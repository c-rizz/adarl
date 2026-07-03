from __future__ import annotations
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_FLAGS"]="--xla_gpu_triton_gemm_any=true"

from adarl.adapters.mujoco_utils import (add_arrow_to_renderer, aggregate_models, apply_opt_preset, 
                                         get_renderdata_into, log_largest_dataclass_fields,
                                         model_element_separator, print_mj_model, apply_dof_overrides)
from adarl.adapters.BaseVecAdapter import JointProperties, JointType
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import pkgutil_get_path, exc_to_str, build_pose, to_string_tensor, quat_xyzw_between_vecs_py
from typing import Any
import jax
import jax.tree_util
import mujoco
from mujoco import mjx
import mujoco.viewer
import time
from typing_extensions import override
from typing import overload, Sequence, Mapping, Literal
import torch as th
import jax.numpy as jnp
import jax.scipy.spatial.transform
import adarl.utils.dbg.ggLog as ggLog
import torch.utils.dlpack as thdlpack
import numpy as np
import copy
from typing import Iterable
from functools import partial
import dataclasses
from dataclasses import dataclass 
import pprint
from adarl.utils.tensor_trees import map_tensor_tree
from packaging.version import Version
import faulthandler
from adarl.utils.base_utils import record_time, print_recorded_times, record_region_start, record_region_end
from adarl.utils.session import default_session

faulthandler.enable()

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_compilation_cache", True)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_transfer_guard_device_to_host", "log") # Should log implicit device-to-host transfers
jax.config.update("jax_debug_nans", True) # May have a performance impact?
# jax.config.update("jax_debug_infs", True) # May have a performance impact?
# jax.config.update("jax_disable_jit", True)  # 
# jax.config.update("jax_check_tracer_leaks", True) # May have a performance impact
# jax.config.update("jax_explain_cache_misses", True) # May have a performance impact
# jax.config.update("jax_enable_x64",True)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
jax.config.update('jax_default_matmul_precision', "highest")

# def inplace_deepcopy(dst, src, strict = False, exclude : Iterable = []):
#     if type(src) != type(dst):
#         raise RuntimeError(f"src and dst should be of same class, but they are respectively {type(src)} {type(dst)}")
#     exclude = set(exclude)
#     attrs = [a for a in dir(src) if not a.startswith('__') and not callable(getattr(src, a))]
#     src_copy = copy.deepcopy(src) # deepcopy fails if copying the attibutes one by one
#     for attr in attrs:
#         if attr not in exclude:
#             try:
#                 setattr(dst, attr, getattr(src_copy, attr))
#             except AttributeError as e:
#                 if not strict:
#                     ggLog.warn(f"Error setting attribute '{attr}': {e}")
#                 else:
#                     raise e

from mujoco.mjx._src.forward import euler, forward, fwd_acceleration, fwd_actuation, fwd_position, fwd_velocity
from mujoco.mjx._src import sensor
from mujoco.mjx._src import solver
from packaging.version import Version


import adarl.adapters.mujoco_utils as mjutils


def _patch_mjx_warp_tileset_pytree_metadata():
    """Make mjx's warp-backend ``TileSet`` safe to use as JAX pytree metadata.

    With the warp backend, ``Model._impl`` is a ``ModelWarp`` whose field
    ``qM_tiles: Tuple[TileSet, ...]`` is NOT a jax.Array-bearing type, so mjx's
    pytree registration places it in the *static metadata* (aux_data) partition.
    mjx only content-hash-wraps bare ``np.ndarray`` / ``Tuple[np.ndarray, ...]``
    metadata, so the ``TileSet`` instances are stored raw. ``TileSet`` is a
    frozen dataclass holding ``adr: np.ndarray``, so its auto-generated
    ``__eq__``/``__hash__`` operate on numpy arrays: when JAX compares two Model
    treedefs for a jit cache lookup it hits ``np.ndarray == np.ndarray`` and
    raises "The truth value of an array with more than one element is
    ambiguous". This patches ``TileSet`` with array-safe equality/hashing.

    This is backend-specific (only the warp Model triggers it) and version
    independent. No-op if the warp types aren't importable, or if a mujoco build
    that already fixes this upstream is installed (see
    https://github.com/google-deepmind/mujoco/pull/3299).
    """
    try:
        from mujoco.mjx.warp import types as _mjxw_types  # type: ignore
    except Exception:
        return
    TileSet = getattr(_mjxw_types, "TileSet", None)
    if TileSet is None or getattr(TileSet, "_adarl_array_safe", False):
        return

    # Skip if TileSet already has array-safe eq/hash (e.g. mujoco PR #3299 landed).
    try:
        _probe_a = TileSet(adr=np.array([1, 2, 3]), size=4)
        _probe_b = TileSet(adr=np.array([1, 2, 3]), size=4)
        if (_probe_a == _probe_b) is True and hash(_probe_a) == hash(_probe_b):
            return
    except Exception:
        pass  # current methods are broken (the bug we're fixing) -> patch below

    def _eq(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.size == other.size and np.array_equal(self.adr, other.adr)

    def _hash(self):
        a = np.ascontiguousarray(self.adr)
        return hash((self.size, a.shape, a.dtype.str, a.tobytes()))

    TileSet.__eq__ = _eq
    TileSet.__hash__ = _hash
    TileSet._adarl_array_safe = True


_patch_mjx_warp_tileset_pytree_metadata()


if Version(jax.__version__) < Version("0.8.0"):
    def th2jax(tensor : th.Tensor, jax_device : mjutils.jax_Device):
        # apparently there are issues with non-contiguous tensors, should be fixed in 0.8.0 (https://github.com/jax-ml/jax/issues/7657)
        # and with CPU tensors, should be fixed since Jan 2025 (https://github.com/jax-ml/jax/issues/25066#issuecomment-2494697463)
        return jnp.from_dlpack(tensor.contiguous().cuda(non_blocking=False)).to_device(jax_device)
    
    def jax2th(array : jnp.ndarray, th_device : th.device):
        return thdlpack.from_dlpack(array.to_device(jax_Devices("gpu")[0])).to(th_device, non_blocking=th_device.type=="cuda")
else:
    def th2jax(tensor : th.Tensor, jax_device : mjutils.jax_Device):
        return jnp.from_dlpack(tensor.contiguous()).to_device(jax_device)
                                                    
    def jax2th(array : jnp.ndarray, th_device : th.device):
        return thdlpack.from_dlpack(array).to(th_device)

jitted_scan = jax.jit(jax.lax.scan, static_argnames=("length", "reverse", "unroll"))
devices_th2jax = {}
devices_jax2th = {}

def _build_th2jax_dev_mapping():
    global devices_th2jax
    global devices_jax2th
    th_devs : Sequence[th.device] = [th.device("cuda", i) for i in range(th.cuda.device_count())]
    th_devs.append(th.device("cpu"))
    devices_th2jax = {}
    for dev in th_devs:
        z = th.zeros((1,), device=dev, dtype=th.float32)
        jz = jnp.from_dlpack(z)
        devices_th2jax[dev] = jz.device
    devices_jax2th = {v:k for k,v in devices_th2jax.items()}

_build_th2jax_dev_mapping()

def _forward_pre(mjx_model : mjx.Model, mjx_data : mjx.Data):
    # Some values in mjData depend on qfrc_applied, and must be recomputed befor the step
    # see for example what happens in fwd_acceleration.
    # To have these correctly set we have to compute at least a part of the forward pass before the step,
    # but after we have decided qfrc_applied.
    # To avoid doing operations twice
    # A forward pass is also done after the step to have updated values for computing observations (e.g. forward kinematics).
    # To avoid doing redundant work, and avoiding to compute force/acceleratons with inproper torque inputs, actuation-dependant
    # forward will only be computed before the physics integration, here in this method.

    # The following is adapted from mjx.forward
    # some of these steps can be pointed to the pipeline at https://mujoco.readthedocs.io/en/stable/computation/index.html#simulation-pipeline
    mjx_data = fwd_actuation(mjx_model, mjx_data) # pipeline step 19
    mjx_data = fwd_acceleration(mjx_model, mjx_data) # pipeline step 20

    if mjx_data.efc_J.size == 0:
        mjx_data = mjx_data.replace(qacc=mjx_data.qacc_smooth)
        return mjx_data

    with jax.named_scope("MjxAdapter._act_forward"):
        mjx_data = solver.solve(mjx_model, mjx_data) # pipeline step 21, writes in to qacc, qacc_warmstart, qfrc_constraint, efc_force
    
    mjx_data = sensor.sensor_acc(mjx_model, mjx_data) # pipeline step 22

    return mjx_data

def _forward_post(mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
    # See comment on _forward_pre
    mjx_data = fwd_position(mjx_model, mjx_data)
    mjx_data = sensor.sensor_pos(mjx_model, mjx_data)
    mjx_data = fwd_velocity(mjx_model, mjx_data)
    mjx_data = sensor.sensor_vel(mjx_model, mjx_data)
    mjx_data = sensor.sensor_acc(mjx_model, mjx_data) # Also comutes cacc. This ends up being done two times, maybe I don't need to do it before the step?
    return mjx_data

def mjx_integrate_and_forward_split(m: mjx.Model, d: mjx.Data) -> mjx.Data:
    """First integrate the physics, then compute forward kinematics/dynamics.
        This is a flipped-around version of mjx.step(), essentially doing mj_step2 and then mj_step1.
        By doing so, the simulation state is already updated after the step, however it is important
        to call forward once before calling this for the first time. I believe dm_control does
        something similar.
        This is correct only if using the Euler integrator."""
    # see: https://github.com/google-deepmind/mujoco/issues/430#issuecomment-1208489785
    #TODO: Check if it is the same to call mjx.step2 and then mjx.step1
    d = _forward_pre(m,d) # compute values that depend on qfrc_applied
    d = euler(m, d)
    # d = forward(m,d) # compute forward, so that we have an up-to-date sim state for computing observation
    d = _forward_post(m, d) # compute forward, so that we have an up-to-date sim state for computing observation
    return d

def mjx_integrate_and_forward_full(m: mjx.Model, d: mjx.Data) -> mjx.Data:
    """ use the MJX functions to first step the sim and then forward the dynamics. This ensures the simlation state is updated
        after the step. However, it is slighlty wasteful as the underlying forward functions are called twice, after the step and also
        at the beginnign of the next one. However, this is needed when using non-euler integrator, and right now is necessary when using Warp, 
        as step1/step2 and the necessary forwward functions are not exposed."""
    d = mjx.step(m, d)
    d = mjx.forward(m,d)
    return d

def set_rows_cols(array : jnp.ndarray,
                  index_arrs : Sequence[jnp.ndarray],
                  vals : jnp.ndarray):
    """Sets values in the specified subarray. index_arrs indicates which indexes of each dimension to set
       the values in. For example you can write in a 2D array at the rows [2,3,4] and columns [0,3,5,7] 
       (which identify a 3x4 array) by setting index_arrs=(jnp.array([2,3,4]), jnp.array([0,3,5,7])) and
       passing a 3x4 array in vals.

    Parameters
    ----------
    array : jnp.ndarray
        The array to be modified (Will not be written to)
    index_arrs : Sequence[jnp.ndarray]
        The indexes in each dimension.
    vals : jnp.ndarray
        The values to write

    Returns
    -------
    jnp.ndarray
        The edited array
    """
    # ggLog.info(f"set_rows_cols(\n"
    #            f"{array}\n"
    #            f"{index_arrs}\n"
    #            f"{vals}\n"
    #            f")")
    index_arrs = [ia if ia.dtype!=bool else jnp.nonzero(ia)[0]    for ia in index_arrs]
    return array.at[jnp.ix_(*index_arrs)].set(vals)

@jax.jit
def set_rows_cols_masks(array : jnp.ndarray,
                        masks : Sequence[jnp.ndarray],
                        vals : jnp.ndarray):
    """Sets values in the specified subarray. index_arrs indicates which indexes of each dimension to set
       the values in. For example you can write in a 2D array at the rows [2,3] and columns [0,2,3] 
       (which identify a 2x3 array) by setting index_arrs=(jnp.array([False, False, True, True]), jnp.array([True, False, True, True])) and
       passing a 2x3 array in vals.

    Parameters
    ----------
    array : jnp.ndarray
        The array to be modified (Will not be written to)
    index_arrs : Sequence[jnp.ndarray]
        The indexes in each dimension.
    vals : jnp.ndarray
        The values to write

    Returns
    -------
    jnp.ndarray
        The edited array
    """
    # # ggLog.info(f"set_rows_cols(\n"
    # #            f"{array}\n"
    # #            f"{index_arrs}\n"
    # #            f"{vals}\n"
    # #            f")")
    # # index_arrs = [jnp.full(ia.shape[i],-1,device=ia.device).at([])[0]    for i,ia in enumerate(masks)]
    # index_arrs = [mask if mask.dtype!=bool else jnp.where(mask, jnp.arange(mask.shape[0]), mask.shape[0]+1) for mask in masks]
    # print(f"index_arrs = {index_arrs}")
    # print(f"array.at[jnp.ix_(*index_arrs)] = {array[jnp.ix_(*index_arrs)]}")
    # return array.at[jnp.ix_(*index_arrs)].set(vals)
    def to_indices(mask, expected_size):
        if mask.dtype == bool:
            # Fixed-size nonzero for JIT compatibility
            return jnp.nonzero(mask, size=expected_size)[0]
        return mask
    
    index_arrs = [to_indices(m, vals.shape[i]) for i, m in enumerate(masks)]
    return array.at[jnp.ix_(*index_arrs)].set(vals)

# def set_masks(  array : jnp.ndarray,
#                 masks : Sequence[jnp.ndarray],
#                 vals : jnp.ndarray):
#     """Sets values in the specified subarray. index_arrs indicates which indexes of each dimension to set
#        the values in. For example you can write in a 2D array at the rows [2,3] and columns [0,2,3] 
#        (which identify a 2x3 array) by setting index_arrs=(jnp.array([False, False, True, True]), jnp.array([True, False, True, True])) and
#        passing a 2x3 array in vals.

#     Parameters
#     ----------
#     array : jnp.ndarray
#         The array to be modified (Will not be written to)
#     index_arrs : Sequence[jnp.ndarray]
#         The indexes in each dimension.
#     vals : jnp.ndarray
#         The values to write

#     Returns
#     -------
#     jnp.ndarray
#         The edited array
#     """
#     if len(masks) == 1:
#         return array.at[masks[0]].set(vals)
#     # ggLog.info(f"set_rows_cols(\n"
#     #            f"{array}\n"
#     #            f"{index_arrs}\n"
#     #            f"{vals}\n"
#     #            f")")
#     # index_arrs = [jnp.full(ia.shape[i],-1,device=ia.device).at([])[0]    for i,ia in enumerate(masks)]
#     index_arrs = [mask if mask.dtype!=bool else jnp.where(mask, jnp.arange(mask.shape[0]), mask.shape[0]+1) for mask in masks]
#     print(f"index_arrs = {index_arrs}")
#     return array.at[jnp.ix_(*index_arrs)].set(vals)



def get_rows_cols(array : jnp.ndarray,
                  index_arrs : Sequence[jnp.ndarray | Sequence[int] | int]):
    """Gets values of the specified subarray. index_arrs indicates which indexes of each dimension to get
       the values from. For example you can read from a 2D array at the rows [2,3,4] and columns [0,3,5,7] 
       (which identify a 3x4 array) by setting index_arrs=(jnp.array([2,3,4]), jnp.array([0,3,5,7])). Or you
       for example:
            get_rows_cols(jnp.arange(0,24).reshape(2,3,4),
                         (jnp.array([0]), jnp.array([1,2]), jnp.array([0,4])))
            Gives:
            Array([[[ 4,  7],
                    [ 8, 11]]], dtype=int32)

    Parameters
    ----------
    array : jnp.ndarray
        The array to be read
    index_arrs : Sequence[jnp.ndarray]
        The indexes in each dimension.

    Returns
    -------
    jnp.ndarray
        The selected array
    """
    index_arrs = [ia if isinstance(ia, jnp.ndarray) else jnp.array(ia)  for ia in index_arrs]
    zerodim_axes = [i for i in range(len(index_arrs)) if index_arrs[i].ndim==0]
    index_arrs = [ia if ia.ndim>0 else jnp.expand_dims(ia,0)            for ia in index_arrs]
    index_arrs = [ia if ia.dtype!=bool else jnp.nonzero(ia)[0]          for ia in index_arrs]
    return array[jnp.ix_(*index_arrs)].squeeze(zerodim_axes)


from mujoco.mjx._src.io import mjwp, types

from typing import Optional, Dict, Union
from mujoco.mjx._src.io import types, _resolve_impl_and_device, _put_data_jax, _put_data_cpp, _check_warp_installed, _wp_to_np_type, _put_data_public_fields, _get_nested_attr

def _put_data_warp(
    m: mjutils._MjModel,
    d: mjutils.MjData,
    device: Optional[mjutils.jax_Device] = None,
    naconmax: Optional[int] = None,
    naccdmax: Optional[int] = None,
    njmax: Optional[int] = None,
) -> types.Data:
  """Puts mujoco_MjData onto a device, resulting in mjx.Data."""

  from mujoco.mjx.warp import warp as wp
  import mujoco.mjx.warp as mjxw
  
  with wp.ScopedDevice('cpu'):  # pylint: disable=undefined-variable
    dw = mjwp.put_data(m, d, nworld=1, naconmax=naconmax, njmax=njmax, naccdmax=naccdmax)  # pylint: disable=undefined-variable

  fields = _put_data_public_fields(d)
  for k in fields:
    if not hasattr(dw, k):
      continue
    field = _wp_to_np_type(getattr(dw, k))
    if mjxw.types._BATCH_DIM['Data'][k]:  # pylint: disable=protected-access
      field = field.reshape(field.shape[1:])
    fields[k] = field

  impl_fields = {}
  for k in mjxw.types.DataWarp.__annotations__.keys():
    field = _get_nested_attr(dw, k, split='__')
    field = _wp_to_np_type(field)
    if mjxw.types._BATCH_DIM['Data'][k]:  # pylint: disable=protected-access
      field = field.reshape(field.shape[1:])
    impl_fields[k] = field

  data = types.Data(
      **fields,
      _impl=mjxw.types.DataWarp(**impl_fields),
  )

  data = jax.device_put(data, device=device)
  return data

def put_data(
    m: mjutils._MjModel,
    d: mjutils.MjData,
    device: Optional[mjutils.jax_Device] = None,
    impl: Optional[Union[str, types.Impl]] = None,
    naconmax: Optional[int] = None,
    naccdmax: Optional[int] = None,
    njmax: Optional[int] = None,
    dummy_arg_for_batching: Optional[jax.Array] = None,
    keepalive_refs: Optional[Dict[int, Any]] = None,
) -> types.Data:
  """Puts mujoco_MjData onto a device, resulting in mjx.Data.

  Args:
    m: the model to use
    d: the data to put on device
    device: which device to use - if unspecified picks the default device
    impl: implementation to use ('jax', 'warp')
    nconmax: maximum number of contacts to allocate for warp
    naconmax: maximum number of contacts to allocate for warp across all worlds
      Since the number of worlds is **not** pre-defined in JAX, we use the
      `naconmax` argument to set the upper bound for the number of contacts
      across all worlds, rather than the `nconmax` argument from MuJoCo Warp.
    njmax: maximum number of constraints to allocate for warp
    dummy_arg_for_batching: dummy argument to use for batching in cpp
      implementation
    keepalive_refs: optional dict to store references to underlying MuJoCo
      objects, preventing them from being garbage collected.

  Returns:
    an mjx.Data placed on device
  """

  impl, device = _resolve_impl_and_device(impl, device)
  if impl == types.Impl.JAX:
    return _put_data_jax(m, d, device)
  elif impl == types.Impl.CPP:
    return _put_data_cpp(
        m,
        d,
        device,
        dummy_arg_for_batching=dummy_arg_for_batching,
        keepalive_refs=keepalive_refs,
    )
  elif impl == types.Impl.WARP:
    _check_warp_installed()
    naconmax = nconmax if naconmax is None else naconmax
    return _put_data_warp(m, d, device, naconmax, naccdmax, njmax)

  raise NotImplementedError(
      f'put_data for implementation "{impl}" not implemented yet.'
  )




@jax.jit
def jax_mat_to_quat_xyzw(matrices):
    return jax.scipy.spatial.transform.Rotation.from_matrix(matrices).as_quat(scalar_first=False)

def quat_wxyz_to_rotmat(quat_wxyz : jnp.ndarray) -> jnp.ndarray:
    """Convert a MuJoCo-style quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return jnp.array([[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),     2.0 * (xz + wy)],
                      [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                      [2.0 * (xz - wy),       2.0 * (yz + wx),     1.0 - 2.0 * (xx + yy)]],
                     dtype=quat_wxyz.dtype)


@jax.tree_util.register_dataclass
@dataclass
class SimState:
    mjx_data : mjx.Data
    mjx_model : mjx.Model
    requested_qfrc_applied : jnp.ndarray
    sim_time : jnp.ndarray
    stats_step_count : jnp.ndarray
    mon_joint_stats_arr_pvaeep : jnp.ndarray
    mon_links_stats_arr_v : jnp.ndarray
    mon_joint_state_pveae : jnp.ndarray  # precomputed joint states for monitored joints
    mon_link_state : jnp.ndarray  # precomputed link states for monitored links
    """ pose and velocity state in format (pos_x,pos_y,pos_z, quat_w,quat_x,quat_y,quat_z, linvel_x,linvel_y,linvel_z, angvel_x,angvel_y,angvel_z)
    """
    mon_link_acceleration : jnp.ndarray  # precomputed local linear acceleration for monitored links
    mon_collision_mask : jnp.ndarray  # precomputed collision mask for monitored pairs (vec_size, num_pairs)
    impulse_startends_stime : jnp.ndarray
    impulses_xfrc : jnp.ndarray

    def replace_v(self, name : str, value : Any):
        return self.replace_d({name:value})

        return SimState(mjx_data=value if name=="mjx_data" else self.mjx_data,
                        requested_qfrc_applied=value if name=="requested_qfrc_applied" else self.requested_qfrc_applied,
                        sim_time=value if name=="sim_time" else self.sim_time)

    def replace_d(self, name_values : dict[str,Any]):
        # ggLog.info(f"rd0 type(self.mjx_data) = {type(self.mjx_data)}")
        # d = dataclasses.asdict(self) # Recurses into dataclesses and deepcopies
        d = {"mjx_data" : self.mjx_data,
             "mjx_model" : self.mjx_model,
             "requested_qfrc_applied" : self.requested_qfrc_applied,
             "sim_time" : self.sim_time,
             "stats_step_count" : self.stats_step_count,
             "impulse_startends_stime" : self.impulse_startends_stime,
             "impulses_xfrc" : self.impulses_xfrc,
             "mon_joint_stats_arr_pvaeep" : self.mon_joint_stats_arr_pvaeep,
             "mon_links_stats_arr_v" : self.mon_links_stats_arr_v,
             "mon_joint_state_pveae" : self.mon_joint_state_pveae,
             "mon_link_state" : self.mon_link_state,
             "mon_link_acceleration" : self.mon_link_acceleration,
             "mon_collision_mask" : self.mon_collision_mask}
        # ggLog.info(f"d0 = "+str({k:type(v) for k,v in d.items()}))
        d.update(name_values)
        # ggLog.info(f"d1 = "+str({k:type(v) for k,v in d.items()}))
        ret = SimState(**d)
        # ggLog.info(f"type(self.mjx_data) = {type(self.mjx_data)}")
        return ret
    
mj_jnt_type_to_adarl = {
    mjutils._mjtJoint.mjJNT_FREE  : JointType.FLOATING,
    mjutils._mjtJoint.mjJNT_HINGE : JointType.REVOLUTE,
    mjutils._mjtJoint.mjJNT_SLIDE : JointType.PRISMATIC,
    mjutils._mjtJoint.mjJNT_BALL : JointType.SPHERICAL
}







@jax.tree_util.register_dataclass
@dataclass
class StaticSimConf:
    vec_size : int

    def __hash__(self) -> int:
        return self.vec_size.__hash__()


@jax.tree_util.register_dataclass
@dataclass
class SimConf:
    monitored_qpadr : jnp.ndarray
    monitored_qvadr : jnp.ndarray
    monitored_lids : jnp.ndarray
    monitored_jids : jnp.ndarray
    body_rootid : jnp.ndarray  # maps all bodies to their root body (for acceleration computation)
    monitored_collision_pairs : jnp.ndarray  # (num_pairs, 2) body id pairs to check for collision (JAX backend path)
    pair_sensor_adrs : jnp.ndarray  # (num_pairs,) sensordata adrs of __pair_i__ contact sensors (warp backend path)
    geom_bodyid : jnp.ndarray  # maps geom ids to body ids
    sim_dt : jnp.ndarray
    jnt_qposadr : jnp.ndarray
    jnt_dofadr : jnp.ndarray
    monitored_sids : jnp.ndarray  # site IDs (in MuJoCo space) to monitor as links
    site_bodyid : jnp.ndarray  # maps site id -> parent body id (for velocity derivation)



    def replace_d(self, name_values : dict[str,Any], non_strict : bool = False):
        # ggLog.info(f"rd0 type(self.mjx_data) = {type(self.mjx_data)}")
        # d = dataclasses.asdict(self) # Recurses into dataclesses and deepcopies
        d = {"monitored_qpadr" : self.monitored_qpadr,
            "monitored_qvadr" : self.monitored_qvadr,
            "monitored_lids" : self.monitored_lids,
            "monitored_jids" : self.monitored_jids,
            "body_rootid" : self.body_rootid,
            "monitored_collision_pairs" : self.monitored_collision_pairs,
            "pair_sensor_adrs" : self.pair_sensor_adrs,
            "geom_bodyid" : self.geom_bodyid,
            "sim_dt" : self.sim_dt,
            "jnt_qposadr" : self.jnt_qposadr,
            "jnt_dofadr" : self.jnt_dofadr,
            "monitored_sids" : self.monitored_sids,
            "site_bodyid" : self.site_bodyid}
        if non_strict:
            name_values = {k:v for k,v in name_values.items() if k in d}
        # ggLog.info(f"d0 = "+str({k:type(v) for k,v in d.items()}))
        d.update(name_values)
        # ggLog.info(f"d1 = "+str({k:type(v) for k,v in d.items()}))
        ret = SimConf(**d)
        # ggLog.info(f"type(self.mjx_data) = {type(self.mjx_data)}")
        return ret





class PublicCommand:
    def build_internal_command(self, adapter : MjxAdapter) -> _InternalCommand:
        raise NotImplementedError()

@dataclass(frozen=True)
class ForwardCommand(PublicCommand):
    def build_internal_command(self, adapter : MjxAdapter) -> _InternalCommand:
        return _InternalForwardCommand()


@dataclass(frozen=True)
class SetJointsStateDirectCommand(PublicCommand):
    joint_names : Sequence[tuple[str, str]]
    joint_states_pve : th.Tensor
    vec_mask : th.Tensor | None = None

    def build_internal_command(self, adapter : MjxAdapter) -> _InternalCommand:
        joint_qpadr_qvadr, joint_states_pve_jnp = adapter._prepare_joint_set_data(
            self.joint_names,
            self.joint_states_pve,
        )
        return _InternalSetJointsStateDirectCommand(
            vec_mask=adapter._vec_mask_to_jax(self.vec_mask),
            joint_qpadr_qvadr=joint_qpadr_qvadr,
            joint_states_pve=joint_states_pve_jnp,
        )


@dataclass(frozen=True)
class SetLinksStateDirectCommand(PublicCommand):
    link_names : Sequence[tuple[str, str]]
    link_states_pose_vel : th.Tensor
    vec_mask : th.Tensor | None = None

    def build_internal_command(self, adapter : MjxAdapter) -> _InternalCommand:
        (mjmodel_lids,
         mjmodel_pose_xyz_xyzw,
         mjdata_qpadrs_qvadrs,
         mjdata_poses_xyzxyzw_vel_xyzxyz) = adapter._prepare_links_set_data(
            self.link_names,
            self.link_states_pose_vel,
        )
        return _InternalSetLinksStateDirectCommand(
            vec_mask=adapter._vec_mask_to_jax(self.vec_mask),
            mjmodel_lids=mjmodel_lids,
            mjmodel_pose_xyz_xyzw=mjmodel_pose_xyz_xyzw,
            mjdata_qpadrs_qvadrs=mjdata_qpadrs_qvadrs,
            mjdata_poses_xyzxyzw_vel_xyzxyz=mjdata_poses_xyzxyzw_vel_xyzxyz,
        )


@dataclass(frozen=True)
class AlterModelCommand(PublicCommand):
    link_masses : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    link_frictions : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    joint_armature_ratios : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    joint_damping_ratios : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    joint_frictionloss_ratios : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    com_position_diffs : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    com_quatxyzw_diffs : tuple[Sequence[int] | np.ndarray | jnp.ndarray, th.Tensor] | None = None
    vec_mask : th.Tensor | None = None
    reset_first : bool = True

    def build_internal_command(self, adapter : MjxAdapter) -> _InternalCommand:

        if self.link_masses is not None:
            link_masses_body_ids = adapter._to_jax_ids(self.link_masses[0])
            body_masses_ratio_change = th2jax(self.link_masses[1], jax_device=adapter._jax_device)
        else:
            link_masses_body_ids = adapter._empty_jax_array((0,), jnp.int32)
            body_masses_ratio_change = adapter._empty_jax_array((adapter._vec_size, 0), jnp.float32)

        if self.link_frictions is not None:
            frictions_body_ids = adapter._to_jax_ids(self.link_frictions[0])
            body_frictions_ratio_change = th2jax(self.link_frictions[1], jax_device=adapter._jax_device)
        else:
            frictions_body_ids = adapter._empty_jax_array((0,), jnp.int32)
            body_frictions_ratio_change = adapter._empty_jax_array((adapter._vec_size, 0, 3), jnp.float32)

        if self.joint_armature_ratios is not None:
            dof_armatures_dof_ids = adapter._sim_conf.jnt_dofadr[adapter._to_jax_ids(self.joint_armature_ratios[0])]
            dof_armatures_ratio_change = th2jax(self.joint_armature_ratios[1], jax_device=adapter._jax_device)
        else:
            dof_armatures_dof_ids = adapter._empty_jax_array((0,), jnp.int32)
            dof_armatures_ratio_change = adapter._empty_jax_array((adapter._vec_size, 0), jnp.float32)

        if self.joint_damping_ratios is not None:
            dof_dampings_dof_ids = adapter._sim_conf.jnt_dofadr[adapter._to_jax_ids(self.joint_damping_ratios[0])]
            dof_dampings_ratio_change = th2jax(self.joint_damping_ratios[1], jax_device=adapter._jax_device)
        else:
            dof_dampings_dof_ids = adapter._empty_jax_array((0,), jnp.int32)
            dof_dampings_ratio_change = adapter._empty_jax_array((adapter._vec_size, 0), jnp.float32)

        if self.joint_frictionloss_ratios is not None:
            dof_frictionloss_dof_ids = adapter._sim_conf.jnt_dofadr[adapter._to_jax_ids(self.joint_frictionloss_ratios[0])]
            dof_frictionloss_ratio_change = th2jax(self.joint_frictionloss_ratios[1], jax_device=adapter._jax_device)
        else:
            dof_frictionloss_dof_ids = adapter._empty_jax_array((0,), jnp.int32)
            dof_frictionloss_ratio_change = adapter._empty_jax_array((adapter._vec_size, 0), jnp.float32)

        if self.com_position_diffs is not None:
            com_body_pos_ids = adapter._to_jax_ids(self.com_position_diffs[0])
            com_position_diff_xyz = th2jax(self.com_position_diffs[1], jax_device=adapter._jax_device)
        else:
            com_body_pos_ids = adapter._empty_jax_array((0,), jnp.int32)
            com_position_diff_xyz = adapter._empty_jax_array((adapter._vec_size, 0, 3), jnp.float32)

        if self.com_quatxyzw_diffs is not None:
            com_body_quat_ids = adapter._to_jax_ids(self.com_quatxyzw_diffs[0])
            com_quat_diff_xyzw = th2jax(self.com_quatxyzw_diffs[1], jax_device=adapter._jax_device)
        else:
            com_body_quat_ids = adapter._empty_jax_array((0,), jnp.int32)
            com_quat_diff_xyzw = adapter._empty_jax_array((adapter._vec_size, 0, 4), jnp.float32)

        return _InternalAlterModelCommand(
            vec_mask=adapter._vec_mask_to_jax(self.vec_mask),
            link_masses_body_ids=link_masses_body_ids,
            body_masses_ratio_change=body_masses_ratio_change,
            frictions_body_ids=frictions_body_ids,
            body_frictions_ratio_change=body_frictions_ratio_change,
            dof_armatures_dof_ids=dof_armatures_dof_ids,
            dof_armatures_ratio_change=dof_armatures_ratio_change,
            dof_dampings_dof_ids=dof_dampings_dof_ids,
            dof_dampings_ratio_change=dof_dampings_ratio_change,
            dof_frictionloss_dof_ids=dof_frictionloss_dof_ids,
            dof_frictionloss_ratio_change=dof_frictionloss_ratio_change,
            com_body_pos_ids=com_body_pos_ids,
            com_position_diff_xyz=com_position_diff_xyz,
            com_body_quat_ids=com_body_quat_ids,
            com_quat_diff_xyzw=com_quat_diff_xyzw,
        )


class _InternalCommand:
    def has_effect(self) -> bool:
        raise NotImplementedError()

    def marks_forward_needed(self) -> bool:
        return self.has_effect()

    def runs_forward(self) -> bool:
        return False

    def run_jax(self, adapter : MjxAdapter, sim_state : SimState, sim_conf : SimConf, static_sim_conf : StaticSimConf) -> SimState:
        raise NotImplementedError()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _InternalForwardCommand(_InternalCommand):
    def has_effect(self) -> bool:
        return True

    def marks_forward_needed(self) -> bool:
        return False

    def runs_forward(self) -> bool:
        return True

    def run_jax(self, adapter : MjxAdapter, sim_state : SimState, sim_conf : SimConf, static_sim_conf : StaticSimConf) -> SimState:
        return adapter._forward_all(sim_state, sim_conf)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _InternalSetJointsStateDirectCommand(_InternalCommand):
    vec_mask : jnp.ndarray
    joint_qpadr_qvadr : jnp.ndarray
    joint_states_pve : jnp.ndarray

    def has_effect(self) -> bool:
        return self.joint_qpadr_qvadr.shape[1] > 0

    def run_jax(self, adapter : MjxAdapter, sim_state : SimState, sim_conf : SimConf, static_sim_conf : StaticSimConf) -> SimState:
        if not self.has_effect():
            return sim_state
        return adapter._set_joint_state_data(
            sim_state=sim_state,
            sim_conf=sim_conf,
            vec_mask_jnp=self.vec_mask,
            qpadr_qvadr=self.joint_qpadr_qvadr,
            js_pve=self.joint_states_pve,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _InternalSetLinksStateDirectCommand(_InternalCommand):
    vec_mask : jnp.ndarray
    mjmodel_lids : jnp.ndarray
    mjmodel_pose_xyz_xyzw : jnp.ndarray
    mjdata_qpadrs_qvadrs : jnp.ndarray
    mjdata_poses_xyzxyzw_vel_xyzxyz : jnp.ndarray

    def has_effect(self) -> bool:
        return (self.mjmodel_lids.shape[0] > 0
                or self.mjdata_qpadrs_qvadrs.shape[1] > 0)

    def run_jax(self, adapter : MjxAdapter, sim_state : SimState, sim_conf : SimConf, static_sim_conf : StaticSimConf) -> SimState:
        if not self.has_effect():
            return sim_state
        return adapter._set_link_poses(
            sim_state=sim_state,
            vec_size=static_sim_conf.vec_size,
            mjmodel_lids=self.mjmodel_lids,
            mjmodel_pose_xyz_xyzw=self.mjmodel_pose_xyz_xyzw,
            mjdata_qpadrs_qvadrs=self.mjdata_qpadrs_qvadrs,
            mjdata_poses_xyzxyzw_vel_xyzxyz=self.mjdata_poses_xyzxyzw_vel_xyzxyz,
            vec_mask_jnp=self.vec_mask,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _InternalAlterModelCommand(_InternalCommand):
    vec_mask : jnp.ndarray
    link_masses_body_ids : jnp.ndarray
    body_masses_ratio_change : jnp.ndarray
    frictions_body_ids : jnp.ndarray
    body_frictions_ratio_change : jnp.ndarray
    dof_armatures_dof_ids : jnp.ndarray
    dof_armatures_ratio_change : jnp.ndarray
    dof_dampings_dof_ids : jnp.ndarray
    dof_dampings_ratio_change : jnp.ndarray
    dof_frictionloss_dof_ids : jnp.ndarray
    dof_frictionloss_ratio_change : jnp.ndarray
    com_body_pos_ids : jnp.ndarray
    com_position_diff_xyz : jnp.ndarray
    com_body_quat_ids : jnp.ndarray
    com_quat_diff_xyzw : jnp.ndarray

    def has_effect(self) -> bool:
        return (   self.link_masses_body_ids.shape[0] > 0
                or self.frictions_body_ids.shape[0] > 0
                or self.dof_armatures_dof_ids.shape[0] > 0
                or self.dof_dampings_dof_ids.shape[0] > 0
                or self.dof_frictionloss_dof_ids.shape[0] > 0
                or self.com_body_pos_ids.shape[0] > 0
                or self.com_body_quat_ids.shape[0] > 0)

    def run_jax(self, adapter : MjxAdapter, sim_state : SimState, sim_conf : SimConf, static_sim_conf : StaticSimConf) -> SimState:
        if not self.has_effect():
            return sim_state
        apply_link_masses = self.link_masses_body_ids.shape[0] > 0
        apply_link_frictions = self.frictions_body_ids.shape[0] > 0
        apply_dof_armature_ratios = self.dof_armatures_dof_ids.shape[0] > 0
        apply_dof_damping_ratios = self.dof_dampings_dof_ids.shape[0] > 0
        apply_dof_frictionloss_ratios = self.dof_frictionloss_dof_ids.shape[0] > 0
        apply_com_position_diffs = self.com_body_pos_ids.shape[0] > 0
        apply_com_quatxyzw_diffs = self.com_body_quat_ids.shape[0] > 0
        mjx_model = adapter._alter_model_jax(
            mjx_model=sim_state.mjx_model,
            vec_size=static_sim_conf.vec_size,
            geom_bodyid=sim_conf.geom_bodyid,
            vec_mask=self.vec_mask,
            apply_link_masses=apply_link_masses,
            link_masses_body_ids=self.link_masses_body_ids,
            body_masses_ratio_change=self.body_masses_ratio_change,
            apply_link_frictions=apply_link_frictions,
            frictions_body_ids=self.frictions_body_ids,
            body_frictions_ratio_change=self.body_frictions_ratio_change,
            apply_dof_armature_ratios=apply_dof_armature_ratios,
            dof_armatures_dof_ids=self.dof_armatures_dof_ids,
            dof_armatures_ratio_change=self.dof_armatures_ratio_change,
            apply_dof_damping_ratios=apply_dof_damping_ratios,
            dof_dampings_dof_ids=self.dof_dampings_dof_ids,
            dof_dampings_ratio_change=self.dof_dampings_ratio_change,
            apply_dof_frictionloss_ratios=apply_dof_frictionloss_ratios,
            dof_frictionloss_dof_ids=self.dof_frictionloss_dof_ids,
            dof_frictionloss_ratio_change=self.dof_frictionloss_ratio_change,
            apply_com_position_diffs=apply_com_position_diffs,
            com_body_pos_ids=self.com_body_pos_ids,
            com_position_diff_xyz=self.com_position_diff_xyz,
            apply_com_quatxyzw_diffs=apply_com_quatxyzw_diffs,
            com_body_quat_ids=self.com_body_quat_ids,
            com_quat_diff_xyzw=self.com_quat_diff_xyzw,
            reset_first=True,
            original_mjx_model=adapter._original_mjx_model,
        )
        return sim_state.replace_v("mjx_model", mjx_model)


@dataclass
class SimElementsState:
    """Aggregated state of monitored sim elements, returned by MjxAdapter.get_sim_elements_state().

    All tensors are on the adapter's output torch device.
    Shapes use: V = vec_size, J = num_requested_joints, L = num_requested_links, P = num_monitored_collision_pairs.
    """
    joint_state_pveae  : th.Tensor | None 
    """(V, J, 5)  pos, vel, cmd_effort, acc, actual_effort"""
    link_state         : th.Tensor | None
    """(V, L, 13) pos_xyz, quat_xyzw, linvel_xyz, angvel_xyz"""
    link_linacc        : th.Tensor | None
    """(V, L, 3)  local linear acceleration"""
    collision_mask     : th.Tensor | None
    """(V, P)     bool, True if pair is in contact"""
    joint_stats_pvaeep  : th.Tensor | None
    """(V, 4, J, 6) step stats [min,max,avg,std] of joint quantities (reordered to position,velocity,acceleration,cmd_effort,actual_effort,cmd_power)"""
    link_stats_v       : th.Tensor | None
    """(V, 4, L, 6) step stats [min,max,avg,std] of link linear+angular velocity"""


class MjxAdapter(BaseVecSimulationAdapter, BaseVecJointEffortAdapter):
    """ Adapter that uses Mujoco MJX as the underlying simulation engine, and provides a vectorized interface to it. 
    This is the most mature and feature-complete adapter in the ADARL suite.
    """

    @dataclass
    class DebugInfo():
        wtime_running : float = 0.0
        wtime_running_since_build : float = 0.0
        wtime_simulating : float = 0.0
        wtime_simulating_since_build : float = 0.0
        wtime_controlling : float = 0.0
        rt_factor_vec : float = 0.0
        rt_factor_single : float = 0.0
        stime_ran : float = 0.0
        stime : float = 0.0
        fps_vec : float = 0.0
        fps_single : float = 0.0
        iterations : int = 0
        run_fps_vec : float = 0.0
        avg_render_time : float = 0.0
        avg_render_update_time : float = 0.0

    def __init__(self, vec_size : int,
                        enable_rendering : bool,
                        jax_device : mjutils.jax_Device,
                        output_th_device : th.device,
                        sim_step_dt : float = 2/1024,
                        step_length_sec : float = 10/1024,
                        realtime_factor : float | None = None,
                        show_gui : bool = False,
                        gui_frequency : float = 15,
                        gui_env_index : int = 0,
                        add_ground : bool = True,
                        add_sky : bool = True,
                        log_freq : int = -1,
                        opt_preset : Literal["fast","faster","fastest","mujoco_default","slow","slower"] | None = "fast",
                        log_folder : str = "./",
                        record_whole_joint_trajectories : bool = False,
                        log_freq_joints_trajectories : int = 1000,
                        safe_revolute_dof_armature = 0.01,
                        revolute_dof_armature_override = None,
                        revolute_dof_damping_override = None,
                        revolute_dof_frictionloss_override = None,
                        opt_override : dict[str,Any] | None = None,
                        render_backend : Literal["cpu", "warp"] = "cpu",
                        mjx_impl : Literal["jax","warp"] = "jax",
                        disable_builtin_actuators : bool = True,
                        geom_overrides : dict[str,Any] | None = None,
                        warp_nccdmax : int = 10,
                        warp_nconmax : int = 20):
        """_summary_

        Parameters
        ----------
        vec_size : int
            _description_
        enable_rendering : bool
            _description_
        jax_device : mjutils.jax_Device
            _description_
        output_th_device : th.device
            _description_
        sim_step_dt : float, optional
            _description_, by default 2/1024
        step_length_sec : float, optional
            _description_, by default 10/1024
        realtime_factor : float | None, optional
            _description_, by default None
        show_gui : bool, optional
            _description_, by default False
        gui_frequency : float, optional
            _description_, by default 15
        gui_env_index : int, optional
            _description_, by default 0
        add_ground : bool, optional
            _description_, by default True
        add_sky : bool, optional
            _description_, by default True
        log_freq : int, optional
            _description_, by default -1
        opt_preset : Literal[&quot;fast&quot;,&quot;faster&quot;,&quot;fastest&quot;,&quot;mujoco_default&quot;,&quot;slow&quot;,&quot;slower&quot;] | None, optional
            _description_, by default "fast"
        log_folder : str, optional
            _description_, by default "./"
        record_whole_joint_trajectories : bool, optional
            _description_, by default False
        log_freq_joints_trajectories : int, optional
            _description_, by default 1000
        safe_revolute_dof_armature : float, optional
            _description_, by default 0.01
        revolute_dof_armature_override : _type_, optional
            _description_, by default None
        revolute_dof_damping_override : _type_, optional
            _description_, by default None
        revolute_dof_frictionloss_override : _type_, optional
            _description_, by default None
        opt_override : dict[str,Any] | None, optional
            _description_, by default None
        render_backend : Literal[&quot;cpu&quot;, &quot;warp&quot;], optional
            _description_, by default "cpu"
        mjx_impl : Literal[&quot;jax&quot;,&quot;warp&quot;], optional
            _description_, by default "jax"
        disable_builtin_actuators : bool, optional
            _description_, by default True
        geom_overrides : dict[str,Any] | None, optional
            _description_, by default None
        warp_nccdmax : int, optional
            per-world max number of mesh contacts (handled by the CCD collider), by default 10
        warp_nconmax : int, optional
            per-world max number of overall contacts, by default 20
        """
        super().__init__(vec_size=vec_size,
                         output_th_device=output_th_device)
        self._disable_builtin_actuators = disable_builtin_actuators
        self._enable_rendering = enable_rendering
        self._render_backend : Literal["cpu", "warp"] = render_backend
        self._jax_device = jax_device
        self._sim_step_dt = sim_step_dt
        self._sim_step_dt_th = th.as_tensor(sim_step_dt, device=output_th_device)
        self._step_length_sec = step_length_sec
        self._simTime = 0.0
        self._total_iterations = 0
        self._sim_step_count_since_build = 0
        self._add_ground = add_ground
        self._add_sky = add_sky
        self._log_freq = log_freq
        self._log_folder = log_folder
        self._last_log_iters = -log_freq 
        self._opt_preset = opt_preset
        self._safe_revolute_dof_armature = safe_revolute_dof_armature
        self._safe_revolute_dof_damping = 1.0
        self._safe_revolute_dof_frictionloss = 0.2
        self._revolute_dof_armature_override = revolute_dof_armature_override #0.5
        self._revolute_dof_damping_override = revolute_dof_damping_override
        self._revolute_dof_frictionloss_override = revolute_dof_frictionloss_override
        self._discardvisual = False
        self._opt_override = opt_override
        self._geom_overrides = geom_overrides
        self._mjx_impl = mjx_impl
        self._jax_float_dtype = jnp.float32
        self._out_cuda = self._out_th_device.type == "cuda"
        self._warp_nccdmax = warp_nccdmax
        self._warp_nconmax = warp_nconmax

        self._realtime_factor = realtime_factor
        self._sim_state = SimState( mjx_data=jnp.empty((0,), device = jax_device), # type: ignore
                                    mjx_model=jnp.empty((0,), device = jax_device), # type: ignore
                                    requested_qfrc_applied=jnp.empty((0,), device = jax_device),
                                    sim_time=jnp.empty((0,), device = jax_device),
                                    stats_step_count=jnp.zeros((1,), device = jax_device),
                                    mon_joint_stats_arr_pvaeep=jnp.empty((self._vec_size,0,6), device = jax_device),
                                    mon_links_stats_arr_v=jnp.empty((self._vec_size,0,6), device = jax_device),
                                    mon_joint_state_pveae=jnp.empty((self._vec_size,0,5), device = jax_device),
                                    mon_link_state=jnp.empty((self._vec_size,0,13), device = jax_device),
                                    mon_link_acceleration=jnp.empty((self._vec_size,0,3), device = jax_device),
                                    mon_collision_mask=jnp.empty((self._vec_size,0), device = jax_device, dtype=jnp.bool_),
                                    impulse_startends_stime=jnp.empty((0,), device = jax_device),
                                    impulses_xfrc=jnp.empty((0,), device = jax_device))
        self._static_sim_conf = StaticSimConf(vec_size=vec_size)
        self._sim_conf = SimConf(   monitored_qpadr=jnp.empty((0,), device = jax_device),
                                    monitored_qvadr=jnp.empty((0,), device = jax_device),
                                    monitored_jids=jnp.empty((0,),  device = jax_device, dtype=jnp.int32),
                                    monitored_lids=jnp.empty((0,),  device = jax_device, dtype=jnp.int32),
                                    body_rootid=jnp.empty((0,), device = jax_device, dtype=jnp.int32),
                                    monitored_collision_pairs=jnp.empty((0,2), device = jax_device, dtype=jnp.int32),
                                    pair_sensor_adrs=jnp.empty((0,), device = jax_device, dtype=jnp.int32),
                                    geom_bodyid=jnp.empty((0,), device = jax_device, dtype=jnp.int32),
                                    sim_dt=jnp.array(sim_step_dt, device = self._jax_device),
                                    jnt_qposadr=jnp.empty((0,), device = jax_device, dtype=jnp.int32),
                                    jnt_dofadr=jnp.empty((0,), device = jax_device, dtype=jnp.int32),
                                    monitored_sids=jnp.empty((0,), device = jax_device, dtype=jnp.int32),
                                    site_bodyid=jnp.empty((0,), device = jax_device, dtype=jnp.int32))
        self._renderer : mujoco.Renderer | None = None
        self._check_sizes = True
        self._show_gui = show_gui
        self._gui_env_index = gui_env_index
        self._last_gui_update_wtime = 0.0
        self._gui_freq = gui_frequency
        self._prev_step_end_wtime = 0.0
        self._viewer = None
        self._warp_render_context = None
        self._warp_render_context_pytree = None
        self._warp_render_has_rgb = False
        self._warp_render_has_depth = False
        self._all_vecs = jnp.ones((vec_size,), dtype=bool, device=self._jax_device)
        self._no_vecs = jnp.zeros((vec_size,), dtype=bool, device=self._jax_device)
        self._all_vecs_th = th.ones((vec_size,), dtype=th.bool, device=self._out_th_device)
        self._no_vecs_th = th.zeros((vec_size,), dtype=th.bool, device=self._out_th_device)
        self._all_vecs_thcpu = th.ones((vec_size,), dtype=th.bool, device="cpu")
        self._no_vecs_thcpu = th.zeros((vec_size,), dtype=th.bool, device="cpu")

        self._dbg_info = MjxAdapter.DebugInfo()

        self._mon_body_count = 0
        self._mon_site_count = 0
        self._mon_link_jax_to_user = th.empty((0,), dtype=th.long, device=self._out_th_device)
        self._scenario_built = False
        self._monitored_collision_pairs : list[tuple[tuple[str,str], tuple[str,str]]] = []
        self._monitored_collision_pair_to_idx : dict[tuple[int,int], int] = {}
        self._record_joint_hist = record_whole_joint_trajectories
        ggLog.info(f"MjxAdapter initialized with record_whole_joint_trajectories={record_whole_joint_trajectories}")
        self._joints_pveae_history = []
        self._log_freq_joints_trajcetories = log_freq_joints_trajectories

    @override
    def sim_step_duration(self):
        return self._sim_step_dt_th

    def _mj_name_to_pair(self, mjid : int, objtype):
        mjname = mjutils._mj_id2name(self._mj_model, objtype, mjid)
        if mjname is None:
            if objtype == mjutils._mjtObj.mjOBJ_BODY:
                objtype_name = "body"
            elif objtype == mjutils._mjtObj.mjOBJ_JOINT:
                objtype_name = "joint"
            elif objtype == mjutils._mjtObj.mjOBJ_GEOM:
                objtype_name = "geom"
            elif objtype == mjutils._mjtObj.mjOBJ_CAMERA:
                objtype_name = "camera"
            elif objtype == mjutils._mjtObj.mjOBJ_SITE:
                objtype_name = "site"
            else:
                raise RuntimeError(f"Unsupported objtype {objtype}")
            mjname = f"unknownmodel#{objtype_name}_{mjid}"
        if mjname == "world":
            return mjname,mjname
        sep = mjname.find(model_element_separator)
        if mjname.count(model_element_separator) != 1:
            raise RuntimeError(f"Invalid mjName: must contain one and only one '{model_element_separator}' substring, but it's {mjname}")
        return mjname[:sep],mjname[sep+len(model_element_separator):]

    def _recompute_mjxmodel_inaxes(self, mjx_model):
        # ggLog.info(f"jax.tree.structure(mjx_model) = {jax.tree.structure(mjx_model)}")
        out_axes = jax.tree_util.tree_map(lambda l:None, mjx_model)
        out_axes = out_axes.tree_replace({"body_mass":0,
                                          "geom_friction":0,
                                          "body_ipos":0,
                                          "body_iquat":0,
                                          "dof_armature":0,
                                          "dof_frictionloss":0,
                                          "dof_damping":0,
                                          "body_pos":0,
                                          "body_quat":0}) # model fields to be vmapped
        # out_axes = map_tensor_tree(mjx_model, lambda l:None)
        # out_axes = out_axes.tree_replace({"body_mass":0,
        #             "geom_friction":0,
        #             "body_ipos":0,
        #             "body_iquat":0})
        self._mjx_model_in_axes = out_axes

    def _rebuild_lower_funcs(self):
        # ggLog.info(f"Rebuilding with self._mjx_model_in_axes= {self._mjx_model_in_axes}")
        self._mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(self._mjx_model_in_axes, 0)))
        # self._mjx_forward_post = jax.jit(jax.vmap(_forward_post, in_axes=(self._mjx_model_in_axes, 0)))
        if self._mjx_impl == "jax":
            if self._mj_model.opt.integrator == mjutils._mjtIntegrator.mjINT_EULER:
                stepping_func = mjx_integrate_and_forward_split
            else:
                stepping_func = mjx_integrate_and_forward_full
        elif self._mjx_impl == "warp":
            stepping_func = mjx_integrate_and_forward_full
        self._mjx_integrate_and_forward = jax.jit(jax.vmap(stepping_func, in_axes=(self._mjx_model_in_axes, 0))) #, donate_argnames=["d"]) donating args make it crash
        self._mjx_ray_vec = jax.jit(jax.vmap(
                                fun=jax.vmap(mjx.ray,
                                             in_axes=(None, None, 0, 0, None)), # map over number of rays
                                in_axes=((self._mjx_model_in_axes, 0, 0, 0, None)) # map over sims and per-env rays positions/directions
                            ))

    def _init_warp_render_context(self, resolutions : dict[str, tuple[int,int]] | None = None):
        if not self._enable_rendering or self._render_backend != "warp":
            return
        missing_symbols = [
            name for name in ("create_render_context", "refit_bvh", "render", "get_rgb", "get_depth")
            if not hasattr(mjx, name)
        ]
        if missing_symbols:
            raise RuntimeError(
                f"render_backend='warp' requested, but mujoco.mjx is missing rendering symbols: {missing_symbols}"
            )

        cam_active = [self._cid2cname[cid] in self._monitored_cameras for cid in range(self._mj_model.ncam)]
        cam_res_wh = [(self._camera_sizes_hw[c][1], self._camera_sizes_hw[c][0])
                        for c in self._monitored_cameras]
        self._warp_render_context = mjx.create_render_context(
            self._mj_model,
            nworld=self._vec_size,
            cam_res=cam_res_wh,
            render_rgb=True,
            render_depth=True,
            cam_active=cam_active,
        )
        self._warp_render_context_pytree = self._warp_render_context.pytree()
        self._warp_render_cname2idx = {cname: i for i, cname in enumerate(self._monitored_cameras)}
        self._warp_render_has_rgb = True
        self._warp_render_has_depth = True
        ggLog.info(f"monitored_cameras = {self._monitored_cameras}")
        ggLog.info(f"_cid2cname = {self._cid2cname}")
        ggLog.info(f"Warp render context initialized with cam_active={cam_active}, res={cam_res_wh}")

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0))
    def _vec_get_rgb(render_context, cidx, pixels):
        return mjx.get_rgb(render_context, cidx, pixels)

    @staticmethod
    @partial(jax.jit, static_argnames=["render_context","cam_indexes"])
    def _render_warp_jax(sim_state : SimState, render_context : Any, cam_indexes : tuple[int]):
        if render_context is None:
            raise RuntimeError("Warp render context not initialized")
        mjx_data = mjx.refit_bvh(sim_state.mjx_model, sim_state.mjx_data, render_context)
        pixels, aux = mjx.render(sim_state.mjx_model, mjx_data, render_context)
        rgbs = []
        for cidx in cam_indexes:
            rgb = MjxAdapter._vec_get_rgb(render_context, cidx, pixels)
            rgbs.append(rgb)
        return rgbs

    def _aggregate_models(self, models, log_folder):
        """Aggregate models into a single MuJoCo model. Override in subclasses to modify the model before compilation."""
        # Build post-attach body-name pairs for contact sensors. Sensors are warp-only;
        # the JAX backend keeps reading mjx_data.contact, so no sensors get injected for it.
        contact_pairs : list[tuple[str, str]] | None = None
        if self._mjx_impl == "warp" and len(self._monitored_collision_pairs) > 0:
            def _to_body_name(p):
                if p == ("world", "world"):
                    return "world"
                return p[0] + model_element_separator + p[1]
            contact_pairs = [(_to_body_name(a), _to_body_name(b)) for a, b in self._monitored_collision_pairs]
        return aggregate_models(models,
                                add_ground=self._add_ground,
                                add_sky=self._add_sky,
                                uneven_ground=self._uneven_ground,
                                discardvisual=self._discardvisual,
                                log_folder=log_folder,
                                geom_overrides=self._geom_overrides,
                                contact_pairs=contact_pairs,
                                opt_preset=self._opt_preset,
                                opt_overrides=self._opt_override,
                                revolute_dof_armature_override=self._revolute_dof_armature_override,
                                revolute_dof_damping_override=self._revolute_dof_damping_override,
                                revolute_dof_frictionloss_override=self._revolute_dof_frictionloss_override,
                                safe_revolute_dof_armature=self._safe_revolute_dof_armature,
                                safe_revolute_dof_damping=self._safe_revolute_dof_damping,
                                safe_revolute_dof_frictionloss=self._safe_revolute_dof_frictionloss)

    @override
    def build_scenario(self, models : list[ModelSpawnDef],
                       default_link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]] | None = None):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
        ggLog.info(f"MjxAdapter building scenario")
        scenario_logs_folder = self._log_folder+"/MjxAdapter/scenario_logs"
        # jax.profiler.start_server(9999)
        self._uneven_ground = False
        self._mj_model, big_speck = self._aggregate_models(models, scenario_logs_folder)
        # self._mj_model = apply_opt_preset(self._mj_model, self._opt_preset, self._opt_override)
        self._mj_model.opt.timestep = self._sim_step_dt
        if self._disable_builtin_actuators:
            self._mj_model.opt.disableactuator = -1 # disable all built-in actuators, we will apply forces/torques directly to the joints in the control step
            # MJX and warp actually seem to ignore disableactuator, so we also set the corresponding disable flag to be sure:
            self._mj_model.opt.disableflags |= mjutils._mjtDisableBit.mjDSBL_ACTUATION
        # I prevent slipping by using a big impratio see for example:
        # - https://github.com/google-deepmind/mujoco_menagerie/blob/d98292efc73511aa7a4ca958eaaf226403d56cb7/anybotics_anymal_b/anymal_b.xml#L4 
        # and the discussion at these links:
        # - https://github.com/google-deepmind/mujoco/discussions/656#discussioncomment-4416347
        # - https://mujoco.readthedocs.io/en/latest/modeling.html#cslippage
        # - https://mujoco.readthedocs.io/en/latest/overview.html#softness-and-slip
        
        
        # self._mj_model = apply_dof_overrides(
        #                     self._mj_model, 
        #                     revolute_dof_armature_override=self._revolute_dof_armature_override,
        #                     revolute_dof_damping_override=self._revolute_dof_damping_override,
        #                     revolute_dof_frictionloss_override=self._revolute_dof_frictionloss_override,
        #                     safe_revolute_dof_armature=self._safe_revolute_dof_armature,
        #                     safe_revolute_dof_damping=self._safe_revolute_dof_damping,
        #                     safe_revolute_dof_frictionloss=self._safe_revolute_dof_frictionloss)
        # ggLog.info(f"big_speck.degree = {big_speck.compiler.degree}")
        os.makedirs(scenario_logs_folder, exist_ok=True)
        with open(scenario_logs_folder+"/mujoco_opt.txt", "w") as text_file:
            text_file.write(str(self._mj_model.opt))
        
        # model = models[0]
        # if model.format == "urdf.xacro":
        #     urdf_string = compile_xacro_string( model_definition_string=model.definition_string,
        #                                                     model_kwargs=model.kwargs)
        # elif model.format == "urdf":
        #     urdf_string = model.definition_string
        # else:
        #     raise RuntimeError(f"Unsupported model format '{model.format}'")
        # self._model_name = model.name
        # # Make model, data, and renderer
        # self._mj_model = mujoco_MjModel.from_xml_string(urdf_string)

        self._jid2jname : dict[int, tuple[str,str]] = {jid:self._mj_name_to_pair(jid, mjutils._mjtObj.mjOBJ_JOINT)
                           for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname : dict[int, tuple[str,str]] = {lid:self._mj_name_to_pair(lid, mjutils._mjtObj.mjOBJ_BODY)
                           for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}
        # Sites are exposed as links with IDs offset by nbody
        self._nbody = self._mj_model.nbody
        self._sid2sname : dict[int, tuple[str,str]] = {sid:self._mj_name_to_pair(sid, mjutils._mjtObj.mjOBJ_SITE)
                           for sid in range(self._mj_model.nsite)}
        self._sname2sid = {sn:sid for sid,sn in self._sid2sname.items()}
        # Merge sites into the link namespace (site unified lid = nbody + sid)
        for sid, sname in self._sid2sname.items():
            if sname in self._lname2lid:
                ggLog.warn(f"Site name {sname} collides with an existing body name, skipping site. "
                           f"Rename the site in the MJCF to make it addressable as a link.")
                continue
            unified_lid = self._nbody + sid
            self._lid2lname[unified_lid] = sname
            self._lname2lid[sname] = unified_lid
        self._cid2cname : dict[int, str] = {cid:self._mj_name_to_pair(cid, mjutils._mjtObj.mjOBJ_CAMERA)[1]
                           for cid in range(self._mj_model.ncam)}
        self._cname2cid = {cn:cid for cid,cn in self._cid2cname.items()}

        # Caches of name-derived, device-resident address arrays for state-setting commands.
        # Keyed by the (normalized) joint/link name tuple; the model and device are frozen
        # after build, so entries never need invalidation (a rebuild rebuilds these too).
        self._joint_addr_cache : dict[tuple[tuple[str,str], ...], jnp.ndarray] = {}
        self._link_addr_cache : dict[tuple[tuple[str,str], ...],
                                     tuple[jnp.ndarray, th.Tensor, th.Tensor, jnp.ndarray]] = {}

        # Resolve buffered monitored collision pairs once lname2lid is available.
        # The JAX backend uses the (P,2) body-id array; the warp backend uses the
        # sensor adrs recovered from the __pair_<i>__ sensors injected at compile time.
        if len(self._monitored_collision_pairs) > 0:
            lid_pairs = [ (self._lname2lid[la], self._lname2lid[lb]) for la,lb in self._monitored_collision_pairs]
            self._monitored_collision_pair_to_idx = {}
            for idx, (a, b) in enumerate(lid_pairs):
                self._monitored_collision_pair_to_idx[(a, b)] = idx
                self._monitored_collision_pair_to_idx[(b, a)] = idx
            self._sim_conf.monitored_collision_pairs = jnp.array(lid_pairs, device=self._jax_device, dtype=jnp.int32)
            if self._mjx_impl == "warp":
                pair_adrs = []
                for i in range(len(self._monitored_collision_pairs)):
                    sname = f"__pair_{i}__"
                    sid = mjutils._mj_name2id(self._mj_model, mjutils._mjtObj.mjOBJ_SENSOR, sname)
                    if sid < 0:
                        raise RuntimeError(f"Internal error: contact sensor '{sname}' missing after compile")
                    pair_adrs.append(int(self._mj_model.sensor_adr[sid]))
                self._sim_conf.pair_sensor_adrs = jnp.array(pair_adrs, device=self._jax_device, dtype=jnp.int32)

        if default_link_group_collisions is not None:
            # the size of some internal fields in mjx_data (e.g. nefc) are determined by the number of possible collisions 
            # So it may be necessary to set the collisions masks before creatign mjx_data
            geom_contype, geom_conaffinity, body_contype, body_conaffinity = self._compute_collision_masks(default_link_group_collisions)
            self._mj_model.geom_contype = geom_contype
            self._mj_model.geom_conaffinity = geom_conaffinity
            self._mj_model.body_contype = body_contype
            self._mj_model.body_conaffinity = body_conaffinity

        print_mj_model(self._mj_model, full_dump=True, file=scenario_logs_folder+"/mj_model_full.txt")


        self._mj_data = mjutils.MjData(self._mj_model)
        mjutils._mj_resetData(self._mj_model, self._mj_data)

        if self._mjx_impl == "warp":
            import mujoco.mjx.warp as mjxw
            mjx_model = mjx.put_model(self._mj_model, device = self._jax_device, impl=self._mjx_impl) #, graph_mode=mjxw.types.GraphMode.WARP_STAGED_EX)
        else:
            mjx_model = mjx.put_model(self._mj_model, device = self._jax_device, impl=self._mjx_impl)
        self._body_rootid = jax.device_put(self._mj_model.body_rootid, device=self._jax_device) # maps bodies to their root body
        self._sim_conf.body_rootid = self._body_rootid
        self._sim_conf.site_bodyid = jax.device_put(jnp.array(self._mj_model.site_bodyid, dtype=jnp.int32), device=self._jax_device)
        # mjx_model.opt.timestep.at[:].set(self._sim_step_dt)
        import operator
        model_nbytes = jax.tree_util.tree_map(lambda x: x.nbytes, mjx_model)
        single_mjmodel_nbytes = jax.tree.reduce(operator.add, model_nbytes)
        self._recompute_mjxmodel_inaxes(mjx_model)
        mjx_model = jax.vmap(lambda: mjx_model, in_axes=None, axis_size=self._vec_size, out_axes=self._mjx_model_in_axes)()
        # mjx_model = jax.vmap(lambda: mjx_model, axis_size=self._vec_size, in_axes=None)()
        model_nbytes = jax.tree_util.tree_map(lambda x: x.nbytes, mjx_model)
        vec_mjmodel_nbytes = jax.tree.reduce(operator.add, model_nbytes)
        
        self._geom_bodyid_jax = jnp.array(mjx_model.geom_bodyid, device = self._jax_device) # for some reason it's a numpy array, so I cannot use it properly in jit
        self._sim_conf.geom_bodyid = self._geom_bodyid_jax
        self._sim_conf.jnt_qposadr = jnp.array(mjx_model.jnt_qposadr, device = self._jax_device, dtype=jnp.int32) # for some reason it's a numpy array, so I cannot use it properly in jit
        self._sim_conf.jnt_dofadr = jnp.array(mjx_model.jnt_dofadr, device = self._jax_device, dtype=jnp.int32) # for some reason it's a numpy array, so I cannot use it properly in jit

        if self._mjx_impl == "warp":
            # self._warp_nccdmax = 10 # per-world max number of mesh contacts (handled by the CCD collider)
            # self._warp_nconmax = 20 # per-world max number of overall contacts
            mjx_data = put_data(self._mj_model, self._mj_data, device = self._jax_device, impl=self._mjx_impl,
                                    naconmax = self._vec_size*self._warp_nconmax, njmax = 100, naccdmax = self._vec_size*self._warp_nccdmax)
        else:
            mjx_data = mjx.put_data(self._mj_model, self._mj_data, device = self._jax_device, impl=self._mjx_impl)
        data_nbytes = jax.tree_util.tree_map(lambda x: x.nbytes, mjx_data)
        single_mjdata_nbytes = jax.tree.reduce(operator.add, data_nbytes)
        mjx_data = jax.vmap(lambda: mjx_data, axis_size=self._vec_size)()
        data_nbytes = jax.tree_util.tree_map(lambda x: x.nbytes, mjx_data)
        vec_mjdata_nbytes = jax.tree.reduce(operator.add, data_nbytes)

        ggLog.info(f"mjx_data size = {single_mjdata_nbytes} bytes")
        ggLog.info(f"vectorized ({self._vec_size}) mjx_data size = {vec_mjdata_nbytes/1024**2} MB")
        ggLog.info(f"mjx_model size = {single_mjmodel_nbytes} bytes")
        ggLog.info(f"vectorized ({self._vec_size}) mjx_model size = {vec_mjmodel_nbytes/1024**2} MB")
        log_largest_dataclass_fields(mjx_model, f"vectorized mjx_model (vec_size={self._vec_size})")
        log_largest_dataclass_fields(mjx_data, f"vectorized mjx_data (vec_size={self._vec_size})")

        # mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(jnp.arange(self._vec_size), mjx_data)
        # ggLog.info(f"mjx_data.qpos.shape = {mjx_data.qpos.shape}")
        # ggLog.info(f"mjx_data.qLD.shape = {mjx_data.qLD.shape}")
        # ggLog.info(f"mj_data.qLD.shape = {self._mj_data.qLD.shape}")
        # ggLog.info(f"mjx_model.nM = {mjx_model.nM}")
        # ggLog.info(f"self._mj_model.nM = {self._mj_model.nM}")
        # mujoco.mj_forward(self._mj_model, self._mj_data) # Compute all fields


        self._original_mjx_data = copy.deepcopy(mjx_data)
        self._original_mjx_model = copy.deepcopy(mjx_model)
        self._original_mj_data = copy.deepcopy(self._mj_data)
        self._original_mj_model = copy.deepcopy(self._mj_model)

        # _ = self._mjx_step(mjx_model, copy.deepcopy(mjx_data)) # trigger jit compile
        self._rebuild_lower_funcs()
        
        requested_qfrc_applied = jnp.copy(mjx_data.qfrc_applied)
        sim_time = jnp.zeros((1,), jnp.float32, device=self._jax_device)
        self._sim_state = self._sim_state.replace_d({   "mjx_data":mjx_data,
                                                        "mjx_model":mjx_model,
                                                        "requested_qfrc_applied":requested_qfrc_applied,
                                                        "sim_time":sim_time,
                                                        "impulses_xfrc" : jnp.zeros_like(mjx_data.xfrc_applied),
                                                        "impulse_startends_stime" : jnp.full(shape=(self._vec_size, mjx_model.nbody, 2), fill_value=-1) })
        
        # self._check_model_inaxes()        
        
        # ggLog.info(f"compiled")
        # self._check_model_inaxes()        
        
        self.set_monitored_joints([])
        self.set_monitored_links([])
        self._sim_state = self._reset_monitored_data_and_stats(self._sim_state, self._sim_conf)



        if self._show_gui:
            self._viewer_mj_data : mjutils.MjData = mjx.get_data(self._mj_model, jax.tree_util.tree_map(lambda l: l[self._gui_env_index], self._sim_state.mjx_data))
            mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_util.tree_map(lambda l: l[self._gui_env_index], self._sim_state.mjx_data))
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._viewer_mj_data)

        self._camera_sizes_hw_by_id : dict[int,tuple[int,int]] = {cid:(self._mj_model.cam_resolution[cid][1],self._mj_model.cam_resolution[cid][0]) for cid in self._cid2cname}
        self._camera_sizes_hw :dict[str,tuple[int,int]] = {self._cid2cname[cid]:hw for cid, hw in self._camera_sizes_hw_by_id.items()}
        if self._enable_rendering:
            if self._render_backend == "cpu":
                def make_renderer(h,w):
                    ggLog.info(f"Making CPU renderer for size {h}x{w} MUJOCO_GL='{os.environ['MUJOCO_GL']}' MUJOCO_EGL_DEVICE_ID='{os.environ.get('MUJOCO_EGL_DEVICE_ID', None)}' (set this to select manually the device)")
                    # If you are having issues with the renderer trying to use a card that it cannot access 
                    # (e.g. an integrated GPU without proper permissions), you can try somthing like this:
                    # sudo setfacl -m u:crizz:rw /dev/dri/renderD128
                    # To be sure what exact device path to use you can navigate the folders
                    # Otherwise you can alsoe set MUJOCO_EGL_DEVICE_ID to force egl to use a certain device
                    # You can see the egl devices with eglinfo -B
                    # If things get stuck you may need : apt-get install -y   libegl1-mesa-dev libgl1-mesa-dri mesa-utils mesa-utils-bin
                    return mujoco.Renderer(self._mj_model,height=h,width=w)
                self._render_scene_option = mjutils._MjvOption()
                # self._render_scene_option.flags[mjutils._mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                # self._render_scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1
                # self._render_scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
                self._renderers : dict[tuple[int,int],mujoco.Renderer]= {resolution:make_renderer(resolution[0],resolution[1])
                                for resolution in set(self._camera_sizes_hw.values())}
                self._renderers_mj_datas : list[mjutils.MjData] = [copy.deepcopy(self._mj_data) for _ in range(self.vec_size())]
            elif self._render_backend == "warp":
                self._renderers = {}
                self._renderers_mj_datas = []
                self._init_warp_render_context()
            else:
                raise RuntimeError(f"Unknown render backend '{self._render_backend}'")
        else:
            self._renderers = {}
        self._visualize_xfrc_applied = True

        self._precompute_depth_cam_params()

        self._lid2geoms : dict[int,jnp.ndarray] = {}
        all_links = list(self._lname2lid.keys())
        for lname in all_links:
            body_id = self._lname2lid[lname]
            if body_id >= self._nbody:
                continue  # sites have no geoms
            self._lid2geoms[body_id] = self._sim_state.mjx_model.body_geomadr[body_id:body_id+self._sim_state.mjx_model.body_geomnum[body_id]]
        self._is_geom_visual = jnp.logical_and(self._mj_model.geom_contype==0, self._mj_model.geom_conaffinity==0)


        ggLog.info(f"MJXAdapter: links  lname2lid = {pprint.pformat(self._lname2lid)}")
        ggLog.info(f"MJXAdapter: joints jname2jid = {pprint.pformat(self._jname2jid)}")
        ggLog.info(f"MJXAdapter: {self._mj_model.ncam} cameras cname2cid = {pprint.pformat(self._cname2cid)}")

        ggLog.info(f"MJXAdapter: Joint limits:\n"+("\n".join([f" - {jn}: {r}" for jn,r in {jname:self._mj_model.jnt_range[jid] for jid,jname in self._jid2jname.items()}.items()])))
        ggLog.info(f"MJXAdapter: Joint child bodies:\n"+("\n".join([f" - {jn}: {r}" for jn,r in {jname:self._mj_model.jnt_bodyid[jid] for jid,jname in self._jid2jname.items()}.items()])))
        ggLog.info(f"MJXAdapter: dof armatures:{self._mj_model.dof_armature}")
        
        ggLog.info(f"MJXAdapter: Bodies parentid:\n"+("\n".join([f" - body_parentid[{lid}({self._lid2lname[lid]})]= {self._mj_model.body_parentid[lid]}" for lid in self._lid2lname.keys() if lid < self._nbody])))
        ggLog.info(f"MJXAdapter: Bodies jnt_num:\n"+("\n".join([f" - body_jntnum[{lid}({self._lid2lname[lid]})]= {self._mj_model.body_jntnum[lid]}" for lid in self._lid2lname.keys() if lid < self._nbody])))
        
        # print(f"got cam resolutions {self._camera_sizes}")
        # self._check_model_inaxes()        
        # ggLog.info(f"self._sim_state.mj_model.nconmax = {self._mj_model.nconmax}")
        # ggLog.info(f"self._sim_state.mjx_model.nconmax = {self._sim_state.mjx_model.nconmax}")
        # ggLog.info(f"self._sim_state.mjx_data.contact.geom.shape = {self._sim_state.mjx_data.contact.geom.shape}")

        self._scenario_built = True


    def startup(self):
        ggLog.info(f"Compiling mjx.forward....")
        data = self._mjx_forward(self._sim_state.mjx_model, self._sim_state.mjx_data)
        ggLog.info(f"Compiled forward.")
        self._sim_state = self._sim_state.replace_v("mjx_data", data) # compute initial mjData
        ggLog.info(f"Compiling mjx_integrate_and_forward....")
        _ = self._mjx_integrate_and_forward(self._sim_state.mjx_model, copy.deepcopy(self._sim_state.mjx_data)) # trigger jit compile
        ggLog.info(f"Compiled.")

    def _compute_collision_masks(self,  link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]],
                                        explicit_groups : list[tuple[tuple[str,str],...]] = []) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """_summary_

        Parameters
        ----------
        link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]]
            List of pairs (link, colliding_links) specifying what links each single links collides with
        explicit_groups : list[tuple[tuple[str,str],...]], optional
            List of explicitly defined collision groups

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            _description_

        Raises
        ------
        RuntimeError
            _description_
        """
        input_collision_groups = [set(lg[1]) for lg in link_group_collisions]
        ggLog.info(f"input link_group_collisions = {link_group_collisions}")
        ggLog.info(f"input_collision_groups = {input_collision_groups}")

        # Reorganize the links in a small set of groups of links that always collide together
        best_collision_groups : list[set[tuple[str,str]]] = []
        while len(input_collision_groups)>0:
            biggest_common_subgroup = set(input_collision_groups[0])
            for g in input_collision_groups:
                prev_biggest_common_subgroup = biggest_common_subgroup
                biggest_common_subgroup = biggest_common_subgroup.intersection(g)
                if len(biggest_common_subgroup)==0:
                    # Then there is not common subgroup, use the one that was common up to now
                    biggest_common_subgroup = prev_biggest_common_subgroup
            best_collision_groups.append(biggest_common_subgroup)
            input_collision_groups = [(g.difference(biggest_common_subgroup)) for g in input_collision_groups] # remove subgroup
            input_collision_groups = [g for g in input_collision_groups if len(g)>0] # remove empty groups
            # print(f"biggest_common_subgroup = {biggest_common_subgroup}")
            # print(f"input_collision_group = {input_collision_groups}")
            # time.sleep(1)


        best_collision_groups_set = {tuple(g) for g in best_collision_groups}
        best_collision_groups = [set(g) for g in best_collision_groups_set.union(set(explicit_groups))]
        ggLog.info(f"best_collision_groups = {best_collision_groups}")
        link_to_group_ids = {} # Which groups each link is part of
        group_to_links = {}
        self._linkgroup_to_id : dict[tuple[tuple[str,str],...], int] = {}
        for i,g in enumerate(best_collision_groups):
            g_t = tuple(g)
            self._linkgroup_to_id[g_t] = i
            group_to_links[g_t] = []
            for l in g:
                if l not in link_to_group_ids:
                    link_to_group_ids[l] = []
                link_to_group_ids[l].append(i)
                group_to_links[g_t].append(l)
        ggLog.info(f"link_to_groups = \n{pprint.pformat(link_to_group_ids)}")
        ggLog.info(f"group_to_links = \n{pprint.pformat(group_to_links)}")

        link_colliding_groups : dict[tuple[str,str], list[int]] = {} # Which groups each link collides with
        for link,colliding_links in link_group_collisions:
            colliding_links = set(colliding_links)
            for i,g in enumerate(best_collision_groups):
                if g.issubset(colliding_links):
                    # then link collides with the group g
                    if link not in link_colliding_groups:
                        link_colliding_groups[link] = []
                    link_colliding_groups[link].append(i)
        ggLog.info(f"link_colliding_groups = \n{ pprint.pformat(link_colliding_groups)}")

        if len(best_collision_groups) > 32:
            raise RuntimeError(f"Detected more than 32 separate collision groups. Cannot represent in Mujoco collision masks.")
        
        all_links = list(self._lname2lid.keys()) #list(link_to_groups.keys())+list(link_colliding_groups.keys())
        for l in all_links:
            if l not in link_to_group_ids:
                link_to_group_ids[l] = []
            if l not in link_colliding_groups:
                link_colliding_groups[l] = []
        
        for l in link_to_group_ids.keys():
            if l not in all_links:
                raise RuntimeError(f"Link {l} specified as a group in link_group_collisions is not present in the model")
        for l in link_colliding_groups.keys():
            if l not in all_links:
                raise RuntimeError(f"Link {l} specified in link_group_collisions is not present in the model")
        link_contypes = {}
        link_conaffinity = {}
        for l in all_links:
            if self._lname2lid[l] >= self._nbody:
                continue  # skip sites, they don't participate in collisions
            contype_mask = 0
            for gid in link_to_group_ids[l]:
                contype_mask |= 1<<gid
            link_contypes[l] = contype_mask
            conaffinity_mask = 0
            for gid in link_colliding_groups[l]:
                conaffinity_mask |= 1<<gid
            link_conaffinity[l] = conaffinity_mask

        

        # print(f"self._mj_model.geom_contype =     {self._mj_model.geom_contype}")
        # print(f"self._mj_model.geom_conaffinity = {self._mj_model.geom_conaffinity}")
        # print(f"self._sim_state.mjx_model.geom_contype =     {self._sim_state.mjx_model.geom_contype}")
        # print(f"self._sim_state.mjx_model.geom_conaffinity = {self._sim_state.mjx_model.geom_conaffinity}")

        body_contype : jnp.ndarray = self._mj_model.body_contype.copy()
        body_conaffinity : jnp.ndarray = self._mj_model.body_conaffinity.copy()
        geom_contype : jnp.ndarray = self._mj_model.geom_contype.copy()
        geom_conaffinity : jnp.ndarray = self._mj_model.geom_conaffinity.copy()
        for lname in all_links:
            body_id = self._lname2lid[lname]
            if body_id >= self._nbody:
                continue  # skip sites
            for geom_id in range(self._mj_model.body_geomadr[body_id],
                                 self._mj_model.body_geomadr[body_id]+self._mj_model.body_geomnum[body_id]):
                visual = geom_contype[geom_id]==0 and geom_conaffinity[geom_id]==0
        for lname in all_links:
            body_id = self._lname2lid[lname]
            if body_id >= self._nbody:
                continue  # skip sites
            aff = link_conaffinity[lname]
            typ = link_contypes[lname]
            body_conaffinity[body_id] = aff
            body_contype[body_id] = typ
            for geom_id in range(self._mj_model.body_geomadr[body_id],
                                 self._mj_model.body_geomadr[body_id]+self._mj_model.body_geomnum[body_id]):
                # this will pass also through the visual geoms!
                # geom_name = mujoco_mj_id2name(self._mj_model, mujoco_mjtObj.mjOBJ_GEOM, geom_id)
                # geom_rgba = self._mj_model.geom_rgba[geom_id]
                # ggLog.info(f"{lname} [{body_id}] : {geom_id} [{geom_name}]: coll={typ:b}/{aff:b} rgba = {geom_rgba}, aff = {aff}, typ = {typ} ")
                # geom_contype[geom_id] = 1
                visual = geom_contype[geom_id]==0 and geom_conaffinity[geom_id]==0
                if not visual:
                    geom_conaffinity[geom_id] = aff
                    geom_contype[geom_id] = typ
        return geom_contype, geom_conaffinity, body_contype, body_conaffinity
    
    @override
    def set_body_collisions(self,   link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]],
                                    explicit_groups : list[tuple[tuple[str,str],...]] = []):
        geom_contype, geom_conaffinity, body_contype, body_conaffinity = self._compute_collision_masks(link_group_collisions, explicit_groups)
        self._mj_model.geom_contype = geom_contype
        self._mj_model.geom_conaffinity = geom_conaffinity
        self._mj_model.body_contype = body_contype
        self._mj_model.body_conaffinity = body_conaffinity
        ggLog.info(f"Previous geom_contype =     {self._sim_state.mjx_model.geom_contype}")
        ggLog.info(f"Previous geom_conaffinity = {self._sim_state.mjx_model.geom_conaffinity}")
        self._sim_state = self._sim_state.replace_v("mjx_model" , self._sim_state.mjx_model.replace(  
                                                                                            geom_contype = geom_contype,
                                                                                            geom_conaffinity = geom_conaffinity,
                                                                                            body_contype = body_contype,
                                                                                            body_conaffinity = body_conaffinity))
        self._original_mjx_model = self._original_mjx_model.replace(geom_contype = geom_contype,
                                                  geom_conaffinity = geom_conaffinity,
                                                  body_contype = body_contype,
                                                  body_conaffinity = body_conaffinity)
        # print(f"self._mj_model.geom_contype =     {self._mj_model.geom_contype}")
        # print(f"self._mj_model.geom_conaffinity = {self._mj_model.geom_conaffinity}")
        ggLog.info(f"New geom_contype =     {self._sim_state.mjx_model.geom_contype}")
        ggLog.info(f"New geom_conaffinity = {self._sim_state.mjx_model.geom_conaffinity}")
        new_mjxdata = mjx.put_data(self._mj_model, self._mj_data, device = self._jax_device, impl=self._mjx_impl)
        if jnp.any(new_mjxdata.nefc > self._sim_state.mjx_data.nefc):
            # maybe something could be done here by regenereating the mjx_data and coping values from the old one
            raise RuntimeError(f"New collision setup requires a higher number of efc constraints than"
                               f" the initial one ({new_mjxdata.nefc} > {self._sim_state.mjx_data.nefc}), this is not supported yet. ")
        self._mark_forward_needed()
        self._recompute_mjxmodel_inaxes(self._sim_state.mjx_model)
        self._rebuild_lower_funcs()
        self._check_model_inaxes()        



    @override
    def get_detected_joints(self):
        return list(self._jname2jid.keys())
    
    @override
    def get_detected_joints_properties(self) -> dict[tuple[str,str],JointProperties]:
        return {jn:JointProperties(joint_type=mj_jnt_type_to_adarl[self._mj_model.jnt_type[self._jname2jid[jn]]])
                for jn in self.get_detected_joints()}
    
    @override
    def get_detected_links(self):
        return list(self._lname2lid.keys())
    
    @override
    def get_detected_cameras(self):
        return list(self._cname2cid.keys())

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str,str]]):
        super().set_monitored_joints(jointsToObserve)
        # Cache Python versions to avoid JAX->numpy sync in getters
        monitored_jids_list = [self._jname2jid[jn] for jn in self._monitored_joints]
        self._monitored_jid_to_idx = {jid: idx for idx, jid in enumerate(monitored_jids_list)}
        self._sim_conf.monitored_jids = jnp.array(monitored_jids_list, device=self._jax_device, dtype=jnp.int32)
        self._sim_conf.monitored_qpadr = self._sim_conf.jnt_qposadr[self._sim_conf.monitored_jids]
        self._sim_conf.monitored_qvadr = self._sim_conf.jnt_dofadr[self._sim_conf.monitored_jids]
        self._rebuild_step_stats_arrs()
        self._sim_state = self._reset_monitored_data_and_stats(self._sim_state, self._sim_conf)
        if self._record_joint_hist:
            self._full_history_labels = to_string_tensor(sum([[f"{jn[1]}.{v}" for v in ["pos","vel","cmd_eff","acc","eff","constr_eff"]] 
                                                              for jn in self._monitored_joints],[])).unsqueeze(0)

    @override
    def set_monitored_links(self, linksToObserve: Sequence[tuple[str,str]]):
        super().set_monitored_links(linksToObserve)
        # Cache Python versions to avoid JAX->numpy sync in getters
        # Unified lid includes both bodies (lid < nbody) and sites (lid = nbody + sid)
        monitored_lids_list = [self._lname2lid[ln] for ln in self._monitored_links]
        # Partition into body IDs and site IDs (in MuJoCo space), preserving relative order
        monitored_body_ids = [lid for lid in monitored_lids_list if lid < self._nbody]
        monitored_site_ids = [lid - self._nbody for lid in monitored_lids_list if lid >= self._nbody]
        self._sim_conf.monitored_lids = jnp.array(monitored_body_ids, device=self._jax_device, dtype=jnp.int32)
        self._sim_conf.monitored_sids = jnp.array(monitored_site_ids, device=self._jax_device, dtype=jnp.int32)
        self._mon_body_count = len(monitored_body_ids)
        self._mon_site_count = len(monitored_site_ids)
        # JAX internally stores [bodies|sites]. Build permutation from JAX layout -> user order.
        # For each user-order index, find its position in the JAX [bodies|sites] layout.
        body_idx = 0  # running index into the bodies portion
        site_idx = 0  # running index into the sites portion
        jax_to_user = [0] * len(monitored_lids_list)  # jax_layout_idx -> user_idx
        user_to_jax = [0] * len(monitored_lids_list)  # user_idx -> jax_layout_idx
        for user_idx, lid in enumerate(monitored_lids_list):
            if lid < self._nbody:
                jax_idx = body_idx
                body_idx += 1
            else:
                jax_idx = self._mon_body_count + site_idx
                site_idx += 1
            user_to_jax[user_idx] = jax_idx
            jax_to_user[jax_idx] = user_idx
        self._mon_link_jax_to_user = th.tensor(jax_to_user, dtype=th.long, device=self._out_th_device)
        # _monitored_lid_to_idx maps unified lid -> user-order index (for get_monitored_links_ids)
        self._monitored_lid_to_idx = {lid: idx for idx, lid in enumerate(monitored_lids_list)}
        # _monitored_links stays in user order (set by super())
        self._rebuild_step_stats_arrs()

    @override
    def set_monitored_collision_pairs(self, collision_pairs: Sequence[tuple[tuple[str,str], tuple[str,str]]]):
        """Set collision pairs to monitor. Must be called before build_scenario.

        Pairs are buffered here and resolved to body ids / sensor adrs inside build_scenario,
        once the model is compiled and the lname2lid map exists. The warp backend additionally
        gets one mjSENS_CONTACT sensor per pair injected into the spec.

        Parameters
        ----------
        collision_pairs : Sequence[tuple[tuple[str,str], tuple[str,str]]]
            List of link name pairs to monitor for collisions.
            Each pair is ((model_a, link_a), (model_b, link_b)).
        """
        if self._scenario_built:
            raise RuntimeError("Monitored collision pairs cannot be changed after build_scenario has been called.")
        self._monitored_collision_pairs = list(collision_pairs)

    def get_collision_pair_ids(self, collision_pairs: Sequence[tuple[tuple[str,str], tuple[str,str]]]) -> th.Tensor:
        """Get indices into monitored collision pairs array.
        
        Parameters
        ----------
        collision_pairs : Sequence[tuple[tuple[str,str], tuple[str,str]]]
            List of link name pairs to look up.
            
        Returns
        -------
        th.Tensor
            Long tensor of indices into the monitored collision pairs array.
        """
        indices = []
        for pair in collision_pairs:
            body_a = self._lname2lid[pair[0]]
            body_b = self._lname2lid[pair[1]]
            # Try both directions
            idx = self._monitored_collision_pair_to_idx.get((body_a, body_b))
            if idx is None:
                raise KeyError(f"Collision pair {pair} is not in monitored collision pairs")
            indices.append(idx)
        return th.as_tensor(indices, dtype=th.long).to(self._out_th_device, non_blocking=self._out_cuda)

    def get_collision_pair_names(self, pair_ids: th.Tensor) -> list[tuple[tuple[str,str], tuple[str,str]]]:
        """Get collision pair names from indices.
        
        Parameters
        ----------
        pair_ids : th.Tensor
            Indices into monitored collision pairs array.
            
        Returns
        -------
        list[tuple[tuple[str,str], tuple[str,str]]]
            List of link name pairs.
        """
        return [self._monitored_collision_pairs[idx] for idx in pair_ids.tolist()]

    @override
    def initialize_for_step(self):
        self._must_init_step = True

    @override
    def step(self) -> float:
        """Run a simulation step.

        Returns
        -------
        float
            Duration of the step in simulation time (in seconds)"""
        record_region_start("MjxAdapter.step")
        self.initialize_for_step()
        record_time("MjxAdapter.step: initialized for step")
        stepLength = self.run(self._step_length_sec)
        # ggLog.info(f"Mjx.step duration = {tf-t0}s, run = {tf-t1}s")
        record_region_end("MjxAdapter.step")
        return stepLength
    
    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_commands(self, sim_state : SimState) -> SimState:
        sim_state = self._apply_torque_cmds(sim_state)
        return sim_state

    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_torque_cmds(self, sim_state : SimState) -> SimState:
        return sim_state.replace_v( "mjx_data", sim_state.mjx_data.replace(qfrc_applied=sim_state.requested_qfrc_applied))


    # @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state"))
    def _sim_step_fast_for_scan_full_pveae(self, sim_state_conf : tuple[SimState,SimConf], _) -> tuple[tuple[SimState,SimConf], jnp.ndarray | None]:
        sim_state, sim_conf = sim_state_conf
        if self._record_joint_hist:
            # Save pre-forward kinematics (the state the controller acted on)
            pre_pos = sim_state.mjx_data.qpos[:, sim_conf.monitored_qpadr]
            pre_vel = sim_state.mjx_data.qvel[:, sim_conf.monitored_qvadr]
        # Full step: apply_commands, apply_impulses, integrate_and_forward, update caches/stats
        sim_state = self._apply_commands(sim_state)
        sim_state = self._apply_impulses(sim_state)
        new_mjx_data = self._mjx_integrate_and_forward(sim_state.mjx_model, sim_state.mjx_data)
        sim_state = sim_state.replace_d({"mjx_data": new_mjx_data,
                                          "sim_time": sim_state.sim_time + self._sim_step_dt})
        sim_state = self._update_monitored_data_cache(sim_state, sim_conf)
        sim_state = MjxAdapter._update_step_stats(sim_state, sim_conf)
        if self._record_joint_hist:
            joint_state = self._assemble_base_joint_history(sim_state.mjx_data, sim_conf, pre_pos, pre_vel)
        else:
            joint_state = None
        return (sim_state, sim_conf), joint_state

    def _assemble_base_joint_history(self, mjx_data, sim_conf : SimConf, pre_pos : jnp.ndarray, pre_vel : jnp.ndarray) -> jnp.ndarray:
        """Assemble 6-column joint history (pveaec) from pre-forward pos/vel and post-forward real forces."""
        qvadr = sim_conf.monitored_qvadr
        qpadr = sim_conf.monitored_qpadr

        current_pveaec = self._get_vec_joint_states_raw_pveaec(qpadr, qvadr, mjx_data)
        pveaec = current_pveaec.at[..., 0].set(pre_pos)
        pveaec = pveaec.at[..., 1].set(pre_vel)

        return pveaec
    
    @partial(jax.jit, static_argnames=("self","iterations","must_init_step"), donate_argnames=("sim_state"))
    def _run_fast_save_full_jpveae(self, sim_state : SimState, iterations : int, sim_conf : SimConf, must_init_step : bool) -> tuple[SimState, jnp.ndarray | None]:
        if must_init_step:
            sim_state = MjxAdapter._clear_step_stats(sim_state)
        (sim_state, sim_conf), joints_state_history = jax.lax.scan(self._sim_step_fast_for_scan_full_pveae, 
                                                  init = (sim_state, sim_conf),
                                                  xs = (), 
                                                  length = iterations)
        # sim_state, joint_stats_arr  = jax.lax.fori_loop(lower=0, upper=iterations,
        #                                                 body_fun=self._sim_step_fast,
        #                                                 init_val=(sim_state, joint_stats_arr))
        return sim_state, joints_state_history

    def _get_joint_state_for_history(self, sim_state : SimState) -> jnp.ndarray:
        return MjxAdapter._get_vec_joint_states_raw_pveaec( self._sim_conf.monitored_qpadr,
                                                            self._sim_conf.monitored_qvadr,
                                                            sim_state.mjx_data)
    
    def _log_joints_pveae_history(self):
        full_history = jnp.concat(self._joints_pveae_history)
        full_history = jnp.reshape(full_history,shape=(full_history.shape[0],-1))
        # convert to torch and save as hdf5
        dir = f"{self._log_folder}/MjxAdapter_joint_hist"
        os.makedirs(dir, exist_ok=True)
        run_id = default_session.run_info["run_id"]
        out_filename = f"{dir}/MjxAdapter_{run_id}_{self._total_iterations}.hdf5"
        import h5py
        with h5py.File(out_filename, "w") as f:
            # loop through obs, action, reward, terminated, truncation
            try:
                f.create_dataset("joints_history", data=np.array(full_history))
                f.create_dataset("joints_history_labels", data=self._full_history_labels)
            except TypeError as e:
                raise RuntimeError(f"Error saving pveae, exception={e}")

    @override
    def run(self, duration_sec : float):
        """Run the environment for the specified duration"""
        record_region_start("MjxAdapter.run")
        run_fast_jit_compiles = self._run_fast_save_full_jpveae._cache_size() # type: ignore
        wt0 = time.monotonic()
        st0 = self._simTime
        # self._sent_motor_torque_commands_by_bid_jid = {}
        # ggLog.info(f"Starting run")
        iterations = int(duration_sec/self._sim_step_dt)
        record_time("MjxAdapter.run: before fast step")
        self._sim_state, joints_state_history = self._run_fast_save_full_jpveae(self._sim_state, iterations, self._sim_conf, self._must_init_step)
        record_time("MjxAdapter.run: after fast step")
        self._must_init_step = False
        self._forward_needed = False
        if self._record_joint_hist:
            self._joints_pveae_history.append(joints_state_history)
            # ggLog.info(f"int({self._total_iterations} / {self._log_freq_joints_trajcetories}) = {int(self._total_iterations / self._log_freq_joints_trajcetories)} != {int((self._total_iterations+iterations) / self._log_freq_joints_trajcetories)} = {int(self._total_iterations / self._log_freq_joints_trajcetories) != int((self._total_iterations+iterations) / self._log_freq_joints_trajcetories)}")
            if int(self._total_iterations / self._log_freq_joints_trajcetories) != int((self._total_iterations+iterations) / self._log_freq_joints_trajcetories):
                self._log_joints_pveae_history()
                self._joints_pveae_history = []
        # self._read_new_contacts()
        wtime_simulating = time.monotonic()-wt0
        # ggLog.info(f"run done.")
        if self._run_fast_save_full_jpveae._cache_size() > run_fast_jit_compiles and run_fast_jit_compiles>0: # type: ignore
            ggLog.warn(f"run_fast was recompiled")

        self._total_iterations += iterations
        self._simTime += iterations*self._sim_step_dt # faster than taking sim_state.sim_time, as it does not do a sync
        
        # print(f"self._simTime={self._simTime}, mjData.time={self._sim_state.mjx_data.time}")

        if self._realtime_factor is not None and self._realtime_factor>0:
            sleep_time = self._sim_step_dt*(1/self._realtime_factor) - (time.monotonic()-self._prev_step_end_wtime)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._prev_step_end_wtime = time.monotonic()
        self._update_gui()
        # self._last_sent_torques_by_name = {self._bodyAndJointIdToJointName[bid_jid]:torque 
        #                                     for bid_jid,torque in self._sent_motor_torque_commands_by_bid_jid.items()}
        tf = time.monotonic()
        self._dbg_info.wtime_running =     tf-wt0
        self._dbg_info.wtime_running_since_build +=     tf-wt0
        self._dbg_info.wtime_simulating =   wtime_simulating
        self._dbg_info.wtime_simulating_since_build +=   wtime_simulating
        self._dbg_info.wtime_controlling =  0
        self._dbg_info.iterations = iterations
        self._dbg_info.rt_factor_vec =      self._vec_size*(self._simTime-st0)/wtime_simulating
        self._dbg_info.rt_factor_single =   (self._simTime-st0)/wtime_simulating
        self._dbg_info.stime_ran =          self._simTime-st0
        self._dbg_info.stime =              self._simTime
        self._dbg_info.fps_vec =            self._vec_size*iterations/wtime_simulating
        self._dbg_info.fps_single =         iterations/wtime_simulating
        self._dbg_info.run_fps_vec =        self._dbg_info.fps_vec/iterations
        if self._log_freq > 0 and self._sim_step_count_since_build - self._last_log_iters >= self._log_freq:
            self._last_log_iters = self._sim_step_count_since_build
            ggLog.info( "MjxAdapter:\n"+"\n".join(["    "+str(k)+' : '+str(v) for k,v in self.get_debug_info().items()]))
        self._sim_step_count_since_build += iterations
        record_region_end("MjxAdapter.run")
        return self._simTime-st0
    
    
    def get_debug_info(self) -> dict[str,th.Tensor]:
        return {k:th.as_tensor(v) for k,v in dataclasses.asdict(self._dbg_info).items()}


    def _render_rgb(self,   requestedCameras : list[str],
                            vec_mask : th.Tensor,
                            out_th_device : th.device,
                            out : list[th.Tensor] | None = None,
                            use_fixed_shapes : bool = False):
        if use_fixed_shapes:
            selected_vecs = list(range(self._vec_size))
            nvecs = self._vec_size
            active_vecs = vec_mask.to(device="cpu", non_blocking=False).tolist()
        else:
            selected_vecs = th.nonzero(vec_mask, as_tuple=True)[0].to("cpu").tolist()
            nvecs = len(selected_vecs)
            active_vecs = [] # made into a list just for typing
        t0 = time.monotonic()
        
        # mj_data_batch = mjx.get_data(self._mj_model, self._sim_state.mjx_data)
        # print(f"mj_data_batch = {mj_data_batch}")
        times = th.as_tensor(self._simTime).repeat((nvecs,len(requestedCameras))).to(out_th_device, non_blocking=out_th_device.type=="cuda")
        cam_shapes_hw = [(int(self._camera_sizes_hw[cam][0]),int(self._camera_sizes_hw[cam][1])) for cam in requestedCameras]
        arr_shapes = [(nvecs,)+cam_shapes_hw[i]+(3,) for i in range(len(requestedCameras))]
        if out is not None:
            if len(out) != len(requestedCameras):
                raise RuntimeError(f"getRenderings: len(out)={len(out)} != len(requestedCameras)={len(requestedCameras)}")
            for i in range(len(requestedCameras)):
                if out[i].shape != arr_shapes[i]:
                    raise RuntimeError(f"getRenderings: out[{i}].shape={out[i].shape} != expected shape={arr_shapes[i]}")
                if out[i].device.type != "cpu":
                    raise RuntimeError(f"getRenderings: out[{i}].device={out[i].device} != cpu, all output tensors must be on cpu")
            image_batches_hwc = [out[i].numpy() for i in range(len(requestedCameras))]
        else:
            image_batches_hwc = [np.zeros(shape=arr_shapes[i], dtype=np.uint8) for i in range(len(requestedCameras))]
        self._forward_if_needed()
        # print(f"images.shapes = {[i.shape for i in images]}")
        # mj_datas : list[mujoco_MjData] = mjx.get_data(self._mj_model, self._sim_state.mjx_data)
        t_precopy = time.monotonic()
        # get_data_into(self._renderers_mj_datas,self._mj_model, self._sim_state.mjx_data,exclude=["qLD"])
        get_renderdata_into(self._renderers_mj_datas, self._sim_state.mjx_data)
        t_postcopy = time.monotonic()
        tot_copy_time = t_postcopy - t_precopy
        tot_render_time = 0.0
        tot_update_time = 0.0
        # ggLog.info(f"Rendering {nvecs} vecs and {len(requestedCameras)} cameras with resolutions {[self._camera_sizes_hw[cam] for cam in requestedCameras]}...")
        for env_i,env in enumerate(selected_vecs):
            if use_fixed_shapes and not active_vecs[env]:
                continue
            for cam_i in range(len(requestedCameras)):
                cam = requestedCameras[cam_i]
                # print(f"self._mj_model.cam_resolution[cid] = {self._mj_model.cam_resolution[self._cname2cid[cam]]}")
                renderer = self._renderers[self._camera_sizes_hw[cam]]
                mjdata = self._renderers_mj_datas[env]
                mjutils._mj_camlight(self._mj_model, mjdata) # see https://github.com/google-deepmind/mujoco/issues/1806
                t_preupdate = time.monotonic()
                renderer.update_scene(mjdata, self._cname2cid[cam], scene_option=self._render_scene_option)
                if self._visualize_xfrc_applied:
                    for body_id in range(0,self._mj_model.nbody):
                        if np.linalg.norm(mjdata.xfrc_applied[body_id]) != 0.0:
                            # ggLog.info(f"xfrc_applied[{body_id}] = {mjdata.xfrc_applied[body_id]}")
                            force_vec = mjdata.xfrc_applied[body_id,:3]
                            # force = np.linalg.norm(np.linalg.norm(force_vec))
                            # force_quat = quat_xyzw_between_vecs_py(th.as_tensor([1.0,0,0]), th.as_tensor(force_vec, dtype=th.float32)).numpy()
                            body_pos = mjdata.xipos[body_id]
                            # add_geom_to_renderer(renderer,
                            #                         geom_type=mujoco_mjtGeom.mjGEOM_CYLINDER,
                            #                         size_xyz=np.array([0.05, force,0.05]),
                            #                         pos_xyz=body_pos,
                            #                         quat_xyzw=force_quat,
                            #                         rgba=np.array([0.9,0.1,0.1,1.0]))
                            add_arrow_to_renderer(renderer, body_pos, body_pos+force_vec/10, radius=0.03, rgba=[0.8, 0.1, 0.1, 1])
                t_postupdate = time.monotonic()
                t_prerender = time.monotonic()
                renderer.render(out=image_batches_hwc[cam_i][env_i])
                tot_render_time += time.monotonic()-t_prerender
                tot_update_time += t_postupdate - t_preupdate
                # renderer.render(out=images[i][env])
        alpha = 0.999
        curr_avg_render_time = tot_render_time/nvecs/len(requestedCameras)
        curr_avg_render_update_time = tot_update_time/nvecs/len(requestedCameras)
        if self._dbg_info.avg_render_time < 0:
            self._dbg_info.avg_render_time = curr_avg_render_time
            self._dbg_info.avg_render_update_time = curr_avg_render_update_time
        self._dbg_info.avg_render_time = alpha*self._dbg_info.avg_render_time + (1-alpha)*curr_avg_render_time
        self._dbg_info.avg_render_update_time = alpha*self._dbg_info.avg_render_update_time + (1-alpha)*curr_avg_render_update_time
        tf = time.monotonic()
        # ggLog.info(f"getRenderings for {nvecs} vecs and {len(requestedCameras)} cameras took {(tf-t0)*1000:.3f}ms,"
        #            f" rendering took {tot_render_time*1000:.3f}ms,"
        #            f" update took {tot_update_time*1000:.3f}ms,"
        #            f" copy took {tot_copy_time*1000:.3f}ms")
        return [th.as_tensor(img_batch).to(device=out_th_device, non_blocking=out_th_device.type=="cuda") for img_batch in image_batches_hwc], times

    def _render_rgb_warp(self,
                         requestedCameras : list[str],
                         vec_mask : th.Tensor,
                         out_th_device : th.device,
                         out : list[th.Tensor] | None = None,
                         use_fixed_shapes : bool = False):
        if not self._warp_render_has_rgb or self._warp_render_context_pytree is None:
            raise RuntimeError("Warp RGB rendering is not initialized")
        nvecs = self._vec_size if use_fixed_shapes else int(th.count_nonzero(vec_mask).item()) #type: ignore
        times = th.as_tensor(self._simTime).repeat((nvecs, len(requestedCameras))).to(
            out_th_device, non_blocking=out_th_device.type=="cuda"
        )
        self._forward_if_needed()
        rgbs = self._render_warp_jax(self._sim_state,
                                          self._warp_render_context_pytree,
                                          tuple([self._warp_render_cname2idx[cam] for cam in requestedCameras]))
        # ggLog.info(f"got warp render pixels with shape {[i.shape for i in rgbs]} for {nvecs} vecs and {len(requestedCameras)} cameras")

        all_imgs : list[th.Tensor] = []
        for i in range(len(requestedCameras)):
            rgb = rgbs[i]
            rgb_th = jax2th(rgb, th_device=out_th_device)
            vec_mask_dev = vec_mask.to(device=rgb_th.device, non_blocking=rgb_th.device.type=="cuda")
            if use_fixed_shapes:
                rgb_th = rgb_th.clone()
                rgb_th[~vec_mask_dev] = 0
            else:
                rgb_th = rgb_th[vec_mask_dev]
            if out is not None:
                if out[i].shape != rgb_th.shape:
                    raise RuntimeError(f"getRenderings: out[{i}].shape={out[i].shape} != expected shape={rgb_th.shape}")
                out[i].copy_(rgb_th, non_blocking=out_th_device.type=="cuda")
                all_imgs.append(out[i])
            else:
                all_imgs.append(rgb_th)
        return all_imgs, times


    def _render_depth(self, requestedCameras : list[str],
                            vec_mask : th.Tensor,
                            out_th_device : th.device,
                            out : list[th.Tensor] | None = None,
                            use_fixed_shapes : bool = False):
        nvecs : int = self._vec_size if use_fixed_shapes else th.count_nonzero(vec_mask).item() #type: ignore
        times = th.as_tensor(self._simTime).repeat((nvecs,len(requestedCameras))).to(out_th_device, non_blocking=out_th_device.type=="cuda")
        all_imgs = []
        for cam_i, cam in enumerate(requestedCameras):
            imgs = self._get_depth_image(cam, sim_state=self._sim_state).to(device=out_th_device, non_blocking=out_th_device.type=="cuda")
            imgs = imgs.unsqueeze(-1) # add channel dimension
            vec_mask_dev = vec_mask.to(device=imgs.device, non_blocking=imgs.device.type=="cuda")
            if use_fixed_shapes:
                imgs = imgs.clone()
                imgs[~vec_mask_dev] = 0
            else:
                imgs = imgs[vec_mask_dev]
            if out is not None:
                if out[cam_i].shape != imgs.shape:
                    raise RuntimeError(f"getRenderings: out[{cam_i}].shape={out[cam_i].shape} != expected shape={imgs.shape}")
                out[cam_i].copy_(imgs, non_blocking=out_th_device.type=="cuda")
                all_imgs.append(out[cam_i])
            else:
                all_imgs.append(imgs)
        return all_imgs, times

    def _render_depth_warp(self,
                           requestedCameras : list[str],
                           vec_mask : th.Tensor,
                           out_th_device : th.device,
                           out : list[th.Tensor] | None = None,
                           use_fixed_shapes : bool = False):
        if not self._warp_render_has_depth or self._warp_render_context_pytree is None:
            raise RuntimeError("Warp depth rendering is not initialized")
        nvecs = self._vec_size if use_fixed_shapes else int(th.count_nonzero(vec_mask).item()) #type: ignore
        times = th.as_tensor(self._simTime).repeat((nvecs, len(requestedCameras))).to(
            out_th_device, non_blocking=out_th_device.type=="cuda"
        )
        self._forward_if_needed()
        pixels, _ = self._render_warp_jax(self._sim_state)

        all_imgs : list[th.Tensor] = []
        for cam_i, cam in enumerate(requestedCameras):
            cid = self._cname2cid[cam]
            depth = mjx.get_depth(self._warp_render_context_pytree, cid, p)(pixels)
            depth = jnp.expand_dims(depth, axis=-1)
            depth_th = jax2th(depth, th_device=out_th_device)
            vec_mask_dev = vec_mask.to(device=depth_th.device, non_blocking=depth_th.device.type=="cuda")
            if use_fixed_shapes:
                depth_th = depth_th.clone()
                depth_th[~vec_mask_dev] = 0
            else:
                depth_th = depth_th[vec_mask_dev]
            if out is not None:
                if out[cam_i].shape != depth_th.shape:
                    raise RuntimeError(f"getRenderings: out[{cam_i}].shape={out[cam_i].shape} != expected shape={depth_th.shape}")
                out[cam_i].copy_(depth_th, non_blocking=out_th_device.type=="cuda")
                all_imgs.append(out[cam_i])
            else:
                all_imgs.append(depth_th)
        return all_imgs, times
    
    def _depth_img_to_rgb(self, depth_img, maxdist=3.0):
        img_batch = (depth_img/maxdist).to(dtype=th.uint8)
        img_batch = img_batch.unsqueeze(-1).repeat((1,1,1,3))
        return img_batch
    
    @override
    def getRenderings(self,
                      requestedCameras : list[str],
                      vec_mask : th.Tensor | None = None,
                      out_th_device : th.device | None = None,
                      out : list[th.Tensor] | None = None,
                      depth : bool = False,
                      use_fixed_shapes : bool = False) -> tuple[list[th.Tensor], th.Tensor]:
        record_region_start("MjxAdapter.getRenderings")
        if out_th_device is None:
            out_th_device = self._out_th_device
        if not self._enable_rendering:
            raise RuntimeError(f"Called getRenderings, but rendering is not initialized. did you set enable_rendering?")
        if vec_mask is None:
            vec_mask = self._all_vecs_thcpu

        if self._render_backend == "warp":
            if depth:
                r = self._render_depth_warp(requestedCameras, vec_mask, out_th_device, out, use_fixed_shapes)
            else:
                r = self._render_rgb_warp(requestedCameras, vec_mask, out_th_device, out, use_fixed_shapes)
        else:
            if len(self._renderers)==0:
                raise RuntimeError("CPU rendering backend selected, but renderers were not initialized")
            if depth:
                r = self._render_depth(requestedCameras, vec_mask, out_th_device, out, use_fixed_shapes)
            else:
                r = self._render_rgb(requestedCameras, vec_mask, out_th_device, out, use_fixed_shapes)
        record_region_end("MjxAdapter.getRenderings")
        return r


    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | th.Tensor | None = None) -> th.Tensor:
        return self.getExtendedJointsState(requestedJoints)[:,:,:3] # only return pos, vel, cmd_eff, without acc and constr_eff

    
    @override
    def getExtendedJointsState(self, requestedJoints : Sequence[tuple[str,str]] | th.Tensor | None = None) -> th.Tensor:
        if requestedJoints is not None and len(requestedJoints) == 0:
            return th.empty(size=(self._vec_size, 0, 5), dtype=th.float32, device=self._out_th_device)
        
        self._forward_if_needed()
        # Convert to torch first (zero-copy via DLPack), then reorder in torch (lower dispatch overhead than JAX)
        full_state = jax2th(self._sim_state.mon_joint_state_pveae, th_device=self._out_th_device)
        if requestedJoints is None:
            return full_state
        # KeyError will propagate if joint doesn't exist or isn't monitored
        if isinstance(requestedJoints, th.Tensor):
            reorder_indices = requestedJoints
        else:
            reorder_indices = self.get_monitored_joints_ids(requestedJoints)
        return full_state[:, reorder_indices, :]

    def get_sim_elements_state(self,
                               joint_ids : Sequence[tuple[str,str]] | th.Tensor | None = None,
                               link_ids  : Sequence[tuple[str,str]] | th.Tensor | None = None,
                               return_jstates : bool = True,
                               return_lstates : bool = True,
                               return_accelerations : bool = True,
                               return_collisions : bool = True,
                               return_jstats : bool = True,
                               return_lstats : bool = True
                               ) -> SimElementsState:
        """Return the state of all monitored sim elements in a single call.

        Performs a single _forward_if_needed() and four zero-copy DLPack transfers,
        avoiding the repeated Python overhead of calling the individual getters.
        Reordering (if joint_ids / link_ids are provided) is done in torch after the transfer.

        Parameters
        ----------
        joint_ids : sequence of (model, joint) name tuples, pre-resolved th.Tensor of indices, or None.
            Subset and/or reordering of monitored joints to return.  None returns all monitored joints.
        link_ids : sequence of (model, link) name tuples, pre-resolved th.Tensor of indices, or None.
            Subset and/or reordering of monitored links to return.  None returns all monitored links.

        Returns
        -------
        SimElementsState
            joint_state_pveae  : (vec_size, J, 5)      pos, vel, cmd_effort, acc, actual_effort
            link_state         : (vec_size, L, 13)     pos_xyz, ori_xyzw, linvel_xyz, angvel_xyz
            link_acceleration  : (vec_size, L, 3)      local linear acceleration
            collision_mask     : (vec_size, P)          bool contact flags for all monitored pairs
            joint_stats_pvaee  : (vec_size, 4, J, 6)   [min,max,avg,std] of joint state over the step substeps (quantities ordered position, velocity, acceleration, command_effort, actual_effort, power)
            link_stats_v       : (vec_size, 4, L, 6)   [min,max,avg,std] of link linear+angular velocity over the step substeps
        """
        self._forward_if_needed()
        # Zero-copy DLPack transfers for all cached arrays in one shot
        jstate  = jax2th(self._sim_state.mon_joint_state_pveae,          th_device=self._out_th_device) if return_jstates else None
        lstate  = jax2th(self._sim_state.mon_link_state,                  th_device=self._out_th_device) if return_lstates else None
        lacc    = jax2th(self._sim_state.mon_link_acceleration,            th_device=self._out_th_device) if return_accelerations else None
        cmask   = jax2th(self._sim_state.mon_collision_mask,               th_device=self._out_th_device) if return_collisions else None
        jstats  = jax2th(self._sim_state.mon_joint_stats_arr_pvaeep[:, :4], th_device=self._out_th_device) if return_jstats else None  # (V,4,J,6)
        lstats  = jax2th(self._sim_state.mon_links_stats_arr_v[:, :4],     th_device=self._out_th_device) if return_lstats else None  # (V,4,L,6)
        # Reorder joints in torch (avoids JAX retrace on dynamic indices)
        if joint_ids is not None:
            if not isinstance(joint_ids, th.Tensor):
                joint_ids = self.get_monitored_joints_ids(joint_ids)
            if return_jstates:
                jstate  = jstate[:,  joint_ids, :]
            if return_jstats:
                jstats  = jstats[:, :, joint_ids, :]
        # Reorder links in torch
        if link_ids is not None:
            if not isinstance(link_ids, th.Tensor):
                link_ids = self.get_monitored_links_ids(link_ids)
            if return_lstates:
                lstate  = lstate[:,  link_ids, :]
            if return_accelerations:
                lacc    = lacc[:,    link_ids, :]
            if return_lstats:
                lstats  = lstats[:, :, link_ids, :]
        return SimElementsState(joint_state_pveae=jstate,
                                link_state=lstate,
                                link_linacc=lacc,
                                collision_mask=cmask,
                                joint_stats_pvaeep=jstats,
                                link_stats_v=lstats)
    
    @staticmethod
    def _get_vec_joint_states_pveae(sim_conf : SimConf, mjx_data, jids : jnp.ndarray):
        return MjxAdapter._get_vec_joint_states_raw_pveae(  sim_conf.jnt_qposadr[jids],
                                                            sim_conf.jnt_dofadr[jids],
                                                            mjx_data)
    
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pveae(qpadr, qvadr, mjx_data):
        # What should we use as torque readings?
        # - Emo Todorov here https://www.roboti.us/forum/index.php?threads/best-way-to-represent-robots-torque-sensors.4181 says 
        #     that qfrc_unc (now renamed to qfrc_smmooth) + qfrc_constraint shoudl give what a torque sensor would measure
        # - We also could use qfrc_applied, which is the torque we are applying, I think in most cases this should be correct
        # ggLog.info(f"qfrc_applied={mjx_data.qfrc_applied[:,qvadr]}\n"
        #            f"qfrc_smooth={mjx_data.qfrc_smooth[:,qvadr]}\n"
        #            f"qfrc_constraint={mjx_data.qfrc_constraint[:,qvadr]}"
        #            f"qfrc_passive={mjx_data.qfrc_constraint[:,qvadr]}"
        #            f"qfrc_bias={mjx_data.qfrc_constraint[:,qvadr]}")
        return MjxAdapter._get_vec_joint_states_raw_pveaec(qpadr, qvadr, mjx_data)[...,:5]
    
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pveaec(qpadr, qvadr, mjx_data):
        # What should we use as torque readings?
        # - Emo Todorov here https://www.roboti.us/forum/index.php?threads/best-way-to-represent-robots-torque-sensors.4181 says 
        #     that qfrc_unc (now renamed to qfrc_smmooth) + qfrc_constraint shoudl give what a torque sensor would measure
        # - We also could use qfrc_applied, which is the torque we are applying, I think in most cases this should be correct
        # ggLog.info(f"qfrc_applied={mjx_data.qfrc_applied[:,qvadr]}\n"
        #            f"qfrc_smooth={mjx_data.qfrc_smooth[:,qvadr]}\n"
        #            f"qfrc_constraint={mjx_data.qfrc_constraint[:,qvadr]}"
        #            f"qfrc_passive={mjx_data.qfrc_constraint[:,qvadr]}"
        #            f"qfrc_bias={mjx_data.qfrc_constraint[:,qvadr]}")
        commanded_effort = mjx_data.qfrc_applied[:,qvadr] + mjx_data.qfrc_actuator[:,qvadr]
        # sensed_effort = mjx_data.qfrc_smooth[:,qvadr] + mjx_data.qfrc_constraint[:,qvadr]
        sensed_effort = commanded_effort + mjx_data.qfrc_passive[:,qvadr] + mjx_data.qfrc_constraint[:,qvadr]
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            commanded_effort, #commanded effort
                            mjx_data.qacc[:,qvadr],
                            sensed_effort, #actual effort
                            mjx_data.qfrc_constraint[:,qvadr]],
                            axis = 2)
    
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pvae(qpadr, qvadr, mjx_data):
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qacc[:,qvadr],
                            mjx_data.qfrc_applied[:,qvadr]], 
                            axis = 2)

    @staticmethod
    def _clear_step_stats(sim_state : SimState):
        return MjxAdapter._init_stats(sim_state)

    @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"])
    def _init_stats(sim_state : SimState):
        joint_stats_array = sim_state.mon_joint_stats_arr_pvaeep
        joint_stats_array = joint_stats_array.at[:,0].set(float("+inf")) # mins
        joint_stats_array = joint_stats_array.at[:,1].set(float("-inf")) # maxes
        joint_stats_array = joint_stats_array.at[:,2].set(0) # avg
        joint_stats_array = joint_stats_array.at[:,3].set(0) # std
        joint_stats_array = joint_stats_array.at[:,4].set(0)
        joint_stats_array = joint_stats_array.at[:,5].set(0)
        links_stats_array = sim_state.mon_links_stats_arr_v
        links_stats_array = links_stats_array.at[:,0].set(float("+inf")) # mins
        links_stats_array = links_stats_array.at[:,1].set(float("-inf")) # maxes
        links_stats_array = links_stats_array.at[:,2].set(0) # avg
        links_stats_array = links_stats_array.at[:,3].set(0) # std
        links_stats_array = links_stats_array.at[:,4].set(0)
        links_stats_array = links_stats_array.at[:,5].set(0)

        sim_state = sim_state.replace_d({"stats_step_count": 0,
                                         "mon_joint_stats_arr_pvaeep" : joint_stats_array,
                                         "mon_links_stats_arr_v" : links_stats_array})
        return sim_state

    @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"])
    def _update_step_stats_arrs(current_jstate_pvae : jnp.ndarray,
                                current_lstate_v : jnp.ndarray,
                                sim_state : SimState):
        current_jstate_pvaee = current_jstate_pvae
        step_count = sim_state.stats_step_count + 1
        joint_stats_array = sim_state.mon_joint_stats_arr_pvaeep
        jvel = current_jstate_pvaee[:,:,1]
        jtorque = current_jstate_pvaee[:,:,2]
        power_spent = jnp.clip(jvel * jtorque, 0.0, 1e6)
        current_jstate_pvaeep   = jnp.concatenate([current_jstate_pvaee, power_spent[:,:,None]], axis=2)
        values_sum              = jnp.add(    joint_stats_array[:,4], current_jstate_pvaeep)
        values_sum_of_squares   = jnp.add(    joint_stats_array[:,5], jnp.square(current_jstate_pvaeep))
        values_min              = jnp.minimum(joint_stats_array[:,0], current_jstate_pvaeep)
        values_max              = jnp.maximum(joint_stats_array[:,1], current_jstate_pvaeep)
        values_avg              = values_sum/step_count
        values_std              = jnp.sqrt(jnp.clip(values_sum_of_squares/step_count-jnp.square(values_avg),min=0))
        joint_stats_array = joint_stats_array.at[:,0].set(values_min) # min
        joint_stats_array = joint_stats_array.at[:,1].set(values_max) # max
        joint_stats_array = joint_stats_array.at[:,2].set(values_avg) # average values
        joint_stats_array = joint_stats_array.at[:,3].set(values_std) # standard deviation
        joint_stats_array = joint_stats_array.at[:,4].set(values_sum) # sum of values
        joint_stats_array = joint_stats_array.at[:,5].set(values_sum_of_squares) # sum of squares

        link_stats_array = sim_state.mon_links_stats_arr_v
        link_stats_array = link_stats_array.at[:,4].set(jnp.add(    link_stats_array[:,4], current_lstate_v)) # sum of values
        link_stats_array = link_stats_array.at[:,5].set(jnp.add(    link_stats_array[:,5], jnp.square(current_lstate_v))) # sum of squares
        link_stats_array = link_stats_array.at[:,0].set(jnp.minimum(link_stats_array[:,0], current_lstate_v)) # maybe it would be better to do this not per-component
        link_stats_array = link_stats_array.at[:,1].set(jnp.maximum(link_stats_array[:,1], current_lstate_v))
        link_stats_array = link_stats_array.at[:,2].set(link_stats_array[:,4]/step_count) # average values
        link_stats_array = link_stats_array.at[:,3].set(jnp.sqrt(jnp.clip(link_stats_array[:,5]/step_count-jnp.square(link_stats_array[:,2]),min=0))) # standard deviation


        sim_state = sim_state.replace_d({"stats_step_count" : step_count,
                                         "mon_joint_stats_arr_pvaeep" : joint_stats_array,
                                         "mon_links_stats_arr_v" : link_stats_array})
        # jax.debug.print("updated stats: count={c}, arr={arr}", c=step_count, arr=stats_array)
        return sim_state


    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _update_monitored_data_cache(self, sim_state : SimState, sim_conf : SimConf) -> SimState:
        replace = {}
        replace["mon_joint_state_pveae"] = MjxAdapter._get_vec_joint_states_raw_pveae(sim_conf.monitored_qpadr,
                                                                  sim_conf.monitored_qvadr,
                                                                  sim_state.mjx_data)
        body_link_state = MjxAdapter._get_links_state_jax(sim_conf.monitored_lids, sim_state.mjx_data)
        # Compute local link linear acceleration (only for bodies)
        body_link_acceleration = MjxAdapter._get_links_acceleration_static(
            sim_conf.monitored_lids, sim_conf.body_rootid, sim_state.mjx_data)
        # If there are monitored sites, compute their state and concatenate
        if sim_conf.monitored_sids.shape[0] > 0:
            site_link_state = MjxAdapter._get_sites_state_jax(sim_conf.monitored_sids, sim_conf.site_bodyid, sim_state.mjx_data)
            replace["mon_link_state"] = jnp.concatenate([body_link_state, site_link_state], axis=1)
            # Acceleration is unavailable for sites — fill with zeros here, replaced with NaN on the PyTorch side
            vec_size = body_link_state.shape[0]
            site_acceleration_zeros = jnp.zeros((vec_size, sim_conf.monitored_sids.shape[0], 3))
            replace["mon_link_acceleration"] = jnp.concatenate([body_link_acceleration, site_acceleration_zeros], axis=1)
        else:
            replace["mon_link_state"] = body_link_state
            replace["mon_link_acceleration"] = body_link_acceleration
        # Collision mask for monitored pairs — backend-specific path. self._mjx_impl is static
        # under jit, so the Python branch is resolved at trace time and only one path is traced.
        if self._mjx_impl == "warp":
            if sim_conf.pair_sensor_adrs.shape[0] > 0:
                replace["mon_collision_mask"] = sim_state.mjx_data.sensordata[:, sim_conf.pair_sensor_adrs] > 0
        else:
            if len(sim_conf.monitored_collision_pairs) > 0:
                replace["mon_collision_mask"] = MjxAdapter._check_collision_pairs_static(
                    sim_conf.monitored_collision_pairs, sim_conf.geom_bodyid, sim_state.mjx_data)
        # Store precomputed states
        sim_state = sim_state.replace_d(replace)
        return sim_state

    @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"])
    def _update_step_stats(sim_state : SimState, sim_conf : SimConf) -> SimState:        
        jstate_pvaee = sim_state.mon_joint_state_pveae[:,:,[0,1,3,2,4]]
        lstate_v = sim_state.mon_link_state[:,:,7:13] # only linear velocity for stats
        sim_state = MjxAdapter._update_step_stats_arrs( jstate_pvaee, 
                                                        lstate_v,
                                                        sim_state)
        return sim_state

    @staticmethod
    @jax.jit
    def _check_collision_pairs_static(queried_body_pairs : jnp.ndarray, geom_bodyid : jnp.ndarray, mjx_data) -> jnp.ndarray:
        """Static version of collision pair checking for use in _update_step_stats.
        
        Parameters
        ----------
        queried_body_pairs : jnp.ndarray
            Shape (num_pairs, 2) with body id pairs to check
        geom_bodyid : jnp.ndarray
            Maps geom ids to body ids
        mjx_data : mjx.Data
            Current simulation data
            
        Returns
        -------
        jnp.ndarray
            Boolean array of shape (vec_size, num_pairs)
        """
        # Get active contacts
        active_contacts = mjx_data.contact.dist < mjx_data.contact.includemargin  # (vec_size, ncon)
        geom_pairs = mjx_data.contact.geom  # (vec_size, ncon, 2)
        geom_pairs = jnp.where(jnp.expand_dims(active_contacts, -1), geom_pairs, -1)
        colliding_body_pairs = geom_bodyid[geom_pairs]  # (vec_size, ncon, 2)
        
        # Check if any contact matches queried pairs (in either direction)
        # colliding_body_pairs: (vec_size, ncon, 2)
        # queried_body_pairs: (num_pairs, 2)
        colliding_expanded = jnp.expand_dims(colliding_body_pairs, 2)  # (vec_size, ncon, 1, 2)
        a_to_b = jnp.any(jnp.all(colliding_expanded == queried_body_pairs, axis=-1), axis=1)  # (vec_size, num_pairs)
        b_to_a = jnp.any(jnp.all(colliding_expanded == queried_body_pairs[:, [1, 0]], axis=-1), axis=1)
        return jnp.logical_or(a_to_b, b_to_a)

    @staticmethod
    @jax.jit
    def _get_links_acceleration_static(body_ids : jnp.ndarray, body_rootid : jnp.ndarray, mjx_data) -> jnp.ndarray:
        """Static version of local link linear acceleration computation for use in _update_step_stats."""
        @jax.vmap
        @jax.vmap
        def _transform_acceleration(com_linacc, com_angacc, com_linvel, com_angvel, com_offset_xyz, body_rotmat):
            local_angvel = body_rotmat.T @ com_angvel
            local_linvel = body_rotmat.T @ (com_linvel - jnp.cross(com_offset_xyz, com_angvel))
            acc = body_rotmat.T @ (com_linacc - jnp.cross(com_offset_xyz, com_angacc))
            correction = jnp.cross(local_angvel, local_linvel)
            return acc + correction
        
        com_linacc = mjx_data.cacc[:,body_ids,3:6]
        com_angacc = mjx_data.cacc[:,body_ids,:3]
        body_rotmat = mjx_data.xmat[:,body_ids]
        body_pos_xyz = mjx_data.xpos[:,body_ids]
        root_body_ids = body_rootid[body_ids]
        body_com_pos_xyz = mjx_data.subtree_com[:, root_body_ids]
        com_linvel = mjx_data.cvel[:,body_ids,3:6]
        com_angvel = mjx_data.cvel[:,body_ids,0:3]
        com_offset_xyz = body_pos_xyz - body_com_pos_xyz
        
        return _transform_acceleration(com_linacc, com_angacc, com_linvel, com_angvel, com_offset_xyz, body_rotmat)

    def _rebuild_step_stats_arrs(self):
        total_mon_links = self._sim_conf.monitored_lids.shape[0] + self._sim_conf.monitored_sids.shape[0]
        self._sim_state.mon_joint_stats_arr_pvaeep = jnp.zeros(shape=(self._static_sim_conf.vec_size, 6, self._sim_conf.monitored_jids.shape[0],6),
                                                        dtype=self._jax_float_dtype,
                                                        device=self._jax_device)
        self._sim_state.mon_links_stats_arr_v = jnp.zeros(shape=(self._static_sim_conf.vec_size, 6, total_mon_links, 6),
                                                        dtype=self._jax_float_dtype,
                                                        device=self._jax_device)
        # Initialize precomputed state arrays
        self._sim_state.mon_joint_state_pveae = jnp.zeros(shape=(self._static_sim_conf.vec_size, self._sim_conf.monitored_jids.shape[0], 5),
                                                        dtype=self._jax_float_dtype,
                                                        device=self._jax_device)
        self._sim_state.mon_link_state = jnp.zeros(shape=(self._static_sim_conf.vec_size, total_mon_links, 13),
                                                        dtype=self._jax_float_dtype,
                                                        device=self._jax_device)
        self._sim_state.mon_link_acceleration = jnp.zeros(shape=(self._static_sim_conf.vec_size, total_mon_links, 3),
                                                        dtype=self._jax_float_dtype,
                                                        device=self._jax_device)
        self._sim_state.mon_collision_mask = jnp.zeros(shape=(self._static_sim_conf.vec_size, self._sim_conf.monitored_collision_pairs.shape[0]),
                                                        dtype=jnp.bool_,
                                                        device=self._jax_device)

    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _reset_monitored_data_and_stats(self, sim_state : SimState, sim_conf : SimConf) -> SimState:
        # ggLog.info(f"resetting stats")
        sim_state = MjxAdapter._clear_step_stats(sim_state)
        sim_state = self._update_monitored_data_cache(sim_state, sim_conf)
        sim_state = MjxAdapter._update_step_stats(sim_state, sim_conf) # populate with current state, so that there are safe-ish values here
        return sim_state

    def get_joints_state_step_stats(self) -> th.Tensor:
        return jax2th(self._sim_state.mon_joint_stats_arr_pvaeep, self._out_th_device)[:,:4,:,:4]

    def get_joints_state_step_stats_extended(self) -> th.Tensor:
        return jax2th(self._sim_state.mon_joint_stats_arr_pvaeep, self._out_th_device)[:,:4]
    
    def get_links_state_step_stats(self) -> th.Tensor:
        stats = jax2th(self._sim_state.mon_links_stats_arr_v[:,:4], self._out_th_device)
        return stats[:, :, self._mon_link_jax_to_user, :]

    @staticmethod
    @partial(jax.jit)
    def _get_links_state_jax(body_ids : jnp.ndarray, mjx_data) -> jnp.ndarray:
        return jnp.concatenate([mjx_data.xpos[:,body_ids], # frame position
                                mjx_data.xquat[:,body_ids][:,:,[1,2,3,0]], # frame orientation
                                mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]]], axis = -1) # com linear and angular velocity

    @staticmethod
    @partial(jax.jit)
    def _get_sites_state_jax(site_ids : jnp.ndarray, site_bodyid : jnp.ndarray, mjx_data) -> jnp.ndarray:
        """Compute state for sites treated as links.
        Position from site_xpos, orientation from site_xmat->quat,
        velocity derived from parent body velocity via rigid body kinematics.
        """
        site_pos = mjx_data.site_xpos[:, site_ids]  # (V, S, 3)
        site_quat_xyzw = jax_mat_to_quat_xyzw(mjx_data.site_xmat[:, site_ids])  # (V, S, 4)
        # Derive velocity from parent body
        parent_body_ids = site_bodyid[site_ids]
        parent_cvel = mjx_data.cvel[:, parent_body_ids]  # (V, S, 6) = [angvel, linvel]
        parent_angvel = parent_cvel[:, :, 0:3]  # (V, S, 3)
        parent_linvel = parent_cvel[:, :, 3:6]  # (V, S, 3)
        parent_pos = mjx_data.xpos[:, parent_body_ids]  # (V, S, 3)
        r = site_pos - parent_pos  # offset from parent body origin to site
        site_linvel = parent_linvel + jnp.cross(parent_angvel, r)  # (V, S, 3)
        site_angvel = parent_angvel  # same angular velocity as parent (rigidly attached)
        return jnp.concatenate([site_pos, site_quat_xyzw, site_linvel, site_angvel], axis=-1)  # (V, S, 13)
    
    @staticmethod
    @jax.jit
    def _get_links_com_state_jax(body_ids : jnp.ndarray, mjx_data) -> jnp.ndarray:
        """ The COM orientation is aligned along the principal axes of inertia, 
            as stated in https://mujoco.readthedocs.io/en/stable/XMLreference.html?#body-inertial
            I believe this means the x ends up on the highes inertia axis and so on
            So for example on the main body of a usual quadruped the x would point down, the y sideways and the z front/back
        """
        return jnp.concatenate([mjx_data.xipos[:,body_ids], # com position
                                jax_mat_to_quat_xyzw(mjx_data.ximat[:,body_ids]), # com orientation, see above
                                mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]]], axis = -1) #com linear and angular velocity
    
    @override
    def get_links_ids(self, link_names : Sequence[tuple[str,str]]):
        # Return the index among all links
        return jnp.array([self._lname2lid[ln] for ln in link_names], device=self._jax_device, dtype=jnp.uint16)
    
    @override
    def get_joints_ids(self, joint_names : Sequence[tuple[str,str]]):
        return jnp.array([self._jname2jid[jn] for jn in joint_names], device=self._jax_device)

    @override
    def get_monitored_links_ids(self, link_names : Sequence[tuple[str,str]]):
        # Return the index among the monitored links
        return th.as_tensor([self._monitored_lid_to_idx[self._lname2lid[ln]] for ln in link_names], dtype=th.long).to(self._out_th_device, non_blocking=self._out_cuda)

    @override
    def get_monitored_links_ids_names(self, link_ids : th.Tensor):
        # link_ids are indices into monitored links, return the corresponding names
        return [self._monitored_links[idx] for idx in link_ids.tolist()]

    @override
    def get_monitored_joints_ids(self, joint_names : Sequence[tuple[str,str]]):
        # Return the index among the monitored joints
        return th.as_tensor([self._monitored_jid_to_idx[self._jname2jid[jn]] for jn in joint_names], dtype=th.long).to(self._out_th_device, non_blocking=self._out_cuda)

    @override
    def get_monitored_joints_ids_names(self, joint_ids : th.Tensor):
        # joint_ids are indices into monitored joints, return the corresponding names
        return [self._monitored_joints[idx] for idx in joint_ids.tolist()]

    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | th.Tensor | None = None, use_com_pose : bool = False) -> th.Tensor:
        if use_com_pose:
            raise ValueError(f"getLinksState: use_com_pose=True is not supported.")
        
        if requestedLinks is not None and len(requestedLinks) == 0:
            return th.empty(size=(self._vec_size, 0, 13), dtype=th.float32, device=self._out_th_device)
        
        self._forward_if_needed()
        # Convert to torch first (zero-copy via DLPack), then reorder in torch (lower dispatch overhead than JAX)
        full_state = jax2th(self._sim_state.mon_link_state, th_device=self._out_th_device)
        record_time("getLinksState: got full state")
        # JAX stores [bodies|sites]; permute back to user-specified order
        full_state = full_state[:, self._mon_link_jax_to_user, :]
        if requestedLinks is None:
            return full_state
        # Duck type: if element is tuple it's a name, otherwise assume lid
        # KeyError will propagate if link doesn't exist or isn't monitored
        if isinstance(requestedLinks, th.Tensor):
            reorder_indices = requestedLinks
        else:
            reorder_indices = self.get_monitored_links_ids(requestedLinks)
        ret = full_state[:, reorder_indices, :]
        record_time("getLinksState: reordered")

        return ret


    # @partial(jax.jit, static_argnums=(0,))
    # def _get_local_links_linear_acceleration_jax(self, body_ids : jnp.ndarray, mjx_data, mjx_model) -> jnp.ndarray:
    #     #Inspired by mujoco/mjx/_src/sensor.py:513
    #     @jax.vmap
    #     @jax.vmap
    #     def _transform_acceleration(com_linacc, com_angacc, com_linvel, com_angvel, com_offset_xyz, body_rotmat):
    #         local_angvel = body_rotmat.T @ com_angvel
    #         local_linvel = body_rotmat.T @ (com_linvel - jnp.cross(com_offset_xyz, com_angvel))
    #         acc = body_rotmat.T @ (com_linacc - jnp.cross(com_offset_xyz, com_angacc))
    #         correction = jnp.cross(local_angvel, local_linvel)
    #         return acc + correction
    #     com_linacc = mjx_data.cacc[:,body_ids,3:6] # com linear acceleration
    #     com_angacc = mjx_data.cacc[:,body_ids,:3] # com angular acceleration
    #     body_rotmat = mjx_data.xmat[:,body_ids]
    #     body_pos_xyz = mjx_data.xpos[:,body_ids] # body position
    #     root_body_ids = self._body_rootid[body_ids] # root body ids for each body
    #     body_com_pos_xyz = mjx_data.subtree_com[:, root_body_ids]
    #     com_linvel = mjx_data.cvel[:,body_ids,3:6]
    #     com_angvel = mjx_data.cvel[:,body_ids,0:3]
    #     com_offset_xyz = body_pos_xyz - body_com_pos_xyz

    #     frame_acc = _transform_acceleration(com_linacc, com_angacc, com_linvel, com_angvel, com_offset_xyz, body_rotmat)
    #     return frame_acc
    
    @override
    def get_local_link_linear_acceleration(self, requestedLinks : Sequence[tuple[str,str]] | th.Tensor | None = None) -> th.Tensor:
        record_region_start("MjxAdapter.get_local_link_linear_acceleration")
        if requestedLinks is not None and len(requestedLinks) == 0:
            return th.empty(size=(self._vec_size, 0, 3), dtype=th.float32, device=self._out_th_device)
        
        self._forward_if_needed()
        # Convert to torch first (zero-copy via DLPack), then reorder in torch (lower dispatch overhead than JAX)
        full_acc = jax2th(self._sim_state.mon_link_acceleration, th_device=self._out_th_device)
        # Replace site entries with NaN (sites have no acceleration data, stored as zeros on JAX side to avoid jax_debug_nans)
        if self._mon_site_count > 0:
            full_acc = full_acc.clone()
            full_acc[:, self._mon_body_count:, :] = float('nan')
        # JAX stores [bodies|sites]; permute back to user-specified order
        full_acc = full_acc[:, self._mon_link_jax_to_user, :]
        if requestedLinks is None:
            record_region_end("MjxAdapter.get_local_link_linear_acceleration")
            return full_acc
        # KeyError will propagate if link doesn't exist or isn't monitored
        if isinstance(requestedLinks, th.Tensor):
            reorder_indices = requestedLinks
        else:
            reorder_indices = self.get_monitored_links_ids(requestedLinks)
        record_region_end("MjxAdapter.get_local_link_linear_acceleration")
        return full_acc[:, reorder_indices, :]

    @override
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        # ggLog.info(f"MjxAdapter resetting")
        super().resetWorld()

        self._sim_state = self._sim_state.replace_d({   "mjx_data": copy.deepcopy(self._original_mjx_data),
                                                        "mjx_model": copy.deepcopy(self._original_mjx_model)})
        self._mj_data = copy.deepcopy(self._original_mj_data)
        self._mj_model = copy.deepcopy(self._original_mj_model)
        self._sim_state = self._reset_monitored_data_and_stats(self._sim_state, self._sim_conf)
        # self._recompute_mjxmodel_inaxes(self._sim_state.mjx_model)
        


    @override
    def getEnvTimeFromStartup(self) -> float:
        """Get the current time within the simulation."""
        return self._simTime

    @partial(jax.jit, static_argnames=["self"], donate_argnames=("sim_state",))
    def _set_joint_state_data(self, sim_state : SimState, sim_conf : SimConf, vec_mask_jnp : jnp.ndarray, qpadr_qvadr : jnp.ndarray, js_pve : jnp.ndarray):
        qpadr = qpadr_qvadr[0]
        qvadr = qpadr_qvadr[1]
        new_qpos = sim_state.mjx_data.qpos.at[:, qpadr].set(
            jnp.where(vec_mask_jnp[:, None], js_pve[:, :, 0], sim_state.mjx_data.qpos[:, qpadr])
        )
        new_qvel = sim_state.mjx_data.qvel.at[:, qvadr].set(
            jnp.where(vec_mask_jnp[:, None], js_pve[:, :, 1], sim_state.mjx_data.qvel[:, qvadr])
        )
        new_qfrc_applied = sim_state.mjx_data.qfrc_applied.at[:, qvadr].set(
            jnp.where(vec_mask_jnp[:, None], js_pve[:, :, 2], sim_state.mjx_data.qfrc_applied[:, qvadr])
        )
        
        mjx_data = sim_state.mjx_data.replace(qpos=new_qpos, qvel=new_qvel, qfrc_applied=new_qfrc_applied)
        sim_state = sim_state.replace_v("mjx_data", mjx_data)
        # sim_state = MjxAdapter._reset_monitored_data_and_stats(sim_state, self._sim_conf)
        return sim_state

    # @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"], static_argnames=["self","vec_size","run_forward"])
    def _set_joints_and_links_state_data(   self,
                                            sim_state : SimState,
                                            sim_conf : SimConf,
                                            vec_size : int,
                                            run_forward : bool,
                                            vec_mask_jnp : jnp.ndarray,
                                            joint_qpadr_qvadr : jnp.ndarray,
                                            joint_states_pve : jnp.ndarray,
                                            mjmodel_lids : jnp.ndarray,
                                            mjmodel_pose_xyz_xyzw : jnp.ndarray,
                                            mjdata_qpadrs_qvadrs : jnp.ndarray,
                                            mjdata_poses_xyzxyzw_vel_xyzxyz : jnp.ndarray):
        sim_state = self._set_joint_state_data(sim_state,
                                               sim_conf,
                                               vec_mask_jnp,
                                               joint_qpadr_qvadr,
                                               joint_states_pve)
        sim_state = self._set_link_poses(sim_state,
                                              vec_size,
                                              mjmodel_lids,
                                              mjmodel_pose_xyz_xyzw,
                                              mjdata_qpadrs_qvadrs,
                                              mjdata_poses_xyzxyzw_vel_xyzxyz,
                                              vec_mask_jnp)
        if run_forward:
            sim_state = self._forward_all(sim_state, sim_conf)
        return sim_state

    def _resolve_joint_qpadr_qvadr(self, joint_names : Sequence[tuple[str,str]]) -> jnp.ndarray:
        """ Resolve joint names to their (2, njoints) qpos/dof address array on the jax device.

        The result depends only on the names and the (frozen) model, so it is cached per
        name-tuple: the dict lookups, the joint-type validation, and the host->device
        transfer run once per distinct name set instead of on every command build. The
        returned array must be treated as read-only (it is shared across calls).
        """
        key = tuple(map(tuple, joint_names))
        cached = self._joint_addr_cache.get(key)
        if cached is not None:
            return cached
        if len(key) == 0:
            qpadr_qvadr = jnp.array(np.empty((2, 0), dtype=np.int32), device=self._jax_device)
        else:
            jids = np.array([self._jname2jid[jn] for jn in key])
            joint_types = self._mj_model.jnt_type[jids]
            if not np.all(np.logical_or(joint_types == mjutils._mjtJoint.mjJNT_HINGE,
                                        joint_types == mjutils._mjtJoint.mjJNT_SLIDE)):
                raise RuntimeError(f"Cannot control set state for multi-dimensional joint, types = {list(zip(key, joint_types))}")
            qpadr_qvadr = jnp.array(np.stack([self._mj_model.jnt_qposadr[jids],
                                              self._mj_model.jnt_dofadr[jids]]),
                                    device=self._jax_device)
        self._joint_addr_cache[key] = qpadr_qvadr
        return qpadr_qvadr

    def _prepare_joint_set_data(self,
                                joint_names : Sequence[tuple[str,str]] | None = None,
                                joint_states_pve : th.Tensor | None = None,
                                ):
        if joint_names is None:
            if joint_states_pve is not None:
                raise ValueError("joint_states_pve was provided without joint_names")
            qpadr_qvadr = jnp.array(np.empty((2, 0), dtype=np.int32), device=self._jax_device)
            joint_states_pve_jnp = jnp.array(np.empty((self._vec_size, 0, 3), dtype=np.float32), device=self._jax_device)
            return qpadr_qvadr, joint_states_pve_jnp
        if joint_states_pve is None:
            raise ValueError("joint_names was provided without joint_states_pve")

        if self._check_sizes and joint_states_pve.size() != (self._vec_size, len(joint_names), 3):
            raise RuntimeError(f"joint_states_pve should have size {(self._vec_size, len(joint_names), 3)}, but it's {joint_states_pve.size()}")

        qpadr_qvadr = self._resolve_joint_qpadr_qvadr(joint_names)
        joint_states_pve_jnp = th2jax(joint_states_pve, jax_device=self._jax_device)
        return qpadr_qvadr, joint_states_pve_jnp

    def _resolve_link_set_addrs(self, link_names : Sequence[tuple[str,str]]):
        """ Resolve link names to the constant address/index arrays needed to set link states.

        Returns ``(mjmodel_lids, idx_without_parents, idx_connected_to_world_th,
        mjdata_qpadrs_qvadrs)``: the parentless-body ids and free-joint qpos/dof addresses
        (jax, on device) plus the torch index arrays that split the per-call state tensor into
        the two groups. These depend only on the names and the (frozen) model, so they are
        cached per name-tuple: the validation and host->device transfers run once per distinct
        name set. The returned arrays must be treated as read-only (they are shared across calls).
        """
        key = tuple(map(tuple, link_names))
        cached = self._link_addr_cache.get(key)
        if cached is not None:
            return cached

        link_ids = np.array([self._lname2lid[ln] for ln in key])
        site_mask = link_ids >= self._nbody
        if np.any(site_mask):
            site_names = np.array(key)[site_mask]
            raise RuntimeError(f"Cannot set state for sites (they are kinematic, attached to a body): {site_names.tolist()}")
        root_body_ids = self._mj_model.body_rootid[link_ids]
        body_jnt_nums = self._mj_model.body_jntnum[link_ids]
        body_parent_ids = self._mj_model.body_parentid[link_ids]

        are_all_lids_root_bodies = np.all(root_body_ids == link_ids)
        if not are_all_lids_root_bodies:
            nonroot_lids = np.array(key)[root_body_ids != link_ids]
            raise RuntimeError(f"All links in setLinksStateDirect must be root bodies, but links {nonroot_lids} are not.")
        are_links_world = link_ids == 0
        if np.any(are_links_world):
            world_lids = np.array(key)[are_links_world]
            raise RuntimeError(f"Cannot set state for world link, but links {world_lids} are among the requested ones.")

        links_without_parents_mask = np.logical_and(body_jnt_nums == 0, body_parent_ids == 0)
        idx_without_parents = th.as_tensor(np.nonzero(links_without_parents_mask)[0]).to(self._out_th_device, non_blocking=self._out_cuda)
        lids_without_parents = link_ids[links_without_parents_mask]
        mjmodel_lids = jnp.array(lids_without_parents, device=self._jax_device)

        links_conected_to_world_mask = np.logical_and(body_jnt_nums == 1, body_parent_ids == 0)
        idx_connected_to_world = np.nonzero(links_conected_to_world_mask)[0]
        idx_connected_to_world_th = th.as_tensor(idx_connected_to_world).to(self._out_th_device, non_blocking=self._out_cuda)
        lids_connected_to_world = link_ids[links_conected_to_world_mask]

        link_joint_ids = self._mj_model.body_jntadr[lids_connected_to_world]
        link_joint_types = self._mj_model.jnt_type[link_joint_ids]
        all_free_joints_mask = link_joint_types == mjutils._mjtJoint.mjJNT_FREE
        if not np.all(all_free_joints_mask):
            non_free_joints_lids = lids_connected_to_world[~all_free_joints_mask]
            raise RuntimeError(f"Cannot set state for links connected to world with non-free joint, but links {non_free_joints_lids} are among the requested ones.")

        mjdata_qpadrs_qvadrs = jnp.array(np.stack([self._mj_model.jnt_qposadr[link_joint_ids],
                                                   self._mj_model.jnt_dofadr[link_joint_ids]], axis=0),
                                         device=self._jax_device)

        uncategorized_mask = ~(links_without_parents_mask | links_conected_to_world_mask)
        if np.any(uncategorized_mask):
            uncategorized_names = np.array(key)[uncategorized_mask]
            raise RuntimeError(f"Links {uncategorized_names.tolist()} are neither parentless bodies nor free-joint bodies connected to world, cannot set their state.")

        resolved = (mjmodel_lids, idx_without_parents, idx_connected_to_world_th, mjdata_qpadrs_qvadrs)
        self._link_addr_cache[key] = resolved
        return resolved

    def _prepare_links_set_data(self, link_names: Sequence[tuple[str, str]] | None = None,
                                link_states_pose_vel: th.Tensor | None = None):
        if link_names is None:
            if link_states_pose_vel is not None:
                raise ValueError("link_states_pose_vel was provided without link_names")
            mjmodel_lids = jnp.array(np.empty((0,), dtype=np.int32), device=self._jax_device)
            mjmodel_pose_xyz_xyzw = jnp.array(np.empty((self._vec_size, 0, 13), dtype=np.float32), device=self._jax_device)
            mjdata_qpadrs_qvadrs = jnp.array(np.empty((2, 0), dtype=np.int32), device=self._jax_device)
            mjdata_poses_xyzxyzw_vel_xyzxyz = jnp.array(np.empty((self._vec_size, 0, 13), dtype=np.float32), device=self._jax_device)
            return mjmodel_lids, mjmodel_pose_xyz_xyzw, mjdata_qpadrs_qvadrs, mjdata_poses_xyzxyzw_vel_xyzxyz
        if link_states_pose_vel is None:
            raise ValueError("link_names was provided without link_states_pose_vel")

        (mjmodel_lids,
         idx_without_parents,
         idx_connected_to_world_th,
         mjdata_qpadrs_qvadrs) = self._resolve_link_set_addrs(link_names)

        link_states_pose_vel = link_states_pose_vel.to(self._out_th_device, non_blocking=self._out_cuda)
        mjmodel_pose_xyz_xyzw = th2jax(link_states_pose_vel[:, idx_without_parents], jax_device=self._jax_device)
        mjdata_poses_xyzxyzw_vel_xyzxyz = th2jax(link_states_pose_vel[:, idx_connected_to_world_th], jax_device=self._jax_device)
        return mjmodel_lids, mjmodel_pose_xyz_xyzw, mjdata_qpadrs_qvadrs, mjdata_poses_xyzxyzw_vel_xyzxyz

    @override
    def setJointsAndLinksStateDirect(self,
                                     joint_names : Sequence[tuple[str,str]] | None = None,
                                     joint_states_pve : th.Tensor | None = None,
                                     link_names : Sequence[tuple[str,str]] | None = None,
                                     link_states_pose_vel : th.Tensor | None = None,
                                     vec_mask : th.Tensor | None = None):
        record_region_start("MjxAdapter.setJointsAndLinksStateDirect")

        

        

        if joint_names is None and link_names is None:
            record_region_end("MjxAdapter.setJointsAndLinksStateDirect")
            return

        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs
        record_time("MjxAdapter.setJointsAndLinksStateDirect: got vec_mask_jnp")

        joint_qpadr_qvadr, joint_states_pve_jnp = self._prepare_joint_set_data(joint_names, joint_states_pve)
        record_time("MjxAdapter.setJointsAndLinksStateDirect: prepared joint data")

        (mjmodel_lids,
         mjmodel_pose_xyz_xyzw,
         mjdata_qpadrs_qvadrs,
         mjdata_poses_xyzxyzw_vel_xyzxyz) = self._prepare_links_set_data(link_names, link_states_pose_vel)
        record_time("MjxAdapter.setJointsAndLinksStateDirect: prepared link data")

        self._sim_state = self._set_joints_and_links_state_data(sim_state = self._sim_state,
                                                                sim_conf = self._sim_conf,
                                                                vec_size = self._vec_size,
                                                                run_forward = False,
                                                                vec_mask_jnp = vec_mask_jnp,
                                                                joint_qpadr_qvadr = joint_qpadr_qvadr,
                                                                joint_states_pve = joint_states_pve_jnp,
                                                                mjmodel_lids = mjmodel_lids,
                                                                mjmodel_pose_xyz_xyzw = mjmodel_pose_xyz_xyzw,
                                                                mjdata_qpadrs_qvadrs = mjdata_qpadrs_qvadrs,
                                                                mjdata_poses_xyzxyzw_vel_xyzxyz = mjdata_poses_xyzxyzw_vel_xyzxyz)
        record_time("MjxAdapter.setJointsAndLinksStateDirect: setted data")

        self._mark_forward_needed()
        record_region_end("MjxAdapter.setJointsAndLinksStateDirect")

    def _empty_jax_array(self, shape : tuple[int, ...], dtype) -> jnp.ndarray:
        return jnp.empty(shape, dtype=dtype, device=self._jax_device)

    def _to_jax_ids(self, ids : Sequence[int] | np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return jnp.array(ids, dtype=jnp.int32, device=self._jax_device)

    def _vec_mask_to_jax(self, vec_mask : th.Tensor | None) -> jnp.ndarray:
        if vec_mask is None:
            return self._all_vecs
        return th2jax(vec_mask, jax_device=self._jax_device)

    def run_command_sequence(self, command_sequence : Sequence[PublicCommand | None],
                                    forward_if_needed : bool = True):
        record_region_start("MjxAdapter.run_command_sequence")
        if len(command_sequence) == 0:
            record_region_end("MjxAdapter.run_command_sequence")
            return

        internal_commands = []
        has_effects = False
        forward_needed = self._forward_needed

        for command in command_sequence:
            if command is None:
                continue
            internal_command = command.build_internal_command(self)
            record_time(f"MjxAdapter.run_command_sequence: built internal command {type(internal_command).__name__}") 
            internal_commands.append(internal_command)
            has_effects = has_effects or internal_command.has_effect()
            if internal_command.runs_forward():
                forward_needed = False
            if internal_command.marks_forward_needed():
                forward_needed = True
        if forward_needed and forward_if_needed:
            internal_commands.append(ForwardCommand().build_internal_command(self))
            record_time(f"MjxAdapter.run_command_sequence: built internal command forward") 
            forward_needed = False

        record_time("MjxAdapter.run_command_sequence: built internal commands") 
        if not has_effects:
            record_region_end("MjxAdapter.run_command_sequence")
            return

        self._sim_state = self._run_command_sequence_jax(
            sim_state=self._sim_state,
            sim_conf=self._sim_conf,
            static_sim_conf=self._static_sim_conf,
            internal_commands=tuple(internal_commands)
        )
        self._forward_needed = forward_needed
        record_region_end("MjxAdapter.run_command_sequence")

    @partial(jax.jit, static_argnames=("self","static_sim_conf"), donate_argnames=("sim_state",))
    def _run_command_sequence_jax(self,
                                  sim_state : SimState,
                                  sim_conf : SimConf,
                                  static_sim_conf : StaticSimConf,
                                  internal_commands : tuple[_InternalCommand, ...]) -> SimState:
        for internal_command in internal_commands:
            sim_state = internal_command.run_jax(self,
                                                 sim_state=sim_state,
                                                 sim_conf=sim_conf,
                                                 static_sim_conf=static_sim_conf)
        return sim_state

    @override    
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor | None = None):
        # ggLog.info(f"setJointsStateDirect(\n{joint_names}, \n{joint_states_pve}, \n{vec_mask})")

        record_region_start("MjxAdapter.setJointsStateDirect")
        if self._check_sizes and joint_states_pve.size() != (self._vec_size,len(joint_names),3):
            raise RuntimeError(f"joint_states_pve should have size {(self._vec_size,len(joint_names),3)}, but it's {joint_states_pve.size()}")
        
        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs
        jids = np.array([self._jname2jid[jn] for jn in joint_names])
        js_pve = th2jax(joint_states_pve, jax_device=self._jax_device)
        record_time("MjxAdapter.setJointsStateDirect: got js_pve and vec_mask_jnp")


        jtypes = self._mj_model.jnt_type[jids]
        if not np.all(np.logical_or(jtypes == mjutils._mjtJoint.mjJNT_HINGE, jtypes == mjutils._mjtJoint.mjJNT_SLIDE)):
            raise RuntimeError(f"Cannot control set state for multi-dimensional joint, types = {list(zip(joint_names,jtypes))}")
        qpadr_np = self._mj_model.jnt_qposadr[jids]
        qvadr_np = self._mj_model.jnt_dofadr[jids]
        record_time("MjxAdapter.setJointsStateDirect: got adrs")

        
        record_time("MjxAdapter.setJointsStateDirect: def func")
        qpadr_qvadr = jnp.array(np.stack([qpadr_np, qvadr_np]), device=self._jax_device)
        record_time("MjxAdapter.setJointsStateDirect: converted np->jax")
        self._sim_state = self._set_joint_state_data(self._sim_state, self._sim_conf, vec_mask_jnp, qpadr_qvadr, js_pve)
        record_time("MjxAdapter.setJointsStateDirect: setted data")

        # self._sim_state = MjxAdapter._reset_step_stats(self._sim_state, self._sim_conf)
        self._mark_forward_needed()
        record_region_end("MjxAdapter.setJointsStateDirect")

        # self._update_gui(force=True)
        # ggLog.info(f"setted_jstate Simtime [{self._simTime:.9f}] step [{self._sim_step_count_since_build}] monitored jstate:\n{self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._sim_state.mjx_data)}")


    def _update_gui(self, force : bool = False):
        if self._show_gui and (time.monotonic() - self._last_gui_update_wtime > 1/self._gui_freq or force):
            self._forward_if_needed()
            mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_util.tree_map(lambda l: l[self._gui_env_index], self._sim_state.mjx_data))
            self._last_gui_update_wtime = time.monotonic()
            self._viewer.sync()

    @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"], static_argnames=["vec_size"])
    def _set_link_poses(sim_state : SimState,
                        vec_size : int,
                        mjmodel_lids : jnp.ndarray,
                        mjmodel_pose_xyz_xyzw : jnp.ndarray,
                        mjdata_qpadrs_qvadrs : jnp.ndarray,
                        mjdata_poses_xyzxyzw_vel_xyzxyz : jnp.ndarray,
                        vec_mask_jnp : jnp.ndarray):
        """ Set the state of links by changing both mjmodel (for bodies with no parent joint) and mjdata (for bodies with a parent joint), using the same vec_mask_jnp to decide which envs to update in either case.
            Referenced MjData joints are assumed to be free joints.

        Parameters
        ----------
        sim_state : SimState
            The current simulation state to update
        mjmodel_lids : jnp.ndarray
            The link ids corresponding to the bodies to update, used for updating the mjmodel
        mjmodel_body_pos : jnp.ndarray
            The new body positions to set in the mjmodel, for bodies with no parent joint
        mjmodel_body_quat_xyzw : jnp.ndarray
            The new body quaternions to set in the mjmodel, for bodies with no parent joint
        mjdata_qpadrs : jnp.ndarray
            The joint position ids to set in the mjdata, for bodies with a parent joint
        mjdata_qvadrs : jnp.ndarray
            The joint velocities ids to set in the mjdata, for bodies with a parent joint
        mjdata_poses_xyz_xyzw : jnp.ndarray
            The new joint poses to set in the mjdata, for bodies with a parent joint, in xyz+xyzw format (will be converted to qpos format within the function)
        mjdata_vels : jnp.ndarray
            The new joint velocities to set in the mjdata, for bodies with a parent joint, in xyz+rpy format (will be converted to qvel format within the function)
        vec_mask_jnp : jnp.ndarray
            Mask of shape (vec_size,) indicating which environments to update

        """
        new_model_body_pos = sim_state.mjx_model.body_pos.at[:,mjmodel_lids].set(
            jnp.where(vec_mask_jnp[:, None, None], mjmodel_pose_xyz_xyzw[:,:,:3], sim_state.mjx_model.body_pos[:, mjmodel_lids])
        )
        new_model_body_quat = sim_state.mjx_model.body_quat.at[:,mjmodel_lids].set(
            jnp.where(vec_mask_jnp[:, None, None], mjmodel_pose_xyz_xyzw[:,:,[6,3,4,5]], sim_state.mjx_model.body_quat[:, mjmodel_lids])
        )
        
        qpadrs = mjdata_qpadrs_qvadrs[0]
        qvadrs = mjdata_qpadrs_qvadrs[1]

        njoints = qpadrs.shape[0]
        all_qpadrs = (qpadrs[:,None] + jnp.arange(7)).flatten() # free joints have 7 dof, flatten for all dof ids
        mjdata_poses_xyz_wxyz = mjdata_poses_xyzxyzw_vel_xyzxyz[:,:,[0,1,2,6,3,4,5]] # convert to wxyz for free joints
        all_poses = mjdata_poses_xyz_wxyz.reshape(vec_size, njoints*7) # flatten within each env
        new_qpos = sim_state.mjx_data.qpos.at[:, all_qpadrs].set(
            jnp.where(vec_mask_jnp[:, None], all_poses, sim_state.mjx_data.qpos[:, all_qpadrs])
        )
        
        all_qvadrs = (qvadrs[:,None] + jnp.arange(6)).flatten() # free joints have 6 dof, flatten for all dof ids
        mjdata_vels_xyzxyz = mjdata_poses_xyzxyzw_vel_xyzxyz[:,:,7:]
        all_vels = mjdata_vels_xyzxyz.reshape(vec_size, njoints*6) # flatten within each env
        new_qvel = sim_state.mjx_data.qvel.at[:, all_qvadrs].set(
            jnp.where(vec_mask_jnp[:, None], all_vels, sim_state.mjx_data.qvel[:, all_qvadrs])
        )

        mjx_data = sim_state.mjx_data.replace(qpos=new_qpos, qvel=new_qvel)
        mjx_model = sim_state.mjx_model.replace(body_pos=new_model_body_pos, body_quat=new_model_body_quat)
        sim_state = sim_state.replace_d({"mjx_data": mjx_data, "mjx_model": mjx_model})
        return sim_state
        

    @override
    def setLinksStateDirect(self, link_names : list[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor | None = None):

        record_region_start("mjxAdapter.setLinksStateDirect")
        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs

        lids = np.array([self._lname2lid[ln] for ln in link_names])
        site_mask = lids >= self._nbody
        if np.any(site_mask):
            site_names = np.array(link_names)[site_mask]
            raise RuntimeError(f"Cannot set state for sites (they are kinematic, attached to a body): {site_names.tolist()}")
        root_body_ids = self._mj_model.body_rootid[lids]
        body_jnt_nums = self._mj_model.body_jntnum[lids]
        body_parent_ids = self._mj_model.body_parentid[lids]

        are_all_lids_root_bodies = np.all(root_body_ids == lids)
        if not are_all_lids_root_bodies:
            nonroot_lids = np.array(link_names)[root_body_ids != lids]
            raise RuntimeError(f"All links in setLinksStateDirect must be root bodies, but links {nonroot_lids} are not.")
        are_links_world = lids == 0
        if np.any(are_links_world):
            world_lids = np.array(link_names)[are_links_world]
            raise RuntimeError(f"Cannot set state for world link, but links {world_lids} are among the requested ones.")
        
        # - bodies with no parents must be moved changing the mjmodel, 
        #   these can be set directly by knowing the lid
        # - bodies attached with one joint to the world must be moved changing mjdata,
        #   these can be set by knowing the lid and the parent joint id
        
        link_states_pose_vel = link_states_pose_vel.to(self._out_th_device, non_blocking=self._out_cuda)

        links_without_parents_mask = np.logical_and(body_jnt_nums == 0, body_parent_ids == 0)
        idx_without_parents = th.as_tensor(np.nonzero(links_without_parents_mask)[0]).to(self._out_th_device, non_blocking=self._out_cuda)
        lids_without_parents = lids[links_without_parents_mask]
        lids_without_parents_jax = jnp.array(lids_without_parents, device=self._jax_device)
        new_mjmodel_poses_xyz_xyzw = th2jax(link_states_pose_vel[:,idx_without_parents], jax_device=self._jax_device)

        links_conected_to_world_mask = np.logical_and(body_jnt_nums == 1, body_parent_ids == 0)
        idx_connected_to_world = np.nonzero(links_conected_to_world_mask)[0]
        idx_connected_to_world_th = th.as_tensor(idx_connected_to_world).to(self._out_th_device, non_blocking=self._out_cuda)
        lids_connected_to_world = lids[links_conected_to_world_mask]

        jids = self._mj_model.body_jntadr[lids_connected_to_world]
        jtypes = self._mj_model.jnt_type[jids]
        all_free_joints_mask = jtypes == mjutils._mjtJoint.mjJNT_FREE
        if not np.all(all_free_joints_mask):
            non_free_joints_lids = lids_connected_to_world[~all_free_joints_mask]
            raise RuntimeError(f"Cannot set state for links connected to world with non-free joint, but links {non_free_joints_lids} are among the requested ones.")
        qpadrs_qvadrs = np.stack([self._mj_model.jnt_qposadr[jids], self._mj_model.jnt_dofadr[jids]], axis = 0)
        qpadrs_qvadrs = jnp.array(qpadrs_qvadrs, device=self._jax_device)
        new_mjdata_poses_xyzxyzw_vel_xyzxyz = th2jax(link_states_pose_vel[:,idx_connected_to_world_th], jax_device=self._jax_device)

        uncategorized_mask = ~(links_without_parents_mask | links_conected_to_world_mask)
        if np.any(uncategorized_mask):
            uncategorized_names = np.array(link_names)[uncategorized_mask]
            raise RuntimeError(f"Links {uncategorized_names.tolist()} are neither parentless bodies nor free-joint bodies connected to world, cannot set their state.")
        record_time("prepared data")
        self._sim_state = self._set_link_poses( self._sim_state,
                                                self._static_sim_conf.vec_size,
                                                mjmodel_lids = lids_without_parents_jax,
                                                mjmodel_pose_xyz_xyzw = new_mjmodel_poses_xyz_xyzw,
                                                mjdata_qpadrs_qvadrs = qpadrs_qvadrs,
                                                mjdata_poses_xyzxyzw_vel_xyzxyz = new_mjdata_poses_xyzxyzw_vel_xyzxyz,
                                                vec_mask_jnp = vec_mask_jnp)
        record_region_end("mjxAdapter.setLinksStateDirect")


        # model_body_pos = self._sim_state.mjx_model.body_pos
        # model_body_quat = self._sim_state.mjx_model.body_quat
        # data_joint_pos = self._sim_state.mjx_data.qpos
        # data_joint_vel = self._sim_state.mjx_data.qvel
        # link_states_pose_vel_jnp = th2jax(link_states_pose_vel, jax_device=self._jax_device)
        # for i, link_name in enumerate(link_names):
        #     # ggLog.info(f"setting link state for {link_name}")
        #     lid = lids[i]            
        #     # Contrary to what you might expect mujoco associates each body to multiple possible parent joints
        #     # so:
        #     # - mj_model.body_jntnum[link_id] is the number of parent joints of a body
        #     # - mj_model.body_jntadr[link_id] is the id of the first of these parent joints
        #     # - mj_model.jnt_qposadr[joint_id] is the qpos addredd of a specific joint id
        #     parent_joints_num = self._mj_model.body_jntnum[lid]
        #     parent_body_id = self._mj_model.body_parentid[lid]
        #     if parent_joints_num == 0 and parent_body_id==0: # if it has no parent joints
        #         # ggLog.info(f"changing 'fixed joint'")
        #         #    Fixed joints cannot be set to different positions across the vectorized simulations.
        #         #    This because MJX does not vectorize the MjModel, all vec simulations use the same model,
        #         #     and fixed joints are represented as fixed transforms in the model.
        #         # if not jnp.all(jnp.array_equal(link_states_pose_vel_jnp[:,i], jnp.broadcast_to(link_states_pose_vel_jnp[0,i], shape=link_states_pose_vel_jnp[:,i].shape),equal_nan=True)):
        #         #     raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations.\n"
        #         #                        f"{link_states_pose_vel_jnp[0,i]}\n"
        #         #                        f"!=\n"
        #         #                        f"{link_states_pose_vel_jnp[:,i]}")
        #         # if jnp.any(vec_mask_jnp != vec_mask_jnp[0]):
        #         #     raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations, but vec_mask has different values.")
        #         model_body_pos = model_body_pos.at[:,lid].set(link_states_pose_vel_jnp[:,i,:3])
        #         model_body_quat = model_body_quat.at[:,lid].set(link_states_pose_vel_jnp[:,i,[6,3,4,5]])
        #         # print(f"self._sim_state.mjx_model.body_pos = {self._sim_state.mjx_model.body_pos}")
        #     elif parent_joints_num == 1 and parent_body_id==0:
        #         jid = self._mj_model.body_jntadr[lid]
        #         jtype = self._mj_model.jnt_type[jid]
        #         if jtype == mujoco_mjtJoint.mjJNT_FREE:
        #             # ggLog.info(f"writing at qpos[{self._mj_model.jnt_qposadr[jid]}:{self._mj_model.jnt_qposadr[jid]+7}]")
        #             qadr = self._mj_model.jnt_qposadr[jid]
        #             dadr = self._mj_model.jnt_dofadr[jid]
        #             data_joint_pos = set_rows_cols(data_joint_pos,
        #                                            (vec_mask_jnp, jnp.arange(qadr, qadr+7)),
        #                                            get_rows_cols(link_states_pose_vel_jnp, (vec_mask_jnp,i,[0,1,2,6,3,4,5])))
        #             data_joint_vel = set_rows_cols(data_joint_vel,
        #                                            (vec_mask_jnp, jnp.arange(dadr, dadr+6)),
        #                                            link_states_pose_vel_jnp[vec_mask_jnp,i,7:13]) #get_rows_cols(link_states_pose_vel_jnp, (vec_mask_jnp,i,jnp.arange(7,13))))
        #             # data_joint_pos = (data_joint_pos.at[vec_mask_jnp,qadr:qadr+7]
        #             #                                 .set(link_states_pose_vel_jnp[vec_mask_jnp,i,[0,1,2,6,3,4,5]]))
        #             # data_joint_vel = (data_joint_vel.at[vec_mask_jnp,dadr:dadr+6]
        #             #                                 .set(link_states_pose_vel_jnp[vec_mask_jnp,i,7:13]))
        #             # raise NotImplementedError()
        #         else:
        #             raise NotImplementedError(f"Cannot set link state for link {link_name} with parent joint of type {jtype} (see mjtJoint enum)")
        #     else:
        #         raise NotImplementedError(f"Cannot set link state for link {link_name} with {self._mj_model.body_jntnum} parent joints and parent body {parent_body_id}")
        # mjx_model = self._sim_state.mjx_model.replace(body_pos=model_body_pos, body_quat = model_body_quat)
        # mjx_data = self._sim_state.mjx_data.replace(qpos=data_joint_pos, qvel=data_joint_vel)
        # self._sim_state = self._sim_state.replace_d({"mjx_data"  : mjx_data,
        #                                              "mjx_model" : mjx_model})
        # self._recompute_mjxmodel_inaxes(self._sim_state.mjx_model)
        self._mark_forward_needed()
        # print(f"self._sim_state.mjx_model.body_pos = {self._sim_state.mjx_model.body_pos}")        
        # print(f"self._sim_state.mjx_data.qpos = {self._sim_state.mjx_data.qpos}")        
        # self._update_gui(True)
        # ggLog.info(f"setted_lstate Simtime [{self._simTime:.9f}] step [{self._sim_step_count_since_build}] monitored jstate:\n{self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._sim_state.mjx_data)}")

    def _mark_forward_needed(self):
        self._forward_needed = True

    @partial(jax.jit, static_argnames=("self",), donate_argnames=("sim_state",))
    def _forward_all(self, sim_state : SimState, sim_conf: SimConf) -> SimState:
        data = self._mjx_forward(sim_state.mjx_model, sim_state.mjx_data)
        sim_state = sim_state.replace_v("mjx_data", data)
        sim_state = self._update_monitored_data_cache(sim_state, sim_conf)
        return sim_state

    def _forward_if_needed(self):
        record_region_start("MjxAdapter._forward_if_needed")
        if self._forward_needed:
            # self._check_model_inaxes()
            self._sim_state = self._forward_all(self._sim_state, self._sim_conf)
            record_time("MjxAdapter._forward_if_needed: forward done")
            self._forward_needed = False
        record_region_end("MjxAdapter._forward_if_needed")

    def _check_model_inaxes(self):
        # from jax._src.tree_util import prefix_errors
        # all_errors = prefix_errors(self._mjx_model_in_axes, self._sim_state.mjx_model)
        # print(f"{get_caller_info()} prefix check= {all_errors[0]('in_axes')}")
        pass

    @override
    def setupLight(self):
        raise NotImplementedError()

    @override
    def spawn_models(self, models : Sequence[ModelSpawnDef]) -> list[str]:
        raise NotImplementedError("Cannot spawn after simulation setup")

    @override
    def delete_model(self, model_name : str):
        """Remove a model from all of the simulations
        Parameters
        ----------
        model_name : str
            Name of the model to be removed
        """
        raise NotImplementedError()
    
    @override
    def destroy_scenario(self, **kwargs):
        if self._viewer is not None:
            self._viewer.close()

    @override
    def setJointsEffortCommand(self, joint_names : Sequence[tuple[str,str]] | None, efforts : th.Tensor, vec_mask : th.Tensor | None = None) -> None:
        jids = jnp.array([self._jname2jid[jn] for jn in joint_names])
        qeff = th2jax(efforts, jax_device=self._jax_device)
        if vec_mask is not None:
            vec_mask_jax = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jax = None
        self._sim_state = self._set_effort_command(self._sim_state, jids,qeff, sims_mask=vec_mask_jax)
        # ggLog.info(f"self._sim_state.requested_qfrc_applied = {self._sim_state.requested_qfrc_applied}")


    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _set_effort_command(self,   sim_state : SimState,
                                    jids : jnp.ndarray,
                                    qefforts : jnp.ndarray,
                                    sims_mask : jnp.ndarray | None = None) -> SimState:
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.

        Parameters
        ----------
        joint_names : jnp.ndarray
            Array with the joint ids for each effort command, 1-dimensional
        efforts : th.Tensor
            Tensor of shape (vec_size, len(jids)) containing the effort for each joint in each environment.
        """
        qvadr = self._sim_conf.jnt_dofadr[jids]
        if sims_mask is None:
            sim_state = sim_state.replace_v( "requested_qfrc_applied",
                                  sim_state.requested_qfrc_applied.at[:,qvadr].set(qefforts[:,:]))
            # sim_state.requested_qfrc_applied = sim_state.requested_qfrc_applied.at[:,qvadr].set(qefforts[:,:])
        else:
            sim_state = sim_state.replace_v( "requested_qfrc_applied",
                                  set_rows_cols_masks(sim_state.requested_qfrc_applied, [sims_mask, qvadr], qefforts[:,:]))
            # sim_state.requested_qfrc_applied = set_rows_cols_masks(sim_state.requested_qfrc_applied, [sims_mask, qvadr], qefforts[:,:])
        return sim_state

    def reset_model_alterations(self, vec_mask : th.Tensor | None = None):
        # ggLog.info(f"setJointsStateDirect(\n{link_names}, \n{link_states_pose_vel}, \n{vec_mask})")
        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs
        # print(f"r0 self._sim_state.mjx_model.body_ipos.shape {self._sim_state.mjx_model.body_ipos.shape}")
        # print(f"r0 self._sim_state.mjx_model.body_mass.shape {self._sim_state.mjx_model.body_mass.shape}")
        self._sim_state.mjx_model = self._reset_model_alterations(vec_mask_jnp, self._sim_state, self._original_mjx_model)
        # print(f"r1 self._sim_state.mjx_model.body_mass.shape {self._sim_state.mjx_model.body_mass.shape}")
        # print(f"r1 self._sim_state.mjx_model.body_ipos.shape {self._sim_state.mjx_model.body_ipos.shape}")

    @staticmethod
    @partial(jax.jit, donate_argnames=["mjx_model"])
    def _reset_model_alterations(vec_mask : jnp.ndarray, mjx_model : mjx.Model, original_mjx_model : mjx.Model) -> mjx.Model:
        resetted_body_mass = jnp.where(jnp.expand_dims(vec_mask,1),
                                       original_mjx_model.body_mass, mjx_model.body_mass)
        resetted_geom_friction = jnp.where(jnp.expand_dims(vec_mask,(1,2)),
                                          original_mjx_model.geom_friction, mjx_model.geom_friction)
        resetted_body_ipos = jnp.where(jnp.broadcast_to(vec_mask, original_mjx_model.body_ipos.shape[::-1]).T,
                                       original_mjx_model.body_ipos, mjx_model.body_ipos)
        resetted_dof_armature = jnp.where(jnp.expand_dims(vec_mask, 1),
                                       original_mjx_model.dof_armature, mjx_model.dof_armature)
        resetted_dof_damping = jnp.where(jnp.expand_dims(vec_mask, 1),
                                       original_mjx_model.dof_damping, mjx_model.dof_damping)
        resetted_dof_frictionloss = jnp.where(jnp.expand_dims(vec_mask, 1),
                                       original_mjx_model.dof_frictionloss, mjx_model.dof_frictionloss)
        resetted_body_iquat = jnp.where(jnp.broadcast_to(vec_mask, original_mjx_model.body_iquat.shape[::-1]).T,
                                        original_mjx_model.body_iquat, mjx_model.body_iquat)
        resetted_model = mjx_model.replace( body_mass = resetted_body_mass,
                                            body_ipos = resetted_body_ipos,
                                            body_iquat = resetted_body_iquat,
                                            geom_friction = resetted_geom_friction,
                                            dof_armature = resetted_dof_armature,
                                            dof_damping = resetted_dof_damping,
                                            dof_frictionloss = resetted_dof_frictionloss)
        return resetted_model

    @staticmethod
    @partial(jax.jit, static_argnames=[ "vec_size",
                                        "apply_link_masses", "apply_link_frictions",
                                       "apply_dof_armature_ratios", "apply_dof_damping_ratios", "apply_dof_frictionloss_ratios",
                                       "apply_com_position_diffs", "apply_com_quatxyzw_diffs",
                                       "reset_first"],
                        donate_argnames=["mjx_model"])
    def _alter_model_jax(mjx_model : mjx.Model,
                         vec_size : int,
                         geom_bodyid : jnp.ndarray,
                         vec_mask : jnp.ndarray,
                         apply_link_masses : bool,
                         link_masses_body_ids : jnp.ndarray | None,
                         body_masses_ratio_change : jnp.ndarray | None,
                         apply_link_frictions : bool,
                         frictions_body_ids : jnp.ndarray | None,
                         body_frictions_ratio_change : jnp.ndarray | None,
                         apply_dof_armature_ratios : bool,
                         dof_armatures_dof_ids : jnp.ndarray | None,
                         dof_armatures_ratio_change : jnp.ndarray | None,
                         apply_dof_damping_ratios : bool,
                         dof_dampings_dof_ids : jnp.ndarray | None,
                         dof_dampings_ratio_change : jnp.ndarray | None,
                         apply_dof_frictionloss_ratios : bool,
                         dof_frictionloss_dof_ids : jnp.ndarray | None,
                         dof_frictionloss_ratio_change : jnp.ndarray | None,
                         apply_com_position_diffs : bool,
                         com_body_pos_ids : jnp.ndarray | None,
                         com_position_diff_xyz : jnp.ndarray | None,
                         apply_com_quatxyzw_diffs : bool,
                         com_body_quat_ids : jnp.ndarray | None,
                         com_quat_diff_xyzw : jnp.ndarray | None,
                         reset_first : bool = True, 
                         original_mjx_model : mjx.Model | None = None) -> mjx.Model:
        if reset_first:
            mjx_model = MjxAdapter._reset_model_alterations(vec_mask, mjx_model, original_mjx_model)
        replacements = {}
        if apply_link_masses:
            if body_masses_ratio_change is None or link_masses_body_ids is None:
                raise ValueError("To apply link mass changes, both body_masses_ratio_change and link_masses_body_ids must be provided")
            new_body_mass = mjx_model.body_mass.at[:, link_masses_body_ids].mul(body_masses_ratio_change)
            new_body_mass = jnp.where(vec_mask[:, None], new_body_mass, mjx_model.body_mass)
            replacements["body_mass"] = jnp.clip(new_body_mass, min=0.0001)

        if apply_link_frictions:
            frictions_body_ids_mask = jnp.zeros_like(mjx_model.geom_bodyid, shape=(mjx_model.nbody,), dtype=jnp.bool)
            frictions_body_ids_mask = frictions_body_ids_mask.at[frictions_body_ids].set(True)
            frictions_geoms_ids_mask = frictions_body_ids_mask[geom_bodyid] # which geoms to alter, geom_bodyid is shape (ngeom,), so frictions_geoms_ids_mask is shape (ngeom,)
            all_body_frictions_ratio_change = jnp.ones_like(mjx_model.body_mass, shape=(vec_size, mjx_model.nbody, 3), dtype=jnp.float32)
            all_body_frictions_ratio_change = all_body_frictions_ratio_change.at[:, frictions_body_ids].set(
                                                        body_frictions_ratio_change)
            all_geom_friction_ratios = all_body_frictions_ratio_change[:, geom_bodyid] # ratio for each geom
            new_allsim_allgeom_frictions = mjx_model.geom_friction * all_geom_friction_ratios
            new_allsim_allgeom_frictions = jnp.clip(new_allsim_allgeom_frictions, min=0.0)
            new_allsim_geom_frictions = jnp.where(
                jnp.expand_dims(frictions_geoms_ids_mask, 1).repeat(repeats=3, axis=1),
                new_allsim_allgeom_frictions,
                mjx_model.geom_friction,
            )
            replacements["geom_friction"] = jnp.where(
                vec_mask[:, None, None], new_allsim_geom_frictions, mjx_model.geom_friction
            )

        if apply_dof_armature_ratios:
            if dof_armatures_ratio_change is None or dof_armatures_dof_ids is None:
                raise ValueError("To apply joint armature changes, both dof_armatures_ratio_change and dof_armatures_dof_ids must be provided")
            new_armatures = mjx_model.dof_armature.at[:, dof_armatures_dof_ids].mul(dof_armatures_ratio_change)
            new_armatures = jnp.clip(new_armatures, min=0.0001)
            replacements["dof_armature"] = jnp.where(vec_mask[:, None], new_armatures, mjx_model.dof_armature)

        if apply_dof_damping_ratios:
            if dof_dampings_ratio_change is None or dof_dampings_dof_ids is None:
                raise ValueError("To apply joint damping changes, both dof_dampings_ratio_change and dof_dampings_dof_ids must be provided")
            new_dampings = mjx_model.dof_damping.at[:, dof_dampings_dof_ids].mul(dof_dampings_ratio_change)
            new_dampings = jnp.clip(new_dampings, min=0.0001)
            replacements["dof_damping"] = jnp.where(vec_mask[:, None], new_dampings, mjx_model.dof_damping)

        if apply_dof_frictionloss_ratios:
            if dof_frictionloss_ratio_change is None or dof_frictionloss_dof_ids is None:
                raise ValueError("To apply joint frictionloss changes, both dof_frictionloss_ratio_change and dof_frictionloss_dof_ids must be provided")
            new_frictionloss = mjx_model.dof_frictionloss.at[:, dof_frictionloss_dof_ids].mul(dof_frictionloss_ratio_change)
            new_frictionloss = jnp.where(vec_mask[:, None], new_frictionloss, mjx_model.dof_frictionloss)
            replacements["dof_frictionloss"] = jnp.clip(new_frictionloss, min=0.0001)

        if apply_com_position_diffs:
            new_body_ipos = mjx_model.body_ipos.at[:, com_body_pos_ids].add(com_position_diff_xyz)
            replacements["body_ipos"] = jnp.where(vec_mask[:, None, None], new_body_ipos, mjx_model.body_ipos)

        if apply_com_quatxyzw_diffs:
            if com_quat_diff_xyzw is None or com_body_quat_ids is None:
                raise ValueError("To apply COM orientation changes, both com_quat_diff_xyzw and com_body_quat_ids must be provided")
            quat_mul = mjx._src.math.quat_mul
            altered_quat = quat_mul(
                com_quat_diff_xyzw[:, :, [3, 0, 1, 2]],
                mjx_model.body_iquat[:, com_body_quat_ids],
            )
            new_body_iquat = mjx_model.body_iquat.at[:, com_body_quat_ids].set(altered_quat)
            replacements["body_iquat"] = jnp.where(vec_mask[:, None, None], new_body_iquat, mjx_model.body_iquat)

        return mjx_model.replace(**replacements)

    def alter_model(self, link_masses : tuple[jnp.ndarray, th.Tensor] | None = None,
                              link_frictions : tuple[jnp.ndarray, th.Tensor] | None = None,
                              joint_armature_ratios : tuple[jnp.ndarray, th.Tensor] | None = None,
                              joint_damping_ratios : tuple[jnp.ndarray, th.Tensor] | None = None,
                              joint_frictionloss_ratios : tuple[jnp.ndarray, th.Tensor] | None = None,
                              com_position_diffs : tuple[jnp.ndarray, th.Tensor] | None = None,
                              com_quatxyzw_diffs : tuple[jnp.ndarray, th.Tensor] | None = None,
                              vec_mask : th.Tensor | None = None,
                              reset_first : bool = True):
        """_summary_

        Parameters
        ----------
        link_masses : tuple[jnp.ndarray, th.Tensor]
            tuple containing  alist of link ids (from get_link_id) and corresponding
            body masses, body masses should be in a tensor of size (vec_size, len(link_ids))
            body mass will be set to old_mass*(1+ratio)
        link_frictions : tuple[jnp.ndarray, th.Tensor]
            tuple containing a list of link ids (from get_link_id) and corresponding
            body friction ratios, where the new friction will be computed as old_friction*(1+ratio).
        joint_armature_ratios : tuple[jnp.ndarray, th.Tensor]
            tuple containing a list of joint ids (from get_joint_id) and corresponding
            joint armature ratios, where the new armature will be computed as old_armature*(1+ratio).
        joint_damping_ratios : tuple[jnp.ndarray, th.Tensor]
            tuple containing a list of joint ids (from get_joint_id) and corresponding
            joint damping ratios, where the new damping will be computed as old_damping*(1+ratio).
        joint_frictionloss_ratios : tuple[jnp.ndarray, th.Tensor]
            tuple containing a list of joint ids (from get_joint_id) and corresponding
            joint frictionloss ratios, where the new frictionloss will be computed as old_frictionloss*(1+ratio).
        com_position_diffs : tuple[jnp.ndarray, th.Tensor]
            tuple containing a list of link ids (from get_link_id) and corresponding
            COM position differences, where the new COM position will be computed as old_COM_position + diff
        com_quatxyzw_diffs : tuple[jnp.ndarray, th.Tensor]
            tuple containing a list of link ids (from get_link_id) and corresponding
            COM orientation differences in quatxyzw format, where the new COM orientation will be computed as old_COM_orientation + diff 
        vec_mask : th.Tensor
            Mask of shape (vec_size,) indicating which environments to update, if None all environments will be updated
        reset_first : bool
            Whether to reset the previous alterations before applying the new ones. If False, new alterations will
            be applied on top of the current model parameters, which might lead to compounding effects if
            the same parameters are altered multiple times.
        """
        record_region_start("MjxAdapter.alter_model")
        # We need to be able to alter:
        #    body masses (body_mass)
        #    body frictions (geom_friction, in the xml there are sliding, torsional and rolling friction, where are they in mjmodel?)
        #    joint coulomb friction (dof_frictionloss, see https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
        #    joint rotational inertia (dof_armature, see https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
        #    some world parameters? e.g. gravity
        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs
        
        record_time("MjxAdapter.alter_model: reset done")
        body_masses_ratio_change=th2jax(link_masses[1], jax_device=self._jax_device) if link_masses is not None else None
        body_frictions_ratio_change=th2jax(link_frictions[1], jax_device=self._jax_device) if link_frictions is not None else None
        dof_armatures_ratio_change=th2jax(joint_armature_ratios[1], jax_device=self._jax_device) if joint_armature_ratios is not None else None
        dof_frictionloss_ratio_change=th2jax(joint_frictionloss_ratios[1], jax_device=self._jax_device) if joint_frictionloss_ratios is not None else None
        dof_dampings_ratio_change=th2jax(joint_damping_ratios[1], jax_device=self._jax_device) if joint_damping_ratios is not None else None
        com_position_diff_xyz=th2jax(com_position_diffs[1], jax_device=self._jax_device) if com_position_diffs is not None else None
        com_quat_diff_xyzw=th2jax(com_quatxyzw_diffs[1], jax_device=self._jax_device) if com_quatxyzw_diffs is not None else None

        record_time("MjxAdapter.alter_model: data prepared")
        self._sim_state.mjx_model = MjxAdapter._alter_model_jax(
            mjx_model=self._sim_state.mjx_model,
            vec_size=self._static_sim_conf.vec_size,
            geom_bodyid=self._sim_conf.geom_bodyid,
            vec_mask=vec_mask_jnp,
            apply_link_masses=link_masses is not None,
            link_masses_body_ids=link_masses[0] if link_masses is not None else None,
            body_masses_ratio_change=body_masses_ratio_change,
            apply_link_frictions=link_frictions is not None,
            frictions_body_ids=link_frictions[0] if link_frictions is not None else None,
            body_frictions_ratio_change=body_frictions_ratio_change,

            apply_dof_armature_ratios=joint_armature_ratios is not None,
            dof_armatures_dof_ids=self._sim_conf.jnt_dofadr[joint_armature_ratios[0]] if joint_armature_ratios is not None else None,
            dof_armatures_ratio_change=dof_armatures_ratio_change,
            
            apply_dof_frictionloss_ratios=joint_frictionloss_ratios is not None,
            dof_frictionloss_dof_ids=self._sim_conf.jnt_dofadr[joint_frictionloss_ratios[0]] if joint_frictionloss_ratios is not None else None,
            dof_frictionloss_ratio_change=dof_frictionloss_ratio_change,

            apply_dof_damping_ratios=joint_damping_ratios is not None,
            dof_damping_dof_ids=self._sim_conf.jnt_dofadr[joint_damping_ratios[0]] if joint_damping_ratios is not None else None,
            dof_dampings_ratio_change=dof_dampings_ratio_change,

            apply_com_position_diffs=com_position_diffs is not None,
            com_body_pos_ids=com_position_diffs[0] if com_position_diffs is not None else None,
            com_position_diff_xyz=com_position_diff_xyz,
            apply_com_quatxyzw_diffs=com_quatxyzw_diffs is not None,
            com_body_quat_ids=com_quatxyzw_diffs[0] if com_quatxyzw_diffs is not None else None,
            com_quat_diff_xyzw=com_quat_diff_xyzw,
            reset_first=reset_first,
            original_mjx_model=self._original_mjx_model
        )
        record_time("MjxAdapter.alter_model: model altered")
        # ggLog.info(f"altering model with {replacements}")
        # self._recompute_mjxmodel_inaxes() # Is it really necessary?
        record_region_end("MjxAdapter.alter_model")

    @override
    def check_colliding_links(self, requested_pairs: Sequence[tuple[tuple[str,str], tuple[str,str]]] | th.Tensor | None = None) -> th.Tensor:
        """Check if link pairs are colliding.
        
        Parameters
        ----------
        requested_pairs : Sequence[tuple[tuple[str,str], tuple[str,str]]] | th.Tensor | None
            If None, returns mask for all monitored pairs.
            If th.Tensor, indices into monitored pairs array.
            If Sequence, link name pairs to look up (must be monitored).
        
        Returns
        -------
        th.Tensor
            Boolean tensor of shape (vec_size, num_pairs) indicating collision status.
        """
        if requested_pairs is not None and len(requested_pairs) == 0:
            return th.empty(size=(self._vec_size, 0), dtype=th.bool, device=self._out_th_device)
        
        self._forward_if_needed()
        full_mask = jax2th(self._sim_state.mon_collision_mask, th_device=self._out_th_device)
        
        if requested_pairs is None:
            return full_mask
        
        if isinstance(requested_pairs, th.Tensor):
            reorder_indices = requested_pairs
        else:
            reorder_indices = self.get_collision_pair_ids(requested_pairs)
        return full_mask[:, reorder_indices]
    
    # def get_current_contacts_num(self) -> th.Tensor:
    #     """Gets the number of contacts in this instant.
    #
    #     Returns
    #     -------
    #     th.Tensor
    #         Tensor of size (self.vec_size,) with the number of contacts in each environment
    #     """
    #     self._forward_if_needed()
    #     return jax2th(self._sim_state.mjx_data.ncon, th_device=self._out_th_device)
    #
    # @partial(jax.jit, static_argnames=["self"])
    # def _get_current_colliding_link_id_pairs(self, sim_state : SimState) -> jnp.ndarray:
    #     # ggLog.info(f"self._sim_state.mjx_data.contact.geom.shape = {self._sim_state.mjx_data.contact.geom.shape}")
    #     # self._forward_if_needed()
    #     if self._mjx_impl == "warp":
    #         raise NotImplementedError("")

    #     active_contacts = sim_state.mjx_data.contact.dist < sim_state.mjx_data.contact.includemargin # size (vec_size, ncon)
    #     # print(f"sim_state.mjx_data.contact.dist = {sim_state.mjx_data.contact.dist}")
    #     # print(f"sim_state.mjx_data.contact.includemargin = {sim_state.mjx_data.contact.includemargin}")
    #     # print(f"active_contacts = {active_contacts}")
    #     geom_pairs = sim_state.mjx_data.contact.geom # size (vec_size, ncon, 2)
    #     geom_pairs = jnp.where(jnp.expand_dims(active_contacts,-1), geom_pairs, -1)
    #     # print(f"geom_pairs = {geom_pairs}")
    #     body_pairs = self._geom_bodyid_jax[geom_pairs]
    #     # print(f"body_pairs = {body_pairs}")
    #     # body_pairs = body_pairs.at[:,sim_state.mjx_data.ncon:].set(-1)
    #     # print(f"ncon = {sim_state.mjx_data.ncon}")
    #     # print(f"body_pairs = {body_pairs}")
    #     return body_pairs # size (vec_size, ncon, 2)

    #     print(f"ncon = {sim_state.mjx_data.ncon}")
    #     geom_pairs = sim_state.mjx_data.contact.geom
    #     print(f"geom_pairs = {geom_pairs}, size = {geom_pairs.shape}")
    #     body_pairs = self._geom_bodyid_jax[geom_pairs]
    #     body_pairs = body_pairs.at[:,self._sim_state.mjx_data.ncon:].set(-1)
    #     return body_pairs
    
    # @partial(jax.jit, static_argnames=["self"])
    # def _check_links_colliding(self, sim_state : SimState, queried_body_pairs : jnp.ndarray) -> jnp.ndarray:
    #     """Returns a boolean array of shape (vec_size, queried_body_pairs.shape[0]).

    #     Parameters
    #     ----------
    #     sim_state : SimState
    #         _description_
    #     queried_body_pairs : jnp.ndarray
    #         The body pairs to check for, array of shape (number_of_pairs,2)

    #     Returns
    #     -------
    #     jnp.ndarray
    #         Boolean array of shape (vec_size, queried_body_pairs.shape[0])
    #     """
    #     colliding_body_pairs = self._get_current_colliding_link_id_pairs(sim_state)
    #     # print(f"colliding_body_pairs = {colliding_body_pairs}")
    #     # colliding_body_pairs is of shape (vec_size, collision_num, 2)
    #     # body_pairs is of shape (num_queried_pairs, 2)
    #     a_to_b = jnp.any(jnp.all(jnp.expand_dims(colliding_body_pairs,2) == queried_body_pairs, axis = -1), axis=1)
    #     b_to_a = jnp.any(jnp.all(jnp.expand_dims(colliding_body_pairs,2) == queried_body_pairs[:,[1,0]], axis = -1), axis=1)
    #     return jnp.logical_or(a_to_b,b_to_a)


    # def _get_contacts_for_pairs(self, sim_state : SimState, queried_body_pairs : jnp.ndarray) -> jnp.ndarray:
    #     """Get a mask that indicates which of the current contacts are between the queried pairs.

    #     Parameters
    #     ----------
    #     sim_state : SimState
    #         _description_
    #     queried_body_pairs : jnp.ndarray
    #         size (number_of_pairs,2)

    #     Returns
    #     -------
    #     jnp.ndarray
    #         _description_
    #     """
    #     colliding_body_pairs = self._get_current_colliding_link_id_pairs(sim_state) # (vec_size, ncon, 2)
    #     colliding_body_pairs = jnp.expand_dims(colliding_body_pairs,2)
    #     # print(f"colliding_body_pairs = {colliding_body_pairs}")
    #     # colliding_body_pairs is of shape (vec_size, collision_num, 1, 2)
    #     # body_pairs is of shape                    (num_queried_pairs, 2)
    #     # comparison is (vec_size, collision_num, num_queried_pairs, 2)
    #     a_to_b = jnp.all(colliding_body_pairs == queried_body_pairs,          axis = -1)
    #     b_to_a = jnp.all(colliding_body_pairs == queried_body_pairs[:,[1,0]], axis = -1)
    #     is_queried = jnp.any(jnp.logical_or(a_to_b,b_to_a), axis=-1) # (vec_size, collision_num)
    #     return is_queried
    #
    # @partial(jax.jit, static_argnames=["self"])
    # def _get_total_contact_forces_for_pairs(self, sim_state : SimState, queried_body_pairs : jnp.ndarray):
    #     """Get the total contact forces for a set of body pairs.

    #     Parameters
    #     ----------
    #     sim_state : SimState
    #         _description_
    #     queried_body_pairs : jnp.ndarray
    #         size (number_of_pairs,2)

    #     Returns
    #     -------
    #     jnp.ndarray
    #         size (vec_size, number_of_pairs, 6) with the total force:torque for each environment and each body pair.
    #         If there are no contacts for an environment and body pair, the force:torque will be zero.
    #     """
    #     total_forces = jax.vmap(lambda x: self._get_total_contact_force_for_pair(sim_state, x), in_axes=[None, 0])(queried_body_pairs) # (number_of_pairs, vec_size, 6)
    #     return jnp.transpose(total_forces, (1,0,2)) # (vec_size, number_of_pairs, 6)

    # def _get_total_contact_force_for_pair(self, sim_state : SimState, queried_body_pair : jnp.ndarray):
    #     """Get the total contact force for a specific body pair.

    #     Parameters
    #     ----------
    #     sim_state : SimState
    #         _description_
    #     queried_body_pair : jnp.ndarray
    #         size (2,)

    #     Returns
    #     -------
    #     jnp.ndarray
    #         size (vec_size, 6) with the total force:torque for each environment.
    #         If there are no contacts for an environment, the force:torque will be zero.
    #     """
    #     contacts_mask = self._get_contacts_for_pairs(sim_state, jnp.expand_dims(queried_body_pair,0)) # (vec_size, ncon)
    #     # print(f"contacts_mask = {contacts_mask}")
    #     net_force = jax.vmap(lambda x, y: self._get_net_6d_force_for_contacts(x, y), in_axes=(0, 0))(sim_state.mjx_data, contacts_mask) # (vec_size, 6)
    #     return net_force

    # def _get_net_6d_force_for_contacts(self, single_mjx_data : mjx.Data, single_contacts_mask : jnp.ndarray):
    #     """Get the net 6D force for a set of contacts, on a single simulation.

    #     Parameters
    #     ----------
    #     single_mjx_data : mjx.Data
    #         The Mujoco data object containing the contact forces. (not vectorized)
    #     single_contacts_mask : jnp.ndarray
    #         A mask indicating which contacts to consider.

    #     Returns
    #     -------
    #     jnp.ndarray
    #         The net force:torque for the specified contacts, size (6,).
    #     """
    #     forces = self._get_force(single_mjx_data, single_mjx_data.contact.efc_address) # (ncon, 10)
    #     forces : jnp.ndarray = jnp.where(jnp.expand_dims(single_contacts_mask,-1), forces, 0.0)
    #     if self._mj_model.opt.cone == mjx.ConeType.ELLIPTIC:
    #         cond_dims = single_mjx_data.contact.dim # (ncon,)
    #         invalid_components = jnp.arange(0,forces.shape[-1]) >= cond_dims # (ncon,10)
    #         forces = forces.at[:,invalid_components].set(0.0)
    #         # net_3d_force = jnp.sum(forces[:,:3], axis=0) # (3,)
    #         # net_3d_torque = jnp.sum(forces[:,3:6], axis=0) # (3,)
    #         net_6d_force = jnp.sum(forces, axis=0) # (6,) # can I sum directly?
    #         return net_6d_force
    #     else:
    #         raise NotImplementedError("Pyramidal friction cone not implemented yet")
    #         forces = __contact_force_decode_pyramid(cond_dim, efc_force_pyramid=state_sim.efc_force[efc_addr:], friction_mu=state_sim.contact.friction[contact_id])

    # @staticmethod
    # @partial(jax.vmap, in_axes=(None, 1))
    # def _get_force(single_mjx_data : mjx.Data, efc_addr : jnp.ndarray):
    #     """Get the contact force for a specific contact.

    #     Parameters
    #     ----------
    #     mjx_data : mjx.Data
    #         The Mujoco data object containing the contact forces. (not vectorized)
    #     efc_addr : jnp.ndarray
    #         The address of the contact force in the Mujoco data.

    #     Returns
    #     -------
    #     jnp.ndarray
    #         The contact force for the specified contact.
    #     """
    #     return single_mjx_data.efc_force[efc_addr:efc_addr + 10] # return the maximum size of each


    # def _get_current_colliding_link_id_pairs_and_forces(self, sim_state : SimState) -> jnp.ndarray:
    #     # ggLog.info(f"self._sim_state.mjx_data.contact.geom.shape = {self._sim_state.mjx_data.contact.geom.shape}")
    #     # self._forward_if_needed()
    #     from mujoco.mjx._src.support import contact_force


    #     mjx_data = sim_state.mjx_data

    #     cond_dims = mjx_data.contact.dim
    #     efc_addrs = mjx_data.contact.efc_address
    #     if self._mj_model.opt.cone == mjx.ConeType.PYRAMIDAL:
    #         raise NotImplementedError("Pyramidal friction cone not implemented yet")
    #         forces = __contact_force_decode_pyramid(cond_dim, efc_force_pyramid=state_sim.efc_force[efc_addr:], friction_mu=state_sim.contact.friction[contact_id])
    #     else:
    #         forces = mjx_data.efc_force[contact_ids:contact_ids + cond_dims]

    #     if transform_in_world_frame:
    #         frame_rot_mat = mjx_data.contact.frame[contact_id]
    #         assert cond_dim == 3
    #         # rotate forces into global frame
    #         # see: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_vis_visualize.c#L230C19-L230C22
    #         forces = frame_rot_mat.T @ forces

    #     return body_pairs

    # def get_contact_force(self, sim_state : SimState,
    #                             contact_id: int,
    #                             transform_in_world_frame=False):
    #     """
    #     Get 6D force:torque for one contact, in contact frame.
    #     If con_dim of contact is just 3, this will return just the 3D forces.
    #     See: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_support.c#L1707

    #     :param sys:
    #     :param state_sim:
    #     :param contact_id: the index of the contact e.g. in state_sim.contact.geom[contact_id, :]
    #     :param transform_in_world_frame: if true, the force will be transformed into the world frame (just works with con_dim = 3).
    #     :return:
    #     """
    #     mjx_data = sim_state.mjx_data
    #     cond_dim = mjx_data.contact.dim[contact_id]
    #     efc_addr = mjx_data.contact.efc_address[contact_id]
    #     if self._mj_model.opt.cone == mjx.ConeType.PYRAMIDAL:
    #         raise NotImplementedError("Pyramidal friction cone not implemented yet")
    #         forces = __contact_force_decode_pyramid(cond_dim, efc_force_pyramid=state_sim.efc_force[efc_addr:], friction_mu=state_sim.contact.friction[contact_id])
    #     else:
    #         forces = mjx_data.efc_force[contact_id:contact_id + cond_dim]

    #     if transform_in_world_frame:
    #         frame_rot_mat = mjx_data.contact.frame[contact_id]
    #         assert cond_dim == 3
    #         # rotate forces into global frame
    #         # see: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_vis_visualize.c#L230C19-L230C22
    #         forces = frame_rot_mat.T @ forces
    #     return forces

    @override
    def set_link_impulses(self, link_ids : jnp.ndarray,
                                force_torque_xyzxyz : th.Tensor,
                                durations : th.Tensor, delays : th.Tensor,
                                vec_mask : th.Tensor) -> None:
        bodies_num = link_ids.shape[0]
        self._sim_state = self._set_link_impulse(self._sim_state,
                                                 link_ids,
                                                 force_torques=th2jax(force_torque_xyzxyz.view((self._vec_size,bodies_num,6)), jax_device=self._jax_device),
                                                 durations=th2jax(durations.view(self._vec_size,bodies_num), jax_device=self._jax_device),
                                                 delays=th2jax(delays.view(self._vec_size,bodies_num), jax_device=self._jax_device),
                                                 vec_mask=th2jax(vec_mask.view(self._vec_size,), jax_device=self._jax_device)
                                                 )


    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _set_link_impulse(self, sim_state : SimState,
                                body_ids : jnp.ndarray,
                                force_torques : jnp.ndarray,
                                durations : jnp.ndarray,
                                delays : jnp.ndarray,
                                vec_mask : jnp.ndarray | None):
        starts = sim_state.sim_time + delays # (vec_size,body_ids.shape[0])
        ends = starts + durations # (vec_size,body_ids.shape[0])
        startends = jnp.stack([starts,ends], axis = -1)  # (vec_size,body_ids.shape[0],2)
        if vec_mask is None:
            impulse_startends_stime = sim_state.impulse_startends_stime.at[:,body_ids].set(startends)
            impulses_xfrc = sim_state.impulses_xfrc.at[:,body_ids].set(force_torques)
        else:
            impulse_startends_stime = set_rows_cols_masks(sim_state.impulse_startends_stime, [vec_mask,body_ids], startends)
            impulses_xfrc = set_rows_cols_masks(sim_state.impulses_xfrc, [vec_mask,body_ids], force_torques)
        sim_state = sim_state.replace_d({"impulse_startends_stime" : impulse_startends_stime,
                                         "impulses_xfrc" : impulses_xfrc})
        return sim_state


    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_impulses(self, sim_state : SimState):
        started_impulses = sim_state.sim_time > sim_state.impulse_startends_stime[:,:,0] # vec_size*nbody
        ended_impulses = sim_state.sim_time > sim_state.impulse_startends_stime[:,:,1] # vec_size*nbody
        # expand to match sizesof xfrc_applied or impulse_startends_stime (i.e. vec_size*nbody*6 and vec_size*nbody*2)
        ended_impulses = jnp.expand_dims(ended_impulses, -1) # vec_size*nbody*1
        started_impulses = jnp.expand_dims(started_impulses, -1) # vec_size*nbody*1
        xfrc_applied = jnp.where(ended_impulses, 0, sim_state.mjx_data.xfrc_applied) # clear ended impulses (vec_size*nbody*6)
        impulse_startends_stime = jnp.where(ended_impulses, -1, sim_state.impulse_startends_stime) # clear ended impulses start/ends (vec_size*nbody*2)
        impulses_xfrc =           jnp.where(ended_impulses, 0,  sim_state.impulses_xfrc) # clear ended impulses force/torques (vec_size*nbody*6)
        active_impulses = jnp.logical_and(started_impulses, jnp.logical_not(ended_impulses))
        
        xfrc_applied = jnp.where(active_impulses, sim_state.impulses_xfrc, xfrc_applied)
        # jax.debug.print("xfrc_applied={xfrc_applied}, ended_impulses={ended_impulses}, started_impulses={started_impulses}",
        #                  xfrc_applied=xfrc_applied,
        #                  started_impulses=started_impulses,
        #                  ended_impulses=ended_impulses)       
        
        mjx_data = sim_state.mjx_data.replace(xfrc_applied = xfrc_applied)
        sim_state = sim_state.replace_d({"impulse_startends_stime" : impulse_startends_stime,
                                         "impulses_xfrc" : impulses_xfrc,
                                         "mjx_data" : mjx_data})
        return sim_state



    # @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    # def _apply_disturbances(self, sim_state : SimState):
    #     prng_key = sim_state.prng_key
    #     ending_disturbances = sim_state.sim_time > sim_state.disturbance_terminations_stime
    #     xfrc_applied = jnp.where(ending_disturbances, 0, sim_state.mjx_data.xfrc_applied)
    #     disturbance_terminations_stime = jnp.where(ending_disturbances, -1, sim_state.disturbance_terminations_stime)
        
    #     prng_key, subkey = jax.random.split(prng_key)
    #     starting_disturbances = jax.random.uniform(subkey, shape=ending_disturbances.shape) < self._xfrc_disturbance_probability
        
    #     prng_key, subkey = jax.random.split(prng_key)
    #     new_disturbance_durations = jax.random.normal(subkey, shape=disturbance_terminations_stime.shape) * self._disturbance_duration_std + self._disturbance_duration_mean
    #     new_xfrc_applied_disturbed = jax.random.normal(subkey, shape=xfrc_applied.shape) * self._xfrc_disturbance_std + self._xfrc_disturbance_mu
    #     xfrc_applied = jnp.where(starting_disturbances, new_xfrc_applied_disturbed, xfrc_applied)
    #     disturbance_terminations_stime = jnp.where(starting_disturbances,sim_state.stime + new_disturbance_durations, sim_state.disturbance_terminations_stime)
        
    #     mjx_data = sim_state.mjx_data.replace(xfrc_applied = xfrc_applied)
    #     sim_state = sim_state.replace_d({"prng_key":prng_key,
    #                                      "disturbance_terminations_stime":disturbance_terminations_stime,
    #                                      "mjx_data" : mjx_data})

    @partial(jax.jit, static_argnums=(0,))
    def _get_height_map(self, positions_vec_xy : jnp.ndarray, range_xyxy : jnp.ndarray, resolution_xy : tuple[int,int],
                                ground_linkgroups_ids : jnp.ndarray,
                                sim_state : SimState):
        vsize = positions_vec_xy.shape[0]
        jnp_resolution_xy = jnp.array(resolution_xy)
        width_height = range_xyxy[2:4] - range_xyxy[0:2]
        coords_grid = jnp.swapaxes(jnp.mgrid[:resolution_xy[0],:resolution_xy[1]]/jnp_resolution_xy*width_height-range_xyxy[0:2],0,2) + jnp_resolution_xy/2
        coords_grid = positions_vec_xy + coords_grid
        ray_height = 10.0
        ray_origins = jnp.concatenate([coords_grid,
                                       jnp.full_like(coords_grid[...,0], fill_value=ray_height)[..., None]], axis=-1) # add z coord
        ray_origins_flat = ray_origins.reshape(vsize, -1, 3)
        ray_dirs_flat = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32), ray_origins_flat.shape)
        dists_vec_xy = self._mjx_ray_vec(   sim_state.mjx_model, 
                                            sim_state.mjx_data, 
                                            ray_origins_flat,
                                            ray_dirs_flat,
                                            ground_linkgroups_ids # mask that is true at each group_id to be included
                                            ).reshape(vsize, resolution_xy[0], resolution_xy[1])
        return dists_vec_xy - ray_height

    def get_height_map(self, positions_vec_xy : th.Tensor, range_xyxy : th.Tensor, resolution_xy : tuple[int,int],
                                ground_linkgroups : list[tuple[tuple[str,str],...]]):
        for g in ground_linkgroups:
            if g not in self._linkgroup_to_id:
                raise RuntimeError(f"Group {g} was not already defined, you can add it explicitly in set_body_collisions")
        ground_linkgroups_ids = th.as_tensor([self._linkgroup_to_id[g] for g in ground_linkgroups])
        heights_vec_xy = self._get_height_map(positions_vec_xy=th2jax(positions_vec_xy, jax_device=self._jax_device),
                                range_xyxy=th2jax(range_xyxy, jax_device=self._jax_device),
                                ground_linkgroups_ids = th2jax(ground_linkgroups_ids, jax_device=self._jax_device),
                                resolution_xy=resolution_xy,
                                sim_state = self._sim_state)        
        return jax2th(heights_vec_xy, th_device=self._out_th_device)

    def _get_depth_cam_params(self, cam_id : int):
        cam_height, cam_width = self._camera_sizes_hw_by_id[cam_id]
        cam_pos_local = jnp.array(self._mj_model.cam_pos[cam_id], dtype=jnp.float32, device=self._jax_device)
        cam_rot_local = quat_wxyz_to_rotmat(jnp.array(self._mj_model.cam_quat[cam_id], dtype=jnp.float32, device=self._jax_device))
        fovy_rad = jnp.deg2rad(self._mj_model.cam_fovy[cam_id])
        return cam_height, cam_width, cam_pos_local, cam_rot_local, fovy_rad

    def _build_camera_rays(self, cam_width, cam_height, fovy_rad):
        xs = jnp.arange(cam_width, dtype=jnp.float32)
        ys = jnp.arange(cam_height, dtype=jnp.float32)
        grid_x, grid_y = jnp.meshgrid(xs, ys, indexing="xy")
        x_norm = (grid_x + 0.5 - cam_width / 2.0) / (cam_width / 2.0)
        y_norm = (grid_y + 0.5 - cam_height / 2.0) / (cam_height / 2.0)
        raydirs_camframe = jnp.stack([  x_norm * jnp.tan(fovy_rad / 2.0)* (cam_width / cam_height),
                                    -y_norm * jnp.tan(fovy_rad / 2.0),
                                    -jnp.ones_like(x_norm)], axis=-1)
        raydirs_camframe = raydirs_camframe / jnp.linalg.norm(raydirs_camframe, axis=-1, keepdims=True)
        raydirs_camframe = raydirs_camframe.reshape(cam_width*cam_height, 3)
        return raydirs_camframe
    
    def _precompute_depth_cam_params(self):
        self._depth_cam_params_by_id = {}
        for cam_id in self._camera_sizes_hw_by_id.keys():
            cam_height, cam_width, cam_pos_local, cam_rot_local, fovy_rad = self._get_depth_cam_params(cam_id)
            ggLog.info(f"MjxAdapter depth cam: cam_id {cam_id} cam_height {cam_height} cam_width {cam_width} cam_pos_local {cam_pos_local} cam_rot_local {cam_rot_local} fovy_rad {fovy_rad}")
            raydirs_camframe = self._build_camera_rays(cam_width, cam_height, fovy_rad)
            self._depth_cam_params_by_id[cam_id] = {
                "cam_height": cam_height,
                "cam_width": cam_width,
                "cam_pos_local": cam_pos_local,
                "cam_rot_local": cam_rot_local,
                "fovy_rad": fovy_rad,
                "raydirs_camframe": raydirs_camframe
            }

    @partial(jax.jit, static_argnums=(0,1))
    def _get_depth_image_jax(self, cam_id : int, sim_state : SimState, geom_group : jnp.ndarray | None = None) -> jnp.ndarray:
        body_id = self._mj_model.cam_bodyid[cam_id]
        depth_cam_params = self._depth_cam_params_by_id[cam_id]
        cam_height = depth_cam_params["cam_height"]
        cam_width = depth_cam_params["cam_width"]
        cam_pos_local = depth_cam_params["cam_pos_local"]
        cam_rot_local = depth_cam_params["cam_rot_local"]
        raydirs_camframe = depth_cam_params["raydirs_camframe"]

        cambody_rot = sim_state.mjx_data.xmat[:, body_id]
        cambody_pos = sim_state.mjx_data.xpos[:, body_id]

        nrays = raydirs_camframe.shape[0]
        nsims = cambody_pos.shape[0]

        cambody_rot_world = jnp.matmul(cambody_rot, cam_rot_local)
        cambody_pos_world = cambody_pos + jnp.matmul(cambody_rot, cam_pos_local)

        raydirs_world = jnp.matmul(cambody_rot_world, raydirs_camframe.T).transpose((0,2,1)) # (vec_size, nrays, 3)

        rayorigins_world = jnp.broadcast_to(cambody_pos_world[:, None, :],  (nsims, nrays, 3)) # (vec_size, nrays, 3)

        dists, geoms_ids = self._mjx_ray_vec(sim_state.mjx_model,
                                  sim_state.mjx_data,
                                  rayorigins_world,
                                  raydirs_world,
                                  geom_group)
        return dists.reshape(cambody_pos_world.shape[0], cam_height, cam_width)

    def _get_depth_image(self,
                         camera_name : str,
                         sim_state : SimState | None = None,
                         visible_linkgroups : list[tuple[tuple[str,str],...]] | None = None) -> th.Tensor:
        """
        Cast per-pixel rays from the specified camera and return depth maps for all vector environments.

        Parameters
        ----------
        camera_name : str
            Name of the camera as returned by get_detected_cameras().
        sim_state : SimState | None
            Simulation state to use; defaults to the adapter's internal state.
        visible_linkgroups : list[tuple[tuple[str,str],...]] | None
            List of linkgroups that should be visible to the camera. If None, all linkgroups are visible.

        Returns
        -------
        th.Tensor
            Depth image tensor with shape (vec_size, height, width) in meters.
        """
        if sim_state is None:
            sim_state = self._sim_state
        if sim_state is self._sim_state:
            self._forward_if_needed()
        if visible_linkgroups is not None:
            raise NotImplementedError("visible_linkgroups is not implemented yet")
        else:
            geom_group = None
        
        cam_id = self._cname2cid[camera_name]
        img = self._get_depth_image_jax(cam_id, sim_state, geom_group)
        # ggLog.info(f"Depth image from camera '{camera_name}' (id {cam_id}): shape {img.shape}, min {img.min()}, max {img.max()}")
        
        return jax2th(img, th_device=self._out_th_device)
