from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_FLAGS"]="--xla_gpu_triton_gemm_any=true"

from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import Pose, compile_xacro_string, pkgutil_get_path, exc_to_str, build_pose, to_string_tensor, get_caller_info
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

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_debug_nans", True) # May have a performance impact?
# jax.config.update("jax_debug_infs", True) # May have a performance impact?
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

from mujoco.mjx._src.forward import euler, forward

def th2jax(tensor : th.Tensor, jax_device : jax.Device):
    # apparently there are issues with non-contiguous tensors (https://github.com/jax-ml/jax/issues/7657)
    # and with CPU tensors (https://github.com/jax-ml/jax/issues/25066#issuecomment-2494697463)
    return jnp.from_dlpack(tensor.contiguous().cuda(non_blocking=True)).to_device(jax_device)
                                                    
def jax2th(array : jnp.ndarray, th_device : th.device):
    return thdlpack.from_dlpack(array.to_device(jax.devices("gpu")[0])).to(th_device) #.detach().clone()

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

def path2tstr(path):
    return tuple([n.name for n in path])

def add_compiler_options(urdf_def : str,
                         max_hull_vert : int = 32,
                         discardvisual : bool = False,
                         strippath : bool = False):
    
    mujoco_block = ('<mujoco>\n'+
                    f'    <compiler  discardvisual="{str(discardvisual).lower()}" strippath="{str(strippath).lower()}" maxhullvert="{max_hull_vert:d}"/>\n'
                    '</mujoco>')
    return urdf_def.replace("</robot>",mujoco_block+"\n</robot>")
# def tree_set(tree, leaf_name : str, new_value):
#     return jax.tree_util.tree_map_with_path(lambda path, leaf: leaf if path2tstr(path)!=(leaf_name,) else new_value, tree)

# def tree_replace(tree, leafs_path_value : Mapping[tuple[str,...],Any]):
#     return jax.tree_util.tree_map_with_path(lambda path, leaf: leafs_path_value.get(path2tstr(path),leaf), tree)

def mjx_integrate_and_forward(m: mjx.Model, d: mjx.Data) -> mjx.Data:
    """First integrate the physics, then compute forward kinematics/dynamics.
        This is a flipped-around version of mjx.step(), essentially doing mj_step2 and then mj_step1.
        By doing so, the simulation state is already updated after the step, however it is important
        to call forward once before calling this for the first time. I believe dm_control does
        something similar."""
    # see: https://github.com/google-deepmind/mujoco/issues/430#issuecomment-1208489785
    d = euler(m, d)
    d = forward(m, d)
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
    # index_arrs = [jnp.full(ia.shape[i],-1,device=ia.device).at([])[0]    for i,ia in enumerate(masks)]
    index_arrs = [mask if mask.dtype!=bool else jnp.where(mask, jnp.arange(mask.shape[0]), -1) for mask in masks]
    return array.at[jnp.ix_(*index_arrs)].set(vals)

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

model_element_separator = "#"

@jax.jit
def jax_mat_to_quat_xyzw(matrices):
    return jax.scipy.spatial.transform.Rotation.from_matrix(matrices).as_quat(scalar_first=False)

@jax.tree_util.register_dataclass
@dataclass
class SimState:
    mjx_data : mjx.Data
    requested_qfrc_applied : jnp.ndarray
    sim_time : jnp.ndarray
    stats_step_count : jnp.ndarray
    mon_joint_stats_arr_pvae : jnp.ndarray

    def replace_v(self, name : str, value : Any):
        return self.replace_d({name:value})

        return SimState(mjx_data=value if name=="mjx_data" else self.mjx_data,
                        requested_qfrc_applied=value if name=="requested_qfrc_applied" else self.requested_qfrc_applied,
                        sim_time=value if name=="sim_time" else self.sim_time)

    def replace_d(self, name_values : dict[str,Any]):
        # ggLog.info(f"rd0 type(self.mjx_data) = {type(self.mjx_data)}")
        # d = dataclasses.asdict(self) # Recurses into dataclesses and deepcopies
        d = {"mjx_data" : self.mjx_data,
             "requested_qfrc_applied" : self.requested_qfrc_applied,
             "sim_time" : self.sim_time}
        # ggLog.info(f"d0 = "+str({k:type(v) for k,v in d.items()}))
        d.update(name_values)
        # ggLog.info(f"d1 = "+str({k:type(v) for k,v in d.items()}))
        ret = SimState(**d)
        # ggLog.info(f"type(self.mjx_data) = {type(self.mjx_data)}")
        return ret
    

class MjxAdapter(BaseVecSimulationAdapter, BaseVecJointEffortAdapter):

    @dataclass
    class DebugInfo():
        wtime_running : float = 0.0
        wtime_simulating : float = 0.0
        wtime_controlling : float = 0.0
        rt_factor_vec : float = 0.0
        rt_factor_single : float = 0.0
        stime_ran : float = 0.0
        stime : float = 0.0
        fps_vec : float = 0.0
        fps_single : float = 0.0
        iterations : int = 0
        run_fps_vec : float = 0.0

    def __init__(self, vec_size : int,
                        enable_rendering : bool,
                        jax_device : jax.Device,
                        output_th_device : th.device,
                        sim_step_dt : float = 2/1024,
                        step_length_sec : float = 10/1024,
                        realtime_factor : float | None = None,
                        show_gui : bool = False,
                        gui_frequency : float = 15,
                        gui_env_index : int = 0,
                        add_ground : bool = True,
                        log_freq : int = -1,
                        opt_preset : Literal["fast","none"] = "fast",
                        log_folder : str = "./",
                        record_whole_joint_trajectories : bool = False,
                        log_freq_joints_trajectories : int = 1000,
                        safe_revolute_dof_armature = 0.01,
                        revolute_dof_armature_override = None,
                        opt_override : dict[str,Any] | None = None):
        super().__init__(vec_size=vec_size,
                         output_th_device=output_th_device)
        self._enable_rendering = enable_rendering
        self._jax_device = jax_device
        self._sim_step_dt = sim_step_dt
        self._step_length_sec = step_length_sec
        self._simTime = 0.0
        self._total_iterations = 0
        self._sim_step_count_since_build = 0
        self._sim_stepping_wtime_since_build = 0
        self._run_wtime_since_build = 0
        self._add_ground = add_ground
        self._log_freq = log_freq
        self._log_folder = log_folder
        self._last_log_iters = -log_freq 
        self._opt_preset = opt_preset if not None else "none"
        self._safe_revolute_dof_armature = safe_revolute_dof_armature
        self._revolute_dof_armature_override = revolute_dof_armature_override #0.5
        self._discardvisual = False
        self._opt_override = opt_override

        self._realtime_factor = realtime_factor
        self._wxyz2xyzw = jnp.array([1,2,3,0], device = jax_device)
        self._sim_state = SimState(mjx_data=jnp.empty((0,), device = jax_device),
                                   requested_qfrc_applied=jnp.empty((0,), device = jax_device),
                                   sim_time=jnp.empty((0,), device = jax_device),
                                   stats_step_count=jnp.zeros((1,), device = jax_device),
                                   mon_joint_stats_arr_pvae=jnp.empty((0,), device = jax_device))
        self._renderer : mujoco.Renderer | None = None
        self._check_sizes = True
        self._show_gui = show_gui
        self._gui_env_index = gui_env_index
        self._last_gui_update_wtime = 0.0
        self._gui_freq = gui_frequency
        self._prev_step_end_wtime = 0.0
        self._viewer = None
        self._all_vecs = jnp.ones((vec_size,), dtype=bool, device=self._jax_device)
        self._no_vecs = jnp.zeros((vec_size,), dtype=bool, device=self._jax_device)
        self._all_vecs_th = th.ones((vec_size,), dtype=th.bool, device=self._out_th_device)
        self._no_vecs_th = th.zeros((vec_size,), dtype=th.bool, device=self._out_th_device)
        self._all_vecs_thcpu = th.ones((vec_size,), dtype=th.bool, device="cpu")
        self._no_vecs_thcpu = th.zeros((vec_size,), dtype=th.bool, device="cpu")

        self._dbg_info = MjxAdapter.DebugInfo()

        self._record_joint_hist = record_whole_joint_trajectories
        self._joints_pveae_history = []
        self._log_freq_joints_trajcetories = log_freq_joints_trajectories

    @staticmethod
    def _mj_name_to_pair(mjname : str):
        if mjname == "world":
            return mjname,mjname
        sep = mjname.find(model_element_separator)
        if mjname.count(model_element_separator) != 1:
            raise RuntimeError(f"Invalid mjName: must contain one and only one '{model_element_separator}' substring, but it's {mjname}")
        return mjname[:sep],mjname[sep+len(model_element_separator):]

    def _recompute_mjxmodel_inaxes(self):
        out_axes = jax.tree_util.tree_map(lambda l:None, self._mjx_model)
        out_axes = out_axes.tree_replace({"body_mass":0,
                                          "geom_friction":0}) # model fields to be vmapped
        self._mjx_model_in_axes = out_axes

    def _rebuild_lower_funcs(self):
        self._mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(self._mjx_model_in_axes, 0)))
        self._mjx_integrate_and_forward = jax.jit(jax.vmap(mjx_integrate_and_forward, in_axes=(self._mjx_model_in_axes, 0))) #, donate_argnames=["d"]) donating args make it crash
        

    @override
    def build_scenario(self, models : list[ModelSpawnDef]):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
        ggLog.info(f"MjxAdapter building scenario")
        if self._add_ground:
            models.append(ModelSpawnDef( name="ground",
                                           definition_string="""<mujoco>
                                                                    <compiler angle="radian"/>
                                                                    <asset>
                                                                        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
                                                                        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
                                                                        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
                                                                    </asset>
                                                                        <worldbody>
                                                                        <body name="ground_link">
                                                                            <light pos="0 0 10" dir="-0.3 -0.3 -1" directional="true" 
                                                                                    ambient="0.2 0.2 0.2"
                                                                                    diffuse="0.7 0.7 0.7"
                                                                                    specular="0.5 0.5 0.5"
                                                                                    castshadow="true"/>
                                                                            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin="0.0" />
                                                                        </body>
                                                                    </worldbody>
                                                                </mujoco>""",
                                           format="mjcf",
                                           pose=None,
                                           kwargs={}))
        specs = []
        ggLog.info(f"Spawning models: {[model.name for model in models]}")
        for model in models:
            if model.format.strip().lower()[-6:] == ".xacro":
                def_string = compile_xacro_string( model_definition_string=model.definition_string,
                                                                model_kwargs=model.kwargs)
            elif model.format.strip().lower() in ("urdf","mjcf"):
                def_string = model.definition_string
            else:
                raise RuntimeError(f"Unsupported model format '{model.format}' for model '{model.name}'")
            if model.format.strip().lower() in ("urdf","urdf.xacro"):
                def_string = add_compiler_options(def_string, discardvisual=self._discardvisual)
            ggLog.info(f"Adding model '{model.name}' : \n{def_string}")
            mjSpec = mujoco.MjSpec.from_string(def_string)
            # mjSpec.compiler.discardvisual = False
            mjSpec.compiler.degree = False
            specs.append((model.name, mjSpec))
            if model.pose is not None:
                raise NotImplementedError(f"Error adding model '{model.name}' ModelSpawnDef.pose is not supported yet")
        big_speck = mujoco.MjSpec()
        
        frame = big_speck.worldbody.add_frame()
        big_speck.compiler.degree = False
        # big_speck.compiler.discardvisual = False
        for mname, spec in specs:
            if model_element_separator in mname:
                raise RuntimeError(f"Cannot have models with '#' in their name (this character is used internally). Found model named {mname}")
            # add all th bodies that are direct childern of worldbody
            body = spec.worldbody.first_body()
            # spec.compiler.discardvisual = False
            if spec.compiler.degree:
                raise NotImplementedError(f"model {mname} uses degrees instead of radians.")
            while body is not None:
                frame.attach_body(body, mname+model_element_separator, "")
                body = spec.worldbody.next_body(body)
        # big_speck.compiler.discardvisual = False
        big_speck.memory = 50*1024*1024 #allocate 50mb for arena (this becomes mjmodel.narena and mjdata.narena)
        self._mj_model = big_speck.compile()
        self._mj_model.opt.timestep = self._sim_step_dt
        if self._opt_preset == "fastest":
            # Copied from barkour example
            self._mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            self._mj_model.opt.iterations = 1 # constraint solver iterations
            self._mj_model.opt.ls_iterations = 5 # doc: "Ensures that at most iterations times ls_iterations linesearch iterations are performed during each constraint solve"
            self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        elif self._opt_preset == "faster":
            self._mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            self._mj_model.opt.iterations = 3
            self._mj_model.opt.ls_iterations = 3
            self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        elif self._opt_preset == "fast":
            self._mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            self._mj_model.opt.iterations = 10
            self._mj_model.opt.ls_iterations = 5
            # self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        else:
            raise RuntimeError(f"Unknown opt preset '{self._opt_preset}'")
        if self._opt_override is not None:
             for k,v in self._opt_override.items():
                setattr(self._mj_model.opt,k,v)
        # ggLog.info(f"big_speck.degree = {big_speck.compiler.degree}")
        ggLog.info(f"Spawned: \n{big_speck.to_xml()}")
        ggLog.info(f"mj_model.opt = {self._mj_model.opt}")


        for dof_id in range(self._mj_model.nv):
            joint_type = self._mj_model.jnt_type[self._mj_model.dof_jntid[dof_id]]
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                if self._mj_model.dof_armature[dof_id] == 0:
                    self._mj_model.dof_armature[dof_id] = self._safe_revolute_dof_armature
                if self._revolute_dof_armature_override is not None:
                    self._mj_model.dof_armature[dof_id] = self._revolute_dof_armature_override

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
        # self._mj_model = mujoco.MjModel.from_xml_string(urdf_string)

        self._jid2jname : dict[int, tuple[str,str]] = {jid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid))
                           for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname : dict[int, tuple[str,str]] = {lid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, lid))
                           for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}
        self._cid2cname : dict[int, str] = {jid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, jid))[1]
                           for jid in range(self._mj_model.ncam)}
        self._cname2cid = {cn:cid for cid,cn in self._cid2cname.items()}
        ggLog.info(f"self._lname2lid = {self._lname2lid}")


        # self.set_body_collisions([  (('kyon', 'pelvis'), False),
        #                             (('kyon', 'base_link'), False),
        #                             (('kyon', 'imu_link'), False),
        #                             (('kyon', 'hip_roll_1_link'), False),
        #                             (('kyon', 'hip_pitch_1_link'), False),
        #                             (('kyon', 'knee_pitch_1_link'), False),
        #                             (('kyon', 'contact_1'), False),
        #                             (('kyon', 'hip_roll_2_link'), False),
        #                             (('kyon', 'hip_pitch_2_link'), False),
        #                             (('kyon', 'knee_pitch_2_link'), False),
        #                             (('kyon', 'contact_2'), False),
        #                             (('kyon', 'hip_roll_3_link'), False),
        #                             (('kyon', 'hip_pitch_3_link'), False),
        #                             (('kyon', 'knee_pitch_3_link'), False),
        #                             (('kyon', 'contact_3'), False),
        #                             (('kyon', 'hip_roll_4_link'), False),
        #                             (('kyon', 'hip_pitch_4_link'), False),
        #                             (('kyon', 'knee_pitch_4_link'), False),
        #                             (('kyon', 'contact_4'), False)])
        # self._mj_model.body_conaffinity[:] = 0
        # self._mj_model.geom_conaffinity[:] = 0
        self._mj_data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, self._mj_data)

        self._mjx_model = mjx.put_model(self._mj_model, device = self._jax_device)
        # self._mjx_model.opt.timestep.at[:].set(self._sim_step_dt)
        self._recompute_mjxmodel_inaxes()
        self._mjx_model = jax.vmap(lambda: self._mjx_model, in_axes=None, axis_size=self._vec_size, out_axes=self._mjx_model_in_axes)()
        # self._mjx_model = jax.vmap(lambda: self._mjx_model, axis_size=self._vec_size, in_axes=None)()
        
        self._jnt_dofadr_jax = jnp.array(self._mjx_model.jnt_dofadr, device = self._jax_device) # for some reason it's a numpy array, so I cannot use it properli in jit

        mjx_data = mjx.put_data(self._mj_model, self._mj_data, device = self._jax_device)
        mjx_data = jax.vmap(lambda: mjx_data, axis_size=self._vec_size)()
        # mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(jnp.arange(self._vec_size), mjx_data)
        ggLog.info(f"mjx_data.qpos.shape = {mjx_data.qpos.shape}")


        self._original_mjx_data = copy.deepcopy(mjx_data)
        self._original_mjx_model = copy.deepcopy(self._mjx_model)
        self._original_mj_data = copy.deepcopy(self._mj_data)
        self._original_mj_model = copy.deepcopy(self._mj_model)

        # _ = self._mjx_step(self._mjx_model, copy.deepcopy(mjx_data)) # trigger jit compile
        self._rebuild_lower_funcs()
        # self._mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(self._mjx_model_in_axes, 0)))
        # self._mjx_integrate_and_forward = jax.jit(jax.vmap(mjx_integrate_and_forward, in_axes=(self._mjx_model_in_axes, 0))) #, donate_argnames=["d"]) donating args make it crash
        # ggLog.info(f"Compiling mjx.step....")
        # self._mjx_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0))) #, donate_argnames=["d"]) donating args make it crash
        
        requested_qfrc_applied = jnp.copy(mjx_data.qfrc_applied)
        sim_time = jnp.zeros((1,), jnp.float32, device=self._jax_device)
        self._sim_state = self._sim_state.replace_d({   "mjx_data":mjx_data,
                                                        "requested_qfrc_applied":requested_qfrc_applied,
                                                        "sim_time":sim_time})
        
        # self._check_model_inaxes()        
        # data = self._mjx_forward(self._mjx_model,self._sim_state.mjx_data)
        # ggLog.info(f"compiled")
        # self._check_model_inaxes()        
        
        self.set_monitored_joints([])
        self.set_monitored_links([])
        self._reset_joint_state_step_stats()


        if self._enable_rendering:
            self._renderer = mujoco.Renderer(self._mj_model)
            self._render_scene_option = mujoco.MjvOption()
            # # enable joint visualization option:
            # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        else:
            self._renderer = None

        if self._show_gui:
            self._viewer_mj_data : mujoco.MjData = mjx.get_data(self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._sim_state.mjx_data))
            mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._sim_state.mjx_data))
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._viewer_mj_data)

        self._camera_sizes :dict[str,tuple[int,int]] = {self._cid2cname[cid]:(self._mj_model.cam_resolution[cid][1],self._mj_model.cam_resolution[cid][0]) for cid in self._cid2cname}
        if self._enable_rendering:
            def make_renderer(h,w):
                ggLog.info(f"Making renderer for size {h}x{w}")
                return mujoco.Renderer(self._mj_model,height=h,width=w)
            self._renderers : dict[tuple[int,int],mujoco.Renderer]= {resolution:make_renderer(resolution[0],resolution[1])
                            for resolution in set(self._camera_sizes.values())}
            self._renderers_mj_datas : list[mujoco.MjData] = [copy.deepcopy(self._mj_data) for _ in range(self.vec_size())]
        else:
            self._renderers = {}
        

        self._lid2geoms : dict[int,jnp.ndarray] = {}
        all_links = list(self._lname2lid.keys())
        for lname in all_links:
            body_id = self._lname2lid[lname]
            self._lid2geoms[body_id] = self._mjx_model.body_geomadr[body_id:body_id+self._mjx_model.body_geomnum[body_id]]
        self._is_geom_visual = jnp.logical_and(self._mj_model.geom_contype==0, self._mj_model.geom_conaffinity==0)


        print("Joint limits:\n"+("\n".join([f" - {jn}: {r}" for jn,r in {jname:self._mj_model.jnt_range[jid] for jid,jname in self._jid2jname.items()}.items()])))
        print("Joint child bodies:\n"+("\n".join([f" - {jn}: {r}" for jn,r in {jname:self._mj_model.jnt_bodyid[jid] for jid,jname in self._jid2jname.items()}.items()])))
        print(f"dof armatures:{self._mj_model.dof_armature}")
        
        print("Bodies parentid:\n"+("\n".join([f" - body_parentid[{lid}({self._lid2lname[lid]})]= {self._mj_model.body_parentid[lid]}" for lid in self._lid2lname.keys()])))
        print("Bodies jnt_num:\n"+("\n".join([f" - body_jntnum[{lid}({self._lid2lname[lid]})]= {self._mj_model.body_jntnum[lid]}" for lid in self._lid2lname.keys()])))
        # print(f"got cam resolutions {self._camera_sizes}")
        # self._check_model_inaxes()        


    def startup(self):
        ggLog.info(f"Compiling mjx.forward....")
        data = self._mjx_forward(self._mjx_model, self._sim_state.mjx_data)
        self._sim_state = self._sim_state.replace_v("mjx_data", data) # compute initial mjData
        ggLog.info(f"Compiling mjx_integrate_and_forward....")
        _ = self._mjx_integrate_and_forward(self._mjx_model, copy.deepcopy(self._sim_state.mjx_data)) # trigger jit compile
        ggLog.info(f"Compiled.")

    @override
    def set_body_collisions(self, link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]]):
        input_collision_groups = [set(lg[1]) for lg in link_group_collisions]
        ggLog.info(f"input_collision_groups = {input_collision_groups}")

        best_collision_groups = []
        while len(input_collision_groups)>0:
            biggest_common_subgroup = set(input_collision_groups[0])
            for g in input_collision_groups:
                biggest_common_subgroup.intersection_update(g)
            best_collision_groups.append(biggest_common_subgroup)
            input_collision_groups = [(g.difference(biggest_common_subgroup)) for g in input_collision_groups]
            input_collision_groups = [g for g in input_collision_groups if len(g)>0]

        ggLog.info(f"best_collision_groups = {best_collision_groups}")
        link_to_groups = {} # Which groups each link is part of
        for i,g in enumerate(best_collision_groups):
            for l in g:
                if l not in link_to_groups:
                    link_to_groups[l] = []
                link_to_groups[l].append(i)
        ggLog.info(f"link_to_groups = {link_to_groups}")

        link_colliding_groups = {} # Which group each link collides with
        for link,colliding_links in link_group_collisions:
            colliding_links = set(colliding_links)
            for g in best_collision_groups:
                if g.issubset(colliding_links):
                    if l not in link_colliding_groups:
                        link_colliding_groups[link] = []
                    link_colliding_groups[link].append(i)
        ggLog.info(f"link_colliding_groups = {link_colliding_groups}")

        if len(best_collision_groups) > 32:
            raise RuntimeError(f"Detected more than 32 separate collision group. Cannot represent in Mujoco collision masks.")
        
        all_links = list(self._lname2lid.keys()) #list(link_to_groups.keys())+list(link_colliding_groups.keys())
        for l in all_links:
            if l not in link_to_groups:
                link_to_groups[l] = []
            if l not in link_colliding_groups:
                link_colliding_groups[l] = []
        link_contypes = {}
        link_conaffinity = {}
        for l in all_links:
            contype_mask = 0
            for gid in link_to_groups[l]:
                contype_mask |= 1<<gid
            link_contypes[l] = contype_mask
            conaffinity_mask = 0
            for gid in link_colliding_groups[l]:
                conaffinity_mask |= 1<<gid
            link_conaffinity[l] = conaffinity_mask

        

        # print(f"self._mj_model.geom_contype =     {self._mj_model.geom_contype}")
        # print(f"self._mj_model.geom_conaffinity = {self._mj_model.geom_conaffinity}")
        # print(f"self._mjx_model.geom_contype =     {self._mjx_model.geom_contype}")
        # print(f"self._mjx_model.geom_conaffinity = {self._mjx_model.geom_conaffinity}")

        body_contype = self._mjx_model.body_contype.copy()
        body_conaffinity = self._mjx_model.body_conaffinity.copy()
        geom_contype = self._mjx_model.geom_contype.copy()
        geom_conaffinity = self._mjx_model.geom_conaffinity.copy()
        for lname in all_links:
            body_id = self._lname2lid[lname]
            for geom_id in range(self._mj_model.body_geomadr[body_id],
                                 self._mj_model.body_geomadr[body_id]+self._mj_model.body_geomnum[body_id]):
                visual = geom_contype[geom_id]==0 and geom_conaffinity[geom_id]==0
        for lname in all_links:
            body_id = self._lname2lid[lname]
            aff = link_conaffinity[lname]
            typ = link_contypes[lname]
            body_conaffinity[body_id] = aff
            body_contype[body_id] = typ
            for geom_id in range(self._mj_model.body_geomadr[body_id],
                                 self._mj_model.body_geomadr[body_id]+self._mj_model.body_geomnum[body_id]):
                # this will pass also through the visual geoms!
                # geom_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                # geom_rgba = self._mj_model.geom_rgba[geom_id]
                # ggLog.info(f"{lname} [{body_id}] : {geom_id} [{geom_name}]: coll={typ:b}/{aff:b} rgba = {geom_rgba}, aff = {aff}, typ = {typ} ")
                # geom_contype[geom_id] = 1
                visual = geom_contype[geom_id]==0 and geom_conaffinity[geom_id]==0
                if not visual:
                    geom_conaffinity[geom_id] = aff
                    geom_contype[geom_id] = typ

        self._mj_model.geom_contype = geom_contype
        self._mj_model.geom_conaffinity = geom_conaffinity
        self._mj_model.body_contype = body_contype
        self._mj_model.body_conaffinity = body_conaffinity
        ggLog.info(f"Previous geom_contype =     {self._mjx_model.geom_contype}")
        ggLog.info(f"Previous geom_conaffinity = {self._mjx_model.geom_conaffinity}")
        self._mjx_model = self._mjx_model.replace(geom_contype = geom_contype,
                                                  geom_conaffinity = geom_conaffinity,
                                                  body_contype = body_contype,
                                                  body_conaffinity = body_conaffinity)
        self._original_mjx_model = self._original_mjx_model.replace(geom_contype = geom_contype,
                                                  geom_conaffinity = geom_conaffinity,
                                                  body_contype = body_contype,
                                                  body_conaffinity = body_conaffinity)
        # print(f"self._mj_model.geom_contype =     {self._mj_model.geom_contype}")
        # print(f"self._mj_model.geom_conaffinity = {self._mj_model.geom_conaffinity}")
        ggLog.info(f"New geom_contype =     {self._mjx_model.geom_contype}")
        ggLog.info(f"New geom_conaffinity = {self._mjx_model.geom_conaffinity}")
        self._recompute_mjxmodel_inaxes()
        self._rebuild_lower_funcs()
        self._check_model_inaxes()        




    def get_detected_joints(self):
        return list(self._jname2jid.keys())
    
    def get_detected_links(self):
        return list(self._lname2lid.keys())
    
    def get_detected_cameras(self):
        return list(self._cname2cid.keys())

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str,str]]):
        super().set_monitored_joints(jointsToObserve)
        self._monitored_jids = jnp.array([self._jname2jid[jn] for jn in self._monitored_joints], device=self._jax_device)
        self._monitored_qpadr = self._mjx_model.jnt_qposadr[self._monitored_jids]
        self._monitored_qvadr = self._mjx_model.jnt_dofadr[self._monitored_jids]
        self._reset_joint_state_step_stats()
        if self._record_joint_hist:
            self._full_history_labels = to_string_tensor(sum([[f"{v}.{jn[1]}" for v in ["pos","vel","cmd_eff","acc","eff","constr_eff"]] for jn in self._monitored_joints],[])).unsqueeze(0)



    @override
    def set_monitored_links(self, linksToObserve: Sequence[tuple[str,str]]):
        super().set_monitored_links(linksToObserve)
        self._monitored_lids = jnp.array([self._lname2lid[ln] for ln in self._monitored_links], device=self._jax_device)

    @override
    def step(self) -> float:
        """Run a simulation step.

        Returns
        -------
        float
            Duration of the step in simulation time (in seconds)"""
        t0 = time.monotonic()
        self._sim_state = self._clear_joint_state_step_stats(self._sim_state)
        t1 = time.monotonic()
        stepLength = self.run(self._step_length_sec)
        tf = time.monotonic()
        # ggLog.info(f"Mjx.step duration = {tf-t0}s, run = {tf-t1}s")
        return stepLength
    
    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_commands(self, sim_state : SimState) -> SimState:
        sim_state = self._apply_torque_cmds(sim_state)
        return sim_state

    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_torque_cmds(self, sim_state : SimState) -> SimState:
        return sim_state.replace_v( "mjx_data", sim_state.mjx_data.replace(qfrc_applied=sim_state.requested_qfrc_applied))








    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state"))
    def _sim_step_fast(self, iteration, sim_state : SimState) -> SimState:
        sim_state = self._apply_commands(sim_state)
        new_mjx_data = self._mjx_integrate_and_forward(self._mjx_model,sim_state.mjx_data)
        sim_state = sim_state.replace_d( {"mjx_data": new_mjx_data,
                                          "sim_time": sim_state.sim_time + self._sim_step_dt})
        sim_state = self._update_joint_state_step_stats(sim_state)
        return sim_state 


    # @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state"))
    # def _sim_step_fast_for_scan(self, sim_state : SimState, _) -> tuple[SimState, None]:
    #     sim_state = self._sim_step_fast(0, sim_state)
    #     return sim_state, None
    
    # @partial(jax.jit, static_argnames=("self","iterations"), donate_argnames=("sim_state"))
    # def _run_fast(self, sim_state : SimState, iterations : int) -> SimState:
    #     sim_state = jax.lax.scan(   self._sim_step_fast_for_scan, 
    #                                 init = sim_state,
    #                                 xs = (), 
    #                                 length = iterations)[0]
    #     # sim_state, joint_stats_arr  = jax.lax.fori_loop(lower=0, upper=iterations,
    #     #                                                 body_fun=self._sim_step_fast,
    #     #                                                 init_val=(sim_state, joint_stats_arr))
    #     return sim_state
    
    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state"))
    def _sim_step_fast_for_scan_full_pveae(self, sim_state : SimState, _) -> tuple[SimState, jnp.ndarray | None]:
        sim_state = self._sim_step_fast(0, sim_state)
        if self._record_joint_hist:
            joint_state = self._get_joint_state_for_history(sim_state=sim_state)
        else:
            joint_state = None
        return sim_state, joint_state
    
    @partial(jax.jit, static_argnames=("self","iterations"), donate_argnames=("sim_state"))
    def _run_fast_save_full_jpveae(self, sim_state : SimState, iterations : int) -> tuple[SimState, jnp.ndarray]:
        sim_state, joints_state_history = jax.lax.scan(self._sim_step_fast_for_scan_full_pveae, 
                                                  init = sim_state,
                                                  xs = (), 
                                                  length = iterations)
        # sim_state, joint_stats_arr  = jax.lax.fori_loop(lower=0, upper=iterations,
        #                                                 body_fun=self._sim_step_fast,
        #                                                 init_val=(sim_state, joint_stats_arr))
        return sim_state, joints_state_history

    def _get_joint_state_for_history(self, sim_state):
        return MjxAdapter._get_vec_joint_states_raw_pveaec(  self._monitored_qpadr,
                                                            self._monitored_qvadr,
                                                            sim_state.mjx_data)
    
    def _log_joints_pveae_history(self):
        full_history = jnp.concat(self._joints_pveae_history)
        full_history = jnp.reshape(full_history,shape=(full_history.shape[0],-1))
        # convert to torch and save as hdf5
        dir = f"{self._log_folder}/MjxAdapter_joint_hist"
        os.makedirs(dir, exist_ok=True)
        out_filename = f"{dir}/MjxAdapter_{int(time.time())}.hdf5"
        import h5py
        with h5py.File(out_filename, "w") as f:
            # loop through obs, action, reward, terminated, truncation
            try:
                f.create_dataset(f"joints_history", data=np.array(full_history))
                f.create_dataset("joints_history_labels", data=self._full_history_labels)
            except TypeError as e:
                raise RuntimeError(f"Error saving pveae, exception={e}")

    @override
    def run(self, duration_sec : float):
        """Run the environment for the specified duration"""
        run_fast_jit_compiles = self._run_fast_save_full_jpveae._cache_size()
        wt0 = time.monotonic()
        st0 = self._simTime
        # self._sent_motor_torque_commands_by_bid_jid = {}
        # ggLog.info(f"Starting run")
        iterations = int(duration_sec/self._sim_step_dt)
        self._sim_state, joints_state_history = self._run_fast_save_full_jpveae(self._sim_state, iterations)
        if self._record_joint_hist:
            self._joints_pveae_history.append(joints_state_history)
            if int(self._total_iterations / self._log_freq_joints_trajcetories) != int((self._total_iterations+iterations) / self._log_freq_joints_trajcetories):
                self._log_joints_pveae_history()
        # self._read_new_contacts()
        wtime_simulating = time.monotonic()-wt0
        # ggLog.info(f"run done.")
        if self._run_fast_save_full_jpveae._cache_size() > run_fast_jit_compiles and run_fast_jit_compiles>0:
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
        self._dbg_info.wtime_running =     time.monotonic()-wt0
        self._dbg_info.wtime_simulating =   wtime_simulating
        self._dbg_info.wtime_controlling =  0
        self._dbg_info.iterations = iterations
        self._dbg_info.rt_factor_vec =      self._vec_size*(self._simTime-st0)/wtime_simulating
        self._dbg_info.rt_factor_single =   (self._simTime-st0)/wtime_simulating
        self._dbg_info.stime_ran =          self._simTime-st0
        self._dbg_info.stime =              self._simTime
        self._dbg_info.fps_vec =            self._vec_size*iterations/wtime_simulating
        self._dbg_info.fps_single =         iterations/wtime_simulating
        self._dbg_info.run_fps_vec = self._dbg_info.fps_vec/iterations
        if self._log_freq > 0 and self._sim_step_count_since_build - self._last_log_iters >= self._log_freq:
            self._last_log_iters = self._sim_step_count_since_build
            ggLog.info( "MjxAdapter:\n"+"\n".join(["    "+str(k)+' : '+str(v) for k,v in self.get_debug_info().items()]))
        self._sim_stepping_wtime_since_build += wtime_simulating
        self._sim_step_count_since_build += iterations
        self._run_wtime_since_build += time.monotonic()-wt0

        return self._simTime-st0
    
    # @override
    # def run(self, duration_sec : float):
    #     """Run the environment for the specified duration"""
    #     tf0 = time.monotonic()

    #     # self._sent_motor_torque_commands_by_bid_jid = {}

    #     stepping_wtime = 0
    #     simulating_wtime = 0
    #     control_wtime = 0
    #     vsteps_done = 0
    #     st0 = self._simTime
    #     while self._simTime-st0 < duration_sec:
    #         wtps = time.monotonic()
    #         self._sim_state = self._apply_commands(self._sim_state)
    #         wt1 = time.monotonic()
    #         control_wtime += wt1-wtps
    #         new_mjx_data = self._mjx_integrate_and_forward(self._mjx_model,self._sim_state.mjx_data)
    #         self._sim_state = self._sim_state.replace( "mjx_data", new_mjx_data)
    #         simulating_wtime += time.monotonic()-wt1
    #         self._sim_state = self._sim_state.replace( "sim_time", self._sim_state.sim_time + self._sim_step_dt)

    #         self._monitored_joints_stats = self._update_joint_state_step_stats(self._sim_state, self._monitored_joints_stats)
    #         stepping_wtime += time.monotonic()-wtps
    #         self._sim_step_count_since_build += 1
    #         vsteps_done += 1
    #         # self._read_new_contacts()
            
    #         self._simTime += self._sim_step_dt
    #         if self._realtime_factor is not None and self._realtime_factor>0:
    #             sleep_time = self._sim_step_dt*(1/self._realtime_factor) - (time.monotonic()-self._prev_step_end_wtime)
    #             if sleep_time > 0:
    #                 time.sleep(sleep_time)
    #         self._prev_step_end_wtime = time.monotonic()
    #         self._update_gui()
    #     # self._last_sent_torques_by_name = {self._bodyAndJointIdToJointName[bid_jid]:torque 
    #     #                                     for bid_jid,torque in self._sent_motor_torque_commands_by_bid_jid.items()}
    #     self._dbg_info.wtime_stepping =    stepping_wtime
    #     self._dbg_info.wtime_simulating =  simulating_wtime
    #     self._dbg_info.wtime_controlling = control_wtime
    #     self._dbg_info.rt_factor_vec = self._vec_size*(self._simTime-st0)/stepping_wtime
    #     self._dbg_info.rt_factor_single = (self._simTime-st0)/stepping_wtime
    #     self._dbg_info.stime_ran =  self._simTime-st0
    #     self._dbg_info.stime =  self._simTime
    #     self._dbg_info.fps_vec =     self._vec_size*vsteps_done/stepping_wtime
    #     self._dbg_info.fps_single =  vsteps_done/stepping_wtime
    #     if self._log_freq > 0 and self._sim_step_count_since_build % self._log_freq == 0:
    #         ggLog.info( "\n".join([str(k)+' : '+str(v) for k,v in self.get_debug_info().items()]))
    #     self._sim_stepping_wtime_since_build += stepping_wtime
    #     self._run_wtime_since_build += time.monotonic()-tf0

    #     return self._simTime-st0
    
    def get_debug_info(self) -> dict[str,th.Tensor]:
        return {k:th.as_tensor(v) for k,v in dataclasses.asdict(self._dbg_info).items()}

    # @staticmethod
    # # @jax.jit
    # def _take_batch_element(batch, e):
    #     mjx_data = jax.tree_map(lambda l: l[e], batch)
    #     return mjx.get_data(mj_model, mjx_data)
    
    @staticmethod
    # @jax.jit
    def _tree_unstack(tree):
        leaves, treedef = jax.tree.flatten(tree)
        return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

    @override
    def getRenderings(self, requestedCameras : list[str], vec_mask : th.Tensor | None) -> tuple[list[th.Tensor], th.Tensor]:
        if self._renderer is None:
            raise RuntimeError(f"Called getRenderings, but rendering is not initialized. did you set enable_rendering?")
        if vec_mask is None:
            vec_mask = self._all_vecs_thcpu
        selected_vecs = th.nonzero(vec_mask, as_tuple=True)[0].to("cpu")
        nvecs = selected_vecs.shape[0]
        
        # mj_data_batch = mjx.get_data(self._mj_model, self._sim_state.mjx_data)
        # print(f"mj_data_batch = {mj_data_batch}")
        times = th.as_tensor(self._simTime).repeat((nvecs,len(requestedCameras)))
        image_batches = [np.ones(shape=(nvecs,)+self._camera_sizes[cam]+(3,), dtype=np.uint8) for cam in requestedCameras]
        self._forward_if_needed()
        # print(f"images.shapes = {[i.shape for i in images]}")
        # mj_datas : list[mujoco.MjData] = mjx.get_data(self._mj_model, self._sim_state.mjx_data)
        mjx.get_data_into(self._renderers_mj_datas,self._mj_model, self._sim_state.mjx_data)
        for env_i,env in enumerate(selected_vecs):
            for i in range(len(requestedCameras)):
                cam = requestedCameras[i]
                # print(f"self._mj_model.cam_resolution[cid] = {self._mj_model.cam_resolution[self._cname2cid[cam]]}")
                renderer = self._renderers[self._camera_sizes[cam]]
                mujoco.mj_camlight(self._mj_model, self._renderers_mj_datas[env]) # see https://github.com/google-deepmind/mujoco/issues/1806
                renderer.update_scene(self._renderers_mj_datas[env], self._cname2cid[cam]) #, scene_option=self._render_scene_option)
                image_batches[i][env_i] = renderer.render()
                # renderer.render(out=images[i][env])
        return [th.as_tensor(img_batch) for img_batch in image_batches], times


    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | jnp.ndarray | None = None) -> th.Tensor:
        if requestedJoints is None:
            jids = self._monitored_jids
        elif isinstance(requestedJoints, jnp.ndarray):
            jids = requestedJoints
        else:
            jids = self.get_joints_ids(requestedJoints)
        if len(jids) == 0:
            return th.empty(size=(self._vec_size,self._sim_state.mjx_data.qpos.shape[1],0,3), dtype=th.float32)
        else:
            self._forward_if_needed()           
            t = self._get_vec_joint_states_pve(self._mjx_model, self._sim_state.mjx_data, jids)
        return jax2th(t, th_device=self._out_th_device)
    
    @override
    def getExtendedJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        if requestedJoints is None:
            jids = self._monitored_jids
        else:
            jids = jnp.array([self._jname2jid[jn] for jn in requestedJoints], device=self._jax_device)
        if len(jids) == 0:
            return th.empty(size=(self._vec_size,self._sim_state.mjx_data.qpos.shape[1],0,3), dtype=th.float32)
        else:
            self._forward_if_needed()
            t = self._get_vec_joint_states_pveae(self._mjx_model, self._sim_state.mjx_data, jids)
        return jax2th(t, th_device=self._out_th_device)
    
    @staticmethod
    # @jax.jit
    def _get_vec_joint_states_pveae(mjx_model, mjx_data, jids : jnp.ndarray):
        return MjxAdapter._get_vec_joint_states_raw_pveae(mjx_model.jnt_qposadr[jids],
                                                        mjx_model.jnt_dofadr[jids],
                                                        mjx_data)
    
    @staticmethod
    # @jax.jit
    def _get_vec_joint_states_pve(mjx_model, mjx_data, jids : jnp.ndarray):
        return MjxAdapter._get_vec_joint_states_raw_pve(mjx_model.jnt_qposadr[jids],
                                                        mjx_model.jnt_dofadr[jids],
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
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qfrc_applied[:,qvadr], #commanded effort
                            mjx_data.qacc[:,qpadr],
                            mjx_data.qfrc_smooth[:,qvadr] + mjx_data.qfrc_constraint[:,qvadr]], #actual effort
                            axis = 2)
    
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
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qfrc_applied[:,qvadr], #commanded effort
                            mjx_data.qacc[:,qpadr],
                            mjx_data.qfrc_smooth[:,qvadr] + mjx_data.qfrc_constraint[:,qvadr], #actual effort
                            mjx_data.qfrc_constraint[:,qvadr]],
                            axis = 2)
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pve(qpadr, qvadr, mjx_data):
        # What should we use as torque readings?
        # - Emo Todorov here https://www.roboti.us/forum/index.php?threads/best-way-to-represent-robots-torque-sensors.4181 says 
        #     that qfrc_unc (now renamed to qfrc_smmooth) + qfrc_constraint shoudl give what a torque sensor would measure
        # - We also could use qfrc_applied, which is the torque we are applying, I think in most cases this should be correct
        # ggLog.info(f"qfrc_applied={mjx_data.qfrc_applied[:,qvadr]}\n"
        #            f"qfrc_smooth={mjx_data.qfrc_smooth[:,qvadr]}\n"
        #            f"qfrc_constraint={mjx_data.qfrc_constraint[:,qvadr]}"
        #            f"qfrc_passive={mjx_data.qfrc_constraint[:,qvadr]}"
        #            f"qfrc_bias={mjx_data.qfrc_constraint[:,qvadr]}")
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qfrc_applied[:,qvadr]], 
                            axis = 2)
    
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pvae(qpadr, qvadr, mjx_data):
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qacc[:,qvadr],
                            mjx_data.qfrc_applied[:,qvadr]], 
                            axis = 2)


    def _clear_joint_state_step_stats(self, sim_state : SimState):
        sim_state = self._init_stats(sim_state)
        return sim_state

    @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"])
    def _init_stats(sim_state : SimState):
        stats_array = sim_state.mon_joint_stats_arr_pvae
        stats_array = stats_array.at[:,0].set(float("+inf"))
        stats_array = stats_array.at[:,1].set(float("-inf"))
        stats_array = stats_array.at[:,2].set(float("nan"))
        stats_array = stats_array.at[:,3].set(float("nan"))
        stats_array = stats_array.at[:,4].set(0)
        stats_array = stats_array.at[:,5].set(0)
        sim_state = sim_state.replace_d({"stats_step_count": 0,
                                         "mon_joint_stats_arr_pvae" : stats_array})
        return sim_state

    @staticmethod
    @partial(jax.jit, donate_argnames=["sim_state"])
    def _update_joint_state_step_stats_arrs(current_jstate_pvae : jnp.ndarray,
                                            sim_state : SimState):
        stats_array = sim_state.mon_joint_stats_arr_pvae
        step_count = sim_state.stats_step_count + 1
        stats_array = stats_array.at[:,4].set(jnp.add(    stats_array[:,4], current_jstate_pvae)) # sum of values
        stats_array = stats_array.at[:,5].set(jnp.add(    stats_array[:,5], jnp.square(current_jstate_pvae))) # sum of squares

        stats_array = stats_array.at[:,0].set(jnp.minimum(stats_array[:,0], current_jstate_pvae))
        stats_array = stats_array.at[:,1].set(jnp.maximum(stats_array[:,1], current_jstate_pvae))
        stats_array = stats_array.at[:,2].set(stats_array[:,4]/step_count) # average values
        stats_array = stats_array.at[:,3].set(jnp.sqrt(jnp.clip(stats_array[:,5]/step_count-jnp.square(stats_array[:,2]),min=0))) # standard deviation
        sim_state = sim_state.replace_d({"stats_step_count" : step_count,
                                         "mon_joint_stats_arr_pvae" : stats_array})
        # jax.debug.print("updated stats: count={c}, arr={arr}", c=step_count, arr=stats_array)
        return sim_state

    @partial(jax.jit, static_argnames=["self"], donate_argnames=["sim_state"])
    def _update_joint_state_step_stats(self, sim_state : SimState) -> SimState:
        jstate_pvae = self._get_vec_joint_states_raw_pvae(self._monitored_qpadr, self._monitored_qvadr, sim_state.mjx_data)
        sim_state = self._update_joint_state_step_stats_arrs(jstate_pvae, sim_state)
        return sim_state

    def _reset_joint_state_step_stats(self):
        # ggLog.info(f"resetting stats")
        sim_state = self._sim_state
        sim_state = sim_state.replace_v("mon_joint_stats_arr_pvae",
                                        jnp.zeros(shape=(self._vec_size, 6, len(self._monitored_joints),4),
                                                    dtype=jnp.float32,
                                                    device=self._jax_device))
        sim_state = self._clear_joint_state_step_stats(sim_state)
        sim_state = self._update_joint_state_step_stats(sim_state) # populate with current state, so that there are safe-ish values here
        self._sim_state = sim_state

    def get_joints_state_step_stats(self) -> th.Tensor:
        return jax2th(self._sim_state.mon_joint_stats_arr_pvae[:,:4], self._out_th_device)

    @partial(jax.jit, static_argnums=(0,))
    def _get_links_state_jax(self, body_ids : jnp.ndarray, mjx_data) -> jnp.ndarray:
        return jnp.concatenate([mjx_data.xpos[:,body_ids], # frame position
                                mjx_data.xquat[:,body_ids][:,:,self._wxyz2xyzw], # frame orientation
                                mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]]], axis = -1) # frame linear and angular velocity
    
    @staticmethod
    @jax.jit
    def _get_links_com_state_jax(body_ids : jnp.ndarray, mjx_data) -> jnp.ndarray:
        """ The COM orientation is aligned along the principal axes of inertia, 
            as stated in https://mujoco.readthedocs.io/en/stable/XMLreference.html?utm_source=chatgpt.com#body-inertial
            I believe this means the x ends up on the highes inertia axis and so on
            So for example on the main body of a usual quadruped the x would point down, the y sideways and the z front/back
        """
        return jnp.concatenate([mjx_data.xipos[:,body_ids], # com position
                                jax_mat_to_quat_xyzw(mjx_data.ximat[:,body_ids]), # com orientation, see above
                                mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]]], axis = -1) #com linear and angular velocity
    
    @override
    def get_links_ids(self, link_names : Sequence[tuple[str,str]]):
        return jnp.array([self._lname2lid[ln] for ln in link_names], device=self._jax_device) # TODO: would make sense to return a mask here instead of indexes

    @override
    def get_joints_ids(self, joint_names : Sequence[tuple[str,str]]):
        return jnp.array([self._jname2jid[jn] for jn in joint_names], device=self._jax_device) # TODO: would make sense to return a mask here instead of indexes

    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | jnp.ndarray | None = None, use_com_frame : bool = False) -> th.Tensor:
        # th.cuda.synchronize()
        # t0 = time.monotonic()
        if requestedLinks is None:
            body_ids = self._monitored_lids
        elif isinstance(requestedLinks, jnp.ndarray):
            body_ids = requestedLinks
        else:
            body_ids = self.get_links_ids(requestedLinks)
        # th.cuda.synchronize()
        # t1 = time.monotonic()
        self._forward_if_needed()
        if use_com_frame:
            t = self._get_links_com_state_jax(body_ids, self._sim_state.mjx_data)
        else:
            t = self._get_links_state_jax(body_ids, self._sim_state.mjx_data)
        # th.cuda.synchronize()
        # t2 = time.monotonic()
        r=jax2th(t, th_device=self._out_th_device)
        # th.cuda.synchronize()
        # t3 = time.monotonic()
        # ggLog.info(f"getLinksState: getids={t1-t0} getvals={t2-t1} convert={t3-t2}")
        return r

    @override
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        ggLog.info(f"MjxAdapter resetting")
        self.__lastResetTime = self.getEnvTimeFromStartup()

        self._sim_state = self._sim_state.replace_v( "mjx_data", copy.deepcopy(self._original_mjx_data))
        self._mjx_model = copy.deepcopy(self._original_mjx_model)
        self._mj_data = copy.deepcopy(self._original_mj_data)
        self._mj_model = copy.deepcopy(self._original_mj_model)
        self._reset_joint_state_step_stats()
        self._recompute_mjxmodel_inaxes()
        


    @override
    def getEnvTimeFromStartup(self) -> float:
        """Get the current time within the simulation."""
        return self._simTime


    @override    
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor | None = None):
        # ggLog.info(f"setJointsStateDirect(\n{joint_names}, \n{joint_states_pve}, \n{vec_mask})")

        if self._check_sizes and joint_states_pve.size() != (self._vec_size,len(joint_names),3):
            raise RuntimeError(f"joint_states_pve should have size {(self._vec_size,len(joint_names),3)}, but it's {joint_states_pve.size()}")
        jids = jnp.array([self._jname2jid[jn] for jn in joint_names])
        # contiguous_js = joint_states_pve.contiguous()
        # ggLog.info(f"joint_states_pve.size() = {joint_states_pve.size()}")
        # ggLog.info(f"joint_states_pve.dim_order() = {joint_states_pve.dim_order()}")
        # ggLog.info(f"joint_states_pve.stride() = {joint_states_pve.stride()}")
        # ggLog.info(f"contiguous_js.size() = {contiguous_js.size()}")
        # ggLog.info(f"contiguous_js.dim_order() = {contiguous_js.dim_order()}")
        # ggLog.info(f"contiguous_js.stride() = {contiguous_js.stride()}")
        js_pve = th2jax(joint_states_pve, jax_device=self._jax_device)
        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs
        # ggLog.info(f"js_pve.shape = {js_pve.shape}")

        jtypes = self._mj_model.jnt_type[jids]
        if not jnp.all(jnp.logical_or(jtypes == mujoco.mjtJoint.mjJNT_HINGE, jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
            raise RuntimeError(f"Cannot control set state for multi-dimensional joint, types = {list(zip(joint_names,jtypes))}")
        qpadr = self._mj_model.jnt_qposadr[jids]
        qvadr = self._mj_model.jnt_dofadr[jids]
        # ggLog.info(f"self._sim_state.mjx_data.qpos[{vec_mask_jnp},{qpadr}].shape = {get_rows_cols(self._sim_state.mjx_data.qpos, [vec_mask_jnp,qpadr]).shape}")
        # ggLog.info(f"js_pve[vec_mask_jnp,:,0].shape = {js_pve[vec_mask_jnp,:,0].shape}")
        qpos = set_rows_cols(self._sim_state.mjx_data.qpos,           (vec_mask_jnp,qpadr), js_pve[vec_mask_jnp,:,0])
        qvel = set_rows_cols(self._sim_state.mjx_data.qvel,           (vec_mask_jnp,qvadr), js_pve[vec_mask_jnp,:,1])
        qeff = set_rows_cols(self._sim_state.mjx_data.qfrc_applied,   (vec_mask_jnp,qvadr), js_pve[vec_mask_jnp,:,2])
        mjx_data = self._sim_state.mjx_data.replace(qpos=qpos, qvel=qvel, qfrc_applied=qeff)
        self._sim_state = self._sim_state.replace_v( "mjx_data", mjx_data)
        self._mark_forward_needed()
        # self._update_gui(force=True)
        # ggLog.info(f"setted_jstate Simtime [{self._simTime:.9f}] step [{self._sim_step_count_since_build}] monitored jstate:\n{self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._sim_state.mjx_data)}")


    def _update_gui(self, force : bool = False):
        if self._show_gui and (time.monotonic() - self._last_gui_update_wtime > 1/self._gui_freq or force):
            self._forward_if_needed()
            mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._sim_state.mjx_data))
            self._last_gui_update_wtime = time.monotonic()
            self._viewer.sync()

    @override
    def setLinksStateDirect(self, link_names : list[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor | None = None):
        # ggLog.info(f"setJointsStateDirect(\n{link_names}, \n{link_states_pose_vel}, \n{vec_mask})")

        link_states_pose_vel_jnp = th2jax(link_states_pose_vel, jax_device=self._jax_device)
        if vec_mask is not None:
            vec_mask_jnp = th2jax(vec_mask, jax_device=self._jax_device)
        else:
            vec_mask_jnp = self._all_vecs
        model_body_pos = self._mjx_model.body_pos
        model_body_quat = self._mjx_model.body_quat
        data_joint_pos = self._sim_state.mjx_data.qpos
        data_joint_vel = self._sim_state.mjx_data.qvel
        for i, link_name in enumerate(link_names):
            # ggLog.info(f"setting link state for {link_name}")
            lid = self._lname2lid[link_name]
            root_body_id = self._mj_model.body_rootid[lid]
            if lid != root_body_id:
                raise RuntimeError(f"Can only set link state for root links, but {link_name} is not one.")
            if lid == 0:
                raise RuntimeError(f"Cannot set link state for world link")
            
            # Contrary to what you might expect mujoco associates each body to multiple possible parent joints
            # so:
            # - mj_model.body_jntnum[link_id] is the number of parent joints of a body
            # - mj_model.body_jntadr[link_id] is the id of the first of these parent joints
            # - mj_model.jnt_qposadr[joint_id] is the qpos addredd of a specific joint id
            parent_joints_num = self._mj_model.body_jntnum[lid]
            parent_body_id = self._mj_model.body_parentid[lid]
            if parent_joints_num == 0 and parent_body_id==0: # if it has no parent joints
                # ggLog.info(f"changing 'fixed joint'")
                #    Fixed joints cannot be set to different positions across the vectorized simulations.
                #    This because MJX does not vectorize the MjModel, all vec simulations use the same model,
                #     and fixed joints are represented as fixed transforms in the model.

                #TODO: the following line triggers jit recompile on:  dynamice_slice, squeeze, broadcast_in_dim
                if not jnp.all(jnp.array_equal(link_states_pose_vel_jnp[:,i], jnp.broadcast_to(link_states_pose_vel_jnp[0,i], shape=link_states_pose_vel_jnp[:,i].shape),equal_nan=True)):
                    raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations.\n"
                                       f"{link_states_pose_vel_jnp[0,i]}\n"
                                       f"!=\n"
                                       f"{link_states_pose_vel_jnp[:,i]}")
                #TODO: the following line triggers jit recompile
                if jnp.any(vec_mask_jnp != vec_mask_jnp[0]):
                    raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations, but vec_mask has different values.")
                #TODO: the following line triggers jit recompile on:  dynamice_slice, squeeze, convert_element_type
                model_body_pos = model_body_pos.at[lid].set(link_states_pose_vel_jnp[0,i,:3])
                #TODO: the following line triggers jit recompile, also on add, select_n, concatenate, gather, scatter
                model_body_quat = model_body_quat.at[lid].set(link_states_pose_vel_jnp[0,i,[6,3,4,5]])
                # print(f"self._mjx_model.body_pos = {self._mjx_model.body_pos}")
            elif parent_joints_num == 1 and parent_body_id==0:
                jid = self._mj_model.body_jntadr[lid]
                jtype = self._mj_model.jnt_type[jid]
                if jtype == mujoco.mjtJoint.mjJNT_FREE:
                    # ggLog.info(f"writing at qpos[{self._mj_model.jnt_qposadr[jid]}:{self._mj_model.jnt_qposadr[jid]+7}]")
                    qadr = self._mj_model.jnt_qposadr[jid]
                    dadr = self._mj_model.jnt_dofadr[jid]
                    data_joint_pos = set_rows_cols(data_joint_pos,
                                                   (vec_mask_jnp, jnp.arange(qadr, qadr+7)),
                                                   get_rows_cols(link_states_pose_vel_jnp, (vec_mask_jnp,i,[0,1,2,6,3,4,5])))
                    data_joint_vel = set_rows_cols(data_joint_vel,
                                                   (vec_mask_jnp, jnp.arange(dadr, dadr+6)),
                                                   link_states_pose_vel_jnp[vec_mask_jnp,i,7:13]) #get_rows_cols(link_states_pose_vel_jnp, (vec_mask_jnp,i,jnp.arange(7,13))))
                    # data_joint_pos = (data_joint_pos.at[vec_mask_jnp,qadr:qadr+7]
                    #                                 .set(link_states_pose_vel_jnp[vec_mask_jnp,i,[0,1,2,6,3,4,5]]))
                    # data_joint_vel = (data_joint_vel.at[vec_mask_jnp,dadr:dadr+6]
                    #                                 .set(link_states_pose_vel_jnp[vec_mask_jnp,i,7:13]))
                    # raise NotImplementedError()
                else:
                    raise NotImplementedError(f"Cannot set link state for link {link_name} with parent joint of type {jtype} (see mjtJoint enum)")
            else:
                raise NotImplementedError(f"Cannot set link state for link {link_name} with {self._mj_model.body_jntnum} parent joints and parent body {parent_body_id}")
        self._mjx_model = self._mjx_model.replace(body_pos=model_body_pos, body_quat = model_body_quat)
        mjx_data = self._sim_state.mjx_data.replace(qpos=data_joint_pos, qvel=data_joint_vel)
        self._sim_state = self._sim_state.replace_v( "mjx_data", mjx_data)
        self._recompute_mjxmodel_inaxes()
        self._mark_forward_needed()
        # print(f"self._mjx_model.body_pos = {self._mjx_model.body_pos}")        
        # print(f"self._sim_state.mjx_data.qpos = {self._sim_state.mjx_data.qpos}")        
        # self._mjx_model = mjx.put_model(self._mj_model, device=self._jax_device)

        

        # self._update_gui(True)
        # ggLog.info(f"setted_lstate Simtime [{self._simTime:.9f}] step [{self._sim_step_count_since_build}] monitored jstate:\n{self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._sim_state.mjx_data)}")

    def _mark_forward_needed(self):
        self._forward_needed = True

    def _forward_if_needed(self):
        if self._forward_needed:
            self._check_model_inaxes()

            data = self._mjx_forward(self._mjx_model,self._sim_state.mjx_data)
            self._sim_state = self._sim_state.replace_v( "mjx_data", data)
            self._forward_needed = False

    def _check_model_inaxes(self):
        # from jax._src.tree_util import prefix_errors
        # all_errors = prefix_errors(self._mjx_model_in_axes, self._mjx_model)
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
    def setJointsEffortCommand(self, joint_names : Sequence[tuple[str,str]] | None, efforts : th.Tensor) -> None:
        jids = jnp.array([self._jname2jid[jn] for jn in joint_names])
        qeff = th2jax(efforts, jax_device=self._jax_device)
        self._sim_state = self._set_effort_command(self._sim_state, jids,qeff)
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
        qvadr = self._jnt_dofadr_jax[jids]
        if sims_mask is None:
            sim_state = sim_state.replace_v( "requested_qfrc_applied",
                                  sim_state.requested_qfrc_applied.at[:,qvadr].set(qefforts[:,:]))
            # sim_state.requested_qfrc_applied = sim_state.requested_qfrc_applied.at[:,qvadr].set(qefforts[:,:])
        else:
            sim_state = sim_state.replace_v( "requested_qfrc_applied",
                                  set_rows_cols_masks(sim_state.requested_qfrc_applied, [sims_mask, qvadr], qefforts[:,:]))
            # sim_state.requested_qfrc_applied = set_rows_cols_masks(sim_state.requested_qfrc_applied, [sims_mask, qvadr], qefforts[:,:])
        return sim_state

    # def _set_effort_command(self,   sim_state : SimState,
    #                                 jids : jnp.ndarray,
    #                                 qefforts : jnp.ndarray,
    #                                 sims_mask : jnp.ndarray | None = None) -> SimState:
    #     """Set the efforts to be applied on a set of joints.

    #     Effort means either a torque or a force, depending on the type of joint.

    #     Parameters
    #     ----------
    #     joint_names : jnp.ndarray
    #         Array with the joint ids for each effort command, 1-dimensional
    #     efforts : th.Tensor
    #         Tensor of shape (vec_size, len(jids)) containing the effort for each joint in each environment.
    #     """
    #     qvadr = self._mj_model.jnt_dofadr[jids]
    #     if sims_mask is None:
    #         sim_state.requested_qfrc_applied = sim_state.requested_qfrc_applied.at[:,qvadr].set(qefforts[:,:])
    #     else:
    #         sims_indexes = jnp.nonzero(sims_mask)[0]
    #         sim_state.requested_qfrc_applied = sim_state.requested_qfrc_applied.at[sims_indexes[:,jnp.newaxis],qvadr].set(qefforts[:,:])
    #     return sim_state


    # def alter_model(self, link_masses : tuple[jnp.ndarray, th.Tensor]):
    #     """_summary_

    #     Parameters
    #     ----------
    #     link_masses : tuple[jnp.ndarray, th.Tensor]
    #         tuple containing alist of link ids (from get_link_id) and corresponding
    #         body masses, body masses should be in a tensor of size (vec_size, len(link_ids))
    #     """
    #     # We need to be able to alter:
    #     #    body masses (body_mass)
    #     #    body frictions (geom_friction, in the xml there are sliding, torsional and rolling friction, where are they in mjmodel?)
    #     #    joint coulomb friction (dof_frictionloss, see https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
    #     #    joint rotational inertia (dof_armature, see https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
    #     #    some world parameters? e.g. gravity
    #     body_masses = th2jax(link_masses[1],jax_device=self._jax_device)
    #     body_ids = link_masses[0]
    #     replacements = {}
    #     if len(body_ids)>0:
    #         replacements["body_mass"] = self._mjx_model.body_mass.at[:,body_ids].set(body_masses)
    #     self._mjx_model = self._mjx_model.replace(**replacements)
    #     # self._recompute_mjxmodel_inaxes() # Is it really necessary?



    def alter_model_rel(self, link_masses : tuple[jnp.ndarray, th.Tensor],
                              link_frictions : tuple[jnp.ndarray, th.Tensor] | None = None):
        """_summary_

        Parameters
        ----------
        link_masses : tuple[jnp.ndarray, th.Tensor]
            tuple containing alist of link ids (from get_link_id) and corresponding
            body masses, body masses should be in a tensor of size (vec_size, len(link_ids))
        """
        # We need to be able to alter:
        #    body masses (body_mass)
        #    body frictions (geom_friction, in the xml there are sliding, torsional and rolling friction, where are they in mjmodel?)
        #    joint coulomb friction (dof_frictionloss, see https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
        #    joint rotational inertia (dof_armature, see https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
        #    some world parameters? e.g. gravity
        body_masses_ratio_change = th2jax(link_masses[1],jax_device=self._jax_device)
        if link_frictions is not None:
            body_frictions_ratio_change = th2jax(link_frictions[1],jax_device=self._jax_device)
        replacements = {}

        masses_body_ids = link_masses[0]
        if len(masses_body_ids)>0:            
            current_mass = self._mjx_model.body_mass[:,masses_body_ids]
            replacements["body_mass"] = self._mjx_model.body_mass.at[:,masses_body_ids].set(current_mass + current_mass*body_masses_ratio_change)
        if link_frictions is not None:
            frictions_body_ids = link_frictions[0]
            frictions_body_ids_mask = jnp.zeros(shape=(self._mjx_model.nbody,),dtype=jnp.bool, device=self._jax_device)
            frictions_body_ids_mask = frictions_body_ids_mask.at[frictions_body_ids].set(1)
            frictions_geoms_ids_mask = frictions_body_ids_mask[self._mjx_model.geom_bodyid]
            full_body_frictions_ratio_change = jnp.ones(shape=(self._vec_size, self._mjx_model.nbody,3),dtype=jnp.float32, device=self._jax_device)
            full_body_frictions_ratio_change = full_body_frictions_ratio_change.at[:,frictions_body_ids].set(body_frictions_ratio_change)
            geom_friction_ratios = full_body_frictions_ratio_change[:,self._mjx_model.geom_bodyid]
            replacements["geom_friction"] = jnp.where(jnp.expand_dims(frictions_geoms_ids_mask,1).repeat(repeats=3,axis=1),
                                                      self._mjx_model.geom_friction+self._mjx_model.geom_friction*geom_friction_ratios,
                                                      self._mjx_model.geom_friction)
        self._mjx_model = self._mjx_model.replace(**replacements)
        # self._recompute_mjxmodel_inaxes() # Is it really necessary?
