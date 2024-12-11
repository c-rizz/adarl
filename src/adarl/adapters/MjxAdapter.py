from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_FLAGS"]="--xla_gpu_triton_gemm_any=true"

from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import Pose, compile_xacro_string, pkgutil_get_path, exc_to_str, build_pose
from typing import Any
import jax
import mujoco
from mujoco import mjx
import mujoco.viewer
import time
from typing_extensions import override
from typing import overload, Sequence, Mapping
import torch as th
import jax.numpy as jnp
import jax.scipy.spatial.transform
import adarl.utils.dbg.ggLog as ggLog
import torch.utils.dlpack as thdlpack
import numpy as np
import copy
from typing import Iterable
from functools import partial

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
    return jnp.from_dlpack(tensor.contiguous().cuda()).to_device(jax_device)
                                                    
def jax2th(array : jnp.ndarray, th_device : th.device):
    return thdlpack.from_dlpack(array.to_device(jax.devices("gpu")[0])).to(th_device)

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

class MjxAdapter(BaseVecSimulationAdapter, BaseVecJointEffortAdapter):
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
                        log_freq : int = 1):
        super().__init__(vec_size=vec_size,
                         output_th_device=output_th_device)
        self._enable_rendering = enable_rendering
        self._jax_device = jax_device
        self._sim_step_dt = sim_step_dt
        self._step_length_sec = step_length_sec
        self._simTime = 0.0
        self._sim_step_count_since_build = 0
        self._sim_stepping_wtime_since_build = 0
        self._run_wtime_since_build = 0
        self._add_ground = add_ground
        self._log_freq = log_freq

        self._realtime_factor = realtime_factor
        self._wxyz2xyzw = jnp.array([1,2,3,0], device = jax_device)
        self._mjx_data : mjx.Data
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

    @staticmethod
    def _mj_name_to_pair(mjname : str):
        if mjname == "world":
            return mjname,mjname
        sep = mjname.find(model_element_separator)
        if mjname.count(model_element_separator) != 1:
            raise RuntimeError(f"Invalid mjName: must contain one and only one '{model_element_separator}' substring, but it's {mjname}")
        return mjname[:sep],mjname[sep+len(model_element_separator):]

    @override
    def build_scenario(self, models : list[ModelSpawnDef]):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
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
                                                                            <light pos="0 0 10" dir="0 0 -1" directional="true" />
                                                                            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin="0.0" />
                                                                        </body>
                                                                    </worldbody>
                                                                </mujoco>""",
                                           format="mjcf",
                                           pose=build_pose(0,0,0,0,0,0,1),
                                           kwargs={}))
        specs = []
        for model in models:
            mjSpec = mujoco.MjSpec()
            if model.format.strip().lower()[-6:] == ".xacro":
                def_string = compile_xacro_string( model_definition_string=model.definition_string,
                                                                model_kwargs=model.kwargs)
            elif model.format.strip().lower() in ("urdf","mjcf"):
                def_string = model.definition_string
            else:
                raise RuntimeError(f"Unsupported model format '{model.format}' for model '{model.name}'")
            ggLog.info(f"Adding model: \n{def_string}")
            mjSpec.from_string(def_string)
            specs.append((model.name, mjSpec))
        big_speck = mujoco.MjSpec()
        
        frame = big_speck.worldbody.add_frame()
        big_speck.degree = False
        for mname, spec in specs:
            if model_element_separator in mname:
                raise RuntimeError(f"Cannot have models with '#' in their name (this character is used internally). Found model named {mname}")
            # add all th bodies that are direct childern of worldbody
            body = spec.worldbody.first_body()
            if spec.degree:
                raise NotImplementedError(f"model {mname} uses degrees instead of radians.")
            while body is not None:
                frame.attach_body(body, mname+model_element_separator, "")
                body = spec.worldbody.next_body(body)

        self._mj_model = big_speck.compile()
        self._mj_model.opt.timestep = self._sim_step_dt
        ggLog.info(f"big_speck.degree = {big_speck.degree}")
        ggLog.info(f"Spawing: \n{big_speck.to_xml()}")

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



        self._mj_data = mujoco.MjData(self._mj_model)
        mujoco.mj_resetData(self._mj_model, self._mj_data)

        self._mjx_model = mjx.put_model(self._mj_model, device = self._jax_device)
        self._mjx_data = mjx.put_data(self._mj_model, self._mj_data, device = self._jax_device)
        ggLog.info(f"mjx_data.qpos.shape = {self._mjx_data.qpos.shape}")
        self._mjx_data = jax.vmap(lambda: self._mjx_data, axis_size=self._vec_size)()
        # self._mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(jnp.arange(self._vec_size), self._mjx_data)
        ggLog.info(f"mjx_data.qpos.shape = {self._mjx_data.qpos.shape}")


        self._original_mjx_data = copy.deepcopy(self._mjx_data)
        self._original_mjx_model = copy.deepcopy(self._mjx_model)
        self._original_mj_data = copy.deepcopy(self._mj_data)
        self._original_mj_model = copy.deepcopy(self._mj_model)

        ggLog.info(f"Compiling mjx_integrate_and_forward....")
        self._mjx_integrate_and_forward = jax.jit(jax.vmap(mjx_integrate_and_forward, in_axes=(None, 0))) #, donate_argnames=["d"]) donating args make it crash
        _ = self._mjx_integrate_and_forward(self._mjx_model, copy.deepcopy(self._mjx_data)) # trigger jit compile
        # ggLog.info(f"Compiling mjx.step....")
        # self._mjx_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0))) #, donate_argnames=["d"]) donating args make it crash
        # _ = self._mjx_step(self._mjx_model, copy.deepcopy(self._mjx_data)) # trigger jit compile
        ggLog.info(f"Compiling mjx.forward....")
        self._mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self._mjx_data = self._mjx_forward(self._mjx_model, self._mjx_data) # compute initial mjData
        ggLog.info(f"Compiled MJX.")
        
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
            self._viewer_mj_data : mujoco.MjData = mjx.get_data(self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._mjx_data))
            mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._mjx_data))
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._viewer_mj_data)

        self._jid2jname : dict[int, tuple[str,str]] = {jid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid))
                           for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname : dict[int, tuple[str,str]] = {lid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, lid))
                           for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}
        ggLog.info(f"self._lname2lid = {self._lname2lid}")
        self._cid2cname : dict[int, str] = {jid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, jid))[1]
                           for jid in range(self._mj_model.ncam)}
        self._cname2cid = {cn:cid for cid,cn in self._cid2cname.items()}
        self._camera_sizes :dict[str,tuple[int,int]] = {self._cid2cname[cid]:(self._mj_model.cam_resolution[cid][1],self._mj_model.cam_resolution[cid][0]) for cid in self._cid2cname}
        if self._enable_rendering:
            def make_renderer(h,w):
                ggLog.info(f"Making renderer for size {h}x{w}")
                return mujoco.Renderer(self._mj_model,height=h,width=w)
            self._renderers : dict[tuple[int,int],mujoco.Renderer]= {resolution:make_renderer(resolution[0],resolution[1])
                            for resolution in set(self._camera_sizes.values())}
        else:
            self._renderers = {}
        
        print("Joint limits:\n"+("\n".join([f" - {jn}: {r}" for jn,r in {jname:self._mj_model.jnt_range[jid] for jid,jname in self._jid2jname.items()}.items()])))
        print("Joint child bodies:\n"+("\n".join([f" - {jn}: {r}" for jn,r in {jname:self._mj_model.jnt_bodyid[jid] for jid,jname in self._jid2jname.items()}.items()])))
        
        print("Bodies parentid:\n"+("\n".join([f" - body_parentid[{lid}({self._lid2lname[lid]})]= {self._mj_model.body_parentid[lid]}" for lid in self._lid2lname.keys()])))
        print("Bodies jnt_num:\n"+("\n".join([f" - body_jntnum[{lid}({self._lid2lname[lid]})]= {self._mj_model.body_jntnum[lid]}" for lid in self._lid2lname.keys()])))
        # print(f"got cam resolutions {self._camera_sizes}")
        self._requested_qfrc_applied = jnp.copy(self._mjx_data.qfrc_applied)


    def detected_joints(self):
        return list(self._jname2jid.keys())
    
    def detected_links(self):
        return list(self._lname2lid.keys())
    
    def detected_cameras(self):
        return list(self._cname2cid.keys())

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str,str]]):
        super().set_monitored_joints(jointsToObserve)
        self._monitored_jids = jnp.array([self._jname2jid[jn] for jn in self._monitored_joints], device=self._jax_device)
        self._monitored_qpadr = self._mjx_model.jnt_qposadr[self._monitored_jids]
        self._monitored_qvadr = self._mjx_model.jnt_dofadr[self._monitored_jids]
        self._reset_joint_state_step_stats()


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
        self._reset_joint_state_step_stats()
        stepLength = self.run(self._step_length_sec)
        return stepLength

    def _apply_commands(self):
        self._apply_torque_cmds()

    def _apply_torque_cmds(self):
        self._mjx_data = self._mjx_data.replace(qfrc_applied=self._requested_qfrc_applied)

    @override
    def run(self, duration_sec : float):
        """Run the environment for the specified duration"""
        tf0 = time.monotonic()

        # self._sent_motor_torque_commands_by_bid_jid = {}

        stepping_wtime = 0
        simulating_wtime = 0
        control_wtime = 0
        vsteps_done = 0
        st0 = self._simTime
        while self._simTime-st0 < duration_sec:
            wtps = time.monotonic()
            self._apply_commands()

            wt1 = time.monotonic()
            control_wtime += wt1-wtps
            self._mjx_data = self._mjx_integrate_and_forward(self._mjx_model,self._mjx_data)
            simulating_wtime += time.monotonic()-wt1

            self._update_joint_state_step_stats()
            stepping_wtime += time.monotonic()-wtps
            self._sim_step_count_since_build += 1
            vsteps_done += 1
            # self._read_new_contacts()
            
            self._simTime += self._sim_step_dt
            if self._realtime_factor is not None and self._realtime_factor>0:
                sleep_time = self._sim_step_dt*(1/self._realtime_factor) - (time.monotonic()-self._prev_step_end_wtime)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._prev_step_end_wtime = time.monotonic()
            self._update_gui()
        # self._last_sent_torques_by_name = {self._bodyAndJointIdToJointName[bid_jid]:torque 
        #                                     for bid_jid,torque in self._sent_motor_torque_commands_by_bid_jid.items()}
        self._stats = { "wtime_stepping":    stepping_wtime,
                        "wtime_simulating":  simulating_wtime,
                        "wtime controlling": control_wtime,
                        "rt_factor_vec": self._vec_size*(self._simTime-st0)/stepping_wtime,
                        "rt_factor_single": (self._simTime-st0)/stepping_wtime,
                        "stime_step" : self._simTime-st0,
                        "stime" : self._simTime,
                        "fps_vec" :    self._vec_size*vsteps_done/stepping_wtime,
                        "fps_single" : vsteps_done/stepping_wtime}
        if self._log_freq > 0 and self._sim_step_count_since_build % self._log_freq == 0:
            ggLog.info( "\n".join([str(k)+' : '+str(v) for k,v in self._stats.items()]))
        self._sim_stepping_wtime_since_build += stepping_wtime
        self._run_wtime_since_build += time.monotonic()-tf0

        return self._simTime-st0
    
    @staticmethod
    # @jax.jit
    def _take_batch_element(batch, e):
        mjx_data = jax.tree_map(lambda l: l[e], batch)
        return mjx.get_data(mj_model, mjx_data)
    
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
        
        # mj_data_batch = mjx.get_data(self._mj_model, self._mjx_data)
        # print(f"mj_data_batch = {mj_data_batch}")
        times = th.as_tensor(self._simTime).repeat((nvecs,len(requestedCameras)))
        image_batches = [np.ones(shape=(nvecs,)+self._camera_sizes[cam]+(3,), dtype=np.uint8) for cam in requestedCameras]
        # print(f"images.shapes = {[i.shape for i in images]}")
        mj_datas : list[mujoco.MjData] = mjx.get_data(self._mj_model, self._mjx_data)
        for env_i,env in enumerate(selected_vecs):
            for i in range(len(requestedCameras)):
                cam = requestedCameras[i]
                # print(f"self._mj_model.cam_resolution[cid] = {self._mj_model.cam_resolution[self._cname2cid[cam]]}")
                renderer = self._renderers[self._camera_sizes[cam]]
                renderer.update_scene(mj_datas[env], self._cname2cid[cam]) #, scene_option=self._render_scene_option)
                image_batches[i][env_i] = renderer.render()
                # renderer.render(out=images[i][env])
        return [th.as_tensor(img_batch) for img_batch in image_batches], times


    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        if requestedJoints is None:
            jids = self._monitored_jids
        else:
            jids = jnp.array([self._jname2jid[jn] for jn in requestedJoints], device=self._jax_device)
        if len(jids) == 0:
            return th.empty(size=(self._vec_size,self._mjx_data.qpos.shape[1],0,3), dtype=th.float32)
        else:
           t = self._get_vec_joint_states_pve(self._mjx_model, self._mjx_data, jids)
        return jax2th(t, th_device=self._out_th_device)
    
    @staticmethod
    # @jax.jit
    def _get_vec_joint_states_pve(mjx_model, mjx_data, jids : jnp.ndarray):
        return MjxAdapter._get_vec_joint_states_raw_pve(mjx_model.jnt_qposadr[jids],
                                                        mjx_model.jnt_dofadr[jids],
                                                        mjx_data)
    
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pve(qpadr, qvadr, mjx_data):
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qfrc_smooth[:,qvadr]], # is this the right one? Should I just use my own qfrc_applied? qfrc_smooth? qfrc_inverse?
                            axis = 2)
    
    @staticmethod
    @jax.jit
    def _get_vec_joint_states_raw_pvea(qpadr, qvadr, mjx_data):
        return jnp.stack([  mjx_data.qpos[:,qpadr],
                            mjx_data.qvel[:,qvadr],
                            mjx_data.qfrc_smooth[:,qvadr], # is this the right one? Should I just use my own qfrc_applied? qfrc_smooth? qfrc_inverse?
                            mjx_data.qacc[:,qvadr]],
                            axis = 2)


    def _build_joint_state_step_stats(self):
        self._joint_stats_sample_count = 0
        self._monitored_joints_stats = self._init_stats(jnp.zeros(shape=(self._vec_size, 4, len(self._monitored_joints),4),
                                                                  dtype=jnp.float32,
                                                                  device=self._jax_device))

    @staticmethod
    @partial(jax.jit, donate_argnames=["stats_array"])
    def _init_stats(stats_array : jnp.ndarray):
        stats_array.at[:,0].set(float("-inf"))
        stats_array.at[:,1].set(float("+inf"))
        stats_array.at[:,2].set(float("nan"))
        stats_array.at[:,3].set(float("nan"))
        stats_array.at[:,4].set(0)
        stats_array.at[:,5].set(0)
        return stats_array

    @staticmethod
    @partial(jax.jit, donate_argnames=["stats_array"])
    def _update_joint_state_step_stats_arrs(current_jstate : jnp.ndarray,
                                            stats_array : jnp.ndarray,
                                            sample_count : jnp.ndarray):
        
        stats_array.at[:,4].set(jnp.add(    stats_array[:,4], current_jstate)) # sum of values
        stats_array.at[:,5].set(jnp.add(    stats_array[:,5], jnp.square(current_jstate))) # sum of squares

        stats_array.at[:,0].set(jnp.minimum(stats_array[:,0], current_jstate))
        stats_array.at[:,1].set(jnp.maximum(stats_array[:,1], current_jstate))
        stats_array.at[:,2].set(stats_array[:,4]/sample_count) # average values
        stats_array.at[:,3].set(jnp.sqrt(jnp.clip(stats_array[:,5]/sample_count-jnp.square(stats_array[:,2]),min=0))) # standard deviation
        return stats_array

    def _update_joint_state_step_stats(self):
        if self._joint_stats_sample_count == 0: # if we are at zero whatever is in the current state is invalid
            self._build_joint_state_step_stats()
        if len(self._monitored_joints) == 0:
            return
        self._joint_stats_sample_count += 1
        jstate_pvea = self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._mjx_data)
        self._monitored_joints_stats = self._update_joint_state_step_stats_arrs(jstate_pvea, self._monitored_joints_stats, self._joint_stats_sample_count)

    def _reset_joint_state_step_stats(self):
        # ggLog.info(f"resetting stats")
        self._joint_stats_sample_count = 0 # set to zero so the update rebuilds the stats
        self._update_joint_state_step_stats() # rebuild and populate with current state
        self._joint_stats_sample_count = 0 # so that at the next update these values get canceled (because the value we just wrote actually belong to the previous step)

    def get_joints_state_step_stats(self) -> th.Tensor:
        return jax2th(self._monitored_joints_stats[:,:4], self._out_th_device)


    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | None = None, use_com_frame : bool = False) -> th.Tensor:
        if requestedLinks is None:
            body_ids = self._monitored_lids
        else:
            body_ids = jnp.array([self._lname2lid[jn] for jn in requestedLinks], device=self._jax_device)
        if use_com_frame:
            t = jnp.concatenate([self._mjx_data.xipos[:,body_ids], # com position
                                 jax.scipy.spatial.transform.Rotation.from_matrix(self._mjx_data.ximat[:,body_ids]).as_quat(scalar_first=False), # com orientation
                                 self._mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]]], axis = -1) #com linear and angular velocity
        else:
            t = jnp.concatenate([self._mjx_data.xpos[:,body_ids], # frame position
                                 self._mjx_data.xquat[:,body_ids][:,:,self._wxyz2xyzw], # frame orientation
                                 self._mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]]], axis = -1) # frame linear and angular velocity
        return jax2th(t, th_device=self._out_th_device)

    @override
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        self.__lastResetTime = self.getEnvTimeFromStartup()

        self._mjx_data = copy.deepcopy(self._original_mjx_data)
        self._mjx_model = copy.deepcopy(self._original_mjx_model)
        self._mj_data = copy.deepcopy(self._original_mj_data)
        self._mj_model = copy.deepcopy(self._original_mj_model)
        self._reset_joint_state_step_stats()


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
        # ggLog.info(f"self._mjx_data.qpos[{vec_mask_jnp},{qpadr}].shape = {get_rows_cols(self._mjx_data.qpos, [vec_mask_jnp,qpadr]).shape}")
        # ggLog.info(f"js_pve[vec_mask_jnp,:,0].shape = {js_pve[vec_mask_jnp,:,0].shape}")
        qpos = set_rows_cols(self._mjx_data.qpos,           (vec_mask_jnp,qpadr), js_pve[vec_mask_jnp,:,0])
        qvel = set_rows_cols(self._mjx_data.qvel,           (vec_mask_jnp,qvadr), js_pve[vec_mask_jnp,:,1])
        qeff = set_rows_cols(self._mjx_data.qfrc_applied,   (vec_mask_jnp,qvadr), js_pve[vec_mask_jnp,:,2])
        self._mjx_data = self._mjx_data.replace(qpos=qpos, qvel=qvel, qfrc_applied=qeff)
        self._mjx_data = self._mjx_forward(self._mjx_model,self._mjx_data)    
        self._update_gui(force=True)
        # ggLog.info(f"setted_jstate Simtime [{self._simTime:.9f}] step [{self._sim_step_count_since_build}] monitored jstate:\n{self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._mjx_data)}")


    def _update_gui(self, force : bool = False):
        if self._show_gui and (time.monotonic() - self._last_gui_update_wtime > 1/self._gui_freq or force):
            mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._mjx_data))
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
        data_joint_pos = self._mjx_data.qpos
        data_joint_vel = self._mjx_data.qvel
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
                if not jnp.all(jnp.array_equal(link_states_pose_vel_jnp[:,i], jnp.broadcast_to(link_states_pose_vel_jnp[0,i], shape=link_states_pose_vel_jnp[:,i].shape),equal_nan=True)):
                    raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations.\n"
                                       f"{link_states_pose_vel_jnp[0,i]}\n"
                                       f"!=\n"
                                       f"{link_states_pose_vel_jnp[:,i]}")
                if jnp.any(vec_mask_jnp != vec_mask_jnp[0]):
                    raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations.")
                model_body_pos = model_body_pos.at[lid].set(link_states_pose_vel_jnp[0,i,:3])
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
        self._mjx_data = self._mjx_data.replace(qpos=data_joint_pos, qvel=data_joint_vel)
        # print(f"self._mjx_model.body_pos = {self._mjx_model.body_pos}")        
        # print(f"self._mjx_data.qpos = {self._mjx_data.qpos}")        
        # self._mjx_model = mjx.put_model(self._mj_model, device=self._jax_device)
        self._mjx_data = self._mjx_forward(self._mjx_model,self._mjx_data)
        self._update_gui(True)
        # ggLog.info(f"setted_lstate Simtime [{self._simTime:.9f}] step [{self._sim_step_count_since_build}] monitored jstate:\n{self._get_vec_joint_states_raw_pvea(self._monitored_qpadr, self._monitored_qvadr, self._mjx_data)}")
            

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
        self._set_effort_command(jids,qeff)
        # ggLog.info(f"self._requested_qfrc_applied = {self._requested_qfrc_applied}")

    def _set_effort_command(self, jids : jnp.ndarray, qefforts : jnp.ndarray, sims_mask : jnp.ndarray | None = None) -> None:
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.

        Parameters
        ----------
        joint_names : jnp.ndarray
            Array with the joint ids for each effort command, 1-dimensional
        efforts : th.Tensor
            Tensor of shape (vec_size, len(jids)) containing the effort for each joint in each environment.
        """
        qvadr = self._mj_model.jnt_dofadr[jids]
        if sims_mask is None:
            self._requested_qfrc_applied = self._requested_qfrc_applied.at[:,qvadr].set(qefforts[:,:])
        else:
            sims_indexes = jnp.nonzero(sims_mask)[0]
            self._requested_qfrc_applied = self._requested_qfrc_applied.at[sims_indexes[:,jnp.newaxis],qvadr].set(qefforts[:,:])