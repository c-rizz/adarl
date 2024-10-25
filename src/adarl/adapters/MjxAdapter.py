from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from dataclasses import dataclass
from adarl.utils.utils import Pose, compile_xacro_string, pkgutil_get_path, exc_to_str
from typing import Any
import jax
import mujoco
from mujoco import mjx
import mujoco.viewer
from pathlib import Path
import time
from typing_extensions import override
from typing import overload, Sequence, Mapping
import torch as th
import jax.numpy as jnp
import jax.scipy.spatial.transform
from adarl.utils.utils import build_pose
from dlpack import asdlpack
import adarl.utils.dbg.ggLog as ggLog
import torch.utils.dlpack as thdlpack
import numpy as np
import copy
from typing import Iterable

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


class MjxAdapter(BaseVecSimulationAdapter, BaseVecJointEffortAdapter):
    def __init__(self, vec_size : int,
                        enable_rendering : bool,
                        jax_device : jax.Device,
                        sim_step_dt : float = 2/1024,
                        step_length_sec : float = 10/1024,
                        realtime_factor : float | None = None,
                        show_gui : bool = False,
                        gui_frequency : float = 15,
                        gui_env_index : int = 0):
        super().__init__(vec_size=vec_size)
        self._enable_rendering = enable_rendering
        self._jax_device = jax_device
        self._sim_step_dt = sim_step_dt
        self._step_length_sec = step_length_sec
        self._simTime = 0.0
        self._sim_step_count_since_build = 0
        self._sim_stepping_wtime_since_build = 0
        self._run_wtime_since_build = 0

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

        self.set_monitored_joints([])
        self.set_monitored_links([])

    @staticmethod
    def _mj_name_to_pair(mjname : str):
        if mjname == "world":
            return mjname,mjname
        sep = mjname.find("_")
        return mjname[:sep],mjname[sep+1:]

    @override
    def build_scenario(self, models : list[ModelSpawnDef]):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
        
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
            # add all th bodies that are direct childern of worldbody
            body = spec.worldbody.first_body()
            if spec.degree:
                raise NotImplementedError(f"model {mname} uses degrees instead of radians.")
            while body is not None:
                frame.attach_body(body, mname+"_", "")
                body = spec.worldbody.next_body(body)

        self._mj_model = big_speck.compile()
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
        if self._enable_rendering:
            self._renderer = mujoco.Renderer(self._mj_model)
        else:
            self._renderer = None
        mujoco.mj_resetData(self._mj_model, self._mj_data)

        self._mjx_model = mjx.put_model(self._mj_model, device = self._jax_device)
        self._mjx_data = mjx.put_data(self._mj_model, self._mj_data, device = self._jax_device)
        ggLog.info(f"mjx_data.qpos.shape = {self._mjx_data.qpos.shape}")
        self._mjx_data = jax.vmap(lambda: self._mjx_data, axis_size=self._vec_size)()
        # self._mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(jnp.arange(self._vec_size), self._mjx_data)
        ggLog.info(f"mjx_data.qpos.shape = {self._mjx_data.qpos.shape}")

        ggLog.info(f"Compiling mjx.step....")
        self._mjx_step2_step1 = jax.jit(jax.vmap(mjx_integrate_and_forward, in_axes=(None, 0))) #, donate_argnames=["d"]) donating args make it crash
        _ = self._mjx_step2_step1(self._mjx_model, copy.deepcopy(self._mjx_data)) # trigger jit compile
        ggLog.info(f"Compiling mjx.forward....")
        self._mjx_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self._mjx_data = self._mjx_forward(self._mjx_model, self._mjx_data) # compute initial mjData
        ggLog.info(f"Compiled MJX.")
        


        if self._enable_rendering:
            self._renderer = mujoco.Renderer(self._mj_model)
            self._render_scene_option = mujoco.MjvOption()
            # # enable joint visualization option:
            # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

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
        self._camera_sizes = {self._cid2cname[cid]:(self._mj_model.cam_resolution[cid][1],self._mj_model.cam_resolution[cid][0]) for cid in self._cid2cname}
        if self._enable_rendering:
            self._renderers : dict[tuple[int,int],mujoco.Renderer]= {resolution:mujoco.Renderer(self._mj_model,height=resolution[0],width=resolution[1])
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
        t0 = self._simTime
        while self._simTime-t0 < duration_sec:
            wtps = time.monotonic()
            self._apply_commands()
            # ggLog.info(f"qfrc_applied0 = {self._mjx_data.qfrc_applied}")
            # ggLog.info(f"qfrc_applied1 = {self._mjx_data.qfrc_applied}")
            # ggLog.info(f"nu = {self._mj_model.nu}")
            self._mjx_data = self._mjx_step2_step1(self._mjx_model,self._mjx_data)
            # ggLog.info(f"qfrc_applied2 = {self._mjx_data.qfrc_applied}")
            stepping_wtime += time.monotonic()-wtps
            self._sim_step_count_since_build += 1
            # self._read_new_contacts()
            # self._update_joint_state_step_stats()
            self._simTime += self._sim_step_dt
            if self._realtime_factor is not None and self._realtime_factor>0:
                sleep_time = self._sim_step_dt*(1/self._realtime_factor) - (time.monotonic()-self._prev_step_end_wtime)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._prev_step_end_wtime = time.monotonic()
            if self._show_gui and time.monotonic() - self._last_gui_update_wtime > 1/self._gui_freq:
                # inplace_deepcopy(self._viewer_mj_data, mjx.get_data(self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._mjx_data)),
                #                  exclude=mj_data_copy_exclude)
                mjx.get_data_into(self._viewer_mj_data,self._mj_model, jax.tree_map(lambda l: l[self._gui_env_index], self._mjx_data))
                self._last_gui_update_wtime = time.monotonic()
                self._viewer.sync()
        # self._last_sent_torques_by_name = {self._bodyAndJointIdToJointName[bid_jid]:torque 
        #                                     for bid_jid,torque in self._sent_motor_torque_commands_by_bid_jid.items()}
        self._sim_stepping_wtime_since_build += stepping_wtime
        self._run_wtime_since_build += time.monotonic()-tf0

        return self._simTime-t0
    
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
    def getRenderings(self, requestedCameras : list[str]) -> tuple[list[th.Tensor], th.Tensor]:
        if self._renderer is None:
            raise RuntimeError(f"Called getRenderings, but rendering is not initialized. did you set enable_rendering?")
        # mj_data_batch = mjx.get_data(self._mj_model, self._mjx_data)
        # print(f"mj_data_batch = {mj_data_batch}")
        times = th.as_tensor(self._simTime).repeat((self._vec_size,len(requestedCameras)))
        image_batches = [np.ones(shape=(self._vec_size,)+self._camera_sizes[cam]+(3,), dtype=np.uint8) for cam in requestedCameras]
        # print(f"images.shapes = {[i.shape for i in images]}")
        mj_datas : list[mujoco.MjData] = mjx.get_data(self._mj_model, self._mjx_data)
        for env in range(self._vec_size):
            for i in range(len(requestedCameras)):
                cam = requestedCameras[i]
                # print(f"self._mj_model.cam_resolution[cid] = {self._mj_model.cam_resolution[self._cname2cid[cam]]}")
                renderer = self._renderers[image_batches[i][env].shape[:2]]
                renderer.update_scene(mj_datas[env], self._cname2cid[cam]) #, scene_option=self._render_scene_option)
                image_batches[i][env] = renderer.render()
                # renderer.render(out=images[i][env])
        return [th.as_tensor(img_batch) for img_batch in image_batches], times


    @override
    @overload
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]]) -> th.Tensor:
        ...

    @override
    @overload
    def getJointsState(self) -> th.Tensor:
        ...

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
        return thdlpack.from_dlpack(asdlpack(t))
    
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

    def get_joints_state_step_stats(self) -> th.Tensor:
        """Returns joint state statistics over the last step for the monitored joints. The value of these statistics after a call to run()
        is currently undefined.

        Returns
        -------
        th.Tensor
            Torch tensor of size (4,len(monitored_joints),3) containing min,max,average,std of the position,velocity
             and effort of each monitored joint. The joints are in the order use din set_monitored_joints.
        """
        raise NotImplementedError()

    @override
    @overload
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]], use_com_frame : bool = False) -> th.Tensor:
        ...

    @override
    @overload
    def getLinksState(self, requestedLinks : None = None, use_com_frame : bool = False) -> th.Tensor:
        ...

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
        return thdlpack.from_dlpack(asdlpack(t))

    @override
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        self.__lastResetTime = self.getEnvTimeFromStartup()


    @override
    def getEnvTimeFromStartup(self) -> float:
        """Get the current time within the simulation."""
        return self._simTime


    @override    
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor):
        if self._check_sizes and joint_states_pve.size() != (self._vec_size,len(joint_names),3):
            raise RuntimeError(f"joint_states_pve should have size {(self._vec_size,len(joint_names),3)}, but it's {joint_states_pve.size()}")
        jids = jnp.array([self._jname2jid[jn] for jn in joint_names])
        js_pve = jnp.from_dlpack(thdlpack.to_dlpack(joint_states_pve))
        jtypes = self._mj_model.jnt_type[jids]
        if not jnp.all(jnp.logical_or(jtypes == mujoco.mjtJoint.mjJNT_HINGE, jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
            raise RuntimeError(f"Cannot control set state for multi-dimensional joint, types = {list(zip(joint_names,jtypes))}")
        qpadr = self._mj_model.jnt_qposadr[jids]
        qvadr = self._mj_model.jnt_dofadr[jids]
        qpos = self._mjx_data.qpos.at[:,qpadr].set(js_pve[:,:,0])
        qvel = self._mjx_data.qvel.at[:,qvadr].set(js_pve[:,:,1])
        qeff = self._mjx_data.qfrc_applied.at[:,qvadr].set(js_pve[:,:,2])
        self._mjx_data = self._mjx_data.replace(qpos=qpos, qvel=qvel, qfrc_applied=qeff)
    
    @override
    def setLinksStateDirect(self, link_names : list[tuple[str,str]], link_states_pose_vel : th.Tensor):
        """Set the state for a set of links


        Parameters
        ----------
        link_names : list[tuple[str,str]]
            The names of the links to set the state for
        link_states_pose_vel : th.Tensor
            A tensor of shape (vec_size, len(link_names), 13), containing, for each joint, position_xyz, orientation_xyzw, linear_velocity_xyz, angular_velocity_xyz
        """
        # js_pve = jnp.from_dlpack(thdlpack.to_dlpack(joint_states_pve))
        link_states_pose_vel_jnp = jnp.from_dlpack(thdlpack.to_dlpack(link_states_pose_vel)).to_device(self._jax_device)
        model_body_pos = self._mjx_model.body_pos
        model_body_quat = self._mjx_model.body_quat
        data_joint_pos = self._mjx_data.qpos
        data_joint_vel = self._mjx_data.qvel
        for i, link_name in enumerate(link_names):
            ggLog.info(f"setting link state for {link_name}")
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
                ggLog.info(f"changing 'fixed joint'")
                if jnp.any(link_states_pose_vel_jnp[:,i] != link_states_pose_vel_jnp[0,i]):
                    raise RuntimeError(f"Fixed joints cannot be set to different positions across the vectorized simulations."
                                       f"This because MJX does not vectorize the MjModel, all vec simulations use the same model,"
                                       f" and fixed joints are represented as fixed transforms in the model.")
                model_body_pos = model_body_pos.at[lid].set(link_states_pose_vel_jnp[0,i,:3])
                model_body_quat = model_body_quat.at[lid].set(link_states_pose_vel_jnp[0,i,[6,3,4,5]])
                # print(f"self._mjx_model.body_pos = {self._mjx_model.body_pos}")
            elif parent_joints_num == 1 and parent_body_id==0:
                jid = self._mj_model.body_jntadr[lid]
                jtype = self._mj_model.jnt_type[jid]
                if jtype == mujoco.mjtJoint.mjJNT_FREE:
                    ggLog.info(f"writing at qpos[{self._mj_model.jnt_qposadr[jid]}:{self._mj_model.jnt_qposadr[jid]+7}]")
                    data_joint_pos = data_joint_pos.at[:,self._mj_model.jnt_qposadr[jid]:self._mj_model.jnt_qposadr[jid]+7].set(link_states_pose_vel_jnp[:,i,[0,1,2,6,3,4,5]])
                    data_joint_vel = data_joint_vel.at[:,self._mj_model.jnt_dofadr[jid]:self._mj_model.jnt_dofadr[jid]+6].set(link_states_pose_vel_jnp[:,i,7:13])
                    # raise NotImplementedError()
                else:
                    raise NotImplementedError(f"Cannot set link state for link {link_name} with parent joint of type {jtype} (see mjtJoint enum)")
            else:
                raise NotImplementedError(f"Cannot set link state for link {link_name} with {self._mj_model.body_jntnum} parent joints and parent body {parent_body_id}")
        self._mjx_model = self._mjx_model.replace(body_pos=model_body_pos, body_quat = model_body_quat)
        self._mjx_data = self._mjx_data.replace(qpos=data_joint_pos, qvel=data_joint_vel)
        # print(f"self._mjx_model.body_pos = {self._mjx_model.body_pos}")        
        print(f"self._mjx_data.qpos = {self._mjx_data.qpos}")        
        # self._mjx_model = mjx.put_model(self._mj_model, device=self._jax_device)
        self._mjx_forward(self._mjx_model,self._mjx_data)
            

    @override
    def setupLight(self):
        raise NotImplementedError()

    @override
    def spawn_model(self,   model_name : str,
                            model_definition_string : str | None = None,
                            model_format : str | None = None,
                            model_file : str  | None = None,
                            pose : Pose = build_pose(0,0,0,0,0,0,1),
                            model_kwargs : dict[Any,Any] = {}) -> str:
        """Spawn a model in the simulation in all of the simulations.

        Parameters
        ----------
        model_definition_string : str
            Model definition specified in as a string. e.g. an SDF definition
        model_format : str
            Format of the model definition. E.g. 'sdf' or 'urdf'
        model_file : _type_
            File to load the model definition from
        model_name : str
            Name to give to the spawned model
        pose : Pose
            Pose to spawn the model at
        model_kwargs : Dict[Any,Any]
            Arguments to use in interpreting the model definition

        Returns
        -------
        str
            The model name
        """
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
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            List of the joint names
        efforts : th.Tensor
            Tensor of shape (vec_size, len(joint_names)) containing the effort for each joint in each environment.
        """
        jids = jnp.array([self._jname2jid[jn] for jn in joint_names])
        qeff = jnp.from_dlpack(thdlpack.to_dlpack(efforts))
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