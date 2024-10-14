from __future__ import annotations
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from dataclasses import dataclass
from adarl.utils.utils import Pose, compile_xacro_string, pkgutil_get_path
from typing import Any
import jax
import mujoco
from mujoco import mjx
from pathlib import Path
import time
from typing_extensions import override
from typing import overload, Sequence
import torch as th
import jax.numpy as jnp
import jax.scipy.spatial.transform
from adarl.utils.utils import build_pose
from dlpack import asdlpack
import adarl.utils.dbg.ggLog as ggLog
import torch.utils.dlpack as thdlpack

   

class MjxAdapter(BaseVecSimulationAdapter):
    def __init__(self, vec_size : int,
                        enable_rendering : bool,
                        jax_device : jax.Device,
                        sim_step_dt : float = 2/1024,
                        step_length_sec : float = 10/1024,
                        realtime_factor : float | None = None):
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


        self.set_monitored_joints([])
        self.set_monitored_links([])

    @override
    def build_scenario(self, models : list[ModelSpawnDef]):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
        
        specs = []
        for model in models:
            mjSpec = mujoco.MjSpec()
            if model.format == "urdf.xacro":
                urdf_string = compile_xacro_string( model_definition_string=model.definition_string,
                                                                model_kwargs=model.kwargs)
            elif model.format == "urdf":
                urdf_string = model.definition_string
            else:
                raise RuntimeError(f"Unsupported model format '{model.format}' for model '{model.name}")
            mjSpec.from_string(urdf_string)
            specs.append(mjSpec)
        big_spec = mujoco.MjSpec()
        site = big_spec.worldbody.add_site()
        for spec in specs:
            site.attach(spec.worldbody, prefix="", suffix="")

        self._mj_model = big_spec.compile()

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
        ggLog.info(f"mjx_data.qpos.shape = {self._mjx_data.qpos.shape}")

        self._mjx_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        _ = self._mjx_step(self._mjx_model, self._mjx_data) # trigger jit compile
        # dm_control calls first step2, then step1, to be more efficient and avoiding forward kinematics recomputation
        # see: https://github.com/google-deepmind/mujoco/issues/430#issuecomment-1208489785
        


        if self._enable_rendering:
            self._renderer = mujoco.Renderer(self._mj_model)
            self._render_scene_option = mujoco.MjvOption()
            # # enable joint visualization option:
            # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self._jid2jname : dict[int, tuple[str,str]] = {jid:(self._model_name, mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)) 
                           for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname : dict[int, tuple[str,str]] = {lid:(self._model_name, mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, lid))
                           for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}
        self._cid2cname : dict[int, str] = {jid:mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, jid)
                           for jid in range(self._mj_model.ncam)}
        self._cname2cid = {cn:cid for cid,cn in self._cid2cname.items()}
        self._camera_sizes = {self._cid2cname[cid]:self._mj_model.cam_resolution[cid] for cid in self._cid2cname}


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
    def destroy_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment when closing."""
        pass

    @override
    def step(self) -> float:
        """Run a simulation step.

        Returns
        -------
        float
            Duration of the step in simulation time (in seconds)"""

        stepLength = self.run(self._step_length_sec)
        return stepLength

    @override
    def run(self, duration_sec : float):
        """Run the environment for the specified duration"""
        tf0 = time.monotonic()

        # self._sent_motor_torque_commands_by_bid_jid = {}

        stepping_wtime = 0
        t0 = self._simTime
        while self._simTime-t0 < duration_sec:
            wtps = time.monotonic()
            self._mjx_data = self._mjx_step(self._mjx_model,self._mjx_data)
            stepping_wtime += time.monotonic()-wtps
            self._sim_step_count_since_build += 1
            # self._read_new_contacts()
            # self._update_joint_state_step_stats()
            self._simTime += self._sim_step_dt
            if self._realtime_factor is not None and self._realtime_factor>0:
                sleep_time = self._sim_step_dt - (time.monotonic()-self._prev_step_end_wall_time)
                if sleep_time > 0:
                    time.sleep(sleep_time*(1/self._realtime_factor))
            self._prev_step_end_wall_time = time.monotonic()
        # self._last_sent_torques_by_name = {self._bodyAndJointIdToJointName[bid_jid]:torque 
        #                                     for bid_jid,torque in self._sent_motor_torque_commands_by_bid_jid.items()}
        self._sim_stepping_wtime_since_build += stepping_wtime
        self._run_wtime_since_build += time.monotonic()-tf0

        return self._simTime-t0
    

    @override
    def getRenderings(self, requestedCameras : list[str]) -> tuple[list[th.Tensor], th.Tensor]:
        if self._renderer is None:
            raise RuntimeError(f"Called getRenderings, but rendering is not initialized. did you set enable_rendering?")
        mj_data_batch = mjx.get_data(self._mj_model, self._mjx_data)
        # print(f"mj_data_batch = {mj_data_batch}")
        times = th.as_tensor(self._simTime).repeat((self._vec_size,len(requestedCameras)))
        images = [th.empty(size=(self._vec_size,)+self._camera_sizes[cam]) for cam in requestedCameras]
        for env in range(self._vec_size):
            mj_data = jax.tree_map(lambda l: l[env], mj_data_batch)
            for i in range(len(requestedCameras)):
                cam = requestedCameras[i]
                self._renderer.update_scene(mj_data, self._cname2cid[cam], scene_option=self._render_scene_option)
                images[i][env] = th.as_tensor(self._renderer.render())
        return images, times


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
            t = jnp.stack([self._mjx_data.qpos[:,jids],self._mjx_data.qvel[:,jids],self._mjx_data.qfrc_actuator[:,jids]], axis = 2)
        return thdlpack.from_dlpack(asdlpack(t))


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
            xiquat = jax.scipy.spatial.transform.Rotation.from_matrix(self._mjx_data.ximat[:,body_ids]).as_quat(scalar_first=False)
            t = jnp.concatenate([self._mjx_data.xipos[:,body_ids], # com position
                                 xiquat, # com orientation
                                 self._mjx_data.cvel[:,body_ids,[3,4,5,0,1,2]]], axis = -1) #com linear and angular velocity
        else:
            # print(f"xpos.shape = {self._mjx_data.xpos.shape}")
            # print(f"xipos.shape = {self._mjx_data.xipos.shape}")
            # print(f"cvel.shape = {self._mjx_data.cvel.shape}")
            # print(f"pos shape = {self._mjx_data.xpos[:,body_ids].shape}")
            # print(f"ori shape = {self._mjx_data.xquat[:,body_ids][:,:,self._wxyz2xyzw].shape}")
            # print(f"vel shape = {self._mjx_data.cvel[:,body_ids][:,:,[3,4,5,0,1,2]].shape}")
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
        qpos = self._mjx_data.qpos.at[:,jids].set(js_pve[:,:,0])
        qvel = self._mjx_data.qvel.at[:,jids].set(js_pve[:,:,1])
        qeff = self._mjx_data.qfrc_actuator.at[:,jids].set(js_pve[:,:,2])
        self._mjx_data = self._mjx_data.replace(qpos=qpos, qvel=qvel, qfrc_actuator=qeff)
    
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    @override
    def delete_model(self, model_name : str):
        """Remove a model from all of the simulations
        Parameters
        ----------
        model_name : str
            Name of the model to be removed
        """
        raise NotImplementedError()