from __future__ import annotations
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
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
from dlpack import asdlpack


@dataclass
class ModelSpawnDef:
    model_name : str
    model_definition_string : str | None
    model_format : str | None
    model_file : str | None
    pose : Pose | None
    model_kwargs : dict[Any,Any]

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

    @override
    def build_scenario(self, models : list[ModelSpawnDef]):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
        if len(models) > 1:
            raise NotImplementedError(f"Only one model is supported for now")

        model_definition_string = compile_xacro_string(  model_definition_string=Path(pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
                                                                        model_kwargs={})
        # Make model, data, and renderer
        self._mj_model = mujoco.MjModel.from_xml_string(model_definition_string)
        self._mj_data = mujoco.MjData(self._mj_model)
        if self._enable_rendering:
            self._renderer = mujoco.Renderer(self._mj_model)
        else:
            self._renderer = None
        mujoco.mj_resetData(mj_model, mj_data)

        self._mjx_model = mjx.put_model(self._mj_model, device = self._jax_device)
        self._mjx_data = mjx.put_data(self._mj_model, self._mj_data, device = self._jax_device)

        self._mjx_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        _ = self._mjx_step(self._mjx_model, self._data_batch) # trigger jit compile

        # # enable joint visualization option:
        # scene_option = mujoco.MjvOption()
        # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self._jid2jname = {jid:mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname = {lid:mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, lid) for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str,str]]):
        super().set_monitored_joints(jointsToObserve)
        self._monitored_jids = jnp.array([self._jname2jid[jn] for jn in self._monitored_joints], device=self._jax_device)

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
            self._mjx_step(self._mjx_model,self._mjx_data)
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
    def getRenderings(self, requestedCameras : list[str]) -> tuple[th.Tensor, th.Tensor]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        List[sensor_msgs.msg.Image]
            List contyaining the images for the cameras specified in requestedCameras, in the same order

        """
        raise NotImplementedError()


    @override
    @overload
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]]) -> th.Tensor:
        ...

    @override
    @overload
    def getJointsState(self) -> th.Tensor:
        """Get the state of the monitored joints.

        Returns
        -------
        th.Tensor
            Tensor of shape (joints_num, 3) with position,velocity,effort for each joint in set_monitored_joints()

        """
        ...

    @override
    def getJointsState(self, requestedJoints : Sequence[JointName] | None = None) -> th.Tensor:
        if requestedJoints is None:
            jids = self._monitored_jids
        else:
            jids = jnp.array([self._jname2jid[jn] for jn in requestedJoints], device=self._jax_device)
        t = jnp.stack([self._mjx_data.qpos[:,jids],self._mjx_data.qvel[:,jids],self._mjx_data.qfrc_actuator[:,jids]], axis = 2)
        return th.from_dlpack(asdlpack(t))


    def get_joints_state_step_stats(self) -> th.Tensor:
        """Returns joint state statistics over the last step for the monitored joints. The value of these statistics after a call to run()
        is currently undefined.

        Returns
        -------
        th.Tensor
            Torch tensor of size (4,len(monitored_joints),3) containing min,max,average,std of the position,velocity
             and effort of each monitored joint. The joints are in the order use din set_monitored_joints.
        """
        ...

    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName]) -> Dict[LinkName,LinkState]:
        """Get the state of the requested links.

        Parameters
        ----------
        linkNames : List[str]
            Names of the link to get the state of

        Returns
        -------
        Dict[str,LinkState]
            Dictionary, indexed by link name containing the state of each link

        """
        ...
    @overload
    def getLinksState(self, requestedLinks : None) -> th.Tensor:
        """Get the state of the monitored links.

        Returns
        -------
        th.Tensor
            Tensor containing the link state for each monitored link

        """
        ...
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName] | None) -> Dict[LinkName,LinkState] | th.Tensor:
        raise NotImplementedError()

    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        self.__lastResetTime = self.getEnvTimeFromStartup()


    def getEnvTimeFromStartup(self) -> float:
        """Get the current time within the simulation."""
        raise NotImplementedError()

    
    @abstractmethod
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor):        
        """Set the state for a set of joints


        Parameters
        ----------
        joint_names : list[tuple[str,str]]
            The names of the joints to set the state for
        joint_states_pve : th.Tensor
            A tensor of shape (vec_size, len(joint_names), 3) containins, position,velocity and effort for each joint
            
        """
        raise NotImplementedError()
    
    @abstractmethod
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

    @abstractmethod
    def setupLight(self):
        raise NotImplementedError()

    @abstractmethod
    def spawn_model(self,   model_name : str,
                            model_definition_string : Optional[str] = None,
                            model_format : Optional[str] = None,
                            model_file : Optional[str] = None,
                            pose : Pose = build_pose(0,0,0,0,0,0,1),
                            model_kwargs : Dict[Any,Any] = {}) -> str:
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

    @abstractmethod
    def delete_model(self, model_name : str):
        """Remove a model from all of the simulations
        Parameters
        ----------
        model_name : str
            Name of the model to be removed
        """
        raise NotImplementedError()