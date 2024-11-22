from __future__ import annotations

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from typing_extensions import override
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.PyBulletJointImpedanceAdapter import PyBulletJointImpedanceAdapter
from typing import Sequence, Any
from adarl.utils.utils import Pose, build_pose, JointState, LinkState
import torch as th

class VecPyBulletJointImpedanceAdapter(BaseVecSimulationAdapter, BaseVecJointImpedanceAdapter):
    
    def __init__(self,  vec_size : int,
                        th_device : th.device,
                        stepLength_sec : float = 0.004166666666,
                        restore_on_reset = True,
                        debug_gui : bool = False,
                        real_time_factor : float | None = None,
                        global_max_torque_position_control : float = 100,
                        joints_max_torque_position_control : dict[tuple[str,str],float] = {},
                        global_max_velocity_position_control : float = 1,
                        joints_max_velocity_position_control : dict[tuple[str,str],float] = {},
                        global_max_acceleration_position_control : float = 10,
                        joints_max_acceleration_position_control : dict[tuple[str,str],float] = {},
                        simulation_step = 1/960,
                        enable_rendering = True):
        self._sub_adapter = PyBulletJointImpedanceAdapter(  stepLength_sec = stepLength_sec,
                                                            restore_on_reset  = restore_on_reset,
                                                            debug_gui  = debug_gui,
                                                            real_time_factor  = real_time_factor,
                                                            global_max_torque_position_control  = global_max_torque_position_control,
                                                            joints_max_torque_position_control  = joints_max_torque_position_control,
                                                            global_max_velocity_position_control  = global_max_velocity_position_control,
                                                            joints_max_velocity_position_control  = joints_max_velocity_position_control,
                                                            global_max_acceleration_position_control  = global_max_acceleration_position_control,
                                                            joints_max_acceleration_position_control  = joints_max_acceleration_position_control,
                                                            simulation_step  = simulation_step,
                                                            enable_rendering  = enable_rendering)
        self._vec_size = vec_size
        self._th_device = th_device
        if vec_size!=1: 
            raise NotImplementedError()


    @override
    def getRenderings(self, requestedCameras : list[str], vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if vec_mask.item():
            rdict = self._sub_adapter.getRenderings(requestedCameras=requestedCameras)
            imgs =  [th.as_tensor(rdict[n][0], device=self._th_device).unsqueeze(0) for n in requestedCameras]
            times = th.stack([th.as_tensor(rdict[n][1], device=self._th_device) for n in requestedCameras]).expand(self._vec_size, len(requestedCameras))
        else:
            imgs  = [th.empty((0,3,9,16), dtype = th.uint8, device=self._th_device)]
            times = th.empty((0,1), dtype = th.float32, device=self._th_device)
        return imgs, times
        
    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        if requestedJoints is None:
            requestedJoints = self._monitored_joints
        jstate = self._sub_adapter.getJointsState(requestedJoints)
        return th.stack([th.as_tensor([ jstate[k].position.item(),
                                        jstate[k].rate.item(),
                                        jstate[k].effort.item()]) for k in requestedJoints]).unsqueeze(0)

    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        return self._sub_adapter.get_joints_state_step_stats().unsqueeze(0)
        
    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | None, use_com_frame : bool = False) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        ls = self._sub_adapter.getLinksState(requestedLinks, use_com_frame=use_com_frame)
        r = th.stack([th.cat([ ls[k].pose.position,
                                ls[k].pose.orientation_xyzw,
                                ls[k].pos_velocity_xyz,
                                ls[k].ang_velocity_xyz])
                        for k in requestedLinks]).unsqueeze(0)
        return r


    @override
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor):
        if vec_mask.item():
            self._sub_adapter.setJointsStateDirect({n:JointState(position = joint_states_pve[0,i,0],
                                                                rate =     joint_states_pve[0,i,1],
                                                                effort =   joint_states_pve[0,i,2]) 
                                                    for i,n in enumerate(joint_names)})
    
    @override
    def setLinksStateDirect(self, link_names : list[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor):
        if vec_mask.item():
            self._sub_adapter.setLinksStateDirect({n:LinkState(position_xyz         = link_states_pose_vel[0,i, 0:3],
                                                            orientation_xyzw     = link_states_pose_vel[0,i, 3:7],
                                                            pos_com_velocity_xyz = link_states_pose_vel[0,i, 7:10],
                                                            ang_velocity_xyz     = link_states_pose_vel[0,i,10:13])
                                                for i,n in  enumerate(link_names)})

    @override
    def setupLight(self,    lightDirection,
                            lightColor,
                            lightDistance,
                            enable_shadows,
                            lightAmbientCoeff,
                            lightDiffuseCoeff,
                            lightSpecularCoeff):
        self._sub_adapter.setupLight(lightDirection = lightDirection,
                                     lightColor = lightColor,
                                     lightDistance = lightDistance,
                                     enable_shadows = enable_shadows,
                                     lightAmbientCoeff = lightAmbientCoeff,
                                     lightDiffuseCoeff = lightDiffuseCoeff,
                                     lightSpecularCoeff = lightSpecularCoeff)

    @override
    def spawn_model(self,   model_name : str,
                            model_definition_string : str | None = None,
                            model_format : str | None = None,
                            model_file : str | None = None,
                            pose : Pose = build_pose(0,0,0,0,0,0,1),
                            model_kwargs : dict[Any,Any] = {}) -> str:
        return self._sub_adapter.spawn_model(   model_name = model_name,
                                                model_definition_string = model_definition_string,
                                                model_format = model_format,
                                                model_file = model_file,
                                                pose = pose,
                                                model_kwargs = model_kwargs)

    @override
    def delete_model(self, model_name : str):
        """Remove a model from all of the simulations
        Parameters
        ----------
        model_name : str
            Name of the model to be removed
        """
        self._sub_adapter.delete_model(model_name)
    
    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None,
                                        joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        if vec_mask is None or vec_mask.item():
            if isinstance(delay_sec,th.Tensor):
                delay_sec = delay_sec.item()
            if joint_names is None:
                self._sub_adapter.setJointsImpedanceCommand(joint_impedances_pvesd[0],delay_sec)
            else:
                self._sub_adapter.setJointsImpedanceCommand({n:tuple(joint_impedances_pvesd[0,i].tolist()) for i,n in enumerate(joint_names)},
                                                            delay_sec=delay_sec)

    
    @override
    def reset_joint_impedances_commands(self):
        self._sub_adapter.clear_commands()

    @override
    def set_current_joint_impedance_command(self,   joint_impedances_pvesd : th.Tensor,
                                                    vec_mask : th.Tensor | None = None,
                                                    joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        if vec_mask is None or vec_mask.item():
            if joint_names is None:
                self._sub_adapter.apply_joint_impedances(joint_impedances_pvesd[0])
            else:
                self._sub_adapter.apply_joint_impedances({n:tuple(joint_impedances_pvesd[0,i].tolist()) for i,n in enumerate(joint_names)})
    
    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[tuple[str,str]]):
        self._sub_adapter.set_impedance_controlled_joints(joint_names)
    

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str, str]]):
        return self._sub_adapter.set_monitored_joints(jointsToObserve)
    
    @override
    def set_monitored_links(self, linksToObserve: Sequence[tuple[str, str]]):
        return self._sub_adapter.set_monitored_links(linksToObserve)
    
    @override
    def set_monitored_cameras(self, camera_names: Sequence[tuple[str, str]]):
        return self._sub_adapter.set_monitored_joints(camera_names)
    
    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str,str]]:
        return self.get_impedance_controlled_joints()
    
    @override
    def build_scenario(self, models: Sequence[ModelSpawnDef] = [], **kwargs):
        return self._sub_adapter.build_scenario(models, **kwargs)
    
    @override
    def destroy_scenario(self, **kwargs):
        return self._sub_adapter.destroy_scenario(**kwargs)
    
    @override
    def run(self, duration_sec : float):
        self._sub_adapter.run(duration_sec)

    @override
    def step(self) -> float:
        return self._sub_adapter.step()
    
    @override
    def resetWorld(self):
        return self._sub_adapter.resetWorld()
    
    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._sub_adapter.getEnvTimeFromStartup()
    
    @override
    def getEnvTimeFromReset(self) -> float:
        return self._sub_adapter.getEnvTimeFromReset()
    
    @override
    def get_last_applied_command(self) -> th.Tensor:
        return self._sub_adapter.get_last_applied_command().unsqueeze(0)