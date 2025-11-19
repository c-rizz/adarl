from __future__ import annotations

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from typing_extensions import override
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecJointPositionAdapter import BaseVecJointPositionAdapter
from adarl.adapters.ZmqXbotAdapter import ZmqXbotAdapter
from typing import Sequence, Any, Optional
import adarl.utils.dbg.ggLog as ggLog
import torch as th

class VecZmqXbotAdapter(BaseVecJointImpedanceAdapter, BaseVecJointPositionAdapter):
    
    def __init__(self,  vec_size : int,
                        th_device : th.device,
                        adapter : ZmqXbotAdapter):
        super().__init__(vec_size=vec_size,
                         output_th_device=th_device)
        if not isinstance(adapter, ZmqXbotAdapter):
            raise RuntimeError(f"adapter must be a ZmqXbotAdapter")
        self._sub_adapter = adapter
        if vec_size!=1: 
            raise NotImplementedError()


    @override
    def getRenderings(self, requestedCameras : list[str], vec_mask : th.Tensor | None = None) -> tuple[list[th.Tensor], th.Tensor]:
        raise NotImplementedError()
        if vec_mask is None or vec_mask.item():
            rdict = self._sub_adapter.getRenderings(requestedCameras=requestedCameras)
            imgs =  [th.as_tensor(rdict[n][0], device=self._out_th_device).unsqueeze(0) for n in requestedCameras]
            times = th.stack([th.as_tensor(rdict[n][1], device=self._out_th_device) for n in requestedCameras]).expand(self._vec_size, len(requestedCameras))
        else:
            imgs  = [th.empty((0,3,9,16), dtype = th.uint8, device=self._out_th_device)]
            times = th.empty((0,1), dtype = th.float32, device=self._out_th_device)
        return imgs, times
        
    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        if requestedJoints is None:
            requestedJoints = self.sub_adapter()._monitored_joints
        jstate_pve = self._sub_adapter.getJointsState(requestedJoints)
        return jstate_pve.unsqueeze(0).to(self._out_th_device)


    @override
    def getExtendedJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        raise NotImplementedError()
        if requestedJoints is None:
            requestedJoints = self.sub_adapter()._monitored_joints
        jstate = self._sub_adapter.getJointsState(requestedJoints)
        return th.stack([th.as_tensor([ jstate[k].position.item(),
                                        jstate[k].rate.item(),
                                        jstate[k].effort.item(),
                                        0.0,  # TODO: fix this
                                        jstate[k].effort.item() # TODO: fix this
                                        ]) for k in requestedJoints]).unsqueeze(0).to(self._out_th_device)
    
    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        raise NotImplementedError()
        return self._sub_adapter.get_joints_state_step_stats().unsqueeze(0)
        
    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | None, use_com_pose : bool = False) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self.sub_adapter()._monitored_links
        ls = self._sub_adapter.getLinksState(requestedLinks, use_com_pose=use_com_pose)
        r = th.stack([th.cat([ ls[k].pose.position,
                                ls[k].pose.orientation_xyzw,
                                ls[k].pos_velocity_xyz,
                                ls[k].ang_velocity_xyz])
                        for k in requestedLinks]).unsqueeze(0).to(self._out_th_device)
        return r

    @override
    def get_link_gravity_direction(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self.sub_adapter()._monitored_links
        return self._sub_adapter.get_link_gravity_direction(requestedLinks=requestedLinks).unsqueeze(0).to(device=self._out_th_device,
                                                                                                           dtype=self._out_th_float_dtype)

    @override
    def get_link_relative_angular_velocity(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self.sub_adapter()._monitored_links
        return self._sub_adapter.get_link_relative_angular_velocity(requestedLinks=requestedLinks).unsqueeze(0).to(device=self._out_th_device,
                                                                                                           dtype=self._out_th_float_dtype)

    @override
    def get_local_link_linear_acceleration(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self.sub_adapter()._monitored_links
        return self._sub_adapter.get_local_link_linear_acceleration(requestedLinks=requestedLinks).unsqueeze(0).to(device=self._out_th_device,
                                                                                                           dtype=self._out_th_float_dtype)

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
                self._sub_adapter.apply_joint_impedances({n:joint_impedances_pvesd[0,i] for i,n in enumerate(joint_names)})
    
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
        self._sub_adapter.build_scenario(models=models, **kwargs)
    
    @override
    def startup(self):
        self._sub_adapter.startup()

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
    def getEnvTimeFromEpStart(self) -> float:
        return self._sub_adapter.getEnvTimeFromEpStart()
    
    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        return self._sub_adapter.get_current_joint_impedance_command().unsqueeze(0)
    
    def sub_adapter(self):
        return self._sub_adapter
    
    @override
    def moveToJointPoseSync(self,   joint_names : Sequence[tuple[str,str]],
                                    positions : th.Tensor,
                                    velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None,
                                    joint_position_tolerance : float = 0.01,
                                    max_time_s : float = 60,
                                    joint_velocity_scaling : dict[tuple[str,str],float] = {}) -> None:
        positions=positions[0]
        self._sub_adapter.moveToJointPoseSync(  jointPositions={jn:jp.item() for jn,jp in zip(joint_names,positions)},
                                                velocity_scaling=velocity_scaling,
                                                acceleration_scaling=acceleration_scaling,
                                                joint_position_tolerance = joint_position_tolerance,
                                                max_time_s = max_time_s,
                                                joint_velocity_scaling=joint_velocity_scaling)
        
    @override
    def setJointsPositionCommand(self, joint_names : Sequence[tuple[str,str]], positions : th.Tensor,
                                        velocity_scaling : Optional[float] = None,
                                        acceleration_scaling : Optional[float] = None) -> None:
        positions=positions[0]
        self._sub_adapter.setJointsPositionCommand(jointPositions={jn:jp.item() for jn,jp in zip(joint_names,positions)},
                                                   velocity_scaling=velocity_scaling,
                                                   acceleration_scaling=acceleration_scaling)


    @override
    def control_period(self) -> th.Tensor:
        return th.as_tensor([self.sub_adapter.control_period()], device=self._out_th_device, dtype=self._out_th_float_dtype)
    
    @override
    def initialize_for_episode(self, vec_mask: th.Tensor | None = None):
        if vec_mask is None or vec_mask[0]:
            self._sub_adapter.initialize_for_episode()

    @override
    def initialize_for_step(self):
        return self.sub_adapter.initialize_for_step()