from __future__ import annotations

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from typing_extensions import override
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from typing import Sequence, Any
from adarl.utils.utils import Pose, build_pose, JointState, LinkState
import torch as th

class VecSimJointImpedanceAdapterWrapper(BaseVecSimulationAdapter, BaseVecJointImpedanceAdapter):
    
    def __init__(self,  vec_size : int,
                        th_device : th.device,
                        adapter):
        super().__init__(vec_size=vec_size,
                         output_th_device=th_device)
        if not isinstance(adapter, BaseJointImpedanceAdapter):
            raise RuntimeError(f"adapter Must be a BaseJointImpedanceAdapter")
        if not isinstance(adapter, BaseSimulationAdapter):
            raise RuntimeError(f"adapter Must be a BaseSimulationAdapter")
        self._sub_adapter = adapter
        if vec_size!=1: 
            raise NotImplementedError()


    @override
    def getRenderings(self, requestedCameras : list[str], vec_mask : th.Tensor | None = None) -> tuple[list[th.Tensor], th.Tensor]:
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
        jstate = self._sub_adapter.getJointsState(requestedJoints)
        return th.stack([th.as_tensor([ jstate[k].position.item(),
                                        jstate[k].rate.item(),
                                        jstate[k].effort.item()]) for k in requestedJoints]).unsqueeze(0).to(self._out_th_device)


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
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor | None = None):
        if vec_mask is None or vec_mask.item():
            self._sub_adapter.setJointsStateDirect({n:JointState(position = joint_states_pve[0,i,0],
                                                                rate =     joint_states_pve[0,i,1],
                                                                effort =   joint_states_pve[0,i,2]) 
                                                    for i,n in enumerate(joint_names)})
    
    @override
    def setLinksStateDirect(self, link_names : list[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor | None = None):
        if vec_mask is None or vec_mask.item():
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
    def spawn_models(self, models : Sequence[ModelSpawnDef]) -> list[str]:
        names = []
        for model_def in models:
            names.append( self._sub_adapter.spawn_model(    model_name = model_def.name,
                                                            model_definition_string = model_def.definition_string,
                                                            model_format = model_def.format,
                                                            pose = model_def.pose,
                                                            model_kwargs = model_def.kwargs))
        return names

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
    def initialize_for_step(self):
        return self._sub_adapter.initialize_for_step()

    @override
    def step(self) -> float:
        return self._sub_adapter.step()
    
    @override
    def control_period(self) -> th.Tensor:
        return self._sub_adapter.control_period()
    
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
    def sim_step_duration(self) -> float:
        return self._sub_adapter.sim_step_duration()