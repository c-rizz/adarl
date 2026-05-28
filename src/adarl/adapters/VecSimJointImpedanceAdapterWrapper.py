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
    """Vectorizing wrapper for simulated joint impedance adapters implementing BaseVecJointImpedanceAdapter."""
    def __init__(self,  th_device : th.device,
                        adapters):
        vec_size = len(adapters)
        super().__init__(vec_size=vec_size,
                         output_th_device=th_device)
        if not isinstance(adapters, list):
            adapters = [adapters]
        for a in adapters:
            if not isinstance(a, BaseJointImpedanceAdapter):
                raise RuntimeError(f"adapter Must be a BaseJointImpedanceAdapter")
            if not isinstance(a, BaseSimulationAdapter):
                raise RuntimeError(f"adapter Must be a BaseSimulationAdapter")
        self._sub_adapters = adapters


    @override
    def getRenderings(self, requestedCameras : list[str], vec_mask : th.Tensor | None = None) -> tuple[list[th.Tensor], th.Tensor]:
        vimgs = []
        vtimes = []
        for a in self._sub_adapters:
            if vec_mask is None or vec_mask.item():
                rdict = a.getRenderings(requestedCameras=requestedCameras)
                imgs =  [th.as_tensor(rdict[n][0], device=self._out_th_device).unsqueeze(0) for n in requestedCameras]
                times = th.stack([th.as_tensor(rdict[n][1], device=self._out_th_device) for n in requestedCameras]).expand(self._vec_size, len(requestedCameras))
            else:
                imgs  = [th.empty((0,3,9,16), dtype = th.uint8, device=self._out_th_device)]
                times = th.empty((0,1), dtype = th.float32, device=self._out_th_device)
            vimgs.append(imgs)
            vtimes.append(times)
        # concatenate along vec dimension
        imgs = [th.cat([vimgs[i][j] for i in range(len(vimgs))], dim=0) for j in range(len(requestedCameras))]
        times = th.cat(vtimes, dim=0)
        return imgs, times
        
    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        vjstates = []
        for a in self._sub_adapters:
            if requestedJoints is None:
                requestedJoints = a._monitored_joints
            jstate = a.getJointsState(requestedJoints)
            jstate_th = th.stack([th.as_tensor([jstate[k].position.item(),
                                                jstate[k].rate.item(),
                                                jstate[k].effort.item()]) for k in requestedJoints]).unsqueeze(0).to(self._out_th_device)
            vjstates.append(jstate_th)
        jstate_th = th.cat(vjstates, dim=0)
        return jstate_th


    @override
    def getExtendedJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        raise NotImplementedError()
    
    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        vstats = [a.get_joints_state_step_stats() for a in self._sub_adapters]
        return th.stack(vstats, dim=0)
        
    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | None, use_com_pose : bool = False) -> th.Tensor:
        vr = []
        for a in self._sub_adapters:
            if requestedLinks is None:
                requestedLinks = a._monitored_links
            ls = a.getLinksState(requestedLinks, use_com_pose=use_com_pose)
            r = th.stack([th.cat([  ls[k].pose.position,
                                    ls[k].pose.orientation_xyzw,
                                    ls[k].pos_velocity_xyz,
                                    ls[k].ang_velocity_xyz])
                            for k in requestedLinks]).unsqueeze(0).to(self._out_th_device)
            vr.append(r)
        r = th.cat(vr, dim=0)
        return r


    @override
    def setJointsStateDirect(self, joint_names : list[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor | None = None):
        for vi, a in enumerate(self._sub_adapters):
            if vec_mask is None or vec_mask.item():
                a.setJointsStateDirect({n:JointState(position = joint_states_pve[vi,i,0],
                                                     rate =     joint_states_pve[vi,i,1],
                                                     effort =   joint_states_pve[vi,i,2]) 
                                        for i,n in enumerate(joint_names)})
    
    @override
    def setLinksStateDirect(self, link_names : list[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor | None = None):
        for vi, a in enumerate(self._sub_adapters):
            if vec_mask is None or vec_mask.item():
                a.setLinksStateDirect({n:LinkState( position_xyz         = link_states_pose_vel[vi,i, 0:3],
                                                    orientation_xyzw     = link_states_pose_vel[vi,i, 3:7],
                                                    pos_com_velocity_xyz = link_states_pose_vel[vi,i, 7:10],
                                                    ang_velocity_xyz     = link_states_pose_vel[vi,i,10:13])
                                                for i,n in  enumerate(link_names)})

    @override
    def setupLight(self,    lightDirection,
                            lightColor,
                            lightDistance,
                            enable_shadows,
                            lightAmbientCoeff,
                            lightDiffuseCoeff,
                            lightSpecularCoeff):
        for a in self._sub_adapters:
            a.setupLight(lightDirection = lightDirection,
                         lightColor = lightColor,
                         lightDistance = lightDistance,
                         enable_shadows = enable_shadows,
                         lightAmbientCoeff = lightAmbientCoeff,
                         lightDiffuseCoeff = lightDiffuseCoeff,
                         lightSpecularCoeff = lightSpecularCoeff)

    @override
    def spawn_models(self, models : Sequence[ModelSpawnDef]) -> list[str]:
        for a in self._sub_adapters:
            names = []
            for model_def in models:
                names.append( a.spawn_model(model_name = model_def.name,
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
        for a in self._sub_adapters:
            a.delete_model(model_name)
    
    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None,
                                        joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        for vi, a in enumerate(self._sub_adapters):
            if vec_mask is None or vec_mask.item():
                if isinstance(delay_sec,th.Tensor):
                    delay_sec = delay_sec.item()
                if joint_names is None:
                    a.setJointsImpedanceCommand(joint_impedances_pvesd[vi],delay_sec)
            else:
                a.setJointsImpedanceCommand({n:tuple(joint_impedances_pvesd[vi,i].tolist()) for i,n in enumerate(joint_names)},
                                                            delay_sec=delay_sec)

    
    @override
    def reset_joint_impedances_commands(self):
        for a in self._sub_adapters:
            a.clear_commands()

    @override
    def set_current_joint_impedance_command(self,   joint_impedances_pvesd : th.Tensor,
                                                    vec_mask : th.Tensor | None = None,
                                                    joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        for vi, a in enumerate(self._sub_adapters):
            if vec_mask is None or vec_mask.item():
                if joint_names is None:
                    a.apply_joint_impedances(joint_impedances_pvesd[vi])
                else:
                    a.apply_joint_impedances({n:tuple(joint_impedances_pvesd[vi,i].tolist()) for i,n in enumerate(joint_names)})
    
    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[tuple[str,str]]):
        for a in self._sub_adapters:
            a.set_impedance_controlled_joints(joint_names)
    

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str, str]]):
        for a in self._sub_adapters:
            a.set_monitored_joints(jointsToObserve)
    
    @override
    def set_monitored_links(self, linksToObserve: Sequence[tuple[str, str]]):
        for a in self._sub_adapters:
            a.set_monitored_links(linksToObserve)
    
    @override
    def set_monitored_cameras(self, camera_names: Sequence[tuple[str, str]]):
        for a in self._sub_adapters:
            a.set_monitored_cameras(camera_names)
    
    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str,str]]:
        return self._sub_adapters[0].get_impedance_controlled_joints()
    
    @override
    def build_scenario(self, models: Sequence[ModelSpawnDef] = [], **kwargs):
        for a in self._sub_adapters:
            a.build_scenario(models=models, **kwargs)

    @override
    def startup(self):
        for a in self._sub_adapters:
            a.startup()
    
    @override
    def destroy_scenario(self, **kwargs):
        for a in self._sub_adapters:
            a.destroy_scenario(**kwargs)
    
    @override
    def run(self, duration_sec : float):
        for a in self._sub_adapters:
            a.run(duration_sec)

    @override
    def initialize_for_step(self):
        for a in self._sub_adapters:
            a.initialize_for_step()

    @override
    def step(self) -> float:
        r = []
        for a in self._sub_adapters:
            r.append(a.step())
        if not all(r_i == r[0] for r_i in r):
            raise RuntimeError("Sub-adapters returned different step durations")
        return r[0]
    
    @override
    def control_period(self) -> th.Tensor:        
        return self._sub_adapters[0].control_period()
    
    @override
    def resetWorld(self):
        for a in self._sub_adapters:
            a.resetWorld()
    
    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._sub_adapters[0].getEnvTimeFromStartup()
    
    @override
    def getEnvTimeFromEpStart(self) -> float:
        return self._sub_adapters[0].getEnvTimeFromEpStart()
    
    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        vjimp = []
        for a in self._sub_adapters:
            vjimp.append(a.get_current_joint_impedance_command().unsqueeze(0))
        return th.cat(vjimp, dim=0)
    
    def sub_adapters(self):
        return self._sub_adapters
    
    @override
    def sim_step_duration(self) -> float:
        return self._sub_adapters[0].sim_step_duration()
    
    @override
    def get_links_state_step_stats(self) -> th.Tensor:
        vstats = [a.get_joints_state_step_stats() for a in self._sub_adapters]
        return th.stack(vstats, dim=0)