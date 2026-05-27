from __future__ import annotations
from abc import abstractmethod
from typing import Any, Sequence
from adarl.utils.utils import Pose, build_pose
from adarl.adapters.BaseVecAdapter import BaseVecAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef

import torch as th


class BaseVecSimulationAdapter(BaseVecAdapter):

    def setJointsAndLinksStateDirect(self,
                                     joint_names : Sequence[tuple[str,str]] | None = None,
                                     joint_states_pve : th.Tensor | None = None,
                                     link_names : Sequence[tuple[str,str]] | None = None,
                                     link_states_pose_vel : th.Tensor | None = None,
                                     vec_mask : th.Tensor | None = None):
        """Set joint and link state together, using a shared vec mask.

        The default implementation preserves existing behavior by delegating to
        `setLinksStateDirect()` first and `setJointsStateDirect()` second.
        Adapters can override this to fuse preprocessing or device-side updates.
        """
        if joint_names is None:
            if joint_states_pve is not None:
                raise ValueError("joint_states_pve was provided without joint_names")
        elif joint_states_pve is None:
            raise ValueError("joint_names was provided without joint_states_pve")

        if link_names is None:
            if link_states_pose_vel is not None:
                raise ValueError("link_states_pose_vel was provided without link_names")
        elif link_states_pose_vel is None:
            raise ValueError("link_names was provided without link_states_pose_vel")

        if link_names is not None:
            self.setLinksStateDirect(link_names=link_names,
                                     link_states_pose_vel=link_states_pose_vel,
                                     vec_mask=vec_mask)
        if joint_names is not None:
            self.setJointsStateDirect(joint_names=joint_names,
                                      joint_states_pve=joint_states_pve,
                                      vec_mask=vec_mask)

    @abstractmethod
    def setJointsStateDirect(self, joint_names : Sequence[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor | None = None):        
        """Set the state for a set of joints

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            The names of the joints to set the state for
        joint_states_pve : th.Tensor
            A tensor of shape (vec_size, len(joint_names), 3) containins, position,velocity and effort for each joint
        vec_mask : th.Tensor
            Tensor of size (vec_size,), indicating which simulators to use or not use. If None, apply to all
        """
        raise NotImplementedError()
    
    @abstractmethod
    def setLinksStateDirect(self, link_names : Sequence[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor | None = None):
        """Set the state for a set of links

        Parameters
        ----------
        link_names : Sequence[tuple[str,str]]
            The names of the links to set the state for
        link_states_pose_vel : th.Tensor
            A tensor of shape (vec_size, len(link_names), 13), containing, for each link, position_xyz, orientation_xyzw, linear_velocity_xyz, angular_velocity_xyz
        vec_mask : th.Tensor
            Tensor of size (vec_size,), indicating which simulators to use or not use. If None, apply to all
        """
        raise NotImplementedError()

    @abstractmethod
    def setupLight(self):
        raise NotImplementedError()

    @abstractmethod
    def build_scenario(self, models : Sequence[ModelSpawnDef] = [], **kwargs):
        raise NotImplementedError()
        
    @abstractmethod
    def spawn_models(self, models : Sequence[ModelSpawnDef]) -> list[str]:
        """Spawn models in all of the simulations. Each model in the provided list will be spawned in all simulations.


        Parameters
        ----------
        models : Sequence[ModelSpawnDef]
            The models to spawn

        Returns
        -------
        list[str]
            The names of the spawned models
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
    
    def set_body_collisions(self, link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]]):
        raise NotImplementedError()
    
    def set_link_impulses(self, link_ids : Sequence[Any],
                                force_torque_xyzxyz : th.Tensor,
                                durations : th.Tensor, delays : th.Tensor,
                                vec_mask : th.Tensor) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def sim_step_duration(self) -> th.Tensor:
        raise NotImplementedError()
