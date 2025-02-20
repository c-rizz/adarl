from __future__ import annotations
from abc import abstractmethod
from typing import Any, Sequence
from adarl.utils.utils import Pose, build_pose
from adarl.adapters.BaseVecAdapter import BaseVecAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef

import torch as th


class BaseVecSimulationAdapter(BaseVecAdapter):

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
        raise NotImplementedError