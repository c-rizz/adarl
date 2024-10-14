from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
from adarl.utils.utils import JointState, LinkState, Pose, build_pose
from adarl.adapters.BaseVecAdapter import BaseVecAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter

import torch as th


class BaseVecSimulationAdapter(BaseVecAdapter, BaseSimulationAdapter):

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