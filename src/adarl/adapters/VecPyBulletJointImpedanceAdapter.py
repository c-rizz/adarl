from __future__ import annotations

from typing_extensions import override
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from typing import Sequence, Any
from adarl.utils.utils import Pose, build_pose
import torch as th

class VecPyBulletJointImpedanceAdapter(BaseVecSimulationAdapter, BaseVecJointImpedanceAdapter):
    
    @override
    def getRenderings(self, requestedCameras : list[str]) -> tuple[list[th.Tensor], th.Tensor]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            A tuple with a batch of batches images in the first element and the sim time of each image in the second. The order is that of the requestedCameras argument.
            The first element containin a list if length len(requestedCameras) containin tensors of shape
             (vec_size, <image_shape>) and the second has shape(vec_size, len(requestedCameras))

        """
        raise NotImplementedError()
        
    @override
    def getJointsState(self, requestedJoints : Sequence[tuple[str,str]] | None = None) -> th.Tensor:
        raise NotImplementedError()

    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        """Returns joint state statistics over the last step for the monitored joints. The value of these statistics after a call to run()
        is currently undefined.

        Returns
        -------
        th.Tensor
            Torch tensor of size (vec_size, 4,len(monitored_joints),3) containing min,max,average,std of the position,velocity
             and effort of each monitored joint. The joints are in the order specified in set_monitored_joints.
        """
        ...
        
    @override
    def getLinksState(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        raise NotImplementedError()


    @override
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
                            model_file : str | None = None,
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
    
    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor, delay_sec : th.Tensor | float = 0.0, joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        raise NotImplementedError()
    
    @override
    def reset_joint_impedances_commands(self):
        """Reset any internal queue of joint impedances commands. Makes it such as if no commands were ever sent.
        """
        ...

    @override
    def set_current_joint_impedance_command(self,   joint_impedances_pvesd : th.Tensor,
                                                    joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        """Sets the current joint impedance command. This command will be applied immediately, however it may be overridden
            by any previously sent command that is supposed to be applied at a simtime 
            simultaneous or previous to the current one. 

        Parameters
        ----------
        joint_impedances_pvesd : th.Tensor
            Tensor of size (vec_size, len(joint_names), 5), containing the command for each joint in each environment.
            The last dimension contains respectively (<position_reference>,<velocity_reference>,<effort_reference>,<position_gain>,<velocity_gain>)
        joint_names : Sequence[tuple[str,str]] | None, optional
            The joint to apply the commands to. If None then it is assumed it is equal to all impedance_controlled_joints.

        Raises
        ------
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """
        ...
    
    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[tuple[str,str]]):
        """Set the joints that will be controlled by the adapter

        Parameters
        ----------
        joint_names : Sequence[Tuple[str,str]]
            List of the controlled joint names

        """
        raise NotImplementedError()
    

    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str,str]]:
        """Get the names of the joints that are controlled by this adapter

        Returns
        -------
        List[Tuple[str,str]]
            The list of the joint names

        """
        raise NotImplementedError()