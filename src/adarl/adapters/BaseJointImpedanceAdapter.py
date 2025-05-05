from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Mapping, Sequence, overload
from adarl.adapters.BaseAdapter import BaseAdapter
import torch as th

class BaseJointImpedanceAdapter(BaseAdapter):

    @overload
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]], delay_sec : float = 0.0) -> None:
        """ Sets a joint impedance command. The command only is appied at the next call of step().
        After step() is called the command gets cleared out. Position, velocity and torque commands may override this.
        For each joint the command is a tuple of the format (position,velocity,effort,stiffness,damping)

        Parameters
        ----------
        joint_impedances_pvesd : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]
            Dictionary with:
             - key=(<model_name>,<joint_name>)
             - value=(<position_reference>,<velocity_reference>,<effort_reference>,<position_gain>,<velocity_gain>)
        delay_sec : float, optional
            Delay the application of the command of this time duration. By default, 0.0
        """
        ...
    @overload
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor, delay_sec : float = 0.0) -> None:
        """ Sets a joint impedance command. The command only is appied at the next call of step().
        After step() is called the command gets cleared out. Position, velocity and torque commands may override this.
        The command for each joint specified in set_controlled_joints is placed in a row of the provided tensor, 
        in the order (position,velocity,effort,stiffness,damping)

        Parameters
        ----------
        joint_impedances_pvesd : th.Tensor
            Torch tensor of shape (len(get_impedance_controlled_joints()),5). Each row represents the joint impedance command in the
            order (position,velocity,effort,stiffness,damping)
        delay_sec : float, optional
            Delay the application of the command of this time duration. By default, 0.0
        """
        ...
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]] | th.Tensor, delay_sec : float = 0.0) -> None:
        raise NotImplementedError()
    
    @abstractmethod    
    def apply_joint_impedances(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]] | th.Tensor) -> None:
        """ Applies a joint impedance command immediately.
            Meant to be used outside of the normal control loop, just for resetting/initializing.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def set_impedance_controlled_joints(self, joint_names : Sequence[Tuple[str,str]]):
        """Set the joints that will be controlled by the adapter

        Parameters
        ----------
        joint_names : Sequence[Tuple[str,str]]
            List of the controlled joint names

        """
        raise NotImplementedError()
    

    @abstractmethod
    def get_impedance_controlled_joints(self) -> List[Tuple[str,str]]:
        """Get the names of the joints that are controlled by this adapter

        Returns
        -------
        List[Tuple[str,str]]
            The list of the joint names

        """
        raise NotImplementedError()


    @abstractmethod
    def get_current_joint_impedance_command(self) -> th.Tensor:
        """Returns the last command that was applied to the controlled joints.

        Returns
        -------
        th.Tensor
            Tensor of size (len(impedance_controlled_joints), 5)
        """
        ...