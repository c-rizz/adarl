from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Mapping, Sequence, overload
from adarl.adapters.BaseVecAdapter import BaseVecAdapter
import torch as th

class BaseVecJointImpedanceAdapter(BaseVecAdapter):

    @overload
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None,
                                        joint_names : Sequence[tuple[str,str]] = None) -> None:
        """ Sets a joint impedance command. The command only is appied at the next call of step().
        After step() is called the command does NOT get cleared out. Position, velocity and torque commands may override this.
        For each joint the command is a tuple of the format (position,velocity,effort,stiffness,damping).

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]:
            The joint to apply the commands to.
        joint_impedances_pvesd : th.Tensor
            Tensor of size (vec_size, len(joint_names), 5), containing the command for each joint in each environment.
            The last dimension contains respectively (<position_reference>,<velocity_reference>,<effort_reference>,<position_gain>,<velocity_gain>)
        delay_sec : th.Tensor | float, optional
            Delay the application of the command of this time duration. It this is a Tensor, then it must be of shape (vec_size,), specifying
            the delay applied in each environment, if it is a float then the same delay is applied to all environments.
        vec_mask : th.Tensor
            Tensor of size (vec_size,), indicating which simulators to use or not use. If None, apply to all
        """
        ...
    @overload
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor, 
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None) -> None:
        """ Sets a joint impedance command. The command only is appied at the next call of step().
        After step() is called the command does NOT get cleared out. Position, velocity and torque commands may override this.
        For each joint the command is a tuple of the format (position,velocity,effort,stiffness,damping).
        This overload always specifies command for all the controlled joints (see set_impedance_controlled_joints()).

        Parameters
        ----------
        joint_impedances_pvesd : th.Tensor
            Tensor of size (vec_size, len(impedance_controlled_joints), 5), containing the command for each joint in each environment.
            The last dimension contains respectively (<position_reference>,<velocity_reference>,<effort_reference>,<position_gain>,<velocity_gain>)
        delay_sec : th.Tensor | float, optional
            Delay the application of the command of this time duration. It this is a Tensor, then it must be of shape (vec_size,), specifying
            the delay applied in each environment, if it is a float then the same delay is applied to all environments.
        vec_mask : th.Tensor
            Tensor of size (vec_size,), indicating which simulators to use or not use. If None, apply to all
        """
        ...
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor, 
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None, 
                                        joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def reset_joint_impedances_commands(self):
        """Reset any internal queue of joint impedances commands. Makes it such as if no commands were ever sent.
        """
        ...

    @abstractmethod
    def set_current_joint_impedance_command(self,   joint_impedances_pvesd : th.Tensor,
                                                    joint_names : Sequence[tuple[str,str]] | None = None,
                                                    vec_mask : th.Tensor | None = None) -> None:
        """Sets the current joint impedance command. This command will be applied immediately, however
            it may be overridden by any previously sent command that is supposed to be applied at a 
            simtime simultaneous or previous to the current one (due to action delaying). 

        Parameters
        ----------
        joint_impedances_pvesd : th.Tensor
            Tensor of size (vec_size, len(joint_names), 5), containing the command for each joint in each environment.
            The last dimension contains respectively (<position_reference>,<velocity_reference>,<effort_reference>,<position_gain>,<velocity_gain>)
        joint_names : Sequence[tuple[str,str]] | None, optional
            The joint to apply the commands to. If None then it is assumed it is equal to all impedance_controlled_joints.

        """
        ...
    
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
    def get_last_applied_command(self) -> th.Tensor:
        """Returns the last command that was applied to the controlled joints.

        Returns
        -------
        th.Tensor
            Tensor of size (vec_size, len(impedance_controlled_joints), 5)
        """
        ...