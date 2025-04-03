from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch as th
from typing import Sequence

class BaseVecJointPositionAdapter(ABC):
    @abstractmethod
    def setJointsPositionCommand(self, joint_names : Sequence[tuple[str,str]], positions : th.Tensor) -> None:
        """Set the position to be requested on a set of joints.

        The position can be an angle (in radiants) or a length (in meters), depending
        on the type of joint.

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            List of the joint names
        positions : th.Tensor
            Tensor of shape (vec_size, len(joint_names)) containing the position for each joint in each environment.

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()

    @abstractmethod
    def moveToJointPoseSync(self,   joint_names : Sequence[tuple[str,str]],
                                    positions : th.Tensor,
                                    velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None) -> None:
        
        """ Moves the joints to the specified positions following a continuous trajectory (not instantaneously as 
            could be done in a simulation). This is not intended to be used during the episode, but just to reposition
            joints during reset/initialization procedures.        

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            List of the joint names
        positions : th.Tensor
            Tensor of shape (vec_size, len(joint_names)) containing the position for each joint in each environment.
        velocity_scaling : Optional[float], optional
            Scales the velocity of the movement
        acceleration_scaling : Optional[float], optional
            Scales the acceleration of the movement

        Raises
        ------
        MoveFailError
            If the movement fails
        """
        raise NotImplementedError()