from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from lr_gym.utils.utils import MoveFailError

class BaseJointPositionAdapter(ABC):
    @abstractmethod
    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float]) -> None:
        """Set the position to be requested on a set of joints.

        The position can be an angle (in radiants) or a length (in meters), depending
        on the type of joint.

        Parameters
        ----------
        jointPositions : Dict[Tuple[str,str],float]]
            List containing the position command for each joint. Each element of the list
            is a tuple of the form (model_name, joint_name, position)

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()

    @abstractmethod
    def moveToJointPoseSync(self,   jointPositions : Dict[Tuple[str,str],float],
                                    velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None) -> None:
        
        """ Moves the joints to the specified positions following a continuous trajectory (not instantaneously as 
            could be done in a simulation). This is not intended to be used during the episode, but just to reposition
            joints during reset/initialization procedures.        

        Parameters
        ----------
        jointPositions : Dict[Tuple[str,str],float]
            Maps joint names (model_name, joint_name) to joint positions.
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