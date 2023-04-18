from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from nptyping import NDArray
import numpy as np

class CartesianPositionEnvController(ABC):
    @abstractmethod
    def setCartesianPoseCommand(self, linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]]) -> None:
        """Request a set of links to be placed at a specific cartesian pose.

        This is mainly meant as a way to perform cartesian end effector control. Meaning
        inverse kinematics will be computed to accomodate the request.

        Parameters
        ----------
        linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]]]
            Dict containing the pose command for each link. Each element of the dict
            is identified by a key of the form (model_name, joint_name). The pose is specified as
            a numpy array in the format: (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)

        Returns
        -------
        None
            Nothing is returned

        """
        raise NotImplementedError()
    
    @abstractmethod
    def moveToEePoseSync(self,  poses : Dict[Tuple[str,str],List[float]] = None, 
                                do_cartesian = False, velocity_scaling :Optional[float] = None,
                                acceleration_scaling : Optional[float] = None, ee_link : Optional[str] = None,
                                reference_frame : Optional[str] = None):
        raise NotImplementedError()
