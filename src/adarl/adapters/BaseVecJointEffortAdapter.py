from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence
from adarl.adapters.BaseAdapter import BaseAdapter
import torch as th

class BaseVecJointEffortAdapter(ABC):
    @abstractmethod
    def setJointsEffortCommand(self, joint_names : Sequence[tuple[str,str]], efforts : th.Tensor) -> None:
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            List of the joint names
        efforts : th.Tensor
            Tensor of shape (vec_size, len(joint_names)) containing the effort for each joint in each environment.
        """
        raise NotImplementedError()