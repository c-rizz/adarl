from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence
import torch as th
from adarl.adapters.BaseVecAdapter import BaseVecAdapter

class BaseVecJointEffortAdapter(BaseVecAdapter, ABC):
    @abstractmethod
    def setJointsEffortCommand(self, joint_names : Sequence[tuple[str,str]], efforts : th.Tensor, vec_mask : th.Tensor | None = None) -> None:
        """Set the efforts to be applied on a set of joints.

        Effort means either a torque or a force, depending on the type of joint.

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            List of the joint names
        efforts : th.Tensor
            Tensor of shape (vec_size, len(joint_names)) containing the effort for each joint in each environment.
        vec_mask : th.Tensor | None
            Optional tensor mask of shape (vec_size,) indicating which environments the command should be applied to. If None,
            the command is applied to all environments.
        """
        raise NotImplementedError()