"""This file implements the Envitronment controller class, whic is the superclass for all th environment controllers."""
#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Optional
from typing_extensions import deprecated

from abc import ABC, abstractmethod
from threading import Thread, RLock
import torch as th
from adarl.utils.utils import JointState, LinkState
from typing import overload, Sequence
from adarl.adapters.BaseAdapter import BaseAdapter
JointName = Tuple[str,str]
LinkName = Tuple[str,str]

class BaseVecAdapter(BaseAdapter):
    """Base class for implementing environment adapters. Adapters allow to interface with a variety of 
    execution environments, being them real or simulated. Different capabilities may be available depending
    on the kind of environment they are implemented for, which can allow very different capabilities.
    Simulation environments can offer different ways to manipulated the environment, but also environments
    based on different control approaches can offer different ways to command robot hardware, and different
    sensor setups can give access to different sensor data.

    It is an abstract class, it is meant to be extended with sub-classes for specific environments
    """

    def __init__(self, vec_size : int):
        """Initialize the adapter.
        """
        self._vec_size = vec_size
        super().__init__()

    
    @abstractmethod
    def getRenderings(self, requestedCameras : List[str]) -> tuple[th.Tensor, th.Tensor]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            A tuple with a batch of batches images in the first element and the sim time of each image in the second. The order is that of the requestedCameras argument.
            The first tensor has shape (vec_size, len(requestedCameras),<image_shape>) and the second has shape(vec_size, len(requestedCameras))

        """
        raise NotImplementedError()


    @abstractmethod
    @overload
    def getJointsState(self, requestedJoints : Sequence[Tuple[str,str]]) -> th.Tensor:
        """Get the state of the requested joints.

        Parameters
        ----------
        requestedJoints : List[Tuple[str,str]]
            Joints to tget the state of. Each element of the list represents a joint in the format [model_name, joint_name]

        Returns
        -------
        th.Tensor
            A tensor of shape (vec_size, len(requestedJoints),3) containig respecitvely position, velocity and effort for each requested joint

        """
        ...
    @abstractmethod
    @overload
    def getJointsState(self) -> th.Tensor:
        """Get the state of the monitored joints.

        Returns
        -------
        th.Tensor
            Tensor of shape (vec_size, len(requestedJoints),3) containig respecitvely position, velocity and effort for each joint in set_monitored_joints()

        """
        ...
    @abstractmethod
    def getJointsState(self, requestedJoints : Sequence[JointName] | None = None) -> th.Tensor:
        raise NotImplementedError()

    @abstractmethod
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

    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName]) -> th.Tensor:
        """Get the state of the requested links.

        Parameters
        ----------
        linkNames : List[str]
            Names of the link to get the state of

        Returns
        -------
        th.Tensor
            Tensor of shape (vec_size, len(requestedJoints),13) containig the link state of each of the requested joints.
            The link state is a concatenation of position_xyz,orientation_xyzw,linear_velocity_xyz,angular_velocity_xyz.

        """
        ...

    @abstractmethod
    @overload
    def getLinksState(self) -> th.Tensor:
        """Get the state of the monitored links.

        Returns
        -------
        th.Tensor
            Tensor containing the link state for each monitored link.

        """
        ...
    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName] | None) -> th.Tensor:
        raise NotImplementedError()
