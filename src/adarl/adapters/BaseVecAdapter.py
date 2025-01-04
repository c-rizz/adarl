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

    def __init__(self, vec_size : int,
                       output_th_device : th.device):
        """Initialize the adapter.
        """
        self._vec_size = vec_size
        self._out_th_device = output_th_device
        super().__init__()

    def vec_size(self):
        return self._vec_size
    
    def output_th_device(self) -> th.device:
        return self._out_th_device
    
    @abstractmethod
    def getRenderings(self, requestedCameras : List[str], vec_mask : th.Tensor | None = None) -> tuple[list[th.Tensor], th.Tensor]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            A tuple with a batch of batches of images in the first element and the sim time of each image
            in the second. The order is that of the requestedCameras argument.
            The first element contains a list of length len(requestedCameras) containing tensors of shape
            (th.count_nonzero(vec_mask), <image_shape>) and the second has shape(th.count_nonzero(vec_mask), len(requestedCameras))
        vec_mask: th.Tensor
            Which envirnoments to render, if None, renders all

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
            Tensor of shape (vec_size, len(monitored_joints),3) containig respecitvely position, velocity and effort for each joint in set_monitored_joints()

        """
        ...
    @abstractmethod
    def getJointsState(self, requestedJoints : Sequence[JointName] | None = None) -> th.Tensor:
        raise NotImplementedError()
    

    @abstractmethod
    @overload
    def getExtendedJointsState(self, requestedJoints : Sequence[Tuple[str,str]]) -> th.Tensor:
        """Get the state of the requested joints.

        Parameters
        ----------
        requestedJoints : List[Tuple[str,str]]
            Joints to tget the state of. Each element of the list represents a joint in the format [model_name, joint_name]

        Returns
        -------
        th.Tensor
            A tensor of shape (vec_size, len(requestedJoints),5) containig respecitvely position, velocity, applied effort,
             acceleration, measured effort for each requested joint

        """
        ...

    @abstractmethod
    @overload
    def getExtendedJointsState(self) -> th.Tensor:
        """Get the state of the monitored joints.

        Returns
        -------
        th.Tensor
            Tensor of shape (vec_size, len(monitored_joints),3) containig respecitvely position, velocity, applied effort,
             acceleration, measured effort for each joint in set_monitored_joints()

        """
        ...

    @abstractmethod
    def getExtendedJointsState(self, requestedJoints : Sequence[JointName] | None = None) -> th.Tensor:
        raise NotImplementedError()
    

    @abstractmethod
    def get_joints_state_step_stats(self) -> th.Tensor:
        """Returns joint state statistics over the last step for the monitored joints. The value of these statistics after a call to run()
        is currently undefined.

        Returns
        -------
        th.Tensor
            Torch tensor of size (vec_size, 4,len(monitored_joints),4) containing min,max,average,std of the position,velocity,
            acceleration and effort of each monitored joint. The joints are in the order specified in set_monitored_joints.
        """
        ...

    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName], use_com_frame : bool = False) -> th.Tensor:
        """Get the state of the requested links.

        Parameters
        ----------
        linkNames : List[str]
            Names of the link to get the state of
        use_com_frame : bool
            Defines if th elink information should be referring to the frame pose or the center of mass pose

        Returns
        -------
        th.Tensor
            Tensor of shape (vec_size, len(linkNames),13) containig the link state of each of the requested joints.
            The link state is a concatenation of position_xyz,orientation_xyzw,linear_com_velocity_xyz,angular_velocity_xyz.

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
