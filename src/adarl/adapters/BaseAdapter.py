"""This file implements the Envitronment controller class, whic is the superclass for all th environment controllers."""
#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Optional

from abc import ABC, abstractmethod
from threading import Thread, RLock
import torch as th
from adarl.utils.utils import JointState, LinkState
from typing import overload, Sequence

JointName = Tuple[str,str]
LinkName = Tuple[str,str]

class BaseAdapter(ABC):
    """Base class for implementing environment adapters. Adapters allow to interface with a variety of 
    execution environments, being them real or simulated. Different capabilities may be available depending
    on the kind of environment they are implemented for, which can allow very different capabilities.
    Simulation environments can offer different ways to manipulated the environment, but also environments
    based on different control approaches can offer different ways to command robot hardware, and different
    sensor setups can give access to different sensor data.

    It is an abstract class, it is meant to be extended with sub-classes for specific environments
    """

    def __init__(self):
        """Initialize the adapter.
        """
        self._running_run_async_lock = RLock()
        self._running_run_async = False
        self.__lastResetTime = 0
        self._jointsToObserve = []
        self._linksToObserve = []
        self._camerasToObserve = []

    def set_monitored_joints(self, jointsToObserve : List[JointName]):
        """Set which joints should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        jointsToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, joint_name)

        """
        self._jointsToObserve = jointsToObserve


    def set_monitored_links(self, linksToObserve : List[LinkName]):
        """Set which links should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        linksToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, link_name)

        """
        self._linksToObserve = linksToObserve

    def set_monitored_cameras(self, camerasToRender : List[str] = []):
        """Set which camera should be rendered after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        camerasToRender : List[str]
            List of the names of the cameras

        """
        self._camerasToObserve = camerasToRender


    @abstractmethod
    def startup(self):
        """Start up the controller. This must be called after set_monitored_cameras, set_monitored_links and set_monitored_joints."""
        raise NotImplementedError()

    def stopController(self):
        pass
    
    @abstractmethod
    def step(self) -> float:
        """Run a simulation step.

        Returns
        -------
        float
            Duration of the step in simulation time (in seconds)"""

        raise NotImplementedError()

    @abstractmethod
    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[th.Tensor, float]]:
        """Get the images for the specified cameras.

        Parameters
        ----------
        requestedCameras : List[str]
            List containing the names of the cameras to get the images of

        Returns
        -------
        List[sensor_msgs.msg.Image]
            List contyaining the images for the cameras specified in requestedCameras, in the same order

        """
        raise NotImplementedError()


    @abstractmethod
    @overload
    def getJointsState(self, requestedJoints : Sequence[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        """Get the state of the requested joints.

        Parameters
        ----------
        requestedJoints : List[Tuple[str,str]]
            Joints to tget the state of. Each element of the list represents a joint in the format [model_name, joint_name]

        Returns
        -------
        Dict[Tuple[str,str],JointState]
            Dictionary containig the state of the joints. The keys are in the format [model_name, joint_name]

        """
        ...
    @abstractmethod
    @overload
    def getJointsState(self) -> th.Tensor:
        """Get the state of the monitored joints.

        Returns
        -------
        th.Tensor
            Tensor of shape (joints_num, 3) with position,velocity,effort for each joint in set_monitored_joints()

        """
        ...
    @abstractmethod
    def getJointsState(self, requestedJoints : Sequence[JointName] | None = None) -> Dict[JointName,JointState] | th.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_joints_state_step_stats(self) -> th.Tensor:
        """Returns joint state statistics over the last step for the monitored joints. The value of these statistics after a call to run()
        is currently undefined.

        Returns
        -------
        th.Tensor
            Torch tensor of size (4,len(monitored_joints),3) containing min,max,average,std of the position,velocity
             and effort of each monitored joint. The joints are in the order use din set_monitored_joints.
        """
        ...

    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName]) -> Dict[LinkName,LinkState]:
        """Get the state of the requested links.

        Parameters
        ----------
        linkNames : List[str]
            Names of the link to get the state of

        Returns
        -------
        Dict[str,LinkState]
            Dictionary, indexed by link name containing the state of each link

        """
        ...
    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : None) -> th.Tensor:
        """Get the state of the monitored links.

        Returns
        -------
        th.Tensor
            Tensor containing the link state for each monitored link

        """
        ...
    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName] | None) -> Dict[LinkName,LinkState] | th.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        self.__lastResetTime = self.getEnvTimeFromStartup()


    @abstractmethod
    def getEnvTimeFromStartup(self) -> float:
        """Get the current time within the simulation."""
        raise NotImplementedError()

    def getEnvTimeFromReset(self) -> float:
        """Get the current time within the simulation."""
        return self.getEnvTimeFromStartup() - self.__lastResetTime

    @abstractmethod
    def run(self, duration_sec : float):
        """Run the environment for the specified duration"""
        raise NotImplementedError()


    def run_async_loop(self, on_finish_callback : Optional[Callable[[], None]]):
        # ggLog.info(f"run async loop")
        should_run = True
        t_remaining = 1 # Always do at least one step # self._run_async_timeout - self.getEnvTimeFromStartup()
        while should_run and t_remaining > 0:
            # ggLog.info(f"running")
            self.run(duration_sec = min(0.5,t_remaining))
            with self._running_run_async_lock:
                should_run = self._running_run_async
            t_remaining =  self._run_async_timeout - self.getEnvTimeFromStartup()
        with self._running_run_async_lock:
            self._running_run_async = False
        if on_finish_callback is not None:
            on_finish_callback()

    def run_async(self, duration_sec : float = float("+inf"), on_finish_callback = None):
        """ Asynchronously run the simulation from a parallel thread.

        Parameters
        ----------
        duration_sec : float, optional
            Run the environment for this duration (in case of simulation, this is simulated time).
             This can be preempted by stop_run_async(). By default float("+inf")
        """
        # ggLog.info(f"run async({duration_sec})")
        with self._running_run_async_lock:
            # ggLog.info(f"run async acquired lock")
            self._run_async_duration_sec = duration_sec
            self._run_async_timeout = self.getEnvTimeFromStartup() + duration_sec
            self._running_run_async = True
            self._run_async_thread = Thread(target=self.run_async_loop, args=[on_finish_callback])
            self._run_async_thread.start()

    def wait_run_async(self):
        with self._running_run_async_lock:
            if self._run_async_thread is None:
                return
        self._run_async_thread.join()

    def stop_run_async(self):        
        with self._running_run_async_lock:
            if not self._running_run_async:
                return
            self._running_run_async = False

    @abstractmethod
    def build_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of controller"""
        raise NotImplementedError()
    

    @abstractmethod
    def destroy_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of controller"""
        raise NotImplementedError()
