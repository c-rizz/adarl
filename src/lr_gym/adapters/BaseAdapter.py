"""This file implements the Envitronment controller class, whic is the superclass for all th environment controllers."""
#!/usr/bin/env python3
from typing import List, Tuple, Dict, Callable, Optional

from lr_gym.utils.utils import JointState
from lr_gym.utils.utils import LinkState
from abc import ABC, abstractmethod
from threading import Thread, RLock
import torch as th

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
        self._running_freerun_async_lock = RLock()
        self._running_freerun_async = False
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
    def getJointsState(self, requestedJoints : List[JointName]) -> Dict[JointName,JointState]:
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
        raise NotImplementedError()

    @abstractmethod
    def getLinksState(self, requestedLinks : List[LinkName]) -> Dict[LinkName,LinkState]:
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
    def freerun(self, duration_sec : float):
        """Run the environment for the specified duration"""
        raise NotImplementedError()


    def freerun_async_loop(self, on_finish_callback : Optional[Callable[[], None]]):
        # ggLog.info(f"Freerun async loop")
        should_run = True
        t_remaining = 1 # Always do at least one step # self._freerun_async_timeout - self.getEnvTimeFromStartup()
        while should_run and t_remaining > 0:
            # ggLog.info(f"Freerunning")
            self.freerun(duration_sec = min(0.5,t_remaining))
            with self._running_freerun_async_lock:
                should_run = self._running_freerun_async
            t_remaining =  self._freerun_async_timeout - self.getEnvTimeFromStartup()
        with self._running_freerun_async_lock:
            self._running_freerun_async = False
        if on_finish_callback is not None:
            on_finish_callback()

    def freerun_async(self, duration_sec : float = float("+inf"), on_finish_callback = None):
        """ Asynchronously run the simulation from a parallel thread.

        Parameters
        ----------
        duration_sec : float, optional
            Run the environment for this duration (in case of simulation, this is simulated time).
             This can be preempted by stop_freerun_async(). By default float("+inf")
        """
        # ggLog.info(f"Freerun async({duration_sec})")
        with self._running_freerun_async_lock:
            # ggLog.info(f"Freerun async acquired lock")
            self._freerun_async_duration_sec = duration_sec
            self._freerun_async_timeout = self.getEnvTimeFromStartup() + duration_sec
            self._running_freerun_async = True
            self._freerun_async_thread = Thread(target=self.freerun_async_loop, args=[on_finish_callback])
            self._freerun_async_thread.start()

    def wait_freerun_async(self):
        with self._running_freerun_async_lock:
            if self._freerun_async_thread is None:
                return
        self._freerun_async_thread.join()

    def stop_freerun_async(self):        
        with self._running_freerun_async_lock:
            if not self._running_freerun_async:
                return
            self._running_freerun_async = False

    @abstractmethod
    def build_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of controller"""
        raise NotImplementedError()
    

    @abstractmethod
    def destroy_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of controller"""
        raise NotImplementedError()
