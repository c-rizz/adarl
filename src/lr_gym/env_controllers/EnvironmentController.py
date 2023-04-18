"""This file implements the Envitronment controller class, whic is the superclass for all th environment controllers."""
#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any

import sensor_msgs

from lr_gym.utils.utils import JointState
from lr_gym.utils.utils import LinkState
from abc import ABC, abstractmethod
from threading import Thread, RLock
import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog


class EnvironmentController(ABC):
    """This class allows to control the execution of a simulation.

    It is an abstract class, it is meant to be extended with sub-classes for specific simulators
    """

    def __init__(   self):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        self._running_freerun_async_lock = RLock()
        self._running_freerun_async = False
        self.__lastResetTime = 0
        self.setJointsToObserve([])
        self.setLinksToObserve([])
        self.setCamerasToObserve([])

    def setJointsToObserve(self, jointsToObserve : List[Tuple[str,str]]):
        """Set which joints should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        jointsToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, joint_name)

        """
        self._jointsToObserve = jointsToObserve


    def setLinksToObserve(self, linksToObserve : List[Tuple[str,str]]):
        """Set which links should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        linksToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, link_name)

        """
        self._linksToObserve = linksToObserve

    def setCamerasToObserve(self, camerasToRender : List[str] = []):
        """Set which camera should be rendered after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        camerasToRender : List[str]
            List of the names of the cameras

        """
        self._camerasToObserve = camerasToRender


    @abstractmethod
    def startController(self):
        """Start up the controller. This must be called after setCamerasToObserve, setLinksToObserve and setJointsToObserve."""
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
    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[np.ndarray, float]]:
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
    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
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
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
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

    
    def getEnvSimTimeFromStart(self) -> float:
        """Deprecated"""
        return self.getEnvTimeFromStartup()

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


    def freerun_async_loop(self):
        # ggLog.info(f"Freerun async loop")
        should_run = True
        t_remaining = 1 # Always do at least one step # self._freerun_async_timeout - self.getEnvTimeFromStartup()
        while should_run and t_remaining > 0:
            # ggLog.info(f"Freerunning")
            self.freerun(duration_sec = min(0.2,t_remaining))
            with self._running_freerun_async_lock:
                should_run = self._running_freerun_async
            t_remaining =  self._freerun_async_timeout - self.getEnvTimeFromStartup()
        with self._running_freerun_async_lock:
            self._running_freerun_async = False

    def freerun_async(self, duration_sec : float = float("+inf")):
        # ggLog.info(f"Freerun async({duration_sec})")
        with self._running_freerun_async_lock:
            # ggLog.info(f"Freerun async acquired lock")
            self._freerun_async_duration_sec = duration_sec
            self._freerun_async_timeout = self.getEnvTimeFromStartup() + duration_sec
            self._running_freerun_async = True
            self._freerun_async_thread = Thread(target=self.freerun_async_loop)
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
        self._freerun_async_thread.join()

    @abstractmethod
    def build_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of controller"""
        raise NotImplementedError()
    

    @abstractmethod
    def destroy_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of controller"""
        raise NotImplementedError()
