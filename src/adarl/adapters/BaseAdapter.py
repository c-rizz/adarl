"""This file implements the base Adapter class, which is the superclass for all environment adapters."""
#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Optional
from typing_extensions import deprecated

from abc import ABC, abstractmethod
from threading import Thread, RLock
import torch as th
from adarl.utils.utils import JointState, LinkState, th_quat_rotate
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
        self._monitored_joints = []
        self._monitored_links = []
        self._monitored_cameras = []

    def set_monitored_joints(self, jointsToObserve : Sequence[tuple[str,str]]):
        """Set which joints should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        jointsToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, joint_name)

        """
        self._monitored_joints = list(jointsToObserve)


    def set_monitored_links(self, linksToObserve : Sequence[tuple[str,str]]):
        """Set which links should be observed after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        linksToObserve : List[Tuple[str,str]]
            List of tuples of the format (model_name, link_name)

        """
        self._monitored_links = list(linksToObserve)

    def set_monitored_cameras(self, camera_names : Sequence[str] = []):
        """Set which camera should be rendered after each simulation step. This information allows for more efficient communication with the simulator.

        Parameters
        ----------
        camera_names : List[str]
            List of the names of the cameras

        """
        self._monitored_cameras = list(camera_names)

    def get_monitored_joints(self):
        return self._monitored_joints
    
    def get_monitored_links(self):
        return self._monitored_links
    
    def get_monitored_cameras(self):
        return self._monitored_cameras

    def startup(self):
        """Start up the adapter."""
        pass
    
    @abstractmethod
    def build_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of adapter"""
        raise NotImplementedError()
    

    @abstractmethod
    def destroy_scenario(self, **kwargs):
        """Build and setup the environment scenario. Should be called by the environment. Arguments depend on the type of adapter"""
        raise NotImplementedError()

    @abstractmethod
    def step(self) -> float:
        """Run a the environment for one step.

        Returns
        -------
        float
            Duration of the step in environment time (in seconds)"""

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
        Dict[str, Tuple[th.Tensor, float]]
            Dict containing the resulting images and the simulation time of their renderings

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
            Torch tensor of size (4,len(monitored_joints),4) containing min,max,average,std of the position,velocity, acceleration
             and effort of each monitored joint. The joints are in the order use din set_monitored_joints.
        """
        ...

    @abstractmethod
    @overload
    def getLinksState(self, requestedLinks : Sequence[LinkName], use_com_pose : bool = False) -> Dict[LinkName,LinkState]:
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


    def get_link_gravity_direction(self, requestedLinks : Sequence[LinkName]) -> th.Tensor:
        ls = self.getLinksState(requestedLinks=requestedLinks)
        gdirs = {ln:th_quat_rotate(th.as_tensor([-1., 0., 0.]), state.pose.orientation_xyzw) for ln,state in ls.items()}
        return th.stack([gdirs[ln] for ln in requestedLinks])


    @abstractmethod
    def resetWorld(self):
        """Reset the environmnet to its start configuration.

        Returns
        -------
        None
            Nothing is returned

        """
        self.__lastResetTime = self.getEnvTimeFromStartup()
        self.initialize_for_episode()

    def initialize_for_episode(self):
        """Performs initializations steps necessary to start an episode.
            After this you can start stepping the environment using step().
        """
        self.__episode_start_env_time = self.getEnvTimeFromStartup()

    @abstractmethod
    def getEnvTimeFromStartup(self) -> float:
        """Get the current time within the simulation."""
        raise NotImplementedError()

    def getEnvTimeFromEpStart(self) -> float:
        """Get the current time within the simulation."""
        return self.getEnvTimeFromStartup() - self.__episode_start_env_time

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

    def get_debug_info(self) -> dict[str,th.Tensor]:
        return {}

    @property
    @deprecated("Just for back compatibility, do not use",)
    def _jointsToObserve(self):
        return self._monitored_joints
    
    @property
    @deprecated("Just for back compatibility, do not use",)
    def _linksToObserve(self):
        return self._monitored_links
    
    @property
    @deprecated("Just for back compatibility, do not use",)
    def _camerasToObserve(self):
        return self._monitored_cameras