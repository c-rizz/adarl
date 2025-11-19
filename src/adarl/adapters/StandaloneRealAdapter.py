#!/usr/bin/env python3
import os
import time
from threading import Lock
from typing import Dict, List, Tuple, Union

# import adarl.utils.beep
import adarl.utils.dbg.ggLog as ggLog
from adarl.adapters.BaseAdapter import BaseAdapter, JointName, LinkName
import torch as th
from typing_extensions import override
import resource

def precise_sleep(delay_sec : float):
    """Tries to sleep a bit more precisely than time.sleep(), but it is still quite bad,
        as we cannot avoid thread switches while sleeping.

    Parameters
    ----------
    delay_sec : float
        Time to sleep for, in seconds
    """
    # usage = resource.getrusage(resource.RUSAGE_SELF)

    target = time.perf_counter_ns() + delay_sec * 1000_000_000
    while time.perf_counter_ns() < target:
        pass

    # newusage = resource.getrusage(resource.RUSAGE_SELF)    
    # prev_switches = usage.ru_nivcsw + usage.ru_nvcsw
    # new_switches = newusage.ru_nivcsw + newusage.ru_nvcsw
    # if new_switches > prev_switches:
    #     ggLog.info(f"precise_sleep of {delay_sec} sec had {new_switches - prev_switches} context switches")
    #     pass


class AlteredClock():
    def __init__(self, realtime_factor : float = 1.0):
        self._realtime_factor = realtime_factor

    def sleep(self, duration_sec : float):
        """Sleep for the specified duration, altered by the realtime factor."""
        duration_sec = duration_sec / self._realtime_factor
        t0 = time.monotonic()
        long_sleep = max(0,duration_sec-0.2)
        if long_sleep>0:
            time.sleep(long_sleep)
        t = time.monotonic()
        while t-t0 < duration_sec:
            precise_sleep(duration_sec-(t-t0)) # can still be quite bad if a thread switch happens while we sleep
            t = time.monotonic()

    def time(self) -> float:
        """Get the current time, altered by the realtime factor."""
        return time.monotonic() * self._realtime_factor



class StandaloneRealAdapter(BaseAdapter):
    def __init__(   self,   stepLength_sec : float = 0.001,
                            walltime_factor : float = 1.0):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__()
        self._stepLength_sec = stepLength_sec

        self._wall_clock = AlteredClock(realtime_factor=walltime_factor)


    def run(self, duration_sec : float):
        self._wall_clock.sleep(duration_sec)

    def step(self) -> float:
        """Wait for the step time to pass."""
        #TODO: it may make sense to keep track of the time spend in the rest of the processing
        tn = self.getEnvTimeFromStartup()
        sleepDuration = self._stepLength_sec - (tn - self._last_step_end_env_time)
        ggLog.info(f"{__class__.__name__} will sleep of {sleepDuration} = {self._stepLength_sec} - ({tn - self._last_step_end_env_time})")
        if sleepDuration <= 0:
            ggLog.warn("Too much time passed since last step call. Cannot respect step frequency, required sleepDuration = "+str(sleepDuration))
        self.run(max(sleepDuration,0))
        t = self.getEnvTimeFromStartup()
        step_duration = t - self._last_step_end_env_time
        self._last_step_end_env_time = t
        return step_duration
    

    def startup(self):
        self._startup_env_time = self._wall_clock.time()
        self._last_step_end_env_time = self.getEnvTimeFromStartup() #Will be overwritten by resetWorld

    @override
    def resetWorld(self):
        # ggLog.info("Average link_state age ="+str(self._linkStateMsgAgeAvg.getAverage()))
        # ggLog.info("Average joint_state age ="+str(self._jointStateMsgAgeAvg.getAverage()))
        # ggLog.info("Average camera image age ="+str(self._cameraMsgAgeAvg.getAverage()))
        # ggLog.info("Average link_state wait ="+str(self._linkMsgWaitAvg.getAverage()))
        # ggLog.info("Average joint_state wait ="+str(self._jointMsgWaitAvg.getAverage()))
        # ggLog.info("Average camera image wait ="+str(self._cameraMsgWaitAvg.getAverage()))
        self._last_step_end_env_time = self.getEnvTimeFromStartup()

    @override
    def initialize_for_episode(self):
        super().initialize_for_episode()
        self._last_step_end_env_time = self.getEnvTimeFromStartup()

    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._wall_clock.time() - self._startup_env_time

    @override
    def build_scenario(self, **kwargs):
        pass

    @override
    def destroy_scenario(self):
        pass

    # @override
    # def getJointsState(self, requestedJoints : Sequence[JointName] | None = None) -> Dict[JointName,JointState] | th.Tensor:
    #     raise NotImplementedError()
    
    # @override
    # def getLinksState(self, requestedLinks : Sequence[LinkName], use_com_pose : bool = False) -> Dict[LinkName,LinkState]:
    #     raise NotImplementedError()
    
    # @override
    # def getRenderings(self, requestedCameras: List[str]) -> Dict[str, Tuple[th.Tensor | float]]:
    #     raise NotImplementedError()

    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        raise NotImplementedError()
    
