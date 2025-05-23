#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

from adarl.envs.BaseEnv import BaseEnv
from typing import TypeVar, Generic
from adarl.adapters.BaseAdapter import BaseAdapter
import adarl.utils.spaces as spaces
import adarl.utils.dbg.ggLog as ggLog

EnvControllerType = TypeVar("EnvControllerType", bound=BaseAdapter)

class ControlledEnv(Generic[EnvControllerType], BaseEnv):
    """This is a base-class for implementing OpenAI-gym environments using environment controllers derived from BaseAdapter.

    It implements part of the methods defined in BaseEnv relying on an BaseAdapter
    (not all methods are available on non-simulated BaseAdapters like RosAdapter, at least for now).

    The idea is that environments created from this will be able to run on different simulators simply by using specifying
    environmentController objects in the constructor

    You can extend this class with a sub-class to implement specific environments.
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxStepsPerEpisode,
                 stepLength_sec,
                 environmentController : EnvControllerType,
                 action_space : spaces.gym_spaces.Space,
                 observation_space : spaces.gym_spaces.Space,
                 state_space : spaces.gym_spaces.Space,
                 startSimulation : bool = False,
                 is_timelimited : bool = False,
                 allow_multiple_steps : bool = False,
                 step_precision_tolerance : float = 0.0):
        """
        """


        if environmentController is None:
            raise AttributeError("You must specify environmentController")
        self._adapter : EnvControllerType = environmentController
        self._estimatedSimTime = 0.0 # Estimated from the results of each environmentController.step()
        self._intendedStepLength_sec = stepLength_sec
        self._allow_multiple_steps = allow_multiple_steps
        self._step_precision_tolerance = step_precision_tolerance

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         startSimulation = startSimulation,
                         is_timelimited=is_timelimited,
                         observation_space=observation_space,
                         action_space=action_space,
                         state_space=state_space)





    def performStep(self) -> None:
        super().performStep()
        estimatedStepDuration_sec = 0
        while True: # Do at least one step, then check if we need more
            estimatedStepDuration_sec += self._adapter.step()
            if estimatedStepDuration_sec >= self._intendedStepLength_sec - self._step_precision_tolerance:
                break
            elif not self._allow_multiple_steps:
                raise RuntimeError(f"Simulation stepped less than required step length (stepped {estimatedStepDuration_sec} instead of {self._intendedStepLength_sec})\n"
                                   f"If you are running in siumlation chack your environment dt is an exact multiple of your physiscs dt (careful, in binary).\n"
                                   f"In the real you may want to raise step_precision_tolerance")
        self._estimatedSimTime += estimatedStepDuration_sec
        if abs(estimatedStepDuration_sec - self._intendedStepLength_sec) > self._step_precision_tolerance:
            ggLog.warn(f"Step duration is different than intended: {estimatedStepDuration_sec} != {self._intendedStepLength_sec}")



    def performReset(self, options = {}):
        super().performReset(options)
        self._adapter.resetWorld()
        self._estimatedSimTime = 0.0
        self.initializeEpisode(options)


    def getSimTimeSinceBuild(self):
        return self._adapter.getEnvTimeFromStartup()
