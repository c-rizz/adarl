#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""


import numpy as np
from typing import Tuple, Dict, Any, Sequence
from adarl.envs.BaseEnv import BaseEnv
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.spaces as spaces

class RandomEnv(BaseEnv):

    action_space = None
    observation_space = None
    reward_space = spaces.gym_spaces.Box(low=np.array([float("-inf")]), high=np.array([float("+inf")]), dtype=np.float32)
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 action_space,
                 observation_space,
                 reward_space,
                 start_state = 0,
                 maxStepsPerEpisode : int = 500,
                 is_timelimited : bool = True):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_space = reward_space

        self._state = hash(start_state)
        self._rng = np.random.default_rng(seed = np.abs(self._state))
        self._action = 0

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         startSimulation = False,
                         simulationBackend = None,
                         is_timelimited=is_timelimited)


    def submitAction(self, action) -> None:
        self._action = np.array(action)

    def _sample_space(self, space, rng = None):
        if rng == None:
            rng = self._rng
        if isinstance(space,spaces.gym_spaces.Box):
            if np.issubdtype(space.dtype, np.floating):
                return rng.random(size=space.shape, dtype=space.dtype)*(space.high - space.low)+space.low
            elif np.issubdtype(space.dtype, np.integer):
                return rng.integers(low = space.low, high=space.high, size = space.shape, dtype= space.dtype)

        elif isinstance(space, spaces.gym_spaces.Dict):
            return {k : self._sample_space(v) for k,v in space.spaces.items()}
        else:
            raise NotImplementedError(f"Unsupported space {space}")


    def computeReward(self, previousState, state, action, env_conf = None) -> float:
        rew =  self._sample_space(self.reward_space)
        ggLog.info(f"reward = {rew}")
        return rew

    def getObservation(self, state) -> np.ndarray:
        return self._sample_space(self.observation_space)

    def getState(self) -> Sequence:
        return self._state


    def performStep(self) -> None:
        super().performStep()
        self._state += hash(self._action.data.tobytes())
        self._rng = np.random.default_rng(seed = np.abs(self._state))

    def performReset(self, options = {}) -> None:
        self._state = self._rng.integers(-1000000000,1000000000)


    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        return np.zeros(shape=(32,32), dtype=np.float32)

    def getInfo(self,state=None) -> Dict[Any,Any]:
        return super().getInfo()


    def build(self, backend : str = "gazebo"):
        pass

    def _destroy(self):
        pass

    def getSimTimeSinceBuild(self):
        return self._stepCounter

    def close(self):
        self._destroy()
