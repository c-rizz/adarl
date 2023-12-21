#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""


from lr_gym.envs.LrWrapper import LrWrapper
from lr_gym.envs.BaseEnv import BaseEnv
import lr_gym.utils.spaces as spaces

class ObsToDict(LrWrapper):

    def __init__(self,
                 env : BaseEnv,
                 key : str = "obs"):
        super().__init__(env=env)
        self._key = key
        self.observation_space = spaces.gym_spaces.Dict({key: env.observation_space})
        self.action_space = env.action_space
        self.metadata = env.metadata

    def getObservation(self, state):
        obs = self.env.getObservation(state)
        return {self._key : obs}