from __future__ import annotations
import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch as th
from typing import Any, SupportsFloat, Tuple, Dict
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.tensor_trees import unstack_tensor_tree, is_all_finite, is_all_bounded
from adarl.utils.dbg.dbg_checks import dbg_check_finite, dbg_check_bounded
import copy
import adarl.utils.session as session
import time
from packaging.version import Version

if Version(gym.__version__) < Version("1.0.0"):
    VectorWrapper = gym.vector.VectorEnvWrapper
else:
    VectorWrapper = gym.vector.VectorWrapper

class VectorEnvChecker(
    VectorWrapper, gym.utils.RecordConstructorArgs
):
    def __init__(self,  env: VectorEnv,
                        just_warn : bool = False,
                        max_obs_value : float = 255.0,
                        max_rew_value : float = 100.0):
        self._just_warn = just_warn
        self._obs_min = -max_obs_value
        self._obs_max =  max_obs_value
        self._rew_min = -max_rew_value
        self._rew_max =  max_rew_value
        super().__init__(env)

    def step(
        self, action
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Steps through the environment.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)``

        """
        observation, reward, terminated, truncated, infos = self.env.step(action)
        self._check(observation, reward)
        return observation, reward, terminated, truncated, infos
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        observation, info = self.env.reset(seed=seed, options=options)
        self._check(observation, None)
        return observation, info
    
    def _check(self, observation, reward):
        if observation is not None:
            dbg_check_finite(observation, async_assert=True, assert_msg="VectorEnvChecker: Non finite values in observation")
            dbg_check_bounded(observation, min=self._obs_min, max=self._obs_max, async_assert=not self._just_warn, just_warn=self._just_warn, assert_msg="Observation out of bounds")
        if reward is not None:
            dbg_check_finite(reward, async_assert=True, assert_msg="VectorEnvChecker: Non finite values in reward")
            dbg_check_bounded(reward, min = self._rew_min, max = self._rew_max, async_assert=not self._just_warn, just_warn=self._just_warn, assert_msg="Reward out of bounds")