from __future__ import annotations
import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch as th
from typing import Any, SupportsFloat, Tuple, Dict
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.tensor_trees import unstack_tensor_tree, is_all_finite, is_all_bounded
import copy
import adarl.utils.session as session
import time

class VectorEnvChecker(
    gym.vector.VectorEnvWrapper, gym.utils.RecordConstructorArgs
):
    def __init__(self,  env: VectorEnv,
                        just_warn : bool = False):
        self._just_warn = just_warn
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
        if observation is not None and not is_all_finite(observation):
            msg = f"Non-finite values in obs {observation}"
            if self._just_warn:
                ggLog.warn(msg)
            else:
                raise RuntimeError(msg)
        if observation is not None and not is_all_bounded(observation, min=-100, max=100):
            msg = f"Values over 100 in obs {observation}"
            if self._just_warn:
                ggLog.warn(msg)
            else:
                raise RuntimeError(msg)
        if reward is not None and not is_all_finite(reward):
            msg = f"Non-finite values in reward {reward}"
            if self._just_warn:
                ggLog.warn(msg)
            else:
                raise RuntimeError(msg)
        if reward is not None and not is_all_bounded(reward, min=-100, max=100):
            msg = f"Values over 100 in reward {reward}"
            if self._just_warn:
                ggLog.warn(msg)
            else:
                raise RuntimeError(msg)