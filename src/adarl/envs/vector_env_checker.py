from __future__ import annotations
import gymnasium as gym
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
        if not is_all_finite(observation):
            raise RuntimeError(f"Non-finite values in obs {self._last_obs}")
        if not is_all_bounded(observation, min=-100, max=100):
            raise RuntimeError(f"Values over 100 in obs {self._last_obs}")
        if not is_all_finite(reward):
            raise RuntimeError(f"Non-finite values in reward {reward}")
        if not is_all_bounded(reward, min=-100, max=100):
            raise RuntimeError(f"Values over 100 in reward {reward}")
        return observation, reward, terminated, truncated, infos
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        observation, info = self.env.reset(seed=seed, options=options)
        if not is_all_finite(observation):
            raise RuntimeError(f"Non-finite values in obs {observation}")
        if not is_all_bounded(observation, min=-10, max=10):
            raise RuntimeError(f"Values over 100 in obs {observation}")
        return observation, info