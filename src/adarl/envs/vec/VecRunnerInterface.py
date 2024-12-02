#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

# import traceback
from __future__ import annotations
import adarl.utils.dbg.ggLog as ggLog

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional, Mapping, Callable, Protocol
import time
import csv

import adarl

from adarl.envs.vec.BaseVecEnv import BaseVecEnv
import os
import adarl.utils.dbg.ggLog as ggLog

import multiprocessing as mp
import signal
import adarl.utils.session
import adarl.utils.utils
from adarl.utils.wandb_wrapper import wandb_log
from numpy.typing import NDArray
import torch as th
import copy
from adarl.utils.tensor_trees import TensorTree
from typing_extensions import override, final
from abc import ABC, abstractmethod
from adarl.utils.spaces import gym_spaces

ObsType = TypeVar("ObsType", bound=Mapping[str | tuple[str,...], th.Tensor])


class VecRunnerInterface(ABC, Generic[ObsType]):

    class OnEpEndCallbackProtocol(Protocol):
        def __call__(self,  envs_ended_mask : th.Tensor,
                            last_observations : ObsType,
                            last_actions : th.Tensor | None,
                            last_infos : TensorTree[th.Tensor],
                            last_rewards : th.Tensor,
                            last_terminateds : th.Tensor,
                            last_truncateds : th.Tensor):
            ...

    spec = None

    def __init__(self,
                 num_envs : int,
                 vec_observation_space : gym_spaces.Space,
                 vec_action_space : gym_spaces.Space,
                 vec_reward_space : gym_spaces.Space,
                 info_space : gym_spaces.Space,
                 single_observation_space : gym_spaces.Space,
                 single_action_space : gym_spaces.Space,
                 single_reward_space : gym_spaces.Space,
                 autoreset : bool,
                 ui_render_envs_indexes : th.Tensor,
                 th_device : th.device):
        self.num_envs = num_envs
        self.autoreset = autoreset
        self.th_device = th_device
        self.on_ep_end_callbacks = []

        self.vec_observation_space = vec_observation_space
        self.vec_action_space = vec_action_space
        self.vec_reward_space = vec_reward_space
        self.info_space = info_space
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.single_reward_space = single_reward_space
        self.ui_render_envs_indexes = ui_render_envs_indexes
        self.ui_render_envs_mask = th.zeros((self.num_envs,), device=ui_render_envs_indexes.device, dtype=th.bool)
        self.ui_render_envs_mask[self.ui_render_envs_indexes] = True

    @abstractmethod
    def step(self, actions, autoreset : bool | None = None) -> Tuple[ ObsType,
                                                        ObsType,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        TensorTree[th.Tensor],
                                                        TensorTree[th.Tensor],
                                                        th.Tensor]:
        ...

    @abstractmethod
    def reinit_envs(self,   reinit_envs_mask : th.Tensor,
                            last_terminateds : th.Tensor,
                            last_truncateds : th.Tensor,
                            last_observations : ObsType,
                            last_actions : th.Tensor | None,
                            last_infos : TensorTree[th.Tensor],
                            last_rewards : th.Tensor):
        """ Reinitialize the envs indicated in reinit_envs_mask. Also calls the on_episode end callback, giving the provided 
            last_terminateds, last_truncateds, last_observations, last_actions, last_infos, last_rewards

        Parameters
        ----------
        reinit_envs_mask : th.Tensor
            _description_
        last_terminateds : th.Tensor
            _description_
        last_truncateds : th.Tensor
            _description_
        last_observations : ObsType
            _description_
        last_actions : th.Tensor | None
            _description_
        last_infos : TensorTree[th.Tensor]
            _description_
        last_rewards : th.Tensor
            _description_
        """
        ...
    
    @abstractmethod
    def reset(self, seed = None, options = {}) -> tuple[ObsType, TensorTree[th.Tensor]]:
        ...

    @abstractmethod
    def get_ui_renderings(self) -> list[th.Tensor]:
        ...

    @abstractmethod
    def close(self):
        ...

    def __del__(self):
        # This is only called when the object is garbage-collected, so users should
        # still call close themselves, we don't know when garbage collection will happen
        self.close()

    @abstractmethod
    def get_max_episode_steps(self) -> th.Tensor:
        ...

    def add_on_ep_end_callback(self, on_ep_end_callback : OnEpEndCallbackProtocol):
        self.on_ep_end_callbacks.append(on_ep_end_callback)

    @final
    def _on_episode_end(self,   envs_ended_mask : th.Tensor,
                                last_observations : ObsType,
                                last_actions : th.Tensor | None,
                                last_infos : TensorTree[th.Tensor],
                                last_rewards : th.Tensor,
                                last_terminateds : th.Tensor, 
                                last_truncateds : th.Tensor):
        for callback in self.on_ep_end_callbacks:
            callback(   envs_ended_mask = envs_ended_mask,
                        last_observations = last_observations,
                        last_actions = last_actions,
                        last_infos = last_infos,
                        last_rewards = last_rewards,
                        last_terminateds = last_terminateds,
                        last_truncateds = last_truncateds)