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
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional, Mapping
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
from typing_extensions import override
from abc import ABC, abstractmethod
from adarl.utils.spaces import gym_spaces
from adarl.envs.vec.VecRunnerInterface import VecRunnerInterface, ObsType


class VecRunnerWrapper(VecRunnerInterface[ObsType], Generic[ObsType]):

    spec = None

    def __init__(self,
                 runner : VecRunnerInterface[ObsType]):
        self._runner = runner
        super().__init__(   num_envs = runner.num_envs,
                            autoreset = runner.autoreset,
                            vec_observation_space = runner.vec_observation_space,
                            vec_action_space = runner.vec_action_space,
                            vec_reward_space = runner.vec_reward_space,
                            single_observation_space = runner.single_observation_space,
                            single_action_space = runner.single_action_space,
                            single_reward_space = runner.single_reward_space,
                            info_space = runner.info_space,
                            ui_render_envs_indexes=runner.ui_render_envs_indexes,
                            th_device = runner.th_device)
        self._runner.add_on_ep_end_callback(self._on_episode_end)


    @override
    def step(self, actions, autoreset : bool | None = None) -> Tuple[ ObsType,
                                                        ObsType,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        TensorTree[th.Tensor],
                                                        TensorTree[th.Tensor]]:
        return self._runner.step(actions = actions, autoreset=autoreset)

    @override

    def reinit_envs(self,   reinit_envs : th.Tensor,
                            terminateds : th.Tensor,
                            truncateds : th.Tensor,
                            last_observations : ObsType,
                            last_actions : th.Tensor,
                            last_infos : TensorTree[th.Tensor],
                            last_rewards : th.Tensor):
        return self._runner.reinit_envs(reinit_envs_mask=reinit_envs,
                                        last_terminateds=terminateds,
                                        last_truncateds=truncateds,
                                        last_observations=last_observations,
                                        last_actions=last_actions,
                                        last_infos=last_infos,
                                        last_rewards=last_rewards)

    @override    
    def reset(self, seed = None, options = {}) -> tuple[ObsType, TensorTree[th.Tensor]]:
        return self._runner.reset(seed=seed,options=options)

    @override
    def get_ui_renderings(self) -> list[th.Tensor]:
        return self._runner.get_ui_renderings()

    @override
    def close(self):
        return self._runner.close()

    @override
    def get_max_episode_steps(self) -> th.Tensor:
        return self._runner.get_max_episode_steps()