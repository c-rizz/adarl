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
from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.EnvRunnerWrapper import EnvRunnerWrapper
from adarl.utils.tensor_trees import map_tensor_tree

ObsType = TypeVar("ObsType", bound=Mapping[str | tuple[str,...], th.Tensor])

def take_first(tensor_tree):
    return map_tensor_tree(tensor_tree, lambda t: t[0])

class GymRunnerWrapper(gym.Env, Generic[ObsType]):

    spec = None

    def __init__(self,
                 runner : EnvRunnerInterface[ObsType],
                 quiet : bool = True):
        
        self.vec_runner = runner
        self.action_space = runner.single_action_space
        self.observation_space = runner.single_observation_space
        if isinstance(runner, EnvRunner):
            self.metadata = runner.get_base_env().metadata
        if isinstance(runner, EnvRunnerWrapper):
            envrunner = runner.get_base_runner()
            if isinstance(envrunner, EnvRunner):
                self.metadata = envrunner.get_base_env().metadata
        else:
            self.metadata = None
        if self.vec_runner.num_envs != 1:
            raise RuntimeError(f"Only 1-env runners are supported")
        self._reinit_needed = False
        
        if th.any(self.vec_runner.get_max_episode_steps()!=self.vec_runner.get_max_episode_steps()[0]):
            raise RuntimeError(f"All sub environments must have the same max_episode_steps, instead"
                               f" they have: {self.vec_runner.get_max_episode_steps()}")
        self.spec = EnvSpec(id=f"GymEnvWrapper-env-v0_{id(runner)}_{int(time.monotonic()*1000)}",
                            entry_point=None,
                            max_episode_steps=int(self.vec_runner.get_max_episode_steps()[0].item()))
        self._max_episode_steps = self.spec.max_episode_steps # For compatibility, some libraries read this instead of spec

        super().__init__()


    def step(self, actions) -> Tuple[ObsType, th.Tensor, th.Tensor, th.Tensor, TensorTree[th.Tensor]]:
        # Steps even if some envs were supposed to be re-initialized
        # Then, after stepping, it does the re-initialization
        # Transitions that previously had "terminated or truncated" must then not be considered valid transitions
        (consequent_observations,
         next_start_observations,
         reward,
         terminated,
         truncated,
         consequent_info,
         next_start_infos,
         reinit_done) = self.vec_runner.step(actions.unsqueeze(0))
        # next_start_infos["final_infos"] = consequent_info
        return (take_first(next_start_observations),
                reward[0],
                terminated[0],
                truncated[0],
                take_first(next_start_infos))
    

    @override
    def reset(self, seed = None, options = {}):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        obss, infos = self.vec_runner.reset(seed=seed, options=options)
        return take_first(obss), take_first(infos)

    @override
    def render(self, mode : str = 'rgb_array') -> th.Tensor:
        if mode!="rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        th_images = self.vec_runner.get_ui_renderings()
        return th_images[0]

    @override
    def close(self):
        """Close the environment.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.vec_runner.close()
