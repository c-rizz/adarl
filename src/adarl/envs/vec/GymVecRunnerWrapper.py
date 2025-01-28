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

ObsType = TypeVar("ObsType", bound=Mapping[str | tuple[str,...], th.Tensor])

class GymVecRunnerWrapper(gym.vector.VectorEnv, Generic[ObsType]):

    spec = None

    def __init__(self,
                 runner : EnvRunnerInterface[ObsType],
                 verbose : bool = False,
                 quiet : bool = False,
                 render_cam_index : int = 0,
                 step_v1_x : bool = False):        
        self.vec_runner = runner
        self.action_space = runner.vec_action_space
        self.observation_space = runner.vec_observation_space
        self.single_action_space = runner.single_action_space
        self.single_observation_space = runner.single_observation_space
        if isinstance(runner, EnvRunner):
            self.metadata = runner.get_base_env().metadata
        if isinstance(runner, EnvRunnerWrapper):
            envrunner = runner.get_base_runner()
            if isinstance(envrunner, EnvRunner):
                self.metadata = envrunner.get_base_env().metadata
        else:
            self.metadata = None
        self.use_step_v1_x = step_v1_x
        self._all_vecs = th.ones((self.vec_runner.num_envs,), dtype=th.bool, device=self.vec_runner.th_device)
        self._no_vecs = th.zeros((self.vec_runner.num_envs,), dtype=th.bool, device=self.vec_runner.th_device)
        self._reinit_needed = False
        self._render_cam_index = render_cam_index
        
        if th.any(self.vec_runner.get_max_episode_steps()!=self.vec_runner.get_max_episode_steps()[0]):
            raise RuntimeError(f"All sub environments must have the same max_episode_steps, instead"
                               f" they have: {self.vec_runner.get_max_episode_steps()}")
        self.spec = EnvSpec(id=f"GymEnvWrapper-env-v0_{id(runner)}_{int(time.monotonic()*1000)}",
                            entry_point=None,
                            max_episode_steps=int(self.vec_runner.get_max_episode_steps()[0].item()))
        self._max_episode_steps = self.spec.max_episode_steps # For compatibility, some libraries read this instead of spec

        self._verbose = verbose
        self._quiet = quiet


        self._last_step_end_etime = 0
        self._wtime_spent_stepping_ep = 0

        self._wtime_spent_stepping_tot = 0

        self._last_terminated = th.zeros((self.vec_runner.num_envs,), device=self.vec_runner.th_device, dtype=th.bool)
        self._last_truncated = th.zeros_like(self._last_terminated)

        super().__init__(num_envs=self.vec_runner.num_envs,
                         observation_space=self.vec_runner.single_observation_space,
                         action_space=self.vec_runner.single_action_space)


    def step_v0_29(self, actions) -> Tuple[ObsType, th.Tensor, th.Tensor, th.Tensor, TensorTree[th.Tensor]]:
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
         reinit_done) = self.vec_runner.step(actions)
        next_start_infos["final_observation"] = consequent_observations
        next_start_infos["final_info"] = consequent_info
        # next_start_infos["final_infos"] = consequent_info
        return next_start_observations, reward, terminated, truncated, next_start_infos
    

    def step_v1_0(self, actions) -> Tuple[ObsType, th.Tensor, th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        # Steps even if some envs were supposed to be re-initialized
        # Then, after stepping, it does the re-initialization
        # Transitions that previously had "terminated or truncated" must then not be considered valid transitions
        prev_terms, prev_truncs = self._last_terminated, self._last_truncated
        consequent_observations, next_start_observations, reward, terminated, truncated, info, next_start_infos = self.step_full(actions, autoreset=False)
        next_start_observations, next_start_infos =  self.vec_runner.reinit_envs(prev_terms, prev_truncs)
        return next_start_observations, reward, terminated, truncated, next_start_infos

    @override
    def step(self, actions):
        # ggLog.info(f"Starting step # {self._total_vsteps}")
        if self.use_step_v1_x:
            ret = self.step_v1_0(actions)
        else:
            ret = self.step_v0_29(actions)
        return ret

    @override
    def step_async(self, actions):
        self._async_actions = actions
    
    @override    
    def step_wait(self, **kwargs):
        return self.step(actions=self._async_actions)


    @override
    def reset_async(self, seed: int | adarl.utils.session.List[int] | None = None, options: Dict | None = None):
        pass
    
    @override
    def reset_wait(self, seed: int | adarl.utils.session.List[int] | None = None, options: Dict | None = None):
        return self.reset(seed=seed, options=options)

    @override
    def reset(self, seed = None, options = {}):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        return self.vec_runner.reset(seed=seed, options=options)

    @override
    def render(self, mode : str = 'rgb_array') -> tuple[th.Tensor,...]:
        if mode!="rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        th_images = self.vec_runner.get_ui_renderings()
        ret : list[th.Tensor] = [None]*self.num_envs #type: ignore
        c = 0
        for i in range(self.num_envs):
            if self.vec_runner.ui_render_envs_mask[i]:
                ret[i] = th_images[c]
                c += 1
        return tuple(ret)

    @override
    def close(self):
        """Close the environment.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.vec_runner.close()


    def get_base_env(self) -> BaseVecEnv[ObsType]:
        """Get the underlying adarl base environment

        Returns
        -------
        BaseEnv
            The adarl.BaseEnv object.
        """
        return self.vec_runner.get_base_env()