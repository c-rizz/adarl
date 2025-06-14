#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""
from __future__ import annotations
from adarl.envs.vec.BaseVecEnv import BaseVecEnv, Observation
from typing import TypeVar, Generic
from adarl.adapters.BaseVecAdapter import BaseVecAdapter
import adarl.utils.spaces as spaces
import adarl.utils.dbg.ggLog as ggLog
import torch as th
import numpy as np
from typing_extensions import override
import time

EnvAdapterType = TypeVar("EnvAdapterType", bound=BaseVecAdapter)

class ControlledVecEnv(Generic[EnvAdapterType, Observation], BaseVecEnv[Observation]):

    def __init__(self,  single_action_space : spaces.gym_spaces.Space,
                        single_observation_space : spaces.gym_spaces.Space,
                        single_state_space : spaces.gym_spaces.Space,
                        info_space : spaces.gym_spaces.Space,
                        step_duration_sec : float,
                        adapter : EnvAdapterType,
                        th_device : th.device,
                        seed : int = 0,
                        obs_dtype : th.dtype = th.float32,
                        single_reward_space = spaces.gym_spaces.Box(low=np.array([float("-inf")]), high=np.array([float("+inf")]), dtype=np.float32),
                        metadata = {},
                        max_episode_steps : int | th.Tensor = 1000,
                        allow_multiple_steps : bool = False,
                        step_precision_tolerance : float = 0.0,
                        build_and_initialize_ep : bool = False):
        """
        """

        self._adapter : EnvAdapterType = adapter
        num_envs = self._adapter.vec_size()
        self._estimated_env_times = th.zeros(size=(num_envs,), device=th_device, dtype=th.float32) # Estimated from the results of each environmentController.step()
        self._intendedStepLength_sec = step_duration_sec
        self._allow_multiple_steps = allow_multiple_steps
        self._step_precision_tolerance = step_precision_tolerance

        super().__init__(   num_envs = num_envs,
                            single_action_space = single_action_space,
                            single_observation_space = single_observation_space,
                            single_state_space = single_state_space,
                            info_space = info_space,
                            th_device = th_device,
                            single_reward_space = single_reward_space,
                            metadata = metadata,
                            max_episode_steps = max_episode_steps,
                            obs_dtype=obs_dtype,
                            seed=seed,
                            build_and_initialize_ep=build_and_initialize_ep)


    def step(self) -> None:
        self.pre_step()
        estimated_step_duration_sec = 0.0
        adapter_step_count = 0
        t0 = time.monotonic()
        while True: # Do at least one step, then check if we need more
            estimated_step_duration_sec += self._adapter.step()
            adapter_step_count+=1
            residual_step_length = self._intendedStepLength_sec - estimated_step_duration_sec
            if abs(residual_step_length) <= self._step_precision_tolerance:
                break
            else:
                if residual_step_length > 0:
                    if not self._allow_multiple_steps:
                        raise RuntimeError(f"Simulation stepped less than required step length ({estimated_step_duration_sec}<{self._intendedStepLength_sec})\n"
                                           f"Tolerance is {self._step_precision_tolerance}, increase it to ignore these errors")
                else:
                    ggLog.warn( f"Simulation stepped more than required step length ({estimated_step_duration_sec}>{self._intendedStepLength_sec})\n"
                                f"Tolerance is {self._step_precision_tolerance}, increase it to ignore these errors")
                    break
        t1 = time.monotonic()
        th.add(self._ep_step_counter,1,out=self._ep_step_counter)
        self._tot_step_counter+=1
        self._estimated_env_times += estimated_step_duration_sec
        if abs(estimated_step_duration_sec - self._intendedStepLength_sec) > self._step_precision_tolerance:
            ggLog.warn(f"Step duration is different than intended: {estimated_step_duration_sec} != {self._intendedStepLength_sec}")
        self.post_step()
        tf = time.monotonic()
        # ggLog.info(f"ControlledEnv: adapter_step_count = {adapter_step_count} adapter_step = {t1-t0:.6f} (vec_env_fps={1/(t1-t0)*self.num_envs:.2f}, rt={estimated_step_duration_sec*self.num_envs/(t1-t0):.2f}), env_step={tf-t0:.6f} (vec_env_fps={1/(tf-t0)*self.num_envs:.2f}, rt={estimated_step_duration_sec*self.num_envs/(tf-t0):.2f})")

    @override
    def reset(self, options = {}):
        super().reset()
        self._adapter.resetWorld()

    @override
    def initialize_episodes(self, vec_mask: th.Tensor | None = None, options: dict = {}):
        self._estimated_env_times[vec_mask] = 0.0
        super().initialize_episodes(vec_mask, options)
        self._adapter.initialize_for_episode(vec_mask=vec_mask)

    @override
    def get_times_since_ep_start(self) -> th.Tensor:
        return self._estimated_env_times

    @override
    def get_times_since_build(self) -> th.Tensor:
        return th.as_tensor(self._adapter.getEnvTimeFromStartup()).to(device=self._th_device, dtype=th.float32, non_blocking=True).expand((self.num_envs,))
