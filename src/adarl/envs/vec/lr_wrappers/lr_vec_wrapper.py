from abc import ABC, abstractmethod
import adarl.utils.spaces as spaces
import numpy as np
import torch as th
from typing import final, TypeVar, Mapping, Generic
from gymnasium.vector.utils.spaces import batch_space
import adarl.utils.dbg.ggLog as ggLog
from adarl.envs.vec.BaseVecEnv import BaseVecEnv
from typing_extensions import override

State = Mapping[str | tuple[str,...], th.Tensor]
Observation = TypeVar("Observation", bound=Mapping[str | tuple[str,...], th.Tensor])
Action = th.Tensor

class LrVecWrapper(BaseVecEnv[Observation]):

    def __init__(self,
                 env : BaseVecEnv[Observation]):
        self.env = env
        super().__init__(   num_envs=env.num_envs,
                            single_action_space=env.single_action_space,
                            single_observation_space=env.single_observation_space,
                            single_reward_space=env.single_reward_space,
                            single_state_space=env.single_action_space,
                            info_space=env.info_space,
                            th_device=env._th_device,
                            metadata = env.metadata,
                            max_episode_steps = env._max_ep_steps,
                            seed = env._input_seed,
                            obs_dtype = env._obs_dtype,
                            build_and_initialize_ep=False)
    
    @override
    def initialize_episodes(self, vec_mask: th.Tensor | None = None, options: dict = ...):
        return self.env.initialize_episodes(vec_mask, options)


    @override
    def get_ep_step_counter(self):
        return self.env.get_ep_step_counter()

    @override
    def _initialize_episodes(self, vec_mask: th.Tensor | None = None, options: dict = ...):
        return self.env._initialize_episodes(vec_mask, options)

    @override
    def _build(self):
        return self.env._build()

    @override
    def reset(self):
        return self.env.reset()

    @override
    def get_states(self) -> dict[str, th.Tensor]:
        return self.env.get_states()

    @override
    def get_observations(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> Observation:
        return self.env.get_observations(states)

    @override
    def submit_actions(self, actions: th.Tensor):
        return self.env.submit_actions(actions)

    @override
    def post_step(self):
        return self.env.post_step()

    @override
    def step(self):
        return self.env.step()

    @override
    def get_times_since_build(self) -> th.Tensor:
        return self.env.get_times_since_build()
    
    @override
    def get_times_since_ep_start(self) -> th.Tensor:
        return self.env.get_times_since_ep_start()

    @override
    def are_states_timedout(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> th.Tensor:
        return self.env.are_states_timedout(states)

    
    @override
    def are_states_terminal(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> th.Tensor:
        return self.env.are_states_terminal(states)
    
    @override
    def compute_rewards(self, states: Mapping[str | tuple[str, ...], th.Tensor], sub_rewards_return: dict | None = None) -> th.Tensor:
        return self.env.compute_rewards(states, sub_rewards_return)

    @override
    def get_ui_renderings(self, vec_mask: th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        return self.env.get_ui_renderings(vec_mask)

    @override
    def get_infos(self, state, labels: dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        return self.env.get_infos(state, labels)

    @override
    def get_max_episode_steps(self) -> th.Tensor:
        return self.env.get_max_episode_steps()
    
    @override
    def set_max_episode_steps(self, max_episode_steps: th.Tensor):
        return self.env.set_max_episode_steps(max_episode_steps)

    @override
    def close(self):
        return self.env.close()

    @override
    def set_seeds(self, seeds: th.Tensor):
        return self.env.set_seeds(seeds)

    @override
    def get_seeds(self):
        return self.env.get_seeds()

    @override
    def _thtens(self, tensor : th.Tensor):
        return self.env._thtens(tensor)

    @override
    def _thzeros(self, size: tuple[int, ...]):
        return self.env._thzeros(size)

    @override
    def _thrand(self, size: tuple[int, ...]):
        return self.env._thrand(size)