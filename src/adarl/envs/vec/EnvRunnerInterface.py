# import traceback
from __future__ import annotations
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional, Mapping, Callable, Protocol, Union
from adarl.envs.vec.BaseVecEnv import BaseVecEnv
import adarl.utils.utils
import torch as th
from adarl.utils.tensor_trees import TensorTree
from typing_extensions import override, final
from abc import ABC, abstractmethod
from adarl.utils.spaces import gym_spaces, ThBox, ThDict

ObsType = TypeVar("ObsType", bound=Union[   Mapping[Tuple[str,...], th.Tensor],
                                            Mapping[str, th.Tensor],
                                            Mapping[Union[str, Tuple[str,...]], th.Tensor]])


class EnvRunnerInterface(ABC, Generic[ObsType]):

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
                 vec_observation_space : ThDict,
                 vec_action_space : ThBox,
                 vec_reward_space : ThBox,
                 info_space : gym_spaces.Dict,
                 single_observation_space : ThDict,
                 single_action_space : ThBox,
                 single_reward_space : ThBox,
                 autoreset : bool,
                 ui_render_envs_indexes : th.Tensor,
                 th_device : th.device):
        """Initialize the Env Runner

        Parameters
        ----------
        num_envs : int
            The number of parallel environments.
        vec_observation_space : ThDict
            The batched observation space for all environments.
        vec_action_space : ThBox
            The batched action space for all environments.
        vec_reward_space : ThBox
            The batched reward space for all environments.
        info_space : gym_spaces.Dict
            The info space for all environments.
        single_observation_space : ThDict
            The observation space for a single environment.
        single_action_space : ThBox
            The action space for a single environment.
        single_reward_space : ThBox
            The reward space for a single environment.
        autoreset : bool
            Whether to automatically reset environments within step() at the end of an episode.
        ui_render_envs_indexes : th.Tensor
            The indexes of the environments to render as UI.
        th_device : th.device
            The device to use for PyTorch tensors.
        """
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
    def step(self, actions : th.Tensor, autoreset : bool | None = None) -> Tuple[ ObsType,
                                                        ObsType,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        TensorTree[th.Tensor],
                                                        TensorTree[th.Tensor],
                                                        th.Tensor]:
        """Step the environment, and auto-reset it if necessary

        Parameters
        ----------
        actions : th.Tensor
            The actions to take in the environment.
        autoreset : bool | None, optional
            Whether to automatically reset environments within step() at the end of an episode, by default None

        Returns
        -------
        Tuple[ ObsType, ObsType, th.Tensor, th.Tensor, th.Tensor, TensorTree[th.Tensor], TensorTree[th.Tensor], th.Tensor]
            A tuple containing:
            - consequent_observations: The observations that are consequence of the actions taken.
            - next_start_observations: The next observations at the start of the next step (differs from consequent_observations if the environment was reset).
            - rewards: The rewards received after taking the actions.
            - terminateds: A boolean tensor indicating which environments have terminated.
            - truncateds: A boolean tensor indicating which environments have been truncated.
            - consequent_infos: A tensor tree containing the info dictionaries that are consequence of the actions taken.
            - next_start_infos: A tensor tree containing the info dictionaries for each environment at the start of the next step (differs from consequent_infos if the environment was reset).
            - envs_ended_mask: A boolean tensor indicating which environments have ended (terminated or truncated).
        """
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
            
    @abstractmethod

    def get_base_env(self) -> BaseVecEnv[ObsType]:
        """Get the underlying adarl base environment

        Returns
        -------
        BaseEnv
            The adarl.BaseEnv object.
        """
        ...