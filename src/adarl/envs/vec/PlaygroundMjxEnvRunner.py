from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence, Tuple, Literal

import gymnasium as gym
import jax
import numpy as np
import torch as th
from gymnasium.vector.utils.spaces import batch_space
from typing_extensions import override

from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface, ObsType
from adarl.utils.spaces import ThBox, ThDict, gym_spaces
from adarl.utils.tensor_trees import TensorTree, clone_tensor_tree, map_tensor_tree
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import isinstance_noimport
import mujoco_playground._src.mjx_env
import mujoco_playground._src.wrapper
import brax.envs.wrappers.training


def _jax_to_torch(tensor):
  import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

  tensor = tpack.from_dlpack(tensor)
  return tensor


def _torch_to_jax(tensor):
  from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel

  tensor = from_dlpack(tensor)
  return tensor

def _make_th_box(shape: tuple[int,...], device: th.device, low : float, high: float) -> ThBox:
    return ThBox(low=low, high=high, shape=shape, dtype=th.float32, torch_device=device)

def _build_observation_space(shape: tuple[int, ...] | Mapping,
                             device: th.device,
                            low: float = float("-inf"),
                            high: float = float("inf")) -> gym_spaces.Space:
    if isinstance(shape, Mapping):
        return ThDict(
            OrderedDict( (key, _build_observation_space(sub_shape, device))
                            for key, sub_shape in shape.items()
            )
        )
    return _make_th_box(shape, device, low=low, high=high)
    

MjpObsType = dict[str, th.Tensor]

class PlaygroundMjxEnvRunner(EnvRunnerInterface[MjpObsType]):
    """Adapter that lets ``EnvRunnerInterface`` clients use plain gym vector envs."""

    def __init__(
        self,
        env: mujoco_playground._src.mjx_env.MjxEnv,
        num_envs: int,
        seed : int ,
        device: th.device | str | None = None,
        ui_render_envs: Sequence[int] | None = None,
        render_camera_name : str = ""
    ) -> None:
        if device is None:
            device = th.device("cpu")
        else:
            device = th.device(device)
        # if isinstance(action_device, str):
        #     if action_device != "numpy":
        #         raise ValueError("action_device string must be 'numpy'")
        # elif not isinstance(action_device, th.device):
        #     raise TypeError("action_device must be 'numpy' or a torch.device instance")
        # self._action_device = action_device
        self._seed = seed
        self._mjp_env = env
        self._jax_rng_key = jax.device_put(jax.random.PRNGKey(self._seed), jax.devices("gpu")[device.index])
        self._rng_reset_key, self._rng_randomization_key = jax.random.split(self._jax_rng_key)
        self._render_camera_name = render_camera_name

        if not isinstance(self._mjp_env, mujoco_playground._src.mjx_env.MjxEnv):
            raise TypeError(f"env must be an instance of mujoco_playground._src.mjx_env.MjxEnv, got {type(self._mjp_env)}")
        
        wrappers = []
        if isinstance(self._mjp_env, mujoco_playground._src.wrapper.Wrapper):
            current_env = self._mjp_env
            while isinstance(current_env, mujoco_playground._src.wrapper.Wrapper):
                wrappers.append(current_env)
                current_env = current_env.env
            wrapper_types = [type(w) for w in wrappers]        
        if mujoco_playground._src.wrapper.BraxAutoResetWrapper not in wrapper_types:
            raise RuntimeError("The provided env must be wrapped in a BraxAutoResetWrapper for GymEnvRunner to work, but that wrapper was not found in the env's wrapper stack.")
        for w in wrappers:
            if isinstance(w, brax.envs.wrappers.training.EpisodeWrapper):
                max_episode_steps = w.episode_length
        
        
        self._mjpenv_autoreset_mode = "only_next_start" # need to implement a better BraxAutoResetWrapper to get the "consequent_in_info" mode working
        if self._mjpenv_autoreset_mode == "consequent_in_info":
            self._consequent_obs_info_key = "final_obs"
            self._consequent_info_info_key = "final_info" # This actually generally isn't there

        if self._mjpenv_autoreset_mode != "consequent_in_info":
            ggLog.warn( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "The mujoco playground environment does not support providing the consequent observation \n"
                        "and info at reset within the step(), but autoreset is enabled.\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                        )

        ui_render_envs = list(ui_render_envs) if ui_render_envs is not None else []


        self.num_envs = num_envs
        action_dim = self._mjp_env.action_size
        obs_size = self._mjp_env.observation_size
        self._max_episode_steps = th.full((self.num_envs,), max_episode_steps, dtype=th.long, device=self.th_device)
        
        single_observation_space = _build_observation_space(obs_size, device)
        if not isinstance(single_observation_space, ThDict):
            raise RuntimeError(f"The base observation space must be a Dict space wrap it to be one if it isn't, right now it's {single_observation_space}")

        vec_observation_space : ThDict = batch_space(single_observation_space, self.num_envs) #type: ignore[assignment]

        single_action_space =  ThBox(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32, torch_device=device)
        vec_action_space : ThBox = batch_space(single_action_space, self.num_envs) #type: ignore[assignment]

        single_reward_space = ThBox(low=float("-inf"),
                                    high=float("inf"),
                                    shape=(1,),
                                    dtype=np.float32,
                                    torch_device=device)
        vec_reward_space : ThBox = batch_space(single_reward_space, self.num_envs) #type: ignore[assignment]

        info_space = gym_spaces.Dict({})
        ui_indexes = th.as_tensor(ui_render_envs, dtype=th.long, device=device)

        super().__init__(
            num_envs=self.num_envs,
            vec_observation_space=vec_observation_space,
            vec_action_space=vec_action_space,
            vec_reward_space=vec_reward_space,
            info_space=info_space,
            single_observation_space=single_observation_space,
            single_action_space=single_action_space,
            single_reward_space=single_reward_space,
            autoreset=True,
            ui_render_envs_indexes=ui_indexes,
            th_device=device,
        )


        self._reset_fn = jax.jit(self._mjp_env.reset)
        self._step_fn = jax.jit(self._mjp_env.step)

        self._reward_shape = single_reward_space.shape
        self._empty_mask = th.zeros((self.num_envs,), dtype=th.bool, device=self.th_device)
        self._last_actions: th.Tensor | None = None

    @override
    def step(self, actions: th.Tensor, autoreset: bool | None = None) -> Tuple[ MjpObsType,
                                                                                MjpObsType,
                                                                                th.Tensor,
                                                                                th.Tensor,
                                                                                th.Tensor,
                                                                                TensorTree[th.Tensor],
                                                                                TensorTree[th.Tensor],
                                                                                th.Tensor]:
        if autoreset is None:
            autoreset = self.autoreset
        if not autoreset:
            raise RuntimeError("Only autoreset=True is supported, as underlying environments only support autoreset")


        actions_jax = _torch_to_jax(actions)
        self.env_state = self._step_fn(self.env_state, actions_jax)
        obs_th  : MjpObsType = map_tensor_tree(self.env_state.obs, _jax_to_torch)
        
        info = self.env_state.info
        done = _jax_to_torch(self.env_state.done).to(self.th_device, non_blocking=self.th_device == "cuda")

        rewards_tensor =     _jax_to_torch(self.env_state.reward).to(self.th_device, non_blocking=self.th_device == "cuda")
        truncateds_tensor =  _jax_to_torch(info["truncation"]).to(self.th_device, non_blocking=self.th_device == "cuda")
        terminateds_tensor = done & ~truncateds_tensor

        if self._mjpenv_autoreset_mode == "only_next_start":
            consequent_obss = obs_th
            consequent_infos = info
            next_start_obs = obs_th
            next_start_infos = info
        else:
            raise NotImplementedError()

        reinit_done = terminateds_tensor | truncateds_tensor

        self._on_episode_end(
            envs_ended_mask=reinit_done,
            last_observations=consequent_obss,
            last_actions=actions,
            last_infos=consequent_infos,
            last_rewards=rewards_tensor,
            last_terminateds=terminateds_tensor,
            last_truncateds=truncateds_tensor,
        )

        return (consequent_obss,
                next_start_obs,
                rewards_tensor,
                terminateds_tensor,
                truncateds_tensor,
                consequent_infos,
                next_start_infos,
                reinit_done)
    
    @override
    def reinit_envs(
        self,
        reinit_envs_mask: th.Tensor,
        last_terminateds: th.Tensor,
        last_truncateds: th.Tensor,
        last_observations: ObsType,
        last_actions: th.Tensor | None,
        last_infos: TensorTree[th.Tensor],
        last_rewards: th.Tensor,
        *,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, TensorTree[th.Tensor]]:
        raise NotImplementedError("")
    #     if options is None:
    #         options = {}

    #     self._on_episode_end(
    #         envs_ended_mask=reinit_envs_mask,
    #         last_observations=last_observations,
    #         last_actions=last_actions,
    #         last_infos=last_infos,
    #         last_rewards=last_rewards,
    #         last_terminateds=last_terminateds,
    #         last_truncateds=last_truncateds,
    #     )

    #     if not th.any(reinit_envs_mask):
    #         return last_observations, last_infos

    #     indices = th.nonzero(reinit_envs_mask, as_tuple=False).squeeze(-1).tolist()
    #     reset_fn = getattr(self._gym_env, "reset_done", None)
    #     if reset_fn is None:
    #         raise RuntimeError("The wrapped gym vector env does not expose reset_done; autoreset cannot work.")

    #     obs, infos = reset_fn(indices=indices, options=options)
    #     obs_tensor = self._to_tensor_dicttree(obs)
    #     infos_tensor = self._to_tensor_dicttree(infos)

    #     obs_batch = self._tree_batch_size(obs_tensor)
    #     if obs_batch is not None and obs_batch != self.num_envs:
    #         next_obs = clone_tensor_tree(last_observations, detach=False)
    #         self._scatter_subset(next_obs, obs_tensor, indices)
    #     else:
    #         next_obs = obs_tensor

    #     info_batch = self._tree_batch_size(infos_tensor)
    #     if info_batch is not None and info_batch != self.num_envs:
    #         next_infos = clone_tensor_tree(last_infos, detach=False)
    #         self._scatter_subset(next_infos, infos_tensor, indices)
    #     else:
    #         next_infos = infos_tensor

    #     return next_obs, next_infos

    # def _tree_batch_size(self, tree: Any) -> int | None:
    #     tensor = self._first_tensor_leaf(tree)
    #     if tensor is None or tensor.ndim == 0:
    #         return None
    #     return tensor.shape[0]

    # def _first_tensor_leaf(self, tree: Any) -> th.Tensor | None:
    #     if isinstance(tree, Mapping):
    #         for value in tree.values():
    #             leaf = self._first_tensor_leaf(value)
    #             if leaf is not None:
    #                 return leaf
    #         return None
    #     if isinstance(tree, th.Tensor):
    #         return tree
    #     return None

    # def _scatter_subset(self, dest: Any, src: Any, indices: Sequence[int]) -> None:
    #     if isinstance(dest, Mapping) and isinstance(src, Mapping):
    #         for key in dest.keys():
    #             self._scatter_subset(dest[key], src[key], indices)
    #         return
    #     if isinstance(dest, th.Tensor) and isinstance(src, th.Tensor):
    #         for local_pos, env_idx in enumerate(indices):
    #             dest[env_idx].copy_(src[local_pos])
    #         return
    #     raise RuntimeError("Cannot scatter observations/infos with mismatched structures")

    @override
    def reset(self, seed: int | Sequence[int] | None = None, options: dict | None = None) -> tuple[MjpObsType, TensorTree[th.Tensor]]:
        if isinstance(seed, Sequence):
            seed = list(seed)
        self._reset_fn(self._rng_reset_key)
        obs_th  : MjpObsType = map_tensor_tree(self.env_state.obs, _jax_to_torch)
        infos_th = {}
        return obs_th, infos_th

    @override
    def get_ui_renderings(self) -> list[th.Tensor]:
        frame = self._mjp_env.render([self.env_state],
                                    camera=self._render_camera_name,
                                    height=480,
                                    width=640,
                                    # scene_option=scene_option,
                                    )
        if isinstance(frame, np.ndarray):
            return [th.as_tensor(frame, device=self.th_device)]
        if isinstance(frame, th.Tensor):
            return [frame.to(self.th_device)]
        if frame is None:
            return [th.empty(0, device=self.th_device)]
        raise RuntimeError(f"Unsupported frame type {type(frame)} returned by gym environment render() method")

    @override
    def close(self) -> None:
        pass

    @override
    def get_max_episode_steps(self) -> th.Tensor:
        return self._max_episode_steps

    @override
    def get_base_env(self) -> gym.Env:
        return self._mjp_env



