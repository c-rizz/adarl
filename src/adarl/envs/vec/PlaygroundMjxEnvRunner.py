from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["MUJOCO_GL"] = "egl"

from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence, Tuple, Literal, Callable

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
from mujoco_playground._src import mjx_env
import mujoco_playground._src.wrapper
import brax.envs.wrappers.training
import brax.envs.base
from jax import numpy as jp
from mujoco import mjx
from mujoco_playground._src.wrapper import Wrapper, BraxDomainRandomizationVmapWrapper
from brax.envs.wrappers import training as brax_training

def _jax_to_torch(tensor):
  import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

  tensor = tpack.from_dlpack(tensor)
  return tensor


def _torch_to_jax(tensor, device: jax.Device | None = None):
  from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel

  tensor = from_dlpack(tensor, device=device)
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
        env: mjx_env.MjxEnv,
        num_envs: int,
        seed : int ,
        device: th.device | str | None = None,
        ui_render_envs: Sequence[int] | None = None,
        render_camera_name : str = "",
        action_device: th.device | str = "cuda",
        randomize_step_timeout_counters: bool = False
    ) -> None:
        if device is None:
            device = th.device("cpu")
        else:
            device = th.device(device)
        self._action_device_th = th.device(action_device)
        if self._action_device_th.index is None:
            self._action_device_th = th.device(self._action_device_th.type, index=0)
        self._action_device_jax = jax.devices("gpu")[self._action_device_th.index] if self._action_device_th.type == "cuda" else jax.devices("cpu")[0]
        self._seed = seed
        self._mjp_env = env
        self._jax_rng_key = jax.device_put(jax.random.PRNGKey(self._seed), jax.devices("gpu")[device.index])
        self._rng_reset_key, self._rng_randomization_key = jax.random.split(self._jax_rng_key)
        self._rng_reset_key = jax.random.split(self._rng_reset_key, num_envs)
        self._rng_randomization_key = jax.random.split(self._rng_randomization_key, num_envs)
        self._render_camera_name = render_camera_name
        self._randomize_stpes_timeout_counters = randomize_step_timeout_counters

        if not isinstance(self._mjp_env, mjx_env.MjxEnv):
            raise TypeError(f"env must be an instance of mujoco_playground._src.mjx_env.MjxEnv, got {type(self._mjp_env)}")
        
        wrappers = []
        wrapper_types = []
        if isinstance(self._mjp_env, mujoco_playground._src.wrapper.Wrapper):
            current_env = self._mjp_env
            while isinstance(current_env, (mujoco_playground._src.wrapper.Wrapper, brax.envs.base.Wrapper)):
                wrappers.append(current_env)
                current_env = current_env.env
            wrapper_types = [type(w) for w in wrappers]
        if BraxAutoResetWrapper_finalobs not in wrapper_types:
            raise RuntimeError("The provided env must be wrapped in a BraxAutoResetWrapper_finalobs for PlaygroundMjxEnvRunner to work, but that wrapper was not found in the env's wrapper stack.")
        if brax.envs.wrappers.training.EpisodeWrapper not in wrapper_types:
            raise RuntimeError("The provided env must be wrapped in a EpisodeWrapper for PlaygroundMjxEnvRunner to work, but that wrapper was not found in the env's wrapper stack.")
        for w in wrappers:
            if isinstance(w, brax.envs.wrappers.training.EpisodeWrapper):
                max_episode_steps = w.episode_length

        ui_render_envs = list(ui_render_envs) if ui_render_envs is not None else []


        self.num_envs = num_envs
        action_dim = self._mjp_env.action_size
        obs_size = self._mjp_env.observation_size
        if isinstance(obs_size, int):
            self._wrap_obs_space = True
            self._obs_dict_wrapping_key = "obs"
            obs_size = {self._obs_dict_wrapping_key: (obs_size,)}
        self._max_episode_steps = max_episode_steps
        self._max_episode_steps_th = th.full((self.num_envs,), max_episode_steps, dtype=th.long, device=device)
        
        single_observation_space = _build_observation_space(obs_size, device)
        
        vec_observation_space : ThDict = batch_space(single_observation_space, self.num_envs) #type: ignore[assignment]

        single_action_space =  ThBox(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32, torch_device=device)
        vec_action_space : ThBox = batch_space(single_action_space, self.num_envs) #type: ignore[assignment]

        single_reward_space = ThBox(low=float("-inf"),
                                    high=float("inf"),
                                    shape=(1,),
                                    dtype=np.float32,
                                    torch_device=device)
        vec_reward_space : ThBox = batch_space(single_reward_space, self.num_envs) #type: ignore[assignment]

        info_space = gym_spaces.Dict({"steps" : ThBox(low=0, high=max_episode_steps, shape=(), dtype=np.int32, torch_device=device)})
        ui_indexes = th.as_tensor(ui_render_envs, dtype=th.int32, device=device)

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


        actions_jax = _torch_to_jax(actions, device=self._action_device_jax)
        self.env_state, consequent_obs = self._step_fn(self.env_state, actions_jax)
        obs = self.env_state.obs
        if not isinstance(obs, Mapping):
            obs = {self._obs_dict_wrapping_key: obs}
        if not isinstance(consequent_obs, Mapping):
            consequent_obs = {self._obs_dict_wrapping_key: consequent_obs}
        next_start_obss_th  : MjpObsType = map_tensor_tree(obs, _jax_to_torch)
        consequent_obss_th : MjpObsType = map_tensor_tree(consequent_obs, _jax_to_torch)
        
        info = self.env_state.info
        done = _jax_to_torch(self.env_state.done).to(self.th_device, non_blocking=self.th_device == "cuda", dtype=th.bool)

        rewards_tensor =     _jax_to_torch(self.env_state.reward).to(self.th_device, non_blocking=self.th_device == "cuda")
        truncateds_tensor =  _jax_to_torch(info["truncation"]).to(self.th_device, non_blocking=self.th_device == "cuda", dtype=th.bool)
        terminateds_tensor = done & ~truncateds_tensor

        reinit_done = terminateds_tensor | truncateds_tensor

        # For now skip the infos, as they aren't really provided in a consistent way
        consequent_infos = {"steps" : _jax_to_torch(info["steps"])}
        next_start_infos = consequent_infos

        # print(f"steps tensor: {consequent_infos['steps']}")

        self._on_episode_end(
            envs_ended_mask=reinit_done,
            last_observations=consequent_obss_th,
            last_actions=actions,
            last_infos=consequent_infos,
            last_rewards=rewards_tensor,
            last_terminateds=terminateds_tensor,
            last_truncateds=truncateds_tensor,
        )

        # ggLog.info(f"step() returning:\n"
        #            f"consequent_obss: {consequent_obss}\n"
        #            f"next_start_obs: {next_start_obs}\n"
        #            f"rewards_tensor: {rewards_tensor}\n"
        #            f"terminateds_tensor: {terminateds_tensor}\n"
        #            f"truncateds_tensor: {truncateds_tensor}\n"
        #            f"consequent_infos: {consequent_infos}\n"
        #            f"next_start_infos: {next_start_infos}\n"
        #            f"reinit_done: {reinit_done}\n"
        #            )

        return (consequent_obss_th,
                next_start_obss_th,
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
        self.env_state = self._reset_fn(self._rng_reset_key)
        if self._randomize_stpes_timeout_counters:
            ggLog.info("Randomizing steps to timeout for envs that were reset")
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            self.env_state.info['steps'] = jax.random.randint(subkey, shape=(self.num_envs,), minval=0, maxval=self._max_episode_steps)
            self.env_state.info['truncation'] = jp.zeros((self.num_envs,), dtype=jp.bool_)
        obs = self.env_state.obs
        if not isinstance(obs, Mapping):
            obs = {self._obs_dict_wrapping_key: obs}
        obs_th  : MjpObsType = map_tensor_tree(obs, _jax_to_torch)
        infos_th = {}
        return obs_th, infos_th

    @override
    def get_ui_renderings(self) -> list[th.Tensor]:
        # frame = np.zeros((len(self.ui_render_envs_indexes), 16, 16, 3), dtype=np.uint8)
        frames = []
        for env_idx in self.ui_render_envs_indexes:
            single_env_state = jax.tree_util.tree_map(lambda x: x[env_idx.item()], self.env_state)
            # ggLog.info(f"Rendering UI for env {env_idx}, single_env_state done, now rendering...")
            frame_hwc = self._mjp_env.render(single_env_state,
                                        camera=self._render_camera_name,
                                        height=480,
                                        width=640,
                                        # scene_option=scene_option,
                                        )
            # ggLog.info(f"Got UI rendering for env {env_idx} with shape {frame_hwc.shape} and dtype {frame_hwc.dtype}")
            frames.append(th.as_tensor(frame_hwc, device=self.th_device))
        frames = th.stack(frames, dim=0) # stack envs
        return [frames] # return as list to be compatible with VecEnvRunner which supports multiple renderings per env, here we just return one

    @override
    def close(self) -> None:
        pass

    @override
    def get_max_episode_steps(self) -> th.Tensor:
        return self._max_episode_steps_th
    
    @override
    def get_max_possible_episode_steps(self) -> int:
        return self._max_episode_steps

    @override
    def get_base_env(self) -> gym.Env:
        return self._mjp_env



class BraxAutoResetWrapper_finalobs(mujoco_playground._src.wrapper.BraxAutoResetWrapper):
    """A custom version of BraxAutoResetWrapper that provides the final observation the step() return when an episode ends."""
    @override
    def step(self, state: mjx_env.State, action: jax.Array) -> tuple[mjx_env.State, mjx_env.Observation]:
        # grab the reset state.
        reset_state = None
        rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f'{self._info_key}_first_data']
            reset_obs = state.info[f'{self._info_key}_first_obs']

        if 'steps' in state.info:
            # reset steps to 0 if done.
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        consequent_obs = state.obs
        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)

        next_info = state.info
        done_count_key = f'{self._info_key}_done_count'
        if self._full_reset and reset_state:
            next_info = jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]

            if 'steps' in next_info:
                next_info['steps'] = state.info['steps']
            preserve_info_key = f'{self._info_key}_preserve_info'
            if preserve_info_key in next_info:
                next_info[preserve_info_key] = state.info[preserve_info_key]

        next_info[done_count_key] += state.done.astype(int)
        next_info[f'{self._info_key}_rng'] = reset_rng

        return state.replace(data=data, obs=obs, info=next_info), consequent_obs
    



def wrap_for_adarl_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> Wrapper:
  """
  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate, should
      equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over
    full_reset: whether to call `env.reset` during `env.step` on done rather
      than resetting to a cached first state. Setting full_reset=True may
      increase wallclock time because it forces full resets to random states.

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if vision:
    env = MadronaWrapper(env, num_vision_envs, randomization_fn)
  elif randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper_finalobs(env, full_reset=full_reset)
  return env