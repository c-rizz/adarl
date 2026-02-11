from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence, Tuple, Literal

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector.utils.spaces import batch_space
from typing_extensions import override

from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface, ObsType
from adarl.utils.spaces import ThBox, ThDict, gym_spaces
from adarl.utils.tensor_trees import TensorTree, clone_tensor_tree, map_tensor_tree
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import isinstance_noimport

def _resolve_max_episode_steps(gym_env : gym.Env) -> int:
    candidate_specs = []
    env_spec = getattr(gym_env, "spec", None)
    if env_spec is not None:
        candidate_specs.append(env_spec)
    envs = getattr(gym_env, "envs", None)
    if envs is not None:
        candidate_specs.extend(getattr(sub_env, "spec", None) for sub_env in envs)

    max_steps = None
    for spec in candidate_specs:
        if spec is not None and getattr(spec, "max_episode_steps", None):
            max_steps = int(spec.max_episode_steps)
            break
    if max_steps is None:
        max_steps = 0
    return max_steps

def _make_th_box(space: gym_spaces.Space, device: th.device) -> ThBox:
    if isinstance(space, gym_spaces.Box):
        return ThBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype, torch_device=device)
    if isinstance(space, gym_spaces.Discrete):
        low = np.zeros((1,), dtype=np.float32)
        high = np.array([space.n - 1], dtype=np.float32)
        return ThBox(low=low, high=high, shape=(1,), dtype=np.float32, torch_device=device)
    if isinstance(space, gym_spaces.MultiBinary):
        low = np.zeros(space.shape, dtype=np.int8)
        high = np.ones(space.shape, dtype=np.int8)
        return ThBox(low=low, high=high, shape=space.shape, dtype=np.int8, torch_device=device)
    if isinstance(space, gym_spaces.MultiDiscrete):
        low = np.zeros_like(space.nvec, dtype=np.int64)
        high = space.nvec - 1
        return ThBox(low=low, high=high, shape=space.nvec.shape, dtype=np.int64, torch_device=device)
    raise NotImplementedError(f"Unsupported action/observation space type {type(space)}")

def _build_observation_space(
        space: gym_spaces.Space,
        device: th.device,
    ) -> gym_spaces.Space:
    if isinstance(space, gym_spaces.Dict):
        return ThDict(
            OrderedDict( (key, _build_observation_space(sub_space, device))
                            for key, sub_space in space.spaces.items()
            )
        )
    if isinstance(space, gym_spaces.Tuple):
        raise RuntimeError(
            "Tuple observation spaces are not supported in GymEnvRunner. "
            "Please wrap them in a Dict or convert them to a single Box before instantiating the runner."
        )
    return _make_th_box(space, device)
    

class GymEnvRunner(EnvRunnerInterface[ObsType]):
    """Adapter that lets ``EnvRunnerInterface`` clients use plain gym vector envs."""

    def __init__(
        self,
        env: gym.vector.VectorEnv | gym.Env,
        *,
        autoreset: bool = True,
        device: th.device | str | None = None,
        ui_render_envs: Sequence[int] | None = None,
        action_device: Literal["numpy"] | th.device = "numpy",
    ) -> None:
        if device is None:
            device = th.device("cpu")
        else:
            device = th.device(device)
        if isinstance(action_device, str):
            if action_device != "numpy":
                raise ValueError("action_device string must be 'numpy'")
        elif not isinstance(action_device, th.dtype):
            raise TypeError("action_device must be 'numpy' or a torch.dtype instance")
        self._action_device = action_device

        self._gym_env = env
        if isinstance(env, gym.vector.VectorEnv):
            self._gymenv_autoreset_mode = env.metadata.get("autoreset_mode", None)
            if self._gymenv_autoreset_mode is None:
                ggLog.warn("Gym environment does not specify 'autoreset_mode' in metadata.")
            if gym.__version__ >= "1.1.0":
                if self._gymenv_autoreset_mode != gym.vector.AutoresetMode.SAME_STEP:
                    raise RuntimeError(f"Unsupported gym environment autoreset mode {self._gymenv_autoreset_mode}. ")
            elif gym.__version__ >= "1.0.0":
                raise RuntimeError(f"Gym version {gym.__version__} does not support specifying autoreset_mode in metadata, but this is required for GymEnvRunner to function correctly. Please upgrade gym to version 1.1.0 or later.")
            self._gymenv_autoreset_mode = "consequent_in_info"
            self._consequent_obs_info_key = "final_obs"
            self._consequent_info_info_key = "final_info" # This actually generally isn't there
            raw_single_observation_space = env.single_observation_space
            raw_single_action_space = env.single_action_space
            num_envs = env.num_envs
        elif isinstance(env, gym.Env):
            raw_single_observation_space = env.observation_space
            raw_single_action_space = env.action_space
            self._gymenv_autoreset_mode = "only_next_start" # We will only be able to get the first observation of the next episode, not the final observation of the previous episode, since the environment will have already reset by the time we get access to the observation.
            if isinstance_noimport(env, ("isaaclab.envs.DirectRLEnv", "isaaclab.envs.ManagerBasedRLEnv")):
                num_envs = env.num_envs
            else:
                raise RuntimeError("Cannot determine number of environments from a non-vector gym.Env")
        else:
            raise TypeError("env must be an instance of gym.vector.VectorEnv or gym.Env")

        if self._gymenv_autoreset_mode != "consequent_in_info" and autoreset:
            ggLog.warn( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "The gym environment does not support providing the consequent observation \n"
                        "and info at reset within the step(), but autoreset is enabled.\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
                        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                        )

        ui_render_envs = list(ui_render_envs) if ui_render_envs is not None else []

        single_observation_space = _build_observation_space(raw_single_observation_space, device)
        if not isinstance(single_observation_space, ThDict):
            raise RuntimeError(f"The base observation space must be a Dict space wrap it to be one if it isn't, right now it's {base_observation_space}")

        vec_observation_space : ThDict = batch_space(single_observation_space, env.num_envs) #type: ignore[assignment]

        single_action_space = _make_th_box(raw_single_action_space, device)
        vec_action_space : ThBox = batch_space(single_action_space, env.num_envs) #type: ignore[assignment]

        single_reward_space = ThBox(low=np.array([env.reward_range[0]], dtype=np.float32),
                                    high=np.array([env.reward_range[1]], dtype=np.float32),
                                    shape=(1,),
                                    dtype=np.float32,
                                    torch_device=device)
        vec_reward_space : ThBox = batch_space(single_reward_space, env.num_envs) #type: ignore[assignment]

        info_space = gym_spaces.Dict({})
        ui_indexes = th.as_tensor(ui_render_envs, dtype=th.long, device=device)

        super().__init__(
            num_envs=num_envs,
            vec_observation_space=vec_observation_space,
            vec_action_space=vec_action_space,
            vec_reward_space=vec_reward_space,
            info_space=info_space,
            single_observation_space=single_observation_space,
            single_action_space=single_action_space,
            single_reward_space=single_reward_space,
            autoreset=autoreset,
            ui_render_envs_indexes=ui_indexes,
            th_device=device,
        )

        self.spec = getattr(env, "spec", None)
        self._max_episode_steps = th.full((self.num_envs,), _resolve_max_episode_steps(env), dtype=th.long, device=self.th_device)
        self._reward_shape = single_reward_space.shape
        self._reward_dtype = single_reward_space.torch_dtype
        self._empty_mask = th.zeros((self.num_envs,), dtype=th.bool, device=self.th_device)
        self._last_actions: th.Tensor | None = None

    @override
    def step(self, actions: th.Tensor, autoreset: bool | None = None) -> Tuple[ ObsType,
                                                                                ObsType,
                                                                                th.Tensor,
                                                                                th.Tensor,
                                                                                th.Tensor,
                                                                                TensorTree[th.Tensor],
                                                                                TensorTree[th.Tensor],
                                                                                th.Tensor]:
        if autoreset is None:
            autoreset = self.autoreset

        action_tensor = actions if isinstance(actions, th.Tensor) else th.as_tensor(actions, device=self.th_device)
        self._last_actions = action_tensor.detach().clone()
        if self._action_device == "numpy":  # numpy pathway
            env_actions = action_tensor.detach().to("cpu").numpy()
        elif isinstance(self._action_device, th.device):
            env_actions = action_tensor.detach().to(device=self._action_device)
        else:
            raise RuntimeError("Invalid action_device configuration")

        obs, rewards, terminateds, truncateds, infos = self._gym_env.step(env_actions)

        if self._gymenv_autoreset_mode == "consequent_in_info":
            consequent_infos = self._to_tensor_tree(infos.get(self._consequent_info_info_key, {}))
            consequent_obss : ObsType = self._to_tensor_tree(infos[self._consequent_obs_info_key]) #type: ignore[index]
            next_start_obs = self._to_tensor_tree(obs )#type: ignore
            next_start_infos = self._to_tensor_tree(infos)
        else:
            consequent_obss = self._to_tensor_tree(obs) #type: ignore
            consequent_infos = self._to_tensor_tree(infos)
            next_start_obs = consequent_obss
            next_start_infos = consequent_infos

        rewards_tensor = th.as_tensor(rewards, device=self.th_device, dtype=self._reward_dtype)
        terminateds_tensor = th.as_tensor(terminateds, device=self.th_device, dtype=th.bool)
        truncateds_tensor = th.as_tensor(truncateds, device=self.th_device, dtype=th.bool)

        reinit_done = terminateds_tensor | truncateds_tensor

        self._on_episode_end(
            envs_ended_mask=reinit_done,
            last_observations=consequent_obss,
            last_actions=action_tensor,
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
    def reset(self, seed: int | Sequence[int] | None = None, options: dict | None = None) -> tuple[ObsType, TensorTree[th.Tensor]]:
        if isinstance(seed, Sequence):
            seed = list(seed)
        obs, infos = self._gym_env.reset(seed=seed, options=options)
        obs_tensor = self._to_tensor_tree(obs)
        infos_tensor = self._to_tensor_tree(infos)
        return obs_tensor, infos_tensor

    @override
    def get_ui_renderings(self) -> list[th.Tensor]:
        frame = self._gym_env.render()
        if isinstance(frame, np.ndarray):
            return [th.as_tensor(frame, device=self.th_device)]
        if isinstance(frame, th.Tensor):
            return [frame.to(self.th_device)]
        if frame is None:
            return [th.empty(0, device=self.th_device)]
        raise RuntimeError(f"Unsupported frame type {type(frame)} returned by gym environment render() method")

    @override
    def close(self) -> None:
        if hasattr(self._gym_env, "close"):
            self._gym_env.close()

    @override
    def get_max_episode_steps(self) -> th.Tensor:
        return self._max_episode_steps

    @override
    def get_base_env(self) -> gym.Env:
        return self._gym_env

    def _to_tensor_tree(self, value: TensorTree) -> TensorTree[th.Tensor]:
        return map_tensor_tree(value, lambda v: th.as_tensor(v, device=self.th_device))

