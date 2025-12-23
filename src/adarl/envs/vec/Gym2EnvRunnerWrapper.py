from __future__ import annotations

import numpy as np
import torch as th
from typing import Mapping, Tuple, Union, TypeVar, Generic
from typing_extensions import override
from gymnasium.vector.utils.spaces import batch_space

from adarl.envs.vec.BaseVecEnv import BaseVecEnv
from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.EnvRunnerWrapper import EnvRunnerWrapper
from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface
from adarl.utils.spaces import ThBox, ThDict, gym_spaces
from adarl.utils.tensor_trees import TensorTree

ObsType = TypeVar("ObsType", bound=Mapping[Union[str, Tuple[str, ...]], th.Tensor])


class _RslRlVecEnvAdapter(BaseVecEnv[ObsType]):
    """Adapter turning an RSL-RL VecEnv into an adarl BaseVecEnv."""

    def __init__(
        self,
        vec_env,
        obs_key: str = "obs",
        privileged_obs_key: str = "privileged_obs",
        max_episode_steps: int | None = None,
        info_space: gym_spaces.Dict | None = None,
        th_device: th.device | None = None,
    ):
        self._vec_env = vec_env
        self._obs_key = obs_key
        self._privileged_obs_key = privileged_obs_key
        self._obs_is_dict = isinstance(getattr(vec_env, "observation_space", None), gym_spaces.Dict)
        device = self._pick_device(th_device)
        action_space = self._space_to_thbox(self._get_env_space("action_space"), device)
        observation_space = self._prepare_obs_space(self._get_env_space("observation_space"), device)
        reward_space = ThBox(
            low=np.array([float("-inf")], dtype=np.float32),
            high=np.array([float("+inf")], dtype=np.float32),
            torch_device=device,
        )
        if info_space is None:
            info_space = gym_spaces.Dict({})
        self._dt = self._infer_dt()
        max_steps = self._infer_max_episode_steps(max_episode_steps)
        super().__init__(
            num_envs=self._vec_env.num_envs,
            single_action_space=action_space,
            single_observation_space=observation_space,
            single_state_space=observation_space,
            info_space=info_space,
            th_device=device,
            single_reward_space=reward_space,
            metadata=getattr(self._vec_env, "metadata", {}),
            max_episode_steps=max_steps,
            build_and_initialize_ep=False,
        )
        self._pending_actions: th.Tensor | None = None
        self._last_obs: dict[str, th.Tensor] | None = None
        self._last_privileged_obs: th.Tensor | None = None
        self._last_rewards = th.zeros((self.num_envs, 1), device=device, dtype=th.float32)
        self._last_terminated = th.zeros((self.num_envs,), device=device, dtype=th.bool)
        self._last_truncated = th.zeros_like(self._last_terminated)
        self._last_infos: dict[str, th.Tensor] = {}
        self._states_initialized = False
        self._last_auto_reset_mask = th.zeros((self.num_envs,), device=device, dtype=th.bool)
        self._sim_times = th.zeros((self.num_envs,), device=device, dtype=th.float32)
        self._total_sim_time = th.zeros((self.num_envs,), device=device, dtype=th.float32)

    def _pick_device(self, override_device: th.device | None) -> th.device:
        if override_device is not None:
            return override_device
        if hasattr(self._vec_env, "device"):
            return th.device(self._vec_env.device)
        if hasattr(self._vec_env, "sim_device"):
            return th.device(self._vec_env.sim_device)
        return th.device("cpu")

    def _space_to_thbox(self, space: gym_spaces.Space, device: th.device) -> ThBox:
        if isinstance(space, ThBox):
            return space
        if isinstance(space, gym_spaces.Box):
            return ThBox(low=space.low, high=space.high, dtype=space.dtype, torch_device=device)
        raise NotImplementedError(f"Unsupported space type {type(space)}")

    def _prepare_obs_space(self, observation_space: gym_spaces.Space, device: th.device) -> ThDict:
        if isinstance(observation_space, gym_spaces.Dict):
            subspaces: dict[str, gym_spaces.Space] = {}
            for k, v in observation_space.spaces.items():
                if isinstance(v, (gym_spaces.Box, ThBox)):
                    subspaces[k] = self._space_to_thbox(v, device)
                else:
                    raise NotImplementedError(f"Unsupported observation subspace type {type(v)} for key '{k}'")
            return ThDict(subspaces)  # type: ignore[arg-type]
        if isinstance(observation_space, (gym_spaces.Box, ThBox)):
            return ThDict({self._obs_key: self._space_to_thbox(observation_space, device)})
        raise NotImplementedError(f"Unsupported observation space type {type(observation_space)}")

    def _get_env_space(self, attr_name: str) -> gym_spaces.Space:
        if hasattr(self._vec_env, attr_name):
            return getattr(self._vec_env, attr_name)
        raise NotImplementedError(f"RSL-RL VecEnv is missing required attribute '{attr_name}'")

    def _infer_dt(self) -> float:
        if hasattr(self._vec_env, "dt"):
            return float(getattr(self._vec_env, "dt"))
        if hasattr(self._vec_env, "control_dt"):
            return float(getattr(self._vec_env, "control_dt"))
        sim_params = getattr(self._vec_env, "sim_params", None)
        if sim_params is not None and hasattr(sim_params, "dt"):
            return float(sim_params.dt)
        return 1.0

    def _infer_max_episode_steps(self, max_episode_steps: int | None) -> int:
        if max_episode_steps is not None:
            return int(max_episode_steps)
        for attr in ("max_episode_length", "max_episode_steps", "episode_length"):
            if hasattr(self._vec_env, attr):
                value = getattr(self._vec_env, attr)
                if isinstance(value, th.Tensor):
                    return int(value.item())
                if isinstance(value, (int, float)):
                    return int(value)
        raise NotImplementedError("Cannot infer max episode length from the provided VecEnv")

    def _as_tensor(self, data) -> th.Tensor:
        return th.as_tensor(data, device=self._th_device, dtype=self._obs_dtype)

    def _wrap_obs(self, obs) -> dict[str, th.Tensor]:
        if self._obs_is_dict:
            if not isinstance(obs, Mapping):
                raise RuntimeError("Expected dict-like observations from VecEnv")
            return {k: self._as_tensor(v) for k, v in obs.items()}
        return {self._obs_key: self._as_tensor(obs)}

    def _mask_to_indices(self, vec_mask: th.Tensor | None) -> th.Tensor | None:
        if vec_mask is None:
            return None
        if vec_mask.numel() == 0:
            return th.empty((0,), device=self._th_device, dtype=th.long)
        return th.nonzero(vec_mask, as_tuple=False).squeeze(-1)

    def _extract_obs_from_result(self, result) -> tuple[dict[str, th.Tensor], th.Tensor | None]:
        if result is None:
            raise RuntimeError("Reset/step returned None, cannot build observations")
        if isinstance(result, tuple):
            if len(result) == 1:
                obs = result[0]
                priv = None
            elif len(result) == 2:
                obs, maybe_priv = result
                priv = maybe_priv if isinstance(maybe_priv, (np.ndarray, th.Tensor, Mapping)) else None
            elif len(result) >= 4:
                obs = result[0]
                maybe_priv = result[1]
                priv = maybe_priv if isinstance(maybe_priv, (np.ndarray, th.Tensor, Mapping)) else None
            else:
                raise NotImplementedError("Unsupported reset/step return format")
        else:
            obs, priv = result, None
        return self._wrap_obs(obs), self._as_tensor(priv) if priv is not None else None

    def _current_observation_from_env(self) -> tuple[dict[str, th.Tensor], th.Tensor | None]:
        if hasattr(self._vec_env, "get_observations"):
            obs = self._vec_env.get_observations()
            return self._extract_obs_from_result(obs)
        if hasattr(self._vec_env, "get_state"):
            obs = self._vec_env.get_state()
            return self._extract_obs_from_result(obs)
        raise NotImplementedError("VecEnv does not expose a way to fetch observations after reset")

    def _format_info(self, info: Mapping) -> dict[str, th.Tensor]:
        formatted: dict[str, th.Tensor] = {}
        for key, value in info.items():
            tens = th.as_tensor(value, device=self._th_device)
            if tens.numel() == 1:
                tens = tens.expand((self.num_envs,))
            elif tens.shape[0] != self.num_envs:
                raise RuntimeError(f"Info entry '{key}' has incompatible shape {tens.shape}")
            formatted[key] = tens
        return formatted

    def _update_from_reset(self, reset_result, vec_mask: th.Tensor | None):
        obs, priv = (
            self._extract_obs_from_result(reset_result) if reset_result is not None else self._current_observation_from_env()
        )
        mask = self._all_envs if vec_mask is None else vec_mask
        for k, v in obs.items():
            self._last_obs = self._last_obs or {kk: vv.clone() for kk, vv in obs.items()}
            self._last_obs[k][mask] = v[mask]
        self._last_privileged_obs = priv
        self._last_rewards[mask] = 0.0
        self._last_terminated[mask] = False
        self._last_truncated[mask] = False
        self._sim_times[mask] = 0.0
        self._last_infos = {}
        self._last_auto_reset_mask.zero_()
        self._states_initialized = True

    def _parse_step_result(self, step_result):
        if not isinstance(step_result, tuple):
            raise RuntimeError("VecEnv.step must return a tuple")
        if len(step_result) == 4:
            obs, rewards, dones, infos = step_result
            priv = None
        elif len(step_result) == 5:
            obs, priv, rewards, dones, infos = step_result
        else:
            raise NotImplementedError("Unsupported VecEnv.step return format")
        return obs, priv, rewards, dones, infos

    def submit_actions(self, actions: th.Tensor):
        actions_tensor = th.as_tensor(actions, device=self._th_device, dtype=self._obs_dtype)
        if actions_tensor.shape[0] != self.num_envs:
            raise RuntimeError(f"Expected actions for {self.num_envs} envs, got {actions_tensor.shape[0]}")
        self._pending_actions = actions_tensor

    def _update_from_step(self, step_result):
        obs, priv, rewards, dones, infos = self._parse_step_result(step_result)
        obs_map, priv_tensor = self._extract_obs_from_result((obs, priv))
        rewards_tensor = th.as_tensor(rewards, device=self._th_device, dtype=th.float32).view(self.num_envs, -1)
        dones_tensor = th.as_tensor(dones, device=self._th_device, dtype=th.bool).view(self.num_envs)
        info_dict = infos if isinstance(infos, Mapping) else {}
        truncations = info_dict.get("time_outs", None)
        truncated_tensor = (
            th.as_tensor(truncations, device=self._th_device, dtype=th.bool).view(self.num_envs)
            if truncations is not None
            else th.zeros_like(dones_tensor)
        )
        terminated_tensor = th.logical_and(dones_tensor, th.logical_not(truncated_tensor))
        if self._last_obs is None:
            self._last_obs = obs_map
        else:
            for k, v in obs_map.items():
                self._last_obs[k] = v
        self._last_privileged_obs = priv_tensor
        self._last_rewards = rewards_tensor
        self._last_terminated = terminated_tensor
        self._last_truncated = truncated_tensor
        self._last_infos = self._format_info(info_dict)
        self._last_auto_reset_mask = dones_tensor
        self._states_initialized = True
        self._sim_times += self._dt
        self._total_sim_time += self._dt

    @override
    def step(self):
        if self._pending_actions is None:
            raise RuntimeError("submit_actions must be called before step")
        step_result = self._vec_env.step(self._pending_actions)
        self._update_from_step(step_result)
        th.add(self._ep_step_counter, 1, out=self._ep_step_counter)
        self._tot_step_counter += 1
        self._th_tot_step_counter += 1

    @override
    def _initialize_episodes(self, vec_mask: th.Tensor | None = None, options: dict = {}):
        env_ids = self._mask_to_indices(vec_mask)
        should_skip_reset = vec_mask is not None and th.all(vec_mask == th.logical_and(vec_mask, self._last_auto_reset_mask))
        if should_skip_reset or (env_ids is not None and env_ids.numel() == 0):
            self._sim_times[self._all_envs if vec_mask is None else vec_mask] = 0.0
            self._last_auto_reset_mask.zero_()
            return
        if env_ids is None:
            reset_result = self._vec_env.reset()
        else:
            if hasattr(self._vec_env, "reset_idx"):
                reset_result = self._vec_env.reset_idx(env_ids)
            else:
                raise NotImplementedError("Partial resets are unsupported by the provided VecEnv")
        self._update_from_reset(reset_result, vec_mask)

    @override
    def _build(self):
        # Nothing to build for the adapter.
        return None

    @override
    def reset(self):
        self._vec_env.reset()

    @override
    def get_states(self) -> Mapping[str, th.Tensor]:
        if not self._states_initialized or self._last_obs is None:
            raise RuntimeError("Environment has not been initialized yet")
        state: dict[str, th.Tensor] = {k: v for k, v in self._last_obs.items()}
        state["rewards"] = self._last_rewards
        state["terminateds"] = self._last_terminated
        state["truncateds"] = self._last_truncated
        if self._last_privileged_obs is not None:
            state[self._privileged_obs_key] = self._last_privileged_obs
        return state

    @override
    def get_observations(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> ObsType:
        if not self._states_initialized or self._last_obs is None:
            raise RuntimeError("Environment has not been initialized yet")
        return self._last_obs  # type: ignore[return-value]

    @override
    def compute_rewards(self, states: Mapping[str | tuple[str, ...], th.Tensor], sub_rewards_return: dict | None = None) -> th.Tensor:
        if not self._states_initialized:
            raise RuntimeError("Environment has not been initialized yet")
        return self._last_rewards

    @override
    def are_states_terminal(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> th.Tensor:
        if not self._states_initialized:
            raise RuntimeError("Environment has not been initialized yet")
        return self._last_terminated

    @override
    def are_states_timedout(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> th.Tensor:
        if not self._states_initialized:
            raise RuntimeError("Environment has not been initialized yet")
        return self._last_truncated

    @override
    def get_ui_renderings(self, vec_mask: th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if not hasattr(self._vec_env, "render"):
            raise NotImplementedError("Underlying VecEnv does not implement render")
        frame = self._vec_env.render(mode="rgb_array")
        if frame is None:
            raise NotImplementedError("VecEnv.render returned None")
        img_tensor = th.as_tensor(frame, device=self._th_device)
        if img_tensor.shape[0] == self.num_envs:
            return [img_tensor[vec_mask]], self._sim_times[vec_mask]
        if int(th.count_nonzero(vec_mask)) == 1:
            return [img_tensor.unsqueeze(0)], self._sim_times[vec_mask]
        raise NotImplementedError("VecEnv.render did not return per-environment frames")

    @override
    def get_infos(self, state, labels: dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        return self._last_infos

    @override
    def get_times_since_build(self) -> th.Tensor:
        return self._total_sim_time

    @override
    def get_times_since_ep_start(self) -> th.Tensor:
        return self._sim_times

    @override
    def close(self):
        if hasattr(self._vec_env, "close"):
            self._vec_env.close()

    @override
    def set_seeds(self, seeds: th.Tensor):
        if hasattr(self._vec_env, "seed"):
            self._vec_env.seed(int(seeds[0].item()))
        super().set_seeds(seeds)


class Gym2EnvRunnerWrapper(EnvRunnerWrapper[ObsType], Generic[ObsType]):
    """Wrap an RSL-RL VecEnv into an adarl EnvRunner."""

    def __init__(
        self,
        vec_env,
        autoreset: bool = True,
        render_envs: list[int] | None = None,
        obs_key: str = "obs",
        privileged_obs_key: str = "privileged_obs",
        max_episode_steps: int | None = None,
        info_space: gym_spaces.Dict | None = None,
        th_device: th.device | None = None,
    ):
        adapter = _RslRlVecEnvAdapter(
            vec_env=vec_env,
            obs_key=obs_key,
            privileged_obs_key=privileged_obs_key,
            max_episode_steps=max_episode_steps,
            info_space=info_space,
            th_device=th_device,
        )
        runner = EnvRunner(
            env=adapter,
            autoreset=autoreset,
            render_envs=render_envs or [],
        )
        self._adapter = adapter
        self._vec_env = vec_env
        super().__init__(runner=runner)

    @override
    def get_base_env(self) -> BaseVecEnv[ObsType]:
        return self._adapter


class RslRlEnvRunner(EnvRunnerInterface[ObsType], Generic[ObsType]):
    """EnvRunnerInterface implementation that drives an RSL-RL VecEnv directly."""

    def __init__(
        self,
        vec_env,
        autoreset: bool = True,
        render_envs: list[int] | None = None,
        obs_key: str = "obs",
        privileged_obs_key: str = "privileged_obs",
        max_episode_steps: int | None = None,
        info_space: gym_spaces.Dict | None = None,
        th_device: th.device | None = None,
        env_autoresets: bool = False,
    ):
        self._vec_env = vec_env
        self._obs_key = obs_key
        self._privileged_obs_key = privileged_obs_key
        self._obs_is_dict = isinstance(getattr(vec_env, "observation_space", None), gym_spaces.Dict)
        device = self._pick_device(th_device)
        num_envs = self._get_num_envs()
        single_observation_space = self._prepare_obs_space(self._require_space("observation_space"), device)
        single_action_space = self._space_to_thbox(self._require_space("action_space"), device)
        single_reward_space = ThBox(
            low=np.array([float("-inf")], dtype=np.float32),
            high=np.array([float("+inf")], dtype=np.float32),
            torch_device=device,
        )
        vec_observation_space = batch_space(single_observation_space, n=num_envs)
        vec_action_space = batch_space(single_action_space, n=num_envs)
        vec_reward_space = batch_space(single_reward_space, n=num_envs)
        if info_space is None:
            info_space = gym_spaces.Dict({})
        max_steps = self._infer_max_episode_steps(max_episode_steps)
        ui_render_envs_indexes = th.as_tensor(render_envs or [], device=device, dtype=th.long)
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
            ui_render_envs_indexes=ui_render_envs_indexes,
            th_device=device,
        )
        self._max_episode_steps = th.full((num_envs,), max_steps, device=device, dtype=th.long)
        self._dt = self._infer_dt()
        self._env_autoresets = env_autoresets
        self._last_observations: ObsType | None = None
        self._last_privileged_obs: th.Tensor | None = None
        self._last_rewards = th.zeros((num_envs, single_reward_space.shape[0]), device=device, dtype=th.float32)
        self._last_terminated = th.zeros((num_envs,), device=device, dtype=th.bool)
        self._last_truncated = th.zeros_like(self._last_terminated)
        self._last_infos: dict[str, th.Tensor] = {}
        self._initialized = False
        self._last_actions: th.Tensor | None = None
        self._sim_times = th.zeros((num_envs,), device=device, dtype=th.float32)
        self._total_times = th.zeros((num_envs,), device=device, dtype=th.float32)

    def _pick_device(self, override_device: th.device | None) -> th.device:
        if override_device is not None:
            return override_device
        if hasattr(self._vec_env, "device"):
            return th.device(self._vec_env.device)
        if hasattr(self._vec_env, "sim_device"):
            return th.device(self._vec_env.sim_device)
        return th.device("cpu")

    def _get_num_envs(self) -> int:
        if hasattr(self._vec_env, "num_envs"):
            return int(self._vec_env.num_envs)
        raise NotImplementedError("VecEnv must expose num_envs")

    def _require_space(self, attr_name: str) -> gym_spaces.Space:
        if hasattr(self._vec_env, attr_name):
            return getattr(self._vec_env, attr_name)
        raise NotImplementedError(f"VecEnv is missing required attribute '{attr_name}'")

    def _space_to_thbox(self, space: gym_spaces.Space, device: th.device) -> ThBox:
        if isinstance(space, ThBox):
            return space
        if isinstance(space, gym_spaces.Box):
            return ThBox(low=space.low, high=space.high, dtype=space.dtype, torch_device=device)
        raise NotImplementedError(f"Unsupported space type {type(space)}")

    def _prepare_obs_space(self, observation_space: gym_spaces.Space, device: th.device) -> ThDict:
        if isinstance(observation_space, gym_spaces.Dict):
            subspaces: dict[str, gym_spaces.Space] = {}
            for k, v in observation_space.spaces.items():
                if isinstance(v, (gym_spaces.Box, ThBox)):
                    subspaces[k] = self._space_to_thbox(v, device)
                else:
                    raise NotImplementedError(f"Unsupported observation subspace type {type(v)} for key '{k}'")
            return ThDict(subspaces)  # type: ignore[arg-type]
        if isinstance(observation_space, (gym_spaces.Box, ThBox)):
            return ThDict({self._obs_key: self._space_to_thbox(observation_space, device)})
        raise NotImplementedError(f"Unsupported observation space type {type(observation_space)}")

    def _infer_max_episode_steps(self, max_episode_steps: int | None) -> int:
        if max_episode_steps is not None:
            return int(max_episode_steps)
        for attr in ("max_episode_length", "max_episode_steps", "episode_length"):
            if hasattr(self._vec_env, attr):
                value = getattr(self._vec_env, attr)
                if isinstance(value, th.Tensor):
                    return int(value.item())
                if isinstance(value, (int, float)):
                    return int(value)
        raise NotImplementedError("Cannot infer max episode length from the provided VecEnv")

    def _infer_dt(self) -> float:
        if hasattr(self._vec_env, "dt"):
            return float(getattr(self._vec_env, "dt"))
        if hasattr(self._vec_env, "control_dt"):
            return float(getattr(self._vec_env, "control_dt"))
        sim_params = getattr(self._vec_env, "sim_params", None)
        if sim_params is not None and hasattr(sim_params, "dt"):
            return float(sim_params.dt)
        return 1.0

    def _wrap_obs(self, obs) -> dict[str, th.Tensor]:
        if self._obs_is_dict:
            if not isinstance(obs, Mapping):
                raise RuntimeError("Expected dict-like observations from VecEnv")
            return {k: th.as_tensor(v, device=self.th_device) for k, v in obs.items()}
        return {self._obs_key: th.as_tensor(obs, device=self.th_device)}

    def _extract_obs_priv(self, result) -> tuple[dict[str, th.Tensor], th.Tensor | None]:
        if result is None:
            raise RuntimeError("VecEnv returned None observations")
        if isinstance(result, tuple):
            if len(result) == 1:
                obs = result[0]
                priv = None
            elif len(result) == 2:
                obs, maybe_priv = result
                priv = maybe_priv
            elif len(result) >= 4:
                obs = result[0]
                priv = result[1]
            else:
                raise NotImplementedError("Unsupported reset/step return format")
        else:
            obs, priv = result, None
        obs_map = self._wrap_obs(obs)
        priv_tensor = th.as_tensor(priv, device=self.th_device) if priv is not None else None
        return obs_map, priv_tensor

    def _parse_step_result(self, step_result):
        if not isinstance(step_result, tuple):
            raise RuntimeError("VecEnv.step must return a tuple")
        if len(step_result) == 4:
            obs, rewards, dones, infos = step_result
            priv = None
        elif len(step_result) == 5:
            obs, priv, rewards, dones, infos = step_result
        else:
            raise NotImplementedError("Unsupported VecEnv.step return format")
        return obs, priv, rewards, dones, infos

    def _split_dones(self, dones, infos) -> tuple[th.Tensor, th.Tensor]:
        dones_tensor = th.as_tensor(dones, device=self.th_device, dtype=th.bool).view(self.num_envs)
        info_dict = infos if isinstance(infos, Mapping) else {}
        truncations = info_dict.get("time_outs", info_dict.get("timeouts", None))
        if truncations is None:
            truncated = th.zeros_like(dones_tensor)
        else:
            truncated = th.as_tensor(truncations, device=self.th_device, dtype=th.bool).view(self.num_envs)
        terminated = th.logical_and(dones_tensor, th.logical_not(truncated))
        return terminated, truncated

    def _format_rewards(self, rewards) -> th.Tensor:
        rewards_tensor = th.as_tensor(rewards, device=self.th_device, dtype=th.float32)
        return rewards_tensor.view(self.num_envs, -1)

    def _format_info(self, info: Mapping) -> dict[str, th.Tensor]:
        formatted: dict[str, th.Tensor] = {}
        for key, value in info.items():
            tens = th.as_tensor(value, device=self.th_device)
            if tens.numel() == 1:
                tens = tens.expand((self.num_envs,))
            elif tens.shape[0] != self.num_envs:
                raise RuntimeError(f"Info entry '{key}' has incompatible shape {tens.shape}")
            formatted[key] = tens
        return formatted

    def _call_reset(self, seed=None):
        if not hasattr(self._vec_env, "reset"):
            raise NotImplementedError("VecEnv does not implement reset")
        if seed is None:
            return self._vec_env.reset()
        try:
            return self._vec_env.reset(seed=seed)
        except TypeError:
            if hasattr(self._vec_env, "seed"):
                self._vec_env.seed(seed)
                return self._vec_env.reset()
            raise NotImplementedError("Seeding is not supported by this VecEnv")

    @override
    def step(self, actions, autoreset: bool | None = None) -> Tuple[
        ObsType,
        ObsType,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        TensorTree[th.Tensor],
        TensorTree[th.Tensor],
        th.Tensor,
    ]:
        if not self._initialized:
            self.reset()
        if autoreset is None:
            autoreset = self.autoreset
        actions_tensor = th.as_tensor(actions, device=self.th_device)
        self._last_actions = actions_tensor
        obs, priv, rewards, dones, infos = self._parse_step_result(self._vec_env.step(actions_tensor))
        consequent_observations, priv_obs = self._extract_obs_priv((obs, priv))
        rewards_tensor = self._format_rewards(rewards)
        terminateds, truncateds = self._split_dones(dones, infos)
        consequent_infos = self._format_info(infos) if isinstance(infos, Mapping) else {}
        self._last_observations = consequent_observations
        self._last_privileged_obs = priv_obs
        self._last_rewards = rewards_tensor
        self._last_terminated = terminateds
        self._last_truncated = truncateds
        self._last_infos = consequent_infos
        self._sim_times += self._dt
        self._total_times += self._dt
        reinit_mask = th.logical_or(terminateds, truncateds)
        if autoreset and th.any(reinit_mask):
            if self._env_autoresets:
                next_start_observations = consequent_observations
                next_start_infos = consequent_infos
            else:
                next_start_observations, next_start_infos = self.reinit_envs(
                    reinit_envs_mask=reinit_mask,
                    last_terminateds=terminateds,
                    last_truncateds=truncateds,
                    last_observations=consequent_observations,
                    last_actions=actions_tensor,
                    last_infos=consequent_infos,
                    last_rewards=rewards_tensor,
                )
            reinit_done = reinit_mask
        else:
            next_start_observations = consequent_observations
            next_start_infos = consequent_infos
            reinit_done = th.zeros_like(reinit_mask)
        return (
            consequent_observations,
            next_start_observations,
            rewards_tensor,
            terminateds,
            truncateds,
            consequent_infos,
            next_start_infos,
            reinit_done,
        )

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
    ):
        self._on_episode_end(
            envs_ended_mask=reinit_envs_mask,
            last_observations=last_observations,
            last_actions=last_actions,
            last_infos=last_infos,
            last_rewards=last_rewards,
            last_terminateds=last_terminateds,
            last_truncateds=last_truncateds,
        )
        if not th.any(reinit_envs_mask):
            return self._last_observations, self._last_infos
        if self._last_observations is None:
            raise RuntimeError("reset must be called before reinitializing environments")
        idxs = th.nonzero(reinit_envs_mask, as_tuple=False).squeeze(-1)
        if hasattr(self._vec_env, "reset_idx"):
            reset_result = self._vec_env.reset_idx(idxs)
        elif th.all(reinit_envs_mask):
            reset_result = self._call_reset()
        else:
            raise NotImplementedError("Partial resets are unsupported by this VecEnv")
        obs_map, priv_obs = self._extract_obs_priv(reset_result)
        sample_key = next(iter(obs_map.keys()))
        if obs_map[sample_key].shape[0] == idxs.shape[0]:
            for k, v in obs_map.items():
                self._last_observations[k][idxs] = v
        elif obs_map[sample_key].shape[0] == self.num_envs:
            for k, v in obs_map.items():
                self._last_observations[k][reinit_envs_mask] = v[reinit_envs_mask]
        else:
            raise RuntimeError("Reset returned observations with unexpected batch dimension")
        if priv_obs is not None:
            if self._last_privileged_obs is None:
                self._last_privileged_obs = th.zeros(
                    (self.num_envs,) + priv_obs.shape[1:], device=self.th_device, dtype=priv_obs.dtype
                )
            if priv_obs.shape[0] == idxs.shape[0]:
                self._last_privileged_obs[idxs] = priv_obs
            elif priv_obs.shape[0] == self.num_envs:
                self._last_privileged_obs[reinit_envs_mask] = priv_obs[reinit_envs_mask]
            else:
                raise RuntimeError("Privileged observations have unexpected batch dimension")
        self._last_rewards[reinit_envs_mask] = 0.0
        self._last_terminated[reinit_envs_mask] = False
        self._last_truncated[reinit_envs_mask] = False
        self._sim_times[reinit_envs_mask] = 0.0
        self._last_infos = {}
        return self._last_observations, self._last_infos

    @override
    def reset(self, seed=None, options: dict | None = None) -> tuple[ObsType, TensorTree[th.Tensor]]:
        if options:
            raise NotImplementedError("Reset options are not supported for this VecEnv runner")
        reset_result = self._call_reset(seed=seed)
        observations, priv_obs = self._extract_obs_priv(reset_result)
        self._last_observations = observations
        self._last_privileged_obs = priv_obs
        self._last_rewards.zero_()
        self._last_terminated.zero_()
        self._last_truncated.zero_()
        self._last_infos = {}
        self._sim_times.zero_()
        self._initialized = True
        return observations, {}

    @override
    def get_ui_renderings(self) -> list[th.Tensor]:
        if not hasattr(self._vec_env, "render"):
            raise NotImplementedError("Underlying VecEnv does not implement render")
        frame = self._vec_env.render(mode="rgb_array")
        if frame is None:
            raise NotImplementedError("VecEnv.render returned None")
        img_tensor = th.as_tensor(frame, device=self.th_device)
        selected: list[th.Tensor] = []
        if img_tensor.shape[0] == self.num_envs:
            for idx in self.ui_render_envs_indexes:
                selected.append(img_tensor[int(idx.item())])
            return selected
        if self.ui_render_envs_indexes.numel() == 1:
            selected.append(img_tensor)
            return selected
        raise NotImplementedError("VecEnv.render did not return per-environment frames")

    @override
    def close(self):
        if hasattr(self._vec_env, "close"):
            self._vec_env.close()

    @override
    def get_max_episode_steps(self) -> th.Tensor:
        return self._max_episode_steps

    @override
    def get_base_env(self) -> BaseVecEnv[ObsType]:
        raise NotImplementedError("No adarl BaseVecEnv is available for this runner")
