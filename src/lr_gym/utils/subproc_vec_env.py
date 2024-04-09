import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
import time
from lr_gym.utils.shared_env_data import SharedEnvData, SimpleCommander, SharedData
from lr_gym.utils.tensor_trees import space_from_tree, map_tensor_tree

import torch as th
import copy
import lr_gym.utils.mp_helper as mp_helper

def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    simple_commander : SimpleCommander,
    shared_env_data : SharedEnvData,
    env_idx : int
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_observation = None
    observation, reset_info = env.reset()
    reset_info["terminal_observation"] = observation # make it as if we always have terminal_observation
    # print(f"reset_info = {reset_info}")
    info_space = space_from_tree(reset_info)
    reset_info = None
    while True:
        try:
            cmd = simple_commander.wait_command()
            if cmd == b"step":
                action = shared_env_data.wait_actions()[env_idx]
                observation, reward, terminated, truncated, info = env.step(action)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                # info = {}
                info["TimeLimit.truncated"] = truncated and not terminated
                info["terminal_observation"] = observation # always put last observation in info, even if not resetting
                if done:
                    # save final observation where user can get it, then reset
                    reset_observation, reset_info = env.reset()
                    observation = reset_observation # To follow VecEnv's behaviour
                    reset_info["terminal_observation"] = reset_observation  # always put last observation in info
                # To be compatible with VecEnvs the worker in case of reset will
                # put the reset observation in obs and the last step observation in info
                # the reset info is always in reset_info (only updated at each reset)
                info = map_tensor_tree(info, func=lambda x: th.as_tensor(x))
                if reset_info is not None:
                    reset_info = map_tensor_tree(reset_info, func=lambda x: th.as_tensor(x))
                # print(f"observation['vec'].size() = {observation['vec'].size()}")
                shared_env_data.fill_data(env_idx = env_idx,
                                          observation=observation,
                                          reward=reward,
                                          action=action,
                                          terminated=terminated,
                                          truncated=truncated,
                                          info = info,
                                          reset_info = reset_info,
                                          reset_observation = reset_observation)
                reset_info = None
            elif cmd == b"reset":
                data = remote.recv()
                maybe_options = {"options": data[1]} if data[1] else {}
                reset_observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((reset_observation, reset_info))
                reset_info = None
            elif cmd == b"render":
                remote.send(env.render())
            elif cmd == b"close":
                env.close()
                remote.close()
                shared_env_data.close()
                break
            elif cmd == b"get_spaces":
                remote.send((env.observation_space, env.action_space, info_space))
            elif cmd == b"env_method":
                data = remote.recv()
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == b"get_attr":
                data = remote.recv()
                remote.send(getattr(env, data))
            elif cmd == b"set_attr":
                data = remote.recv()
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == b"is_wrapped":
                data = remote.recv()
                remote.send(is_wrapped(env, data))
            elif cmd == b"set_data_struct":
                data = remote.recv()
                shared_env_data.set_data_struct(data)
            else:
                pass
                # print(f"`{cmd}` is not implemented in the worker"))
            if cmd is not None:
                simple_commander.mark_done()
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp_helper.get_context(start_method)
        self._simple_commander = SimpleCommander(ctx, n_envs=n_envs, timeout_s=60)
        self._shared_env_data = SharedEnvData(n_envs=n_envs, mp_context=ctx, timeout_s=60)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for i in range(n_envs):
            work_remote, remote, env_fn = self.work_remotes[i], self.remotes[i], env_fns[i]
            args = (work_remote, remote, CloudpickleWrapper(env_fn), self._simple_commander, self._shared_env_data, i)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=False)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.num_envs = len(env_fns)
        observation_spaces, action_spaces, info_spaces = self.get_spaces()
        observation_space, action_space, info_space = observation_spaces[0], action_spaces[0], info_spaces[0]

        self._shared_data = SharedData(observation_space=observation_space,
                                        action_space=action_space,
                                        info_space = info_space,
                                        n_envs=n_envs,
                                        device=th.device("cpu"))
        self._shared_env_data.set_data_struct(self._shared_data)
        self._simple_commander.set_command("set_data_struct")
        for remote in self.remotes: remote.send(self._shared_data)
        self._simple_commander.wait_done()

        super().__init__(len(env_fns), observation_space, action_space)

    def get_spaces(self):
        self._simple_commander.set_command("get_spaces")
        obs_spaces = [None]*self.num_envs
        act_spaces = [None]*self.num_envs
        info_spaces = [None]*self.num_envs
        self._simple_commander.wait_done()
        for i in range(len(self.remotes)):
            obs_spaces[i], act_spaces[i], info_spaces[i] = self.remotes[i].recv()
        return obs_spaces, act_spaces, info_spaces

    def step_async(self, actions: np.ndarray) -> None:
        self._shared_env_data.mark_waiting_data()
        self._simple_commander.set_command("step")
        self._shared_env_data.fill_actions(th.as_tensor(actions))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        # To be compatible with VecEnvs the worker in case of reset will
        # put the reset observation in obs and the last step observation in info
        # the reset info is always in reset_info (only updated at each reset)
        # So reset_obss is not used as it is already in obss
        self._simple_commander.wait_done()
        obss, rews, terminateds, truncateds, infos, reset_obss, reset_infos = copy.deepcopy(self._shared_env_data.wait_data())
        self.reset_infos = reset_infos
        dones = th.logical_or(truncateds, terminateds)
        # results = [remote.recv() for remote in self.remotes]
        # t = time.monotonic()
        self.waiting = False
        # obss, rews, dones, infos, self.reset_infos, tsend = zip(*results)  # type: ignore[assignment]
        # print(f"twait = {[t-ts for ts in tsend]}")
        obss = unstack_tensor_tree(obss)
        infos = unstack_tensor_tree(infos)
        return _flatten_obs(obss, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        self._simple_commander.set_command("reset")
        for env_idx, remote in enumerate(self.remotes):
            remote.send((self._seeds[env_idx], self._options[env_idx]))
        results = [remote.recv() for remote in self.remotes]
        self._simple_commander.wait_done()
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            self.step_wait()
        self._simple_commander.set_command("close")
        self._simple_commander.wait_done()
        for process in self.processes:
            process.join()
        self.closed = True
        for remote in self.remotes : remote.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        self._simple_commander.set_command("render")
        outputs = [remote.recv() for remote in self.remotes]
        self._simple_commander.wait_done()
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        self._simple_commander.set_command("get_attr")
        for remote in target_remotes: remote.send((attr_name))
        ret = [remote.recv() for remote in target_remotes]
        self._simple_commander.wait_done()
        return ret

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        self._simple_commander.set_command("set_attr")
        for remote in target_remotes:
            remote.send((attr_name, value))
        for remote in target_remotes:
            remote.recv()
        self._simple_commander.wait_done()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        self._simple_commander.set_command("env_method")
        for remote in target_remotes:
            remote.send((method_name, method_args, method_kwargs))
        ret = [remote.recv() for remote in target_remotes]
        self._simple_commander.wait_done()
        return ret

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        self._simple_commander.set_command("is_wrapped")
        for remote in target_remotes: remote.send(wrapper_class)
        ret = [remote.recv() for remote in target_remotes]
        self._simple_commander.wait_done()
        return ret

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]
