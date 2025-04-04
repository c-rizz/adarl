"""An async vector environment."""
from __future__ import annotations

import multiprocessing as mp
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union, Any, Callable, Dict, Literal

import numpy as np
import cloudpickle
import copy

from gymnasium import logger
from gymnasium.core import ActType, Env, ObsType, RenderFrame
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.vector.utils import concatenate, create_empty_array
from gymnasium.vector.vector_env import VectorEnv
from adarl.utils.shared_env_data import SharedEnvData, SimpleCommander, SharedData
from adarl.utils.tensor_trees import space_from_tree, map_tensor_tree, stack_tensor_tree, to_contiguous_tensor
import torch as th
import adarl.utils.mp_helper as mp_helper
import time
import adarl.utils.dbg.ggLog as ggLog
import os
import setproctitle
import cProfile
from adarl.utils.tensor_trees import TensorTree
import traceback
__all__ = ["AsyncVectorEnv", "AsyncState"]


class AsyncState(Enum):
    """The AsyncVectorEnv possible states given the different actions."""

    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    cloudpickle_func: bytes,
    simple_commander : SimpleCommander,
    shared_env_data : SharedEnvData,
    env_idx : int,
    action_device = "numpy",
    worker_init_fn = None,
    worker_init_kwargs = {}
) -> None:
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    if worker_init_fn is not None:
        worker_init_fn(**worker_init_kwargs)
    # ggLog.info(f"async_vector_env: starting worker {mp.current_process().name}, env_idx = {env_idx}, pid = {os.getpid()}")
    setproctitle.setproctitle(mp.current_process().name)
    # th.cuda.memory._record_memory_history()
    # pr = cProfile.Profile()
    # pr.enable()
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    # env = _patch_env(env_fn_wrapper.var())
    env = cloudpickle.loads(cloudpickle_func)()
    next_start_observation, next_start_info = env.reset()
    consequent_observation, consequent_info = next_start_observation, next_start_info
    # next_start_info["final_observation"] = consequent_observation  # always put an observation in info, this is not actually correct, but it hase the same structure
    # next_start_info["terminal_observation"] = consequent_observation  # always put an observation in info, this is not actually correct, but it hase the same structure
    # next_start_info["real_next_observation"] = consequent_observation  # always put an observation in info, this is not actually correct, but it hase the same structure
    # print(f"reset_info = {reset_info}")
    info_space = space_from_tree(map_tensor_tree(next_start_info, to_contiguous_tensor))
    next_start_info = None
    running = True
    stepcount = 0
    while running:
        try:
            cmd = simple_commander.wait_command()
            # ggLog.info(f"async_vector_env worker got cmd {cmd}")
            if cmd == b"step":
                # ggLog.info(f"async_vec_env step: 1 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                action = shared_env_data.wait_actions()[env_idx]
                # ggLog.info(f"async_vec_env step: 2 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                if action_device == "numpy":
                    if isinstance(action, th.Tensor):
                        action = action.cpu().numpy()
                    else:
                        action = np.array(action)
                elif isinstance(action_device, th.device):
                    action = th.as_tensor(action, device = action_device)
                # ggLog.info(f"async_vec_env step: 3 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                consequent_observation, reward, terminated, truncated, consequent_info = env.step(action)
                # convert to SB3 VecEnv api
                # ggLog.info(f"async_vec_env step: 4 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                done = terminated or truncated
                # info = {}
                consequent_info["TimeLimit.truncated"] = truncated and not terminated # used by SB3 VecEnv specification
                    
                # When there is a reset the experience collector will need to receive both:
                # - The "consequent_observation" observation, which is the consequence of the action
                # - The "next_start_observation" observation, which is after the reset, the first of
                #   the new episode.
                # Same applies to the info, there is the one consequence of the action and the one of the next
                # transition, which may come from the reset or be the same of the consequential one.
                # To keep everything clear I always return both the consequential and next input observations
                # and infos, even when there is no reset. If there is no reset the will simply be the same thing.
                # So I will return:
                # - consequent_observation
                # - next_start_observation
                # - consequent_info
                # - next_start_info
                # This will then be transplated to gymnasium conventions on the server side.
                
                if done:
                    # save final observation where user can get it, then reset
                    next_start_observation, next_start_info = env.reset()
                    # ggLog.info(f"async_vec_env step: 5 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                    # reset_info["final_info"] = reset_info  # always put an observation in info, this is not actually correct, but it hase the same structure
                    # next_start_info["final_observation"] = next_start_observation  # always put an observation in info, this is not actually correct, but it hase the same structure
                    # next_start_info["terminal_observation"] = next_start_observation  # always put an observation in info, this is not actually correct, but it hase the same structure
                    # next_start_info["real_next_observation"] = next_start_observation  # always put an observation in info, this is not actually correct, but it hase the same structure
                else:
                    next_start_observation = consequent_observation
                    next_start_info = consequent_info

                # consequent_info["final_observation"] = consequent_observation # We always put the newest observation in info, even if not resetting
                # consequent_info["terminal_observation"] = consequent_observation # We always put the newest observation in info, even if not resetting
                # consequent_info["real_next_observation"] = consequent_observation # this is the actual next_observation, the one that is a consequence of action
                # info["final_info"] = info # We always put the newest observation in info, even if not resetting
                
                # As a result now we have:
                # - observation contains the newest observation (either the action's consequence or the first of a new episode)
                # - info contains the observation that is consequence of the action (the same obs at the 3 different keys)
                # reset_info always contains the info of the last reset() (it is only updated at each reset)

                # In any case (i.e. truncated, terminated, both, neither) the real next observation is the one in info
                # then the algorithm will not consider reward propagation if termination==True (it instead has to
                # consider reward propagation if termination==False and truncation==True)
                # ggLog.info(f"async_vec_env step: 6 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                consequent_observation = map_tensor_tree(consequent_observation, func=lambda x: th.as_tensor(x))
                if next_start_observation is not consequent_observation:
                    next_start_observation = map_tensor_tree(next_start_observation, func=lambda x: th.as_tensor(x))
                reward = map_tensor_tree(reward, func=th.as_tensor)
                action = map_tensor_tree(action, func=th.as_tensor)
                terminated = map_tensor_tree(terminated, func=th.as_tensor)
                truncated = map_tensor_tree(truncated, func=th.as_tensor)
                consequent_info = map_tensor_tree(consequent_info, func=th.as_tensor)
                if next_start_info is not consequent_info:
                    next_start_info = map_tensor_tree(next_start_info, func=th.as_tensor)
                # print(f"observation shape = {map_tensor_tree(observation, lambda t: t.size())}")
                # ggLog.info(f"async_vec_env step: 7 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                shared_env_data.fill_data(env_idx = env_idx,
                                          next_start_info = next_start_info,
                                          next_start_observation = next_start_observation,
                                          action=action,
                                          reward=reward,
                                          terminated=terminated,
                                          truncated=truncated,
                                          consequent_observation=consequent_observation,
                                          consequent_info = consequent_info)
                # ggLog.info(f"async_vec_env step: 8 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                next_start_info = None
                stepcount+=1
                # th.cuda.memory._dump_snapshot(f"worker{env_idx}_step{stepcount}.pickle")
            elif cmd == b"reset":
                # ggLog.info(f"async_vec_env reset: 1 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                data = remote.recv()
                # ggLog.info(f"async_vec_env reset: 2 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                maybe_options = {"options": data[1]} if data[1] else {}
                next_start_observation, next_start_info = env.reset(seed=data[0], **maybe_options)
                # ggLog.info(f"async_vec_env reset: 3 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                remote.send((next_start_observation, next_start_info))
                # ggLog.info(f"async_vec_env reset: 4 allocated {th.cuda.memory_allocated()} init {th.cuda.is_initialized()}")
                next_start_info = None
            elif cmd == b"render":
                remote.send(env.render())
            elif cmd == b"close":
                env.close()
                remote.close()
                running = False
            elif cmd == b"get_spaces":
                spaces = (env.observation_space, env.action_space, info_space)
                remote.send(spaces)
            elif cmd == b"env_method":
                data = remote.recv()
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == b"get_attr":
                data = remote.recv()
                remote.send(getattr(env, data))
            elif cmd == b"set_attr":
                data = remote.recv()
                setattr(env, data[0], data[1])
            elif cmd == b"is_wrapped":
                data = remote.recv()
                remote.send(is_wrapped(env, data))
            elif cmd == b"set_data_struct":
                data = remote.recv()
                shared_env_data.set_data_struct(data)
            elif cmd is None:
                ggLog.debug(f"async_vector_env: no command received")
            else:
                ggLog.warn(f"async_vector_env: `{cmd}` is not implemented in the worker")
            if cmd is not None:
                simple_commander.mark_done()
        except EOFError:
            break
    # pr.disable()
    # ggLog.info(f"async_vector_env: worker {mp.current_process().name} terminating, pid = {os.getpid()}")
    # if env_idx == 1:
    #     pr.print_stats(sort='cumtime')

class AsyncVectorEnvShmem(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, shared memory, and pipes for communication.

    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        context : Optional[str] = None,
        purely_numpy = False,
        shared_mem_device = th.device("cpu"),
        env_action_device : th.device|Literal["numpy"]= "numpy",
        copy_data = False,
        worker_init_fn = None,
        worker_init_kwargs = {},
        daemonic_workers = True
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            context: Context for `multiprocessing`. If ``None``, then the default context is used.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
            ValueError: If observation_space is a custom space (i.e. not a default space in Gym,
                such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
        """
        self.env_fns = env_fns        
        self._purely_numpy = purely_numpy
        self.num_envs = len(env_fns)
        self._shared_mem_device = shared_mem_device
        self._env_action_device = env_action_device
        self._copy_data = copy_data
        self._daemonic_workers = daemonic_workers
        if context is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            context = "forkserver" if forkserver_available else "spawn"
        ctx = mp_helper.get_context(context)
        self._simple_commander = SimpleCommander(ctx, n_envs=self.num_envs, timeout_s=60)
        self._shared_env_data = SharedEnvData(n_envs=self.num_envs, mp_context=ctx, timeout_s=60)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for i in range(self.num_envs):
            work_remote, remote, env_fn = self.work_remotes[i], self.remotes[i], env_fns[i]
            args = (work_remote, remote, cloudpickle.dumps(env_fn), self._simple_commander, self._shared_env_data, i, self._env_action_device,
                    worker_init_fn, worker_init_kwargs)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, name=f"async_vector_env.worker{i}", daemon=self._daemonic_workers)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        observation_spaces, action_spaces, info_spaces = self.get_spaces()
        observation_space, action_space, info_space = observation_spaces[0], action_spaces[0], info_spaces[0]

        self.single_info_space = info_space

        ggLog.info(f"Building shared data for space {observation_space}, self._shared_mem_device={self._shared_mem_device}")
        self._shared_data = SharedData(observation_space=observation_space,
                                        action_space=action_space,
                                        info_space = info_space,
                                        n_envs=self.num_envs,
                                        device=self._shared_mem_device)
        self._shared_env_data.set_data_struct(self._shared_data)
        self._simple_commander.set_command("set_data_struct")
        for remote in self.remotes: remote.send(self._shared_data)
        self._simple_commander.wait_done()
        
        

        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        # self.observations = create_empty_array(
        #     self.single_observation_space, n=self.num_envs, fn=np.zeros
        # )

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    @property
    def np_random_seed(self) -> Tuple[int, ...]:
        """Returns the seeds of the wrapped envs."""
        return self.get_attr("np_random_seed")

    @property
    def np_random(self) -> Tuple[np.random.Generator, ...]:
        """Returns the numpy random number generators of the wrapped envs."""
        return self.get_attr("np_random")

    def reset(
        self,
        *,
        seed: Union[int, List[int], None] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets all sub-environments in parallel and return a batch of concatenated observations and info.

        Args:
            seed: The environment reset seeds
            options: If to return the options

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        self.reset_async(seed=seed, options=options)
        return self.reset_wait()


    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.

        Args:
            seed: List of seeds for each environment
            options: The reset option

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`step_async`). This can be caused by two consecutive
                calls to :meth:`reset_async`, with no call to :meth:`reset_wait` in between.
        """
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                str(self._state.value),
            )

        self._simple_commander.set_command("reset")
        for env_idx, remote in enumerate(self.remotes):
            remote.send((seed[env_idx], options))
        
        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            timeout: Number of seconds before the call to ``reset_wait`` times out. If `None`, the call to ``reset_wait`` never times out.

        Returns:
            A tuple of batched observations and list of dictionaries

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`reset_wait` was called without any prior call to :meth:`reset_async`.
            TimeoutError: If :meth:`reset_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        self._simple_commander.wait_done(timeout=timeout)
        if not all([remote.poll(timeout=1) for remote in self.remotes]): # should all be already available due to wait_done
            raise RuntimeError(f"Timed out waiting for reset")
        results = [remote.recv() for remote in self.remotes]
        observations, infos = zip(*results)

        if self._purely_numpy:
            observations = concatenate(
                self.single_observation_space, observations, out=None
            )
        else:
            observations = stack_tensor_tree(observations)

        self._state = AsyncState.DEFAULT

        if self._copy_data:
            observations = copy.deepcopy(observations)

        return (observations, infos)

    def step(
        self, actions: ActType
    ) -> Tuple[ObsType, np.ndarray,np.ndarray,np.ndarray, Dict[str, Any]]:
        """Take an action for each parallel environment.

        Args:
            actions: element of :attr:`action_space` batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray):
        """Send the calls to :meth:`Env.step` to each sub-environment.

        Args:
            actions: Batch of actions. element of :attr:`VectorEnv.action_space`

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`reset_async`). This can be caused by two consecutive
                calls to :meth:`step_async`, with no call to :meth:`step_wait` in
                between.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )


        if isinstance(actions,th.Tensor):
            actions = actions.to(self._shared_mem_device)
        self._shared_env_data.mark_waiting_data()
        self._shared_env_data.fill_actions(actions)
        self._simple_commander.set_command("step")
        self._t0_step = time.monotonic()

        self._state = AsyncState.WAITING_STEP

    def step_wait(
        self, timeout: int | float | None = None
    ) -> (Tuple[TensorTree[np.ndarray], np.ndarray, np.ndarray, np.ndarray, TensorTree[np.ndarray]] | 
          Tuple[TensorTree[th.Tensor],  th.Tensor,  th.Tensor,  th.Tensor,  TensorTree[th.Tensor]]):
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out. If ``None``, the call to :meth:`step_wait` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`step_wait` was called without any prior call to :meth:`step_async`.
            TimeoutError: If :meth:`step_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )
        
        # The worker in case of reset will  put the reset observation in obs and 
        # the last step observation in info. reset_infos always contains the last
        # reset's info (only updated at each reset)
        # reset_obss is not used as it is already in obss
        self._simple_commander.wait_done(timeout=timeout)
        # t1_step = time.monotonic()
        data = self._shared_env_data.wait_data()
        # t2_step = time.monotonic()
        if self._copy_data:
            data = copy.deepcopy(data)
        # observations, rewards, terminateds, truncateds, infos, reset_obss, reset_infos = data
        consequent_observations, rewards, terminations, truncations, consequent_infos, next_start_observations, next_start_infos = data
        # t3_step = time.monotonic()
        # self.reset_infos = reset_infos
        
        next_start_infos["final_observation"] = consequent_observations
        next_start_infos["final_info"] = consequent_infos
        # obss = unstack_tensor_tree(obss)
        # self.observations = concatenate(
        #     self.single_observation_space,
        #     obss,
        #     self.observations,
        # )
        # infos = unstack_tensor_tree(infos)
        ret = (
            next_start_observations,
            rewards,
            terminations,
            truncations,
            next_start_infos
        )
        if self._purely_numpy:
            ret_np : Tuple[TensorTree[np.ndarray], np.ndarray, np.ndarray, np.ndarray, TensorTree[np.ndarray]] = map_tensor_tree(ret, func=lambda x: np.array(x))
            ret = ret_np
        # self.observations = observations # it is already a stacked observation
        
        # return _flatten_obs(obss, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

        self._state = AsyncState.DEFAULT

        # tf_step = time.monotonic()
        # ggLog.info(f"collection: {t1_step-self._t0_step} {t2_step-t1_step} {t3_step-t2_step} {tf_step-t3_step}")


        return ret

    def call(self, name: str, *args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Call a method from each parallel environment with args and kwargs.

        Args:
            name (str): Name of the method or property to call.
            *args: Position arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def render(self) -> Tuple[RenderFrame, ...] | None:
        """Returns a list of rendered frames from the environments."""
        return self.call("render")

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `call_async` while waiting for a pending call to `{self._state.value}` to complete.",
                str(self._state.value),
            )
        
        self._simple_commander.set_command("env_method")
        for remote in self.remotes:
            remote.send((name, args, kwargs))

        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> Tuple[Any, ...]:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out.
                If ``None`` (default), the call to :meth:`step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling :meth:`call_wait` without any prior call to :meth:`call_async`.
            TimeoutError: The call to :meth:`call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        self._simple_commander.wait_done(timeout=timeout)
        if not all([remote.poll(timeout=1) for remote in self.remotes]): # should all be already available due to wait_done
            raise RuntimeError(f"Timed out waiting for call to method")
        ret = [remote.recv() for remote in self.remotes]
        return tuple(ret)

    def get_attr(self, name: str) -> Tuple[Any, ...]:
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: Union[List[Any], Tuple[Any], object], timeout = None):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling :meth:`set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )
        
        self._simple_commander.set_command("set_attr")
        for i in range(len(self.remotes)):
            remote = self.remotes[i]
            remote.send((name, values[i]))
        self._simple_commander.wait_done(timeout=timeout)

    def close_extras(
        self, timeout: Optional[Union[int, float]] = None, terminate: bool = False
    ):
        """Close the environments & clean up the extra resources (processes and pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            self._simple_commander.set_command("close")
            self._simple_commander.wait_done()
            for remote in self.remotes : remote.close()
            for process in self.processes: process.join(30)            
            for process in self.processes:
                if process.is_alive():
                    logger.error(f"Subprocess still alive after timeot, sending SIGTERM.")
                    process.terminate()
        for process in self.processes: process.join(30)
        for process in self.processes:
            if process.is_alive():
                logger.error(f"Subprocess still alive after timeot, sending SIGKILL.")
                process.kill()

    def get_spaces(self):
        self._simple_commander.set_command("get_spaces")
        obs_spaces = [None]*self.num_envs
        act_spaces = [None]*self.num_envs
        info_spaces = [None]*self.num_envs
        for i in range(len(self.remotes)):
            # ggLog.info(f"Waiting spaces")
            obs_spaces[i], act_spaces[i], info_spaces[i] = self.remotes[i].recv()
            # ggLog.info(f"got spaces")
        self._simple_commander.wait_done()
        return obs_spaces, act_spaces, info_spaces
    
    def _check_spaces(self):
        self._assert_is_running()
        spaces = (self.single_observation_space, self.single_action_space)
        sub_spaces = self.get_spaces()
        for i in range(self.num_envs):
            if self.single_observation_space != sub_spaces[0][i]:
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )
            if self.single_action_space != sub_spaces[1][i]:
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )
            if self.single_info_space != sub_spaces[2][i]:
                raise RuntimeError(
                    "Some environments have an info space different from "
                    f"`{self.single_info_space}`. In order to batch infos, the "
                    "info spaces from all environments must be equal."
                )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
            )

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)
