from __future__ import annotations
from abc import ABC, abstractmethod
from adarl.utils.tensor_trees import map_tensor_tree
from adarl.utils.utils import dbg_check_finite
from cmath import inf
from dataclasses import dataclass
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer, DictReplayBufferSamples
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from typing import Union, List, Dict, Any, Optional, Callable, NamedTuple
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.mp_helper as mp_helper
import ctypes
import numpy as np
import psutil
import random
import time
import torch as th
from typing_extensions import override
import warnings

# class ReplayBuffer_updatable(ReplayBuffer):
#     def update(self, buffer : ReplayBuffer):

#         if (self.optimize_memory_usage or buffer.optimize_memory_usage):
#             raise RuntimeError("Memory optimizatio is not supported")
        
#         copied = 0
#         while copied < buffer.size():
#             space_to_end = self.buffer_size - self.pos
#             to_copy = min(space_to_end, buffer.size()-copied)

#             self.actions[self.pos:self.pos + to_copy] = buffer.actions[copied:copied+to_copy]
#             self.rewards[self.pos:self.pos + to_copy] = buffer.rewards[copied:copied+to_copy]
#             self.dones[self.pos:self.pos + to_copy]   = buffer.dones[copied:copied+to_copy]
#             self.observations[self.pos:self.pos + to_copy] = buffer.observations[copied:copied+to_copy]
#             self.next_observations[self.pos:self.pos + to_copy] = buffer.next_observations[copied:copied+to_copy]
#             if self.handle_timeout_termination:
#                 self.timeouts[self.pos:self.pos + to_copy] = buffer.timeouts[copied:copied+to_copy]
                
#             self.pos += to_copy
#             if self.pos == self.buffer_size:
#                 self.full = True
#                 self.pos = 0
#             copied += to_copy

class BaseBuffer(ABC):
    
    def __init__(self,  buffer_size: int,
                        observation_space: spaces.Space,
                        action_space: spaces.Space,
                        device: Union[th.device, str] = "auto",
                        n_envs: int = 1,):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.device = device
        self.th_device = th.device(device)
        self.n_envs = n_envs

    @abstractmethod
    def memory_size(self):
        raise NotImplementedError()
       
    @abstractmethod
    def add(
        self,
        obs: Dict[str, th.Tensor],
        next_obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        truncated: th.Tensor,
        terminated: th.Tensor,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> TransitionBatch:
        raise NotImplementedError()

    @abstractmethod
    def _get_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> TransitionBatch:
        raise NotImplementedError()

    @abstractmethod
    def storage_torch_device(self):
        raise NotImplementedError()

    @abstractmethod
    def stored_frames(self):
        raise NotImplementedError()
    
    def size(self):
        return self.stored_frames()
    
    @abstractmethod
    def collected_frames(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, buffer : ThDReplayBuffer):
        raise NotImplementedError()

    @abstractmethod
    def predict_memory_consumption(self) -> tuple[float,int | float]:
        raise NotImplementedError()

    

@dataclass
class TransitionBatch():
    observations : Union[th.Tensor, Dict]
    actions : th.Tensor
    next_observations : Union[th.Tensor, Dict]
    terminated : th.Tensor
    rewards : th.Tensor


class RandomHoldoutBuffer(DictReplayBuffer):
    def __init__(self,
                buffer_size: int,
                observation_space: spaces.Space,
                action_space: spaces.Space,
                device: Union[th.device, str] = "cpu",
                n_envs: int = 1,
                optimize_memory_usage: bool = False,
                handle_timeout_termination: bool = True,
                storage_torch_device: str = "cpu",
                buffer_class : type = None,
                validation_ratio : float = 0.1,
                disable : bool = False):
        super().__init__(buffer_size = buffer_size,
                         observation_space = observation_space,
                         action_space = action_space,
                         device = device,
                         n_envs = n_envs,
                         optimize_memory_usage = optimize_memory_usage,
                         handle_timeout_termination = handle_timeout_termination)
        self._storage_torch_device = storage_torch_device
        self._validation_ratio = validation_ratio
        self._train_buffer = buffer_class(buffer_size=buffer_size,
                                            observation_space = observation_space,
                                            action_space = action_space,
                                            device = device,
                                            n_envs = n_envs,
                                            optimize_memory_usage = optimize_memory_usage,
                                            handle_timeout_termination = handle_timeout_termination,
                                            storage_torch_device = storage_torch_device)
        self._disable = disable
        if disable:
            self._validation_buffer = None
        else:                                            
            self._validation_buffer = buffer_class( buffer_size=int(buffer_size),
                                                    observation_space = observation_space,
                                                    action_space = action_space,
                                                    device = device,
                                                    n_envs = n_envs,
                                                    optimize_memory_usage = optimize_memory_usage,
                                                    handle_timeout_termination = handle_timeout_termination,
                                                    storage_torch_device = storage_torch_device)

    def size(self):
        if self._disable:
            return self._train_buffer.size()
        else:
            return self._train_buffer.size() + self._validation_buffer.size()


    def memory_size(self):
        if self._disable:
            return self._train_buffer.memory_size()
        else:
            return self._train_buffer.memory_size() + self._validation_buffer.memory_size()

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]]) -> None:
        if random.random() > self._validation_ratio or self._disable:
            buffer_to_use = self._train_buffer
        else:
            buffer_to_use = self._validation_buffer

        buffer_to_use.add(obs,next_obs, action, reward, done, infos)

    def update(self, buffer : DictReplayBuffer):
        self._train_buffer.update(buffer._train_buffer)
        if not self._disable: # if not disabled
            self._validation_buffer.update(buffer._validation_buffer)
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, validation_set : bool = False) -> DictReplayBufferSamples:
        if validation_set and not self._disable:
            buffer_to_use = self._validation_buffer
        else:
            buffer_to_use = self._train_buffer

        return buffer_to_use.sample(batch_size=batch_size, env=env)

    def storage_torch_device(self):
        return self._storage_torch_device


# from https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict = {
        bool       : th.bool,
        np.uint8      : th.uint8,
        np.int8       : th.int8,
        np.int16      : th.int16,
        np.int32      : th.int32,
        np.int64      : th.int64,
        np.float16    : th.float16,
        np.float32    : th.float32,
        np.float64    : th.float64,
        np.complex64  : th.complex64,
        np.complex128 : th.complex128
    }

def numpy_to_torch_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype_type = dtype.type
    else:
        dtype_type = dtype
    return numpy_to_torch_dtype_dict[dtype_type]




class BasicStorage():
    """
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        n_envs: int = 1,
        storage_torch_device: Union[str,th.device] = "cpu",
        share_mem : bool = True,
        allow_rollover = False
    ):

        assert isinstance(observation_space, spaces.Dict), "BasicStorage must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)
        self._observation_space = observation_space
        self._action_space = action_space
        self.obs_shape : Dict = get_obs_shape(self._observation_space) #type: ignore
        self._storage_torch_device = th.device(storage_torch_device)
        self.n_envs = n_envs
        self._share_mem = share_mem
        self._allow_rollover = allow_rollover
        
        self._allocate_buffers(self.buffer_size)
        # self._addcount = 0
        # self._completed_episodes = 0
        # self._addcount_since_clear = 0

        ctx = mp_helper.get_context(method="forkserver")
        # these are shared and not synchronized, so be careful to use them properly.
        # It should not be an issue, you should use the whole class in a properly synchronized manner
        # In the end using torch tensors as shared emmeory is just the easiest thing (mp.Values don't like to be sent through pipes)
        self._counters = th.tensor([0,0,0], dtype=th.int64, device="cpu").share_memory_()

    @property
    def _addcount(self):
        return self._counters[0].item()
    @_addcount.setter
    def _addcount(self, value):
        self._counters[0] = value

    @property
    def _completed_episodes(self):
        return self._counters[1].item()
    @_completed_episodes.setter
    def _completed_episodes(self, value):
        self._counters[1] = value

    @property
    def _addcount_since_clear(self):
        return self._counters[2].item()
    @_addcount_since_clear.setter
    def _addcount_since_clear(self, value):
        self._counters[2] = value

    def _allocate_buffers(self, buffer_size):
        self.observations = {
            key: th.zeros(  (buffer_size, self.n_envs) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: th.zeros(  (buffer_size, self.n_envs) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = th.zeros((buffer_size, self.n_envs) + self._action_space.shape, 
                                dtype=numpy_to_torch_dtype(self._action_space.dtype),
                                device = self._storage_torch_device)
        self.rewards = th.zeros((buffer_size, self.n_envs), dtype=th.float32, device = self._storage_torch_device)
        self.terminated = th.zeros((buffer_size, self.n_envs), dtype=th.uint8, device = self._storage_torch_device)
        self.truncated = th.zeros((buffer_size, self.n_envs), dtype=th.uint8,  device = self._storage_torch_device)

        if self._storage_torch_device.type == "cpu":
            for k in self.observations.keys():
                self.observations[k] = self.observations[k].pin_memory()
            for k in self.next_observations.keys():
                self.next_observations[k] = self.next_observations[k].pin_memory()
            self.actions = self.actions.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.terminated = self.terminated.pin_memory()
            self.truncated = self.truncated.pin_memory()

        if self._share_mem:
            for k in self.observations.keys():
                self.observations[k] = self.observations[k].share_memory_()
            for k in self.next_observations.keys():
                self.next_observations[k] = self.next_observations[k].share_memory_()
            self.actions = self.actions.share_memory_()
            self.rewards = self.rewards.share_memory_()
            self.terminated = self.terminated.share_memory_()
            self.truncated = self.truncated.share_memory_()

        
    def clear(self):
        self._addcount_since_clear = 0

        

    def add(
        self,
        obs: Dict[str, th.Tensor | np.ndarray],
        next_obs: Dict[str, th.Tensor],
        action: th.Tensor | np.ndarray,
        reward: th.Tensor | np.ndarray,
        terminated: th.Tensor | np.ndarray,
        truncated: th.Tensor | np.ndarray
    ) -> None:
        if not self._allow_rollover and int(self._addcount_since_clear) >= self.buffer_size:
            raise RuntimeError(f"Called add with full buffer and allow_rollover is false")
        dbg_check_finite([obs,next_obs,action,reward,truncated,terminated])
        pos = int(self._addcount_since_clear) % self.buffer_size
        # ggLog.info(f"{type(self)}: Adding step {self._addcount}, {self.size()}")
        # Copy to avoid modification by reference
        devices_to_sync = {t.device for t in [action,reward,terminated,truncated] if isinstance(t, th.Tensor)}
        devices_to_sync.add(self._storage_torch_device)
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self._observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][pos].copy_(th.as_tensor(obs[key]), non_blocking=True)
            if isinstance(obs,th.Tensor): devices_to_sync.add(obs.device)

        for key in self.next_observations.keys():
            if isinstance(self._observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][pos].copy_(th.as_tensor(next_obs[key]), non_blocking=True)
            if isinstance(next_obs,th.Tensor): devices_to_sync.add(next_obs.device)

        if isinstance(terminated, np.ndarray):
            terminated = th.as_tensor(terminated, device="cpu") # should be a no-copy operation
        if isinstance(truncated, np.ndarray):
            truncated = th.as_tensor(truncated, device="cpu") # should be a no-copy operation
        ep_ends = th.logical_or(terminated, truncated).count_nonzero().to(device="cpu", non_blocking=True)
        # ggLog.info(f"ep_ends = {ep_ends}")

        # print(f"storing action[{pos}] {action}")
        self.actions[pos].copy_(th.as_tensor(action), non_blocking=True)
        self.rewards[pos].copy_(th.as_tensor(reward), non_blocking=True)
        self.terminated[pos].copy_(th.as_tensor(terminated), non_blocking=True)
        self.truncated[pos].copy_(th.as_tensor(truncated), non_blocking=True)
        for device in devices_to_sync:
            if device.type == "cuda":
                th.cuda.synchronize(device)

        self._addcount = self._addcount + 1
        self._completed_episodes = self._completed_episodes + ep_ends.item()
        self._addcount_since_clear +=1
    
    def size(self):
        return min(self.buffer_size,int(self._addcount_since_clear))

    def storage_torch_device(self):
        return self._storage_torch_device

    def stored_frames(self):
        return self.size()*self.n_envs
    
    def added_frames(self) -> int:
        """Frames that were added to the buffer since its creation.
        This is not resetted by clear() and still increases when the buffer is full.

        Returns
        -------
        int
            Number of frames that were added to the buffer since its creation
        """
        return self._addcount * self.n_envs
    
    def added_completed_episodes(self) -> int:
        """Episodes that were added to the buffer since its creation. It only increases when a full episode is added.
        Frames of non-completed episodes are not accounte for.
        This is not resetted by clear() and still increases when the buffer is full.

        Returns
        -------
        int
            Number of episodes that were added to the buffer since its creation
        """
        return int(self._completed_episodes)

    def replay(self):
        addcount_since_clear = int(self._addcount_since_clear)
        pos = addcount_since_clear % self.buffer_size if addcount_since_clear > self.buffer_size else 0
        ret_vsteps = 0
        while ret_vsteps < self.size():
            obs = {k:v[pos] for k,v in self.observations.items()}
            next_obs = {k:v[pos] for k,v in self.next_observations.items()}
            action = self.actions[pos]
            reward = self.rewards[pos]
            terminated = self.terminated[pos]
            truncated = self.truncated[pos]
            # print(f"replaying action[{pos}] {action}")
            yield (obs, next_obs, action, reward, terminated, truncated)
            pos = (pos+1) % self.buffer_size
            ret_vsteps +=1

class ThDReplayBuffer(BaseBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Derived from the stable baselines dict replay buffer. 
    It uses torch tensors to store data and is meant for gymnasium environments.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        storage_torch_device: Union[str,th.device] = "cpu",
        fallback_to_cpu_storage: bool = True,
        copy_outputs : bool = True
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)
        self._observation_space = observation_space
        self._action_space = action_space
        self._copy_outputs = copy_outputs
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        if handle_timeout_termination == False:
            raise NotImplementedError()

        storage_torch_device = th.device(storage_torch_device)
        self._storage_torch_device = storage_torch_device
        if storage_torch_device == "cuda" and device == "cpu":
            raise AttributeError(f"Storage device is gpu, and output device is cpu. This doesn't make much sense. Use either [gpu,gpu], [cpu,gpu], or [cpu,cpu]")

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage
        
        self.obs_shape : Dict
        self.observations : Dict[Any, th.Tensor] = {}
        self.next_observations : Dict[Any, th.Tensor] = {}
        self.actions : th.Tensor = th.empty(size=(0,))
        self.rewards : th.Tensor = th.empty(size=(0,))
        self.terminated : th.Tensor = th.empty(size=(0,))
        self.truncated : th.Tensor = th.empty(size=(0,))

        if self._storage_torch_device.type == "cuda" and fallback_to_cpu_storage:
            pred_avail = self.predict_memory_consumption()
            consumptionRatio = pred_avail[0]/pred_avail[1]
            if consumptionRatio>0.5:
                warnings.warn(   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                                f"Not enough memory on requested device {self._storage_torch_device} (Would consume {consumptionRatio*100:.0f}% = {pred_avail[0]/1024/1024/1024:.3f} GiB)\n"
                                 "Falling back to CPU memory\n"
                                 "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                time.sleep(3)
                self._storage_torch_device = th.device("cpu")
        pred_avail = self.predict_memory_consumption()
        consumptionRatio = pred_avail[0]/pred_avail[1]
        if consumptionRatio>0.5:
            warnings.warn(   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            f"Replay buffer will use {consumptionRatio*100:.0f}% ({pred_avail[0]/1024/1024/1024:.3f} GiB) of available memory on device {self._storage_torch_device}\n"
                             "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            time.sleep(3)

        self._allocate_buffers(self.buffer_size)
        self._addcount = 0
        
    @override
    def predict_memory_consumption(self):
        testRatio = 0.01
        self._allocate_buffers(buffer_size=int(self.buffer_size*testRatio))
        predicted_mem_usage = self.memory_size()/testRatio

        mem_available = float("+inf")
        if self._storage_torch_device.type == "cpu":
            mem_available = psutil.virtual_memory().available
        elif self._storage_torch_device.type == "cuda":
            mem_available = th.cuda.mem_get_info(self._storage_torch_device)[0]

        return predicted_mem_usage, mem_available

    @override
    def memory_size(self):
        obs_nbytes = 0
        for _, obs in self.observations.items():
            obs_nbytes += obs.element_size()*obs.nelement()

        action_nbytes = self.actions.element_size()*self.actions.nelement()
        rewards_nbytes = self.rewards.element_size()*self.rewards.nelement()
        terminated_nbytes = self.terminated.element_size()*self.terminated.nelement()

        total_memory_usage = obs_nbytes + action_nbytes + rewards_nbytes + terminated_nbytes
        if self.next_observations is not None:
            next_obs_nbytes = 0
            for _, obs in self.observations.items():
                next_obs_nbytes += obs.element_size()*obs.nelement()
            total_memory_usage += next_obs_nbytes
        return total_memory_usage

    def _allocate_buffers(self, buffer_size):

        self.observations = {
            key: th.zeros(  (buffer_size, self.n_envs) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: th.zeros(  (buffer_size, self.n_envs) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = th.zeros((buffer_size, self.n_envs, self.action_dim),
                                dtype=numpy_to_torch_dtype(self._action_space.dtype),
                                device = self._storage_torch_device)
        self.rewards = th.zeros((buffer_size, self.n_envs),
                                dtype=th.float32,
                                device = self._storage_torch_device)
        self.terminated = th.zeros((buffer_size, self.n_envs), dtype=th.uint8,
                              device = self._storage_torch_device)

        self.truncated = th.zeros((buffer_size, self.n_envs), dtype=th.uint8,  device = self._storage_torch_device)

        if self._storage_torch_device.type == "cpu":
            for k in self.observations.keys():
                self.observations[k] = self.observations[k].pin_memory()
            for k in self.next_observations.keys():
                self.next_observations[k] = self.next_observations[k].pin_memory()
            self.actions = self.actions.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.terminated = self.terminated.pin_memory()
            self.truncated = self.truncated.pin_memory()


        
    @override
    def add(
        self,
        obs: Dict[str, th.Tensor],
        next_obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        truncated: th.Tensor,
        terminated: th.Tensor,
    ) -> None:
        # if infos is not None and truncated is not None:
        #     raise RuntimeError(f"Can only provided either inifo or truncated")
        # ggLog.info(f"{type(self)}: Adding step {self._addcount}, {self.size()}")
        dbg_check_finite([obs,next_obs,action,reward,truncated,terminated])
        self._addcount+=1

        devices_to_sync = {t.device for t in [action,reward,terminated,truncated] if isinstance(t, th.Tensor)}
        devices_to_sync.add(self._storage_torch_device)
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self._observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos].copy_(th.as_tensor(obs[key]), non_blocking=True)
            if isinstance(obs,th.Tensor): devices_to_sync.add(obs.device)

        for key in self.next_observations.keys():
            if isinstance(self._observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos].copy_(next_obs[key], non_blocking=True)
            if isinstance(next_obs,th.Tensor): devices_to_sync.add(next_obs.device)

        # Same reshape, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos].copy_(th.as_tensor(action), non_blocking=True)
        self.rewards[self.pos].copy_(th.as_tensor(reward), non_blocking=True)            
        self.truncated[self.pos].copy_(th.as_tensor(truncated), non_blocking=True)
        self.terminated[self.pos].copy_(th.as_tensor(terminated), non_blocking=True)
        for device in devices_to_sync:
            if device.type == "cuda":
                th.cuda.synchronize(device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    @override
    def collected_frames(self):
        return self._addcount*self.n_envs

    @override
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, validation_set : bool = False) -> TransitionBatch:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if validation_set:
            raise RuntimeError(f"Validation set not avaliable")
        if (self.optimize_memory_usage):
            raise RuntimeError("Memory optimization is not supported")
        
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.randint(0, upper_bound, size=(batch_size,), device = self._storage_torch_device)
        return self._get_samples(batch_inds, env=env)
        # return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> TransitionBatch:
        # Sample randomly the env idx
        # env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        if not isinstance(batch_inds, th.Tensor):
            batch_inds = th.tensor(batch_inds, device = self._storage_torch_device) # cuda sync point (if not already a cuda tensor)
        else:
            batch_inds.to(self._storage_torch_device) # ensure it is on cuda if it should (avoids synchronizations when indexing later on)
        env_indices = th.randint(low = 0, high=self.n_envs, size=(len(batch_inds),), device = self._storage_torch_device)

        data = TransitionBatch(
            observations={key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},
            actions=self.actions[batch_inds, env_indices],
            next_observations={key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()},
            terminated=self.terminated[batch_inds, env_indices].reshape(-1, 1),
            rewards=self.rewards[batch_inds, env_indices].reshape(-1, 1)
        )
        
        if self._copy_outputs:
            data = map_tensor_tree(data, lambda t: t.detach().clone())
        map_tensor_tree(data, lambda t: t.to(device = self.th_device, non_blocking=True))
        if self.th_device.type == "cuda":
            th.cuda.synchronize(self.th_device)
        dbg_check_finite(data)

        return data

    @override
    def storage_torch_device(self):
        return self._storage_torch_device

    @override
    def stored_frames(self):
        return (self.buffer_size if self.full else self.pos)*self.n_envs

    @override
    def update(self, buffer : ThDReplayBuffer):

        if (self.optimize_memory_usage or buffer.optimize_memory_usage):
            raise RuntimeError("Memory optimization is not supported")
        prev_size = self.size()
        copied = 0
        while copied < buffer.size():
            space_to_end = self.buffer_size - self.pos
            to_copy = min(space_to_end, buffer.size()-copied)

            self.actions[self.pos:self.pos + to_copy] = buffer.actions[copied:copied+to_copy]
            self.rewards[self.pos:self.pos + to_copy] = buffer.rewards[copied:copied+to_copy]
            self.terminated[self.pos:self.pos + to_copy]   = buffer.terminated[copied:copied+to_copy]

            for key in self.observations.keys():
                self.observations[key][self.pos:self.pos + to_copy] = buffer.observations[key][copied:copied+to_copy]
            for key in self.next_observations.keys():
                self.next_observations[key][self.pos:self.pos + to_copy] = buffer.next_observations[key][copied:copied+to_copy]
                
            self.truncated[self.pos:self.pos + to_copy] = buffer.truncated[copied:copied+to_copy]
                
            self.pos += to_copy
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
            copied += to_copy     
        new_size = self.size()
        if new_size-prev_size != buffer.size() and not self.full: raise RuntimeError(f"Error updating buffer {new_size}-{prev_size}!={buffer.size()}")       


    # def replay(self):
    #     pos = self.pos if self.full else 0
    #     ret_vsteps = 0
    #     while ret_vsteps < self.size():
    #         obs = {k:v[pos] for k,v in self.observations.items()}
    #         next_obs = {k:v[pos] for k,v in self.next_observations.items()}
    #         action = self.actions[pos]
    #         reward = self.rewards[pos]
    #         done = self.dones[pos]
    #         truncated = self.truncated[pos]
    #         yield (obs, next_obs, action, reward, done, truncated)
    #         pos = (pos+1) % self.buffer_size


class GenericHerReplayBuffer(HerReplayBuffer):
    def __init__(self,
                    buffer_size : int,
                    observation_space: spaces.Space,
                    action_space: spaces.Space,
                    device: Union[th.device, str] = "cpu",
                    n_envs: int = 1,
                    optimize_memory_usage: bool = False,
                    env: VecEnv = None,
                    buffer_class = None,
                    buffer_kwargs = None,
                    max_episode_length: Optional[int] = None,
                    n_sampled_goal: int = 4,
                    goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
                    online_sampling: bool = True,
                    handle_timeout_termination: bool = True):

        if online_sampling:
            if buffer_class is not None:
                raise AttributeError("Cannot specify replay_buffer_class with online_sampling")
            replay_buffer = None
        else:
            replay_buffer = buffer_class(   buffer_size,
                                            observation_space,
                                            action_space,
                                            device = device,
                                            n_envs = n_envs,
                                            optimize_memory_usage = optimize_memory_usage,
                                            **buffer_kwargs)
        super().__init__(env = env,
                         buffer_size = buffer_size,
                         device = device,
                         replay_buffer = replay_buffer,
                         max_episode_length = max_episode_length,
                         n_sampled_goal = n_sampled_goal,
                         goal_selection_strategy = goal_selection_strategy,
                         online_sampling = online_sampling,
                         handle_timeout_termination = handle_timeout_termination)

    
    def memory_size(self):
        return self.replay_buffer.memory_size()

    
    def storage_torch_device(self):
        return self.replay_buffer.storage_torch_device()