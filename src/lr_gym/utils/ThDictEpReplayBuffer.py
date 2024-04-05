from cmath import inf
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer, DictReplayBufferSamples
import numpy as np
from gymnasium import spaces
from typing import Union, List, Dict, Any, Optional
import torch as th
from stable_baselines3.common.vec_env import VecNormalize
import psutil
import warnings
import time
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.utils.buffers import numpy_to_torch_dtype
import random
import threading


def take_frames(buff, episodes, frames):
    # ggLog.info(f"episodes.size() = {episodes.size()}")
    # ggLog.info(f"frames.size() = {frames.size()}")
    max_ep_len = buff.size()[1]
    elem_size = buff.size()[2:]
    # ggLog.info(f"max_ep_len = {max_ep_len}")
    flat_indexes = ((max_ep_len*episodes).unsqueeze(1) + frames).flatten()
    # ggLog.info(f"flat_indexes.size() = {flat_indexes.size()}")
    flat_buff = buff.view(buff.size()[0]*max_ep_len,-1)
    # ggLog.info(f"flat_buff.size() = {flat_buff.size()}")
    selected = flat_buff[flat_indexes].view((episodes.size()[0],frames.size()[1],)+elem_size)
    # ggLog.info(f"selected.size() = {selected.size()}")
    return selected


class ThDictEpReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    class Storage():
        def __init__(self, episodes_num, max_episode_duration, buffer, min_ep_length):
            self._added_episodes = 0
            self._stored_frames_count = 0
            self._max_episodes = episodes_num
            self.full = False
            self._storage_torch_device = buffer._storage_torch_device
            self._output_device = th.device(buffer.device)
            self._min_ep_length = min_ep_length
            self._max_episode_duration = max_episode_duration
            self._buffer = buffer

            #TODO: Make more efficient by using only one observation buffer
            self.observations = {
                key: th.zeros(  size = (episodes_num, max_episode_duration) + _obs_shape,
                                dtype=numpy_to_torch_dtype(buffer._observation_space[key].dtype),
                                device = self._storage_torch_device)
                for key, _obs_shape in buffer.obs_shape.items()
            }
            self.next_observations = {
                key: th.zeros(  size = (episodes_num, max_episode_duration) + _obs_shape,
                                dtype=numpy_to_torch_dtype(buffer._observation_space[key].dtype),
                                device = self._storage_torch_device)
                for key, _obs_shape in buffer.obs_shape.items()
            }

            self.actions = th.zeros(size = (episodes_num, max_episode_duration, buffer.action_dim),
                                    dtype=numpy_to_torch_dtype(buffer._action_space.dtype),
                                    device = self._storage_torch_device)
            self.rewards = th.zeros(size = (episodes_num, max_episode_duration,),
                                    dtype=th.float32,
                                    device = self._storage_torch_device)
            self.dones = th.zeros(size = (episodes_num, max_episode_duration,), dtype=th.float32,
                                device = self._storage_torch_device)

            self.timeouts = th.zeros(size = (episodes_num, max_episode_duration), dtype=th.float32,  device = buffer._storage_torch_device)


            self.remaining_frames = th.zeros(size = (episodes_num, max_episode_duration),
                                        dtype=th.float32,
                                        device = self._storage_torch_device)
            
            if self._storage_torch_device.type == "cpu":
                for k in self.observations.keys():
                    self.observations[k] = self.observations[k].pin_memory()
                    self.next_observations[k] = self.next_observations[k].pin_memory()
                self.actions = self.actions.pin_memory()
                self.rewards = self.rewards.pin_memory()
                self.dones = self.dones.pin_memory()
                self.timeouts = self.timeouts.pin_memory()

            # ggLog.info(f"Allocated buffer of size {self.memory_size()/1024/1024/1024} GiB")
        
        def memory_size(self):
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.element_size()*obs.nelement()
            next_obs_nbytes = obs_nbytes
            
            action_nbytes = self.actions.element_size()*self.actions.nelement()
            rewards_nbytes = self.rewards.element_size()*self.rewards.nelement()
            dones_nbytes = self.dones.element_size()*self.dones.nelement()
            timeouts_nbytes = self.timeouts.element_size()*self.timeouts.nelement()
            remaining_frames_nbytes = self.remaining_frames.element_size()*self.remaining_frames.nelement()

            total_memory_usage = obs_nbytes + next_obs_nbytes + action_nbytes + rewards_nbytes + dones_nbytes + timeouts_nbytes + remaining_frames_nbytes

            return total_memory_usage
        
        def add_episode(self, buf, ep_len):
            prev_buff_size = self._stored_frames_count
            ep_idx = self._added_episodes%self._max_episodes
            overridden_frames = self.remaining_frames[ep_idx][0]
            if self._added_episodes>=self._max_episodes:
                self._stored_frames_count -= self.remaining_frames[ep_idx][0] # remove these frames, they will be overridden
            for key in self.observations.keys():
                self.observations[key][ep_idx] = buf.observations[key][0]
                self.next_observations[key][ep_idx] = buf.next_observations[key][0]
            self.actions[ep_idx] = buf.actions[0]
            self.rewards[ep_idx] = buf.rewards[0]
            self.dones[ep_idx] = buf.dones[0]
            self.remaining_frames[ep_idx] = buf.remaining_frames[0]
            self.timeouts[ep_idx] = buf.timeouts[0]
            self._added_episodes += 1
            self._stored_frames_count += ep_len

            if self._added_episodes>=self._max_episodes:
                self.full = True

            if abs(prev_buff_size - self._stored_frames_count) > self._max_episode_duration:
                ggLog.error(f"Buffer size changed from {prev_buff_size} to {self._stored_frames_count} input ep_len={ep_len}, overridden_frames = {overridden_frames}")

            
        def stored_episodes(self):
            return min(self._added_episodes, self._max_episodes)
        
        def stored_frames(self):
            # ggLog.warn(f"buffer size = {self._stored_frames_count}")
            return self._stored_frames_count
        
        def size(self):
            return self.stored_frames()
        
        def sample(self, batch_size: int, env: Optional[VecNormalize] = None, sample_duration = None) -> DictReplayBufferSamples:
            
            return_traj = sample_duration is not None
            if sample_duration is None:
                sample_duration = 1
            
            upper_bound = self.stored_episodes()
            sampled_episodes = th.empty((batch_size,), dtype=th.int64, device = self._storage_torch_device)
            valid_count = 0
            iter = 0
            while valid_count<batch_size:
                if iter>1000:
                    raise RuntimeError(f"Unable to sample episodes long at least {sample_duration}, got {valid_count} out of {batch_size}")
                new_sampled_episodes = th.randint(0, upper_bound, size=(batch_size-valid_count,), device = self._storage_torch_device)
                ep_lenghts = self.remaining_frames[new_sampled_episodes][:,0]
                # ggLog.info(f"new_sampled_episodes.size() = {new_sampled_episodes.size()}")
                # ggLog.info(f"ep_lenghts.size() = {ep_lenghts.size()}")
                if self._min_ep_length > sample_duration:
                    sampled_episodes = new_sampled_episodes
                    break
                valid_idx = (ep_lenghts>=sample_duration).nonzero().squeeze()
                # ggLog.info(f"valid_idx.size() = {valid_idx.size()}")
                sampled_episodes[valid_count:valid_count+len(valid_idx)] = new_sampled_episodes[valid_idx]
                valid_count += len(valid_idx)
                iter += 1
            ep_lenghts = self.remaining_frames[sampled_episodes][:,0]
            ranges = ep_lenghts-sample_duration+1
            # ggLog.info(f"ranges.size() = {ranges.size()}")
            # ggLog.info(f"sample_duration = {sample_duration}")
            sampled_start_frames = (ranges*th.rand(size = sampled_episodes.size(), device=self._storage_torch_device)).int()
          
            
            trajs_num = len(sampled_episodes)
            trajs_len = sample_duration

            # ggLog.info(f"trajs_num = {trajs_num}, trajs_len = {trajs_len}")
            # ggLog.info(f"sampled_start_frames.size() = {sampled_start_frames.size()}")


            sampled_frames = sampled_start_frames.unsqueeze(1) + th.arange(sample_duration, device = sampled_start_frames.device)

            trajs_obs_ =        {key:take_frames(b, sampled_episodes, sampled_frames).to(self._output_device) 
                                        for key, b in self.observations.items()} 
            trajs_next_obs_ =   {key:take_frames(b, sampled_episodes, sampled_frames).to(self._output_device) 
                                        for key, b in self.next_observations.items()} 
            trajs_dones = take_frames(self.dones, sampled_episodes, sampled_frames).view(trajs_num,trajs_len,1).to(self._output_device)
            trajs_timeouts = take_frames(self.timeouts,sampled_episodes, sampled_frames).view(trajs_num,trajs_len,1).to(self._output_device)
            trajs_actions = take_frames(self.actions, sampled_episodes, sampled_frames).to(self._output_device) #.view(trajs_num,trajs_len,-1)
            trajs_rewards = take_frames(self.rewards, sampled_episodes, sampled_frames).view(trajs_num,trajs_len,1).to(self._output_device)

            for key in self.observations:
                obs_shape = trajs_obs_[key].size()[2:]
                # ggLog.info(f"obs_shape = {obs_shape}")
                trajs_obs_[key] = self._buffer._normalize_obs(trajs_obs_[key].view((trajs_num*trajs_len,)+obs_shape),
                                                    env).view((trajs_num,trajs_len,)+obs_shape)
                trajs_next_obs_[key] = self._buffer._normalize_obs(trajs_next_obs_[key].view((trajs_num*trajs_len,)+obs_shape),
                                                        env).view((trajs_num,trajs_len,)+obs_shape)
            # ggLog.info(f"1 trajs_dones.size() = {trajs_dones.size()}")
            # ggLog.info(f"trajs_timeouts.size() = {trajs_timeouts.size()}")
            trajs_dones = (trajs_dones*(1-trajs_timeouts)).view(trajs_num,trajs_len,1)
            # ggLog.info(f"2 trajs_dones.size() = {trajs_dones.size()}")
            trajs_rewards = self._buffer._normalize_reward(trajs_rewards.view(trajs_num*trajs_len,1)).view(trajs_num,trajs_len,1)

            if return_traj:
                observations = trajs_obs_
                actions = trajs_actions
                next_observations = trajs_next_obs_
                dones = trajs_dones
                rewards = trajs_rewards
            else:
                observations = {}
                next_observations = {}
                for key in self.observations:
                    obs_shape = trajs_obs_[key].size()[2:]
                    observations[key] = trajs_obs_[key].view((trajs_num,)+obs_shape)
                    next_observations[key] = trajs_next_obs_[key].view((trajs_num,)+obs_shape)
                actions = trajs_actions.view((trajs_num,)+trajs_actions.size()[2:])
                # ggLog.info(f"3 trajs_dones.size() = {trajs_dones.size()}")
                dones = trajs_dones.view(trajs_num,1)
                rewards = trajs_rewards.view(trajs_num,1)

            # ggLog.info(f"observations = {[str(k)+':'+str(v.size()) for k,v in observations.items()]}")
            # ggLog.info(f"next_observations = {[str(k)+':'+str(v.size()) for k,v in next_observations.items()]}")
            # ggLog.info(f"actions = {actions.size()}")
            # ggLog.info(f"dones = {dones.size()}")
            # ggLog.info(f"sampled rewards {trajs_num} trajs of {trajs_len}, rewards = {rewards.size()}")
            

            # ggLog.info(f"actions.device = {actions.device}")

            return DictReplayBufferSamples(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                dones=dones,
                rewards=rewards
            )
        


        def update(self, src_storage):
            
            # ggLog.info(f"Updating storage of size {self.stored_frames()} with storage of size {src_storage.stored_frames()}")
            prev_size = self.size()
            copied = 0
            overridden_frames = 0
            while copied < src_storage.stored_episodes():
                pos = self._added_episodes%self._max_episodes
                space_to_end = self._max_episodes - pos
                to_copy = min(space_to_end, src_storage.stored_episodes()-copied) # either the dest space or the remaining stuff to copy

                if self._added_episodes>=self._max_episodes:
                    # it's always covering either a completely empty area or a completely full one
                    # if self is full the overridden frames this round are:
                    overridden_frames += th.sum(self.remaining_frames[pos:pos + to_copy,0])

                self.actions[pos:pos + to_copy] = src_storage.actions[copied:copied+to_copy]
                self.rewards[pos:pos + to_copy] = src_storage.rewards[copied:copied+to_copy]
                self.dones[pos:pos + to_copy]   = src_storage.dones[copied:copied+to_copy]
                self.timeouts[pos:pos + to_copy] = src_storage.timeouts[copied:copied+to_copy]
                self.remaining_frames[pos:pos + to_copy] = src_storage.remaining_frames[copied:copied+to_copy]

                for key in self.observations.keys():
                    self.observations[key][pos:pos + to_copy] = src_storage.observations[key][copied:copied+to_copy]
                    self.next_observations[key][pos:pos + to_copy] = src_storage.next_observations[key][copied:copied+to_copy]                    
                

                self._added_episodes += to_copy
                copied += to_copy
            self._stored_frames_count += src_storage._stored_frames_count - overridden_frames
            # ggLog.info(f"self._stored_frames_count = {self._stored_frames_count}")
            new_size = self.size()
            if self._added_episodes>=self._max_episodes:
                self.full = True
            if new_size-prev_size != src_storage.size() and not self.full: raise RuntimeError(f"Error updating buffer {new_size}-{prev_size}!={src_storage.size()}")       


                    
















    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        storage_torch_device: str = "cpu",
        fallback_to_cpu_storage: bool = True,
        max_episode_duration = 1000,
        validation_buffer_size = 0,
        validation_holdout_ratio = 0,
        min_episode_duration = 0,
        disable_validation_set = True,
        fill_val_buffer_to_min_at_ep = float("+inf"),
        val_buffer_min_size = 0
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.max_frames = buffer_size
        self.buffer_size = max(buffer_size // n_envs, 1)
        self._observation_space = observation_space
        self._action_space = action_space
        self.handle_timeout_termination = handle_timeout_termination
        storage_torch_device = th.device(storage_torch_device)
        self._storage_torch_device = storage_torch_device
        if storage_torch_device == "cuda" and device == "cpu":
            raise AttributeError(f"Storage device is gpu, and output device is cpu. This doesn't make much sense. Use either [gpu,gpu], [cpu,gpu], or [cpu,cpu]")
        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        self.optimize_memory_usage = optimize_memory_usage
        self._max_episode_duration = max_episode_duration
        assert buffer_size%max_episode_duration == 0, f"buffer_size must be a multiple of max_episode_duration but they are respectively {buffer_size} and {max_episode_duration}"
        self._max_episodes = int(self.max_frames/max_episode_duration)
        self._validation_buffer_size = int(validation_buffer_size)
        self._max_val_episodes = int(self._validation_buffer_size/max_episode_duration)
        self._validation_holdout_ratio = validation_holdout_ratio
        self._min_episode_duration = min_episode_duration
        self._disable_validation_set = disable_validation_set
        if self._disable_validation_set:
            self._validation_holdout_ratio = -1
        else:
            assert validation_buffer_size%max_episode_duration == 0, f"validation_buffer_size must be a multiple of max_episode_duration bit they are respectively {buffer_size} and {max_episode_duration}"
        

        if self._storage_torch_device.type == "cuda" and fallback_to_cpu_storage:
            pred_avail = self.predict_memory_consumption()
            consumptionRatio = pred_avail[0]/pred_avail[1]
            if consumptionRatio>0.6:
                warnings.warn(   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                                f"Not enough memory on requested device {self._storage_torch_device} (Would consume {consumptionRatio*100:.0f}% = {pred_avail[0]/1024/1024/1024:.3f} GiB)\n"
                                 "Falling back to CPU memory\n"
                                 "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                time.sleep(3)
                self._storage_torch_device = th.device("cpu")
        pred_avail = self.predict_memory_consumption()
        consumptionRatio = pred_avail[0]/pred_avail[1]
        if consumptionRatio>0.6:
            warnings.warn(   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            f"Replay buffer will use {consumptionRatio*100:.0f}% ({pred_avail[0]/1024/1024/1024:.3f} GiB) of available memory on device {self._storage_torch_device}\n"
                             "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            time.sleep(3)
        # ggLog.info(f"Buffer will consume {consumptionRatio*100:.0f}% = {pred_avail[0]/1024/1024/1024:.3f} GiB")

        self._allocate_buffers(self._max_episodes, self._max_val_episodes)
        self._addcount = 0
        self._added_eps_count = 0
        self._fill_val_buffer_to_min_at_ep = fill_val_buffer_to_min_at_ep
        self._val_buff_min_size = val_buffer_min_size
        
    def validation_set_enabled(self):
        return not self._disable_validation_set
    
    def memory_size(self):
        return self._storage.memory_size() + self._validation_storage.memory_size()
    
    def predict_memory_consumption(self):
        testRatio = 0.01
        self._allocate_buffers(int(self._max_episodes*testRatio), int(self._max_val_episodes*testRatio))
        predicted_mem_usage = self.memory_size()/testRatio

        mem_available = float("+inf")
        if self._storage_torch_device.type == "cpu":
            mem_available = psutil.virtual_memory().available
        elif self._storage_torch_device.type == "cuda":
            mem_available = th.cuda.mem_get_info(self._storage_torch_device)[0]

        return predicted_mem_usage, mem_available


    def _allocate_buffers(self, episodes_number, validation_episodes_number):
        self._storage = ThDictEpReplayBuffer.Storage(episodes_number, self._max_episode_duration, self, self._min_episode_duration)
        self._validation_storage = ThDictEpReplayBuffer.Storage(validation_episodes_number, self._max_episode_duration, self, self._min_episode_duration)
        self._last_eps_buffers = [ThDictEpReplayBuffer.Storage(1, self._max_episode_duration, self, self._min_episode_duration) for _ in range(self.n_envs)]
        self._last_eps_lengths = [0 for _ in range(self.n_envs)] 


        

    def add(self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]]) -> None:
        # All inputs are batches of size n_envs
        # All should be copied to avoid modification by reference

        self._addcount+=1

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))
        for key in obs.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])

        # Experience is collected in parallel, but it is stored sequentially in the main buffer
        # To rearrange it we first save experience in a temporary buffer, then when
        # an episode is completed we move it to the main storage

        # copy each transition into the buffer for the respective env
        for env_idx in range(self.n_envs):
            buf = self._last_eps_buffers[env_idx]
            ep_frame = self._last_eps_lengths[env_idx]
            for key in obs.keys():
                buf.observations[key][0][ep_frame] = th.as_tensor(obs[key][env_idx]).to(self._storage_torch_device, non_blocking=True)
                buf.next_observations[key][0][ep_frame] = th.as_tensor(next_obs[key][env_idx]).to(self._storage_torch_device, non_blocking=True)
            buf.actions[0][ep_frame] = th.as_tensor(action[env_idx]).to(self._storage_torch_device, non_blocking=True)
            buf.rewards[0][ep_frame] = th.as_tensor(reward[env_idx]).to(self._storage_torch_device, non_blocking=True)
            buf.dones[0][ep_frame]   = th.as_tensor(done[env_idx]).to(self._storage_torch_device, non_blocking=True)
            if self.handle_timeout_termination:
                buf.timeouts[0][ep_frame] = th.as_tensor(infos[env_idx].get("TimeLimit.truncated", False)).to(self._storage_torch_device, non_blocking=True)
            
            self._last_eps_lengths[env_idx] += 1
            # If it is done copy it to the main buffer
            if done[env_idx]:
                # ggLog.info(f"Storing completed episode of {self._last_eps_lengths[env_idx]} steps")
                r = np.random.random()
                # ggLog.info(f"{r}<{self._validation_holdout_ratio}?")
                if (r<self._validation_holdout_ratio or 
                    (self._added_eps_count > self._fill_val_buffer_to_min_at_ep and self._val_buff_min_size > self._validation_storage.size())):
                    store = self._validation_storage
                else:
                    store = self._storage
                buf.remaining_frames[0] = self._last_eps_lengths[env_idx] - th.arange(self._max_episode_duration, device=self._storage_torch_device)
                # store.add_episode(buf, ep_len = self._last_eps_lengths[env_idx])
                store.update(buf)
                buf.remaining_frames[0][:] = 0
                self._last_eps_lengths[env_idx] = 0
                self._added_eps_count += 1

        th.cuda.current_stream().synchronize() #Wait for non_blocking transfers (they are not automatically synchronized when used as inputs! https://discuss.pytorch.org/t/how-to-wait-on-non-blocking-copying-from-gpu-to-cpu/157010/2)

        # ggLog.info(f"{threading.get_ident()}: Added step, count = {self._addcount}, size = {self.size()}, val_size = {self.size(validation_set=True)}")
            
            
    def stored_episodes(self, validation_set = False):
        if validation_set and not self._disable_validation_set:
            return self._validation_storage.stored_episodes()
        else:
            return self._storage.stored_episodes()
    
    def stored_frames(self, validation_set = False):
        if validation_set and not self._disable_validation_set:
            return self._validation_storage.stored_frames()
        else:
            return self._storage.stored_frames()
    
    def size(self, validation_set = False):
        return self.stored_frames(validation_set=validation_set)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, sample_duration = None, validation_set : bool = False) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if (self.optimize_memory_usage):
            raise RuntimeError("Memory optimization is not supported")
        if validation_set and not self._disable_validation_set:
            # ggLog.info(f"Sampling validation set")
            return self._validation_storage.sample(batch_size, env, sample_duration)
        else:
            return self._storage.sample(batch_size, env, sample_duration)

    def _get_samples(self, sampled_episodes, sampled_start_frames, env: Optional[VecNormalize] = None, sample_duration = 1) -> DictReplayBufferSamples:
        raise NotImplementedError()


    def storage_torch_device(self):
        return self._storage_torch_device

    
    def update(self, src_buffer):
        self._storage.update(src_buffer._storage)
        self._validation_storage.update(src_buffer._validation_storage)


