from __future__ import annotations
import numpy as np
from gymnasium import spaces
from typing import Union, List, Dict, Any, Optional
import torch as th
from stable_baselines3.common.vec_env import VecNormalize
import psutil
import warnings
import time
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.buffers import numpy_to_torch_dtype, TransitionBatch, BaseValidatingBuffer
from adarl.utils.tensor_trees import is_all_finite, map_tensor_tree
from typing_extensions import override
from adarl.utils.dbg.dbg_checks import dbg_check_finite

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



class EpisodeStorage():
    def __init__(self, episodes_num, max_episode_duration, buffer, min_ep_length):
        self._max_episodes = episodes_num
        self._storage_torch_device = buffer._storage_torch_device
        self._output_device = th.device(buffer.device)
        self._min_ep_length = min_ep_length
        self._max_episode_duration = max_episode_duration
        self._buffer = buffer

        self._added_episodes = 0
        self._stored_frames_count = 0
        self._stored_episodes = 0
        self._current_ep_frame_count = 0
        self.full = False
        self._use_nonblocking_adds = self._storage_torch_device.type == "cuda"

        #TODO: Make more efficient by using only one observation buffer
        self.observations = {
            key: th.full(  fill_value=42,
                            size = (self._max_episodes, self._max_episode_duration) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._buffer._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self._buffer.obs_shape.items()
        }
        self.next_observations = {
            key: th.full(  fill_value=42,
                            size = (self._max_episodes, self._max_episode_duration) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._buffer._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self._buffer.obs_shape.items()
        }

        self.actions = th.full( fill_value=42,
                                size = (self._max_episodes, self._max_episode_duration, self._buffer.action_dim),
                                dtype=numpy_to_torch_dtype(self._buffer._action_space.dtype),
                                device = self._storage_torch_device)
        self.rewards = th.full( fill_value=42,
                                size = (self._max_episodes, self._max_episode_duration,), dtype=th.float32,
                                device = self._storage_torch_device)
        self.terminated = th.full(  fill_value=42,
                                    size = (self._max_episodes, self._max_episode_duration,), dtype=th.uint8,
                                    device = self._storage_torch_device)
        self.truncated = th.full(   fill_value=42,
                                    size = (self._max_episodes, self._max_episode_duration), dtype=th.uint8,
                                    device = self._buffer._storage_torch_device)


        self.episode_durations = th.zeros(size = (self._max_episodes,),
                                    dtype=th.int32,
                                    device = self._storage_torch_device)
        
        if self._storage_torch_device.type == "cpu" and self._use_nonblocking_adds:
            for k in self.observations.keys():
                self.observations[k] = self.observations[k].pin_memory()
                self.next_observations[k] = self.next_observations[k].pin_memory()
            self.actions = self.actions.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.terminated = self.terminated.pin_memory()
            self.truncated = self.truncated.pin_memory()


    def clear(self):
        self._added_episodes = 0
        self._stored_frames_count = 0
        self._stored_episodes = 0
        self._current_ep_frame_count = 0
        self.full = False
        self.episode_durations[:] = 0

    def memory_size(self):
        obs_nbytes = 0
        for _, obs in self.observations.items():
            obs_nbytes += obs.element_size()*obs.nelement()
        next_obs_nbytes = obs_nbytes
        
        action_nbytes = self.actions.element_size()*self.actions.nelement()
        rewards_nbytes = self.rewards.element_size()*self.rewards.nelement()
        terminated_nbytes = self.terminated.element_size()*self.terminated.nelement()
        timeouts_nbytes = self.truncated.element_size()*self.truncated.nelement()
        ep_durations = self.episode_durations.element_size()*self.episode_durations.nelement()

        total_memory_usage = obs_nbytes + next_obs_nbytes + action_nbytes + rewards_nbytes + terminated_nbytes + timeouts_nbytes + ep_durations

        return total_memory_usage
    
    def add_frame(self, observation, action, next_observation, reward, terminated, truncated, sync_stream = True):
        ep_idx = self._added_episodes%self._max_episodes
        frame_idx = self._current_ep_frame_count

        dbg_check_finite((observation, next_observation, action, reward))
        # if not is_all_finite(observation):
        #     raise RuntimeError(f"nonfinite values in added observation "
        #                        f"isnan_count = {map_tensor_tree(observation, lambda t: th.sum(th.isnan(t)))} "
        #                        f"isinf_count = {map_tensor_tree(observation, lambda t: th.sum(th.isinf(t)))} "
        #                        f"{observation} ")
        # if not is_all_finite(next_observation):
        #     raise RuntimeError(f"nonfinite values in added next_observation "
        #                        f"isnan_count = {map_tensor_tree(next_observation, lambda t: th.sum(th.isnan(t)))} "
        #                        f"isinf_count = {map_tensor_tree(next_observation, lambda t: th.sum(th.isinf(t)))} "
        #                        f"{next_observation}")
        # if not is_all_finite(action):
        #     raise RuntimeError(f"nonfinite values in added action isnan_count = {th.sum(th.isnan(action))} {action}")
        # if not is_all_finite(reward):
        #     raise RuntimeError(f"nonfinite values in added reward isnan_count = {th.sum(th.isnan(reward))} {reward}")


        if self._current_ep_frame_count == 0:
            self._stored_frames_count -= self.episode_durations[ep_idx].cpu().item()
            if self._added_episodes < self._max_episodes:
                self._stored_episodes += 1
            self.episode_durations[ep_idx] = 0    
        nb = self._use_nonblocking_adds
        for key in self.observations.keys():
            self.observations[key][ep_idx,frame_idx].copy_(th.as_tensor(observation[key]), non_blocking=nb)
            self.next_observations[key][ep_idx,frame_idx].copy_(th.as_tensor(next_observation[key]), non_blocking=nb)
        self.actions[ep_idx,frame_idx].copy_(th.as_tensor(action), non_blocking=nb)
        self.rewards[ep_idx,frame_idx].copy_(th.as_tensor(reward), non_blocking=nb)
        self.terminated[ep_idx,frame_idx].copy_(th.as_tensor(terminated), non_blocking=nb)
        self.truncated[ep_idx,frame_idx].copy_(th.as_tensor(truncated), non_blocking=nb)
        self.episode_durations[ep_idx] += 1
        self._current_ep_frame_count += 1
        self._stored_frames_count += 1
        if terminated or truncated:
            self._current_ep_frame_count = 0
            self._added_episodes += 1
        if self._added_episodes>=self._max_episodes:
            self.full = True
            
        if sync_stream:
            th.cuda.current_stream().synchronize() # sync non_blocking operations
        # ggLog.info(f"EpisodeStorage{id(self)}: added frame {ep_idx},{frame_idx}. term={terminated} trunc={truncated}")

    
    # def add_episode(self, buf : Storage, ep_len):
    #     prev_buff_size = self._stored_frames_count
    #     ep_idx = self._added_episodes%self._max_episodes
    #     overridden_frames = self.remaining_frames[ep_idx][0]
    #     if self._added_episodes>=self._max_episodes:
    #         self._stored_frames_count -= self.remaining_frames[ep_idx][0] # remove these frames, they will be overridden
    #     for key in self.observations.keys():
    #         self.observations[key][ep_idx] = buf.observations[key][0]
    #         self.next_observations[key][ep_idx] = buf.next_observations[key][0]
    #     self.actions[ep_idx] = buf.actions[0]
    #     self.rewards[ep_idx] = buf.rewards[0]
    #     self.dones[ep_idx] = buf.dones[0]
    #     self.truncated[ep_idx] = buf.truncated[0]
    #     self.remaining_frames[ep_idx] = buf.remaining_frames[0]
    #     self._added_episodes += 1
    #     self._stored_frames_count += ep_len

    #     if self._added_episodes>=self._max_episodes:
    #         self.full = True

    #     if abs(prev_buff_size - self._stored_frames_count) > self._max_episode_duration:
    #         ggLog.error(f"Buffer size changed from {prev_buff_size} to {self._stored_frames_count} input ep_len={ep_len}, overridden_frames = {overridden_frames}")

        
    def stored_episodes(self):
        return min(self._added_episodes, self._max_episodes)
    
    def stored_frames(self):
        # ggLog.warn(f"buffer size = {self._stored_frames_count}")
        return self._stored_frames_count
    
    def size(self):
        return self.stored_frames()
    
    def sample(self, batch_size: int, sample_duration = None) -> TransitionBatch:
        
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
            ep_lengths = self.episode_durations[new_sampled_episodes]
            # ggLog.info(f"new_sampled_episodes.size() = {new_sampled_episodes.size()}")
            # ggLog.info(f"ep_lenghts.size() = {ep_lenghts.size()}")
            if self._min_ep_length > sample_duration:
                sampled_episodes = new_sampled_episodes
                break
            valid_eps_idxs = (ep_lengths>=sample_duration).nonzero().squeeze()
            # ggLog.info(f"valid_idx.size() = {valid_idx.size()}")
            sampled_episodes[valid_count:valid_count+len(valid_eps_idxs)] = new_sampled_episodes[valid_eps_idxs]
            valid_count += len(valid_eps_idxs)
            iter += 1
        ep_lengths = self.episode_durations[sampled_episodes]
        ranges = ep_lengths-sample_duration+1
        # ggLog.info(f"ranges.size() = {ranges.size()}")
        # ggLog.info(f"sample_duration = {sample_duration}")
        sampled_start_frames = (ranges*th.rand(size = sampled_episodes.size(), device=self._storage_torch_device)).int()
        
        
        trajs_num = len(sampled_episodes)
        trajs_len = sample_duration

        # ggLog.info(f"trajs_num = {trajs_num}, trajs_len = {trajs_len}")
        # ggLog.info(f"sampled_start_frames.size() = {sampled_start_frames.size()}")


        sampled_frames = sampled_start_frames.unsqueeze(1) + th.arange(sample_duration, device = sampled_start_frames.device)

        # ggLog.info(f"EpisodeStorage{id(self)}: sampled {list(zip(sampled_episodes,sampled_frames))}")

        trajs_obs_ =        {key:take_frames(b, sampled_episodes, sampled_frames).to(self._output_device, non_blocking=self._output_device.type=="cuda") 
                                    for key, b in self.observations.items()} 
        trajs_next_obs_ =   {key:take_frames(b, sampled_episodes, sampled_frames).to(self._output_device, non_blocking=self._output_device.type=="cuda") 
                                    for key, b in self.next_observations.items()} 
        trajs_terminateds = take_frames(self.terminated, sampled_episodes, sampled_frames).view(trajs_num,trajs_len,1).to(self._output_device, non_blocking=self._output_device.type=="cuda")
        trajs_actions = take_frames(self.actions, sampled_episodes, sampled_frames).to(self._output_device, non_blocking=self._output_device.type=="cuda") #.view(trajs_num,trajs_len,-1)
        trajs_rewards = take_frames(self.rewards, sampled_episodes, sampled_frames).view(trajs_num,trajs_len,1).to(self._output_device, non_blocking=self._output_device.type=="cuda")

        for key in self.observations:
            obs_shape = trajs_obs_[key].size()[2:]
            # ggLog.info(f"obs_shape = {obs_shape}")
            trajs_obs_[key] = trajs_obs_[key].view((trajs_num,trajs_len,)+obs_shape)
            trajs_next_obs_[key] = trajs_next_obs_[key].view((trajs_num,trajs_len,)+obs_shape)
        # ggLog.info(f"1 trajs_dones.size() = {trajs_dones.size()}")
        # ggLog.info(f"trajs_timeouts.size() = {trajs_timeouts.size()}")
        # ggLog.info(f"2 trajs_dones.size() = {trajs_dones.size()}")
        trajs_rewards = trajs_rewards.view(trajs_num,trajs_len,1)

        if return_traj:
            observations = trajs_obs_
            actions = trajs_actions
            next_observations = trajs_next_obs_
            terminateds = trajs_terminateds
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
            terminateds = trajs_terminateds.view(trajs_num,1)
            rewards = trajs_rewards.view(trajs_num,1)

        # ggLog.info(f"observations = {[str(k)+':'+str(v.size()) for k,v in observations.items()]}")
        # ggLog.info(f"next_observations = {[str(k)+':'+str(v.size()) for k,v in next_observations.items()]}")
        # ggLog.info(f"actions = {actions.size()}")
        # ggLog.info(f"dones = {dones.size()}")
        # ggLog.info(f"sampled rewards {trajs_num} trajs of {trajs_len}, rewards = {rewards.size()}")
        

        # ggLog.info(f"actions.device = {actions.device}")
        dbg_check_finite((observations, next_observations, actions, rewards))

        # if not is_all_finite(observations):
        #     raise RuntimeError(f"nonfinite values in sampled observation "
        #                        f"isnan_count = {map_tensor_tree(observations, lambda t: th.sum(th.isnan(t)))}"
        #                        f"isinf_count = {map_tensor_tree(observations, lambda t: th.sum(th.isinf(t)))}"
        #                        f"{observations} ")
        # if not is_all_finite(next_observations):
        #     raise RuntimeError(f"nonfinite values in sampled next_observation "
        #                        f"isnan_count = {map_tensor_tree(next_observations, lambda t: th.sum(th.isnan(t)))}"
        #                        f"isinf_count = {map_tensor_tree(next_observations, lambda t: th.sum(th.isinf(t)))}"
        #                        f"{next_observations}")
        # if not is_all_finite(actions):
        #     raise RuntimeError(f"nonfinite values in sampled action isnan_count = {th.sum(th.isnan(actions))} {actions}")
        # if not is_all_finite(rewards):
        #     raise RuntimeError(f"nonfinite values in sampled reward isnan_count = {th.sum(th.isnan(rewards))} {rewards}")

        return TransitionBatch(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            terminated=terminateds,
            rewards=rewards
        )
    


    def update(self, src_storage : EpisodeStorage):
        
        # ggLog.info(f"Updating storage of size {self.stored_frames()} (id={id(self)}) with storage of size {src_storage.stored_frames()} (id={id(src_storage)})")
        prev_size = self.size()
        copied_eps = 0
        overridden_frames = 0
        while copied_eps < src_storage.stored_episodes():
            pos = self._added_episodes%self._max_episodes
            eps_to_end = self._max_episodes - pos
            eps_to_copy = min(eps_to_end, src_storage.stored_episodes()-copied_eps) # either the dest space or the remaining stuff to copy

            if self._added_episodes>=self._max_episodes:
                # it's always covering either a completely empty area or a completely full one
                # if self is full the overridden frames this round are:
                overridden_frames += th.sum(self.episode_durations[pos:pos + eps_to_copy])
            else:
                overridden_frames = 0

            # ggLog.info(f"Copying {copied_eps}:{copied_eps+eps_to_copy} to {pos}:{pos + eps_to_copy}")
            # ggLog.info(f"Overwriting {overridden_frames} frames")

            self.actions[pos:pos + eps_to_copy] = src_storage.actions[copied_eps:copied_eps+eps_to_copy]
            self.rewards[pos:pos + eps_to_copy] = src_storage.rewards[copied_eps:copied_eps+eps_to_copy]
            self.terminated[pos:pos + eps_to_copy]   = src_storage.terminated[copied_eps:copied_eps+eps_to_copy]
            self.truncated[pos:pos + eps_to_copy] = src_storage.truncated[copied_eps:copied_eps+eps_to_copy]
            self.episode_durations[pos:pos + eps_to_copy] = src_storage.episode_durations[copied_eps:copied_eps+eps_to_copy]

            for key in self.observations.keys():
                self.observations[key][pos:pos + eps_to_copy] = src_storage.observations[key][copied_eps:copied_eps+eps_to_copy]
                self.next_observations[key][pos:pos + eps_to_copy] = src_storage.next_observations[key][copied_eps:copied_eps+eps_to_copy]   
            
            self._added_episodes += eps_to_copy
            copied_eps += eps_to_copy
        self._stored_frames_count += src_storage._stored_frames_count - overridden_frames
        # ggLog.info(f"new self._stored_frames_count = {self._stored_frames_count}")
        # ggLog.info(f"new self._added_episodes = {self._added_episodes}")
        new_size = self.size()
        if self._added_episodes>=self._max_episodes:
            self.full = True
        if new_size-prev_size != src_storage.size() and not self.full: raise RuntimeError(f"Error updating buffer {new_size}-{prev_size}!={src_storage.size()}")       


    def replay(self):
        ep = self._added_episodes%self._max_episodes if self.full else 0
        frame = 0
        eps = self._stored_episodes
        ret_eps = 0
        while ret_eps < eps:
            obs = {k:v[ep,frame] for k,v in self.observations.items()}
            next_obs = {k:v[ep,frame] for k,v in self.next_observations.items()}
            action = self.actions[ep,frame]
            reward = self.rewards[ep,frame]
            done = self.terminated[ep,frame]
            truncated = self.truncated[ep,frame]
            yield (obs, next_obs, action, reward, done, truncated)
            frame += 1
            if done: ep = (ep+1) % self._max_episodes

                

class ThDictEpReplayBuffer(BaseValidatingBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_episode_duration : int | float,
        device: th.device = th.device("cpu"),
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        storage_torch_device: th.device = th.device("cpu"),
        fallback_to_cpu_storage: bool = True,
        validation_buffer_size : int = 0,
        validation_holdout_ratio : float = 0.0,
        min_episode_duration : int = 0,
        disable_validation_set : bool = True,
        fill_val_buffer_to_min_at_ep : float = float("+inf"),
        fill_val_buffer_to_min_at_step : float = float("+inf"),
        val_buffer_min_size : int = 0
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.max_frames = buffer_size
        self.buffer_size = max(buffer_size // n_envs, 1)
        self._observation_space = observation_space
        self._action_space = action_space
        storage_torch_device = th.device(storage_torch_device)
        self._storage_torch_device = storage_torch_device
        if storage_torch_device.type == "cuda" and device.type == "cpu":
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
        if consumptionRatio > 1.0:
            raise RuntimeError(f"Not enough memory on device {self.storage_torch_device}, would use {consumptionRatio*100:.0f}% ({pred_avail[0]/1024/1024/1024:.3f} GiB) of available memory")
        # ggLog.info(f"Buffer will consume {consumptionRatio*100:.0f}% = {pred_avail[0]/1024/1024/1024:.3f} GiB")

        self._allocate_buffers(self._max_episodes, self._max_val_episodes)
        self._addcount = 0
        self._collected_frames = 0
        self._added_eps_count = 0
        self._fill_val_buffer_to_min_at_ep = fill_val_buffer_to_min_at_ep
        self._fill_val_buffer_to_min_at_step = fill_val_buffer_to_min_at_step
        self._val_buff_min_size = val_buffer_min_size
        
    def validation_set_enabled(self):
        return not self._disable_validation_set
    
    @override
    def memory_size(self):
        return self._storage.memory_size() + self._validation_storage.memory_size()
    
    @override
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
        self._storage = EpisodeStorage(episodes_number, self._max_episode_duration, self, self._min_episode_duration)
        self._validation_storage = EpisodeStorage(validation_episodes_number, self._max_episode_duration, self, self._min_episode_duration)
        self._last_eps_buffers = [EpisodeStorage(1, self._max_episode_duration, self, self._min_episode_duration) for _ in range(self.n_envs)]


    @override
    def add(self,
            obs: Dict[str, th.Tensor],
            next_obs: Dict[str, th.Tensor],
            action: th.Tensor,
            reward: th.Tensor,
            terminated: th.Tensor,
            truncated: th.Tensor) -> None:
        # All inputs are batches of size n_envs
        # All should be copied to avoid modification by reference

        self._addcount+=1

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))
        obs = {k:v for k,v in obs.items()} # shallow copy the observations
        next_obs = {k:v for k,v in next_obs.items()} # shallow copy the observations
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
            buf.add_frame(  observation = {k:v[env_idx] for k,v in obs.items()},
                            action = action[env_idx],
                            next_observation = {k:v[env_idx] for k,v in next_obs.items()},
                            reward = reward[env_idx],
                            terminated = terminated[env_idx],
                            truncated = truncated[env_idx],
                            sync_stream = False)
            self._collected_frames += 1

        th.cuda.current_stream().synchronize() #Wait for non_blocking transfers (they are not automatically synchronized when used as inputs! https://discuss.pytorch.org/t/how-to-wait-on-non-blocking-copying-from-gpu-to-cpu/157010/2)

        for env_idx in range(self.n_envs):
            # If it is done copy it to the main buffer
            if terminated[env_idx] or truncated[env_idx]:
                buf = self._last_eps_buffers[env_idx] 
                # ggLog.info(f"ThDictEpReplayBuffer: idx {env_idx} ep terminated, copying to main storage")
                r = np.random.random()
                should_fill_up_validation = self._added_eps_count > self._fill_val_buffer_to_min_at_ep or self.stored_frames() > self._fill_val_buffer_to_min_at_step
                if (r<self._validation_holdout_ratio or 
                    (should_fill_up_validation and self._val_buff_min_size > self._validation_storage.size())):
                    store = self._validation_storage
                    # ggLog.info(f"Putting ep in validation storage")
                else:
                    store = self._storage
                    # ggLog.info(f"Putting ep in training storage")
                # store.add_episode(buf, ep_len = self._last_eps_lengths[env_idx])
                store.update(buf)
                buf.clear()
                self._added_eps_count += 1
                # ggLog.info(f"training storage contains {self._storage.stored_episodes()} eps {self._storage.stored_frames()} frames")
                # ggLog.info(f"validation storage contains {self._validation_storage.stored_episodes()} eps {self._validation_storage.stored_frames()} frames")


        # ggLog.info(f"{threading.get_ident()}: Added step, count = {self._addcount}, size = {self.size()}, val_size = {self.size(validation_set=True)}")
            
    

    def replay(self):
        pos = self.pos if self.full else 0
        ret_vsteps = 0
        while ret_vsteps < self.size():
            obs = {k:v[pos] for k,v in self.observations.items()}
            next_obs = {k:v[pos] for k,v in self.next_observations.items()}
            action = self.actions[pos]
            reward = self.rewards[pos]
            done = self.dones[pos]
            truncated = self.truncated[pos]
            yield (obs, next_obs, action, reward, done, truncated)
            pos = (pos+1) % self.buffer_size

            
    @override
    def collected_frames(self):
        return self._collected_frames

            
    def stored_episodes(self, validation_set = False, training_set = True):
        if validation_set and not self._disable_validation_set:
            return self._validation_storage.stored_episodes()
        elif training_set:
            return self._storage.stored_episodes()
        else:
            return self._storage.stored_episodes()+self._validation_storage.stored_episodes()
    
    @override
    def stored_frames(self, validation_set = False, training_set = True):
        if validation_set and not self._disable_validation_set:
            return self._validation_storage.stored_frames()
        elif training_set:
            return self._storage.stored_frames()
        else:
            return self._storage.stored_frames()+self._validation_storage.stored_frames()
    
    @override
    def stored_validation_frames(self) -> int:
        return self.stored_frames(validation_set=True)

    @override
    def size(self, validation_set = False, training_set = True):
        return self.stored_frames(validation_set=validation_set, training_set=training_set)

    @override
    def sample_validation(self, batch_size: int, sample_duration = None):
        return self._sample(batch_size=batch_size,sample_duration=sample_duration,validation_set=True)

    @override
    def sample(self, batch_size: int, sample_duration = None):
        return self._sample(batch_size=batch_size,sample_duration=sample_duration,validation_set=False)

    def _sample(self, batch_size: int, sample_duration = None, validation_set : bool = False) -> TransitionBatch:
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
            return self._validation_storage.sample(batch_size, sample_duration)
        else:
            return self._storage.sample(batch_size, sample_duration)

    def _get_samples(self, sampled_episodes, sampled_start_frames, env: Optional[VecNormalize] = None, sample_duration = 1) -> TransitionBatch:
        raise NotImplementedError()


    @override
    def storage_torch_device(self):
        return self._storage_torch_device

    
    @override
    def update(self, src_buffer):
        if isinstance(src_buffer, ThDictEpReplayBuffer):
            self._storage.update(src_buffer._storage)
            self._validation_storage.update(src_buffer._validation_storage)
        else:
            raise NotImplementedError()
            for (obs, next_obs, action, reward, terminated, truncated) in src_buffer.replay():
                self.add(obs=obs,
                         next_obs=next_obs,
                         action=action,
                         reward=reward,
                         terminated=terminated,
                         truncated=truncated)

