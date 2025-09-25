from __future__ import annotations
import numpy as np
from typing import Union, List, Dict, Any, Optional
import torch as th
from stable_baselines3.common.vec_env import VecNormalize
import psutil
import warnings
import time
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.buffers import numpy_to_torch_dtype, TransitionBatch, BaseValidatingBuffer, BaseBuffer
from adarl.utils.tensor_trees import is_all_finite, map_tensor_tree, flatten_tensor_tree
from typing_extensions import override
from adarl.utils.dbg.dbg_checks import dbg_check_finite, dbg_check_size, dbg_check
import adarl.utils.spaces as spaces
from adarl.utils.utils import masked_assign, masked_to_masked_assign

def take_frames(buff : th.Tensor , episodes : th.Tensor, frames : th.Tensor) -> th.Tensor:
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



class VecEpisodeStorage():
    """Vectorized episode-based experience storage. Organizing by episodes allows to sample trajectories,
        but incurs in a potential overhead at sampling time, as trajectories of a given length need to be found.
    """
    def __init__(self,  buffer_size : int,
                        vec_size : int,
                        storage_torch_device : th.device,
                        output_device : th.device,
                        observation_space : spaces.ThDict,
                        action_space : spaces.ThBox,
                        min_episode_length : int = 0,
                        rng : th.Generator | None = None):
        """_summary_

        Parameters
        ----------
        buffer_size : int
            Buffer size, in vectorized transitions (1 vectorized transition is made of vec_size transitions)
        vec_size : int
            Number of parallel environments
        storage_torch_device : th.device
            Device for storing the tensors
        output_device : th.device
            Device on which to output the tensors
        observation_space : spaces.ThDict
            Observation space
        action_space : spaces.ThBox
            Action space
        min_episode_length : int
            Minimum episode length, useful if it always more than the sampling trajectory length, to avoid sampling retries
        """
        self._buffer_size_vframes = buffer_size
        self._storage_torch_device = storage_torch_device
        self._output_device = output_device
        self._observation_space : spaces.ThDict = observation_space
        self._action_space : spaces.ThBox = action_space
        self._action_dim = np.prod(action_space.shape).item()
        self._obs_shape = spaces.get_obs_shape(observation_space)
        self._vec_size = vec_size
        self._min_episode_length = min_episode_length
        self._rng = rng

        # here 'frame' means 'transition'
        self._stored_episodes_counts = th.zeros((self._vec_size,), dtype=th.uint32, device=self._storage_torch_device)
        self._stored_vframes_th = th.as_tensor(0, dtype=th.uint64, device=self._storage_torch_device)
        self._current_ep_frame_counts = th.zeros((self._vec_size,), dtype=th.uint32, device=self._storage_torch_device)
        self._tot_stored_frames = th.as_tensor(0, dtype=th.uint64, device=self._storage_torch_device)
        self._tot_stored_episodes = th.as_tensor(0, dtype=th.uint64, device=self._storage_torch_device)
        self._added_episodes = th.as_tensor(0, dtype=th.uint64, device=self._storage_torch_device)
        self._added_vframes_th = th.as_tensor(0, dtype=th.uint64, device=self._storage_torch_device)
        self._added_vframes = 0
        self.full = False
        self._use_nonblocking_adds = self._storage_torch_device.type == "cuda"

        magic_value = 42
        self.observations = {
            key: th.full(  fill_value=magic_value,
                            size = (self._vec_size, self._buffer_size_vframes,) + _obs_shape,
                            dtype=numpy_to_torch_dtype(self._observation_space[key].dtype),
                            device = self._storage_torch_device)
            for key, _obs_shape in self._obs_shape.items()
        }

        self.actions    = th.full(  fill_value=magic_value,
                                    size = (self._vec_size, self._buffer_size_vframes, self._action_dim),
                                    dtype=numpy_to_torch_dtype(self._action_space.dtype),
                                    device = self._storage_torch_device)
        self.rewards    = th.full(  fill_value=magic_value,
                                    size = (self._vec_size, self._buffer_size_vframes,), dtype=th.float32,
                                    device = self._storage_torch_device)
        self.terminated = th.full(  fill_value=magic_value,
                                    size = (self._vec_size, self._buffer_size_vframes,), dtype=th.uint8,
                                    device = self._storage_torch_device)
        self.truncated  = th.full(  fill_value=magic_value,
                                    size = (self._vec_size, self._buffer_size_vframes,), dtype=th.uint8,
                                    device = self._storage_torch_device)
        self.ep_frame_count = th.zeros(size = (self._vec_size, self._buffer_size_vframes,),
                                            dtype = th.uint32,
                                            device = self._storage_torch_device)
        
        if self._storage_torch_device.type == "cpu" and self._use_nonblocking_adds:
            for k in self.observations.keys():
                self.observations[k] = self.observations[k].pin_memory()
            self.actions = self.actions.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.terminated = self.terminated.pin_memory()
            self.truncated = self.truncated.pin_memory()


    def clear(self):
        self._added_episodes.fill_(0)
        self._tot_stored_frames.fill_(0)
        self._tot_stored_episodes.fill_(0)
        self._current_ep_frame_counts.fill_(0)
        self.ep_frame_count.fill_(0)
        self.full = False

    def memory_size(self):
        obs_nbytes = 0
        for _, obs in self.observations.items():
            obs_nbytes += obs.element_size()*obs.nelement()
        
        action_nbytes = self.actions.element_size()*self.actions.nelement()
        rewards_nbytes = self.rewards.element_size()*self.rewards.nelement()
        terminated_nbytes = self.terminated.element_size()*self.terminated.nelement()
        timeouts_nbytes = self.truncated.element_size()*self.truncated.nelement()
        ep_durations = self.ep_frame_count.element_size()*self.ep_frame_count.nelement()

        total_memory_usage = obs_nbytes + action_nbytes + rewards_nbytes + terminated_nbytes + timeouts_nbytes + ep_durations

        return total_memory_usage
    
    def add_frames(self, observations, actions, next_observations, rewards, terminateds, truncateds, sync_stream = True):
        dbg_check_finite((observations, next_observations, actions, rewards), async_assert=True,
                         assert_msg=f"Nonfinite values in added transition")
        
        # Parallel to storing actual experience, we keep track of the frame count in the episode.
        # In this way, when we sample, we can know if we can take a trajjectory of desired lengths.
        # For example, say we want to sample a trajectory of length 3, sample index K, and we see
        # that ep_count[K] >= 3, then we can take frames[K-2:K+1] as they are all in the same episode.
        # If instead ep_count[K] < 3, we cannot take the trajectory from here. So we resample K (this
        # may be avoided if we know the min episode length).
        # However, when we start overwriting, we invalidate the logic as the episode on the edge
        # of the overwriting gets truncated. To avoid this we will have to avoid sampling indexes
        # that are in [pos:pos+traj_len-1]. This can be easily avoid by just offsetting the
        # sampling in a circular manner as idx=randint(pos+traj_len,pos+added_stored_vframes)%buffer_size.
        # In practice actually, one more frame has to be handled carefully when sampling, as next_observations
        # are stored in the same buffers of observations, so when overriding, we are already overriding also
        # the next slot in the observations buffer, this effectively reduces the capacity of one vframe.

        frame_idx = self._added_vframes_th % self._buffer_size_vframes
        overriding = self._added_vframes_th >= self._buffer_size_vframes

        next_overridden_frames_ep_count = self.ep_frame_count[:,(frame_idx+1)%self._buffer_size_vframes]
        newly_deleted_eps = (next_overridden_frames_ep_count == 0) * overriding
        eps_finishing = th.logical_or(terminateds, truncateds)

        self.ep_frame_count[:,frame_idx] = self._current_ep_frame_counts
        
        nb = self._use_nonblocking_adds
        self.actions[:,frame_idx].copy_(th.as_tensor(actions), non_blocking=nb)
        self.rewards[:,frame_idx].copy_(th.as_tensor(rewards), non_blocking=nb)
        self.terminated[:,frame_idx].copy_(th.as_tensor(terminateds), non_blocking=nb)
        self.truncated[:,frame_idx].copy_(th.as_tensor(truncateds), non_blocking=nb)


        for key in self.observations.keys():
            self.observations[key][:,frame_idx].copy_(th.as_tensor(observations[key]), non_blocking=nb)
            next_frame_idx = (frame_idx+1) % (self._buffer_size_vframes) #careful! handle this correctly at sampling, we may be overwriting something here
            self.observations[key][:,next_frame_idx].copy_(th.as_tensor(next_observations[key]), non_blocking=nb)
        
        self.full = self._added_vframes>=self._buffer_size_vframes
        self._current_ep_frame_counts = (self._current_ep_frame_counts+1)*(th.logical_not(eps_finishing))
        self._added_vframes_th += 1
        self._added_vframes += 1
        self._added_episodes += th.sum(eps_finishing)
        self._stored_episodes_counts += eps_finishing-newly_deleted_eps
        self._stored_vframes_th += th.logical_not(overriding)
        self._tot_stored_episodes += eps_finishing.sum()-newly_deleted_eps.sum()
        self._tot_stored_frames += self._vec_size*th.logical_not(overriding)
            
        if sync_stream:
            th.cuda.current_stream().synchronize() # sync non_blocking operations
        # ggLog.info(f"EpisodeStorage{id(self)}: added frame {ep_idx},{frame_idx}. term={terminated} trunc={truncated}")
        
    def stored_episodes(self):
        return self._tot_stored_episodes
    
    def stored_frames(self):
        return min(self._added_vframes, self._buffer_size_vframes)*self._vec_size
    
    def size(self):
        return self.stored_frames()
    
    def sample(self, batch_size: int, sample_duration : int | None = None) -> TransitionBatch:
        """_summary_

        Parameters
        ----------
        batch_size : int
            _description_
        sample_duration : int, optional
            trajectory length, in number of transitions (=number of actions), by default None, which means 1

        Returns
        -------
        TransitionBatch
            _description_

        Raises
        ------
        RuntimeError
            _description_
        """
        return_traj = sample_duration is not None
        if sample_duration is None:
            sample_duration = 1
        
        # for pos = self._added_vframes%self._buffer_size_vframes
        # If not full we can take data from in [0, pos] (left included, right excluded)
        # If full we can take data from [pos+1, buffer_size] U [0,pos] (the frame at pos is invalid as it's start_obs
        # has been overwritten).
        # For capacity = min(self._added_vframes, buffer_size_vframes), these ranges can be combined as:
        # arange(pos+1*full, pos+capacity))%capacity (the pos frame gets excluded only if the buffer is full)
        # Now, as we are sampling trajectories, we actually sample one idx from those ranges, and then read from 
        # ep_frame_count[idx] to see if we can take a trajectory of desired length finishing at that index.
        # As such, we know that the first sample_duration-1 frames cannot be chosen (they do not have enough frames before them).
        # So the actually valid range is:
        # arange(pos+1*full+sample_duration-1, pos+capacity)%capacity
        # Still we may fall on a frame that does not have enough frames before it, so we need to check 
        # ep_frame_count[idx]>=sample_duration, if its false we have to resample.
        # One exception is if min_episode_length>=sample_duration, then we can actually adjust idx to make it work, though 
        # this may bias the sampling.

        stored_vframes = min(self._added_vframes, self._buffer_size_vframes)
        pos = self._added_vframes % self._buffer_size_vframes
        full = self.full
        sampled_idxs = th.empty((batch_size,2), dtype=self._added_vframes_th.dtype, device = self._storage_torch_device)
        valid_sampled_idx_mask = th.zeros((batch_size,), dtype=th.bool, device = self._storage_torch_device)
        all_sampled = False
        iteration = 0
        while not all_sampled:
            frame_idxs = th.randint(low=pos+1*full+sample_duration-1,
                                    high=pos+stored_vframes,
                                    size=(batch_size,),
                                    generator=self._rng,
                                    dtype=self._added_vframes_th.dtype)%stored_vframes
            env_idxs = th.randint(low=0, high=self._vec_size, size=(batch_size,), generator=self._rng)
            new_sampled_idxs = th.stack((env_idxs, frame_idxs), dim=1)
            ep_step_counter = self.ep_frame_count[env_idxs, frame_idxs]
            new_valid_samples_mask = ep_step_counter>=sample_duration-1
            if iteration == 0:
                masked_assign(valid_sampled_idx_mask, new_valid_samples_mask, th.tensor(True, device=valid_sampled_idx_mask.device))
                masked_assign(sampled_idxs,            new_valid_samples_mask, new_sampled_idxs)
            else:
                elements_set = masked_to_masked_assign(sampled_idxs, th.logical_not(valid_sampled_idx_mask), new_sampled_idxs, new_valid_samples_mask)
                valid_sampled_idx_mask = th.logical_or(valid_sampled_idx_mask, elements_set)
            all_sampled = th.all(valid_sampled_idx_mask).item() # This is unavoidable I think, unless the whole logic was on gpu
            iteration += 1

        # Now, in sampled_idx we have indexes referring to frames that have at least sample_duration frames before them in the same episode.

        frame_offsets = th.arange(-(sample_duration-1),1, device=sampled_idxs.device)
        sampled_traj_idxs = sampled_idxs.unsqueeze(1).repeat(1, sample_duration, 1)
        sampled_traj_idxs[:,:,1] += frame_offsets
        # now the sampled_traj_idxs is of shape (batch_size, sample_duration, 2) and contains the (env_idx, frame_idx) of each sampled trajectory frame

        # simple (non-obs) buffers are of shape (vec_size, buffer_size_vframes, element_dim)
        env_idxs = sampled_traj_idxs[:,:,0]
        frame_idxs = sampled_traj_idxs[:,:,1]
        trajs_actions = self.actions[env_idxs, frame_idxs]
        trajs_rewards = self.rewards[env_idxs, frame_idxs]
        trajs_terminateds = self.terminated[env_idxs, frame_idxs]
        trajs_truncateds = self.truncated[env_idxs, frame_idxs]
        trajs_obs =      {key: self.observations[key][env_idxs, frame_idxs] for key in self.observations.keys()}
        trajs_next_obs = {key: self.observations[key][env_idxs, (frame_idxs+1)%self._buffer_size_vframes] for key in self.observations.keys()}

        dbg_check_size(trajs_actions, (batch_size, sample_duration, self._action_dim), "Sampled actions have wrong size")
        dbg_check_size(trajs_rewards, (batch_size, sample_duration), "Sampled rewards have wrong size")
        dbg_check_size(trajs_terminateds, (batch_size, sample_duration), "Sampled terminateds have wrong size")
        dbg_check_size(trajs_truncateds, (batch_size, sample_duration), "Sampled truncateds have wrong size")
        for key in self.observations.keys():
            dbg_check_size(trajs_obs[key], (batch_size, sample_duration)+self._obs_shape[key], f"Sampled obs[{key}] have wrong size")
            dbg_check_size(trajs_next_obs[key], (batch_size, sample_duration)+self._obs_shape[key], f"Sampled next_obs[{key}] have wrong size")


        if return_traj:
            observations = trajs_obs
            actions = trajs_actions
            next_observations = trajs_next_obs
            terminateds = trajs_terminateds
            rewards = trajs_rewards
        else:
            observations = {}
            next_observations = {}
            for key in self.observations:
                obs_shape = trajs_obs[key].size()[2:]
                observations[key]      = trajs_obs[key].view((batch_size,)+obs_shape)
                next_observations[key] = trajs_next_obs[key].view((batch_size,)+obs_shape)
            actions     = trajs_actions.view((batch_size,)+trajs_actions.size()[2:])
            terminateds = trajs_terminateds.view(batch_size,1)
            rewards     = trajs_rewards.view(batch_size,1)

        dbg_check_finite((observations, next_observations, actions, rewards))
        return TransitionBatch(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            terminated=terminateds,
            rewards=rewards
        )
    


    def update(self, src_storage : VecEpisodeStorage):
        
        prev_size = self.size()
        tot_copied_steps = 0
        overridden_steps = 0
        nb = self._storage_torch_device.type == "cuda" and self._use_nonblocking_adds
        while tot_copied_steps < src_storage.stored_frames():
            src_pos = (src_storage._added_vframes_th + tot_copied_steps)%src_storage._buffer_size_vframes
            dst_pos = self._added_vframes_th%self._buffer_size_vframes
            copiable_steps = min(src_storage._buffer_size_vframes - src_pos, src_storage.stored_frames()-src_pos) # either the dest space or the remaining stuff to copy up to the end of src
            copy_src = slice(src_pos, src_pos+copiable_steps)
            copy_dst = slice(dst_pos, dst_pos+copiable_steps)
            dst_empty_space = self._buffer_size_vframes-dst_pos
            overridden_steps = max(0, copiable_steps - dst_empty_space) # actually this is I think always either 0 or copyable_steps
            overridden_ep_ends = th.sum(th.logical_or(self.terminated[:,copy_dst], self.truncated[:,copy_dst]), dim=1)
            copied_ep_ends = th.sum(th.logical_or(src_storage.terminated[:,copy_src], src_storage.truncated[:,copy_src]), dim=1)

            self.actions[:,copy_dst].copy_(src_storage.actions[:,copy_src], non_blocking=nb)
            self.rewards[:,copy_dst].copy_(src_storage.rewards[:,copy_src], non_blocking=nb)
            self.terminated[:,copy_dst].copy_(src_storage.terminated[:,copy_src], non_blocking=nb)
            self.truncated[:,copy_dst].copy_(src_storage.truncated[:,copy_src], non_blocking=nb)
            self.ep_frame_count[:,copy_dst].copy_(src_storage.ep_frame_count[:,copy_src], non_blocking=nb)
            for key in self.observations.keys():
                self.observations[key][:,copy_dst].copy_(src_storage.observations[key][:,copy_src], non_blocking=nb)
                last_next_obs_idx_dst = copy_dst.stop%self._buffer_size_vframes
                last_next_obs_idx_src = copy_src.stop%src_storage._buffer_size_vframes
                self.observations[key][:,last_next_obs_idx_dst].copy_(src_storage.observations[key][:,last_next_obs_idx_src], non_blocking=nb)

            
            tot_copied_steps += copiable_steps
            self._added_vframes_th += copiable_steps
            self._added_vframes += copiable_steps
            self._added_episodes += th.sum(copied_ep_ends)
            self._stored_episodes_counts += copied_ep_ends - overridden_ep_ends
            self._stored_vframes_th += copiable_steps - overridden_steps
            self._tot_stored_frames += (copiable_steps - overridden_steps)*self._vec_size
            self._tot_stored_episodes += copied_ep_ends.sum() - overridden_ep_ends.sum()
            self.full = self._added_vframes>=self._buffer_size_vframes

        self._current_ep_frame_counts = src_storage._current_ep_frame_counts            
        new_size = self.size()
        dbg_check(lambda: new_size-prev_size == src_storage.size() or self.full, lambda:f"Error updating buffer {new_size}-{prev_size}!={src_storage.size()}")
        dbg_check(self._check_ep_counts, lambda: f"Error updating buffer, current_ep_frame_counts inconsistent with last ep_frame_count")

    def _check_ep_counts(self):
        prev_idx = (self._added_vframes-1)%self._buffer_size_vframes
        recomp_counts = (self.ep_frame_count[:,prev_idx]+1)*th.logical_not(th.logical_or(self.terminated[:,prev_idx],
                                                                                        self.truncated[:,prev_idx]))
        return self._current_ep_frame_counts==recomp_counts

    def replay(self):

        step = 0
        start_pos = self._added_vframes%self._buffer_size_vframes
        stored_vframes = min(self._added_vframes, self._buffer_size_vframes)
        while step < stored_vframes:
            idx = (start_pos+step)%self._buffer_size_vframes
            actions = self.actions[:,idx]
            rewards = self.rewards[:,idx]
            terminateds = self.terminated[:,idx]
            truncateds = self.truncated[:,idx]
            obs = {}
            next_obs = {}
            for key in self.observations.keys():
                obs[key] = self.observations[key][:,idx]
                next_obs[key] = self.observations[key][:,(idx+1)%self._buffer_size_vframes]
            step += 1
            yield (obs, next_obs, actions, rewards, terminateds, truncateds)

                

class ThVecDictEpReplayBuffer(BaseValidatingBuffer):
    """A Replay buffer that handles dict observation spaces and manages experience in episodes,
        allowing to sample trajectories and not just single transition samples.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.ThDict,
        action_space: spaces.ThBox,
        max_episode_duration : int | float,
        output_device: th.device = th.device("cpu"),
        n_envs: int = 1,
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
        self.obs_shape : dict[str, tuple[int, ...]]
        super().__init__(buffer_size, observation_space, action_space, output_device, n_envs=n_envs)

        storage_torch_device = th.device(storage_torch_device)
        output_device = th.device(output_device)
        if not isinstance(self.obs_shape, dict):
            raise ValueError("Observation space must be of dict type")
        if buffer_size % n_envs != 0:
            raise ValueError(f"Buffer size {buffer_size} must be a multiple of n_envs {n_envs}")
        if storage_torch_device.type == "cuda" and output_device.type == "cpu":
            raise AttributeError(f"Storage device is gpu, and output device is cpu. This doesn't make much sense. Use either [gpu,gpu], [cpu,gpu], or [cpu,cpu]")
        self.max_frames = buffer_size
        self.buffer_size = max(buffer_size // n_envs, 1)
        self._observation_space = observation_space
        self._action_space = action_space
        self._storage_torch_device = storage_torch_device
        
        self._max_episode_duration = max_episode_duration
        self._min_episode_duration = min_episode_duration
        if validation_holdout_ratio > 0:
            raise NotImplementedError("Validation set not implemented yet")
        # Valdation should be reorganized to be collected in two ways:
        # 1) Either by reserving one env for validation only (if n_envs>1)
        # 2) by considering one episode every k episodes for validation (but gets tricky if episodes are of variable length)
        self._validation_buffer_size = int(validation_buffer_size)
        self._validation_holdout_ratio = validation_holdout_ratio
        self._disable_validation_set = disable_validation_set
        self._val_buff_min_size = val_buffer_min_size
        self._fill_val_buffer_to_min_at_ep = fill_val_buffer_to_min_at_ep
        self._fill_val_buffer_to_min_at_step = fill_val_buffer_to_min_at_step
        if self._disable_validation_set:
            self._validation_holdout_ratio = -1
        
        example_obs = self._observation_space.sample()
        obs_size = sum([th.as_tensor(o).nelement()*th.as_tensor(o).element_size() for o in flatten_tensor_tree(example_obs).values()])

        if self._storage_torch_device.type == "cuda" and fallback_to_cpu_storage:
            pred_usage,tot = self.predict_memory_consumption()
            consumptionRatio = pred_usage/tot
            if consumptionRatio>1.0:
                ggLog.warn("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            f"Not enough memory on requested device {self._storage_torch_device} (Would consume {consumptionRatio*100:.0f}% = {pred_usage/1024**3:.3f} GiB)\n"
                            f"Falling back to CPU memory\n"
                            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                ggLog.warn(f"Observations are of size {obs_size} bytes, total buffer size is {self.buffer_size*n_envs} transitions")
                time.sleep(3)
                self._storage_torch_device = th.device("cpu")
        pred_usage,tot = self.predict_memory_consumption()
        consumptionRatio = pred_usage/tot
        if consumptionRatio > 1.0:
            raise RuntimeError(f"Not enough memory on device {self.storage_torch_device()}, would use {consumptionRatio*100:.0f}% ({pred_usage/1024**3:.3f} GiB) of available memory"
                               f"Observations are of size {obs_size} bytes, total buffer size is {self.buffer_size*n_envs} transitions")
        if consumptionRatio>0.6:
            ggLog.warn( "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        f"Replay buffer will use {consumptionRatio*100:.0f}% ({pred_usage/1024**3:.3f} GiB) of available memory on device {self._storage_torch_device}\n"
                        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            ggLog.warn(f"Observations are of size {obs_size} bytes, total buffer size is {self.buffer_size*n_envs} transitions")
            time.sleep(3)

        self._allocate_buffers(self.buffer_size, self._validation_buffer_size)
        self._addcount = 0
        self._collected_frames = 0
        self._collected_eps_th = th.as_tensor(0, dtype=th.uint64, device=self._storage_torch_device)
        
    def validation_set_enabled(self):
        return not self._disable_validation_set
    
    @override
    def memory_size(self):
        return self._storage.memory_size() #+ self._validation_storage.memory_size()
    
    @override
    def predict_memory_consumption(self):
        testRatio = 0.01
        self._allocate_buffers(int(self.buffer_size*testRatio), int(self._validation_buffer_size*testRatio))
        predicted_mem_usage = self.memory_size()/testRatio

        mem_available = float("+inf")
        if self._storage_torch_device.type == "cpu":
            mem_available = psutil.virtual_memory().available
        elif self._storage_torch_device.type == "cuda":
            mem_available = th.cuda.mem_get_info(self._storage_torch_device)[0]

        return predicted_mem_usage, mem_available


    def _allocate_buffers(self, buffer_size, validation_buffer_size):
        self._storage = VecEpisodeStorage(buffer_size, self.n_envs, self._storage_torch_device, self.out_device, self._observation_space, self._action_space, self._min_episode_duration)
        self._validation_storage : VecEpisodeStorage = None


    @override
    def add(self,
            obs: Dict[str, th.Tensor],
            next_obs: Dict[str, th.Tensor],
            action: th.Tensor,
            reward: th.Tensor,
            terminated: th.Tensor,
            truncated: th.Tensor,
            sync_stream : bool = True) -> None:
        # All inputs are batches of size n_envs
        # All should be copied to avoid modification by reference

        self._addcount+=1

        action = action.view((self.n_envs, self.action_dim))
        reward = reward.view((self.n_envs,))
        terminated = terminated.view((self.n_envs,)).to(th.uint8)
        truncated = truncated.view((self.n_envs,)).to(th.uint8)
        obs      = {k:v.view((self.n_envs,) + self.obs_shape[k]) for k,v in obs.items()} # shallow copy the observations
        next_obs = {k:v.view((self.n_envs,) + self.obs_shape[k]) for k,v in next_obs.items()} # shallow copy the observations

        self._storage.add_frames(observations=obs,
                                 actions=action,
                                 next_observations=next_obs,
                                 rewards=reward,
                                 terminateds=terminated,
                                 truncateds=truncated,
                                 sync_stream = False)
        self._collected_frames += self.n_envs
        self._collected_eps_th += th.logical_or(terminated, truncated).sum()
        if sync_stream:
            th.cuda.current_stream().synchronize() #Wait for non_blocking transfers (they are not automatically synchronized when used as inputs! https://discuss.pytorch.org/t/how-to-wait-on-non-blocking-copying-from-gpu-to-cpu/157010/2)
    

    def replay(self):
        yield from self._storage.replay()
            
    @override
    def collected_frames(self):
        return self._collected_frames

    def stored_episodes(self, validation_set = False, training_set = True):
        if validation_set and self._disable_validation_set:
            return self._storage.stored_episodes()+self._validation_storage.stored_episodes()
        elif training_set:
            return self._storage.stored_episodes()
        elif validation_set:
            return self._validation_storage.stored_episodes()
    
    @override
    def stored_frames(self, validation_set = False, training_set = True):
        if validation_set and self._disable_validation_set:
            return self._storage.stored_frames()+self._validation_storage.stored_frames()
        elif training_set:
            return self._storage.stored_frames()
        elif validation_set:
            return self._validation_storage.stored_frames()
        else:
            raise RuntimeError("Invalid combination of validation_set and training_set")
    
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
        if validation_set and not self._disable_validation_set:
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
        if isinstance(src_buffer, ThVecDictEpReplayBuffer):
            self._storage.update(src_buffer._storage)
            # self._validation_storage.update(src_buffer._validation_storage)
        else:
            raise NotImplementedError("Can only update from another ThVecDictEpReplayBuffer")

