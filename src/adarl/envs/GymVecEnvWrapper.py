#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

# import traceback
from __future__ import annotations
import adarl.utils.dbg.ggLog as ggLog

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional, Mapping
import time
import csv

import adarl

from adarl.envs.BaseVecEnv import BaseVecEnv
import os
import adarl.utils.dbg.ggLog as ggLog

import multiprocessing as mp
import signal
import adarl.utils.session
import adarl.utils.utils
from adarl.utils.wandb_wrapper import wandb_log
import torch as th
from torch.nn.functional import one_hot
import copy
from adarl.utils.tensor_trees import TensorTree

ObsType = TypeVar("ObsType", bound=Mapping[str | tuple[str,...], th.Tensor])

class GymVecEnvWrapper(gym.vector.VectorEnv, Generic[ObsType]):

    spec = None

    def __init__(self,
                 env : BaseVecEnv[ObsType],
                 verbose : bool = False,
                 quiet : bool = False,
                 episodeInfoLogFile : Optional[str] = None,
                 render_envs : list[int] = [],
                 step_v1_x : bool = False):
        
        self._adarl_env = env
        self.action_space = env.vec_action_space
        self.observation_space = env.vec_observation_space
        self.single_action_space = env.single_action_space
        self.single_observation_space = env.single_observation_space
        self.metadata = env.metadata
        self.use_step_v1_x = step_v1_x
        
        if th.any(self._adarl_env.get_max_episode_steps()!=self._adarl_env.get_max_episode_steps()[0]):
            raise RuntimeError(f"All sub environments must have the same max_episode_steps, instead"
                               f" they have: {self._adarl_env.get_max_episode_steps()}")
        self.spec = EnvSpec(id=f"GymEnvWrapper-env-v0_{id(env)}_{int(time.monotonic()*1000)}",
                            entry_point=None,
                            max_episode_steps=int(self._adarl_env.get_max_episode_steps()[0].item()))
        self._max_episode_steps = self.spec.max_episode_steps # For compatibility, some libraries read this instead of spec

        self._verbose = verbose
        self._quiet = quiet

        self._tot_ep_reward = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.float32)
        self._tot_ep_sub_rewards = {}
        self._cached_states : dict[str,th.Tensor] | None = None
        self._cache_ep_step_counts = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.bool)
        self._cache_ep_counts = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.bool)
        self._tot_vstep_count = 0
        self._ep_step_counts = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.bool)
        self._ep_counts = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.bool)
        self._build_time = time.monotonic()
        self._dbg_info = {}

        self._reset_count = 0
        self._totalSteps = 0
        self._first_step_start_wtime = -1
        self._last_step_end_wtime = -1


        self._envStepDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._submitActionDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._getStateDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._getPrevStateDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._getObsRewDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._simStepWallDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._last_step_end_etime = 0
        self._wtime_spent_stepping_ep = 0

        self._wtime_spent_stepping_tot = 0

        self._render_envs = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.bool)
        self._render_envs[render_envs] = True
        self._terminated = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.bool)
        self._truncated = th.zeros_like(self._terminated)

        super().__init__(num_envs=self._adarl_env.num_envs,
                         observation_space=self._adarl_env.single_observation_space,
                         action_space=self._adarl_env.single_action_space)


    def step_full(self, actions, autoreset = True) -> Tuple[ObsType,
                                                            ObsType,
                                                            th.Tensor,
                                                            th.Tensor,
                                                            th.Tensor,
                                                            TensorTree[th.Tensor],
                                                            TensorTree[th.Tensor]]:

        t0 = time.monotonic()
        if self._ep_step_counts==0:
            self._first_step_start_wtime = t0
        if th.any(th.logical_or(self._terminated, self._truncated)):
            ggLog.warn(f"Calling step on terminated/truncated episodes ({th.count_nonzero(self._terminated)} terminated, {th.count_nonzero(self._truncated)} truncated)")

        self._totalSteps += 1

        # Setup action to perform
        with self._submitActionDurationAverage:
            self._adarl_env.submit_actions(actions)

        # Step the environment
        with self._simStepWallDurationAverage:
            self._adarl_env.step()
            self._ep_step_counts+=1

        #Get new observation
        with self._getStateDurationAverage:
            consequent_states = self._get_states_caching()

        # Assess the situation
        with self._getObsRewDurationAverage:
            terminateds = self._adarl_env.are_states_terminal(consequent_states)
            truncateds = self._adarl_env.are_states_timedout(consequent_states)
            sub_rewardss : Dict[str,th.Tensor] = {}
            rewards = self._adarl_env.compute_rewards(consequent_states, sub_rewards_return = sub_rewardss)
            consequent_observations = self._adarl_env.get_observations(consequent_states)
            
            self._terminated = terminateds
            self._truncated = truncateds
            self._tot_ep_reward += rewards
            # if self._total_sub_rewards is None:
            #     self._total_sub_rewards = {k:v for k,v in sub_rewards.items()}
            if len(sub_rewardss) > 0 and th.any(th.sum(th.stack(list(sub_rewardss.values()), dim=1),dim=1) - rewards > 0.001):
                raise RuntimeError(f"sub_rewards do not sum up to reward: {rewards}!=sum({sub_rewardss})")
            for k,v in sub_rewardss.items():
                self._tot_ep_sub_rewards[k] += v
        
        if autoreset:
            # If some environments have terminated/truncated then re-initialize them
            envs_to_reinitialize = th.logical_or(terminateds, truncateds)
            ggLog.info(f"envs_to_reinitialize = {envs_to_reinitialize}") has one excess dimension
            self._adarl_env.initialize_episodes(envs_to_reinitialize)
            self._ep_step_counts[envs_to_reinitialize] = 0
            self._ep_counts[envs_to_reinitialize] += 1
        # Get the start observation for the next step
        next_start_states = self._get_states_caching()
        next_start_observations = self._adarl_env.get_observations(next_start_states)
        next_start_infos = self._adarl_env.get_infos(next_start_states)


        tf = time.monotonic()
        stepDuration = tf - t0
        self._envStepDurationAverage.addValue(newValue = stepDuration)
        self._last_step_end_etime = self._adarl_env.get_times_since_build()
        self._last_step_end_wtime = tf
        self._wtime_spent_stepping_ep += stepDuration
        self._wtime_spent_stepping_tot += stepDuration

        self._fill_dbg_info()
        info = self._build_info()
        return consequent_observations, next_start_observations, rewards, terminateds, truncateds, info, next_start_infos

    def step_v0_29(self, actions) -> Tuple[ObsType, th.Tensor, th.Tensor, th.Tensor, TensorTree[th.Tensor]]:
        # Steps even if some envs were supposed to be re-initialized
        # Then, after stepping, it does the re-initialization
        # Transitions that previously had "terminated or truncated" must then not be considered valid transitions
        (consequent_observations,
         next_start_observations,
         reward,
         terminated,
         truncated,
         consequent_info,
         next_start_infos) = self.step_full(actions)
        next_start_infos["final_observation"] = consequent_observations
        # next_start_infos["final_infos"] = consequent_info
        return next_start_observations, reward, terminated, truncated, next_start_infos
    

    def step_v1_0(self, actions) -> Tuple[ObsType, th.Tensor, th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        # Steps even if some envs were supposed to be re-initialized
        # Then, after stepping, it does the re-initialization
        # Transitions that previously had "terminated or truncated" must then not be considered valid transitions
        envs_to_reinitialize = th.logical_or(self._terminated, self._truncated)
        consequent_observations, next_start_observations, reward, terminated, truncated, info, next_start_infos = self.step_full(actions, autoreset=False)
        self._adarl_env.initialize_episodes(envs_to_reinitialize)
        self._ep_step_counts[envs_to_reinitialize] = 0
        self._ep_counts[envs_to_reinitialize] += 1
        next_start_states = self._get_states_caching()
        next_start_observations = self._adarl_env.get_observations(next_start_states)
        next_start_infos = self._adarl_env.get_infos(next_start_states)
        terminated[envs_to_reinitialize] = False
        truncated[envs_to_reinitialize] = False
        return next_start_observations, reward, terminated, truncated, next_start_infos

    def step(self, actions):
        if self.use_step_v1_x:
            return self.step_v1_0(actions)
        else:
            return self.step_v0_29(actions)




    def reset(self, seed = None, options = {}):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        if seed is not None:
            self._adarl_env.set_seeds(th.as_tensor(seed, device=self._adarl_env.th_device).expand(self.num_envs))
        if self._verbose:
            ggLog.info(" ------- Resetting Environment (#"+str(self._reset_count)+")-------")

        if self._reset_count > 0:
            self._fill_dbg_info()
            if self._ep_step_counts == 0:
                ggLog.info(f"No step executed in episode {self._reset_count-1}")
            else:
                if self._verbose:
                    for k,v in self._dbg_info.items():
                        ggLog.info(k," = ",v)
                elif not self._quiet:
                    msg =  (f"ep = {self._dbg_info['reset_count']:d}"+
                            f" rwrd = {self._dbg_info['ep_reward']:.3f}"+
                            f" stps = {self._dbg_info['ep_frames_count']:d}"+
                            f" wHz = {self._dbg_info['wall_fps']:.3f}"+
                            f" wHzFtl = {self._dbg_info['wall_fps_first_to_last']:.3f}"+
                            f" avgStpWt = {self._dbg_info['avg_env_step_wall_duration']:f}"+
                            f" avgSimWt = {self._dbg_info['avg_sim_step_wall_duration']:f}"+
                            f" avgActWt = {self._dbg_info['avg_act_wall_duration']:f}"+
                            f" avgStaWt = {self._dbg_info['avg_sta_wall_duration']:f}"+
                            f" avgObsWt = {self._dbg_info['avg_obs_rew_wall_duration']:f}"+
                            f" tstep%ftl = {self._dbg_info['ratio_time_spent_stepping_first_to_last']:.2f}"+
                            f" tstep% = {self._dbg_info['ratio_time_spent_stepping']:.2f}"+
                            f" wEpDur = {self._dbg_info['tot_ep_wall_duration']:.2f}"+
                            f" sEpDur = {self._dbg_info['tot_ep_sim_duration']:.2f}")
                    if "success_ratio" in self._dbg_info.keys():
                            msg += f" succ_ratio = {self._dbg_info['success_ratio']:.2f}"
                    if "success" in self._dbg_info.keys():
                            msg += f" succ = {self._dbg_info['success']:.2f}"
                    ggLog.info(msg)

        self._lastPreResetTime = time.monotonic()
        #reset simulation state
        self._adarl_env.reset()
        self._adarl_env.initialize_episodes(options=options)
        self._lastPostResetTime = time.monotonic()

        self._reset_count += 1
        self._ep_step_counts = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.long)
        self._cached_states = None
        self._tot_ep_reward = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env.th_device, dtype=th.float32)
        self._tot_ep_sub_rewards = {}
        self._last_step_end_wtime = -1.0
        self._wtime_spent_stepping_ep = 0.0

        self._envStepDurationAverage.reset()
        self._submitActionDurationAverage.reset()
        self._getStateDurationAverage.reset()
        self._getPrevStateDurationAverage.reset()
        self._getObsRewDurationAverage.reset()
        self._simStepWallDurationAverage.reset()
        self._fill_dbg_info()

        observation = self._adarl_env.get_observations(self._get_states_caching())
        return observation, self._build_info()







    def render(self, mode : str = 'rgb_array') -> tuple[th.Tensor,...]:
        """Get a rendering of the environment.

        This rendering is not synchronized with the end of the step() function

        Parameters
        ----------
        mode : string
            type of rendering to generate. Only "rgb_array" is supported

        Returns
        -------
        type
            A rendering in the format of a numpy array of shape (width, height, 3), BGR channel order.
            OpenCV-compatible

        Raises
        -------
        NotImplementedError
            If called with mode!="rgb_array"

        """
        if mode!="rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        th_images, imageTimes = self._adarl_env.get_ui_renderings(vec_mask=self._render_envs)
        ret : list[th.Tensor] = [None]*self.num_envs #type: ignore
        c = 0
        for i in range(self.num_envs):
            if self._render_envs[i]:
                ret[i] = th_images[c]
                c += 1
        return tuple(ret)








    def close(self):
        """Close the environment.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._adarl_env.close()


    def _get_states_caching(self) -> Any:
        """Get the an observation of the environment keeping a cache of the last observation.

        Returns
        -------
        Any
            An observation of the environment. See the environment implementation for details on its format

        """
        if (self._cached_states is None or
            th.any(self._cache_ep_counts != self._ep_counts) or
            th.any(self._cache_ep_step_counts != self._ep_step_counts)):
            self._cached_states = self._adarl_env.get_states()
            self._cache_ep_step_counts[:] = self._ep_step_counts
            self._cache_ep_counts[:] = self._ep_counts
        return self._cached_states

    def get_base_env(self) -> BaseVecEnv[ObsType]:
        """Get the underlying adarl base environment

        Returns
        -------
        BaseEnv
            The adarl.BaseEnv object.
        """
        return self._adarl_env

    def __del__(self):
        # This is only called when the object is garbage-collected, so users should
        # still call close themselves, we don't know when garbage collection will happen
        self.close()


    def _fill_dbg_info(self):
        state = self._get_states_caching()
        if self._ep_step_counts>0:
            avgSimTimeStepDuration = self._last_step_end_etime/self._ep_step_counts
            totEpisodeWallDuration = time.monotonic() - self._lastPostResetTime
            epWallDurationUntilDone = self._last_step_end_wtime - self._lastPostResetTime
            resetWallDuration = self._lastPostResetTime-self._lastPreResetTime
            wallFps = self._ep_step_counts/totEpisodeWallDuration
            wall_fps_until_done = self._ep_step_counts/epWallDurationUntilDone
            wall_fps_only_stepping = self._ep_step_counts/self._wtime_spent_stepping_ep
            ratio_time_spent_stepping = self._wtime_spent_stepping_ep/totEpisodeWallDuration
            wall_fps_first_to_last = self._ep_step_counts/(self._last_step_end_wtime - self._first_step_start_wtime)
            ratio_time_spent_stepping_first_to_last = self._wtime_spent_stepping_ep/(self._last_step_end_wtime - self._first_step_start_wtime)
        else:
            avgSimTimeStepDuration = float("NaN")
            totEpisodeWallDuration = 0
            resetWallDuration = float("NaN")
            wallFps = float("NaN")
            wall_fps_until_done = float("NaN")
            wall_fps_only_stepping = float("nan")
            ratio_time_spent_stepping = 0
            wall_fps_first_to_last = float("NaN")
            ratio_time_spent_stepping_first_to_last = 0
            # state = map_tensor_tree(state, func=lambda x: th.as_tensor(x))

        self._dbg_info["avg_env_step_wall_duration"] = self._envStepDurationAverage.getAverage()
        self._dbg_info["avg_sim_step_wall_duration"] = self._simStepWallDurationAverage.getAverage()
        self._dbg_info["avg_act_wall_duration"] = self._submitActionDurationAverage.getAverage()
        self._dbg_info["avg_sta_wall_duration"] = self._getStateDurationAverage.getAverage()
        self._dbg_info["avg_pst_wall_duration"] = self._getPrevStateDurationAverage.getAverage()
        self._dbg_info["avg_obs_rew_wall_duration"] = self._getObsRewDurationAverage.getAverage()
        self._dbg_info["avg_step_sim_duration"] = avgSimTimeStepDuration
        self._dbg_info["tot_ep_wall_duration"] = totEpisodeWallDuration
        self._dbg_info["tot_ep_sim_duration"] = self._last_step_end_etime
        self._dbg_info["reset_wall_duration"] = resetWallDuration
        self._dbg_info["ep_frames_count"] = self._ep_step_counts
        self._dbg_info["ep_reward"] = self._tot_ep_reward
        self._dbg_info["wall_fps"] = wallFps
        self._dbg_info["wall_fps_until_done"] = wall_fps_until_done
        self._dbg_info["reset_count"] = self._reset_count
        self._dbg_info["ratio_time_spent_stepping"] = ratio_time_spent_stepping
        self._dbg_info["time_from_start"] = time.monotonic() - self._build_time
        self._dbg_info["total_steps"] = self._totalSteps
        self._dbg_info["wall_fps_first_to_last"] = wall_fps_first_to_last
        self._dbg_info["ratio_time_spent_stepping_first_to_last"] = ratio_time_spent_stepping_first_to_last
        self._dbg_info["seed"] = self._adarl_env.get_seeds()
        self._dbg_info["alltime_stepping_time"] = self._wtime_spent_stepping_tot
        self._dbg_info["wall_fps_only_stepping"] = wall_fps_only_stepping

        # self._dbg_info.update(self._ggEnv.getInfo(state))
        if len(self._tot_ep_sub_rewards)==0: # at the first step and episode this must be populated to at least know which fields we'll have
            sub_rewards = {}
            # Not really setting the rewards, just populating the fields with zeros
            try:
                _ = self._adarl_env.compute_rewards(state, sub_rewards_return=sub_rewards)
            except ValueError:
                pass
            self._tot_ep_sub_rewards = {k: v*0.0 for k,v in sub_rewards.items()}
        # ggLog.info(f'self._total_sub_rewards = {self._total_sub_rewards}')
        self._dbg_info.update({"ep_sub_"+k:v for k,v in self._tot_ep_sub_rewards.items()})

    def _build_info(self):
        info = {}
        states = self._get_states_caching()
        only_truncated = self._adarl_env.are_states_timedout(states) and not self._adarl_env.are_states_terminal(states)
        info["TimeLimit.truncated"] = only_truncated
        info["timed_out"] = self._adarl_env.are_states_timedout(states)
        adarl_env_info = self._adarl_env.get_infos(states)
        adarl_env_info["is_success"] = adarl_env_info.get("success",
                                            th.as_tensor(False, device=self._adarl_env.th_device).expand((self._adarl_env.num_envs,)))
        info.update(self._dbg_info)
        info.update(adarl_env_info)
        return copy.deepcopy(info)