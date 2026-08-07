#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

# import traceback
from __future__ import annotations
import adarl.utils.dbg.ggLog as ggLog

from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional, Mapping
import time
import adarl
from adarl.envs.vec.BaseVecEnv import BaseVecEnv
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.spaces as spaces
import adarl.utils.utils
import torch as th
import copy
from adarl.utils.tensor_trees import TensorTree
from typing_extensions import override
from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface, ObsType
from adarl.utils.utils import to_string_tensor, masked_assign
from adarl.utils.tensor_trees import clone_tensor_tree
from adarl.utils.base_utils import record_time, print_recorded_times, record_region_start, record_region_end, clear_recorded_times, trace_malloc_diffs
import os
from adarl.utils.async_cuda2cpu_queue import log_async

class EnvRunner(EnvRunnerInterface, Generic[ObsType]):

    spec = None

    def __init__(self,
                 env : BaseVecEnv[ObsType],
                 verbose : bool = False,
                 quiet : bool = False,
                 episodeInfoLogFile : Optional[str] = None,
                 ui_render_envs : list[int] = [],
                 autoreset : bool = True,
                 log_freq : int = -1,
                 sync_on_reinit : bool = False):
        """Initialize the Env Runner

        Parameters
        ----------
        env : BaseVecEnv[ObsType]
            The underlying adarl base environment.
        verbose : bool, optional
            Whether to enable verbose logging, by default False
        quiet : bool, optional
            Whether to suppress logging, by default False
        episodeInfoLogFile : Optional[str], optional
            The file to log episode information, by default None
        render_envs : list[int], optional
            The indexes of the environments to render as UI, by default []
        autoreset : bool, optional
            Whether to automatically reset environments within step() at the end of an episode, by default True
        log_freq : int, optional
            The frequency of logging, by default -1 (no logging)
        Raises
        ------
        RuntimeError
            If the sub-environments have different max_episode_steps.
        """
        
        super().__init__(num_envs=env.num_envs,
                         vec_observation_space=env.vec_observation_space,
                         vec_action_space=env.vec_action_space,
                         vec_reward_space=env.vec_reward_space,
                         single_observation_space=env.single_observation_space,
                         single_action_space=env.single_action_space,
                         info_space=env.info_space,
                         single_reward_space=env.single_reward_space,
                         autoreset=autoreset,
                         ui_render_envs_indexes=th.as_tensor(ui_render_envs),
                         th_device=env._th_device)
        self._adarl_env = env
        self._all_vecs = th.ones((self._adarl_env.num_envs,), dtype=th.bool, device=self._adarl_env._th_device)
        self._no_vecs = th.zeros((self._adarl_env.num_envs,), dtype=th.bool, device=self._adarl_env._th_device)
        self._envs_needing_reinit = self._no_vecs.detach().clone()
        self._last_actions = None
        import cProfile
        self._profiler = cProfile.Profile()
        self._profid = str(int(time.monotonic()*1000))
        self._reinit_count = 0
        # if th.any(self._adarl_env.get_max_episode_steps()!=self._adarl_env.get_max_episode_steps()[0]):
        #     raise RuntimeError(f"All sub environments must have the same max_episode_steps, instead"
        #                        f" they have: {self._adarl_env.get_max_episode_steps()}")
        self._max_possible_episode_steps = self._adarl_env.get_max_possible_episode_steps()
        self.spec = EnvSpec(id=f"GymEnvWrapper-env-v0_{id(env)}_{int(time.monotonic()*1000)}",
                            entry_point=None,
                            max_episode_steps=self._max_possible_episode_steps)
        self._max_episode_steps = self.spec.max_episode_steps # For compatibility, some libraries read this instead of spec

        self._verbose = verbose
        self._quiet = quiet
        self._rewards_num = spaces.get_1d_space_size(env.single_reward_space)

        self._tot_ep_rewards = th.zeros((self._adarl_env.num_envs, self._rewards_num), dtype=th.float32, device=self._no_vecs.device)
        self._sub_rewards_names = []
        self._tot_ep_sub_rewards = th.zeros((self._adarl_env.num_envs, self._rewards_num), dtype=th.float32, device=self._no_vecs.device)
        self._ep_sub_rewards : dict[str,th.Tensor] = {}
        self._cached_states : dict[str,th.Tensor] | None = None
        self._cache_ep_step_counts = th.zeros_like(self._no_vecs, dtype=th.int64)
        self._cache_ep_counts = th.zeros_like(self._no_vecs, dtype=th.int64)
        self._ep_step_counts = th.zeros_like(self._no_vecs, dtype=th.int64)
        self._ep_counts = th.zeros_like(self._no_vecs, dtype=th.int64)
        self._build_time = time.monotonic()
        self._dbg_info = {}
        self._vec_ep_info = {}

        self._reset_count = 0
        self._total_vsteps = 0
        self._last_step_end_wtime = time.monotonic()
        self._end_to_end_step_wtime = 1
        self._end_to_end_step_etime = 1
        self._log_freq = log_freq


        self._runnerStepDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 10)
        self._submitActionDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 10)
        self._getStateDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 10)
        self._getObsRewDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 10)
        self._reinitDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 10)
        self._envStepWallDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 10)
        self._last_step_end_etime = 0
        self._wtime_spent_stepping_tot = 0
        self._wtime_spent_reinit_tot = 0
        self._wtime_spent_really_reinit_tot = 0
        self._wtime_spent_stepping_adarl_tot = 0
        self._wtime_first_step = time.monotonic() # for now fill it with a reasonable time

        self._last_terminated = th.zeros((self._adarl_env.num_envs,), device=self._adarl_env._th_device, dtype=th.bool)
        self._last_truncated = th.zeros_like(self._last_terminated)
        self._sync_on_reinit = sync_on_reinit
        self._state_cache_dirty = True
        self._info_cache_dirty = True

    def _mark_caches_dirty(self):
        self._state_cache_dirty = True
        self._info_cache_dirty = True

    @override
    def step(self, actions : th.Tensor, autoreset : bool | None = None) -> Tuple[ ObsType,
                                                        ObsType,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        th.Tensor,
                                                        TensorTree[th.Tensor],
                                                        TensorTree[th.Tensor],
                                                        th.Tensor]:
        if autoreset is None:
            autoreset = self.autoreset
        if self._total_vsteps == 0:
            self._wtime_first_step = time.monotonic()

        record_region_start("EnvRunner loop ----------------------------")
        with self._runnerStepDurationAverage:
            self._total_vsteps += 1
            # if th.any(self._envs_needing_reinit):
            #     ggLog.warn(f"Calling step on terminated/truncated episodes")
            etime_step_start = th.sum(self._adarl_env.get_times_since_build())

            # Setup action to perform
            with self._submitActionDurationAverage:
                if isinstance(actions, np.ndarray):
                    actions = th.as_tensor(actions)
                self._last_actions = actions.detach().clone()
                self._adarl_env.submit_actions(actions)
            record_time("EnvRunner sent action")

            # Step the environment
            with self._envStepWallDurationAverage:
                self._adarl_env.step()
                self._ep_step_counts+=1
                self._mark_caches_dirty()
            record_time("EnvRunner stepped env")

            #Get new observation
            with self._getStateDurationAverage:
                consequent_states = self._get_states_caching()

            record_time("EnvRunner got state")

            # Assess the situation
            with self._getObsRewDurationAverage:
                # ts = []
                # ts.append(time.monotonic())
                terminateds = self._adarl_env.are_states_terminal(consequent_states)
                # ts.append(time.monotonic())
                truncateds = self._adarl_env.are_states_timedout(consequent_states)
                record_time("EnvRunner computed termination/truncation")
                # ts.append(time.monotonic())
                sub_rewardss : Dict[str,th.Tensor] = {}
                rewards = self._adarl_env.compute_rewards(consequent_states, sub_rewards_return = sub_rewardss)
                record_time("EnvRunner computed rewards")

                # ts.append(time.monotonic())
                consequent_observations = self._adarl_env.get_observations(consequent_states)
                record_time("EnvRunner got observations")
                # ts.append(time.monotonic())
                consequent_infos = self._build_info(consequent_states)
                record_time("EnvRunner built infos")
                # ts.append(time.monotonic())
                
                self._last_terminated = terminateds
                self._last_truncated = truncateds
                self._tot_ep_rewards += rewards.view(-1, self._rewards_num)
                # if self._total_sub_rewards is None:
                #     self._total_sub_rewards = {k:v for k,v in sub_rewards.items()}
                # dbg_check(lambda: len(sub_rewardss) ==0 or th.all(th.sum(th.stack(list(sub_rewardss.values()), dim=1),dim=1) - rewards < 0.001),
                #           lambda: f"sub_rewards do not sum up to reward: {rewards}!=sum({sub_rewardss})")
                # if len(sub_rewardss) > 0 and th.any(th.sum(th.stack(list(sub_rewardss.values()), dim=1),dim=1) - rewards > 0.001):
                #     raise RuntimeError(f"sub_rewards do not sum up to reward: {rewards}!=sum({sub_rewardss})")
                sub_rewards_tens = th.stack([sub_rewardss[k] for k in self._sub_rewards_names], dim = 1)
                self._tot_ep_sub_rewards += sub_rewards_tens
                for k,v in self._ep_sub_rewards.items():
                    v += sub_rewardss[k]
                # ts.append(time.monotonic())
                # ggLog.info(f"obs rew comp times = tot {ts[-1]-ts[0]} = sum("+str([f"{(ts[i+1]-ts[i])/(ts[-1]-ts[0]):.3f}" for i in range(len(ts)-1)])+")")

            record_time("EnvRunner got obs and rewards")

            with self._reinitDurationAverage:
                # reinit_ratio = th.mean(self._envs_needing_reinit.float()).item()
                # # ggLog.info(f"Step {self._total_vsteps-1}: {reinit_ratio*100:.2f}% (num_env={self.num_envs}) of envs need reinit")
                if autoreset:
                    t_prereinit_real = time.monotonic()
                    self._envs_needing_reinit = th.logical_or(terminateds, truncateds)
                    # ggLog.info(f"Reinit ratio = {th.mean(self._envs_needing_reinit.float()).item()*100:.2f}% (num_env={self.num_envs}) at step {self._total_vsteps-1}")
                    if (not self._sync_on_reinit) or th.any(self._envs_needing_reinit):
                        reinit_done = self._envs_needing_reinit
                        next_start_observations, next_start_infos = self.reinit_envs(reinit_envs_mask=self._envs_needing_reinit,
                                                                                    terminateds=terminateds,
                                                                                    truncateds=truncateds,
                                                                                    last_observations=consequent_observations,
                                                                                    last_actions=actions,
                                                                                    last_infos=consequent_infos,
                                                                                    last_rewards=rewards)
                    else:
                        reinit_done = self._no_vecs
                        next_start_observations = consequent_observations
                        next_start_infos = consequent_infos
                    self._wtime_spent_really_reinit_tot += time.monotonic() - t_prereinit_real
                else:
                    next_start_observations = consequent_observations
                    next_start_infos = consequent_infos
                    reinit_done = self._no_vecs

        record_region_end("EnvRunner loop ----------------------------")
        # trace_malloc_diffs(self._total_vsteps, 100)
        if self._total_vsteps % self._log_freq == 0:
            print_recorded_times(f"num_envs = {self.num_envs}, step {self._total_vsteps-1}")
        else:
            clear_recorded_times()

        tf = time.monotonic()
        self._last_step_end_etime = self._adarl_env.get_times_since_build()
        self._end_to_end_step_wtime = tf- self._last_step_end_wtime
        self._end_to_end_step_etime = th.sum(self._adarl_env.get_times_since_build()) - etime_step_start
        self._last_step_end_wtime = tf
        self._wtime_spent_stepping_tot += self._runnerStepDurationAverage.getLast()
        self._wtime_spent_reinit_tot += self._reinitDurationAverage.getLast()
        self._wtime_spent_stepping_adarl_tot += self._envStepWallDurationAverage.getLast()

        self._fill_dbg_info()
        if self._total_vsteps%self._log_freq==0:
            self.print_dbg_info()
        # ggLog.info(f"Finishing step # {self._total_vsteps-1} ep_step_counts={self._ep_step_counts-1} terminateds,truncateds = {terminateds, truncateds}")
        return consequent_observations, next_start_observations, rewards, terminateds, truncateds, consequent_infos, next_start_infos, reinit_done


    @override
    def reinit_envs(self,   reinit_envs_mask : th.Tensor,
                            terminateds : th.Tensor,
                            truncateds : th.Tensor,
                            last_observations : ObsType,
                            last_actions : th.Tensor | None,
                            last_infos : TensorTree[th.Tensor],
                            last_rewards : th.Tensor,
                            options = {}):
        # if self._reinit_count > 10:
        #     self._profiler.enable()
        record_region_start("EnvRunner reinit")
        self._on_episode_end(   envs_ended_mask = reinit_envs_mask,
                                last_observations = last_observations,
                                last_actions = last_actions,
                                last_infos = last_infos,
                                last_rewards = last_rewards,
                                last_terminateds = terminateds,
                                last_truncateds = truncateds)
        record_time("EnvRunner: done _on_episode_end")
        self._adarl_env.initialize_episodes(reinit_envs_mask, options=options)
        record_time("EnvRunner: done env.initialize_episodes")
        self._mark_caches_dirty()
        self._ep_step_counts[reinit_envs_mask] = 0
        self._ep_counts += 1*reinit_envs_mask
        self._tot_ep_rewards[reinit_envs_mask] = 0
        self._tot_ep_sub_rewards[reinit_envs_mask] = 0
        for k,v in self._ep_sub_rewards.items():
                v[reinit_envs_mask] = 0
        masked_assign(self._envs_needing_reinit, reinit_envs_mask, th.zeros_like(self._envs_needing_reinit))
        record_time("EnvRunner: updated counts")
        next_start_states = self._get_states_caching()
        record_time("EnvRunner: got states")
        next_start_observations = self._adarl_env.get_observations(next_start_states)
        record_time("EnvRunner: got obs")
        next_start_infos = self._build_info(next_start_states)
        record_time("EnvRunner: built info")

        record_region_end("EnvRunner reinit")

        # if self._reinit_count > 10:
        #     self._profiler.disable()
        #     os.makedirs("profiles", exist_ok=True)
        #     if self._reinit_count % 10 == 0:
        #         self._profiler.dump_stats(f"profiles/profiler_{self._profid}_save_{self._reinit_count}.prof")
        self._reinit_count+=1
        return next_start_observations, next_start_infos

    def print_dbg_info(self):
        log_async(  "EnvRunner: vsteps[{vsteps}]"+
                    " wHz={wall_fps}"+
                    " rt_estep={rtfactor_estep}"+
                    " rt_rstep={rtfactor_rstep}"+
                    " rt_tot={rtfactor_tot}"+
                    " rt_wall={rtfactor}"+
                    " aStpWt={avg_env_step_wall_duration}"+
                    " aSimWt={avg_adarl_step_wall_duration}"+
                    " aActWt={avg_act_wall_duration}"+
                    " aStaWt={avg_sta_wall_duration}"+
                    " aObsWt={avg_obs_rew_wall_duration}"+
                    " aReiWt={avg_reinit_wall_duration}"+
                    " tstep%wt={ratio_time_spent_stepping}"+
                    " tinit%wt={ratio_time_spent_reinit}"+
                    " trinit%wt={ratio_time_spent_really_reinit}"+
                    " tstep%st={ratio_time_spent_simulating}",
                    {"vsteps":self._dbg_info['vsteps'],
                     "wall_fps" : self._dbg_info['wall_fps'],
                     "rtfactor_estep" : self._dbg_info['rtfactor_estep'],
                     "rtfactor_rstep" : self._dbg_info['rtfactor_rstep'],
                     "rtfactor_tot" : self._dbg_info['rtfactor_tot'],
                     "rtfactor" : self._dbg_info['rtfactor'],
                     "avg_env_step_wall_duration" : self._dbg_info['avg_env_step_wall_duration'],
                     "avg_adarl_step_wall_duration" : self._dbg_info['avg_adarl_step_wall_duration'],
                     "avg_act_wall_duration" : self._dbg_info['avg_act_wall_duration'],
                     "avg_sta_wall_duration" : self._dbg_info['avg_sta_wall_duration'],
                     "avg_obs_rew_wall_duration" : self._dbg_info['avg_obs_rew_wall_duration'],
                     "avg_reinit_wall_duration" : self._dbg_info['avg_reinit_wall_duration'],
                     "ratio_time_spent_stepping" : self._dbg_info['ratio_time_spent_stepping'],
                     "ratio_time_spent_reinit" : self._dbg_info['ratio_time_spent_reinit'],
                     "ratio_time_spent_really_reinit" : self._dbg_info['ratio_time_spent_really_reinit'],
                     "ratio_time_spent_simulating" : self._dbg_info['ratio_time_spent_simulating'],
                    })

    @override
    def reset(self, seed = None, options = {}) -> tuple[ObsType, TensorTree[th.Tensor]]:
        if options is None:
            options = {}
        if seed is not None:
            self._adarl_env.set_seeds(th.as_tensor(seed, device=self._adarl_env._th_device).expand(self.num_envs))
        if self._verbose:
            ggLog.info(" ------- Resetting Environment (#"+str(self._reset_count)+")-------")

        if self._reset_count > 0:
            self._fill_dbg_info()
            # if self._ep_step_counts == 0:
            #     ggLog.info(f"No step executed in episode {self._reset_count-1}")
            # else:
            #     if self._verbose:
            #         for k,v in self._dbg_info.items():
            #             ggLog.info(k," = ",v)
            #     elif not self._quiet:
            #         msg =  (f"ep = {self._dbg_info['reset_count']:d}"+
            #                 f" rwrd = {self._dbg_info['ep_reward']:.3f}"+
            #                 f" stps = {self._dbg_info['ep_frames_count']:d}"+
            #                 f" wHz = {self._dbg_info['wall_fps']:.3f}"+
            #                 f" wHzFtl = {self._dbg_info['wall_fps_first_to_last']:.3f}"+
            #                 f" avgStpWt = {self._dbg_info['avg_env_step_wall_duration']:f}"+
            #                 f" avgSimWt = {self._dbg_info['avg_adarl_step_wall_duration']:f}"+
            #                 f" avgActWt = {self._dbg_info['avg_act_wall_duration']:f}"+
            #                 f" avgStaWt = {self._dbg_info['avg_sta_wall_duration']:f}"+
            #                 f" avgObsWt = {self._dbg_info['avg_obs_rew_wall_duration']:f}"+
            #                 f" tstep%ftl = {self._dbg_info['ratio_time_spent_stepping_first_to_last']:.2f}"+
            #                 f" tstep% = {self._dbg_info['ratio_time_spent_stepping']:.2f}"+
            #                 f" wEpDur = {self._dbg_info['tot_ep_wall_duration']:.2f}"+
            #                 f" sEpDur = {self._dbg_info['tot_ep_sim_duration']:.2f}")
            #         if "success_ratio" in self._dbg_info.keys():
            #                 msg += f" succ_ratio = {self._dbg_info['success_ratio']:.2f}"
            #         if "success" in self._dbg_info.keys():
            #                 msg += f" succ = {self._dbg_info['success']:.2f}"
            #         ggLog.info(msg)


        states = self._get_states_caching()
        terminateds = self._adarl_env.are_states_terminal(states)
        truncateds = self._adarl_env.are_states_timedout(states)
        rewards = self._adarl_env.compute_rewards(states, sub_rewards_return = {})
        observations = self._adarl_env.get_observations(states)
        infos = self._build_info(states)

        #reset simulation state
        self._adarl_env.reset()
        self._mark_caches_dirty()


        self.reinit_envs(reinit_envs_mask=self._all_vecs,
                         terminateds=terminateds, 
                         truncateds=truncateds, 
                         last_actions=self._last_actions,
                         last_observations=observations,
                         last_infos=infos,
                         last_rewards=rewards,
                         options=options)        
        self._mark_caches_dirty()

        self._reset_count += 1
        self._cached_states = None

        self._runnerStepDurationAverage.reset()
        self._submitActionDurationAverage.reset()
        self._getStateDurationAverage.reset()
        self._getObsRewDurationAverage.reset()
        self._reinitDurationAverage.reset()
        self._envStepWallDurationAverage.reset()
        self._fill_dbg_info()

        states = self._get_states_caching()
        observation = self._adarl_env.get_observations(states)
        return observation, self._build_info(states)


    @override
    def get_ui_renderings(self) -> list[th.Tensor]:
        th_images, imageTimes = self._adarl_env.get_ui_renderings(vec_mask=self.ui_render_envs_mask)
        return th_images

    @override
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
        if self._state_cache_dirty:
            self._cached_states = self._adarl_env.get_states()
            self._state_cache_dirty = False
        return self._cached_states

    def get_base_env(self) -> BaseVecEnv[ObsType]:
        """Get the underlying adarl base environment

        Returns
        -------
        BaseEnv
            The adarl.BaseEnv object.
        """
        return self._adarl_env


    def _fill_dbg_info(self):
        state = self._get_states_caching()


        t = time.monotonic()
        wtime_since_simstart = (t-self._wtime_first_step)
        self._dbg_info["vsteps"] = self._total_vsteps
        self._dbg_info["ratio_time_spent_stepping"] = self._wtime_spent_stepping_tot/wtime_since_simstart
        self._dbg_info["ratio_time_spent_reinit"] = self._wtime_spent_reinit_tot/wtime_since_simstart
        self._dbg_info["ratio_time_spent_really_reinit"] = self._wtime_spent_really_reinit_tot/wtime_since_simstart
        self._dbg_info["ratio_time_spent_simulating"] = self._wtime_spent_stepping_adarl_tot/wtime_since_simstart
        self._dbg_info["wall_fps"] = self._adarl_env.num_envs*self._total_vsteps/wtime_since_simstart
        self._dbg_info["rtfactor"] = self._end_to_end_step_etime/self._end_to_end_step_wtime
        self._dbg_info["rtfactor_estep"] = th.sum(self._adarl_env.get_times_since_build())/self._wtime_spent_stepping_adarl_tot
        self._dbg_info["rtfactor_rstep"] = th.sum(self._adarl_env.get_times_since_build())/self._wtime_spent_stepping_tot
        self._dbg_info["rtfactor_tot"] = th.sum(self._adarl_env.get_times_since_build())/wtime_since_simstart
        self._dbg_info["fps_only_sim"] = self._total_vsteps/self._wtime_spent_stepping_adarl_tot if self._wtime_spent_stepping_adarl_tot!=0 else float("nan")
        self._dbg_info["fps_only_env"] = self._total_vsteps/self._wtime_spent_stepping_tot if self._wtime_spent_stepping_tot!=0 else float("nan")
        self._dbg_info["avg_env_step_wall_duration"] = self._runnerStepDurationAverage.getAverage()
        self._dbg_info["avg_adarl_step_wall_duration"] = self._envStepWallDurationAverage.getAverage()
        self._dbg_info["avg_act_wall_duration"] = self._submitActionDurationAverage.getAverage()
        self._dbg_info["avg_sta_wall_duration"] = self._getStateDurationAverage.getAverage()
        self._dbg_info["avg_obs_rew_wall_duration"] = self._getObsRewDurationAverage.getAverage()
        self._dbg_info["avg_reinit_wall_duration"] = self._reinitDurationAverage.getAverage()
        # self._dbg_info["tot_ep_sim_duration"] = self._last_step_end_etime
        # self._dbg_info["reset_count"] = self._reset_count
        # self._dbg_info["time_from_start"] = t - self._build_time
        # self._dbg_info["total_steps"] = self._total_vsteps
        # self._dbg_info["seed"] = self._adarl_env.get_seeds()
        # self._dbg_info["alltime_stepping_time"] = self._wtime_spent_stepping_tot
        # self._dbg_info["alltime_fps"] = self._total_vsteps*self.num_envs/(t-self._build_time)
        # self._dbg_info["alltime_stepping_fps"] = self._total_vsteps*self.num_envs/self._wtime_spent_stepping_tot if self._wtime_spent_stepping_tot>0 else float("nan")

        self._vec_ep_info["ep_frames_count"] = self._ep_step_counts
        self._vec_ep_info["ep_reward"] = self._tot_ep_rewards
        self._vec_ep_info["ep_rewards_sum"] = th.sum(self._tot_ep_rewards, dim=1) # Sum of the subrewards, if using explicit sub rewards (self._tot_ep_rewards is then a 2D tensor)
        # ggLog.info(f"_tot_ep_rewards = {self._tot_ep_rewards}")
        # self._dbg_info.update(self._ggEnv.getInfo(state))
        if len(self._sub_rewards_names)==0: # at the first step and episode this must be populated to at least know which fields we'll have
            sub_rewards : dict[str, th.Tensor] = {}
            # Not really setting the rewards, just populating the fields with zeros
            try:
                _ = self._adarl_env.compute_rewards(state, sub_rewards_return=sub_rewards)
            except ValueError:
                pass
            self._sub_rewards_names = list(sub_rewards.keys())
            self._tot_ep_sub_rewards = th.zeros((self._adarl_env.num_envs, len(self._sub_rewards_names)), dtype=th.float32, device=self._adarl_env._th_device)
            self._ep_sub_rewards = {k : th.zeros(size=(self.num_envs,), dtype=th.float32, device=self._adarl_env._th_device) for k in self._sub_rewards_names}
        # ggLog.info(f'self._total_sub_rewards = {self._total_sub_rewards}')
        self._vec_ep_info["ep_sub_rewards"] = self._tot_ep_sub_rewards
        self._vec_ep_info["ep_sub_rewards_labels"] = to_string_tensor(self._sub_rewards_names).expand(self._adarl_env.num_envs,len(self._sub_rewards_names),-1)
        self._vec_ep_info.update({"ep_sub_reward_"+k : v for k,v in self._ep_sub_rewards.items()})

    def _build_info(self, states) -> TensorTree[th.Tensor]:
        info = {}
        # timed_out = self._adarl_env.are_states_timedout(states)
        # terminated = self._adarl_env.are_states_terminal(states)
        # only_truncated = th.logical_and(timed_out, th.logical_not(terminated))
        # info["TimeLimit.truncated"] = only_truncated
        # info["timed_out"] = timed_out
        adarl_env_info = self._adarl_env.get_infos(states)
        # adarl_env_info["is_success"] = adarl_env_info.get("success",
        #                                     th.as_tensor(False).to(device=self._adarl_env._th_device, non_blocking=self._adarl_env._th_device.type=="cuda").expand((self._adarl_env.num_envs,)))
        info.update({k:th.as_tensor(v) for k,v in self._vec_ep_info.items()})
        info.update(adarl_env_info)
        return clone_tensor_tree(info)

        # if self._info_cache_dirty:
        #     info = {}
        #     # timed_out = self._adarl_env.are_states_timedout(states)
        #     # terminated = self._adarl_env.are_states_terminal(states)
        #     # only_truncated = th.logical_and(timed_out, th.logical_not(terminated))
        #     # info["TimeLimit.truncated"] = only_truncated
        #     # info["timed_out"] = timed_out
        #     adarl_env_info = self._adarl_env.get_infos(states)
        #     # adarl_env_info["is_success"] = adarl_env_info.get("success",
        #     #                                     th.as_tensor(False).to(device=self._adarl_env._th_device, non_blocking=self._adarl_env._th_device.type=="cuda").expand((self._adarl_env.num_envs,)))
        #     info.update({k:th.as_tensor(v) for k,v in self._vec_ep_info.items()})
        #     info.update(adarl_env_info)
        #     self._cached_info = info
        #     self._info_cache_dirty = False
        # return clone_tensor_tree(self._cached_info, detach=True)
    
    @override
    def get_max_episode_steps(self):
        return self._adarl_env.get_max_episode_steps()
    
    @override
    def get_max_possible_episode_steps(self):
        return self._max_possible_episode_steps