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
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional
import time
import csv

import adarl

from adarl.envs.BaseEnv import BaseEnv
import os
import adarl.utils.dbg.ggLog as ggLog

import multiprocessing as mp
import signal
import adarl.utils.session
import adarl.utils.utils
from adarl.utils.wandb_wrapper import wandb_log
import torch as th
from adarl.utils.tensor_trees import map_tensor_tree
import copy

ObsType = TypeVar("ObsType")

class GymEnvWrapper(gym.Env, Generic[ObsType]):
    """This class is a wrapper to convert adarl environments in OpenAI Gym environments.

    It also implements a simple cache for the state of the environment and keeps track
    of some useful metrics.

    """

    # action_space = None
    # observation_space = None
    # metadata = None # e.g. {'render.modes': ['rgb_array']}
    spec = None

    def __init__(self,
                 env : BaseEnv,
                 verbose : bool = False,
                 quiet : bool = False,
                 episodeInfoLogFile : Optional[str] = None,
                 logs_id : str | None = "",
                 use_wandb = True):
        """Short summary.

        Parameters
        ----------

        """
        
        self._use_wandb = use_wandb
        self._logs_id = logs_id if logs_id is not None else ""
        self._ggEnv = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        if self._ggEnv.is_timelimited:
            reg_env_id = f"GymEnvWrapper-env-v0_{id(env)}_{int(time.monotonic()*1000)}"
            # gym.register(id=reg_env_id, entry_point=None, max_episode_steps = self._ggEnv.get_max_episode_steps())
            # self.spec = gym.spec(env_id=reg_env_id)
            self.spec = EnvSpec(id=reg_env_id, entry_point=None, max_episode_steps=self._ggEnv.get_max_episode_steps())
            self._max_episode_steps = self.spec.max_episode_steps # For compatibility, some libraries read this instead of spec

        self._verbose = verbose
        self._quiet = quiet
        self._logEpisodeInfo = episodeInfoLogFile is not None
        self._episodeInfoLogFile : str = episodeInfoLogFile if episodeInfoLogFile is not None else ""

        self._framesCounter = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = -1
        self._cumulativeImagesAge = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0.0
        self._total_sub_rewards = {}
        self._resetCount = 0
        self._init_time = time.monotonic()
        self._totalSteps = 0
        self._first_step_start_time = -1
        self._last_step_finish_time = -1


        self._envStepDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._submitActionDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._getStateDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._getPrevStateDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._getObsRewDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._simStepWallDurationAverage =adarl.utils.utils.AverageKeeper(bufferSize = 100)
        self._lastStepEndSimTimeFromStart = 0
        self._lastValidStepWallTime = -1
        self._timeSpentStepping_ep = 0
        self._success_ratio = 0.0
        self._successes = [0]*50
        self._last_ep_succeded = False
        self._logFileCsvWriter = None
        self._dbg_info = {}

        self._terminated = False
        self._alltime_stepping_time = 0


    def _update_dbg_info(self):
        state = self._getStateCached()
        if self._framesCounter>0:
            avgSimTimeStepDuration = self._lastStepEndSimTimeFromStart/self._framesCounter
            totEpisodeWallDuration = time.monotonic() - self._lastPostResetTime
            epWallDurationUntilDone = self._lastValidStepWallTime - self._lastPostResetTime
            resetWallDuration = self._lastPostResetTime-self._lastPreResetTime
            wallFps = self._framesCounter/totEpisodeWallDuration
            wall_fps_until_done = self._framesCounter/epWallDurationUntilDone
            wall_fps_only_stepping = self._framesCounter/self._timeSpentStepping_ep
            ratio_time_spent_stepping = self._timeSpentStepping_ep/totEpisodeWallDuration
            wall_fps_first_to_last = self._framesCounter/(self._last_step_finish_time - self._first_step_start_time)
            ratio_time_spent_stepping_first_to_last = self._timeSpentStepping_ep/(self._last_step_finish_time - self._first_step_start_time)
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
        self._dbg_info["tot_ep_sim_duration"] = self._lastStepEndSimTimeFromStart
        self._dbg_info["reset_wall_duration"] = resetWallDuration
        self._dbg_info["ep_frames_count"] = self._framesCounter
        self._dbg_info["ep_reward"] = self._totalEpisodeReward
        self._dbg_info["wall_fps"] = wallFps
        self._dbg_info["wall_fps_until_done"] = wall_fps_until_done
        self._dbg_info["reset_count"] = self._resetCount
        self._dbg_info["ratio_time_spent_stepping"] = ratio_time_spent_stepping
        self._dbg_info["time_from_start"] = time.monotonic() - self._init_time
        self._dbg_info["total_steps"] = self._totalSteps
        self._dbg_info["wall_fps_first_to_last"] = wall_fps_first_to_last
        self._dbg_info["ratio_time_spent_stepping_first_to_last"] = ratio_time_spent_stepping_first_to_last
        self._dbg_info["success_ratio"] = self._success_ratio
        self._dbg_info["max_success_ratio"] = max(self._success_ratio, self._dbg_info.get("max_success_ratio",0))
        self._dbg_info["success"] = self._last_ep_succeded
        self._dbg_info["seed"] = self._ggEnv.get_seed()
        self._dbg_info["alltime_stepping_time"] = self._alltime_stepping_time
        self._dbg_info["wall_fps_only_stepping"] = wall_fps_only_stepping

        # self._dbg_info.update(self._ggEnv.getInfo(state))
        if len(self._total_sub_rewards)==0: # at the first step and episode this must be populated to at leat know which fields we'll have
            sub_rewards = {}
            # Not really setting the rewards, just populating the fields with zeros
            try:
                _ = self._ggEnv.computeReward(state,state,self._ggEnv.action_space.sample(), sub_rewards=sub_rewards, env_conf = self._ggEnv.get_configuration())
            except ValueError:
                pass
            self._total_sub_rewards = {k: v*0.0 for k,v in sub_rewards.items()}
        # ggLog.info(f'self._total_sub_rewards = {self._total_sub_rewards}')
        self._dbg_info.update({"ep_sub_"+k:v for k,v in self._total_sub_rewards.items()})


    def _logInfoCsv(self):
        if self._logFileCsvWriter is None:
            try:
                log_dir = os.path.dirname(self._episodeInfoLogFile)
                if log_dir != "":
                    os.makedirs(log_dir, exist_ok=True)
            except FileExistsError:
                pass
            existed = os.path.isfile(self._episodeInfoLogFile)
            if existed:
                with open(self._episodeInfoLogFile) as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    try:
                        columns = next(csvreader)
                    except StopIteration:
                        raise RuntimeError(f"A log file is already present but it is empty.")
                    lastRow = ""
                    for row in csvreader:
                        lastRow = row
                    self._resetCount += int(lastRow[columns.index("reset_count")])
                    self._totalSteps += int(lastRow[columns.index("total_steps")])
                    self._success_ratio += float(lastRow[columns.index("success_ratio")])
                    self._update_dbg_info()
            self._logFile = open(self._episodeInfoLogFile, "a")
            self._logFileCsvWriter = csv.writer(self._logFile, delimiter = ",")
            if not existed:
                self._logFileCsvWriter.writerow(self._dbg_info.keys())
        #print("writing csv")
        self._logFileCsvWriter.writerow([v.cpu().item() if isinstance(v, th.Tensor) else v for v in self._dbg_info.values()])
        self._logFile.flush()
        if self._use_wandb and adarl.utils.session.default_session.is_wandb_enabled():
            if self._logs_id is not None and self._logs_id!= "":
                prefix = self._logs_id+"/"
            else:
                prefix = ""
            d = {str(k) : v if type(v) is not bool else int(v) for k,v in self._dbg_info.items()}
            wandb_dict = {}
            wandb_dict.update({"lrg/"+k:v for k,v in d.items()})
            wandb_dict.update({prefix+k:v for k,v in d.items()})
            wandb_log(wandb_dict)

    def _build_info(self):
        info = {}
        state = self._getStateCached()
        truncated = self._ggEnv.reachedTimeout() and not self._ggEnv.is_timelimited() # If this env is time-limited this is not a truncation, it's the proper ending
        info["TimeLimit.truncated"] = truncated
        info["timed_out"] = self._ggEnv.reachedTimeout()
        ggInfo = self._ggEnv.getInfo(state=state)
        ggInfo["is_success"] = ggInfo.get("success", False)
        self._last_ep_succeded = ggInfo.get("success", False)
        if self._terminated:
            self._successes[self._resetCount%len(self._successes)] = int(self._last_ep_succeded)
            self._success_ratio = sum(self._successes)/min(len(self._successes), self._resetCount)
        info.update(self._dbg_info)
        info.update(ggInfo)
        return copy.deepcopy(info)

    def step(self, action) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Run one step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action
            Defines the action to be performed. See the environment implementation to know its format

        Returns
        -------
        Tuple[Sequence, int, bool, Dict[str,Any]]
            The first element is the observation. See the environment implementation to know its format
            The second element is the reward. See the environment implementation to know its format
            The third is True if the episode finished, False if it isn't
            The fourth is a dict containing auxiliary info. It contains the "simTime" element,
             which indicates the time reached by the simulation

        Raises
        -------
        AttributeError
            If an invalid action is provided

        """
        #ggLog.info("step()")

        t0 = time.monotonic()
        if self._framesCounter==0:
            self._first_step_start_time = t0
            self._last_step_finish_time = -1 # reset it
        if self._terminated:
            if self._verbose:
                ggLog.warn("Episode already finished")
            observation : ObsType = self._ggEnv.getObservation(self._getStateCached())
            reward = 0
            terminated = True
            truncated = self._ggEnv.reachedTimeout() and not self._ggEnv.is_timelimited() # If this env is time-limited this is not a truncation, it's the proper ending
            self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeSinceBuild()
            self._alltime_stepping_time += time.monotonic() - t0
            info = self._build_info()
            return (observation, reward, terminated, truncated, info)

        self._totalSteps += 1
        # Get previous observation

        with self._getPrevStateDurationAverage:
            previousState = self._getStateCached()

        # Setup action to perform
        with self._submitActionDurationAverage:
            self._ggEnv.submitAction(action)

        # Step the environment
        with self._simStepWallDurationAverage:
            self._lastStepStartEnvTime = self._ggEnv.getSimTimeSinceBuild()
            self._ggEnv.performStep()
            self._framesCounter+=1

        #Get new observation
        with self._getStateDurationAverage:
            state = self._getStateCached()
            self._lastStepEndEnvTime = self._ggEnv.getSimTimeSinceBuild()

        # Assess the situation
        with self._getObsRewDurationAverage:
            truncated = self._ggEnv.reachedTimeout() and not self._ggEnv.is_timelimited()
            self._terminated = self._ggEnv.reachedTerminalState(previousState, state)
            sub_rewards : Dict[str,th.Tensor] = {}
            reward = self._ggEnv.computeReward(previousState, state, action, env_conf=self._ggEnv.get_configuration(), sub_rewards = sub_rewards)
            observation = self._ggEnv.getObservation(state)
            self._totalEpisodeReward += reward
            # if self._total_sub_rewards is None:
            #     self._total_sub_rewards = {k:v for k,v in sub_rewards.items()}
            if len(sub_rewards) > 0 and sum(sub_rewards.values()) - reward > 0.001: raise RuntimeError(f"sub_rewards do not sum up to reward: {reward}!=sum({sub_rewards})")
            for k,v in sub_rewards.items():
                self._total_sub_rewards[k] += v
        

        tf = time.monotonic()
        self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeSinceBuild()
        self._lastValidStepWallTime = tf
        stepDuration = tf - t0
        self._envStepDurationAverage.addValue(newValue = stepDuration)
        self._timeSpentStepping_ep += stepDuration
        if self._framesCounter>1:
            self._last_step_finish_time = time.monotonic()
        #ggLog.info("stepped")
        self._alltime_stepping_time += time.monotonic() - t0

        self._update_dbg_info()
        info = self._build_info()
        return observation, reward, self._terminated, truncated, info






    def reset(self, seed = None, options = {}):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        if seed is not None:
            self._ggEnv.seed(seed)
        if self._verbose:
            ggLog.info(" ------- Resetting Environment (#"+str(self._resetCount)+")-------")

        if self._resetCount > 0:
            self._update_dbg_info()
            if self._framesCounter == 0:
                ggLog.info(f"No step executed in episode {self._resetCount}")
            else:
                if self._logEpisodeInfo:
                    self._logInfoCsv()
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
        self._ggEnv.performReset(options)
        self._lastPostResetTime = time.monotonic()

        if self._framesCounter!=0 and self._cumulativeImagesAge!=0:
            if self._cumulativeImagesAge/float(self._framesCounter)>0.01:
                ggLog.warn("Average delay of renderings = {:.4f}s".format(self._cumulativeImagesAge/float(self._framesCounter)))

        self._resetCount += 1
        self._framesCounter = 0
        self._cumulativeImagesAge = 0
        self._lastStepStartEnvTime = -1
        self._lastStepEndEnvTime = 0
        self._lastStepGotState = -1
        self._lastState = None
        self._totalEpisodeReward = 0.0
        self._total_sub_rewards = {}
        self._lastValidStepWallTime = -1
        self._timeSpentStepping_ep = 0

        #time.sleep(1)


        self._terminated = False


        self._envStepDurationAverage.reset()
        self._submitActionDurationAverage.reset()
        self._getStateDurationAverage.reset()
        self._getPrevStateDurationAverage.reset()
        self._getObsRewDurationAverage.reset()
        self._simStepWallDurationAverage.reset()
        self._update_dbg_info()

        #ggLog.info("reset() return")
        observation = self._ggEnv.getObservation(self._getStateCached())
        # print("observation space = "+str(self.observation_space)+" high = "+str(self.observation_space.high)+" low = "+str(self.observation_space.low))
        # print("observation = "+str(observation))
        # ggLog.info("GymEnvWrapper.reset() done")
        return observation, self._build_info()







    def render(self, mode : str = 'rgb_array') -> np.ndarray:
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

        npArrImage, imageTime = self._ggEnv.getUiRendering()

        # if imageTime < self._lastStepStartEnvTime:
        #     ggLog.warn(f"render(): The most recent camera image is older than the start of the last step! (by {self._lastStepStartEnvTime-imageTime}s, imageTime = {imageTime})")

        cameraImageAge = self._lastStepEndEnvTime - imageTime
        #ggLog.info("Rendering image age = "+str(cameraImageAge)+"s")
        self._cumulativeImagesAge += cameraImageAge


        return npArrImage









    def close(self):
        """Close the environment.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self._logEpisodeInfo and self._logFileCsvWriter is not None:
            self._logFile.close()
        self._ggEnv.close()









    # def seed(self, seed=None):
    #     """Set the seed for this env's random number generator(s).

    #     Note:
    #         Some environments use multiple pseudorandom number generators.
    #         We want to capture all such seeds used in order to ensure that
    #         there aren't accidental correlations between multiple generators.
    #     Returns:
    #         list<bigint>: Returns the list of seeds used in this env's random
    #           number generators. The first value in the list should be the
    #           "main" seed, or the value which a reproducer should pass to
    #           'seed'. Often, the main seed equals the provided 'seed', but
    #           this won't be true if seed=None, for example.
    #     """
    #     return self._ggEnv.seed(seed)





    def _getStateCached(self) -> Any:
        """Get the an observation of the environment keeping a cache of the last observation.

        Returns
        -------
        Any
            An observation of the environment. See the environment implementation for details on its format

        """
        if self._framesCounter != self._lastStepGotState:
            # ggLog.info("State cache miss")
            self._lastStepGotState = self._framesCounter
            self._lastState = self._ggEnv.getState()

        return self._lastState

    def getBaseEnv(self) -> BaseEnv:
        """Get the underlying adarl base environment

        Returns
        -------
        BaseEnv
            The adarl.BaseEnv object.
        """
        return self._ggEnv

    def __del__(self):
        # This is only called when the object is garbage-collected, so users should
        # still call close themselves, we don't know when garbage collection will happen
        self.close()

    @staticmethod
    def _compute_reward_nonbatch(achieved_goal, desired_goal, info, setGoalInState, computeReward):
        # The step function in BaseEnv fills the info up with the actual state and action
        # print("Computing reward")
        reachedState = info["gz_gym_base_env_reached_state"]
        previousState = info["gz_gym_base_env_previous_state"]
        action = info["gz_gym_base_env_action"]

        setGoalInState(previousState, desired_goal)
        setGoalInState(reachedState, desired_goal)
        reward = computeReward(previousState, reachedState, action)
        # print("Computed reward")
        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward for the provided inputs. This method must support batched inputs.
            This method follows the deprecated gym.GoalEnv interface.

        Parameters
        ----------
        achieved_goal : _type_
            Goal that was achieved
        desired_goal : _type_
            Goal that was desired
        info : _type_
            Extra info data

        Returns
        -------
        float, np.ndarray
            Reward or batch of rewards
        """
        if isinstance(info,dict):
            return self._compute_reward_nonbatch(achieved_goal,desired_goal,info)
        else:
            batch_size = len(achieved_goal)
            rewards = [None]*batch_size
            #assume its a batch
            # batch_size = desired_goal.shape[0]
            # for i in range(batch_size):
            #     rewards.append(self._compute_reward_nonbatch(achieved_goal[i], desired_goal[i], info[i]))
            if not hasattr(self, "_compute_reward_pool"):
                original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                self._compute_reward_pool = mp.Pool(min(mp.cpu_count()-1, 8))
                signal.signal(signal.SIGINT, original_sigint_handler)
            # print(f"Starting map to compute {batch_size} rewards")
            env_conf = self._ggEnv.get_configuration()
            reward_func = lambda *args, **kwargs: self._ggEnv.computeReward(*args, **kwargs, env_conf=env_conf)
            inputs = zip(  achieved_goal,
                            desired_goal,
                            info,
                            [self._ggEnv.setGoalInState]*batch_size,
                            [reward_func]*batch_size)
            # print(f"input[0] = {inpu  ts[0]}")
            rewards = self._compute_reward_pool.starmap(self._compute_reward_nonbatch, inputs)
            # for i in range(batch_size):
            #     rewards[i] = self._compute_reward_nonbatch(achieved_goal[i], desired_goal[i], info[i], self._ggEnv.setGoalInState, self._ggEnv.computeReward)
            reward= np.array(rewards, dtype=np.float64)
            
        return reward