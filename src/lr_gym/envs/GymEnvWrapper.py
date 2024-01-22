#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

# import traceback

import lr_gym.utils.dbg.ggLog as ggLog

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import Tuple, Dict, Any, SupportsFloat, TypeVar, Generic, Optional
import time
import csv

import lr_gym

from lr_gym.envs.BaseEnv import BaseEnv
import os
import lr_gym.utils.dbg.ggLog as ggLog

import multiprocessing as mp
import signal
import lr_gym.utils.session
import lr_gym.utils.utils
from lr_gym.utils.wandb_wrapper import wandb_log
import torch as th

ObsType = TypeVar("ObsType")

class GymEnvWrapper(gym.Env, Generic[ObsType]):
    """This class is a wrapper to convert lr_gym environments in OpenAI Gym environments.

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
                 logs_id : str = "",
                 use_wandb = True):
        """Short summary.

        Parameters
        ----------

        """
        
        self._use_wandb = use_wandb
        self._logs_id = logs_id
        self._ggEnv = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        if self._ggEnv.is_timelimited:
            reg_env_id = f"GymEnvWrapper-env-v0_{id(env)}_{int(time.monotonic()*1000)}"
            # gym.register(id=reg_env_id, entry_point=None, max_episode_steps = self._ggEnv.getMaxStepsPerEpisode())
            # self.spec = gym.spec(env_id=reg_env_id)
            self.spec = EnvSpec(id=reg_env_id, entry_point=None, max_episode_steps=self._ggEnv.getMaxStepsPerEpisode())
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
        self._first_step_finish_time = -1
        self._last_step_finish_time = -1


        self._envStepDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._submitActionDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._observationDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._wallStepDurationAverage =lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._lastStepEndSimTimeFromStart = 0
        self._lastValidStepWallTime = -1
        self._timeSpentStepping_ep = 0
        self._success_ratio = 0.0
        self._successes = [0]*50
        self._last_ep_succeded = False
        self._logFileCsvWriter = None
        self._info = {}

        self._terminated = False


    def _setInfo(self):
        if self._framesCounter>0:
            avgSimTimeStepDuration = self._lastStepEndSimTimeFromStart/self._framesCounter
            totEpisodeWallDuration = time.monotonic() - self._lastPostResetTime
            epWallDurationUntilDone = self._lastValidStepWallTime - self._lastPostResetTime
            resetWallDuration = self._lastPostResetTime-self._lastPreResetTime
            wallFps = self._framesCounter/totEpisodeWallDuration
            wall_fps_until_done = self._framesCounter/epWallDurationUntilDone
            ratio_time_spent_stepping_until_done = self._timeSpentStepping_ep/epWallDurationUntilDone
            ratio_time_spent_stepping = self._timeSpentStepping_ep/totEpisodeWallDuration
            wall_fps_first_to_last = self._framesCounter/(self._last_step_finish_time - self._first_step_finish_time)
            ratio_time_spent_stepping_first_to_last = self._timeSpentStepping_ep/(self._last_step_finish_time - self._first_step_finish_time)
        else:
            avgSimTimeStepDuration = float("NaN")
            totEpisodeWallDuration = 0
            resetWallDuration = float("NaN")
            wallFps = float("NaN")
            wall_fps_until_done = float("NaN")
            ratio_time_spent_stepping_until_done = 0
            ratio_time_spent_stepping = 0
            wall_fps_first_to_last = float("NaN")
            ratio_time_spent_stepping_first_to_last = 0

        self._info["avg_env_step_wall_duration"] = self._envStepDurationAverage.getAverage()
        self._info["avg_sim_step_wall_duration"] = self._wallStepDurationAverage.getAverage()
        self._info["avg_act_wall_duration"] = self._submitActionDurationAverage.getAverage()
        self._info["avg_obs_wall_duration"] = self._observationDurationAverage.getAverage()
        self._info["avg_step_sim_duration"] = avgSimTimeStepDuration
        self._info["tot_ep_wall_duration"] = totEpisodeWallDuration
        self._info["tot_ep_sim_duration"] = self._lastStepEndSimTimeFromStart
        self._info["reset_wall_duration"] = resetWallDuration
        self._info["ep_frames_count"] = self._framesCounter
        self._info["ep_reward"] = self._totalEpisodeReward
        self._info["ep_sub_rewards"] = self._total_sub_rewards
        self._info["wall_fps"] = wallFps
        self._info["wall_fps_until_done"] = wall_fps_until_done
        self._info["reset_count"] = self._resetCount
        self._info["ratio_time_spent_stepping_until_done"] = ratio_time_spent_stepping_until_done
        self._info["ratio_time_spent_stepping"] = ratio_time_spent_stepping
        self._info["time_from_start"] = time.monotonic() - self._init_time
        self._info["total_steps"] = self._totalSteps
        self._info["wall_fps_first_to_last"] = wall_fps_first_to_last
        self._info["ratio_time_spent_stepping_first_to_last"] = ratio_time_spent_stepping_first_to_last
        self._info["success_ratio"] = self._success_ratio
        self._info["max_success_ratio"] = max(self._success_ratio, self._info.get("max_success_ratio",0))
        self._info["success"] = self._last_ep_succeded
        self._info["seed"] = self._ggEnv.get_seed()

        self._info.update(self._ggEnv.getInfo(self._getStateCached()))


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
                    self._setInfo()
            self._logFile = open(self._episodeInfoLogFile, "a")
            self._logFileCsvWriter = csv.writer(self._logFile, delimiter = ",")
            if not existed:
                self._logFileCsvWriter.writerow(self._info.keys())
        #print("writing csv")
        self._logFileCsvWriter.writerow(self._info.values())
        self._logFile.flush()
        if lr_gym.utils.session.is_wandb_enabled() and self._use_wandb:
            if self._logs_id is not None and self._logs_id!= "":
                prefix = self._logs_id+"/"
            else:
                prefix = ""
            d = {str(k) : v if type(v) is not bool else int(v) for k,v in self._info.items()}
            wandb_dict = {}
            wandb_dict.update({"lrg/"+k:v for k,v in d.items()})
            wandb_dict.update({prefix+k:v for k,v in d.items()})
            wandb_log(lambda: wandb_dict)

    def _build_info(self):
        info = {}
        state = self._getStateCached()
        truncated = self._ggEnv.reachedTimeout() and not self._ggEnv.is_timelimited() # If this env is time-limited this is not a truncation, it's the proper ending
        info["TimeLimit.truncated"] = truncated
        info["timed_out"] = self._ggEnv.reachedTimeout()
        ggInfo = self._ggEnv.getInfo(state=state)
        if self._terminated:
            if "success" in ggInfo:
                ggInfo["is_success"] = ggInfo["success"]
                self._last_ep_succeded = ggInfo["success"]
                self._successes[self._resetCount%len(self._successes)] = int(self._last_ep_succeded)
                self._success_ratio = sum(self._successes)/min(len(self._successes), self._resetCount)
            info.update(ggInfo)
        info.update(self._info)
        return info

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

        if self._terminated:
            if self._verbose:
                ggLog.warn("Episode already finished")
            observation : ObsType = self._ggEnv.getObservation(self._getStateCached())
            reward = 0
            terminated = True
            truncated = self._ggEnv.reachedTimeout() and not self._ggEnv.is_timelimited() # If this env is time-limited this is not a truncation, it's the proper ending
            self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeFromEpStart()
            info = {}
            info.update(self._ggEnv.getInfo(state=self._getStateCached()))
            info["is_success"] = info.get("success",None)
            info["simTime"] = self._lastStepEndSimTimeFromStart
            info["TimeLimit.truncated"] = truncated
            info.update(self._info)
            return (observation, reward, terminated, truncated, info)

        self._totalSteps += 1
        # Get previous observation
        t0 = time.monotonic()
        previousState = self._getStateCached()

        # Setup action to perform
        t_preAct = time.monotonic()
        self._ggEnv.submitAction(action)
        self._submitActionDurationAverage.addValue(newValue = time.monotonic()-t_preAct)

        # Step the environment

        self._lastStepStartEnvTime = self._ggEnv.getSimTimeFromEpStart()
        t_preStep = time.monotonic()
        self._ggEnv.performStep()
        self._wallStepDurationAverage.addValue(newValue = time.monotonic()-t_preStep)
        self._framesCounter+=1

        #Get new observation
        t_preObs = time.monotonic()
        state = self._getStateCached()
        self._observationDurationAverage.addValue(newValue = time.monotonic()-t_preObs)
        self._lastStepEndEnvTime = self._ggEnv.getSimTimeFromEpStart()

        # Assess the situation
        self._terminated = self._ggEnv.checkEpisodeEnded(previousState, state)
        sub_rewards : Dict[str,th.Tensor] = {}
        reward = self._ggEnv.computeReward(previousState, state, action, env_conf=self._ggEnv.get_configuration(), sub_rewards = sub_rewards)
        observation = self._ggEnv.getObservation(state)
        truncated = self._ggEnv.reachedTimeout() and not self._ggEnv.is_timelimited()
        info = self._build_info()
        info.update({"gz_gym_base_env_reached_state" : state,
                    "gz_gym_base_env_previous_state" : previousState,
                    "gz_gym_base_env_action" : action})
        self._totalEpisodeReward += reward
        for k,v in sub_rewards.items():
            self._total_sub_rewards[k] = self._total_sub_rewards.get(k,0.0) + v
        
        ret = (observation, reward, self._terminated, truncated, info)

        self._lastStepEndSimTimeFromStart = self._ggEnv.getSimTimeFromEpStart()

        self._lastValidStepWallTime = time.monotonic()

        stepDuration = time.monotonic() - t0
        self._envStepDurationAverage.addValue(newValue = stepDuration)
        self._timeSpentStepping_ep += stepDuration
        if self._framesCounter==1:
            self._first_step_finish_time = time.monotonic()
            self._last_step_finish_time = -1
        else:
            self._last_step_finish_time = time.monotonic()
        #ggLog.info("stepped")
        return ret






    def reset(self, seed = None, options = {}):
        """Reset the state of the environment and return an initial observation.

        Returns
        -------
        Any
            the initial observation.

        """
        if len(options) > 0:
            raise NotImplementedError()
        if seed is not None:
            self._ggEnv.seed(seed)
        self._resetCount += 1
        if self._verbose:
            ggLog.info(" ------- Resetting Environment (#"+str(self._resetCount)+")-------")

        if self._framesCounter == 0:
            ggLog.info("No step executed in this episode")
        else:
            self._setInfo()
            if self._logEpisodeInfo:
                self._logInfoCsv()
            if self._verbose:
                for k,v in self._info.items():
                    ggLog.info(k," = ",v)
            elif not self._quiet:
                msg =  (f"ep = {self._info['reset_count']:d}"+
                        f" rwrd = {self._info['ep_reward']:.3f}"+
                        " stps = {:d}".format(self._info["ep_frames_count"])+
                        " wHz = {:.3f}".format(self._info["wall_fps"])+
                        " wHzFtl = {:.3f}".format(self._info["wall_fps_first_to_last"])+
                        " avg_stpWDur = {:f}".format(self._info["avg_env_step_wall_duration"])+
                        # " tstep/ttot_ftl = {:.2f}".format(self._info["ratio_time_spent_stepping_until_done"])+
                        " tstep/ttot = {:.2f}".format(self._info["ratio_time_spent_stepping"])+
                        " wEpDur = {:.2f}".format(self._info["tot_ep_wall_duration"])+
                        " sEpDur = {:.2f}".format(self._info["tot_ep_sim_duration"]))
                if "success_ratio" in self._info.keys():
                        msg += f" succ_ratio = {self._info['success_ratio']:.2f}"
                if "success" in self._info.keys():
                        msg += f" succ = {self._info['success']:.2f}"
                ggLog.info(msg)

        self._lastPreResetTime = time.monotonic()
        #reset simulation state
        self._ggEnv.performReset()
        self._lastPostResetTime = time.monotonic()

        if self._framesCounter!=0 and self._cumulativeImagesAge!=0:
            if self._cumulativeImagesAge/float(self._framesCounter)>0.01:
                ggLog.warn("Average delay of renderings = {:.4f}s".format(self._cumulativeImagesAge/float(self._framesCounter)))

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
        self._observationDurationAverage.reset()
        self._wallStepDurationAverage.reset()

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
        """Get the underlying lr_gym base environment

        Returns
        -------
        BaseEnv
            The lr_gym.BaseEnv object.
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