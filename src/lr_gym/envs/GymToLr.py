#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

from lr_gym.envs.BaseEnv import BaseEnv

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Sequence, SupportsFloat
import torch as th



class GymToLr(BaseEnv):

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self, openaiGym_env : gym.Env, stepSimDuration_sec : float = 1, maxStepsPerEpisode = None):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times

        """

        if maxStepsPerEpisode is None:
            if openaiGym_env.spec is not None:
                maxStepsPerEpisode = openaiGym_env.spec.max_episode_steps
            if maxStepsPerEpisode is None and hasattr(openaiGym_env,"_max_episode_steps"):
                maxStepsPerEpisode = openaiGym_env._max_episode_steps
            if maxStepsPerEpisode is None:
                raise RuntimeError("Cannot determine maxStepsPerEpisode from openaiGym_env env, you need to specify it manually")

        super().__init__(   maxStepsPerEpisode = maxStepsPerEpisode,
                            startSimulation = True,
                            simulationBackend = "OpenAiGym",
                            verbose = False,
                            quiet = False)

        self._openaiGym_env = openaiGym_env

        self._actionToDo = None # This will be set by submitAction an then used in step()
        self._prev_observation = None #Observation before the last
        self._last_observation, self._last_reward, self._last_terminated, self._last_truncated, self._last_info = (None,)*5
        self._stepCount = 0
        self._stepSimDuration_sec = stepSimDuration_sec
        self._envSeed = 0
        self._must_set_seed = True

        self.action_space = self._openaiGym_env.action_space
        self.observation_space = self._openaiGym_env.observation_space


    def submitAction(self, action) -> None:
        super().submitAction(action)
        self._actionToDo = action


    def checkEpisodeEnded(self, previousState, state) -> th.Tensor:
        if self._last_terminated is not None:
            ended = self._last_terminated
        else:
            ended = False
        ended = ended or super().checkEpisodeEnded(previousState, state)
        return ended


    def computeReward(self, previousState, state, action, env_conf = None) -> SupportsFloat:
        if not (state is self._last_observation and action is self._actionToDo and previousState is self._prev_observation):
            raise RuntimeError("GymToLr.computeReward is only valid if used for the last executed step. And it looks like you tried using it for something else.")
        return self._last_reward

    def getObservation(self, state) -> Dict[Any,th.Tensor]:
        return th.as_tensor(state)

    def getState(self) -> Sequence:
        return self._last_observation


    def initializeEpisode(self) -> None:
        pass


    def performStep(self) -> None:
        super().performStep()
        self._prev_observation = self._last_observation
        self._stepCount += 1
        # time.sleep(1)
        # print(f"Step {self._stepCount}, memory usage = {psutil.Process(os.getpid()).memory_info().rss/1024} KB")
        obs, rew, term, trunc, info = self._openaiGym_env.step(self._actionToDo)
        # convert  to dict obs and pytorch tensors
        if not isinstance(obs, Dict):
            obs = {"obs": obs}
        obs = {k:th.as_tensor(v) for k,v in obs.items()}
        rew = th.as_tensor(rew)
        term = th.as_tensor(term)
        trunc = th.as_tensor(trunc)
        info = {k: th.as_tensor(v) if isinstance(v,(np.ndarray,th.Tensor)) else v for k,v in info.items()}
        
        self._last_observation = obs
        self._last_reward = rew
        self._last_terminated = term
        self._last_truncated = trunc
        self._last_info = info

    def performReset(self) -> None:
        super().performReset()
        self._prev_observation = None
        self._stepCount = 0
        if self._must_set_seed:
            seed = self._envSeed
        else:
            seed = None

        obs, info = self._openaiGym_env.reset(seed=seed)

        if not isinstance(obs, Dict):
            obs = {"obs": obs}
        obs = {k:th.as_tensor(v) for k,v in obs.items()}
        info = {k: th.as_tensor(v) if isinstance(v,(np.ndarray,th.Tensor)) else v for k,v in info.items()}        
        self._last_observation = obs
        self._last_info = info



    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        return self._openaiGym_env.render(), self.getSimTimeFromEpStart()

    def getInfo(self,state=None) -> Dict[str,Any]:
        """To be implemented in subclass.

        This method is called by the step method. The values returned by it will be appended in the info variable returned bby step
        """
        r = self._last_info
        if r is None:
            r = {}
        return r

    def getMaxStepsPerEpisode(self):
        """Get the maximum number of frames of one episode, as set by the constructor."""
        return self._maxStepsPerEpisode

    def setGoalInState(self, state, goal):
        """To be implemented in subclass.

        Update the provided state with the provided goal. Useful for goal-oriented environments, especially when using HER.
        It's used by ToGoalEnvWrapper.
        """
        raise NotImplementedError()

    def buildSimulation(self, backend : str = "gazebo"):
        """To be implemented in subclass.

        Build a simulation for the environment.
        """
        pass

    def _destroySimulation(self):
        """To be implemented in subclass.

        Destroy a simulation built by buildSimulation.
        """
        pass

    def getSimTimeFromEpStart(self):
        """Get the elapsed time since the episode start."""
        return self._stepCount * self._stepSimDuration_sec

    def close(self):
        self._destroySimulation()

    def seed(self, seed=None):
        if seed is not None:
            self._envSeed = seed
            self._must_set_seed= True
        # self._openaiGym_env.seed(seed)
        return [self._envSeed]
