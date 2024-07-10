#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

from adarl.envs.BaseEnv import BaseEnv

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Sequence, SupportsFloat, TypeVar, Generic
import torch as th
import adarl.utils.utils
import gymnasium
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.tensor_trees import map_tensor_tree

ObsType = TypeVar("ObsType")

def to_contiguous_tensor(value):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
    return th.as_tensor(value)


class GymToLr(BaseEnv, Generic[ObsType]):

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self, openaiGym_env : gym.Env, stepSimDuration_sec : float = 1, maxStepsPerEpisode = None,
                 copy_observations : bool = False,
                 actions_to_numpy : bool = False):
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
                maxStepsPerEpisode = openaiGym_env._max_episode_steps #type:ignore
            if maxStepsPerEpisode is None:
                raise RuntimeError("Cannot determine maxStepsPerEpisode from openaiGym_env env, you need to specify it manually")

        self._openaiGym_env = openaiGym_env
        self._copy_observations = copy_observations
        self._actions_to_numpy = actions_to_numpy
        state_space = gymnasium.spaces.Dict({
                "internal_info":gymnasium.spaces.Dict({
                        "ep" : gymnasium.spaces.Box(low = np.array(np.iinfo(np.int32).min),high = np.array(np.iinfo(np.int32).max)),
                        "step" : gymnasium.spaces.Box(low = np.array(np.iinfo(np.int32).min),high = np.array(np.iinfo(np.int32).max)),
                        "reward" : gymnasium.spaces.Box(low = np.array(float("-inf")),high = np.array(float("+inf"))),
                        "action" : openaiGym_env.action_space}),
                "obs" : openaiGym_env.observation_space})
        super().__init__(   maxStepsPerEpisode = maxStepsPerEpisode,
                            startSimulation = True,
                            observation_space=self._openaiGym_env.observation_space,
                            action_space = self._openaiGym_env.action_space,
                            state_space=state_space)


        self._actionToDo = None # This will be set by submitAction an then used in step()
        self._prev_observation = None #Observation before the last
        self._last_observation : ObsType = None
        self._last_reward = th.tensor(0.)
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = {}
        self._last_action = self._openaiGym_env.action_space.sample()
        self._stepCount = 0
        self._stepSimDuration_sec = stepSimDuration_sec
        self._envSeed = 0
        self._must_set_seed = True
        self._ep_count = 0


    def submitAction(self, action) -> None:
        super().submitAction(action)
        self._actionToDo = action

    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        return state["internal_info"]["terminated"]

    def computeReward(self, previousState, state, action, env_conf = None, sub_rewards = {}) -> th.Tensor:
        if (state["internal_info"]["ep"] == previousState["internal_info"]["ep"] and
            state["internal_info"]["step"] == previousState["internal_info"]["step"]+1):
            return state["internal_info"]["reward"]
        else:
            raise ValueError(f"Cannot compute reward for this transition.\n state[internal_info'] = {state['internal_info']},\n previousState[internal_info'] = {previousState['internal_info']}, action = {action}")

    def getObservation(self, state) -> ObsType:
        obs = state["obs"]
        return obs

    def getState(self):
        # ggLog.info(f"gymtolr returning obs {self._last_observation}")
        internal_info = {"step":self._stepCounter,
                          "ep":self._ep_count,
                          "reward":self._last_reward,
                          "action":self._last_action,
                          "terminated":self._last_terminated,
                          "truncated":self._last_truncated,}
        ret = {"internal_info":internal_info,
                "obs":self._last_observation}
        # ggLog.info(f"GymToLr(): Returning {ret}")
        return ret


    def initializeEpisode(self) -> None:
        pass


    def performStep(self) -> None:
        super().performStep()
        self._prev_observation = self._last_observation
        self._stepCount += 1
        # time.sleep(1)
        # print(f"Step {self._stepCount}, memory usage = {psutil.Process(os.getpid()).memory_info().rss/1024} KB")
        act = self._actionToDo
        if self._actions_to_numpy and isinstance(act, th.Tensor):
            if act.dim() == 0:
                act = act.unsqueeze(0)
            act = act.cpu().numpy()
        # ggLog.info(f"acting with {act} ({type(act)})")
        obs, rew, term, trunc, info = self._openaiGym_env.step(act)
        # ggLog.info(f"gymtolr stepped, obs = {obs}")
        # convert  to dict obs and pytorch tensors

        obs = map_tensor_tree(obs, to_contiguous_tensor)
        if self._copy_observations:
            obs = map_tensor_tree(obs, lambda t: t.detach().clone())
        self._last_observation = obs
        
        # ggLog.info(f"gymtolr set last_obs to {self._last_observation}")
        self._last_reward = th.as_tensor(rew)
        self._last_action = th.as_tensor(self._actionToDo)
        self._last_terminated = th.as_tensor(term)
        self._last_truncated = th.as_tensor(trunc)
        self._last_info = {k: to_contiguous_tensor(v) for k,v in info.items()}

    def performReset(self, options = {}) -> None:
        super().performReset()
        self._prev_observation = None
        self._stepCount = 0
        self._ep_count += 1
        if self._must_set_seed:
            seed = self._envSeed
        else:
            seed = None

        obs, info = self._openaiGym_env.reset(seed=seed, options=options)
        # ggLog.info(f"gymtolr resetted, obs = {obs}")
        # if not isinstance(obs, Dict):
        #     obs = {"obs": obs}
        obs = map_tensor_tree(obs, to_contiguous_tensor)
        if self._copy_observations:
            obs = map_tensor_tree(obs, lambda t: t.detach().clone())
        self._last_observation = obs
        # ggLog.info(f"gymtolr reset last_obs to {self._last_observation}")
        self._last_info = {k: to_contiguous_tensor(v) for k,v in info.items()}



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

    def get_max_episode_steps(self):
        """Get the maximum number of frames of one episode, as set by the constructor."""
        return self._maxStepsPerEpisode

    # def setGoalInState(self, state, goal):
    #     """To be implemented in subclass.

    #     Update the provided state with the provided goal. Useful for goal-oriented environments, especially when using HER.
    #     It's used by ToGoalEnvWrapper.
    #     """
    #     raise NotImplementedError()

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
