#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""


import numpy as np
import adarl.utils.spaces as spaces
from adarl.envs.BaseEnv import BaseEnv
from adarl.envs.vec.BaseVecEnv import BaseVecEnv
from typing import Tuple, Dict, Any, Sequence, Dict, Union, Optional, final
from abc import ABC, abstractmethod
import torch as th
from typing_extensions import deprecated
from adarl.utils.tensor_trees import map_tensor_tree, TensorTree

class VecToSingle(BaseEnv):
    """This is a base-class for implementing adarl environments.

    It defines more general methods to be implemented than the original gym.Env class.

    You can extend this class with a sub-class to implement specific environments.
    """

     # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 venv : BaseVecEnv):
        """_summary_

        Parameters
        ----------
        venv : adarl.env.BaseVecEnv
            Vec environment to use
        """
        self.venv = venv
        if self.venv.num_envs > 1:
            raise RuntimeError(f"Can only wrap vec environment of 1 env")
        super().__init__(   action_space = venv.single_action_space,
                            observation_space = venv.single_observation_space,
                            state_space = venv.single_state_space,
                            reward_space = venv.single_reward_space,
                            metadata = venv.metadata,
                            maxStepsPerEpisode = int(venv.get_max_episode_steps()[0]),
                            startSimulation = False,
                            is_timelimited = False)

    def submitAction(self, action : th.Tensor) -> None:
        super().submitAction(action)
        self.venv.submit_actions(action.unsqueeze(0))

    def reachedTimeout(self) -> th.Tensor:
        return self.venv.are_states_timedout(self._last_states)[0]

    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        return self.venv.are_states_terminal(state)[0]

    @abstractmethod
    def computeReward(self, previousState, state, action, env_conf = None, sub_rewards : Optional[Dict[str,th.Tensor]] = None) -> th.Tensor:
        return self.venv.compute_rewards(map_tensor_tree(state,lambda tensor: tensor.unsqueeze(0)))

    @abstractmethod
    def getObservation(self, state) -> Dict[Any, th.Tensor]:
        return self.venv.get_observations(states=map_tensor_tree(state,lambda tensor: tensor.unsqueeze(0)))

    @abstractmethod
    def getState(self) -> Dict[Any, th.Tensor]:
        return map_tensor_tree(self.venv.get_states(),lambda tensor: tensor[0])

    def initializeEpisode(self, options = {}):
        self.venv.initialize_episodes()

    @abstractmethod
    def performStep(self):
        super().performStep()
        self.venv.step()
        self._last_states = self.getState()

    @abstractmethod
    def performReset(self, options = {}):
        super().performReset()
        self.venv.reset()
        self._last_states = self.getState()


    @abstractmethod
    def getUiRendering(self) -> Tuple[th.Tensor, th.Tensor]:

        """To be implemented in subclass.

        This method is called by the render method to get the environment rendering for he user to watch (not for agent observation)

        Returns
        -------
        Tuple[th.Tensor, th.Tensor]
            The first tensor is the image, the second is the simulation time at which the image was rendered.

        """
        imgs, times = self.venv.get_ui_renderings(th.ones((self.venv.num_envs,),device=self.venv._th_device, dtype=th.bool))
        return imgs[0], times[0]

    @abstractmethod
    def getInfo(self,state=None) -> Dict[Any,Any]:
        i = super().getInfo()
        if state is None:
            state = self._last_states
        vinfo = self.venv.get_infos(map_tensor_tree(state,lambda tensor: tensor.unsqueeze(0)))
        i.update(vinfo)
        return i


    def get_max_episode_steps(self) -> th.Tensor:
        """Get the maximum number of frames of one episode, as set by the constructor (1-element tensor)."""
        return self.venv.get_max_episode_steps()[0]
    
    def set_max_episode_steps(self, max_steps : th.Tensor):
        """Get the maximum number of frames of one episode, as set by the constructor (1-element tensor)."""
        self.venv.set_max_episode_steps(max_steps.expand((self.venv.num_envs)))


    @deprecated("Do not implement this, just call whatever you need in the __init__ of the env")
    @abstractmethod
    def build(self, backend : str = "gazebo") -> None:
        """To be implemented in subclass.

        Build the environment.
        """
        pass

    @abstractmethod
    def _destroy(self) -> None:
        """To be implemented in subclass.

        Destroy the environment, releasing whatever resource it may be holding.
        Called at the end of the environment lifecycle (i.e. when close gets called)
        """
        pass

    @abstractmethod
    def getSimTimeSinceBuild(self) -> th.Tensor:
        """Get the elapsed time since the episode start."""
        raise NotImplementedError()

    def close(self) -> None:
        if not self._closed:
            self.venv.close()
            self._destroy()
            self._closed = True

    def seed(self, seed : int) -> None:
        if seed is not None:
            self.venv.set_seeds(th.as_tensor(seed).expand((self.venv.num_envs,)))
            self._envSeed = seed

    def get_seed(self):
        return self._envSeed
