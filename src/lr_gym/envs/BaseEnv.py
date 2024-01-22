#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""


import numpy as np
import lr_gym.utils.spaces as spaces
from typing import Tuple, Dict, Any, Sequence, Dict, Union, Optional
from abc import ABC, abstractmethod
import torch as th

class BaseEnv(ABC):
    """This is a base-class for implementing lr_gym environments.

    It defines more general methods to be implemented than the original gym.Env class.

    You can extend this class with a sub-class to implement specific environments.
    """
    #TODO: This should be an abstract class, defined via python's ABC

     # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 action_space : spaces.gym_spaces.Space = None,
                 observation_space : spaces.gym_spaces.Space = None,
                 pure_observation_space : spaces.gym_spaces.Space = None,
                 goal_observation_space : spaces.gym_spaces.Space = None,
                 reward_space = spaces.gym_spaces.Box(low=np.array([float("-inf")]), high=np.array([float("+inf")]), dtype=np.float32),
                 metadata = {},
                 maxStepsPerEpisode : int = 500,
                 startSimulation : bool = False,
                 simulationBackend : str = None,
                 verbose : bool = False,
                 quiet : bool = False,
                 is_timelimited : bool = True):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The checkEpisodeEnded() function will return
            done=True after being called this number of times

        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.pure_observation_space = pure_observation_space
        self.goal_observation_space = goal_observation_space
        self.reward_space = reward_space
        self.metadata = metadata

        self._actionsCounter = 0
        self._stepCounter = 0
        self._maxStepsPerEpisode = th.as_tensor(maxStepsPerEpisode)
        self._backend = simulationBackend
        self._envSeed : int = 0
        self._is_timelimited = th.as_tensor(is_timelimited)
        self._closed = False

        if startSimulation:
            self.buildSimulation(backend=simulationBackend)



    @abstractmethod
    def submitAction(self, action : th.Tensor) -> None:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. It is called while the simulation is paused and
        should perform the provided action.

        Parameters
        ----------
        action : type
            The action to be performed the format of this is up to the subclass to decide

        Returns
        -------
        None

        Raises
        -------
        AttributeError
            Raised if the provided action is not valid

        """
        self._actionsCounter += 1

    def reachedTimeout(self) -> th.Tensor:
        """
        If maxStepsPerEpisode is reached. Usually not supposed to be subclassed.
        """
        return self.getMaxStepsPerEpisode()>0 and self._stepCounter >= self.getMaxStepsPerEpisode()

    
    def checkEpisodeEnded(self, previousState, state) -> th.Tensor:
        """To be implemented in subclass.

        If the episode has finished. In the subclass you should OR this with your own conditions.

        Parameters
        ----------
        previousState : type
            The observation before the simulation was stepped forward
        state : type
            The observation after the simulation was stepped forward

        Returns
        -------
        bool
            Return True if the episode has ended, False otherwise

        """
        return self.reachedTimeout()

    @abstractmethod
    def computeReward(self, previousState, state, action, env_conf = None, sub_rewards : Optional[Dict[str,th.Tensor]] = None) -> th.Tensor:
        """To be implemented in subclass.

        This method is called during the stepping of the simulation. Just after the simulation has been stepped forward
        this method is used to compute the reward for the step.

        Parameters
        ----------
        previousState : type
            The state before the simulation was stepped forward
        state : type
            The state after the simulation was stepped forward

        Returns
        -------
        th.Tensor
            The reward for this step 

        """
        raise NotImplementedError()

    @abstractmethod
    def getObservation(self, state) -> Dict[Any, th.Tensor]:
        """To be implemented in subclass.

        Get an observation of the environment.

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()

    @abstractmethod
    def getState(self) -> Dict[Any, th.Tensor]:
        """To be implemented in subclass.

        Get the state of the environment form the simulation

        Returns
        -------
        Sequence
            An observation of the environment. See the environment implementation for details on its format

        """
        raise NotImplementedError()

    def initializeEpisode(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to allow the sub-class to reset environment-specific details

        """
        pass

    @abstractmethod
    def performStep(self) -> None:
        """To be implemented in subclass.

        This method is called to perform the stepping of the environment. In the case of
        simulated environments this means stepping forward the simulated time.
        It is called after submitAction and before getting the state observation

        """
        self._stepCounter+=1
        return

    @abstractmethod
    def performReset(self) -> None:
        """To be implemented in subclass.

        This method is called by the reset method to perform the actual reset of the environment to its initial state

        """
        self._stepCounter = 0
        self._actionsCounter = 0


    @abstractmethod
    def getUiRendering(self) -> Tuple[Union[np.ndarray, th.Tensor], float]:
        """To be implemented in subclass.

        This method is called by the render method to get the environment rendering

        """

        raise NotImplementedError()

    @abstractmethod
    def getInfo(self,state=None) -> Dict[Any,Any]:
        """To be implemented in subclass.

        This method is called by the step method. The values returned by it will be appended in the info variable returned bby step
        """
        return {"timed_out" : self.reachedTimeout()}


    def getMaxStepsPerEpisode(self) -> th.Tensor:
        """Get the maximum number of frames of one episode, as set by the constructor (1-element tensor)."""
        return self._maxStepsPerEpisode

    @abstractmethod
    def buildSimulation(self, backend : str = "gazebo") -> None:
        """To be implemented in subclass.

        Build a simulation for the environment.
        """
        raise NotImplementedError() #TODO: Move this into the environmentControllers

    @abstractmethod
    def _destroySimulation(self) -> None:
        """To be implemented in subclass.

        Destroy a simulation built by buildSimulation.
        """
        pass

    @abstractmethod
    def getSimTimeFromEpStart(self) -> th.Tensor:
        """Get the elapsed time since the episode start."""
        raise NotImplementedError()

    def close(self) -> None:
        if not self._closed:
            self._destroySimulation()
            self._closed = True

    def seed(self, seed : int) -> None:
        if seed is not None:
            self._envSeed = seed

    def get_seed(self):
        return self._envSeed

    def is_timelimited(self) -> th.Tensor:
        return self._is_timelimited
    
    def get_configuration(self):
        return None