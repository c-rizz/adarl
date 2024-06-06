#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""


import numpy as np

from adarl.envs.panda.PandaEffortBaseEnv import PandaEffortBaseEnv
import adarl





class PandaEffortStayUpEnv(PandaEffortBaseEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep the end-effector as high as possible."""

    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    maxTorques = [100, 100, 100, 100, 100, 100, 100],
                    environmentController : adarl.adapters.BaseAdapter = None,
                    startSimulation : bool = False):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        maxTorques : Tuple[float, float, float, float, float, float, float]
            Maximum torque that can be applied ot each joint
        environmentController : adarl.adapters.BaseAdapter
            The contorller to be used to interface with the environment. If left to None an EffortRosControlAdapter will be used.

        """


        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         render = render,
                         maxTorques = maxTorques,
                         environmentController = environmentController,
                         startSimulation = startSimulation)



    def computeReward(self, previousState : np.typing.NDArray[(20,), np.float32], state : np.typing.NDArray[(20,), np.float32], action : int, env_conf = None) -> float:

        position = state[0:3]
        prevPosition = previousState[0:3]

        reward = (position[2]*position[2]+(position[2]-prevPosition[2])).item()
        return reward
