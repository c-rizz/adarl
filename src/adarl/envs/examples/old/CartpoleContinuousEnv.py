#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""


import adarl.utils.spaces as spaces
import numpy as np
from adarl.envs.CartpoleEnv import CartpoleEnv

class CartpoleContinuousEnv(CartpoleEnv):
    """This is a slight modification on the CartpoleEnv that makes it accept continuous actions.
    Still, the left and right pushes are of a fixed amount. Actions in [0,0.5] push to the left, acitons in [0.5,1] push right.
    """

    action_space = spaces.gym_spaces.Box(low=np.array([0]),high=np.array([1]))

    def submitAction(self, action : int) -> None:
        super(CartpoleEnv, self).submitAction(action) #skip CartpoleEnv's submitAction, call its parent one
        # print("action = ",action)
        # print("type(action) = ",type(action))
        if action < 0.5: #left
            direction = -1
        elif action >= 0.5:
            direction = 1
        else:
            raise AttributeError("Invalid action (it's "+str(action)+")")

        self._adapter.setJointsEffortCommand(jointTorques = [("cartpole_v0","foot_joint", direction * 20)])
