#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any

import pybullet as p

from lr_gym.env_controllers.PyBulletController import PyBulletController
from lr_gym.env_controllers.CartesianPositionEnvController import CartesianPositionEnvController
import lr_gym.utils.PyBulletUtils as PyBulletUtils
import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog
from nptyping import NDArray
from lr_gym.utils.utils import JointState




class PyBullet2DofCartesianController(PyBulletController, CartesianPositionEnvController):
    def __init__(self,  end_effector_link : Tuple[str,str],
                        xjoint : Tuple[str,str],
                        yjoint : Tuple[str,str],
                        stepLength_sec : float = 0.004166666666,
                        start_position = np.array([0.0, 0.0]),
                        restore_on_reset = True):
        super().__init__(stepLength_sec=stepLength_sec,
                         restore_on_reset=restore_on_reset)
        self._xjoint = xjoint
        self._yjoint = yjoint
        self._end_effector_link = end_effector_link
        self._step_timeout = 10
        self._position_tolerance = 0.001
        self._blocking_movement = True
        self._start_position = start_position
        self._target_position = start_position
        self._move_fails_in_last_step = 0

    def setCartesianPoseCommand(self,   linkPoses : Dict[Tuple[str,str],NDArray[(7,), np.float32]], do_cartesian : bool = True,
                                        velocity_scaling : float = 1.0,
                                        acceleration_scaling : float = 1.0) -> None:
        pose = linkPoses[self._end_effector_link]
        # if np.any(pose[2:]!=0):
        #     raise AttributeError(f"PyBullet2DofCartesianController only supports moving in the xy plane, set all other values to zero")
        self.setJointsPositionCommand({self._xjoint : pose[0],
                                       self._yjoint : pose[1]})
        
    def step(self):
        if not self._blocking_movement:
            return super().step()
        
        self._move_fails_in_last_step = 0
        sim_t0 = self.getEnvTimeFromStartup()
        elapsed_time = 0
        keep_going = True
        while keep_going and elapsed_time<self._step_timeout:
            self.freerun(self._stepLength_sec)
            ee_pos = self.getLinksState([self._end_effector_link])[self._end_effector_link].pose.position[0:2]
            err = np.linalg.norm(ee_pos - self._target_position)
            keep_going = err > self._position_tolerance
            elapsed_time = self.getEnvSimTimeFromStart()-sim_t0
        if keep_going:
            ggLog.warn(f"{type(self)}: move timed out (err = {err})")
            self._move_fails_in_last_step += 1
        return elapsed_time

    def resetWorld(self):
        super().resetWorld()

        self.setJointsStateDirect({ self._xjoint  : JointState(position=[self._start_position[0]], rate = [0], effort = [0]),
                                    self._yjoint : JointState(position=[self._start_position[1]], rate = [0], effort = [0])})

    def actionsFailsInLastStep(self):
        return self._move_fails_in_last_step