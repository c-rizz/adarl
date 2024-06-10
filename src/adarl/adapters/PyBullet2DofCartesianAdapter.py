#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any, Optional

import pybullet as p

from adarl.adapters.PyBulletAdapter import PyBulletAdapter
from adarl.adapters.BaseCartesianPositionAdapter import BaseCartesianPositionAdapter
import adarl.utils.PyBulletUtils as PyBulletUtils
import numpy as np
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import JointState
import time



class PyBullet2DofCartesianAdapter(PyBulletAdapter, BaseCartesianPositionAdapter):
    def __init__(self,  end_effector_link : Tuple[str,str],
                        xjoint : Tuple[str,str],
                        yjoint : Tuple[str,str],
                        stepLength_sec : float = 0.0165,
                        start_position = np.array([0.0, 0.0]),
                        restore_on_reset = True,
                        step_timeout_sec = 10.0,
                        debug_gui : bool = False,
                        global_max_torque_position_control : float = 100,
                        joints_max_torque_position_control : float = {}):
        super().__init__(stepLength_sec=stepLength_sec,
                         restore_on_reset=restore_on_reset,
                         debug_gui=debug_gui,
                        global_max_torque_position_control = global_max_torque_position_control,
                        joints_max_torque_position_control = joints_max_torque_position_control)
        self._xjoint = xjoint
        self._yjoint = yjoint
        self._end_effector_link = end_effector_link
        self._step_timeout = step_timeout_sec
        self._position_tolerance = 0.001
        self._blocking_movement = True
        self._start_position = np.array(start_position)
        self._target_position = None
        self._move_fails_in_last_step = 0

    def _ik(self, pos_xy):
        pos_base_frame = np.array(pos_xy) # Assumes the two joints directly control world x and y
        return pos_base_frame

    def setCartesianPoseCommand(self,   linkPoses : Dict[Tuple[str,str],np.typing.NDArray[(7,), np.float32]],
                                        do_cartesian : bool = True,
                                        velocity_scaling : float = 1.0,
                                        acceleration_scaling : float = 1.0) -> None:
        pose = linkPoses[self._end_effector_link]
        if len(linkPoses) > 1:
            raise AttributeError(f"We support moving the end effector, but you requested links {linkPoses.keys()}")
        # if np.any(pose[2:]!=0):
        #     raise AttributeError(f"PyBullet2DofCartesianAdapter only supports moving in the xy plane, set all other values to zero")
        jpose = self._ik(pose[0:2])
        # ggLog.info(f"Got request {pose} \t Moving to join pose {jpose}")
        self._target_position = pose[0:2]
        self.setJointsPositionCommand(  {self._xjoint : jpose[0],
                                         self._yjoint : jpose[1]},
                                        velocity_scaling = velocity_scaling,
                                        acceleration_scaling = acceleration_scaling)


    def moveToEePoseSync(self,  poses : Dict[Tuple[str,str],List[float]] = None, 
                                do_cartesian = False,
                                velocity_scaling : Optional[float] = None,
                                acceleration_scaling : Optional[float] = None,
                                ee_link : Optional[Tuple[str,str]] = None,
                                reference_frame : Optional[str] = None,
                                max_error : float = 0.001,
                                step_time : Optional[float] = 0.1,
                                wall_timeout_sec : float = 30,
                                sim_timeout_sec : float = 30) -> None:
        if step_time is None:
            step_time = self._stepLength_sec
        if ee_link is None:
            ee_link = self._end_effector_link

        links = poses.keys()
        if len(links) > 1 or ee_link not in links:
            raise AttributeError(f"{type(self).__name__}.moveToEePoseSync only supports moving ee_link (={ee_link}), but you requested links {links}")
        link = list(links)[0]
        req_pos = poses[link][0:3]
        curr_pos = self.getLinksState([link])[link].pose.position
        err = np.linalg.norm(req_pos - curr_pos)

        t0_wall = time.monotonic()
        t0_sim = self.getEnvTimeFromStartup()

        while err > max_error:
            self.setCartesianPoseCommand(   linkPoses = poses,
                                            do_cartesian = do_cartesian,
                                            velocity_scaling = velocity_scaling,
                                            acceleration_scaling = acceleration_scaling)
            self.run(duration_sec=step_time)
            curr_pos = self.getLinksState([link])[link].pose.position
            err = np.linalg.norm(req_pos - curr_pos)

            wall_d = time.monotonic()-t0_wall
            if wall_d > wall_timeout_sec:
                raise TimeoutError(f"moveToEePoseSync wall timeout: {wall_d:.2f} > {wall_timeout_sec:.2f} req_pos = {req_pos} curr_pos = {curr_pos} err = {err}")
            sim_d = self.getEnvTimeFromStartup()-t0_sim
            if sim_d > sim_timeout_sec:
                raise TimeoutError(f"moveToEePoseSync sim timeout: {sim_d:.2f} > {sim_timeout_sec:.2f} req_pos = {req_pos} curr_pos = {curr_pos} err = {err}")
            
    def step(self):
        if not self._blocking_movement:
            return super().step()
        
        self._move_fails_in_last_step = 0
        sim_t0 = self.getEnvTimeFromStartup()
        elapsed_time = 0
        keep_going = True
        while keep_going and elapsed_time<self._step_timeout:
            self.run(self._stepLength_sec)
            ee_pos = self.getLinksState([self._end_effector_link])[self._end_effector_link].pose.position[0:2]
            err = np.linalg.norm(ee_pos - self._target_position)
            keep_going = err > self._position_tolerance
            elapsed_time = self.getEnvTimeFromStartup()-sim_t0
        if keep_going:
            # ggLog.warn(f"{type(self)}: move timed out (ee_pos = {ee_pos}, target_position = {self._target_position}, err = {err})")
            self._move_fails_in_last_step += 1
        self.clear_commands()
        return elapsed_time

    def resetWorld(self):
        super().resetWorld()
        self._target_position = self._start_position
        self.setJointsStateDirect({ self._xjoint  : JointState(position=[self._start_position[0]], rate = [0], effort = [0]),
                                    self._yjoint : JointState(position=[self._start_position[1]], rate = [0], effort = [0])})

    def actionsFailsInLastStep(self):
        return self._move_fails_in_last_step