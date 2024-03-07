#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any, Optional

import pybullet
import time
from lr_gym.env_controllers.PyBulletController import PyBulletController



class PyBulletJointImpedanceController(PyBulletController):

    def __init__(self, stepLength_sec : float = 0.004166666666,
                        restore_on_reset = True,
                        debug_gui : bool = False,
                        real_time_factor : Optional[float] = None,
                        global_max_torque_position_control : float = 100,
                        joints_max_torque_position_control : Dict[Tuple[str,str],float] = {},
                        global_max_velocity_position_control : float = 1,
                        joints_max_velocity_position_control : Dict[Tuple[str,str],float] = {},
                        global_max_acceleration_position_control : float = 10,
                        joints_max_acceleration_position_control : Dict[Tuple[str,str],float] = {},
                        simulation_step = 1/960):
        """Initialize the Simulator controller.

        """
        super().__init__(
            stepLength_sec = stepLength_sec,
            restore_on_reset = restore_on_reset,
            debug_gui = debug_gui,
            real_time_factor = real_time_factor,
            global_max_torque_position_control = global_max_torque_position_control,
            joints_max_torque_position_control = joints_max_torque_position_control,
            global_max_velocity_position_control = global_max_velocity_position_control,
            joints_max_velocity_position_control = joints_max_velocity_position_control,
            global_max_acceleration_position_control = global_max_acceleration_position_control,
            joints_max_acceleration_position_control = joints_max_acceleration_position_control,
            simulation_step = simulation_step
        )

    
    def _apply_controls(self):
        self._apply_commanded_joint_impedances()
        self._apply_commanded_torques()
        self._apply_commanded_velocities()
        self._apply_commanded_positions()


    def _clear_commands(self):
        super()._clear_commands()
        self._commanded_joint_impedances = {}


    def _apply_commanded_joint_impedances(self):
        js = super().getJointsState(list(self._commanded_joint_impedances.keys()))
        tcmd = {}
        for jn, cmd in self._commanded_joint_impedances.items():
            pref, vref, tref, pgain, vgain = cmd
            p = js[jn].position[0]
            v = js[jn].rate[0]
            torque = pgain*(pref-p) + vgain*(vref-v) + tref

            max_torque = min(self._max_torque_pos_control, self._max_torques_pos_control.get(jn,float("+inf")))
            if torque>max_torque:
                torque = max_torque
            elif torque<-max_torque:
                torque = -max_torque

            bodyId, jointId = self._getBodyAndJointId(jn)
            if bodyId not in tcmd:
                tcmd[bodyId] = ([],[])
            tcmd[bodyId][0].append(jointId)
            tcmd[bodyId][1].append(torque)
            self._last_step_commanded_torques.append((jn,torque))

        for bodyId, command in tcmd.items():
            pybullet.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=command[0],
                                        controlMode=pybullet.TORQUE_CONTROL,
                                        forces=command[1])


    def setJointsImpedanceCommand(self, jointImpedances : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]) -> None:
        """ Sets a joint impedance command. This gets cleared out at every step or freerun. Position, velocity and torque commands will override this.

        Parameters
        ----------
        jointImpedances : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]
            Dictionary with:
             - key=(<model_name>,<joint_name>)
             - value=(<position_reference>,<velocity_reference>,<torque_reference>,<position_gain>,<velocity_gain>)
        """
        self._commanded_joint_impedances = {ji[0]:ji[1] for ji in jointImpedances}