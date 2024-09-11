#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Sequence, Mapping
from typing_extensions import override

from adarl.adapters.PyBulletAdapter import PyBulletAdapter
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
import torch as th
import copy
import adarl.utils.dbg.ggLog as ggLog

class PyBulletJointImpedanceAdapter(PyBulletAdapter, BaseJointImpedanceAdapter):

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
                        simulation_step = 1/960,
                        enable_redering = True):
        """Initialize the Simulator controller.

        """
        self._jimpedance_controlled_joints : list[tuple[str,str]] = []
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
            simulation_step = simulation_step,
            enable_redering = enable_redering
        )

    
    def _apply_controls(self):
        self._apply_commanded_joint_impedances()
        self._apply_commanded_torques()
        self._apply_commanded_velocities()
        self._apply_commanded_positions()

    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[Tuple[str,str]]):
        self._jimpedance_controlled_joints = list(joint_names)

    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str,str]]:
        return self._jimpedance_controlled_joints

    @override
    def clear_commands(self):
        super().clear_commands()
        self._commanded_joint_impedances : dict[float, dict] = {}

    def _apply_commanded_joint_impedances(self):
        # get the newest command with a time less or equal to now
        cmd = {}
        tcmd = float("-inf")
        future_commands = {} # Commands to be applied in the future
        t = self.getEnvTimeFromStartup()
        # ggLog.info(f"t = {t}")
        # ggLog.info(f"self._commanded_joint_impedances = {self._commanded_joint_impedances}")
        for tc,c in self._commanded_joint_impedances.items():
            if tc > t:
                future_commands[tc] = c
            if tc <= t and tc >= tcmd:
                cmd = c
                tcmd = tc
        future_commands[tcmd] = cmd # Reapply this until a new command is reached
        # ggLog.info(f"cmd = {cmd}")
        # ggLog.info(f"future_commands = {future_commands}")        
        self._commanded_joint_impedances = future_commands # keep commands not applied yet
        self._compute_and_apply_joint_effort(list(cmd.items()))

    @override
    def apply_joint_impedances(self, joint_impedances_pvesd : Dict[Tuple[str,str],Tuple[float,float,float,float,float]]):
        self.setJointsImpedanceCommand(joint_impedances_pvesd=joint_impedances_pvesd)
        self._apply_commanded_joint_impedances()

    def _compute_and_apply_joint_effort(self, joint_impedances_pvesd : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]):

        js = super().getJointsState([jn_cmd[0] for jn_cmd in joint_impedances_pvesd])
        # tcmd = {}
        eff_cmd : Dict[Tuple[str,str],float] = {}
        for jn, cmd in joint_impedances_pvesd:
            pref, vref, tref, pgain, vgain = cmd
            p = js[jn].position[0].cpu().item()
            v = js[jn].rate[0].cpu().item()
            torque = pgain*(pref-p) + vgain*(vref-v) + tref

            max_torque = min(self._max_torque_pos_control, self._max_torques_pos_control.get(jn,float("+inf")))
            if torque>max_torque:
                torque = max_torque
            elif torque<-max_torque:
                torque = -max_torque

            # bodyId, jointId = self._getBodyAndJointId(jn)
            # if bodyId not in tcmd:
            #     tcmd[bodyId] = ([],[])
            # tcmd[bodyId][0].append(jointId)
            # tcmd[bodyId][1].append(torque)
            # self._last_step_commanded_torques.append((jn,torque))
            eff_cmd[jn] = torque

        # for bodyId, command in tcmd.items():
        #     pybullet.setJointMotorControlArray(bodyIndex=bodyId,
        #                                 jointIndices=command[0],
        #                                 controlMode=pybullet.TORQUE_CONTROL,
        #                                 forces=command[1])
        self.setJointsEffortCommand(jointTorques=list(eff_cmd.items()))

    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]] | th.Tensor,
                                        delay_sec : float = 0.0) -> None:
        cmd_time = self.getEnvTimeFromStartup() + delay_sec
        if isinstance(joint_impedances_pvesd, Mapping):
            self._commanded_joint_impedances[cmd_time] = dict(copy.deepcopy(joint_impedances_pvesd))
        elif isinstance(joint_impedances_pvesd, th.Tensor):
            self._commanded_joint_impedances[cmd_time] = dict(zip(self._jimpedance_controlled_joints, joint_impedances_pvesd))
        else:
            raise RuntimeError(f"Unexpected joint_impedances_pvesd type {type(joint_impedances_pvesd)}")