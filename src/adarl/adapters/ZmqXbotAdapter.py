#!/usr/bin/env python3
from __future__ import annotations
import time
from typing import Dict, List, Tuple, Union, Optional, Sequence, Mapping, Literal

import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import JointState, LinkState, RequestFailError, build_1D_vramp_trajectory, MoveFailError, quat_mul_xyzw, th_quat_rotate
from adarl.utils.robot_helpers import Robot
import numpy as np


import torch as th
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from typing_extensions import override
from threading import RLock, Condition
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.StandaloneRealAdapter import StandaloneRealAdapter
from adarl.utils.base_utils import _fix_urdf_package_paths

import pyxbot
from pyxbot.zmq_client import XbotZmqClient, JointsCommand
import numpy as np



# ------------------------------------------------------------------------------------------
# XBOT helper functions
# ------------------------------------------------------------------------------------------


class XbotSafetyError(RuntimeError):
    pass


def detect_simulated():
    # Can I ask xbot?
    return False

class ZmqXbotAdapter(StandaloneRealAdapter, BaseJointImpedanceAdapter, BaseJointPositionAdapter):

    def __init__(self,  model_name : str,
                        stepLength_sec : float,
                        is_floating_base : bool = True,
                        reference_frame : str = "world",
                        torch_device : th.device = th.device("cpu"),
                        fallback_cmd_stiffness : float = 200.0,
                        fallback_cmd_damping : float= 100.0,
                        allow_fallback : bool = True,
                        jpos_cmd_max_vel = {},
                        jpos_cmd_max_vel_default = 0.0,
                        jpos_cmd_max_acc = {},
                        jpos_cmd_max_acc_default = 0.0,
                        enable_filters = True,
                        position_commands_stiffness : float = 100.0,
                        position_commands_damping : float = 10.0,
                        is_simulated : bool | None = False,
                        walltime_factor : float = 1.0,
                        remote_ip : str ='localhost',
                        comm_protocol : Literal["tcp", "ipc"] ='ipc', # or 'ipc'
                        tcp_service_port : int =5557,
                        tcp_state_port : int =5556,
                        tcp_cmd_port : int =5558,
                        ipc_pub_path : str ="/tmp/xbot2_zmq_pub.ipc",
                        ipc_cmd_path : str ="/tmp/xbot2_zmq_cmd.ipc",
                        ipc_service_path : str ="/tmp/xbot2_zmq_rep.ipc",
                        robot_urdf : str | None = None,
                        base_link : str = "base_link",
                        time_source : Literal["wall", "xbot", "sim"] = "wall",
                        sense_timeout_s : float = 0.2,
                        health_check_timeout_s : float = 0.2,
                        health_check_period_s : float = 1.0,
                        health_stale_after_s : float = 5.0):
        super().__init__(stepLength_sec, walltime_factor=walltime_factor)
        self._is_floating_base = is_floating_base
        self._model_name = model_name
        self._reference_frame = reference_frame
        self._torch_device = torch_device
        self._started = False
        self._robot_urdf = robot_urdf
        self._sense_always = True
        self._sense_needed = True
        # self._joint_cmd_fallback_by_jid = {}
        self._fallback_cmd_stiffness = fallback_cmd_stiffness
        self._fallback_cmd_damping = fallback_cmd_damping
        self._allow_fallback = allow_fallback
        
        self._is_safety_triggered = False
        self._position_command_stiffness = position_commands_stiffness
        self._position_command_damping = position_commands_damping
        self._next_commanded_joint_impedances_by_name : dict[tuple[str,str], th.Tensor]= {}
        self._next_commanded_joint_positions : Dict[Tuple[str,str],Tuple[float,float,float]] = {}
        # joint trajectories are ndarrays listing waypoints of format (time, position, velocuty, acceleration)
        self._commanded_joint_trajs_tpva : Dict[Tuple[str,str],np.ndarray]= {}


        self._jpos_cmd_vel_scaling : Dict[Tuple[str,str],float]= {}
        self._jpos_cmd_acc_scaling : Dict[Tuple[str,str],float]= {}
        self._jpos_cmd_vel_scaling_default : float = 1.0
        self._jpos_cmd_acc_scaling_default : float = 1.0
        self._jpos_cmd_max_vel : Dict[Tuple[str,str],float] = jpos_cmd_max_vel
        self._jpos_cmd_max_vel_default : float = jpos_cmd_max_vel_default
        self._jpos_cmd_max_acc : Dict[Tuple[str,str],float] = jpos_cmd_max_acc
        self._jpos_cmd_max_acc_default : float = jpos_cmd_max_acc_default

        self._xbotjname_to_jid : Dict[str, int] # maps joint names to the joint id used by robot_helper
        self._jid_to_xbotjname : Dict[int, str] # maps joint id used by robot_helper to joint names
        self._enable_filters = enable_filters
        self._jimpedance_controlled_joints : list[tuple[str,str]] = [] # The joints that this adapter exposes to its users
        self._is_simulated = is_simulated if is_simulated is not None else detect_simulated()
        self._control_dt = 0.001 # can I get this from somewhere?
        self._base_link = base_link
        self._time_source = "xbot" if time_source == "sim" else time_source
        self._sense_timeout_s = sense_timeout_s
        self._health_check_timeout_s = health_check_timeout_s
        self._health_check_period_s = health_check_period_s
        self._health_stale_after_s = health_stale_after_s
        self._last_successful_sense_wall_time = 0.0
        self._health_cache_until = 0.0
        self._last_health_ok = False
        self._startup_xbot_time = 0.0
        self._reset_xbot_time = 0.0
        self.position_ramp_time = 4.0
        self.impedance_ramp_time = 1.0
        self._position_ramp_tinysleep = 0.01
        self._impedance_ramp_tinysleep = 0.01
        self._base_q_last = np.zeros((1, 4), dtype=np.float64)
        self._base_q_last[:, 0] = 1.0
        self._base_omega_last = np.zeros((1, 3), dtype=np.float64)
        self._base_linacc_last = np.zeros((1, 3), dtype=np.float64)

        self._xbot_zmq_client = XbotZmqClient(  remote_ip = remote_ip,
                                                protocol = comm_protocol,
                                                tcp_service_port = tcp_service_port,
                                                tcp_pub_port = tcp_state_port,
                                                tcp_cmd_port = tcp_cmd_port,
                                                ipc_pub_path = ipc_pub_path,
                                                ipc_cmd_path = ipc_cmd_path,
                                                ipc_service_path = ipc_service_path)

    def is_simulated(self):
        return self._is_simulated

    @override
    def control_period(self):
        return self._control_dt
        
    def _thtens(self, arr: np.ndarray) -> th.Tensor:
        """Convert a numpy array to a torch tensor on the configured device."""
        return th.as_tensor(arr, device=self._torch_device)
    
    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[Tuple[str, str]]):
        self._xbot_joints_to_monitor = jointsToObserve # keep empty the normal jointsToObserve and use this instead
        super().set_monitored_joints(jointsToObserve)

    def startup(self):
        super().startup()

        self._xbot_zmq_client.start()
        detected_joint_names = self._xbot_zmq_client.get_joint_names()
        self._joints_num = len(detected_joint_names)
        self._xbotjname_to_jid = {jname : jid for jid, jname in enumerate(detected_joint_names)}
        self._jid_to_xbotjname = {jid : jname for jname, jid in self._xbotjname_to_jid.items()}
        ggLog.info(f"ZmqXBotAdapter: found joints: {list(self._xbotjname_to_jid.keys())}")
        if self._robot_urdf is None:
            self._robot_urdf = _fix_urdf_package_paths(self._xbot_zmq_client.get_urdf())
        self._robot_helper = Robot(robot_description_string=self._robot_urdf)

        self._jimpedance_controlled_joints_jids = np.array([self._xbotjname_to_jid[jn] for model_name,jn in self._jimpedance_controlled_joints])
        self._started = True
        self._sense_if_needed()
        self._startup_xbot_time = self._last_xbot_time()
        self._reset_xbot_time = self._startup_xbot_time

    def _last_xbot_time(self) -> float:
        return float(getattr(self._xbot_zmq_client, "_last_msg_stamp", 0.0))

    def _check_zmq_plugin(self, timeout_s: float | None = None) -> bool:
        try:
            health = self._xbot_zmq_client.get_health(
                timeout_s=self._health_check_timeout_s if timeout_s is None else timeout_s
            )
        except Exception:
            return False

        self._is_safety_triggered = bool(health.get("safety_triggered", True))
        plugin_running = bool(health.get("zmq_io_state_ok", False)) and health.get("zmq_io_state") == "Running"
        state_age = float(health.get("state_last_publish_age_s", float("inf")))
        state_fresh = 0.0 <= state_age <= self._health_stale_after_s
        return plugin_running and not self._is_safety_triggered and state_fresh

    def _update_safety_status(self, timeout_s: float | None = None) -> bool:
        try:
            status = self._xbot_zmq_client.get_safety_status(
                timeout_s=self._health_check_timeout_s if timeout_s is None else timeout_s
            )
        except Exception:
            return self._is_safety_triggered
        self._is_safety_triggered = bool(status.get("safety_triggered", True))
        return self._is_safety_triggered

    def _raise_if_safety_triggered(self, context: str):
        if self._update_safety_status():
            raise XbotSafetyError(f"XBot2 safety is triggered while {context}; stopping interface cleanly")

    def _mark_successful_sense(self):
        self._last_successful_sense_wall_time = time.monotonic()
        self._last_health_ok = True
        self._health_cache_until = self._last_successful_sense_wall_time + self._health_check_period_s

    def _has_recent_successful_sense(self) -> bool:
        if self._last_successful_sense_wall_time <= 0.0:
            return False
        return time.monotonic() - self._last_successful_sense_wall_time <= self._health_stale_after_s

    def _cached_health_check(self, force: bool = False) -> bool:
        if not self._started:
            return False

        now = time.monotonic()
        if not force and now < self._health_cache_until:
            return self._last_health_ok

        self._last_health_ok = self._check_zmq_plugin()
        self._health_cache_until = now + self._health_check_period_s
        return self._last_health_ok

    @override
    def resetWorld(self):
        super().resetWorld()
        if self._time_source == "xbot" and self._started:
            self._sense_if_needed()
            self._reset_xbot_time = self._last_xbot_time()

    @override
    def getEnvTimeFromStartup(self) -> float:
        if self._time_source == "xbot" and self._started:
            return self._last_xbot_time() - self._startup_xbot_time
        return super().getEnvTimeFromStartup()

    def getEnvTimeFromReset(self) -> float:
        if self._time_source == "xbot" and self._started:
            return self._last_xbot_time() - self._reset_xbot_time
        return self.getEnvTimeFromStartup()

    def is_xbot_running(self):
        if not self._started:
            return False
        if self._has_recent_successful_sense():
            return True
        return self._cached_health_check()

    def is_xbot_control_running(self):
        return self._cached_health_check()

    def fallback_striffness(self):
        return self._fallback_cmd_stiffness

    def fallback_damping(self):
        return self._fallback_cmd_damping

    def get_xbot_controlled_joints(self) -> list[tuple[str,str]]:
        """Get the names of the joint that XBot is controlling

        Returns
        -------
        list[tuple[str,str]]
            The list of the joints
        """
        return [(self._model_name, xbot_jname) for xbot_jname in self._xbotjname_to_jid.keys()]

    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[Tuple[str,str]]):
        self._jimpedance_controlled_joints = list(joint_names)

    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str,str]]:
        return self._jimpedance_controlled_joints


    def getRenderings(self, requestedCameras: Sequence[str]) -> Dict[str, Tuple[th.Tensor | float]]:
        raise NotImplementedError("getRenderings not implemented for ZmqXbotAdapter")
    
    @override
    def getJointsState(self, requestedJoints : Sequence[Tuple[str,str]] | None = None) -> th.Tensor:
        if not self._started:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call set_monitored_joints, and then call startController")

        if requestedJoints is None:
            requestedJoints = self._monitored_joints

        for model, jname in requestedJoints:
            if model != self._model_name:
                raise RuntimeError(f"Requested joint for model different from the monitored one (asked '{model, jname}', but have '{self._model_name}')")
        # jids = [self._xbotjname_to_jid[jname] for model, jname in requestedJoints]
        jnames = [jn[1] for jn in requestedJoints]
        self._sense_if_needed()

        joints_pve = self._xbot_zmq_client.get_joints_state(jnames).pve()

        #TODO: get the delay time somehow
        # obsDelay = float("+inf")
        # while obsDelay > self._maxObsAge:
        #     self.run(0.001)
        #     self._robot_interface.sense(update_model=False)
        #     obsDelay = self._robot_interface.getTime() - self._robot_interface.getTimestampRx()            
        # self._jointStateMsgAgeAvg.addValue(obsDelay)

        return th.as_tensor(joints_pve).view(size=(len(requestedJoints),3)).to(device=self._torch_device, dtype=th.float32)

    @override
    def getLinksState(self, requestedLinks : Sequence[Tuple[str,str]], use_com_pose = False) -> Dict[Tuple[str,str],LinkState]:
        raise RuntimeError("getLinksState not yet implemented for ZmqXbotAdapter") # Could at least be implemented for robot links, relative to the robot

    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]] | th.Tensor,
                                        delay_sec : float = 0) -> None:
        if delay_sec!=0.0:
            raise NotImplementedError("Impedance command delay is not supported")
        
        if isinstance(joint_impedances_pvesd, th.Tensor):
            joint_impedances_pvesd_dict = dict(zip(self._jimpedance_controlled_joints, joint_impedances_pvesd))
        elif isinstance(joint_impedances_pvesd, Mapping):
            joint_impedances_pvesd_dict = joint_impedances_pvesd

        if self.is_safety_triggered():
            raise XbotSafetyError("XBot2 safety is triggered while queueing impedance commands; stopping interface cleanly")
        
        for full_jname, jcmd in joint_impedances_pvesd_dict.items():
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(f"Commanded joint impedance for model different from the controlled one (asked '{model_name, jname}', but have '{self._model_name}')")
            self._next_commanded_joint_impedances_by_name[full_jname] = th.as_tensor(jcmd)

    def _apply_commanded_joint_impedances(self):
        self.apply_joint_impedances(self._next_commanded_joint_impedances_by_name)

    @override
    def apply_joint_impedances(self, joint_impedances_pvesd : Dict[Tuple[str,str], th.Tensor] | th.Tensor):
        # ggLog.info(f"applying joint impedances {joint_impedances_pvesd}")
        if len (joint_impedances_pvesd)==0:
            return
        if self.is_safety_triggered():
            raise XbotSafetyError("XBot2 safety is triggered while applying impedance commands; stopping interface cleanly")
        
        if isinstance(joint_impedances_pvesd, th.Tensor):
            joint_impedances_pvesd_dict = dict(zip(self._jimpedance_controlled_joints, joint_impedances_pvesd))
        elif isinstance(joint_impedances_pvesd, Mapping):
            joint_impedances_pvesd_dict = joint_impedances_pvesd

        commanded_joint_impedances_by_jid : dict[int,np.ndarray] = {}
        for full_jname, jcmd in joint_impedances_pvesd_dict.items():
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(f"Commanded joint impedance for model different from the controleld one (asked '{model_name, jname}', but have '{self._model_name}')")
            jid = self._xbotjname_to_jid[jname]
            commanded_joint_impedances_by_jid[jid] = th.as_tensor(jcmd).detach().cpu().numpy()
        
        prefs, vrefs, erefs, pgains, vgains = (np.zeros(shape=(self._joints_num,), dtype=np.float64) 
                                               for _ in range(5))

        commanded_joint_names = [self._jid_to_xbotjname[jid] for jid in commanded_joint_impedances_by_jid.keys()]
        commanded_pvesd = np.stack([np.array(pvesd) for pvesd in commanded_joint_impedances_by_jid.values()], axis = 0)
        for jid, cmd in commanded_joint_impedances_by_jid.items():
            prefs[jid], vrefs[jid], erefs[jid], pgains[jid], vgains[jid] = cmd
        # curr_pos = self._xbot_zmq_client.getJointPosition()
        # used_fallback = False
        # for jid in range(self._joints_num):
        #     cmd = commanded_joint_impedances_by_jid.get(jid,None)
        #     if cmd is None and self._allow_fallback:
        #         used_fallback = True
        #         ggLog.warn(f"Missing command for joint {self._jid_to_xbotjname[jid]} ({jid}), using fallback.")
        #         raise RuntimeError("Missing command for joint {self._jid_to_xbotjname[jid]} ({jid})")
        #         # keeps current position
        #         cmd = (curr_pos[jid], 0, 0, self._fallback_cmd_stiffness, self._fallback_cmd_damping)
        #     prefs[jid], vrefs[jid], erefs[jid], pgains[jid], vgains[jid] = cmd
        # if used_fallback:
        #     ggLog.warn(f"Used fallback because only had commands for joints_ids:\n {list(commanded_joint_impedances_by_jid.keys())}")
        #     ggLog.warn(f"Which correspond to joint names:\n {[jn for jn,ji in joint_impedances_pvesd_dict.items()]}")
        # ggLog.info(f"Sending commanded joint impedances:\n {commanded_pvesd}")
        self._xbot_zmq_client.send_command(JointsCommand(joint_names = commanded_joint_names,
                                                        pvesd = commanded_pvesd,
                                                        ctrl_mode = np.full(shape=(len(commanded_joint_names),1), fill_value=63, dtype=np.uint32)))

        self._last_sent_pvesd = np.stack([prefs,vrefs,erefs,pgains,vgains], axis = 1)
        # ggLog.info(f"Sent robot_interface command")


    def _apply_commanded_joint_trajectories(self):
        jimp_cmds_pvesd : Dict[Tuple[str,str],Tuple[float,float,float,float,float]] = {}
        t = self.getEnvTimeFromStartup()
        for jn,traj_tpva in self._commanded_joint_trajs_tpva.items():
            sample_idx = np.searchsorted(traj_tpva[:,0], t) # index of the next trajectory sample (the first with time higher of t)
            if sample_idx>=traj_tpva.shape[0]:
                sample_idx = traj_tpva.shape[0]-1
            time, pos, vel, acc = traj_tpva[sample_idx]
            jimp_cmds_pvesd[jn] = (pos, vel, 0, self._position_command_stiffness, self._position_command_damping)
        self.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_cmds_pvesd)

    def _apply_commanded_joint_positions(self):
        if len(self._next_commanded_joint_positions)>0:
            jimp_pvesd_cmds : Dict[Tuple[str,str],Tuple[float,float,float,float,float]] = {}
            joints = list(self._next_commanded_joint_positions.keys())
            js_pve : th.Tensor = self.getJointsState(joints)
            js_dict = joints_state_dict = {jn : pve for jn, pve in zip(joints, js_pve)}
            for jn,p_ref_velsc_accsc in self._next_commanded_joint_positions.items():
                # just to compute the duration
                p_ref, velocity_scaling, acceleration_scaling = p_ref_velsc_accsc
                traj_tpva = build_1D_vramp_trajectory(t0 = 0.0,
                                                    p0 = js_dict[jn][0].item(),
                                                    v0 = js_dict[jn][1].item(),
                                                    pf = p_ref,
                                                    ctrl_freq_hz = 10, # we just use the first sample anyway
                                                    max_vel=self._jpos_cmd_max_vel.get(jn, self._jpos_cmd_max_vel_default)*velocity_scaling,
                                                    max_acc=self._jpos_cmd_max_acc.get(jn, self._jpos_cmd_max_acc_default)*acceleration_scaling)
                t, pos, vel, a = traj_tpva[0]
                jimp_pvesd_cmds[jn] = (pos, vel, 0.0, self._position_command_stiffness, self._position_command_damping)
            self.setJointsImpedanceCommand(jimp_pvesd_cmds)

    def clear_commands(self):
        self._next_commanded_joint_impedances_by_name = {}
        self._next_commanded_joint_positions = {}
        self._commanded_joint_trajs_tpva = {}

    def _apply_controls(self):
        self._apply_commanded_joint_trajectories() # trajectories override positions by setting position commands
        self._apply_commanded_joint_positions() # positions override impedances by setting impedance commands
        self._apply_commanded_joint_impedances() 

    @override
    def run(self, duration_sec: float):
        # ggLog.info(f"XbotAdapter.run()")
        self._apply_controls()
        super().run(duration_sec)

    @override
    def initialize_for_step(self):
        pass

    def _sense_if_needed(self):
        if self._sense_needed or self._sense_always:
            try:
                self._xbot_zmq_client.sense(timeout_s=self._sense_timeout_s)
            except TimeoutError:
                self._last_health_ok = False
                self._health_cache_until = 0.0
                self._raise_if_safety_triggered("waiting for the XBot2/ZMQ state stream")
                raise
            except Exception:
                self._last_health_ok = False
                self._health_cache_until = 0.0
                raise
            self._mark_successful_sense()
            self._sense_needed = False
    @override
    def step(self) -> float:
        step_duration = super().step()
        self.clear_commands()
        self._sense_needed = True

        # model = self._robot_interface.model()
        # model.setJointPosition(self._robot_interface.getJointPosition())
        # model.setJointVelocity(self._robot_interface.getJointVelocity())
        # model.setJointAcceleration(self._robot_interface.getJointAcceleration())
        # model.update()
        # ggLog.info(f"q = {model.getJointPosition()}")
        # ggLog.info(f"v = {model.getJointVelocity()}")
        # ggLog.info(f"a = {model.getJointAcceleration()}")
        # ggLog.info(f"tau = {model.computeInverseDynamics()}")


        return step_duration

    @override
    def initialize_for_episode(self):
        super().initialize_for_episode()
        self.clear_commands()

    @override
    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float],
                                        velocity_scaling : Optional[float] = None,
                                        acceleration_scaling : Optional[float] = None) -> None:
        if velocity_scaling is None:
            velocity_scaling = 1.0
        if acceleration_scaling is None:
            acceleration_scaling = 1.0
        self._next_commanded_joint_positions.update({k:(p,velocity_scaling,acceleration_scaling) for k,p in jointPositions.items()})

        # jic_pvesd = [(jn,(pos, 0.0, 0.0, self._position_command_stiffness, self._position_command_damping)) for jn, pos in jointPositions.items()]
        # self.setJointsImpedanceCommand(joint_impedances_pvesd=jic_pvesd)

    def _setJointTrajectoryCommand(self, jointTrajectories_tpva : Dict[Tuple[str,str], np.ndarray]):
        self._commanded_joint_trajs_tpva.update(jointTrajectories_tpva)

    @override
    def moveToJointPoseSync(self,   jointPositions : Dict[Tuple[str,str],float],
                                    velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None,
                                    joint_position_tolerance : float = 0.01,
                                    max_time_s : float = 60,
                                    joint_velocity_termination_threshold = 0.01,
                                    joint_velocity_scaling : dict[Tuple[str,str],float] = {}) -> None:
        # ggLog.info(f"moveToJointPoseSync called with jointPositions={jointPositions}, velocity_scaling={velocity_scaling}, acceleration_scaling={acceleration_scaling}, joint_position_tolerance={joint_position_tolerance}, max_time_s={max_time_s}, joint_velocity_termination_threshold={joint_velocity_termination_threshold}, joint_velocity_scaling={joint_velocity_scaling}")
        self.clear_commands()
        if velocity_scaling is None:
            velocity_scaling = 1.0
        if acceleration_scaling is None:
            acceleration_scaling = 1.0
        joints = list(jointPositions.keys())
        js_pve = self.getJointsState(joints)
        js_dict = {jn : pve for jn, pve in zip(joints, js_pve)}
        last_refs = self.get_current_joint_impedance_command()
        last_refs_dict = {self._jimpedance_controlled_joints[i]:last_refs[i] for i in range(len(self._jimpedance_controlled_joints))}
        max_traj_duration = 0
        joint_trajs = {}
        for jn,p_ref in jointPositions.items():
            # just to compute the duration
            vs = joint_velocity_scaling.get(jn, velocity_scaling)
            traj_tpva = build_1D_vramp_trajectory(  t0 = 0.0,
                                                    p0 = last_refs_dict[jn][0].item(),
                                                    v0 = js_dict[jn][1].item(),
                                                    pf = p_ref,
                                                    ctrl_freq_hz = 1000.0,
                                                    max_vel=self._jpos_cmd_max_vel.get(jn, self._jpos_cmd_max_vel_default)*vs,
                                                    max_acc=self._jpos_cmd_max_acc.get(jn, self._jpos_cmd_max_acc_default)*acceleration_scaling)
            joint_trajs[jn] = traj_tpva
            traj_duration = traj_tpva[-1][0]
            max_traj_duration = max(max_traj_duration,traj_duration)
        for jn,p_ref in jointPositions.items(): # scale to have all trajectories be the same duration
            joint_traj = joint_trajs[jn]
            traj_duration = joint_traj[-1][0]
            scale = max_traj_duration/traj_duration
            joint_traj[:,0] *= scale # time
            joint_traj[:,2] *= 1/scale # velocity
            joint_traj[:,3] *= 1/(scale**2) # acceleration
        # for jn,jt in joint_trajs.items():
        #     with np.printoptions(threshold=np.inf):
        #         ggLog.info(f"traj_tpva {jn} = \n{jt}")
        timeout_env = max_traj_duration*2+1
        timeout_wall = timeout_env*20
        if max_traj_duration > max_time_s:
            raise RuntimeError(f"Computed trajectory is excessively long, would last {max_traj_duration}s, max_time is set to {max_time_s}s. \n"
                               f"Joint names           : {[jn for jn,ji in js_dict.items()]}\n"
                               f"Initial joint position: "+str([f"{jpve[0].item(): 2.4f}" for jn,jpve in js_dict.items()])+"\n"
                               f"Initial joint velocity: "+str([f"{jpve[1].item(): 2.4f}" for jn,jpve in js_dict.items()])+"\n"
                               f"Target  joint position: "+str([f"{jp: 2.4f}" for jp in jointPositions.values()])+"\n"
                               f"Durations {[(jn,traj_tpva[-1][0]) for jn,traj_tpva in joint_trajs.items()]}\n"
                               f"Raise the max_time if it is actually ok.")
        t0 = self.getEnvTimeFromStartup()
        for jn in joint_trajs.keys():
            joint_trajs[jn][:,0] += t0
        self._setJointTrajectoryCommand(jointTrajectories_tpva = joint_trajs)

        # self.setJointsPositionCommand(jointPositions=jointPositions)
        t0_env = self.getEnvTimeFromStartup()
        t0_wall = time.monotonic()
        js = self.getJointsState(list(jointPositions.keys()))
        errors = [jpve[0].item() - jointPositions[jn] for jn,jpve in js_dict.items()]
        reached_position = all([abs(e) < joint_position_tolerance for e in errors])
        elapsed_env_time = 0.0
        elapsed_wall_time = 0.0
        stopped = False
        while not (reached_position or (stopped and elapsed_env_time>=max_traj_duration)):
            self.run(self._stepLength_sec)
            js = self.getJointsState(list(jointPositions.keys()))
            js_dict = {jn : pve for jn, pve in zip(joints, js)}
            errors = [jpve[0].item() - jointPositions[jn] for jn,jpve in js_dict.items()]
            reached_position = all([abs(e) < joint_position_tolerance for e in errors])
            stopped = all([abs(jpve[1].item())<joint_velocity_termination_threshold for jpve in js_dict.values()])
            elapsed_env_time = self.getEnvTimeFromStartup() - t0_env
            elapsed_wall_time = time.monotonic() - t0_wall
            if elapsed_env_time > timeout_env:
                self.clear_commands()
                raise MoveFailError(f"Timed out waiting for sync joint move (env timeout {elapsed_env_time}>{timeout_env})\n"
                                    f"    target = {jointPositions}\n"
                                    f"    joint state = {[jpve[0].item() for jn,jpve in js_dict.items()]}\n"
                                    f"    errors = {errors}\n"
                                    f"    max_error = {max(errors)}\n"
                                    f"    tolerance = {joint_position_tolerance}")
            if elapsed_wall_time > timeout_wall:
                self.clear_commands()
                raise MoveFailError(f"Timed out waiting for sync joint move (wall timeout {elapsed_wall_time}>{timeout_wall}) errors = {errors} tolerance = {joint_position_tolerance}")



    def is_safety_triggered(self):
        return self._is_safety_triggered
    
    def _get_current_refs_pvesd(self) -> np.ndarray:
        self._sense_if_needed()
        js : pyxbot.zmq_client.JointState = self._xbot_zmq_client.get_joints_state([jn[1] for jn in self._jimpedance_controlled_joints])
        pvesd = js.pvesd_refs()
        return pvesd

    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        ref_j_pvesd = self._get_current_refs_pvesd()
        # pvesd_by_name = {(mn,jn):ref_j_pvesd[self._xbotjname_to_jid[jn]] for mn,jn in self._jimpedance_controlled_joints}
        return th.as_tensor(ref_j_pvesd, device=self._torch_device, dtype=th.float32)

    @override
    def apply_joint_ref_with_ramp(self, joint_impedances_pvesd: th.Tensor, ramp_time=None, tolerance=-1.0):
        if not isinstance(joint_impedances_pvesd, th.Tensor):
            raise TypeError("joint_impedances_pvesd must be a torch.Tensor")
        if ramp_time is None:
            ramp_time = self.position_ramp_time

        curr_pvesd = self.get_current_joint_impedance_command()
        curr_pvesd[:, 1] = 0.0
        curr_pvesd[:, 2] = 0.0
        target_pos_ref = joint_impedances_pvesd[:, 0].clone()
        start_pos_ref = curr_pvesd[:, 0].clone()

        if tolerance >= 0.0 and not (th.abs(target_pos_ref - curr_pvesd[:, 0]) > tolerance).any().item():
            return

        start_time = self.getEnvTimeFromStartup()
        while True:
            elapsed = self.getEnvTimeFromStartup() - start_time
            frac = min(1.0, max(0.0, elapsed / ramp_time))
            curr_pvesd[:, 0] = start_pos_ref + frac * (target_pos_ref - start_pos_ref)
            self.setJointsImpedanceCommand(curr_pvesd)
            self.run(self._position_ramp_tinysleep)
            self._sense_needed = True
            self._sense_if_needed()
            if frac >= 1.0:
                break

    @override
    def apply_joint_impedances_with_ramp(self, joint_impedances_pvesd: th.Tensor, ramp_time=None, tolerance=-1.0):
        if not isinstance(joint_impedances_pvesd, th.Tensor):
            raise TypeError("joint_impedances_pvesd must be a torch.Tensor")
        if ramp_time is None:
            ramp_time = self.impedance_ramp_time

        curr_pvesd = self.get_current_joint_impedance_command()
        curr_pvesd[:, 1] = 0.0
        curr_pvesd[:, 2] = 0.0
        target_stiffness = joint_impedances_pvesd[:, 3].clone()
        start_stiffness = curr_pvesd[:, 3].clone()
        target_damping = joint_impedances_pvesd[:, 4].clone()
        start_damping = curr_pvesd[:, 4].clone()

        if tolerance >= 0.0:
            stiffness_ok = th.abs(target_stiffness - curr_pvesd[:, 3]) <= tolerance
            damping_ok = th.abs(target_damping - curr_pvesd[:, 4]) <= tolerance
            if (stiffness_ok & damping_ok).all().item():
                return

        start_time = self.getEnvTimeFromStartup()
        while True:
            elapsed = self.getEnvTimeFromStartup() - start_time
            frac = min(1.0, max(0.0, elapsed / ramp_time))
            curr_pvesd[:, 3] = start_stiffness + frac * (target_stiffness - start_stiffness)
            curr_pvesd[:, 4] = start_damping + frac * (target_damping - start_damping)
            self.setJointsImpedanceCommand(curr_pvesd)
            self.run(self._impedance_ramp_tinysleep)
            self._sense_needed = True
            self._sense_if_needed()
            if frac >= 1.0:
                break

    def set_filters(self, set_enabled: bool, profile_name="safe"):
        profile_cutoff_hz = {
            "safe": 5.0,
            "medium": 15.0,
            "fast": 25.0,
        }
        cutoff_hz = float(profile_cutoff_hz.get(profile_name, 0.0)) if set_enabled else 0.0
        self._xbot_zmq_client.set_filter_frequency_hz(cutoff_hz, enabled=cutoff_hz > 0.0 and self._enable_filters)

    def read_imu_data(self):
        self._sense_if_needed()
        imu_name = self._xbot_zmq_client.get_imu_names()[0]
        q_xyzw = self._xbot_zmq_client.getImuOrientation([imu_name])[0]
        self._base_q_last[:, :] = np.array([[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]], dtype=np.float64)
        self._base_omega_last[:, :] = self._xbot_zmq_client.getImuAngularVelocity([imu_name]).reshape(1, 3)
        self._base_linacc_last[:, :] = self._xbot_zmq_client.getImuLinearAcceleration([imu_name]).reshape(1, 3)

    def get_base_link_state(self):
        return self._base_link, self._base_q_last, self._base_omega_last, self._base_linacc_last
    
    def _get_imus_for_links(self, requestedLinks : Sequence[tuple[str,str]]) -> dict[str, str]:
        imus = self._xbot_zmq_client.get_imu_names()
        # print(f"imus = {imus}")
        req_links = [ln[1] for ln in requestedLinks] # Remove the model name
        ref_imus : dict[str,str] = {} # What imu to use for which links
        for rl in req_links:
            # Assumes imu names and link names are the same
            ref_imus[rl] = rl if rl in imus else list(imus.keys())[0] # use the first available imu (maybe we can do better than this? find a "best" one?)
        return ref_imus

    @override
    def get_link_gravity_direction(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        self._sense_if_needed()
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        # ref_imus = self._get_imus_for_links(requestedLinks)
        # needed_imus = list(set(ref_imus.values()))
        # quats = self._xbot_zmq_client.getImuOrientation(needed_imus)
        # poses_imu2world = {imu_name:quat for imu_name,quat in zip(needed_imus, quats)}

        # for model_name,link_name in requestedLinks:
        #     pose_link2imu = self._robot_helper.get_frame_poses_xyzxyzw(frames=[link_name], reference_frame=ref_imus[link_name])
        #     quat_link2imu  = pose_link2imu[link_name][3:7]
        
        linksnum = len(requestedLinks)
        requested_linknames = [ln[1] for ln in requestedLinks] # Remove the model name
        imu_name = self._xbot_zmq_client.get_imu_names()[0] # use imu 0 for all links
        poses_link2imu = self._robot_helper.get_frame_poses_xyzxyzw(frames=requested_linknames, reference_frame=imu_name)
        quats_link2imu = th.stack([th.as_tensor(poses_link2imu[ln][3:7]) for ln in requested_linknames])
        
        quat_imu2world = th.as_tensor(self._xbot_zmq_client.getImuOrientation([imu_name])[0])
        quats_link2world = quat_mul_xyzw(quat_imu2world.expand(linksnum, 4), quats_link2imu)
        gdirs = th_quat_rotate(th.as_tensor([0,0,-1.0]).expand(linksnum, 3), quats_link2world.to(dtype=th.float32))
        return gdirs
    
    @override
    def get_link_relative_angular_velocity(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        linksnum = len(requestedLinks)
        requested_linknames = [ln[1] for ln in requestedLinks] # Remove the model name
        imu_name = self._xbot_zmq_client.get_imu_names()[0] # use imu 0 for all links
        poses_link2imu = self._robot_helper.get_frame_poses_xyzxyzw(frames=requested_linknames, reference_frame=imu_name)
        quats_link2imu = th.stack([th.as_tensor(poses_link2imu[ln][3:7]) for ln in requested_linknames])
        
        imulocal_angvel = th.as_tensor(self._xbot_zmq_client.getImuAngularVelocity([imu_name])[0])
        linklocal_angvel = th_quat_rotate(imulocal_angvel.expand(linksnum, 3).to(dtype=th.float32), quats_link2imu.to(dtype=th.float32))
        return linklocal_angvel
    
    
    def get_local_link_linear_acceleration(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        raise NotImplementedError("get_local_link_linear_acceleration not yet implemented for ZmqXbotAdapter")
        imus = self._robot_interface.getImu()
        # print(f"imus = {imus}")
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        req_links = [ln[1] for ln in requestedLinks] # Remove the model name
        ref_imus : dict[str,str] = {} # What imu to use for which links
        for rl in req_links:
            ref_imus[rl] = rl if rl in imus else list(imus.keys())[0] # use the first available imu (maybe we can do better than this? find a "best" one?)
        imu2link_poses : dict[str,Affine3] = {ln:self._robot_interface.model().getPose(ln,ref_imus[ln]) for ln in req_links}
        accelerations : list[th.Tensor] = []
        for ln in req_links:
            imu2link = imu2link_poses[ln]
            imu2link_rotmat = imu2link.matrix()[:3,:3]
            imu = imus[ref_imus[ln]]
            imu_linacc = imu.getLinearAcceleration()
            if np.allclose(imu2link_rotmat, np.eye(imu2link_rotmat.shape[0]), atol=1e-5):
                acceleration = imu_linacc
            else:
                raise RuntimeError(f"Cannot compute local linear acceleration for link '{ln}' as it is not directly attached to an IMU")
                com_offset_xyz = imu2link.translation()
                imu_angvel = imu.getAngularVelocity()
                imu_angacc # Would need this somehow
                imu_linvel # Would need this somehow
                local_angvel = imu2link_rotmat @ imu_linacc
                local_linvel = imu2link_rotmat @ (imu_linvel - np.cross(com_offset_xyz, imu_angvel))
                acc = imu2link_rotmat @ (imu_linacc - np.cross(com_offset_xyz, imu_angacc))
                correction = np.cross(local_angvel, local_linvel)
                acceeleration = acc + correction
            accelerations.append(self._thtens(acceleration))
        return th.stack(accelerations)
    
    @override
    def set_reference_filter(self, reference_filter_cutoff_frequency : th.Tensor):
        self._xbot_zmq_client.set_filter_frequency_hz(reference_filter_cutoff_frequency.item(),
                                                      enabled=reference_filter_cutoff_frequency.item()>0.0 and self._enable_filters)
