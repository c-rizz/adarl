#!/usr/bin/env python3
from __future__ import annotations

import atexit
import time
import os
import signal
import shlex
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch as th
import yaml
import zmq
from typing_extensions import override

import adarl.utils.dbg.ggLog as ggLog
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.ZmqXbotAdapter import ZmqXbotAdapter
from adarl.utils.utils import JointState, LinkState, Pose, build_pose
from xbot2_mujoco.PyXbotMjSim import XBotMjSim


def _yaml_param_value(params: dict, name: str, default: Any) -> Any:
    value = params.get(name, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _load_zmq_params(config_path: str | None) -> dict[str, Any]:
    params = {
        "comm_protocol": "tcp",
        "remote_ip": "localhost",
        "tcp_service_port": 5557,
        "tcp_state_port": 5559,
        "tcp_cmd_port": 5558,
        "ipc_pub_path": "/tmp/xbot2_zmq_pub.ipc",
        "ipc_cmd_path": "/tmp/xbot2_zmq_cmd.ipc",
        "ipc_service_path": "/tmp/xbot2_zmq_rep.ipc",
    }
    if config_path is None:
        return params

    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except OSError:
        return params

    plugin_params = (
        config.get("xbotcore_plugins", {})
        .get("zmq_io", {})
        .get("params", {})
    )
    params["comm_protocol"] = _yaml_param_value(plugin_params, "protocol", params["comm_protocol"])
    params["tcp_service_port"] = int(_yaml_param_value(plugin_params, "tcp_service_port", params["tcp_service_port"]))
    params["tcp_state_port"] = int(_yaml_param_value(plugin_params, "tcp_state_port", params["tcp_state_port"]))
    params["tcp_cmd_port"] = int(_yaml_param_value(plugin_params, "tcp_cmd_port", params["tcp_cmd_port"]))
    params["ipc_pub_path"] = _yaml_param_value(
        plugin_params, "ipc_state_path", _yaml_param_value(plugin_params, "ipc_pub_path", params["ipc_pub_path"])
    )
    params["ipc_cmd_path"] = _yaml_param_value(plugin_params, "ipc_cmd_path", params["ipc_cmd_path"])
    params["ipc_service_path"] = _yaml_param_value(
        plugin_params, "ipc_service_path", _yaml_param_value(plugin_params, "ipc_rep_path", params["ipc_service_path"])
    )
    return params


class XbotMjAdapter(ZmqXbotAdapter, BaseSimulationAdapter):
    """XBot-MuJoCo simulation adapter using the xbot2_zmq plugin for XBot I/O."""

    def __init__(
        self,
        model_fpath: str,
        model_name: str,
        stepLength_sec: float,
        xbot2_config_path: str | None = None,
        headless: bool = False,
        init_steps: int = 0,
        timeout_ms: int = 30000,
        forced_ros_master_uri: str | None = None,
        maxObsDelay=float("+inf"),
        blocking_observation=False,
        is_floating_base: bool = True,
        reference_frame: str = "world",
        torch_device: th.device = th.device("cpu"),
        fallback_cmd_stiffness: float = 200.0,
        fallback_cmd_damping: float = 100.0,
        allow_fallback: bool = True,
        jpos_cmd_max_vel={},
        jpos_cmd_max_vel_default=0.0,
        jpos_cmd_max_acc={},
        jpos_cmd_max_acc_default=0.0,
        enable_filters=True,
        base_link: str = "base_link",
        render_to_file: bool = False,
        render_fps: float = 60,
    ):
        del forced_ros_master_uri, maxObsDelay, blocking_observation

        self._model_fpath = model_fpath
        self._xbot2_config_path = xbot2_config_path
        self._headless = headless
        self._init_steps = init_steps
        self._timeout_ms = timeout_ms
        self._base_link = base_link
        self._render_to_file = render_to_file
        self._render_fps = render_fps
        self._closed = False
        self._xmj_sim = None
        self._xbot2_core_proc = None
        self._xbot2_core_log = None
        self._xbot2_core_log_thread = None
        self._skip_next_reset_world = False
        self._sim_time = 0.0
        self._base_q_last = np.zeros((1, 4), dtype=np.float64)
        self._base_q_last[:, 0] = 1.0
        self._base_omega_last = np.zeros((1, 3), dtype=np.float64)
        self._base_linacc_last = np.zeros((1, 3), dtype=np.float64)
        self.position_ramp_time = 4.0
        self.impedance_ramp_time = 1.0
        self._position_ramp_tinysleep = 0.01
        self._impedance_ramp_tinysleep = 0.01

        self._init_simulation(base_link=base_link)
        atexit.register(self._close)
        if stepLength_sec != self._xmj_sim.physics_dt:
            raise ValueError(f"stepLength_sec {stepLength_sec} is not equal to physics dt {self._xmj_sim.physics_dt}")

        zmq_params = _load_zmq_params(xbot2_config_path)
        super().__init__(
            model_name=model_name,
            stepLength_sec=stepLength_sec,
            is_floating_base=is_floating_base,
            reference_frame=reference_frame,
            torch_device=torch_device,
            fallback_cmd_stiffness=fallback_cmd_stiffness,
            fallback_cmd_damping=fallback_cmd_damping,
            allow_fallback=allow_fallback,
            jpos_cmd_max_vel=jpos_cmd_max_vel,
            jpos_cmd_max_vel_default=jpos_cmd_max_vel_default,
            jpos_cmd_max_acc=jpos_cmd_max_acc,
            jpos_cmd_max_acc_default=jpos_cmd_max_acc_default,
            enable_filters=enable_filters,
            is_simulated=True,
            **zmq_params,
        )

        joints_to_observe = [(model_name, joint) for joint in self._xmj_sim_jnt_names]
        self.set_monitored_joints(joints_to_observe)
        self.set_impedance_controlled_joints(joints_to_observe)
        self.set_monitored_links([])

    def __del__(self):
        self._close()

    def _close(self):
        if not self._closed:
            self._stop_xbot2_core()
            if self._xmj_sim is not None:
                self._xmj_sim.close()
            self._closed = True

    def close(self):
        self._close()

    def _init_simulation(self, base_link: str):
        self._xmj_sim = XBotMjSim(
            model_fname=self._model_fpath,
            xbot2_config_path=self._xbot2_config_path,
            headless=self._headless,
            manual_stepping=True,
            init_steps=max(self._init_steps, 100),
            timeout=self._timeout_ms,
            base_link_name=base_link,
            match_rt_factor=True,
            rt_factor_trgt=1.0,
            render_to_file=self._render_to_file,
            custom_camera_name="custom_camera",
            render_base_path="/tmp",
            render_fps=self._render_fps,
        )

        if not self._xmj_sim.reset():
            raise RuntimeError("Failed to reset XBot-MuJoCo simulation")

        pi = np.zeros((3,), dtype=np.float64)
        qi = np.zeros((4,), dtype=np.float64)
        qi[0] = 1.0
        pi[2] = self._xmj_sim.p[2]
        self._xmj_sim.set_pi(pi)
        self._xmj_sim.set_qi(qi)

        for _ in range(self._init_steps):
            if not self._xmj_sim.step():
                raise RuntimeError("Failed to warm-start XBot-MuJoCo simulation")

        pi[2] = self._xmj_sim.p[2]
        self._xmj_sim.set_pi(pi)
        if not self._xmj_sim.reset():
            raise RuntimeError("Failed to reset XBot-MuJoCo simulation after warm-start")

        self._xmj_sim_jnt_names = self._xmj_sim.jnt_names()
        self._xmj_sim_n_dofs = self._xmj_sim.n_jnts()

    def _start_xbot2_core(self):
        if self._xbot2_config_path is None:
            raise RuntimeError("xbot2_config_path is required to start xbot2-core for XBot-MuJoCo ZMQ")
        if self._xbot2_core_proc is not None and self._xbot2_core_proc.poll() is None:
            return

        log_path = os.path.join(tempfile.gettempdir(), f"xbot2_core_zmq_{os.getpid()}.log")
        self._xbot2_core_log = open(log_path, "w", encoding="utf-8")
        ros_setup = f"/opt/ros/{os.environ.get('ROS_DISTRO', 'jazzy')}/setup.bash"
        xbot_setup = "/opt/xbot/setup.sh"
        setup_cmds = []
        if os.path.isfile(ros_setup):
            setup_cmds.append(f"source {shlex.quote(ros_setup)}")
        if os.path.isfile(xbot_setup):
            setup_cmds.append(f"source {shlex.quote(xbot_setup)}")
        xbot2_cmd = f"xbot2-core -S -C {shlex.quote(self._xbot2_config_path)}"
        setup_cmds.append(f"if command -v stdbuf >/dev/null 2>&1; then exec stdbuf -oL -eL {xbot2_cmd}; else exec {xbot2_cmd}; fi")
        cmd = ["bash", "-lc", " && ".join(setup_cmds)]
        ggLog.info(f"Starting xbot2-core with simulation time: xbot2-core -S -C {self._xbot2_config_path}")
        ggLog.info(f"xbot2-core log: {log_path}")
        self._xbot2_core_proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(self._xbot2_config_path)) or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._xbot2_core_log_thread = threading.Thread(
            target=self._forward_xbot2_core_output,
            args=(self._xbot2_core_proc,),
            name="xbot2-core-log-forwarder",
            daemon=True,
        )
        self._xbot2_core_log_thread.start()

    def _forward_xbot2_core_output(self, proc: subprocess.Popen):
        if proc.stdout is None:
            return

        for line in proc.stdout:
            if self._xbot2_core_log is not None:
                self._xbot2_core_log.write(line)
                self._xbot2_core_log.flush()
            sys.stdout.write(f"[xbot2-core] {line}")
            if not line.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

    def _stop_xbot2_core(self):
        proc = self._xbot2_core_proc
        self._xbot2_core_proc = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5.0)
            except ProcessLookupError:
                pass
        if self._xbot2_core_log_thread is not None:
            self._xbot2_core_log_thread.join(timeout=1.0)
            self._xbot2_core_log_thread = None
        if self._xbot2_core_log is not None:
            self._xbot2_core_log.close()
            self._xbot2_core_log = None

    def _xbot2_core_log_tail(self, n_chars: int = 4000) -> str:
        if self._xbot2_core_log is None:
            return ""
        self._xbot2_core_log.flush()
        try:
            with open(self._xbot2_core_log.name, "r", encoding="utf-8", errors="replace") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - n_chars), os.SEEK_SET)
                return stream.read()
        except OSError:
            return ""

    def _raise_for_xbot2_zmq_log_errors(self):
        tail = self._xbot2_core_log_tail()
        if "zmq_io" in tail and ("Address already in use" in tail or "initialization failed" in tail):
            raise RuntimeError(f"xbot2_zmq plugin failed during startup\n{tail}")

    def _wait_for_zmq_plugin(self, timeout_s: float = 60.0):
        params = _load_zmq_params(self._xbot2_config_path)
        if params["comm_protocol"] == "tcp":
            request_url = f"tcp://{params['remote_ip']}:{params['tcp_service_port']}"
        else:
            request_url = f"ipc://{params['ipc_service_path']}"

        context = zmq.Context.instance()
        t0 = time.monotonic()
        last_error = None
        while time.monotonic() - t0 < timeout_s:
            if self._xbot2_core_proc is not None and self._xbot2_core_proc.poll() is not None:
                raise RuntimeError(
                    f"xbot2-core exited early with code {self._xbot2_core_proc.returncode}\n"
                    f"{self._xbot2_core_log_tail()}"
                )
            self._raise_for_xbot2_zmq_log_errors()

            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 500)
            socket.setsockopt(zmq.SNDTIMEO, 500)
            try:
                socket.connect(request_url)
                socket.send_string(yaml.dump({"type": "joint_names"}))
                response = yaml.safe_load(socket.recv_string()) or {}
                if response.get("success", True) and "data" in response:
                    return
                last_error = response.get("message", response)
            except (zmq.Again, zmq.ZMQError, yaml.YAMLError) as exc:
                last_error = exc
            finally:
                socket.close()
            if self._xmj_sim.is_running() and not self._xmj_sim.step():
                raise RuntimeError("Failed to step XBot-MuJoCo while waiting for xbot2_zmq plugin")
            time.sleep(0.1)

        raise TimeoutError(
            f"Timed out waiting for xbot2_zmq plugin on {request_url}. Last error: {last_error}\n"
            f"{self._xbot2_core_log_tail()}"
        )

    def _sense_with_sim_stepping(self, timeout_s: float = 60.0):
        t0 = time.monotonic()
        last_error = None
        while time.monotonic() - t0 < timeout_s:
            if self._xbot2_core_proc is not None and self._xbot2_core_proc.poll() is not None:
                raise RuntimeError(
                    f"xbot2-core exited early with code {self._xbot2_core_proc.returncode}\n"
                    f"{self._xbot2_core_log_tail()}"
                )
            self._raise_for_xbot2_zmq_log_errors()

            try:
                self._xbot_zmq_client.sense(timeout_s=0.001)
                return
            except TimeoutError as exc:
                last_error = exc

            if self._xmj_sim.is_running():
                if not self._xmj_sim.step():
                    raise RuntimeError("Failed to step XBot-MuJoCo while waiting for xbot2_zmq state")
                self._sim_time += self._xmj_sim.physics_dt
            else:
                time.sleep(0.001)

        raise TimeoutError(
            f"Timeout while waiting for xbot2_zmq state after {timeout_s} seconds. Last error: {last_error}\n"
            f"{self._xbot2_core_log_tail()}"
        )

    def _hold_current_joint_references(self):
        if not self._jimpedance_controlled_joints:
            return

        self._sense_with_sim_stepping(timeout_s=10.0)
        joint_names = [joint_name for _, joint_name in self._jimpedance_controlled_joints]
        pve = np.asarray(self._xbot_zmq_client.get_joints_state(joint_names).pve(), dtype=np.float64)
        pve = pve.reshape(len(joint_names), 3)
        pvesd = np.zeros((len(joint_names), 5), dtype=np.float64)
        pvesd[:, 0] = pve[:, 0]
        pvesd[:, 3] = self._fallback_cmd_stiffness
        pvesd[:, 4] = self._fallback_cmd_damping
        self.apply_joint_impedances(th.as_tensor(pvesd, dtype=th.float32))

        if self._xmj_sim.is_running():
            if not self._xmj_sim.step():
                raise RuntimeError("Failed to step XBot-MuJoCo while initializing joint references")
            self._sim_time += self._xmj_sim.physics_dt
        self._sense_with_sim_stepping(timeout_s=10.0)

    @override
    def startup(self, urdf: str = None, srdf: str = None):
        del srdf
        try:
            if urdf is not None:
                self._robot_urdf = urdf
            if not self._xmj_sim.reset():
                raise RuntimeError("Failed to reset XBot-MuJoCo simulation during startup")
            self._start_xbot2_core()
            self._wait_for_zmq_plugin()
            ZmqXbotAdapter.startup(self)
            self._sense_with_sim_stepping(timeout_s=60.0)
            self._hold_current_joint_references()
            self._skip_next_reset_world = True
            self._sim_time = 0.0
        except Exception:
            self._stop_xbot2_core()
            raise

    def sim_is_running(self):
        return self._xmj_sim.is_running()

    def is_xbot_running(self):
        xbot2_core_running = self._xbot2_core_proc is not None and self._xbot2_core_proc.poll() is None
        return xbot2_core_running and ZmqXbotAdapter.is_xbot_running(self)

    def is_xbot_control_running(self):
        return self.is_xbot_running()

    def xmj_env(self):
        return self._xmj_sim

    def jnt_names(self):
        return list(self._xmj_sim_jnt_names)

    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._sim_time

    def getEnvTimeFromReset(self) -> float:
        return self._sim_time

    @override
    def run(self, duration_sec: float):
        self._apply_controls()
        self.step_sim_for(duration_sec=duration_sec)

    def step_sim_for(self, duration_sec: float):
        sim_dt = self._xmj_sim.physics_dt
        n_sim_steps_to_do = round(duration_sec / sim_dt)
        for _ in range(n_sim_steps_to_do):
            if not self._xmj_sim.step():
                raise RuntimeError("Failed to step XBot-MuJoCo simulation")
            self._sim_time += sim_dt

    @override
    def step(self) -> float:
        stime_before = self._sim_time
        self.run(duration_sec=self._stepLength_sec)
        self.clear_commands()
        self._sense_needed = True
        return self._sim_time - stime_before

    @override
    def resetWorld(self):
        if self._skip_next_reset_world:
            self._skip_next_reset_world = False
        elif not self._xmj_sim.reset():
            raise RuntimeError("Failed to reset XBot-MuJoCo simulation")
        self._sim_time = 0.0
        self._sense_needed = True
        self._health_cache_until = 0.0
        ZmqXbotAdapter.resetWorld(self)

    def set_filters(self, set_enabled: bool, profile_name="safe"):
        profile_cutoff_hz = {
            "safe": 5.0,
            "medium": 15.0,
            "fast": 25.0,
        }
        cutoff_hz = profile_cutoff_hz.get(profile_name, 0.0)
        self.set_reference_filter(th.tensor(cutoff_hz if set_enabled else 0.0))

    def read_imu_data(self):
        self._sense_with_sim_stepping(timeout_s=self._sense_timeout_s)
        imu_name = self._xbot_zmq_client.get_imu_names()[0]
        q_xyzw = self._xbot_zmq_client.getImuOrientation([imu_name])[0]
        self._base_q_last[:, :] = np.array([[q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]], dtype=np.float64)
        self._base_omega_last[:, :] = self._xbot_zmq_client.getImuAngularVelocity([imu_name]).reshape(1, 3)
        self._base_linacc_last[:, :] = self._xbot_zmq_client.getImuLinearAcceleration([imu_name]).reshape(1, 3)

    def get_base_link_state(self):
        return self._base_link, self._base_q_last, self._base_omega_last, self._base_linacc_last

    def _sense_if_needed(self):
        if self._sense_needed or self._sense_always:
            self._sense_with_sim_stepping(timeout_s=self._sense_timeout_s)
            self._last_successful_sense_wall_time = time.monotonic()
            self._last_health_ok = True
            self._health_cache_until = self._last_successful_sense_wall_time + self._health_check_period_s
            self._sense_needed = False

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
            if frac >= 1.0:
                break

    def setJointsStateDirect(self, jointStates: Dict[Tuple[str, str], JointState]):
        raise NotImplementedError()

    def setLinksStateDirect(self, linksStates: Dict[Tuple[str, str], LinkState]):
        raise NotImplementedError()

    def setupLight(self):
        raise NotImplementedError()

    def build_scenario(self, models=[], **kwargs):
        del models, kwargs

    def spawn_model(
        self,
        model_name: str,
        model_definition_string: Optional[str] = None,
        model_format: Optional[str] = None,
        model_file: Optional[str] = None,
        pose: Pose = build_pose(0, 0, 0, 0, 0, 0, 1),
        model_kwargs: Dict[Any, Any] = {},
    ) -> str:
        del model_name, model_definition_string, model_format, model_file, pose, model_kwargs
        raise NotImplementedError()

    def delete_model(self, model_name: str):
        del model_name
        raise NotImplementedError()

    def sim_step_duration(self):
        return self._xmj_sim.physics_dt

    def getLinksState(self, requestedLinks: Sequence[Tuple[str, str]], use_com_pose=False) -> Dict[Tuple[str, str], LinkState]:
        del use_com_pose
        ret = {}
        for requested_link in requestedLinks:
            if self._base_link not in requested_link:
                ggLog.info(f"getLinksState currently supports reading base link ({self._base_link}) state only")
                continue
            ret[requested_link] = LinkState(
                position_xyz=(self._xmj_sim.p[0], self._xmj_sim.p[1], self._xmj_sim.p[2]),
                orientation_xyzw=(self._xmj_sim.q[1], self._xmj_sim.q[2], self._xmj_sim.q[3], self._xmj_sim.q[0]),
                pos_com_velocity_xyz=(self._xmj_sim.twist[0], self._xmj_sim.twist[1], self._xmj_sim.twist[2]),
                ang_velocity_xyz=(self._xmj_sim.twist[3], self._xmj_sim.twist[4], self._xmj_sim.twist[5]),
            )
        return ret

    def get_joints_state_step_stats(self):
        raise NotImplementedError()
