from __future__ import annotations

from adarl.adapters.MujocoAdapter import MujocoAdapter
import heapq
import numpy as np
import torch as th
from typing import Sequence
from typing_extensions import override
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter




class ExponentialFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state: np.ndarray

    def reset(self, initial: np.ndarray) -> None:
        self.state = np.array(initial, dtype=np.float32, copy=True)

    def apply(self, value: np.ndarray) -> np.ndarray:
        v = np.array(value, dtype=np.float32)
        self.state = self.state * self.alpha + v * (1.0 - self.alpha)
        return self.state
    
    @staticmethod
    def _decimation_alpha(sim_step_dt: float, decimation_time: float) -> float:
        if decimation_time <= 0:
            return 0.0
        return float(0.1 ** (sim_step_dt / decimation_time))


class SecondOrderFilter:
    def __init__(self, dt: float, cutoff_freq: float, eps: float = 1.0):
        self.coeffs = self._compute_coeffs(dt, cutoff_freq, eps)
        self.state: np.ndarray

    def reset(self, initial: np.ndarray) -> None:
        init_flat = np.array(initial, dtype=np.float32, copy=False).reshape(-1)
        self.state = np.repeat(init_flat[:, None], 5, axis=1)

    def apply(self, value: np.ndarray) -> np.ndarray:
        v = np.array(value, dtype=np.float32, copy=False)
        flat = v.reshape(-1)
        if self.state is None or self.state.shape[0] != flat.shape[0]:
            self.reset(v)
            flat = self.state[:, 0]
        state = self.state
        state[:, 1:3] = state[:, 0:2]
        state[:, 0] = flat
        y_flat = state @ self.coeffs
        state[:, 4] = state[:, 3]
        state[:, 3] = y_flat
        self.state = state
        return y_flat.reshape(v.shape)

    @staticmethod
    def _compute_coeffs(dt: float, cutoff_freq: float, eps: float = 1.0) -> np.ndarray:
        omega = 2 * np.pi * cutoff_freq
        b1 = 2.0
        b2 = 1.0
        omega_dt = omega * dt
        omega_dt_sq = omega_dt * omega_dt
        a0 = 1.0 + 4.0 * eps / omega_dt + 4.0 / omega_dt_sq
        a1 = 2.0 - 8.0 / omega_dt_sq
        a2 = 1.0 + 4.0 / omega_dt_sq - 4.0 * eps / omega_dt
        return np.array([1.0 / a0, b1 / a0, b2 / a0, -a1 / a0, -a2 / a0], dtype=np.float32)


class MujocoJointImpedanceAdapter(MujocoAdapter, BaseVecJointImpedanceAdapter):
    """Joint impedance controller built on top of the MujocoAdapter (mujoco classic, only vec_size=1).
    
    WARNING: This adapter should still be considered as a Work In Progress."""

    def __init__(self,
                 vec_size: int = 1,
                 sim_step_dt: float = 2 / 1024,
                 step_length_sec: float = 48 / 1024,
                 output_th_device: th.device = th.device("cpu"),
                 default_max_joint_impedance_ctrl_torque: float = 100.0,
                 max_joint_impedance_ctrl_torques: dict[tuple[str, str], float] | None = None,
                 reference_filter_cutoff_frequency: float = 20.0,
                 reference_filter_mode: str = "second_order",
                 reference_exp_filter_decimation_time: float = 0.05,
                 joint_state_filter_decimation_time: float | None = 0.005,
                 gains_exp_filter_decimation_time: float = 0.05,
                 gains_filter_cutoff_frequency: float = 20.0,
                 use_gains_filter: bool = False):
        super().__init__(vec_size=vec_size,
                         sim_step_dt=sim_step_dt,
                         step_length_sec=step_length_sec,
                         output_th_device=output_th_device)
        self._max_torque_default = default_max_joint_impedance_ctrl_torque
        self._max_torque_overrides = max_joint_impedance_ctrl_torques or {}
        self._imp_ctrl_joints: list[tuple[str, str]] = []
        self._imp_ctrl_jids: np.ndarray = np.empty((0,), dtype=int)
        self._current_cmd : np.ndarray = np.zeros((self.vec_size(), 0, 5), dtype=np.float32)
        self._cmd_queue: list[tuple[float, int, np.ndarray]] = []
        self._cmd_queue_counter = 0
        self._control_period_th = th.as_tensor(step_length_sec, device=self._out_th_device, dtype=self._out_th_float_dtype)
        if reference_filter_mode == "second_order":
            self._ref_filter = SecondOrderFilter(self._sim_step_dt, reference_filter_cutoff_frequency)
            if use_gains_filter:
                self._gains_filter = SecondOrderFilter(self._sim_step_dt, gains_filter_cutoff_frequency)
            else:
                self._gains_filter = None
        elif reference_filter_mode == "exponential":
            self._ref_filter = ExponentialFilter(ExponentialFilter._decimation_alpha(self._sim_step_dt, reference_exp_filter_decimation_time))
            if use_gains_filter:
                self._gains_filter = ExponentialFilter(ExponentialFilter._decimation_alpha(self._sim_step_dt, gains_exp_filter_decimation_time))
            else:
                self._gains_filter = None
        elif reference_filter_mode == "none":
            self._ref_filter = None
            self._gains_filter = None
        else:
            raise RuntimeError(f"Unknown reference filter mode '{reference_filter_mode}'")
        if joint_state_filter_decimation_time is not None:
            self._joint_state_filter = ExponentialFilter(ExponentialFilter._decimation_alpha(self._sim_step_dt, joint_state_filter_decimation_time))
        else:
            self._joint_state_filter = None

    @override
    def set_impedance_controlled_joints(self, joint_names: Sequence[tuple[str, str]]):
        self._imp_ctrl_joints = list(joint_names)
        self._imp_ctrl_jids = np.array([self._jname2jid[jn] for jn in joint_names], dtype=int)
        self._current_cmd = np.zeros((self.vec_size(), len(joint_names), 5), dtype=np.float32)
        self._cmd_queue.clear()
        self._cmd_queue_counter = 0
        self._reset_filters()

    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str, str]]:
        return list(self._imp_ctrl_joints)

    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd: th.Tensor,
                                  delay_sec: th.Tensor | float = 0.0,
                                  vec_mask: th.Tensor | None = None,
                                  joint_names: Sequence[tuple[str, str]] | None = None) -> None:
        if joint_names is not None and len(joint_names) != len(self._imp_ctrl_joints):
            raise NotImplementedError("Specifying joint_names is not supported; commands must target the impedance-controlled joints.")
        if vec_mask is not None and not vec_mask.item():
            return
        if joint_impedances_pvesd.shape[0] != 1:
            raise ValueError("joint_impedances_pvesd must have leading dimension 1")
        expected_len = len(self._imp_ctrl_joints)
        if joint_impedances_pvesd.shape[1] != expected_len or joint_impedances_pvesd.shape[2] != 5:
            raise ValueError(f"joint_impedances_pvesd has wrong shape {tuple(joint_impedances_pvesd.shape)}, expected (1,{expected_len},5)")
            
        d = delay_sec.item() if isinstance(delay_sec, th.Tensor) else float(delay_sec)
        cmd_time = self.getEnvTimeFromStartup() + max(0.0, d)
        self._enqueue_command(cmd_time, joint_impedances_pvesd.cpu().numpy())

    @override
    def reset_joint_impedances_commands(self):
        self._cmd_queue.clear()
        self._cmd_queue_counter = 0
        if len(self._imp_ctrl_joints) > 0:
            self._current_cmd[:] = 0
        self._reset_filters()

    @override
    def set_current_joint_impedance_command(self, joint_impedances_pvesd: th.Tensor,
                                            joint_names: Sequence[tuple[str, str]] | None = None,
                                            vec_mask: th.Tensor | None = None) -> None:
        if joint_names is not None:
            raise NotImplementedError("Specifying joint_names is not supported; commands must target the impedance-controlled joints.")
        if vec_mask is not None and not vec_mask.item():
            return
        if joint_impedances_pvesd.shape[0] != 1:
            raise ValueError("joint_impedances_pvesd must have leading dimension 1")
        expected_len = len(self._imp_ctrl_joints)
        if joint_impedances_pvesd.shape[1] != expected_len or joint_impedances_pvesd.shape[2] != 5:
            raise ValueError(f"joint_impedances_pvesd has wrong shape {tuple(joint_impedances_pvesd.shape)}, expected (1,{expected_len},5)")
        self._enqueue_command(self.getEnvTimeFromStartup(), joint_impedances_pvesd.cpu().numpy())
        self._current_cmd = self._cmd_queue[0][2]
        self._apply_impedance_torques()

    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        return th.as_tensor(self._current_cmd, device=self._out_th_device, dtype=self._out_th_float_dtype).clone()

    @override
    def control_period(self) -> th.Tensor:
        return self._control_period_th
        
    def _apply_commands(self):
        self._apply_impedance_torques()
        super()._apply_commands()

    def _apply_impedance_torques(self):
        if len(self._imp_ctrl_joints) == 0:
            return
        now = self.getEnvTimeFromStartup()
        # Apply all commands scheduled up to the current simulation time, keeping the latest one.
        while self._cmd_queue and self._cmd_queue[0][0] <= now:
            _, _, next_cmd = heapq.heappop(self._cmd_queue)
            self._current_cmd = next_cmd
        jids = self._imp_ctrl_jids
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        cmd = self._current_cmd[0]
        pos_ref = cmd[:, 0]
        vel_ref = cmd[:, 1]
        eff_ref = cmd[:, 2]
        kp = cmd[:, 3]
        kd = cmd[:, 4]
        qpos = self._mj_data.qpos[qpos_adrs] #type: ignore
        qvel = self._mj_data.qvel[qvel_adrs] #type: ignore
        qpos,qvel = self._filter_joint_sensing(qpos, qvel)
        pos_ref, vel_ref, eff_ref = self._filter_references(pos_ref, vel_ref, eff_ref)
        kp, kd = self._filter_gains(kp, kd)
        torques = kp * (pos_ref - qpos) + kd * (vel_ref - qvel) + eff_ref
        max_torques = np.array([self._max_torque_overrides.get(jn, self._max_torque_default) for jn in self._imp_ctrl_joints], dtype=np.float64)
        torques = np.clip(torques, -max_torques, max_torques)
        self._set_effort_command(jids, torques[np.newaxis, :])
        # self._mj_data.qfrc_applied[qvel_adrs] += torques

    def _filter_joint_sensing(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._joint_state_filter is not None:
            fq = self._joint_state_filter.apply(np.stack((qpos, qvel), axis=1))
            return fq[:, 0], fq[:, 1]
        else:
            return qpos, qvel
    
    def _filter_references(self, pos_ref: np.ndarray, vel_ref: np.ndarray, eff_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._ref_filter is not None:
            filtered_refs = self._ref_filter.apply(np.stack((pos_ref, vel_ref, eff_ref), axis=1))
            return filtered_refs[:, 0], filtered_refs[:, 1], filtered_refs[:, 2]
        else:
            return pos_ref, vel_ref, eff_ref
    
    def _filter_gains(self, kp:  np.ndarray, kd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._gains_filter is not None:
            gains = np.stack((kp, kd), axis=1)
            filtered = self._gains_filter.apply(gains)
            return filtered[:, 0], filtered[:, 1]
        else:
            return kp, kd

    def _enqueue_command(self, cmd_time: float, cmd: np.ndarray) -> None:
        heapq.heappush(self._cmd_queue, (cmd_time, self._cmd_queue_counter, cmd))
        self._cmd_queue_counter += 1

    def _reset_filters(self) -> None:
        ctrl_joint_count = len(self._imp_ctrl_joints)
        all_qpos = self._mj_data.qpos # type: ignore
        all_qvel = self._mj_data.qvel # type: ignore
        ctrl_qpos = all_qpos[self._mj_model.jnt_qposadr[self._imp_ctrl_jids]].astype(np.float32)
        ctrl_qvel = all_qvel[self._mj_model.jnt_dofadr[self._imp_ctrl_jids]].astype(np.float32)
        
        current_cmd_j_pvesd = self._current_cmd[0]
        if self._joint_state_filter is not None:
            self._joint_state_filter.reset(np.stack((ctrl_qpos, ctrl_qvel), axis=1))
        if self._ref_filter is not None:
            self._ref_filter.reset(current_cmd_j_pvesd[:,:3])
        if self._gains_filter is not None:
            init_gains = current_cmd_j_pvesd[:, 3:]
            self._gains_filter.reset(init_gains)
