from __future__ import annotations

from adarl.adapters.MujocoAdapter import MujocoAdapter
import heapq
import numpy as np
import torch as th
from typing import Sequence, Any
from typing_extensions import override
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter




class ExponentialFilter:
    """Exponential smoothing of one array per environment, state of shape (vec_size, ...), with a factor alpha per
    environment. Computed in float32."""
    def __init__(self, alpha: float, vec_size: int):
        self.alpha = np.full((vec_size,), alpha, dtype=np.float32)
        self.one_minus_alpha = np.full((vec_size,), 1.0 - alpha, dtype=np.float32)
        self.state: np.ndarray | None = None

    def set_alpha(self, envs: np.ndarray, alpha: float) -> None:
        self.alpha[envs] = alpha
        self.one_minus_alpha[envs] = 1.0 - alpha

    def reset(self, initial: np.ndarray, envs: np.ndarray | None = None) -> None:
        """initial: (vec_size, ...), the state of the environments of envs (None: all)."""
        initial = np.array(initial, dtype=np.float32, copy=True)
        if envs is None or self.state is None or self.state.shape != initial.shape:
            self.state = initial
        else:
            self.state[envs] = initial[envs]

    def apply(self, value: np.ndarray, envs: np.ndarray) -> np.ndarray:
        """value: (len(envs), ...), the new inputs of the environments of envs. Returns their filtered values."""
        assert self.state is not None
        v = np.array(value, dtype=np.float32)
        shape = (len(envs),) + (1,)*(v.ndim-1)
        filtered = self.state[envs] * self.alpha[envs].reshape(shape) + v * self.one_minus_alpha[envs].reshape(shape)
        self.state[envs] = filtered
        return filtered

    @staticmethod
    def _decimation_alpha(sim_step_dt: float, decimation_time: float) -> float:
        if decimation_time <= 0:
            return 0.0
        return float(0.1 ** (sim_step_dt / decimation_time))


class SecondOrderFilter:
    """Second-order low-pass filter of one array per environment, with a cutoff per environment. Computed in float32.
    The state of each environment holds, per element, [x_k, x_k-1, x_k-2, y_k-1, y_k-2]."""
    def __init__(self, dt: float, cutoff_freq: float, vec_size: int, eps: float = 1.0):
        self.dt = dt
        self.eps = eps
        self.coeffs = np.repeat(self._compute_coeffs(dt, cutoff_freq, eps)[None, :], vec_size, axis=0) # (vec_size, 5)
        self.state: np.ndarray | None = None # (vec_size, elements, 5)

    def set_cutoff(self, envs: np.ndarray, cutoff_freq: float) -> None:
        self.coeffs[envs] = self._compute_coeffs(self.dt, cutoff_freq, self.eps)

    def reset(self, initial: np.ndarray, envs: np.ndarray | None = None) -> None:
        """initial: (vec_size, ...), the state of the environments of envs (None: all)."""
        init_flat = np.array(initial, dtype=np.float32).reshape(initial.shape[0], -1)
        new_state = np.repeat(init_flat[:, :, None], 5, axis=2)
        if envs is None or self.state is None or self.state.shape != new_state.shape:
            self.state = new_state
        else:
            self.state[envs] = new_state[envs]

    def apply(self, value: np.ndarray, envs: np.ndarray) -> np.ndarray:
        """value: (len(envs), ...), the new inputs of the environments of envs. Returns their filtered values."""
        assert self.state is not None
        v = np.array(value, dtype=np.float32)
        flat = v.reshape(len(envs), -1)
        state = self.state[envs]
        state[:, :, 1:3] = state[:, :, 0:2]
        state[:, :, 0] = flat
        # one matrix-vector product per environment: the same float32 results as for each environment alone
        y_flat = np.matmul(state, self.coeffs[envs][:, :, None])[:, :, 0]
        state[:, :, 4] = state[:, :, 3]
        state[:, :, 3] = y_flat
        self.state[envs] = state
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
    """Joint impedance controller built on top of the MujocoAdapter (mujoco classic), for vec_size environments.

    Each environment has its own command queue, with commands applied after their delay on that environment's
    clock, its own filter states and its own reference filter cutoff. The environments that are not stepped (per
    environment durations in run()) keep their commands and filter states as they are.

    WARNING: This adapter should still be considered as a Work In Progress."""

    def __init__(self,
                 vec_size: int = 1,
                 sim_step_dt: float = 2 / 1024,
                 step_length_sec: float = 48 / 1024,
                 output_th_device: th.device = th.device("cpu"),
                 log_folder: str = "./",
                 show_gui: bool = False,
                 gui_frequency: float = 25.0,
                 opt_preset: str = "mujoco_default",
                 opt_override: dict | None = None,
                 safe_revolute_dof_armature: float = 0.01,
                 revolute_dof_armature_override: DofOverride = None,
                 revolute_dof_damping_override: DofOverride = None,
                 revolute_dof_frictionloss_override: DofOverride = None,
                 default_max_joint_impedance_ctrl_torque: float = 100.0,
                 max_joint_impedance_ctrl_torques: dict[tuple[str, str], float] | None = None,
                 reference_filter_cutoff_frequency: float = 20.0,
                 reference_filter_mode: str = "second_order",
                 reference_exp_filter_decimation_time: float = 0.05,
                 joint_state_filter_decimation_time: float | None = None,
                 gains_exp_filter_decimation_time: float = 0.05,
                 gains_filter_cutoff_frequency: float = 20.0,
                 use_gains_filter: bool = False,
                 geom_overrides : dict[str,Any] | None = None,
                 ground_rgba : tuple[float,float,float,float] | None = None,
                 render_znear : float | None = 0.01,
                 render_shadows : bool = True,
                 sim_threads : int | None = None):
        super().__init__(vec_size=vec_size,
                         sim_step_dt=sim_step_dt,
                         step_length_sec=step_length_sec,
                         output_th_device=output_th_device,
                         log_folder=log_folder,
                         show_gui=show_gui,
                         gui_frequency=gui_frequency,
                         opt_preset=opt_preset,
                         opt_override=opt_override,
                         safe_revolute_dof_armature=safe_revolute_dof_armature,
                         revolute_dof_armature_override=revolute_dof_armature_override,
                         revolute_dof_damping_override=revolute_dof_damping_override,
                         revolute_dof_frictionloss_override=revolute_dof_frictionloss_override,
                         geom_overrides=geom_overrides,
                         ground_rgba=ground_rgba,
                         render_znear=render_znear,
                         render_shadows=render_shadows,
                         sim_threads=sim_threads)
        self._max_torque_default = default_max_joint_impedance_ctrl_torque
        self._max_torque_overrides = max_joint_impedance_ctrl_torques or {}
        self._imp_ctrl_joints: list[tuple[str, str]] = []
        self._imp_ctrl_jids: np.ndarray = np.empty((0,), dtype=int)
        self._max_torques = np.empty((0,), dtype=np.float64)
        self._current_cmd : np.ndarray = np.zeros((vec_size, 0, 5), dtype=np.float32)
        # per environment, a heap of (time, insertion counter, command of shape (joints, 5))
        self._cmd_queues: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(vec_size)]
        self._cmd_queue_counter = 0
        self._control_period_th = th.as_tensor(step_length_sec, device=self._out_th_device, dtype=self._out_th_float_dtype)
        self._reference_filter_mode = reference_filter_mode
        self._reference_filter_cutoff_frequency = np.full((vec_size,), reference_filter_cutoff_frequency, dtype=np.float64)
        # which environments filter their references (set_reference_filter can turn it off per environment)
        self._ref_filter_enabled = np.full((vec_size,), reference_filter_mode != "none", dtype=bool)
        self._ref_filter : ExponentialFilter | SecondOrderFilter | None
        self._gains_filter : ExponentialFilter | SecondOrderFilter | None
        if reference_filter_mode == "second_order":
            self._ref_filter = SecondOrderFilter(self._sim_step_dt, reference_filter_cutoff_frequency, vec_size)
            if use_gains_filter:
                self._gains_filter = SecondOrderFilter(self._sim_step_dt, gains_filter_cutoff_frequency, vec_size)
            else:
                self._gains_filter = None
        elif reference_filter_mode == "exponential":
            self._ref_filter = ExponentialFilter(ExponentialFilter._decimation_alpha(self._sim_step_dt, reference_exp_filter_decimation_time), vec_size)
            if use_gains_filter:
                self._gains_filter = ExponentialFilter(ExponentialFilter._decimation_alpha(self._sim_step_dt, gains_exp_filter_decimation_time), vec_size)
            else:
                self._gains_filter = None
        elif reference_filter_mode == "none":
            self._ref_filter = None
            self._gains_filter = None
        else:
            raise RuntimeError(f"Unknown reference filter mode '{reference_filter_mode}'")
        if joint_state_filter_decimation_time is not None:
            self._joint_state_filter = ExponentialFilter(ExponentialFilter._decimation_alpha(self._sim_step_dt, joint_state_filter_decimation_time), vec_size)
        else:
            self._joint_state_filter = None

    def _check_cmd_shape(self, joint_impedances_pvesd: th.Tensor):
        expected = (self._vec_size, len(self._imp_ctrl_joints), 5)
        if tuple(joint_impedances_pvesd.shape) != expected:
            raise ValueError(f"joint_impedances_pvesd has wrong shape {tuple(joint_impedances_pvesd.shape)}, expected {expected}")

    @override
    def set_impedance_controlled_joints(self, joint_names: Sequence[tuple[str, str]]):
        self._imp_ctrl_joints = list(joint_names)
        self._imp_ctrl_jids = np.array([self._jname2jid[jn] for jn in joint_names], dtype=int)
        self._max_torques = np.array([self._max_torque_overrides.get(jn, self._max_torque_default) for jn in self._imp_ctrl_joints], dtype=np.float64)
        self._current_cmd = np.zeros((self._vec_size, len(joint_names), 5), dtype=np.float32)
        for queue in self._cmd_queues:
            queue.clear()
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
        """joint_impedances_pvesd: (vec_size, joints, 5); delay_sec: one delay, or one per environment (vec_size,).
        Each environment of vec_mask applies its command after its delay, on its own clock."""
        if joint_names is not None and len(joint_names) != len(self._imp_ctrl_joints):
            raise NotImplementedError("Specifying joint_names is not supported; commands must target the impedance-controlled joints.")
        envs = self._all_ids(self._vec_mask_to_ids(vec_mask))
        if len(envs) == 0:
            return
        self._check_cmd_shape(joint_impedances_pvesd)
        if isinstance(delay_sec, th.Tensor):
            delays = np.broadcast_to(delay_sec.detach().cpu().numpy().astype(np.float64).reshape(-1), (self._vec_size,))
        else:
            delays = np.full((self._vec_size,), float(delay_sec), dtype=np.float64)
        cmds = joint_impedances_pvesd.detach().cpu().numpy()
        for env in envs:
            cmd_time = float(self._time_since_startup[env]) + max(0.0, float(delays[env]))
            self._enqueue_command(int(env), cmd_time, cmds[env])

    @override
    def reset_joint_impedances_commands(self):
        for queue in self._cmd_queues:
            queue.clear()
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
        envs = self._all_ids(self._vec_mask_to_ids(vec_mask))
        if len(envs) == 0:
            return
        self._check_cmd_shape(joint_impedances_pvesd)
        cmds = joint_impedances_pvesd.detach().cpu().numpy()
        for env in envs:
            self._enqueue_command(int(env), float(self._time_since_startup[env]), cmds[env])
            self._current_cmd[env] = self._cmd_queues[env][0][2]
        self._apply_impedance_torques(envs)

    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        return th.as_tensor(self._current_cmd, device=self._out_th_device, dtype=self._out_th_float_dtype).clone()

    @override
    def control_period(self) -> th.Tensor:
        return self._control_period_th

    @override
    def set_reference_filter(self, reference_filter_cutoff_frequency : th.Tensor, vec_mask : th.Tensor | None = None):
        """reference_filter_cutoff_frequency: one cutoff, or one per environment (vec_size,), for the environments of
        vec_mask. A cutoff <= 0 turns the filter off for that environment. The filter restarts from the current
        command."""
        envs = self._all_ids(self._vec_mask_to_ids(vec_mask))
        if len(envs) == 0:
            return
        cutoffs = np.broadcast_to(np.asarray(th.as_tensor(reference_filter_cutoff_frequency).detach().cpu().numpy(),
                                             dtype=np.float64).reshape(-1), (self._vec_size,))
        for env in envs:
            cutoff = float(cutoffs[env])
            self._reference_filter_cutoff_frequency[env] = cutoff
            enabled = cutoff > 0.0 and self._reference_filter_mode != "none"
            self._ref_filter_enabled[env] = enabled
            if not enabled:
                continue
            assert self._ref_filter is not None
            if isinstance(self._ref_filter, ExponentialFilter):
                # One-pole low-pass: alpha placing the -3dB cutoff at the requested frequency.
                self._ref_filter.set_alpha(np.array([env]), float(np.exp(-2.0 * np.pi * cutoff * self._sim_step_dt)))
            else:  # "second_order"
                self._ref_filter.set_cutoff(np.array([env]), cutoff)
        enabled_envs = envs[self._ref_filter_enabled[envs]]
        if self._ref_filter is not None and self._batch is not None and len(self._imp_ctrl_joints) > 0 and len(enabled_envs) > 0:
            self._ref_filter.reset(self._current_cmd[:, :, :3], enabled_envs)

    def _apply_commands(self, ids: np.ndarray | None = None):
        self._apply_impedance_torques(ids)
        super()._apply_commands(ids)

    def _apply_impedance_torques(self, ids: np.ndarray | None = None):
        """Compute the impedance torques of the environments of ids (None: all) and request them."""
        if len(self._imp_ctrl_joints) == 0:
            return
        envs = self._all_ids(ids)
        # Apply all commands scheduled up to each environment's current time, keeping the latest one.
        for env in envs:
            queue = self._cmd_queues[env]
            now = float(self._time_since_startup[env])
            while queue and queue[0][0] <= now:
                _, _, next_cmd = heapq.heappop(queue)
                self._current_cmd[env] = next_cmd
        jids = self._imp_ctrl_jids
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        cmd = self._current_cmd[envs]
        pos_ref = cmd[:, :, 0]
        vel_ref = cmd[:, :, 1]
        eff_ref = cmd[:, :, 2]
        kp = cmd[:, :, 3]
        kd = cmd[:, :, 4]
        rows = envs[:, None]
        qpos = self._b_qpos[rows, qpos_adrs]
        qvel = self._b_qvel[rows, qvel_adrs]
        qpos,qvel = self._filter_joint_sensing(qpos, qvel, envs)
        pos_ref, vel_ref, eff_ref = self._filter_references(pos_ref, vel_ref, eff_ref, envs)
        kp, kd = self._filter_gains(kp, kd, envs)
        torques = kp * (pos_ref - qpos) + kd * (vel_ref - qvel) + eff_ref
        torques = np.clip(torques, -self._max_torques, self._max_torques)
        efforts = np.zeros((self._vec_size, len(jids)), dtype=torques.dtype)
        efforts[envs] = torques
        env_mask = np.zeros((self._vec_size,), dtype=bool)
        env_mask[envs] = True
        self._set_effort_command(jids, efforts, vec_mask=th.as_tensor(env_mask))

    def _filter_joint_sensing(self, qpos: np.ndarray, qvel: np.ndarray, envs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._joint_state_filter is not None:
            fq = self._joint_state_filter.apply(np.stack((qpos, qvel), axis=-1), envs)
            return fq[..., 0], fq[..., 1]
        else:
            return qpos, qvel

    def _filter_references(self, pos_ref: np.ndarray, vel_ref: np.ndarray, eff_ref: np.ndarray, envs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._ref_filter is None:
            return pos_ref, vel_ref, eff_ref
        enabled = self._ref_filter_enabled[envs]
        if not np.any(enabled):
            return pos_ref, vel_ref, eff_ref
        refs = np.stack((pos_ref, vel_ref, eff_ref), axis=-1)
        if np.all(enabled):
            filtered_refs = self._ref_filter.apply(refs, envs)
        else: # the environments with the filter turned off pass their references through
            filtered_refs = refs.astype(np.float32)
            filtered_refs[enabled] = self._ref_filter.apply(refs[enabled], envs[enabled])
        return filtered_refs[..., 0], filtered_refs[..., 1], filtered_refs[..., 2]

    def _filter_gains(self, kp:  np.ndarray, kd: np.ndarray, envs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._gains_filter is not None:
            gains = np.stack((kp, kd), axis=-1)
            filtered = self._gains_filter.apply(gains, envs)
            return filtered[..., 0], filtered[..., 1]
        else:
            return kp, kd

    def _enqueue_command(self, env: int, cmd_time: float, cmd: np.ndarray) -> None:
        heapq.heappush(self._cmd_queues[env], (cmd_time, self._cmd_queue_counter, cmd))
        self._cmd_queue_counter += 1

    def _reset_filters(self) -> None:
        """Restart every environment's filters from its current joint state and command."""
        ctrl_qpos = self._b_qpos[:, self._mj_model.jnt_qposadr[self._imp_ctrl_jids]].astype(np.float32)
        ctrl_qvel = self._b_qvel[:, self._mj_model.jnt_dofadr[self._imp_ctrl_jids]].astype(np.float32)
        if self._joint_state_filter is not None:
            self._joint_state_filter.reset(np.stack((ctrl_qpos, ctrl_qvel), axis=-1))
        if self._ref_filter is not None:
            self._ref_filter.reset(self._current_cmd[:, :, :3])
        if self._gains_filter is not None:
            self._gains_filter.reset(self._current_cmd[:, :, 3:])
