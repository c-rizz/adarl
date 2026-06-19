from __future__ import annotations
"""Joint impedance controller built on top of GenesisAdapter."""
import math
import torch as th

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple
from typing_extensions import override

from adarl.adapters.GenesisAdapter import GenesisAdapter, GenesisCameraDef
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
import adarl.utils.dbg.ggLog as ggLog


class TorchExponentialFilter:
    """First-order exponential filter on (vec_size, n, k) tensors, with per-environment alpha."""
    def __init__(self, alpha: th.Tensor):
        self.alpha = alpha.view(-1, 1, 1)  # (vec_size,1,1)
        self.state: th.Tensor | None = None

    def reset(self, initial: th.Tensor) -> None:
        self.state = initial.clone()

    def apply(self, value: th.Tensor) -> th.Tensor:
        self.state = self.state * self.alpha + value * (1.0 - self.alpha)
        return self.state

    def set_alpha(self, alpha: th.Tensor, env_mask: th.Tensor) -> None:
        self.alpha = th.where(env_mask.view(-1, 1, 1), alpha.view(-1, 1, 1), self.alpha)

    @staticmethod
    def decimation_alpha(sim_step_dt: float, decimation_time: float) -> float:
        if decimation_time <= 0:
            return 0.0
        return float(0.1 ** (sim_step_dt / decimation_time))

    @staticmethod
    def cutoff_alpha(cutoff_freq: th.Tensor, dt: float) -> th.Tensor:
        # One-pole low-pass placing the -3dB cutoff at the requested frequency. Disabled (passthrough) where cutoff<=0.
        alpha = th.exp(-2.0 * math.pi * cutoff_freq.clamp(min=1e-9) * dt)
        return th.where(cutoff_freq > 0, alpha, th.zeros_like(alpha))


class TorchSecondOrderFilter:
    """Second-order low-pass filter on (vec_size, n, k) tensors, with per-environment cutoff frequency.
    Same coefficients as MujocoJointImpedanceAdapter.SecondOrderFilter. Environments with cutoff<=0 pass through."""
    def __init__(self, dt: float, cutoff_freq: th.Tensor, eps: float = 1.0):
        self.dt = float(dt)
        self.eps = float(eps)
        self.coeffs = self._compute_coeffs(cutoff_freq)  # (vec_size,1,1,5)
        self.state: th.Tensor | None = None

    def _compute_coeffs(self, cutoff_freq: th.Tensor) -> th.Tensor:
        omega_dt = 2.0 * math.pi * cutoff_freq.clamp(min=1e-9) * self.dt
        omega_dt_sq = omega_dt * omega_dt
        a0 = 1.0 + 4.0 * self.eps / omega_dt + 4.0 / omega_dt_sq
        a1 = 2.0 - 8.0 / omega_dt_sq
        a2 = 1.0 + 4.0 / omega_dt_sq - 4.0 * self.eps / omega_dt
        coeffs = th.stack([1.0 / a0, 2.0 / a0, 1.0 / a0, -a1 / a0, -a2 / a0], dim=-1)  # (vec_size,5)
        passthrough = th.zeros_like(coeffs)
        passthrough[:, 0] = 1.0
        coeffs = th.where((cutoff_freq > 0).unsqueeze(-1), coeffs, passthrough)
        return coeffs.view(-1, 1, 1, 5)

    def reset(self, initial: th.Tensor) -> None:
        # state columns: [x_t, x_t-1, x_t-2, y_t-1, y_t-2]
        self.state = initial.unsqueeze(-1).repeat(1, 1, 1, 5)

    def apply(self, value: th.Tensor) -> th.Tensor:
        s = self.state
        s[..., 1:3] = s[..., 0:2].clone()
        s[..., 0] = value
        y = (s * self.coeffs).sum(dim=-1)
        s[..., 4] = s[..., 3]
        s[..., 3] = y
        return y

    def set_cutoff(self, cutoff_freq: th.Tensor, env_mask: th.Tensor) -> None:
        new_coeffs = self._compute_coeffs(cutoff_freq)
        mask = env_mask.view(-1, 1, 1, 1)
        self.coeffs = th.where(mask, new_coeffs, self.coeffs)


@dataclass
class _QueuedCommand:
    apply_time: th.Tensor  # (vec_size,)
    cmd: th.Tensor         # (vec_size, n_joints, 5)
    pending: th.Tensor     # (vec_size,) bool


class GenesisJointImpedanceAdapter(GenesisAdapter, BaseVecJointImpedanceAdapter):
    """Joint impedance controller on top of GenesisAdapter. At each simulation substep the joint torques are
    computed as torque = kp*(pos_ref - pos) + kd*(vel_ref - vel) + effort_ref, clamped to the maximum torques,
    and applied through genesis force control."""

    def __init__(self,
                 vec_size: int,
                 output_th_device: th.device = th.device("cuda", 0),
                 sim_step_dt: float = 1 / 256,
                 step_length_sec: float = 12 / 256,
                 enable_rendering: bool = False,
                 use_raytracer: bool = False,
                 cameras: Sequence[GenesisCameraDef] = (),
                 render_envs_idx: Sequence[int] | None = None,
                 add_ground: bool = True,
                 show_gui: bool = False,
                 sim_options_override: dict[str, Any] | None = None,
                 rigid_options_override: dict[str, Any] | None = None,
                 genesis_logging_level: str = "info",
                 log_folder: str = "./",
                 default_max_joint_impedance_ctrl_torque: float = 100.0,
                 max_joint_impedance_ctrl_torques: dict[tuple[str, str], float] | None = None,
                 reference_filter_mode: str = "second_order",
                 reference_filter_cutoff_frequency: float = 20.0,
                 reference_exp_filter_decimation_time: float = 0.05,
                 joint_state_filter_decimation_time: float | None = None,
                 use_gains_filter: bool = False,
                 gains_filter_cutoff_frequency: float = 20.0,
                 gains_exp_filter_decimation_time: float = 0.05):
        super().__init__(vec_size=vec_size,
                         output_th_device=output_th_device,
                         sim_step_dt=sim_step_dt,
                         step_length_sec=step_length_sec,
                         enable_rendering=enable_rendering,
                         use_raytracer=use_raytracer,
                         cameras=cameras,
                         render_envs_idx=render_envs_idx,
                         add_ground=add_ground,
                         show_gui=show_gui,
                         sim_options_override=sim_options_override,
                         rigid_options_override=rigid_options_override,
                         genesis_logging_level=genesis_logging_level,
                         log_folder=log_folder)
        if reference_filter_mode not in ("second_order", "exponential", "none"):
            raise RuntimeError(f"Unknown reference filter mode '{reference_filter_mode}'")
        self._max_torque_default = default_max_joint_impedance_ctrl_torque
        self._max_torque_overrides = dict(max_joint_impedance_ctrl_torques or {})
        self._reference_filter_mode = reference_filter_mode
        self._reference_filter_cutoff = reference_filter_cutoff_frequency
        self._reference_exp_filter_decimation_time = reference_exp_filter_decimation_time
        self._joint_state_filter_decimation_time = joint_state_filter_decimation_time
        self._use_gains_filter = use_gains_filter
        self._gains_filter_cutoff = gains_filter_cutoff_frequency
        self._gains_exp_filter_decimation_time = gains_exp_filter_decimation_time
        self._imp_ctrl_joints: list[tuple[str, str]] = []
        self._imp_ctrl_jids = th.empty((0, 2), dtype=th.long)
        self._imp_max_torques = th.empty((1, 0))
        self._current_cmd = th.empty((vec_size, 0, 5))
        self._cmd_queue: list[_QueuedCommand] = []
        self._ref_filter: TorchSecondOrderFilter | TorchExponentialFilter | None = None
        self._gains_filter: TorchSecondOrderFilter | TorchExponentialFilter | None = None
        self._joint_state_filter: TorchExponentialFilter | None = None
        self._control_period_th = th.as_tensor(step_length_sec, device=self._out_th_device, dtype=self._out_th_float_dtype)

    # =================================================================================
    #                              controlled joints setup
    # =================================================================================

    @override
    def set_impedance_controlled_joints(self, joint_names: Sequence[Tuple[str, str]]):
        if self._scene is None:
            raise RuntimeError("set_impedance_controlled_joints() requires the scenario to be built, call build_scenario() first.")
        self._imp_ctrl_joints = [tuple(jn) for jn in joint_names]
        self._imp_ctrl_jids = self._to_joint_ids(self._imp_ctrl_joints)
        self._imp_max_torques = th.as_tensor([self._max_torque_overrides.get(jn, self._max_torque_default)
                                              for jn in self._imp_ctrl_joints],
                                             device=self._sim_dev).view(1, -1)
        self._current_cmd = th.zeros((self._vec_size, len(self._imp_ctrl_joints), 5), device=self._sim_dev)
        self._cmd_queue = []
        self._build_filters()
        self._reset_filters()

    @override
    def get_impedance_controlled_joints(self) -> List[Tuple[str, str]]:
        return list(self._imp_ctrl_joints)

    def _build_filters(self):
        dev = self._sim_dev
        if self._reference_filter_mode == "second_order":
            cutoffs = th.full((self._vec_size,), self._reference_filter_cutoff, device=dev)
            self._ref_filter = TorchSecondOrderFilter(self._sim_step_dt, cutoffs)
            if self._use_gains_filter:
                gains_cutoffs = th.full((self._vec_size,), self._gains_filter_cutoff, device=dev)
                self._gains_filter = TorchSecondOrderFilter(self._sim_step_dt, gains_cutoffs)
            else:
                self._gains_filter = None
        elif self._reference_filter_mode == "exponential":
            alpha = th.full((self._vec_size,), TorchExponentialFilter.decimation_alpha(self._sim_step_dt, self._reference_exp_filter_decimation_time), device=dev)
            self._ref_filter = TorchExponentialFilter(alpha)
            if self._use_gains_filter:
                gains_alpha = th.full((self._vec_size,), TorchExponentialFilter.decimation_alpha(self._sim_step_dt, self._gains_exp_filter_decimation_time), device=dev)
                self._gains_filter = TorchExponentialFilter(gains_alpha)
            else:
                self._gains_filter = None
        else:
            self._ref_filter = None
            self._gains_filter = None
        if self._joint_state_filter_decimation_time is not None:
            alpha = th.full((self._vec_size,), TorchExponentialFilter.decimation_alpha(self._sim_step_dt, self._joint_state_filter_decimation_time), device=dev)
            self._joint_state_filter = TorchExponentialFilter(alpha)
        else:
            self._joint_state_filter = None

    def _reset_filters(self):
        if len(self._imp_ctrl_joints) == 0:
            return
        rs = self._rigid_solver()
        if self._joint_state_filter is not None:
            q = rs.get_qpos(qs_idx=self._imp_ctrl_jids[:, 1])
            qd = rs.get_dofs_velocity(dofs_idx=self._imp_ctrl_jids[:, 0])
            self._joint_state_filter.reset(th.stack([q, qd], dim=2))
        if self._ref_filter is not None:
            self._ref_filter.reset(self._current_cmd[:, :, 0:3].clone())
        if self._gains_filter is not None:
            self._gains_filter.reset(self._current_cmd[:, :, 3:5].clone())

    # =================================================================================
    #                                   commands
    # =================================================================================

    def _validate_cmd(self, joint_impedances_pvesd: th.Tensor, joint_names) -> None:
        if joint_names is not None and [tuple(jn) for jn in joint_names] != self._imp_ctrl_joints:
            raise NotImplementedError("Commands must target all the impedance controlled joints, in the order "
                                      "specified with set_impedance_controlled_joints().")
        expected = (self._vec_size, len(self._imp_ctrl_joints), 5)
        if tuple(joint_impedances_pvesd.shape) != expected:
            raise ValueError(f"joint_impedances_pvesd has shape {tuple(joint_impedances_pvesd.shape)}, expected {expected}")

    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd: th.Tensor,
                                  delay_sec: th.Tensor | float = 0.0,
                                  vec_mask: th.Tensor | None = None,
                                  joint_names: Sequence[tuple[str, str]] | None = None) -> None:
        self._validate_cmd(joint_impedances_pvesd, joint_names)
        now = self.getEnvTimeFromStartup()
        if isinstance(delay_sec, th.Tensor) and delay_sec.numel() > 1:
            apply_time = now + delay_sec.to(self._sim_dev).clamp(min=0.0).view(-1)
        else:
            d = delay_sec.item() if isinstance(delay_sec, th.Tensor) else float(delay_sec)
            apply_time = th.full((self._vec_size,), now + max(0.0, d), device=self._sim_dev)
        if vec_mask is None:
            pending = th.ones((self._vec_size,), dtype=th.bool, device=self._sim_dev)
        else:
            pending = vec_mask.to(device=self._sim_dev, dtype=th.bool).clone()
        if not bool(pending.any()):
            return
        cmd = joint_impedances_pvesd.to(device=self._sim_dev, dtype=th.float32).clone()
        self._cmd_queue.append(_QueuedCommand(apply_time=apply_time, cmd=cmd, pending=pending))

    @override
    def set_current_joint_impedance_command(self, joint_impedances_pvesd: th.Tensor,
                                            joint_names: Sequence[tuple[str, str]] | None = None,
                                            vec_mask: th.Tensor | None = None) -> None:
        self._validate_cmd(joint_impedances_pvesd, joint_names)
        cmd = joint_impedances_pvesd.to(device=self._sim_dev, dtype=th.float32)
        if vec_mask is None:
            self._current_cmd.copy_(cmd)
        else:
            mask = vec_mask.to(device=self._sim_dev, dtype=th.bool)
            self._current_cmd[mask] = cmd[mask]
        self._apply_impedance_torques(apply_queue=False)

    @override
    def reset_joint_impedances_commands(self):
        self._cmd_queue = []
        if len(self._imp_ctrl_joints) > 0:
            self._current_cmd.zero_()
        self._reset_filters()

    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        return self._out(self._current_cmd).clone()

    @override
    def control_period(self) -> th.Tensor:
        return self._control_period_th

    @override
    def set_reference_filter(self, reference_filter_cutoff_frequency: th.Tensor, vec_mask: th.Tensor | None = None):
        if self._reference_filter_mode == "none":
            ggLog.warn("GenesisJointImpedanceAdapter.set_reference_filter() called, but reference_filter_mode is 'none'. Ignoring.")
            return
        if self._ref_filter is None:
            raise RuntimeError("Reference filter is not built, call set_impedance_controlled_joints() first.")
        cutoffs = th.as_tensor(reference_filter_cutoff_frequency, device=self._sim_dev, dtype=th.float32).expand(self._vec_size).clone()
        if vec_mask is None:
            env_mask = th.ones((self._vec_size,), dtype=th.bool, device=self._sim_dev)
        else:
            env_mask = vec_mask.to(device=self._sim_dev, dtype=th.bool)
        if isinstance(self._ref_filter, TorchSecondOrderFilter):
            self._ref_filter.set_cutoff(cutoffs, env_mask)
        else:
            self._ref_filter.set_alpha(TorchExponentialFilter.cutoff_alpha(cutoffs, self._sim_step_dt), env_mask)
        # reset the filter state of the updated environments to the current references
        if self._ref_filter.state is not None and len(self._imp_ctrl_joints) > 0:
            refs = self._current_cmd[:, :, 0:3]
            if isinstance(self._ref_filter, TorchSecondOrderFilter):
                self._ref_filter.state[env_mask] = refs[env_mask].unsqueeze(-1).repeat(1, 1, 1, 5)
            else:
                self._ref_filter.state[env_mask] = refs[env_mask]

    # =================================================================================
    #                              control loop internals
    # =================================================================================

    def _pop_due_commands(self, now: float):
        if len(self._cmd_queue) == 0:
            return
        still_queued = []
        for entry in self._cmd_queue:
            due = entry.pending & (entry.apply_time <= now)
            if bool(due.any()):
                self._current_cmd[due] = entry.cmd[due]
                entry.pending = entry.pending & (~due)
            if bool(entry.pending.any()):
                still_queued.append(entry)
        self._cmd_queue = still_queued

    def _apply_impedance_torques(self, apply_queue: bool = True):
        if len(self._imp_ctrl_joints) == 0:
            return
        if apply_queue:
            self._pop_due_commands(self.getEnvTimeFromStartup())
        rs = self._rigid_solver()
        q = rs.get_qpos(qs_idx=self._imp_ctrl_jids[:, 1])
        qd = rs.get_dofs_velocity(dofs_idx=self._imp_ctrl_jids[:, 0])
        if self._joint_state_filter is not None:
            filtered = self._joint_state_filter.apply(th.stack([q, qd], dim=2))
            q, qd = filtered[:, :, 0], filtered[:, :, 1]
        refs = self._current_cmd[:, :, 0:3]
        if self._ref_filter is not None:
            refs = self._ref_filter.apply(refs)
        gains = self._current_cmd[:, :, 3:5]
        if self._gains_filter is not None:
            gains = self._gains_filter.apply(gains)
        pos_ref, vel_ref, eff_ref = refs[:, :, 0], refs[:, :, 1], refs[:, :, 2]
        kp, kd = gains[:, :, 0], gains[:, :, 1]
        torques = kp * (pos_ref - q) + kd * (vel_ref - qd) + eff_ref
        torques = th.clamp(torques, -self._imp_max_torques, self._imp_max_torques)
        rs.control_dofs_force(torques, dofs_idx=self._imp_ctrl_jids[:, 0])

    @override
    def _apply_commands(self):
        self._apply_impedance_torques()
        super()._apply_commands()
