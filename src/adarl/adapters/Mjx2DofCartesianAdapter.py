from __future__ import annotations

from adarl.adapters.MjxActuatedAdapter import MjxActuatedAdapter
from adarl.adapters.BaseVecCartesianPositionAdapter import BaseVecCartesianPositionAdapter
from typing import Sequence, Any
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import torch as th


class Mjx2DofCartesianAdapter(MjxActuatedAdapter, BaseVecCartesianPositionAdapter):
    """MJX adapter for a 2-DoF planar 'arm' controlled by specifying the end-effector position in the xy plane.

    This is the MJX/vectorized counterpart of :class:`PyBullet2DofCartesianAdapter`. The mechanism is
    assumed to be two prismatic joints whose axes are the world x and y axes, chained so that the
    second one carries the end effector. The inverse kinematics is therefore the identity, and a
    cartesian command is just a position command on the two joints, which is executed by the
    MuJoCo position actuators managed by :class:`MjxActuatedAdapter`.

    Like the PyBullet adapter it replaces, :meth:`step` is *blocking*: it runs the simulation, in
    chunks of ``control_chunk_sec``, until the end effector reached the commanded position. A step
    therefore has a variable duration, which is what :meth:`BaseAdapter.step` returns. Environments
    using this adapter must tolerate a variable step length (see ``step_precision_tolerance`` in
    :class:`ControlledVecEnv`).

    A movement can also be impossible to complete: the end effector can be held back by whatever it
    is pushing, or be commanded past its joint limits. Waiting for the timeout in those cases would
    make every such step cost the maximum amount of simulation, so the step also ends as soon as
    every environment either reached its target or stopped making progress towards it. Commands are
    additionally clamped to the joint ranges, so that a target outside the reachable area does not
    turn into a permanently unreachable one.

    Note that the convergence check needs the end-effector position on the host at every chunk, so
    each chunk costs a device synchronization, and the loop runs until the *slowest* environment of
    the batch is done.
    """

    def __init__(self,  end_effector_link : tuple[str,str],
                        xjoint : tuple[str,str],
                        yjoint : tuple[str,str],
                        position_tolerance : float = 0.001,
                        blocking_movement : bool = True,
                        max_step_duration_sec : float = 2.0,
                        min_progress_per_chunk : float = 1e-5,
                        control_chunk_sec : float | None = None,
                        clamp_commands_to_joint_limits : bool = True,
                        **kwargs : Any):
        """
        Parameters
        ----------
        end_effector_link : tuple[str,str]
            The link that is moved by the two joints, and whose position is commanded.
        xjoint : tuple[str,str]
            The prismatic joint moving along the world x axis.
        yjoint : tuple[str,str]
            The prismatic joint moving along the world y axis.
        position_tolerance : float
            Position error under which a movement is considered completed, in meters.
        blocking_movement : bool
            If False, step() just runs for the adapter's nominal step length, like a regular adapter.
        max_step_duration_sec : float
            Maximum simulated duration of a blocking step.
        min_progress_per_chunk : float
            An environment whose position error decreased by less than this over a control chunk is
            considered stuck, and stops holding up the step. In meters.
        control_chunk_sec : float | None
            Simulated duration of each chunk of a blocking step. The convergence of the movement is
            checked once per chunk. Must be kept constant across chunks to avoid jit recompilations,
            so it cannot be adapted to the residual error. Defaults to the adapter's step length.
        clamp_commands_to_joint_limits : bool
            Clamp commanded positions to the range of the two joints.
        **kwargs
            Forwarded to :class:`MjxActuatedAdapter`.
        """
        super().__init__(**kwargs)
        self._xjoint = xjoint
        self._yjoint = yjoint
        self._end_effector_link = end_effector_link
        self._position_tolerance = position_tolerance
        self._blocking_movement = blocking_movement
        self._max_step_duration_sec = max_step_duration_sec
        self._min_progress_per_chunk = min_progress_per_chunk
        self._control_chunk_sec = control_chunk_sec if control_chunk_sec is not None else self._step_length_sec
        self._clamp_commands_to_joint_limits = clamp_commands_to_joint_limits
        self._target_position_xy = th.zeros((self.vec_size(), 2),
                                            dtype=th.float32, device=self._out_th_device)
        self._command_was_set = False
        self._move_timeouts_in_last_step = 0
        self._stuck_envs_in_last_step = 0
        self._chunks_in_last_step = 0
        self._ee_mon_ids = None
        self._joint_limits_min_xy : th.Tensor | None = None
        self._joint_limits_max_xy : th.Tensor | None = None

    def _ik(self, positions_xy : th.Tensor) -> th.Tensor:
        """Convert end-effector xy positions to joint positions. The two joints move along the world
        x and y axes, so this is the identity."""
        return positions_xy

    @override
    def build_scenario(self, *args : Any, **kwargs : Any):
        ret = super().build_scenario(*args, **kwargs)
        # This adapter controls exactly these two joints, in this order: own the mapping instead of
        # leaving the command layout up to the environment.
        self.set_impedance_controlled_joints((self._xjoint, self._yjoint))
        self._read_joint_limits()
        return ret

    def _read_joint_limits(self):
        mins = []
        maxs = []
        for joint in (self._xjoint, self._yjoint):
            jid = self._jname2jid[joint]
            if self._mj_model.jnt_limited[jid]:
                jrange = self._mj_model.jnt_range[jid]
                mins.append(float(jrange[0]))
                maxs.append(float(jrange[1]))
            else:
                mins.append(float("-inf"))
                maxs.append(float("+inf"))
        self._joint_limits_min_xy = th.as_tensor(mins, dtype=th.float32, device=self._out_th_device)
        self._joint_limits_max_xy = th.as_tensor(maxs, dtype=th.float32, device=self._out_th_device)
        ggLog.info(f"{type(self).__name__}: end effector range x={mins[0]}..{maxs[0]}, y={mins[1]}..{maxs[1]}")

    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[tuple[str,str]]):
        joint_names = tuple(joint_names)
        if joint_names != (self._xjoint, self._yjoint):
            raise RuntimeError(f"{type(self).__name__} controls exactly ({self._xjoint}, {self._yjoint}), "
                               f"but got {joint_names}")
        return super().set_impedance_controlled_joints(joint_names)

    @override
    def set_monitored_links(self, linksToObserve : Sequence[tuple[str,str]]):
        # The end effector position is needed at every control chunk to check for convergence, so it
        # must always be monitored, whatever the environment asked for.
        links = list(linksToObserve)
        if self._end_effector_link not in links:
            links.append(self._end_effector_link)
        ret = super().set_monitored_links(links)
        self._ee_mon_ids = self.get_monitored_links_ids([self._end_effector_link])
        return ret

    def _get_ee_position_xy(self) -> th.Tensor:
        if self._ee_mon_ids is None:
            raise RuntimeError(f"{type(self).__name__}: set_monitored_links() must be called before using the adapter")
        return self.getLinksState(self._ee_mon_ids)[:, 0, 0:2]

    def _send_position_command(self, positions_xy : th.Tensor, vec_mask : th.Tensor | None = None):
        joint_positions = self._ik(positions_xy)
        # MjxActuatedAdapter only uses the position field of the pvesd command, the gains are the
        # ones of the MuJoCo position actuators (see default_actuator_kp/default_actuator_kv).
        cmd_pvesd = th.zeros((self.vec_size(), 2, 5), dtype=th.float32, device=self._out_th_device)
        cmd_pvesd[:, :, 0] = joint_positions
        self.setJointsImpedanceCommand(joint_impedances_pvesd=cmd_pvesd, vec_mask=vec_mask)

    @override
    def setCartesianPoseCommand(self,   link_names : Sequence[tuple[str,str]],
                                        link_poses_xyz_xyzw : th.Tensor,
                                        vec_mask : th.Tensor | None = None) -> None:
        link_names = tuple(link_names)
        if link_names != (self._end_effector_link,):
            raise AttributeError(f"{type(self).__name__} can only move {self._end_effector_link}, "
                                 f"but got a command for {link_names}")
        target_xy = link_poses_xyz_xyzw[:, 0, 0:2].to(dtype=th.float32, device=self._out_th_device)
        if self._clamp_commands_to_joint_limits and self._joint_limits_min_xy is not None:
            target_xy = th.clamp(target_xy, min=self._joint_limits_min_xy, max=self._joint_limits_max_xy)
        if vec_mask is None:
            self._target_position_xy = target_xy.clone()
        else:
            self._target_position_xy = th.where(vec_mask.unsqueeze(-1), target_xy, self._target_position_xy)
        self._command_was_set = True
        self._send_position_command(self._target_position_xy, vec_mask=vec_mask)

    @override
    def get_cartesian_pose_command(self) -> tuple[Sequence[tuple[str,str]], th.Tensor]:
        poses = th.zeros((self.vec_size(), 1, 7), dtype=th.float32, device=self._out_th_device)
        poses[:, 0, 0:2] = self._target_position_xy
        # z and orientation are not controlled by this adapter, they are reported as zero and identity
        poses[:, 0, 6] = 1.0
        return (self._end_effector_link,), poses

    @override
    def get_cartesian_position_error(self) -> th.Tensor:
        err = th.linalg.vector_norm(self._get_ee_position_xy() - self._target_position_xy, dim=-1)
        return err.unsqueeze(-1)

    def reset_cartesian_command_to_current(self, vec_mask : th.Tensor | None = None) -> None:
        """Set the commanded position to the current end-effector position.

        Meant to be called after the joint state was set directly (i.e. on episode initialization),
        so that the controller does not immediately drag the end effector back to the previous target.
        """
        ee_xy = self._get_ee_position_xy()
        if vec_mask is None:
            self._target_position_xy = ee_xy.clone()
        else:
            self._target_position_xy = th.where(vec_mask.unsqueeze(-1), ee_xy, self._target_position_xy)
        self._command_was_set = True
        self._send_position_command(self._target_position_xy, vec_mask=vec_mask)

    def get_move_timeouts_in_last_step(self) -> int:
        """How many times the blocking step hit its time limit instead of settling."""
        return self._move_timeouts_in_last_step

    def get_stuck_envs_in_last_step(self) -> int:
        """How many environments stopped making progress towards their target during the last step."""
        return self._stuck_envs_in_last_step

    def get_chunks_in_last_step(self) -> int:
        """How many control chunks the last blocking step took."""
        return self._chunks_in_last_step

    @override
    def step(self) -> float:
        if not self._blocking_movement or not self._command_was_set:
            return super().step()
        self.initialize_for_step()
        self._move_timeouts_in_last_step = 0
        self._chunks_in_last_step = 0
        elapsed_sec = 0.0
        previous_error = th.full((self.vec_size(),), float("+inf"),
                                 dtype=th.float32, device=self._out_th_device)
        done = th.zeros((self.vec_size(),), dtype=th.bool, device=self._out_th_device)
        while True:
            elapsed_sec += self.run(self._control_chunk_sec)
            self._chunks_in_last_step += 1
            error = self.get_cartesian_position_error()[:, 0]
            reached = error <= self._position_tolerance
            # An environment that is not getting any closer is not going to: it is either pushing
            # something it cannot move or resting against a joint limit. Stop waiting for it.
            stuck = (previous_error - error) < self._min_progress_per_chunk
            previous_error = error
            done = th.logical_or(reached, stuck)
            if bool(done.all()):
                self._stuck_envs_in_last_step = int(th.logical_and(stuck, th.logical_not(reached)).sum())
                break
            if elapsed_sec >= self._max_step_duration_sec:
                self._move_timeouts_in_last_step = int(th.logical_not(done).sum())
                ggLog.warn(f"{type(self).__name__}: step timed out after {elapsed_sec:.3f}s of simulated time "
                           f"with {self._move_timeouts_in_last_step} environments still moving, "
                           f"maximum position error {float(error.max()):.4f}m "
                           f"(tolerance is {self._position_tolerance:.4f}m)")
                break
        return elapsed_sec

    @override
    def moveToEePoseSync(self,  link_names : Sequence[tuple[str,str]],
                                link_poses_xyz_xyzw : th.Tensor,
                                vec_mask : th.Tensor | None = None,
                                max_error : float = 0.001,
                                sim_timeout_sec : float = 30.0) -> None:
        self.setCartesianPoseCommand(link_names=link_names,
                                     link_poses_xyz_xyzw=link_poses_xyz_xyzw,
                                     vec_mask=vec_mask)
        elapsed_sec = 0.0
        while True:
            elapsed_sec += self.run(self._control_chunk_sec)
            max_err = float(self.get_cartesian_position_error().max())
            if max_err <= max_error:
                break
            if elapsed_sec >= sim_timeout_sec:
                raise TimeoutError(f"{type(self).__name__}.moveToEePoseSync timed out after {elapsed_sec:.3f}s "
                                   f"of simulated time, maximum position error is {max_err:.4f}m")
