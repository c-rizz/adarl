from __future__ import annotations

from adarl.adapters.MjxActuatedAdapter import MjxActuatedAdapter
from adarl.adapters.BaseVecCartesianPositionAdapter import BaseVecCartesianPositionAdapter
from typing import Sequence, Any
from typing_extensions import override
from adarl.utils.dbg.dbg_checks import dbg_check
import adarl.utils.dbg.ggLog as ggLog
import torch as th


class Mjx2DofCartesianAdapter(MjxActuatedAdapter, BaseVecCartesianPositionAdapter):
    """MJX adapter for a 2-DoF planar 'arm' controlled by specifying the end-effector position in the xy plane.

    This is the MJX/vectorized counterpart of :class:`PyBullet2DofCartesianAdapter`. The mechanism is
    assumed to be two prismatic joints whose axes are the world x and y axes, chained so that the
    second one carries the end effector. The inverse kinematics is therefore the identity, and a
    cartesian command is just a position command on the two joints.

    A command is not applied as a step reference: like the PyBullet adapter, it is turned into a
    quintic, velocity- and acceleration-limited position trajectory from the measured position to the
    target, and the reference sent to the MuJoCo position actuators follows that trajectory as the
    simulation advances. One trajectory is computed per axis, each with its own duration, exactly as
    :meth:`PyBulletAdapter.setJointsPositionCommand` does per joint.

    Like the PyBullet adapter, :meth:`step` is *blocking*: it runs the simulation, in chunks of
    ``control_chunk_sec``, until the trajectory is over and the end effector reached the commanded
    position. A step therefore has a variable duration, which is what :meth:`BaseAdapter.step`
    returns. Environments using this adapter must tolerate a variable step length (see
    ``step_precision_tolerance`` in :class:`ControlledVecEnv`).

    A movement can also be impossible to complete: the end effector can be held back by whatever it
    is pushing, or be commanded past its joint limits. Waiting for the timeout in those cases would
    make every such step cost the maximum amount of simulation, so the step also ends as soon as
    every environment either reached its target or, with its trajectory already over, stopped making
    progress towards it. Commands are additionally clamped to the joint ranges, so that a target
    outside the reachable area does not turn into a permanently unreachable one.

    Note that the convergence check needs the end-effector position on the host at every chunk, so
    each chunk costs a device synchronization, and the loop runs until the *slowest* environment of
    the batch is done.
    """

    _MIN_TRAJECTORY_DURATION = 0.001

    def __init__(self,  end_effector_link : tuple[str,str],
                        xjoint : tuple[str,str],
                        yjoint : tuple[str,str],
                        position_tolerance : float = 0.001,
                        blocking_movement : bool = True,
                        max_step_duration_sec : float = 2.0,
                        min_progress_per_chunk : float = 1e-5,
                        sim_dt : float | None = None,
                        clamp_commands_to_joint_limits : bool = True,
                        max_velocity : float = 0.9,
                        max_acceleration : float = 5.0,
                        use_trajectory : bool = True,
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
            An environment whose trajectory is over and whose position error decreased by less than
            this over a control chunk is considered stuck, and stops holding up the step. In meters.
        control_chunk_sec : float | None
            Simulated duration of each chunk of a blocking step. The trajectory reference is advanced
            and the convergence of the movement is checked once per chunk. Must be kept constant
            across chunks to avoid jit recompilations. Defaults to the adapter's step length.
        clamp_commands_to_joint_limits : bool
            Clamp commanded positions to the range of the two joints.
        max_velocity : float
            Maximum end-effector velocity of the commanded trajectory, in m/s, per axis.
        max_acceleration : float
            Maximum end-effector acceleration of the commanded trajectory, in m/s^2, per axis.
        use_trajectory : bool
            If False the target is commanded directly as a step reference, without a trajectory.
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
        self._sim_dt = sim_dt if sim_dt is not None else self._step_length_sec
        self._clamp_commands_to_joint_limits = clamp_commands_to_joint_limits
        self._max_velocity = max_velocity
        self._max_acceleration = max_acceleration
        self._use_trajectory = use_trajectory
        self._target_position_xy = th.zeros((self.vec_size(), 2),
                                            dtype=th.float32, device=self._out_th_device)
        # Quintic trajectory state: one trajectory per environment and axis.
        self._traj_start_xy = th.zeros((self.vec_size(), 2), dtype=th.float32, device=self._out_th_device)
        self._traj_range_xy = th.zeros((self.vec_size(), 2), dtype=th.float32, device=self._out_th_device)
        self._traj_duration_xy = th.full((self.vec_size(), 2), self._MIN_TRAJECTORY_DURATION,
                                         dtype=th.float32, device=self._out_th_device)
        self._traj_elapsed = th.zeros((self.vec_size(),), dtype=th.float32, device=self._out_th_device)
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
        # Only the position field has an effect: the gains are those of the MuJoCo position
        # actuators. They must still be stated in the command though, since MjxActuatedAdapter
        # rejects one whose stiffness/damping differ from the actuators' fixed gains.
        cmd_pvesd = th.zeros((self.vec_size(), 2, 5), dtype=th.float32, device=self._out_th_device)
        cmd_pvesd[:, :, 0] = joint_positions
        cmd_pvesd[:, :, 3] = self._default_actuator_kp
        cmd_pvesd[:, :, 4] = self._default_actuator_kv
        self.setJointsImpedanceCommand(joint_impedances_pvesd=cmd_pvesd, vec_mask=vec_mask)

    def _compute_quintic_duration(self, position_range_xy : th.Tensor) -> th.Tensor:
        """Duration of the quintic that covers ``position_range_xy`` within the velocity and
        acceleration limits, per environment and axis.

        Vectorized counterpart of :func:`adarl.utils.base_utils.compute_quintic`. Note that that one
        takes the square root of a signed range, so it yields NaN (and falls back to the minimum
        duration, i.e. a step reference) for negative displacements; here the magnitude is used, so
        both directions get the same profile.
        """
        abs_range = position_range_xy.abs()
        duration_vel_limited = 15*abs_range/(8*self._max_velocity)
        duration_acc_limited = th.sqrt(10*abs_range)/(3**0.25 * self._max_acceleration**0.5)
        duration = th.maximum(duration_vel_limited, duration_acc_limited)
        return th.clamp(duration, min=self._MIN_TRAJECTORY_DURATION)

    def _trajectory_reference_xy(self) -> th.Tensor:
        """The position reference of the commanded trajectory at the current elapsed time."""
        if not self._use_trajectory:
            return self._target_position_xy
        # quintic_pos: range*t^3*(6t^2 - 15t + 10) + start, with t the normalized time in [0,1]
        t = th.clamp(self._traj_elapsed.unsqueeze(-1)/self._traj_duration_xy, max=1.0)
        return self._traj_range_xy*t*t*t*(6*t*t - 15*t + 10) + self._traj_start_xy

    def _trajectories_are_over(self) -> th.Tensor:
        """(vec_size,) mask of the environments whose commanded trajectory has run its course."""
        if not self._use_trajectory:
            return th.ones((self.vec_size(),), dtype=th.bool, device=self._out_th_device)
        return th.all(self._traj_elapsed.unsqueeze(-1) >= self._traj_duration_xy, dim=-1)

    @override
    def setCartesianPoseCommand(self,   link_names : Sequence[tuple[str,str]],
                                        link_poses_xyz_xyzw : th.Tensor,
                                        vec_mask : th.Tensor | None = None) -> None:
        link_names = tuple(link_names)
        if link_names != (self._end_effector_link,):
            raise AttributeError(f"{type(self).__name__} can only move {self._end_effector_link}, "
                                 f"but got a command for {link_names}")
        target_xy = link_poses_xyz_xyzw[:, 0, 0:2].to(dtype=th.float32, device=self._out_th_device)
        # A non-finite target poisons the trajectory (and then the actuator references) for good,
        # so catch it before it reaches the simulation.
        dbg_check(lambda: th.isfinite(target_xy).all(),
                  lambda: f"Non-finite cartesian pose command: {target_xy}")
        if self._clamp_commands_to_joint_limits and self._joint_limits_min_xy is not None:
            target_xy = th.clamp(target_xy, min=self._joint_limits_min_xy, max=self._joint_limits_max_xy)
        # The trajectory starts from the measured position, as PyBulletAdapter does
        start_xy = self._get_ee_position_xy()
        range_xy = target_xy - start_xy
        duration_xy = self._compute_quintic_duration(range_xy)
        if vec_mask is None:
            self._target_position_xy = target_xy.clone()
            self._traj_start_xy = start_xy.clone()
            self._traj_range_xy = range_xy
            self._traj_duration_xy = duration_xy
            self._traj_elapsed = th.zeros_like(self._traj_elapsed)
        else:
            mask_xy = vec_mask.unsqueeze(-1)
            self._target_position_xy = th.where(mask_xy, target_xy, self._target_position_xy)
            self._traj_start_xy = th.where(mask_xy, start_xy, self._traj_start_xy)
            self._traj_range_xy = th.where(mask_xy, range_xy, self._traj_range_xy)
            self._traj_duration_xy = th.where(mask_xy, duration_xy, self._traj_duration_xy)
            self._traj_elapsed = th.where(vec_mask, th.zeros_like(self._traj_elapsed), self._traj_elapsed)
        self._command_was_set = True
        self._send_position_command(self._trajectory_reference_xy(), vec_mask=vec_mask)

    @override
    def get_cartesian_pose_command(self) -> tuple[Sequence[tuple[str,str]], th.Tensor]:
        poses = th.zeros((self.vec_size(), 1, 7), dtype=th.float32, device=self._out_th_device)
        poses[:, 0, 0:2] = self._target_position_xy
        # z and orientation are not controlled by this adapter, they are reported as zero and identity
        poses[:, 0, 6] = 1.0
        return (self._end_effector_link,), poses

    @override
    def get_cartesian_position_error(self) -> th.Tensor:
        """Distance to the final commanded position, not to the current trajectory reference."""
        err = th.linalg.vector_norm(self._get_ee_position_xy() - self._target_position_xy, dim=-1)
        return err.unsqueeze(-1)

    def reset_cartesian_command_to_current(self, vec_mask : th.Tensor | None = None) -> None:
        """Set the commanded position to the current end-effector position.

        Meant to be called after the joint state was set directly (i.e. on episode initialization),
        so that the controller does not immediately drag the end effector back to the previous target.
        """
        ee_xy = self._get_ee_position_xy()
        poses = th.zeros((self.vec_size(), 1, 7), dtype=th.float32, device=self._out_th_device)
        poses[:, 0, 0:2] = ee_xy
        poses[:, 0, 6] = 1.0
        # Going through setCartesianPoseCommand gives a zero-length trajectory at the current
        # position, which leaves no leftover motion from the previous episode.
        self.setCartesianPoseCommand(link_names=(self._end_effector_link,),
                                     link_poses_xyz_xyzw=poses,
                                     vec_mask=vec_mask)

    def get_move_timeouts_in_last_step(self) -> int:
        """How many times the blocking step hit its time limit instead of settling."""
        return self._move_timeouts_in_last_step

    def get_stuck_envs_in_last_step(self) -> int:
        """How many environments stopped making progress towards their target during the last step."""
        return self._stuck_envs_in_last_step

    def get_chunks_in_last_step(self) -> int:
        """How many control chunks the last blocking step took."""
        return self._chunks_in_last_step

    def _advance_trajectory(self, elapsed_increment : float, vec_mask : th.Tensor | None = None):
        """Advance the trajectory clock and send the resulting position reference."""
        self._traj_elapsed = self._traj_elapsed + elapsed_increment
        self._send_position_command(self._trajectory_reference_xy(), vec_mask=vec_mask)

    @override
    def step(self) -> float:
        if not self._blocking_movement or not self._command_was_set:
            ret = super().step()
            self._advance_trajectory(ret)
            return ret
        self.initialize_for_step()
        self._move_timeouts_in_last_step = 0
        self._stuck_envs_in_last_step = 0
        self._chunks_in_last_step = 0
        elapsed_sec = 0.0
        previous_error = th.full((self.vec_size(),), float("+inf"),
                                 dtype=th.float32, device=self._out_th_device)
        while True:
            chunk_sec = self.run(self._sim_dt)
            elapsed_sec += chunk_sec
            self._chunks_in_last_step += 1
            self._advance_trajectory(chunk_sec)
            error = self.get_cartesian_position_error()[:, 0]
            reached = error <= self._position_tolerance
            # An environment that is not getting any closer is not going to: it is either pushing
            # something it cannot move or resting against a joint limit. Stop waiting for it -- but
            # only once its trajectory is over, since a quintic starts with zero velocity and would
            # otherwise look stuck for the first chunks.
            trajectory_over = self._trajectories_are_over()
            stuck = th.logical_and(trajectory_over,
                                   (previous_error - error) < self._min_progress_per_chunk)
            previous_error = error
            # A movement is only complete once the reference stopped moving too
            done = th.logical_or(th.logical_and(reached, trajectory_over), stuck)
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
            elapsed_sec += self.run(self._sim_dt)
            self._advance_trajectory(self._sim_dt)
            max_err = float(self.get_cartesian_position_error().max())
            if max_err <= max_error and bool(self._trajectories_are_over().all()):
                break
            if elapsed_sec >= sim_timeout_sec:
                raise TimeoutError(f"{type(self).__name__}.moveToEePoseSync timed out after {elapsed_sec:.3f}s "
                                   f"of simulated time, maximum position error is {max_err:.4f}m")
