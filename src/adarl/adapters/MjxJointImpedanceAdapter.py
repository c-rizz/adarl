from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

from adarl.adapters.MjxAdapter import (
    MjxAdapter,
    MjxCommandBatch,
    jax2th,
    th2jax,
    SimState,
    SimConf,
    StaticSimConf,
    PublicCommand,
    _InternalCommand,
)
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.utils.utils import to_string_tensor, masked_assign
from typing import Any, Literal, Mapping
import jax
from typing_extensions import override
from typing import overload, Sequence
import torch as th
import jax.numpy as jnp
from functools import partial
import jax.tree_util
from dataclasses import dataclass
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.base_utils import record_time, print_recorded_times, record_region_end, record_region_start
import numpy as np
from adarl.utils.dbg.dbg_checks import dbg_check

@jax.jit
@partial(jax.vmap, in_axes=(0, 0,    0), out_axes=(0, 0)) #vectorize along the number of simulations
@partial(jax.vmap, in_axes=(0, None, 0), out_axes=(0, 0)) #vectorize along the number of references (pos,vel,torque)
@partial(jax.vmap, in_axes=(0, None, 0), out_axes=(0, 0)) #vectorize along the number of joints
def _second_order_filter(u, filter_coeffs, filter_state):
    """Applies a second order filter to the input signal u.
    
    From xbot2:
    This is a second order filter with transfer function
                        1
    P(s) =  -----------------------,  w = natural frequency, eps = damping ratio
            (s/w)^2 + 2*eps/w*s + 1
    
    and discretized according to a trapezoidal (aka Tustin) scheme. This yields
    a difference equation of the following form:
    
        a0*y + a1*yd + a2*ydd = u + b1*ud + b2*udd
        i.e.:
        y = u/a0 + b1/a0*ud + b2/a0*udd - a1/a0*yd - a2/a0*ydd 
    
    where yd = y(k-1), ydd = y(k-2) and so on (d = delayed).

    Parameters
    ----------
    u : jnp.ndarray
        Input signal to be filtered.
    filter_coeffs : jnp.ndarray
        Coefficients of the filter.
    filter_state : jnp.ndarray
        Current state of the filter.
    
    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Filtered output signal and new filter state
    """
    # print(f"in u.shape = {u.shape}, filter_coeffs.shape = {filter_coeffs.shape}, filter_state.shape = {filter_state.shape}")
    # at this point the state is [ u_prev, u_prev2, u_prev3, y_prev, y_prev2]
    new_filter_state = filter_state.at[1:3].set(filter_state[0:2])  # Shift u state
    new_filter_state = new_filter_state.at[0].set(u)  # Update the first state with the new input
    # at this point the state is [ u, u_prev, u_prev2, y_prev, y_prev2]
    y = new_filter_state @ filter_coeffs    
    new_filter_state = new_filter_state.at[4].set(new_filter_state[3])  # Shift the y state
    new_filter_state = new_filter_state.at[3].set(y)  # Update the last state with the output
    # at this point the state is [ u, u_prev, u_prev2, y, y_prev]
    # print(f"out u.shape = {u.shape}, filter_coeffs.shape = {filter_coeffs.shape}, filter_state.shape = {filter_state.shape}")
    return y, new_filter_state

@jax.jit
@partial(jax.vmap, in_axes=(None, 0,    0), out_axes=(0,    0)) #vectorize along the number of simulations
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=(None, 0)) #vectorize along the number of joints
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=(None, 0)) #vectorize along the number of references (pos,vel,torque)
def _compute_filter_coeffs_and_state(dt, cutoff_freq, initial_value, eps=1.0):
    """Computes the coefficients and initial state for a second order filter.

    The filter is designed to have a cutoff frequency of `cutoff_freq` and a damping ratio of `eps`.
    The filter is discretized using the trapezoidal (Tustin) method.
    
    Parameters
    ----------
    dt : float
        Time step for the discretization.
    cutoff_freq : float
        Cutoff frequency for the filter.
    initial_value : float
        Initial value for the filter state.
    eps : float
        Damping ratio for the filter.

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Filter coefficients and initial filter state.
    """
    omega = 2*jnp.pi*cutoff_freq
    b1 = 2.0
    b2 = 1.0

    a0 = 1.0 + 4.0*eps/(omega*dt) + 4.0/jnp.power(omega*dt, 2.0)
    a1 = 2 - 8.0/jnp.power(omega*dt, 2.0)
    a2 = 1.0 + 4.0/jnp.power(omega*dt, 2.0) - 4.0*eps/(omega*dt)
    return jnp.array([1/a0, b1/a0, b2/a0, -a1/a0, -a2/a0], dtype=jnp.float32), jnp.full(fill_value=initial_value, shape=(5,), dtype=jnp.float32)



@jax.tree_util.register_dataclass
@dataclass
class SimStateJimp(SimState):
    cmds_queue : jnp.ndarray
    cmds_queue_times : jnp.ndarray
    vec_impjoints_pveaecpvesde : jnp.ndarray
    filtered_pv_references : jnp.ndarray
    filtered_pve_states : jnp.ndarray
    ref_filter_coeffs : jnp.ndarray
    ref_filter_state : jnp.ndarray

    def replace_d(self, name_values : dict[str,Any]):
        # ggLog.info(f"rd0 type(self.mjx_data) = {type(self.mjx_data)}")
        # d = dataclasses.asdict(self) # Recurses into dataclesses and deepcopies
        d = {"mjx_data" : self.mjx_data,
             "mjx_model" : self.mjx_model,
             "requested_qfrc_applied" : self.requested_qfrc_applied,
             "sim_time" : self.sim_time,
             "cmds_queue" : self.cmds_queue,
             "cmds_queue_times" : self.cmds_queue_times,
             "stats_step_count" : self.stats_step_count,
             "mon_joint_stats_arr_pvaeep" : self.mon_joint_stats_arr_pvaeep,
             "vec_impjoints_pveaecpvesde" : self.vec_impjoints_pveaecpvesde,
             "filtered_pv_references" : self.filtered_pv_references,
             "filtered_pve_states" : self.filtered_pve_states,
             "impulse_startends_stime" : self.impulse_startends_stime,
             "impulses_xfrc" : self.impulses_xfrc,
             "ref_filter_coeffs" : self.ref_filter_coeffs,
             "ref_filter_state" : self.ref_filter_state,
             "mon_links_stats_arr_v" : self.mon_links_stats_arr_v,
             "mon_joint_state_pveae" : self.mon_joint_state_pveae,
             "mon_link_state" : self.mon_link_state,
             "mon_link_acceleration" : self.mon_link_acceleration,
             "mon_collision_mask" : self.mon_collision_mask
            }
        # ggLog.info(f"d0 = "+str({k:type(v) for k,v in d.items()}))
        d.update(name_values)
        # ggLog.info(f"d1 = "+str({k:type(v) for k,v in d.items()}))
        ret = SimStateJimp(**d)
        # ggLog.info(f"type(self.mjx_data) = {type(self.mjx_data)}")
        return ret
    
    # def __setattr__(self, name: str, value: Any) -> None:
    #     if name == "mjx_data":
    #         ggLog.info(f"setting mjx_data to {type(value)}")
    #         traceback.print_stack()
    #     return super().__setattr__(name, value)

@jax.tree_util.register_dataclass
@dataclass
class SimConfJimp(SimConf):
    imp_control_jids : jnp.ndarray
    use_second_order_reference_filter : bool
    use_exponential_reference_filter: bool
    ref_filter_cutoff_freqs : jnp.ndarray
    pv_ref_filter_alpha : jnp.ndarray

    def replace_d(self, name_values : dict[str,Any]):
        # ggLog.info(f"rd0 type(self.mjx_data) = {type(self.mjx_data)}")
        # d = dataclasses.asdict(self) # Recurses into dataclesses and deepcopies
        d = {"imp_control_jids" : self.imp_control_jids,
            "use_second_order_reference_filter" : self.use_second_order_reference_filter,
            "use_exponential_reference_filter" : self.use_exponential_reference_filter,
            "ref_filter_cutoff_freqs" : self.ref_filter_cutoff_freqs,
            "pv_ref_filter_alpha" : self.pv_ref_filter_alpha,
            "monitored_qpadr" : self.monitored_qpadr,
            "monitored_qvadr" : self.monitored_qvadr,
            "monitored_lids" : self.monitored_lids,
            "monitored_jids" : self.monitored_jids,
            "body_rootid" : self.body_rootid,
            "monitored_collision_pairs" : self.monitored_collision_pairs,
            "geom_bodyid" : self.geom_bodyid,
            "sim_dt" : self.sim_dt,
            "jnt_qposadr" : self.jnt_qposadr,
            "jnt_dofadr" : self.jnt_dofadr,
            "monitored_sids" : self.monitored_sids,
            "site_bodyid" : self.site_bodyid
            }
        d.update(name_values)
        ret = SimConfJimp(**d)
        return ret


@jax.tree_util.register_dataclass
@dataclass
class MjxJointImpedanceCommandBatch(MjxCommandBatch):
    current_joint_impedance_command_pvesd : jnp.ndarray


@dataclass(frozen=True)
class SetCurrentJointImpedanceCommand(PublicCommand):
    joint_impedances_pvesd : th.Tensor
    vec_mask : th.Tensor | None = None
    joint_names : Sequence[tuple[str, str]] | None = None

    def build_internal_command(self, adapter : MjxJointImpedanceAdapter) -> _InternalCommand:
        if self.joint_names is not None:
            raise RuntimeError("joint_names is not supported, must be None (controls all impedance_controlled_joints)")
        expected_size = (adapter._vec_size, adapter._sim_conf.imp_control_jids.shape[0], 5)
        if self.joint_impedances_pvesd.size() != expected_size:
            raise RuntimeError(f"joint_impedances_pvesd should have size {expected_size}, but it's {self.joint_impedances_pvesd.size()}")
        return _InternalSetCurrentJointImpedanceCommand(
            vec_mask=adapter._vec_mask_to_jax(self.vec_mask),
            current_joint_impedance_command_pvesd=th2jax(self.joint_impedances_pvesd, adapter._jax_device),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _InternalSetCurrentJointImpedanceCommand(_InternalCommand):
    vec_mask : jnp.ndarray
    current_joint_impedance_command_pvesd : jnp.ndarray

    def has_effect(self) -> bool:
        return self.current_joint_impedance_command_pvesd.shape[1] > 0

    def marks_forward_needed(self) -> bool:
        return False

    def run_jax(self, adapter : MjxJointImpedanceAdapter, sim_state : SimStateJimp, sim_conf : SimConfJimp, static_sim_conf : StaticSimConf) -> SimStateJimp:
        if not self.has_effect():
            return sim_state
        cmd_delay = jnp.where(
            self.vec_mask,
            jnp.full_like(sim_state.sim_time, -1000.0),
            jnp.full_like(sim_state.sim_time, float("+inf")),
        )
        new_queue, new_queue_times, _inserted = adapter._add_impedance_command_jax(
            sim_time=sim_state.sim_time,
            cmds_queue=sim_state.cmds_queue,
            cmds_queue_times=sim_state.cmds_queue_times,
            cmd_joint_pvesd=self.current_joint_impedance_command_pvesd,
            cmd_delay=cmd_delay,
        )
        sim_state = sim_state.replace_d({"cmds_queue_times" : new_queue_times, "cmds_queue" : new_queue})
        return sim_state

class MjxJointImpedanceAdapter(MjxAdapter, BaseVecJointImpedanceAdapter):
    
    def __init__(self, vec_size : int,
                        enable_rendering : bool,
                        jax_device : jax.Device,
                        output_th_device : th.device,
                        sim_step_dt : float = 2/1024,
                        step_length_sec : float = 10/1024,
                        realtime_factor : float | None = None,
                        show_gui : bool = False,
                        gui_frequency : float = 15,
                        gui_env_index : int = 0,
                        default_max_joint_impedance_ctrl_torque : float = 100.0,
                        max_joint_impedance_ctrl_torques : dict[tuple[str,str],float] = {},
                        add_ground : bool = True,
                        add_sky : bool = True,
                        impedance_commands_queue_size : int = 10,
                        log_freq : int = -1,
                        opt_preset : Literal["fast","none"] | str = "fast",
                        record_whole_joint_trajectories : bool = False,
                        log_freq_joints_trajectories = 1000,
                        log_folder="./",
                        safe_revolute_dof_armature = 0.01,
                        revolute_dof_armature_override = None,
                        opt_override : dict[str,Any] | None = None,
                        reference_filter_cutoff_frequency : float = 20.0,
                        reference_filter_mode :  str = "second_order"):
        super().__init__(vec_size=vec_size,
                        enable_rendering = enable_rendering,
                        jax_device = jax_device,
                        sim_step_dt = sim_step_dt,
                        step_length_sec = step_length_sec,
                        realtime_factor = realtime_factor,
                        show_gui = show_gui,
                        gui_frequency = gui_frequency,
                        gui_env_index = gui_env_index,
                        output_th_device=output_th_device,
                        add_ground=add_ground,
                        add_sky = add_sky,
                        log_freq=log_freq,
                        record_whole_joint_trajectories=record_whole_joint_trajectories,
                        log_freq_joints_trajectories=log_freq_joints_trajectories,
                        log_folder=log_folder,
                        safe_revolute_dof_armature=safe_revolute_dof_armature,
                        revolute_dof_armature_override=revolute_dof_armature_override,
                        opt_preset=opt_preset,
                        opt_override=opt_override)
        self._sim_state = SimStateJimp( mjx_data=self._sim_state.mjx_data,
                                        requested_qfrc_applied=self._sim_state.requested_qfrc_applied,
                                        sim_time=self._sim_state.sim_time,
                                        mjx_model=self._sim_state.mjx_model,
                                        cmds_queue=jnp.empty((0,), device = jax_device),
                                        cmds_queue_times=jnp.empty((0,), device = jax_device),
                                        stats_step_count=jnp.zeros((1,), device = jax_device),
                                        mon_joint_stats_arr_pvaeep=jnp.empty((vec_size,0,6), device = jax_device),
                                        vec_impjoints_pveaecpvesde=jnp.empty((vec_size,0,12), device = jax_device),
                                        filtered_pv_references=jnp.empty((vec_size,0,2), device = jax_device),
                                        filtered_pve_states = jnp.empty((vec_size,0,3), device = jax_device),
                                        impulse_startends_stime=jnp.empty((0,), device = jax_device),
                                        impulses_xfrc=jnp.empty((0,), device = jax_device),
                                        ref_filter_coeffs=jnp.empty((vec_size,0,5), device = jax_device),
                                        ref_filter_state=jnp.zeros((vec_size,0,5), device = jax_device),
                                        mon_links_stats_arr_v=jnp.empty((0,), device = jax_device),
                                        mon_joint_state_pveae=self._sim_state.mon_joint_state_pveae,
                                        mon_link_state=self._sim_state.mon_link_state,
                                        mon_link_acceleration=self._sim_state.mon_link_acceleration,
                                        mon_collision_mask=self._sim_state.mon_collision_mask)
        # Reference filter
        if reference_filter_mode == "second_order":
            self._use_second_order_reference_filter = True
            self._use_exponential_reference_filter = False
        elif reference_filter_mode == "exponential":
            self._use_exponential_reference_filter = True
            self._use_second_order_reference_filter = False
        elif reference_filter_mode == "none":
            self._use_second_order_reference_filter = False
            self._use_exponential_reference_filter = False
        else:
            raise RuntimeError(f"Unknown reference filter mode '{reference_filter_mode}'")
        self._ref_filter_cutoff_freqs_th = th.as_tensor(reference_filter_cutoff_frequency).expand(self.vec_size()).clone()
        pv_ref_filter_decimation_time = 0.05 # 90% of the filtered value comes from this duration
        pv_ref_filter_alpha = 0.1**(1/(pv_ref_filter_decimation_time/self._sim_step_dt))
        self._sim_conf = SimConfJimp(   monitored_qpadr=self._sim_conf.monitored_qpadr,
                                        monitored_qvadr=self._sim_conf.monitored_qvadr,
                                        monitored_jids=self._sim_conf.monitored_jids,
                                        monitored_lids=self._sim_conf.monitored_lids,
                                        body_rootid=self._sim_conf.body_rootid,
                                        monitored_collision_pairs=self._sim_conf.monitored_collision_pairs,
                                        geom_bodyid=self._sim_conf.geom_bodyid,
                                        sim_dt=self._sim_conf.sim_dt,
                                        jnt_qposadr=self._sim_conf.jnt_qposadr,
                                        jnt_dofadr=self._sim_conf.jnt_dofadr,
                                        monitored_sids=self._sim_conf.monitored_sids,
                                        site_bodyid=self._sim_conf.site_bodyid,
                                        imp_control_jids=jnp.empty((0,), dtype=jnp.int32, device=self._jax_device),
                                        use_second_order_reference_filter=self._use_second_order_reference_filter,
                                        use_exponential_reference_filter=self._use_exponential_reference_filter,
                                        ref_filter_cutoff_freqs=th2jax(self._ref_filter_cutoff_freqs_th, self._jax_device),
                                        pv_ref_filter_alpha = pv_ref_filter_alpha
                                        )
        # Controlled joint state filter (Only used for the impedance control feedback, not by getJointState)
        pve_sensing_filter_decimation_time = 0.005        
        self._pve_sensing_filter_alpha = 0.1**(1/pve_sensing_filter_decimation_time/self._sim_step_dt)
        

        self._queue_size = impedance_commands_queue_size
        self._max_joint_impedance_ctrl_torques = max_joint_impedance_ctrl_torques
        self._default_max_joint_impedance_ctrl_torque = default_max_joint_impedance_ctrl_torque
        self.set_impedance_controlled_joints([]) # initialize attributes
        self._insert_cmd_to_queue_vec = jax.vmap(jax.jit(self._insert_cmd_to_queue, donate_argnames=["cmds_queue","cmds_queue_times"]))
        self._get_cmd_and_cleanup_vec = jax.vmap(jax.jit(self._get_cmd_and_cleanup, donate_argnames=["cmds_queue","cmds_queue_times"]), in_axes=(0,0,None))
        self._compute_impedance_torques_vec = jax.vmap(jax.jit(self._compute_impedance_torques), in_axes=(0,0,None))

    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None,
                                        joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        delay_sec = th.as_tensor(delay_sec).to(self._out_th_device, non_blocking=True).clamp(min=0)
        # if th.any(delay_sec < 0):
        #     raise RuntimeError(f"Cannot have a negative command delay") # actually we could, but it would mess up the logic of set_current_joint_impedance_command
        if joint_names is not None:
            raise RuntimeError(f"joint_names is not supported, must be None (controls all impedance_controlled_joints)")
        if vec_mask is not None:
            delay_sec = th.where(vec_mask, delay_sec, float("+inf"))
        # ggLog.info(f"Adding joint impedance command {joint_impedances_pvesd}")
        self._add_impedance_command(joint_impedances_pvesd=joint_impedances_pvesd,
                                    delay_sec=delay_sec)

    def reset_joint_impedances_commands(self):
        self._reset_cmd_queue()

    def set_current_joint_impedance_command(self,   joint_impedances_pvesd : th.Tensor,
                                                    joint_names : Sequence[tuple[str,str]] | None = None,
                                        vec_mask : th.Tensor | None = None) -> None:
        
        """ Applies a joint impedance command immediately.
            Meant to be used outside of the normal control loop, just for resetting/initializing.

        Parameters
        ----------
        joint_impedances_pvesd : th.Tensor
            Shape (vec_size, len(impedance_controlled_joints), 5)
        joint_names : Sequence[tuple[str,str]] | None, optional
            _description_, by default None

        Raises
        ------
        RuntimeError
            _description_
        """
        # The queue always contains commands that are in the future, i.e. with time greater or 
        # equal to the current time. Except for the initial command which is placed at time -inf.
        # To have this new command be applied as soon as possible we put it in the past, where
        # it should have already been applied, so at the next loop it gets applied immediately.
        # So we want this command to be placed after -inf and before the current time, so with 
        # a sizable negative delay.
        if joint_names is not None:
            raise RuntimeError(f"joint_names is not supported, must be None (controls all impedance_controlled_joints)")
        if vec_mask is not None:
            delay = th.where(vec_mask, -1000.0, float("+inf"))
        else:
            delay = th.as_tensor(-1000.0).to(self._out_th_device, non_blocking=True)
            # th._assert_async(th.all(vec_mask),f"setJointsImpedanceCommand: vec_mask is not supported, must be None (controls all simulations)")
            # This could probably be implemented fairly easily
        # ggLog.info(f"Setting jimp command {joint_impedances_pvesd}")
        self._add_impedance_command(joint_impedances_pvesd=joint_impedances_pvesd,
                                    delay_sec=delay)
        
    @partial(jax.jit, donate_argnames=("cmds_queue","cmds_queue_times"), static_argnames=("self"))
    def _add_impedance_command_jax(self, sim_time : jnp.ndarray,
                                         cmds_queue : jnp.ndarray,
                                         cmds_queue_times : jnp.ndarray,
                                         cmd_joint_pvesd : jnp.ndarray,
                                         cmd_delay : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # sim_time=sim_state.sim_time
        # cmds_queue=sim_state.cmds_queue,
        # cmds_queue_times=sim_state.cmds_queue_times
        new_cmds_queue, new_cmds_queue_times, inserted = self._insert_cmd_to_queue_vec( cmd=cmd_joint_pvesd,
                                                                                        cmd_time=sim_time + cmd_delay,
                                                                                        cmds_queue=cmds_queue,
                                                                                        cmds_queue_times=cmds_queue_times)
        return new_cmds_queue, new_cmds_queue_times, inserted
        # return sim_state.replace_d({"cmds_queue" : new_cmds_queue, 
        #                             "cmds_queue_times" : new_cmds_queue_times}), inserted
        

    def _add_impedance_command(self,    joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor = 0.0) -> None:
        # No support for having commands that don't contain all joints
        record_region_start("MjxJointImpedanceAdapter.add_impedance_command")
        joint_impedances_pvesd_jax = th2jax(joint_impedances_pvesd, self._jax_device)
        cmd_delay_jax = th2jax(delay_sec.expand(self._vec_size), self._jax_device)
        record_time("MjxJointImpedanceAdapter.add_impedance_command: th->jax done")
        # jids = self._sim_conf.imp_control_jids

        # # Create a command, commands are always of the size of _imp_control_jids
        # cmd = jnp.zeros(shape=(self._vec_size, len(jids), 5), dtype=jnp.float32, device=self._jax_device)
        # # The joints that are actually being commanded are indicated this boolean tensor
        # cmd_idxs = self._jids_to_imp_cmd_idx_np[jids]
        # if np.any(cmd_idxs < 0):
        #     raise RuntimeError(f"Tried to set impedance command for joint that has not been set with set_impedance_controlled_joints")

        # cmd = cmd.at[:,cmd_idxs].set(joint_impedances_pvesd_jax)
        # So now we have a properly formulated command in cmd and cmd_time
        new_cmds, new_cmds_times, inserted = self._add_impedance_command_jax(self._sim_state.sim_time,
                                                                             self._sim_state.cmds_queue,
                                                                             self._sim_state.cmds_queue_times,
                                                                             joint_impedances_pvesd_jax, cmd_delay_jax)
        self._sim_state.cmds_queue = new_cmds
        self._sim_state.cmds_queue_times = new_cmds_times
        record_time("MjxJointImpedanceAdapter.add_impedance_command: update sim state done")
        inserted = jax2th(inserted, self._out_th_device)
        dbg_check(lambda : th.all(inserted), 
                  lambda : f"Failed to insert command, inserted = {inserted}",
                  async_assert=True)
        # ggLog.info(f"added cmd: times = {self._sim_state.cmds_queue_times} \n cmds = {self._sim_state.cmds_queue}")
        record_region_end("MjxJointImpedanceAdapter.add_impedance_command")

    @staticmethod
    def _insert_cmd_to_queue(   cmd : jnp.ndarray,
                                cmd_time : jnp.ndarray,
                                cmds_queue : jnp.ndarray,
                                cmds_queue_times : jnp.ndarray):
        """_summary_

        Parameters
        ----------
        cmd : jnp.ndarray
            shape: (imp_joints_num, 5)
        cmd_time : jnp.ndarray
            shape: (,)
        cmds_queue : jnp.ndarray
            shape (queue_len, imp_joints_num, 5)
        cmds_queue_times : jnp.ndarray
            shape (queue_len,)

        Returns
        -------
        _type_
            _description_
        """
        # insert command in the first slot that has +inf time
        # if there's no space return some specific value in a ndarray
        discard_command = jnp.isposinf(cmd_time) # if the command time is +inf we discard it
        empty_slots = jnp.isinf(cmds_queue_times) # if no command or a command for the same time
        same_time_slots = cmds_queue_times==cmd_time
        # If there is any slot at the same time, then do not use the empty ones
        # If there is a slot at the same time, use that one
        # Thi smounts to the following: (empty and not any(same)) or same
        writable_slots = jnp.logical_or(jnp.logical_and(empty_slots,jnp.logical_not(jnp.any(same_time_slots))), same_time_slots)
        found_slot = jnp.any(writable_slots)
        first_empty = jnp.argmax(writable_slots) # if there is no empty slot, then this is zero, but in that case we end up not selecting it, becuase we use 'found_slot'
        # selected_slots = jnp.zeros_like(cmds_queue_times) # Initialize all False
        # selected_slots = selected_slots.at[first_empty].set(found_slot) # if a slot was found, set its corresponding cell to True, the rest to False

        do_write = jnp.logical_and(found_slot, jnp.logical_not(discard_command))
        cmds_queue_times = cmds_queue_times.at[first_empty].set(jnp.where(do_write, cmd_time, cmds_queue_times[first_empty]))
        cmds_queue = cmds_queue.at[first_empty].set(jnp.where(do_write, cmd, cmds_queue[first_empty]))

        return cmds_queue, cmds_queue_times, found_slot

    @staticmethod
    def _get_cmd_and_cleanup(   cmds_queue : jnp.ndarray,
                                cmds_queue_times : jnp.ndarray,
                                current_time : jnp.ndarray):
        past_cmds_queue_mask = cmds_queue_times <= current_time
        has_cmd = jnp.any(past_cmds_queue_mask)
        current_cmd_idx = jnp.argmax(jnp.where(past_cmds_queue_mask, cmds_queue_times, float("-inf")))
        current_cmd = cmds_queue[current_cmd_idx]
        cmds_queue_to_remove_mask = past_cmds_queue_mask.at[current_cmd_idx].set(False)
        cmds_queue_times = jnp.where(cmds_queue_to_remove_mask, float("+inf"), cmds_queue_times) # mark commands as removed by setting the time to +inf
        
        return current_cmd, has_cmd, cmds_queue, cmds_queue_times


    
    def set_impedance_controlled_joints(self, joint_names : Sequence[tuple[str,str]]):
        """Set the joints that will be controlled by the adapter

        Parameters
        ----------
        joint_names : Sequence[Tuple[str,str]]
            List of the controlled joint names

        """
        self._imp_controlled_joint_names = tuple(joint_names)
        imp_control_jids = [self._jname2jid[jn] for jn in joint_names]
        imp_control_jids_np = np.array(imp_control_jids)
        self._imp_control_max_torque = jnp.array([self._max_joint_impedance_ctrl_torques.get(jn, self._default_max_joint_impedance_ctrl_torque)
                                                    for jn in self._imp_controlled_joint_names], device=self._jax_device)
        imp_control_jids_jax = jnp.array(imp_control_jids, device=self._jax_device)
        self._sim_conf = self._sim_conf.replace_d({"imp_control_jids" : imp_control_jids_jax})
        if len(imp_control_jids) != 0:
            self._jids_to_imp_cmd_qpadr = self._sim_conf.jnt_qposadr[imp_control_jids_np]
            self._jids_to_imp_cmd_qvadr = self._sim_conf.jnt_dofadr[imp_control_jids_np]
            self._jids_to_imp_cmd_idx_np = np.array([imp_control_jids.index(i) if i in imp_control_jids else -1 
                                                   for i in range(max(imp_control_jids)+1)])
            # self._jids_to_imp_cmd_idx[i] tells at which index to put the command for joint i when forming an impedance command
        else:
            self._jids_to_imp_cmd_qpadr = jnp.empty_like(imp_control_jids_jax)
            self._jids_to_imp_cmd_qvadr = jnp.empty_like(imp_control_jids_jax)
            self._jids_to_imp_cmd_idx_np   = np.empty_like(imp_control_jids_np)
        self._reset_cmd_queue()
        self._reset_filters()
        if self._record_joint_hist:
            self._sim_state = self._sim_state.replace_v("vec_impjoints_pveaecpvesde",
                                                        jnp.zeros(shape=(self._vec_size, len(self._imp_controlled_joint_names),12),
                                                                    dtype=jnp.float32,
                                                                    device=self._jax_device))
            self._full_history_labels = to_string_tensor(sum([[f"{v}.{jn[1]}" for v in ["pos","vel","cmd_eff","acc","eff","constr_eff","pref","vref","eref","stiff","damp","neweff"]] 
                                                              for jn in self._imp_controlled_joint_names],[])).unsqueeze(0)
            

    def _reset_cmd_queue(self):
        self._queued_cmds_queue = 1
        self._sim_state = self._sim_state.replace_d({"cmds_queue" : jnp.zeros(shape=(self._vec_size, self._queue_size, len(self._sim_conf.imp_control_jids), 5), dtype=jnp.float32, device=self._jax_device),
                                                     "cmds_queue_times" : jnp.full(fill_value=float("+inf"), shape=(self._vec_size, self._queue_size), dtype=jnp.float32, device=self._jax_device)})
        
    @partial(jax.jit, donate_argnames=("sim_state",), static_argnames=("self","reset_state"))
    def _reset_filters_jax(self, sim_state : SimStateJimp , sim_conf : SimConfJimp, reset_state : bool = True):
        
        current_pve = MjxJointImpedanceAdapter._get_vec_joint_states_pveae(sim_conf, sim_state.mjx_data, sim_conf.imp_control_jids)[...,:3]
        state_repl = {  "filtered_pve_states" : current_pve}
        # For the reference filter, use commanded references from the queue if available,
        # otherwise fall back to measured positions.
        current_cmd, has_cmd, _, _ = self._get_cmd_and_cleanup_vec(sim_state.cmds_queue,
                                                                   sim_state.cmds_queue_times,
                                                                   sim_state.sim_time)
        ref_pve = jnp.where(has_cmd[:, None, None], current_cmd[:, :, :3], current_pve)
        ref_pv = ref_pve[:, :, :2]
        vec_ref_filter_coeffs, ref_filter_state = _compute_filter_coeffs_and_state( sim_conf.sim_dt,
                                                                                    sim_conf.ref_filter_cutoff_freqs,
                                                                                    ref_pve)
        state_repl["ref_filter_coeffs"] = vec_ref_filter_coeffs
        if reset_state:
            state_repl["ref_filter_state"] = ref_filter_state
            state_repl["filtered_pv_references"] = ref_pv
        # ggLog.info(f"resetted filter to coeffs {vec_ref_filter_coeffs} and state {ref_filter_state}")
        # ggLog.info(f"resetted refs filter")
        sim_state = sim_state.replace_d(state_repl)
        return sim_state

    def _reset_filters(self, reset_state : bool = True):
        if len(self._imp_controlled_joint_names)>0:
            # ggLog.info(f"type(mjx_model.jnt_qposadr) = {type(self._sim_state.mjx_model.jnt_qposadr)}")
            self._sim_state = self._reset_filters_jax(self._sim_state, self._sim_conf, reset_state)
        return 
        # sim_conf = self._sim_conf
        # sim_state = self._sim_state
        # if len(self._imp_controlled_joint_names)>0:
        #     current_pve = MjxJointImpedanceAdapter._get_vec_joint_states_pveae(sim_state.mjx_model, sim_state.mjx_data, self._imp_control_jids)[...,:3]
        #     current_pv = current_pve[:,:,:2]
        # else:
        #     current_pve = jnp.zeros(shape=(sim_conf.vec_size, len(self._imp_controlled_joint_names),3),
        #                             dtype=jnp.float32,
        #                             device=self._jax_device)
        #     current_pv = current_pve[:,:,:2]
        # # print(f"current_pve.shape = {current_pve.shape}")
        # # print(f"_ref_filter_cutoff_freqs.shape = {self._ref_filter_cutoff_freqs.shape}")
        # state_repl = {  "filtered_pve_states" : current_pve}
        # if self._use_second_order_reference_filter:
        #     vec_ref_filter_coeffs, ref_filter_state = _compute_filter_coeffs_and_state(th2jax(self.sim_step_duration(), self._jax_device),
        #                                                                             self._ref_filter_cutoff_freqs,
        #                                                                             current_pve)
        #     # expected_coeff_shape = (sim_conf.vec_size, 5)
        #     # if vec_ref_filter_coeffs.shape != expected_coeff_shape:
        #     #     raise RuntimeError(f"ref_filter_coeffs shape {vec_ref_filter_coeffs.shape} does not match expected shape {expected_coeff_shape}")
        #     # expected_state_shape = (sim_conf.vec_size, len(self._imp_controlled_joint_names), 3, 5)
        #     # if reset_state and ref_filter_state.shape != expected_state_shape:
        #     #     raise RuntimeError(f"ref_filter_state shape {ref_filter_state.shape} does not match expected shape {expected_state_shape}")
        #     state_repl["ref_filter_coeffs"] = vec_ref_filter_coeffs
        #     if reset_state:
        #         state_repl["ref_filter_state"] = ref_filter_state
        # elif self._use_exponential_reference_filter:
        #     if reset_state:
        #         state_repl["filtered_pv_references"] = current_pv
        # # ggLog.info(f"resetted filter to coeffs {vec_ref_filter_coeffs} and state {ref_filter_state}")
        # # ggLog.info(f"resetted refs filter")
        # self._sim_state = sim_state.replace_d(state_repl)
        
    def set_reference_filter(self, reference_filter_cutoff_frequency : th.Tensor, vec_mask : th.Tensor | None = None):
        """Set the parameters of the filter applied to the command references

        Parameters
        ----------
        reference_filter_cutoff_frequency : float
            Cutoff frequency of the filter in Hz

        """
        if vec_mask is not None:
            th.where(vec_mask, self._ref_filter_cutoff_freqs_th, reference_filter_cutoff_frequency, out=self._ref_filter_cutoff_freqs_th)
        else:
            self._ref_filter_cutoff_freqs_th = reference_filter_cutoff_frequency.expand(self.vec_size())
        ref_filter_cutoff_freqs = th2jax(self._ref_filter_cutoff_freqs_th, self._jax_device)
        self._sim_conf = self._sim_conf.replace_d({"ref_filter_cutoff_freqs" : ref_filter_cutoff_freqs})
        self._reset_filters_jax(reset_state=False)
        
    def get_impedance_controlled_joints(self) -> tuple[tuple[str,str],...]:
        """Get the names of the joints that are controlled by this adapter

        Returns
        -------
        tuple[tuple[str,str]]
            The list of the joint names

        """
        return self._imp_controlled_joint_names


    @staticmethod
    def _compute_impedance_torques(cmd_j_pvesd : jnp.ndarray, 
                                   state_j_pve : jnp.ndarray,
                                   max_joint_efforts : jnp.ndarray):
        """_summary_

        Parameters
        ----------
        cmd_j_pvesd : jnp.ndarray
            shape (len(self_imp_controlled_jids), 5)
        state_j_pve : jnp.ndarray
            shape (len(self_imp_controlled_jids),3)
        max_joint_efforts : jnp.ndarray
            shape (len(self_imp_controlled_jids),)
        """
        torques = (cmd_j_pvesd[:,3]*(cmd_j_pvesd[:,0]-state_j_pve[:,0]) + 
                   cmd_j_pvesd[:,4]*(cmd_j_pvesd[:,1]-state_j_pve[:,1]) + 
                   cmd_j_pvesd[:,2])
        torques = jnp.clip(torques,min=-max_joint_efforts,max=max_joint_efforts)
        return torques

    @partial(jax.jit, static_argnames=("self"))
    def _compute_imp_cmds(self, current_cmd_v_j_pvesd, max_torques, vec_jstate_pve):
        vec_efforts = self._compute_impedance_torques_vec(current_cmd_v_j_pvesd, vec_jstate_pve, max_torques)
        # jax.debug.print("t={t} \t eff={eff} \t cmd={current_cmd} jstate={jstate}",
        #                 t=sim_time, eff=vec_efforts, jstate=vec_jstate,
        #                 current_cmd=current_cmd)
        return vec_efforts

    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_impedance_cmds(self, sim_state : SimStateJimp):
        
        new_state = {}
        current_cmd_v_j_pvesd, sim_has_cmd, new_cmds_queue, new_cmds_queue_times = self._get_cmd_and_cleanup_vec(   sim_state.cmds_queue,
                                                                                                                    sim_state.cmds_queue_times,
                                                                                                                    sim_state.sim_time)
        new_state["cmds_queue"] = new_cmds_queue
        new_state["cmds_queue_times"] = new_cmds_queue_times
        if self._use_second_order_reference_filter:
            filtered_refs, new_ref_filter_state = _second_order_filter( current_cmd_v_j_pvesd[:,:,:3],
                                                                        sim_state.ref_filter_coeffs,
                                                                        sim_state.ref_filter_state)
            filtered_cmd_v_j_pvesd = current_cmd_v_j_pvesd.at[:,:,:3].set(filtered_refs)
            new_state["ref_filter_state"] = new_ref_filter_state
        elif self._use_exponential_reference_filter:
            new_filtered_pv_references = sim_state.filtered_pv_references*self._sim_conf.pv_ref_filter_alpha + current_cmd_v_j_pvesd[:,:,:2]*(1-self._sim_conf.pv_ref_filter_alpha)
            filtered_cmd_v_j_pvesd = current_cmd_v_j_pvesd.at[:,:,:2].set(new_filtered_pv_references)
            new_state["filtered_pv_references"] = new_filtered_pv_references
        else:
            filtered_cmd_v_j_pvesd = current_cmd_v_j_pvesd
        vec_jstate = self._get_vec_joint_states_raw_pveaec( self._jids_to_imp_cmd_qpadr,
                                                            self._jids_to_imp_cmd_qvadr,
                                                            sim_state.mjx_data)
        vec_jstate_pve = vec_jstate[:,:,:3]
        new_filtered_pve_states = sim_state.filtered_pve_states*self._pve_sensing_filter_alpha + vec_jstate_pve*(1-self._pve_sensing_filter_alpha)
        new_state["filtered_pve_states"] = new_filtered_pve_states
        vec_efforts = self._compute_imp_cmds(   filtered_cmd_v_j_pvesd,
                                                self._imp_control_max_torque,
                                                new_filtered_pve_states)
        if self._record_joint_hist:
            vec_impjoints_pveaecpvesde = jnp.concat([vec_jstate, filtered_cmd_v_j_pvesd, jnp.expand_dims(vec_efforts,2)], axis = 2)
            new_state["vec_impjoints_pveaecpvesde"] = vec_impjoints_pveaecpvesde

        # jax.debug.print("t={t} \t eff={eff} \t cmd={current_cmd} filtered_cmd={filtered_cmd} raw_jstate={jstate} filtered_jstate={filtered_state} prev_filter_state={prev_filter_state} prevfilter_coeffs={prev_filter_coeffs}",
        #                 t=sim_state.sim_time, eff=vec_efforts, jstate=vec_jstate,
        #                 current_cmd=current_cmd_v_j_pvesd, filtered_cmd=filtered_cmd_v_j_pvesd,
        #                 filtered_state=new_filtered_pve_states, prev_filter_state=sim_state.ref_filter_state, prev_filter_coeffs=sim_state.ref_filter_coeffs)
        sim_state = sim_state.replace_d(new_state)        
        # vec_efforts = jnp.zeros_like(vec_efforts)
        sim_state = self._set_effort_command(sim_state, self._sim_conf.imp_control_jids, vec_efforts, sims_mask=sim_has_cmd)
        return sim_state

    @override
    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_commands(self, sim_state : SimStateJimp) -> SimStateJimp:
        sim_state = self._apply_impedance_cmds(sim_state)
        sim_state = self._apply_torque_cmds(sim_state)
        return sim_state
    
    @override
    def resetWorld(self):
        ret = super().resetWorld()
        self.reset_joint_impedances_commands()
        return ret
    
    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        return self._last_applied_jimp_cmd

    @override
    def _get_joint_state_for_history(self, sim_state : SimStateJimp):
        return sim_state.vec_impjoints_pveaecpvesde

    @override
    def _sim_step_fast_for_scan_full_pveae(self, sim_state_conf : tuple[SimStateJimp,SimConf], _) -> tuple[tuple[SimStateJimp,SimConf], jnp.ndarray | None]:
        sim_state, sim_conf = sim_state_conf
        if self._record_joint_hist:
            # Save pre-forward kinematics and command (the state+command the controller acted on)
            pre_pos = sim_state.mjx_data.qpos[:, self._jids_to_imp_cmd_qpadr]
            pre_vel = sim_state.mjx_data.qvel[:, self._jids_to_imp_cmd_qvadr]
        # Full step: apply_commands (stores pvesd in sim_state), apply_impulses, forward, caches/stats
        sim_state = self._apply_commands(sim_state)
        sim_state = self._apply_impulses(sim_state)
        new_mjx_data = self._mjx_integrate_and_forward(sim_state.mjx_model, sim_state.mjx_data)
        sim_state = sim_state.replace_d({"mjx_data": new_mjx_data,
                                          "sim_time": sim_state.sim_time + self._sim_step_dt})
        sim_state = MjxAdapter._update_monitored_data_cache(sim_state, sim_conf)
        sim_state = MjxAdapter._update_step_stats(sim_state, sim_conf)
        if self._record_joint_hist:
            # pvesd (cols 6-10) from the pre-stored filtered command
            pre_pvesd = sim_state.vec_impjoints_pveaecpvesde[:, :, 6:11]
            joint_state = self._assemble_jimp_joint_history(
                sim_state.mjx_data, pre_pos, pre_vel, pre_pvesd)
        else:
            joint_state = None
        return (sim_state, sim_conf), joint_state

    def _assemble_jimp_joint_history(self, mjx_data, pre_pos : jnp.ndarray, pre_vel : jnp.ndarray,
                                     pre_pvesd : jnp.ndarray) -> jnp.ndarray:
        """Assemble 12-column joint history from pre-forward pos/vel/pvesd and post-forward real forces.

        All force data (cols 2-5, 11) comes from post-forward mjx_data — no stale or recomputed values.
        """
        qvadr = self._jids_to_imp_cmd_qvadr
        qpadr = self._jids_to_imp_cmd_qpadr
        qfrc_applied    = mjx_data.qfrc_applied[:, qvadr]
        current_pveaec = self._get_vec_joint_states_raw_pveaec(qpadr, qvadr, mjx_data)
        pveaec = current_pveaec.at[..., 0].set(pre_pos)
        pveaec = pveaec.at[..., 1].set(pre_vel)
        
        return jnp.concat([
            pveaec,                                         # pveaec (6)
            pre_pvesd,                                      # pvesd (5) - filtered command from _apply_impedance_cmds
            jnp.expand_dims(qfrc_applied, 2),               # act_eff = real qfrc_applied (1) (actuators disabled in JIMP)
        ], axis=2)
    
    # @override
    # def setJointsStateDirect(self, joint_names: list[tuple[str, str]], joint_states_pve: th.Tensor, vec_mask: th.Tensor | None = None):
    #     record_time("MjxJointImpedanceAdapter.setJointsStateDirect")
    #     super().setJointsStateDirect(joint_names, joint_states_pve, vec_mask)
    #     record_time("MjxJointImpedanceAdapter.setJointsStateDirect: called super")
    #     self._reset_filters()
    #     record_time("MjxJointImpedanceAdapter.setJointsStateDirect: resetted filters")

    @override
    def control_period(self):
        self._sim_step_dt_th

    @override
    @partial(jax.jit, static_argnames=["self"], donate_argnames=("sim_state",))
    def _set_joint_state_data(self, sim_state : SimState, sim_conf : SimConf, vec_mask_jnp : jnp.ndarray, qpadr_qvadr : jnp.ndarray, js_pve : jnp.ndarray):
        sim_state = super()._set_joint_state_data(sim_state, sim_conf, vec_mask_jnp, qpadr_qvadr, js_pve)
        sim_state = self._reset_filters_jax(sim_state, sim_conf)
        return sim_state
