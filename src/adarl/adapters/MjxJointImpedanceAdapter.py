from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

from adarl.adapters.MjxAdapter import MjxAdapter, jax2th, th2jax, SimState
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import to_string_tensor
from typing import Any, Literal
import jax
import mujoco
from mujoco import mjx
import mujoco.viewer
import time
from typing_extensions import override
from typing import overload, Sequence
import torch as th
import jax.numpy as jnp
import jax.scipy.spatial.transform
from adarl.utils.utils import build_pose
import adarl.utils.dbg.ggLog as ggLog
import jax.core
from functools import partial
import jax.tree_util
from dataclasses import dataclass
import dataclasses
import traceback
@jax.tree_util.register_dataclass
@dataclass
class SimStateJimp(SimState):
    cmds_queue : jnp.ndarray
    cmds_queue_times : jnp.ndarray
    vec_impjoints_pveaecpvesde : jnp.ndarray
    filtered_pv_references : jnp.ndarray
    filtered_pve_states : jnp.ndarray

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
             "mon_joint_stats_arr_pvae" : self.mon_joint_stats_arr_pvae,
             "vec_impjoints_pveaecpvesde" : self.vec_impjoints_pveaecpvesde,
             "filtered_pv_references" : self.filtered_pv_references,
             "filtered_pve_states" : self.filtered_pve_states,
             "impulse_startends_stime" : self.impulse_startends_stime,
             "impulses_xfrc" : self.impulses_xfrc}
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
                        opt_override : dict[str,Any] | None = None):
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
                                        mon_joint_stats_arr_pvae=jnp.empty((vec_size,0,4), device = jax_device),
                                        vec_impjoints_pveaecpvesde=jnp.empty((vec_size,0,12), device = jax_device),
                                        filtered_pv_references=jnp.empty((vec_size,0,2), device = jax_device),
                                        filtered_pve_states = jnp.empty((vec_size,0,3), device = jax_device),
                                        impulse_startends_stime=jnp.empty((0,), device = jax_device),
                                        impulses_xfrc=jnp.empty((0,), device = jax_device))
        pv_ref_filter_decimation_time = 0.05 # 90% of the filtered value comes from this duration
        pve_sens_filter_decimation_time = 0.005
        self._pv_ref_filter_alpha = 0.1**(1/(pv_ref_filter_decimation_time/self._sim_step_dt))
        self._pve_sens_filter_alpha = 0.1**(1/pve_sens_filter_decimation_time/self._sim_step_dt)
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
            th._assert_async(th.all(vec_mask),f"setJointsImpedanceCommand: vec_mask is not supported, must be None (controls all simulations)")
            # This could probably be implemented fairly easily
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
            th._assert_async(th.all(vec_mask),f"setJointsImpedanceCommand: vec_mask is not supported, must be None (controls all simulations)")
            # This could probably be implemented fairly easily
        # ggLog.info(f"Setting jimp command {joint_impedances_pvesd}")
        self._add_impedance_command(joint_impedances_pvesd=joint_impedances_pvesd,
                                    delay_sec=-1000)
        
    def _add_impedance_command(self,    joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0) -> None:
        # No support for having commands that don't contain all joints
        joint_impedances_pvesd_jax = th2jax(joint_impedances_pvesd, self._jax_device)
        delay_sec_j = th2jax(th.as_tensor(delay_sec), self._jax_device)
        jids = self._imp_control_jids

        # Create a command, commands are always of the size of _imp_control_jids
        cmd = jnp.zeros(shape=(self._vec_size, len(self._imp_control_jids), 5), dtype=jnp.float32, device=self._jax_device)
        # The joints that are actually being commanded are indicated this boolean tensor
        cmd_idxs = self._jids_to_imp_cmd_idx[jids]
        if jnp.any(cmd_idxs < 0):
            raise RuntimeError(f"Tried to set impedance command for joint that has not been set with set_impedance_controlled_joints")

        cmd = cmd.at[:,cmd_idxs].set(joint_impedances_pvesd_jax)
        cmd_time = jnp.resize(delay_sec_j, (self._vec_size,)) + self._simTime
        # ggLog.info(f"adding cmd: times = {self._sim_state.cmds_queue_times} \n cmds = {self._sim_state.cmds_queue}")
        # ggLog.info(f"inserting cmd: cmd_time = {cmd_time},  cmds_queue_times = {self._sim_state.cmds_queue_times}")
        # So now we have a properly formulated command in cmd and cmd_time
        new_cmds_queue, new_cmds_queue_times, inserted = self._insert_cmd_to_queue_vec( cmd=cmd,
                                                                                        cmd_time=cmd_time,
                                                                                        cmds_queue=self._sim_state.cmds_queue,
                                                                                        cmds_queue_times=self._sim_state.cmds_queue_times)
        self._sim_state = self._sim_state.replace_d({"cmds_queue" : new_cmds_queue, "cmds_queue_times" : new_cmds_queue_times})
        if not jnp.all(inserted):
            raise RuntimeError(f"Failed to insert commands, inserted = {inserted}")
        # ggLog.info(f"added cmd: times = {self._sim_state.cmds_queue_times} \n cmds = {self._sim_state.cmds_queue}")

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
        empty_slots = jnp.isinf(cmds_queue_times) # if no command or a command for the same time
        same_time_slots = cmds_queue_times==cmd_time
        # If there is any slot at the same time, then do not use the empty ones
        # If there is a slot at the same time, use that one
        # Thi smounts to the following: (empty and not any(same)) or same
        writable_slots = jnp.logical_or(jnp.logical_and(empty_slots,jnp.logical_not(jnp.any(same_time_slots))), same_time_slots)
        found_slot = jnp.any(writable_slots)
        first_empty = jnp.argmax(writable_slots) # if there is no empty slot, then this is zero, but in that case we end up not selecting it, becuase we use 'found_slot'
        selected_slots = jnp.zeros_like(cmds_queue_times) # Initialize all False
        selected_slots = selected_slots.at[first_empty].set(found_slot) # if a slot was found, set its corresponding cell to True, the rest to False

        cmds_queue_times = cmds_queue_times.at[first_empty].set(jnp.where(found_slot, cmd_time, cmds_queue_times[first_empty]))
        cmds_queue = cmds_queue.at[first_empty].set(jnp.where(found_slot, cmd, cmds_queue[first_empty]))

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
        self._imp_control_max_torque = jnp.array([self._max_joint_impedance_ctrl_torques.get(jn, self._default_max_joint_impedance_ctrl_torque)
                                                    for jn in self._imp_controlled_joint_names], device=self._jax_device)
        self._imp_control_jids = jnp.array(imp_control_jids, device=self._jax_device)
        if len(imp_control_jids) != 0:
            self._jids_to_imp_cdm_qpadr = self._mj_model.jnt_qposadr[self._imp_control_jids]
            self._jids_to_imp_cdm_qvadr = self._mj_model.jnt_dofadr[self._imp_control_jids]
            self._jids_to_imp_cmd_idx = jnp.array([imp_control_jids.index(i) if i in imp_control_jids else -1 
                                                   for i in range(max(imp_control_jids)+1)])
            # self._jids_to_imp_cmd_idx[i] tells at which index to put the command for joint i when forming an impedance command
        else:
            self._jids_to_imp_cdm_qpadr = jnp.empty_like(self._imp_control_jids)
            self._jids_to_imp_cdm_qvadr = jnp.empty_like(self._imp_control_jids)
            self._jids_to_imp_cmd_idx   = jnp.empty_like(self._imp_control_jids)
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
        self._sim_state = self._sim_state.replace_d({"cmds_queue" : jnp.zeros(shape=(self._vec_size, self._queue_size, len(self._imp_control_jids), 5), dtype=jnp.float32, device=self._jax_device),
                                                     "cmds_queue_times" : jnp.full(fill_value=float("+inf"), shape=(self._vec_size, self._queue_size), dtype=jnp.float32, device=self._jax_device)})
        

    def _reset_filters(self):
        if len(self._imp_controlled_joint_names)>0:
            current_pve = self._get_vec_joint_states_pve(self._sim_state.mjx_model, self._sim_state.mjx_data, self._imp_control_jids)
            current_pv = current_pve[:,:,:2]
        else:
            current_pve = jnp.zeros( shape=(self._vec_size, len(self._imp_controlled_joint_names),3),
                                    dtype=jnp.float32,
                                    device=self._jax_device)
            current_pv = current_pve[:,:,:2]
        self._sim_state = self._sim_state.replace_d({"filtered_pv_references" : current_pv,
                                                     "filtered_pve_states" : current_pve})
        
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
        # current_cmd, has_cmd, self._sim_state.cmds_queue, self._sim_state.cmds_queue_times = self._get_cmd_and_cleanup_vec( self._sim_state.cmds_queue,
        #                                                                                                 self._sim_state.cmds_queue_times,
        #                                                                                                 self._simTime)
        # vec_jstate = self._get_vec_joint_states_raw_pve(self._jids_to_imp_cdm_qpadr, self._jids_to_imp_cdm_qvadr, self._mjx_data)
        # vec_efforts = self._compute_impedance_torques_vec(current_cmd, vec_jstate, self._imp_control_max_torque)

        current_cmd_v_j_pvesd, sim_has_cmd, new_cmds_queue, new_cmds_queue_times = self._get_cmd_and_cleanup_vec(   sim_state.cmds_queue,
                                                                                                                    sim_state.cmds_queue_times,
                                                                                                                    sim_state.sim_time)
        new_filtered_pv_references = sim_state.filtered_pv_references*self._pv_ref_filter_alpha + current_cmd_v_j_pvesd[:,:,:2]*(1-self._pv_ref_filter_alpha)
        current_cmd_v_j_pvesd = current_cmd_v_j_pvesd.at[:,:,:2].set(new_filtered_pv_references)
        vec_jstate = self._get_vec_joint_states_raw_pveaec(self._jids_to_imp_cdm_qpadr,
                                                            self._jids_to_imp_cdm_qvadr,
                                                            sim_state.mjx_data)
        vec_jstate_pve = vec_jstate[:,:,:3]
        new_filtered_pve_states = sim_state.filtered_pve_states*self._pve_sens_filter_alpha + vec_jstate_pve*(1-self._pve_sens_filter_alpha)
        # jax.debug.print("t={t} \t new_filtered_pve_states={new_filtered_pve_states} \t vec_jstate_pve={vec_jstate_pve}",
        #                 t=sim_state.sim_time, new_filtered_pve_states=new_filtered_pve_states, vec_jstate_pve=vec_jstate)
        # new_filtered_pve_states = vec_jstate_pve
        vec_efforts = self._compute_imp_cmds(   current_cmd_v_j_pvesd,
                                                self._imp_control_max_torque,
                                                new_filtered_pve_states)
        if self._record_joint_hist:
            vec_impjoints_pveaecpvesde = jnp.concat([vec_jstate, current_cmd_v_j_pvesd, jnp.expand_dims(vec_efforts,2)], axis = 2)
        else:
            vec_impjoints_pveaecpvesde = None
        if self._record_joint_hist:
            sim_state = sim_state.replace_d({"cmds_queue" : new_cmds_queue,
                                            "cmds_queue_times" : new_cmds_queue_times,
                                            "vec_impjoints_pveaecpvesde" : vec_impjoints_pveaecpvesde,
                                            "filtered_pv_references" : new_filtered_pv_references,
                                            "filtered_pve_states" : new_filtered_pve_states})
        else:
            sim_state = sim_state.replace_d({"cmds_queue" : new_cmds_queue,
                                            "cmds_queue_times" : new_cmds_queue_times,
                                            "filtered_pv_references" : new_filtered_pv_references,
                                            "filtered_pve_states" : new_filtered_pve_states})

        
        # vec_efforts = jnp.zeros_like(vec_efforts)
        # ggLog.info(f"setting efforts {vec_efforts}")
        sim_state = self._set_effort_command(sim_state, self._imp_control_jids, vec_efforts, sims_mask=sim_has_cmd)
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
    def setJointsStateDirect(self, joint_names: list[tuple[str, str]], joint_states_pve: th.Tensor, vec_mask: th.Tensor | None = None):
        super().setJointsStateDirect(joint_names, joint_states_pve, vec_mask)
        self._reset_filters()

    @override
    def control_period(self):
        self._sim_step_dt_th