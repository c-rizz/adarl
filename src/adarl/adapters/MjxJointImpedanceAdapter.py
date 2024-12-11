from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

from adarl.adapters.MjxAdapter import MjxAdapter, jax2th, th2jax
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from typing import Any
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
                        add_ground : bool = True):
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
                        add_ground=add_ground)
        self._queue_size = 100
        self._max_joint_impedance_ctrl_torques = max_joint_impedance_ctrl_torques
        self._default_max_joint_impedance_ctrl_torque = default_max_joint_impedance_ctrl_torque
        self.set_impedance_controlled_joints([]) # initialize attributes
        self._insert_cmd_to_queue_vec = jax.vmap(jax.jit(self._insert_cmd_to_queue, donate_argnames=["cmds","cmds_times"]))
        self._get_cmd_and_cleanup_vec = jax.vmap(jax.jit(self._get_cmd_and_cleanup, donate_argnames=["cmds","cmds_times"]), in_axes=(0,0,None))
        self._compute_impedance_torques_vec = jax.vmap(jax.jit(self._compute_impedance_torques), in_axes=(0,0,None))

    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None,
                                        joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        delay_sec = th.as_tensor(delay_sec)
        if th.any(delay_sec < 0):
            raise RuntimeError(f"Cannot have a negative command delay") # actually we could, but it would mess up the logic of set_current_joint_impedance_command
        if joint_names is not None:
            raise RuntimeError(f"joint_names is not supported, must be None (controls all impedance_controlled_joints)")
        if vec_mask is not None and not th.all(vec_mask):
            raise RuntimeError(f"vec_mask is not supported, must be None (controls all simulations)") # This could probably be implemented fairly easily
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
        if vec_mask is not None and not th.all(vec_mask):
            raise RuntimeError(f"vec_mask is not supported, must be None (controls all simulations)") # This could probably be implemented fairly easily
        # ggLog.info(f"Setting jimp command {joint_impedances_pvesd}")
        self._add_impedance_command(joint_impedances_pvesd=joint_impedances_pvesd,
                                    delay_sec=-1000)
        
    def _add_impedance_command(self,    joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0) -> None:
        # No support for having commands that don't contain all joints
        joint_impedances_pvesd_jax = th2jax(joint_impedances_pvesd, self._jax_device)
        delay_sec_j = th2jax(th.as_tensor(delay_sec, device = joint_impedances_pvesd.device), self._jax_device)
        jids = self._imp_control_jids

        # Create a command, commands are always of the size of _imp_control_jids
        cmd = jnp.zeros(shape=(self._vec_size, len(self._imp_control_jids), 5), dtype=jnp.float32, device=self._jax_device)
        # The joints that are actually being commanded are indicated this boolean tensor
        cmd_idxs = self._jids_to_imp_cmd_idx[jids]
        if jnp.any(cmd_idxs < 0):
            raise RuntimeError(f"Tried to set impedance command for joint that has not been set with set_impedance_controlled_joints")

        cmd = cmd.at[:,cmd_idxs].set(joint_impedances_pvesd_jax)
        cmd_time = jnp.resize(delay_sec_j, (self._vec_size,)) + self._simTime
        
        # So now we have a properly formulated command in cmd and cmd_time
        self._cmds_queue, self._cmds_queue_times, inserted = self._insert_cmd_to_queue_vec( cmd=cmd,
                                                                                            cmd_time=cmd_time,
                                                                                            cmds=self._cmds_queue,
                                                                                            cmds_times=self._cmds_queue_times)
        if not jnp.all(inserted):
            raise RuntimeError(f"Failed to insert commands, inserted = {inserted}")

    @staticmethod
    def _insert_cmd_to_queue(   cmd : jnp.ndarray,
                                cmd_time : jnp.ndarray,
                                cmds : jnp.ndarray,
                                cmds_times : jnp.ndarray):
        """_summary_

        Parameters
        ----------
        cmd : jnp.ndarray
            shape: (imp_joints_num, 5)
        cmd_time : jnp.ndarray
            shape: (,)
        cmds : jnp.ndarray
            shape (queue_len, imp_joints_num, 5)
        cmds_times : jnp.ndarray
            shape (queue_len,)

        Returns
        -------
        _type_
            _description_
        """
        # insert command in the first slot that has +inf time
        # if there's no space return some specific value in a ndarray
        empty_slots = jnp.isinf(cmds_times)
        found_slot = jnp.any(empty_slots)
        first_empty = jnp.argmax(empty_slots) # if there is no empty slot, then this is zero
        selected_slots = jnp.zeros_like(cmds_times) # all False
        selected_slots = selected_slots.at[first_empty].set(found_slot) # if a slot was found, set its corresponding cell to True, the rest to False

        cmds_times = cmds_times.at[first_empty].set(jnp.where(found_slot, cmd_time, cmds_times[first_empty]))
        cmds = cmds.at[first_empty].set(jnp.where(found_slot, cmd, cmds[first_empty]))

        return cmds, cmds_times, found_slot

    @staticmethod
    def _get_cmd_and_cleanup(   cmds : jnp.ndarray,
                                cmds_times : jnp.ndarray,
                                current_time : jnp.ndarray):
        past_cmds_mask = cmds_times <= current_time
        has_cmd = jnp.any(past_cmds_mask)
        current_cmd_idx = jnp.argmax(jnp.where(past_cmds_mask, cmds_times, float("-inf")))
        current_cmd = cmds[current_cmd_idx]
        cmds_to_remove_mask = past_cmds_mask.at[current_cmd_idx].set(False)
        cmds_times = jnp.where(cmds_to_remove_mask, float("+inf"), cmds_times) # mark commands as removed by setting the time to +inf
        
        return current_cmd, has_cmd, cmds, cmds_times


    
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

    def _reset_cmd_queue(self):
        self._queued_cmds = 1
        self._cmds_queue =       jnp.zeros(shape=(self._vec_size, self._queue_size, len(self._imp_control_jids), 5), dtype=jnp.float32, device=self._jax_device)
        self._cmds_queue_times = jnp.full(fill_value=float("+inf"), shape=(self._vec_size, self._queue_size), dtype=jnp.float32, device=self._jax_device)
        
    def get_impedance_controlled_joints(self) -> tuple[tuple[str,str],...]:
        """Get the names of the joints that are controlled by this adapter

        Returns
        -------
        tuple[tuple[str,str]]
            The list of the joint names

        """
        return self._imp_controlled_joint_names


    @staticmethod
    def _compute_impedance_torques(cmd_j_pvesd : jnp.ndarray, state_j_pve : jnp.ndarray,
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

    @partial(jax.jit, static_argnums=(0,), donate_argnames=("cmds", "cmds_times"))
    def _compute_imp_cmds(self, cmds, cmds_times, sim_time, qpadr, qvadr, mjx_data, max_torques):
        current_cmd, sim_has_cmd, cmds, cmds_times = self._get_cmd_and_cleanup_vec( cmds,
                                                                                cmds_times,
                                                                                sim_time)
        vec_jstate = self._get_vec_joint_states_raw_pve(qpadr,
                                                        qvadr,
                                                        mjx_data)
        vec_efforts = self._compute_impedance_torques_vec(current_cmd, vec_jstate, max_torques)
        return vec_efforts, sim_has_cmd, cmds, cmds_times

    def _apply_impedance_cmds(self):
        # current_cmd, has_cmd, self._cmds_queue, self._cmds_queue_times = self._get_cmd_and_cleanup_vec( self._cmds_queue,
        #                                                                                                 self._cmds_queue_times,
        #                                                                                                 self._simTime)
        # vec_jstate = self._get_vec_joint_states_raw_pve(self._jids_to_imp_cdm_qpadr, self._jids_to_imp_cdm_qvadr, self._mjx_data)
        # vec_efforts = self._compute_impedance_torques_vec(current_cmd, vec_jstate, self._imp_control_max_torque)
        vec_efforts, sim_has_cmd, self._cmds_queue, self._cmds_queue_times = self._compute_imp_cmds(  self._cmds_queue,
                                                            self._cmds_queue_times,
                                                            self._simTime,
                                                            self._jids_to_imp_cdm_qpadr,
                                                            self._jids_to_imp_cdm_qvadr,
                                                            self._mjx_data,
                                                            self._imp_control_max_torque)
        # vec_efforts = jnp.zeros_like(vec_efforts)
        # ggLog.info(f"setting efforts {vec_efforts}")
        self._set_effort_command(self._imp_control_jids, vec_efforts, sims_mask=sim_has_cmd)

    @override
    def _apply_commands(self):
        self._apply_impedance_cmds()
        self._apply_torque_cmds()

    @override
    def resetWorld(self):
        ret = super().resetWorld()
        self.reset_joint_impedances_commands()
        return ret
    
    @override
    def get_last_applied_command(self) -> th.Tensor:
        return self._last_applied_jimp_cmd