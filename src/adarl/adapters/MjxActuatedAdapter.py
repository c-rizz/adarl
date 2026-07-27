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
    set_rows_cols_masks,
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
from adarl.utils.base_utils import record_time, record_region_end, record_region_start
import numpy as np
from adarl.utils.dbg.dbg_checks import dbg_check
import mujoco


@jax.tree_util.register_dataclass
@dataclass
class SimStateActuated(SimState):
    """SimState extension for MjxActuatedAdapter.
    
    Carries additional fields for tracking the current ctrl values
    and the last applied impedance-like command.
    """
    current_ctrl : jnp.ndarray  # (vec_size, nu) current ctrl vector
    current_cmd_pvesd : jnp.ndarray  # (vec_size, n_actuated_joints, 5) last impedance command
    vec_joint_history : jnp.ndarray  # (vec_size, n_actuated_joints, 12) joint history pveaecpvesde

    def replace_d(self, name_values : dict[str,Any]):
        d = {"mjx_data" : self.mjx_data,
             "mjx_model" : self.mjx_model,
             "requested_qfrc_applied" : self.requested_qfrc_applied,
             "sim_time" : self.sim_time,
             "stats_step_count" : self.stats_step_count,
             "mon_joint_stats_arr_pvaeep" : self.mon_joint_stats_arr_pvaeep,
             "mon_links_stats_arr_v" : self.mon_links_stats_arr_v,
             "mon_joint_state_pveae" : self.mon_joint_state_pveae,
             "mon_link_state" : self.mon_link_state,
             "mon_link_acceleration" : self.mon_link_acceleration,
             "mon_collision_mask" : self.mon_collision_mask,
             "impulse_startends_stime" : self.impulse_startends_stime,
             "impulses_xfrc" : self.impulses_xfrc,
             "current_ctrl" : self.current_ctrl,
             "current_cmd_pvesd" : self.current_cmd_pvesd,
             "vec_joint_history" : self.vec_joint_history,
            }
        d.update(name_values)
        ret = SimStateActuated(**d)
        return ret


@jax.tree_util.register_dataclass
@dataclass
class SimConfActuated(SimConf):
    """SimConf extension for MjxActuatedAdapter.
    
    Carries the mapping from controlled-joint index to actuator index.
    """
    actuated_joint_jids : jnp.ndarray   # (n_actuated_joints,) joint IDs
    actuated_joint_actids : jnp.ndarray  # (n_actuated_joints,) actuator IDs for each joint

    def replace_d(self, name_values : dict[str,Any]):
        d = {"monitored_qpadr" : self.monitored_qpadr,
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
            "site_bodyid" : self.site_bodyid,
            "actuated_joint_jids" : self.actuated_joint_jids,
            "actuated_joint_actids" : self.actuated_joint_actids,
            }
        d.update(name_values)
        ret = SimConfActuated(**d)
        return ret


@jax.tree_util.register_dataclass
@dataclass
class MjxActuatedCommandBatch(MjxCommandBatch):
    current_joint_impedance_command_pvesd : jnp.ndarray


@dataclass(frozen=True)
class SetCurrentJointImpedanceCommand(PublicCommand):
    """Command to set the impedance command immediately (for initialization/reset)."""
    joint_impedances_pvesd : th.Tensor
    vec_mask : th.Tensor | None = None
    joint_names : Sequence[tuple[str, str]] | None = None

    def build_internal_command(self, adapter : MjxActuatedAdapter) -> _InternalCommand:
        if self.joint_names is not None:
            raise RuntimeError("joint_names is not supported, must be None (controls all actuated_joints)")
        expected_size = (adapter._vec_size, adapter._sim_conf.actuated_joint_jids.shape[0], 5)
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

    def run_jax(self, adapter : MjxActuatedAdapter, sim_state : SimStateActuated, sim_conf : SimConfActuated, static_sim_conf : StaticSimConf) -> SimStateActuated:
        if not self.has_effect():
            return sim_state
        # Apply command ctrl immediately
        new_ctrl, new_cmd = adapter._apply_pvesd_to_ctrl(
            sim_state.current_ctrl,
            sim_state.current_cmd_pvesd,
            self.current_joint_impedance_command_pvesd,
            sim_conf.actuated_joint_actids,
            self.vec_mask,
        )
        sim_state = sim_state.replace_d({
            "current_ctrl" : new_ctrl,
            "current_cmd_pvesd" : new_cmd,
        })
        return sim_state


class MjxActuatedAdapter(MjxAdapter, BaseVecJointImpedanceAdapter):
    """MJX adapter that uses MuJoCo built-in actuators (via ctrl) instead of custom impedance control.
    
    Unlike MjxJointImpedanceAdapter which computes PD torques in JAX and writes them to qfrc_applied,
    this adapter writes position references to mjx_data.ctrl and lets MuJoCo's fwd_actuation() compute 
    the actuator forces through the standard pipeline.
    
    Position actuators are automatically added for any hinge/slide joint that doesn't already have one
    in the XML model. The gains for auto-added actuators are controlled by default_actuator_kp and
    default_actuator_kv. ctrl is initialized from qpos0 so uncontrolled actuators produce zero initial error.
    
    Limitations compared to MjxJointImpedanceAdapter:
    - Stiffness (pvesd[3]) and damping (pvesd[4]) are IGNORED — gains are fixed in the XML/auto-generated
    - Velocity reference (pvesd[1]) is IGNORED — MuJoCo position actuators damp toward zero velocity
    - Feedforward torque (pvesd[2]) is IGNORED — position actuators don't support additive torque
    - Only position reference (pvesd[0]) is used, written to ctrl
    - Command delays are NOT supported (raise NotImplementedError)
    """
    
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
                        add_ground : bool = True,
                        add_sky : bool = True,
                        render_znear : float | None = 0.01,
                        render_zfar : float | None = 100.0,
                        log_freq : int = -1,
                        opt_preset : Literal["fast","faster","fastest","mujoco_default","slow","slower"] | None = "fast",
                        record_whole_joint_trajectories : bool = False,
                        log_freq_joints_trajectories = 1000,
                        log_folder="./",
                        safe_revolute_dof_armature = 0.01,
                        revolute_dof_armature_override = None,
                        opt_override : dict[str,Any] | None = None,
                        default_actuator_kp : float = 100.0,
                        default_actuator_kv : float = 10.0,
                        default_max_actuator_force : float = 100.0,
                        max_actuator_forces : dict[tuple[str,str],float] = {}):
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
                        render_znear = render_znear,
                        render_zfar = render_zfar,
                        log_freq=log_freq,
                        record_whole_joint_trajectories=record_whole_joint_trajectories,
                        log_freq_joints_trajectories=log_freq_joints_trajectories,
                        log_folder=log_folder,
                        safe_revolute_dof_armature=safe_revolute_dof_armature,
                        revolute_dof_armature_override=revolute_dof_armature_override,
                        opt_preset=opt_preset,
                        opt_override=opt_override,
                        disable_builtin_actuators=False)
        self._default_actuator_kp = default_actuator_kp
        self._default_actuator_kv = default_actuator_kv
        self._default_max_actuator_force = default_max_actuator_force
        self._max_actuator_forces = max_actuator_forces
        self._sim_state = SimStateActuated(
            mjx_data=self._sim_state.mjx_data,
            mjx_model=self._sim_state.mjx_model,
            requested_qfrc_applied=self._sim_state.requested_qfrc_applied,
            sim_time=self._sim_state.sim_time,
            stats_step_count=jnp.zeros((1,), device=jax_device),
            mon_joint_stats_arr_pvaeep=jnp.empty((vec_size,0,6), device=jax_device),
            mon_links_stats_arr_v=jnp.empty((0,), device=jax_device),
            mon_joint_state_pveae=self._sim_state.mon_joint_state_pveae,
            mon_link_state=self._sim_state.mon_link_state,
            mon_link_acceleration=self._sim_state.mon_link_acceleration,
            mon_collision_mask=self._sim_state.mon_collision_mask,
            impulse_startends_stime=jnp.empty((0,), device=jax_device),
            impulses_xfrc=jnp.empty((0,), device=jax_device),
            current_ctrl=jnp.empty((0,), device=jax_device),
            current_cmd_pvesd=jnp.empty((vec_size, 0, 5), device=jax_device),
            vec_joint_history=jnp.empty((vec_size, 0, 12), device=jax_device),
        )
        self._sim_conf = SimConfActuated(
            monitored_qpadr=self._sim_conf.monitored_qpadr,
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
            actuated_joint_jids=jnp.empty((0,), dtype=jnp.int32, device=self._jax_device),
            actuated_joint_actids=jnp.empty((0,), dtype=jnp.int32, device=self._jax_device),
        )
        self._actuated_joint_names : tuple[tuple[str,str],...] = ()

    @override
    def _aggregate_models(self, models, log_folder):
        """Override to auto-add position actuators (with kp=0, kv=0) for hinge/slide joints that lack one.
        
        Auto-added actuators start with zero gains so they produce no torque. Gains are set to
        default_actuator_kp/kv only for impedance-controlled joints in set_impedance_controlled_joints().
        """
        mj_model, big_speck = super()._aggregate_models(models, log_folder)
        
        # Find which joints already have actuators
        actuated_jids : set[int] = set()
        for act_id in range(mj_model.nu):
            if mj_model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                actuated_jids.add(int(mj_model.actuator_trnid[act_id, 0]))
        
        # Add position actuators with zero gains for hinge/slide joints without actuators
        added = 0
        for jid in range(mj_model.njnt):
            jtype = mj_model.jnt_type[jid]
            if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and jid not in actuated_jids:
                jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                act = big_speck.add_actuator()
                act.name = f"auto_pos_{jname}"
                act.target = jname
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.gainprm[0] = 0.0  # zero gain until joint is controlled
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.biasprm[0] = 0.0
                act.biasprm[1] = 0.0  # zero until controlled
                act.biasprm[2] = 0.0  # zero until controlled
                if mj_model.jnt_limited[jid]:
                    act.ctrllimited = True
                    act.ctrlrange = mj_model.jnt_range[jid].tolist()
                # Set force limits (forcerange set to max; will be overridden per-joint in _update_actuator_gains)
                act.forcelimited = True
                act.forcerange = [-self._default_max_actuator_force, self._default_max_actuator_force]
                added += 1
                ggLog.info(f"MjxActuatedAdapter: Auto-added position actuator '{act.name}' for joint '{jname}' "
                          f"(gains=0, forcerange=[-{self._default_max_actuator_force}, {self._default_max_actuator_force}], will be set when controlled)")
        
        if added > 0:
            ggLog.info(f"MjxActuatedAdapter: Added {added} auto-generated position actuators. Recompiling model.")
            mj_model = big_speck.compile()
        
        return mj_model, big_speck

    @override
    def build_scenario(self, models, default_link_group_collisions=None, add_ground=None):
        """Override build_scenario to additionally build the joint-to-actuator mapping
        and initialize the ctrl state. Actuators are kept enabled via disable_builtin_actuators=False
        passed to the parent constructor."""
        super().build_scenario(models, default_link_group_collisions, add_ground=add_ground)
        
        # Build joint-to-actuator mapping from the compiled model
        self._build_joint_actuator_mapping()
        
        # Track which actuator IDs were auto-added (have "auto_pos_" prefix)
        self._auto_added_actids : set[int] = set()
        for act_id in range(self._mj_model.nu):
            aname = self._mj_model.actuator(act_id).name
            if aname.startswith("auto_pos_"):
                self._auto_added_actids.add(act_id)
        
        # Initialize ctrl to zero — auto-added actuators have zero gains so ctrl value is irrelevant
        nu = self._mj_model.nu
        current_ctrl = jnp.zeros((self._vec_size, nu), dtype=jnp.float32, device=self._jax_device)
        self._sim_state = self._sim_state.replace_d({"current_ctrl": current_ctrl})
        
        ggLog.info(f"MjxActuatedAdapter: Actuators enabled. nu={nu}, "
                   f"auto-added={len(self._auto_added_actids)}, "
                   f"actuator names: {[self._mj_model.actuator(i).name for i in range(nu)]}")

    def _build_joint_actuator_mapping(self):
        """Build mapping from joint names to actuator indices.
        
        Uses model.actuator_trntype and model.actuator_trnid to find which actuator
        controls which joint.
        """
        self._jid_to_actid : dict[int, int] = {}
        for act_id in range(self._mj_model.nu):
            trntype = self._mj_model.actuator_trntype[act_id]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                jid = self._mj_model.actuator_trnid[act_id, 0]
                if jid in self._jid_to_actid:
                    ggLog.warn(f"Joint {jid} has multiple actuators. "
                               f"Using actuator {act_id}, previous was {self._jid_to_actid[jid]}.")
                self._jid_to_actid[jid] = act_id

    def set_impedance_controlled_joints(self, joint_names : Sequence[tuple[str,str]]):
        """Set the joints that will be controlled by the adapter.
        
        Each specified joint must have a corresponding MuJoCo actuator (either from the XML model
        or auto-generated at build_scenario time).

        Parameters
        ----------
        joint_names : Sequence[Tuple[str,str]]
            List of the controlled joint names
        """
        self._actuated_joint_names = tuple(joint_names)
        
        if len(joint_names) == 0:
            self._sim_conf = self._sim_conf.replace_d({
                "actuated_joint_jids": jnp.empty((0,), dtype=jnp.int32, device=self._jax_device),
                "actuated_joint_actids": jnp.empty((0,), dtype=jnp.int32, device=self._jax_device),
            })
            self._actuated_qpadr = jnp.empty((0,), dtype=jnp.int32, device=self._jax_device)
            self._actuated_qvadr = jnp.empty((0,), dtype=jnp.int32, device=self._jax_device)
            self._sim_state = self._sim_state.replace_d({
                "current_cmd_pvesd": jnp.empty((self._vec_size, 0, 5), device=self._jax_device),
            })
            # Zero all auto-added actuator gains (if build_scenario has run)
            if hasattr(self, "_auto_added_actids"):
                self._update_actuator_gains(set())
            return

        jids = []
        actids = []
        for jn in joint_names:
            jid = self._jname2jid[jn]
            if jid not in self._jid_to_actid:
                raise RuntimeError(
                    f"Joint {jn} (jid={jid}) does not have a corresponding MuJoCo actuator. "
                    f"This should only happen for free/ball joints, which are not supported. "
                    f"Hinge/slide joints get auto-added position actuators at build_scenario time. "
                    f"Available actuated joints: {[(self._jid2jname[j], a) for j, a in self._jid_to_actid.items()]}"
                )
            jids.append(jid)
            actids.append(self._jid_to_actid[jid])

        jids_jax = jnp.array(jids, dtype=jnp.int32, device=self._jax_device)
        actids_jax = jnp.array(actids, dtype=jnp.int32, device=self._jax_device)
        
        self._actuated_qpadr = self._sim_conf.jnt_qposadr[np.array(jids)]
        self._actuated_qvadr = self._sim_conf.jnt_dofadr[np.array(jids)]
        
        self._sim_conf = self._sim_conf.replace_d({
            "actuated_joint_jids": jids_jax,
            "actuated_joint_actids": actids_jax,
        })
        
        # Initialize the current command state
        self._sim_state = self._sim_state.replace_d({
            "current_cmd_pvesd": jnp.zeros((self._vec_size, len(joint_names), 5), 
                                           dtype=jnp.float32, device=self._jax_device),
        })

        # Set gains and force limits: auto-added actuators that are now controlled get default_kp/kv
        # and per-joint forcerange, auto-added actuators that are NOT controlled stay at zero.
        actid_to_jname = {actids[i]: joint_names[i] for i in range(len(joint_names))}
        self._update_actuator_gains(set(actids), joint_names, actid_to_jname)

        if self._record_joint_hist:
            self._sim_state = self._sim_state.replace_d({
                "vec_joint_history": jnp.zeros((self._vec_size, len(joint_names), 12),
                                               dtype=jnp.float32, device=self._jax_device),
            })
            self._full_history_labels = to_string_tensor(sum(
                [[f"{v}.{jn[1]}" for v in ["pos","vel","cmd_eff","acc","eff","constr_eff","pref","vref","eref","stiff","damp","act_eff"]]
                 for jn in joint_names], [])).unsqueeze(0)

        ggLog.info(f"MjxActuatedAdapter: Controlling {len(joint_names)} joints via actuators: "
                   f"{list(zip(joint_names, actids))}")

    def _update_actuator_gains(self, controlled_actids : set[int], controlled_joint_names : Sequence[tuple[str,str]] = (), actid_to_jname : dict[int, tuple[str,str]] = {}):
        """Set gains and force limits on auto-added actuators: default_kp/kv if controlled, zero if not.
        
        XML-defined actuators are never modified — their gains are always preserved.
        Updates both self._mj_model (CPU) and the mjx_model in self._sim_state (JAX device).
        
        Parameters
        ----------
        controlled_actids : set[int]
            Set of actuator IDs that are being controlled.
        controlled_joint_names : Sequence[tuple[str,str]]
            Joint names corresponding to controlled_actids (used for per-joint force overrides).
        actid_to_jname : dict[int, tuple[str,str]]
            Mapping from actuator ID to joint name pair.
        """
        nu = self._mj_model.nu
        # Build new gain/bias/forcerange arrays from current mjx_model
        mjx_model = self._sim_state.mjx_model
        new_gainprm = mjx_model.actuator_gainprm
        new_biasprm = mjx_model.actuator_biasprm
        new_forcerange = mjx_model.actuator_forcerange
        
        for act_id in self._auto_added_actids:
            if act_id in controlled_actids:
                kp = self._default_actuator_kp
                kv = self._default_actuator_kv
                jname = actid_to_jname.get(act_id, None)
                fmax = self._max_actuator_forces.get(jname, self._default_max_actuator_force) if jname else self._default_max_actuator_force
            else:
                kp = 0.0
                kv = 0.0
                fmax = 0.0
            # Update CPU model
            self._mj_model.actuator_gainprm[act_id, 0] = kp
            self._mj_model.actuator_biasprm[act_id, 0] = 0.0
            self._mj_model.actuator_biasprm[act_id, 1] = -kp
            self._mj_model.actuator_biasprm[act_id, 2] = -kv
            self._mj_model.actuator_forcerange[act_id] = [-fmax, fmax]
            # Update JAX model arrays (shape: (nu, 10) for gainprm/biasprm, (nu, 2) for forcerange)
            new_gainprm = new_gainprm.at[act_id, 0].set(kp)
            new_biasprm = new_biasprm.at[act_id, 0].set(0.0)
            new_biasprm = new_biasprm.at[act_id, 1].set(-kp)
            new_biasprm = new_biasprm.at[act_id, 2].set(-kv)
            new_forcerange = new_forcerange.at[act_id, 0].set(-fmax)
            new_forcerange = new_forcerange.at[act_id, 1].set(fmax)
        
        new_mjx_model = mjx_model.replace(
            actuator_gainprm=new_gainprm,
            actuator_biasprm=new_biasprm,
            actuator_forcerange=new_forcerange,
        )
        self._sim_state = self._sim_state.replace_d({"mjx_model": new_mjx_model})
        
        n_controlled = len(controlled_actids & self._auto_added_actids)
        n_zeroed = len(self._auto_added_actids) - n_controlled
        ggLog.info(f"MjxActuatedAdapter: Updated auto-added actuator gains: "
                   f"{n_controlled} controlled (kp={self._default_actuator_kp}, kv={self._default_actuator_kv}), "
                   f"{n_zeroed} uncontrolled (kp=0, kv=0, forcerange=0)")

    def _validate_pvesd_command(self, joint_impedances_pvesd : th.Tensor):
        """Validate that the pvesd command only contains values supported by this adapter.
        
        MuJoCo position actuators only use position references (pvesd[0]). The other fields must
        match the adapter's fixed values: vel_ref=0, torque_ff=0, stiffness=default_actuator_kp,
        damping=default_actuator_kv.
        
        Parameters
        ----------
        joint_impedances_pvesd : th.Tensor
            Shape (vec_size, n_joints, 5) with fields [pos_ref, vel_ref, torque_ff, stiffness, damping]
        
        Raises
        ------
        RuntimeError
            If any field doesn't match the supported value.
        """
        vel_refs = joint_impedances_pvesd[:, :, 1]
        dbg_check(lambda: th.all(vel_refs == 0.0),
                  async_assert=True,
                  assert_msg="MjxActuatedAdapter does not support velocity references (pvesd[:,:,1]). "
                             "MuJoCo position actuators always damp toward zero velocity. "
                             "Use MjxJointImpedanceAdapter for velocity reference support.")
        torque_ff = joint_impedances_pvesd[:, :, 2]
        dbg_check(lambda: th.all(torque_ff == 0.0),
                  async_assert=True,
                  assert_msg="MjxActuatedAdapter does not support feedforward torques (pvesd[:,:,2]). "
                             "MuJoCo position actuators do not support additive torque offsets. "
                             "Use MjxJointImpedanceAdapter for feedforward torque support.")
        stiffness = joint_impedances_pvesd[:, :, 3]
        dbg_check(lambda: th.all(stiffness == self._default_actuator_kp),
                  async_assert=True,
                  assert_msg=f"MjxActuatedAdapter does not support per-step stiffness (pvesd[:,:,3]). "
                             f"Gains are fixed at default_actuator_kp={self._default_actuator_kp}. "
                             f"Use MjxJointImpedanceAdapter for variable stiffness support.")
        damping = joint_impedances_pvesd[:, :, 4]
        dbg_check(lambda: th.all(damping == self._default_actuator_kv),
                  async_assert=True,
                  assert_msg=f"MjxActuatedAdapter does not support per-step damping (pvesd[:,:,4]). "
                             f"Gains are fixed at default_actuator_kv={self._default_actuator_kv}. "
                             f"Use MjxJointImpedanceAdapter for variable damping support.")

    def setJointsImpedanceCommand(self, joint_impedances_pvesd : th.Tensor,
                                        delay_sec : th.Tensor | float = 0.0,
                                        vec_mask : th.Tensor | None = None,
                                        joint_names : Sequence[tuple[str,str]] | None = None) -> None:
        if joint_names is not None:
            raise RuntimeError("joint_names is not supported, must be None (controls all actuated joints)")
        
        # Check for delays
        delay_sec_t = th.as_tensor(delay_sec)
        dbg_check(lambda: th.all(delay_sec_t <= 0),
                  async_assert=True,
                  assert_msg="MjxActuatedAdapter does not support command delays. "
                             "Use delay_sec=0.0 or use MjxJointImpedanceAdapter for delay support.")
        
        self._validate_pvesd_command(joint_impedances_pvesd)
        
        record_region_start("MjxActuatedAdapter.setJointsImpedanceCommand")
        pvesd_jax = th2jax(joint_impedances_pvesd, self._jax_device)
        
        if vec_mask is not None:
            mask_jax = th2jax(vec_mask, self._jax_device)
        else:
            mask_jax = self._all_vecs
        
        new_ctrl, new_cmd = self._apply_pvesd_to_ctrl(
            self._sim_state.current_ctrl,
            self._sim_state.current_cmd_pvesd,
            pvesd_jax,
            self._sim_conf.actuated_joint_actids,
            mask_jax,
        )
        self._sim_state = self._sim_state.replace_d({
            "current_ctrl": new_ctrl,
            "current_cmd_pvesd": new_cmd,
        })
        record_region_end("MjxActuatedAdapter.setJointsImpedanceCommand")

    @staticmethod
    @partial(jax.jit, donate_argnames=("current_ctrl", "current_cmd_pvesd"))
    def _apply_pvesd_to_ctrl(current_ctrl : jnp.ndarray,
                              current_cmd_pvesd : jnp.ndarray,
                              pvesd : jnp.ndarray,
                              actids : jnp.ndarray,
                              vec_mask : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract position references from pvesd command and write to ctrl at actuator indices.
        
        Parameters
        ----------
        current_ctrl : jnp.ndarray
            Shape (vec_size, nu). Current ctrl vector.
        current_cmd_pvesd : jnp.ndarray
            Shape (vec_size, n_joints, 5). Current stored pvesd command.
        pvesd : jnp.ndarray
            Shape (vec_size, n_joints, 5). New pvesd command.
        actids : jnp.ndarray
            Shape (n_joints,). Actuator IDs for each joint.
        vec_mask : jnp.ndarray
            Shape (vec_size,). Boolean mask for which envs to apply to.
        
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Updated ctrl and pvesd command.
        """
        # Extract position references (index 0 of pvesd)
        pos_refs = pvesd[:, :, 0]  # (vec_size, n_joints)
        
        # Write pos_refs to ctrl at the actuator indices, respecting vec_mask
        new_ctrl = current_ctrl.at[:, actids].set(
            jnp.where(vec_mask[:, None], pos_refs, current_ctrl[:, actids])
        )
        new_cmd = jnp.where(vec_mask[:, None, None], pvesd, current_cmd_pvesd)
        return new_ctrl, new_cmd

    def reset_joint_impedances_commands(self):
        """Reset ctrl and pvesd commands to zero. Uncontrolled auto-added actuators have zero gains
        so ctrl value doesn't matter."""
        n_joints = len(self._actuated_joint_names)
        nu = self._mj_model.nu
        updates = {"current_ctrl": jnp.zeros((self._vec_size, nu), dtype=jnp.float32, device=self._jax_device)}
        if n_joints > 0:
            updates["current_cmd_pvesd"] = jnp.zeros((self._vec_size, n_joints, 5), 
                                                      dtype=jnp.float32, device=self._jax_device)
        self._sim_state = self._sim_state.replace_d(updates)

    def set_current_joint_impedance_command(self, joint_impedances_pvesd : th.Tensor,
                                                  joint_names : Sequence[tuple[str,str]] | None = None,
                                                  vec_mask : th.Tensor | None = None) -> None:
        """Applies a joint impedance command immediately.
        Meant to be used outside of the normal control loop, just for resetting/initializing.
        """
        if joint_names is not None:
            raise RuntimeError("joint_names is not supported, must be None (controls all actuated joints)")
        # Validation is done inside setJointsImpedanceCommand
        self.setJointsImpedanceCommand(joint_impedances_pvesd=joint_impedances_pvesd,
                                       delay_sec=0.0,
                                       vec_mask=vec_mask)

    def set_reference_filter(self, reference_filter_cutoff_frequency : th.Tensor, vec_mask : th.Tensor | None = None):
        """Not supported by MjxActuatedAdapter. Raises an exception."""
        raise RuntimeError(
            "MjxActuatedAdapter does not support reference filters. "
            "Filtering is not applied when using MuJoCo built-in actuators. "
            "Use MjxJointImpedanceAdapter for reference filter support."
        )


    def get_impedance_controlled_joints(self) -> tuple[tuple[str,str],...]:
        return self._actuated_joint_names


    def get_current_joint_impedance_command(self) -> th.Tensor:
        return jax2th(self._sim_state.current_cmd_pvesd, self._out_th_device)


    @override
    @partial(jax.jit, static_argnums=(0,), donate_argnames=("sim_state",))
    def _apply_commands(self, sim_state : SimStateActuated) -> SimStateActuated:
        """Write ctrl to mjx_data, then apply any direct torque commands."""
        # Write current ctrl into mjx_data.ctrl
        new_mjx_data = sim_state.mjx_data.replace(ctrl=sim_state.current_ctrl)
        sim_state = sim_state.replace_d({"mjx_data": new_mjx_data})
        # Also apply any direct torque commands (qfrc_applied) from parent
        sim_state = self._apply_torque_cmds(sim_state)
        return sim_state

    @override
    def _sim_step_fast_for_scan_full_pveae(self, sim_state_conf : tuple[SimStateActuated,SimConf], _) -> tuple[tuple[SimStateActuated,SimConf], jnp.ndarray | None]:
        sim_state, sim_conf = sim_state_conf
        if self._record_joint_hist:
            # Save pre-forward kinematics (the state the controller acted on)
            pre_pos = sim_state.mjx_data.qpos[:, self._actuated_qpadr]  # (vec, nj)
            pre_vel = sim_state.mjx_data.qvel[:, self._actuated_qvadr]  # (vec, nj)
            pre_pvesd = sim_state.current_cmd_pvesd                     # (vec, nj, 5)
        # Run full step: apply_commands, apply_impulses, integrate_and_forward, update caches/stats
        sim_state = self._apply_commands(sim_state)
        sim_state = self._apply_impulses(sim_state)
        new_mjx_data = self._mjx_integrate_and_forward(sim_state.mjx_model, sim_state.mjx_data)
        sim_state = sim_state.replace_d({"mjx_data": new_mjx_data,
                                          "sim_time": sim_state.sim_time + self._sim_step_dt})
        sim_state = MjxAdapter._update_monitored_data_cache(sim_state, sim_conf)
        sim_state = MjxAdapter._update_step_stats(sim_state, sim_conf)
        if self._record_joint_hist:
            joint_state = self._assemble_joint_history(sim_state.mjx_data, pre_pos, pre_vel, pre_pvesd)
        else:
            joint_state = None
        return (sim_state, sim_conf), joint_state

    def _assemble_joint_history(self, mjx_data, pre_pos : jnp.ndarray, pre_vel : jnp.ndarray, pre_pvesd : jnp.ndarray) -> jnp.ndarray:
        """Assemble 12-column joint history from pre-forward kinematics and post-forward real MuJoCo forces.
        
        Columns: [pos, vel, cmd_eff, acc, eff, constr_eff, pref, vref, eref, stiff, damp, act_eff]
        """
        qfrc_actuator   = mjx_data.qfrc_actuator[:, self._actuated_qvadr]
        qfrc_applied    = mjx_data.qfrc_applied[:, self._actuated_qvadr]
        qacc            = mjx_data.qacc[:, self._actuated_qvadr]
        qfrc_passive    = mjx_data.qfrc_passive[:, self._actuated_qvadr]
        qfrc_constraint = mjx_data.qfrc_constraint[:, self._actuated_qvadr]
        pveaec = jnp.stack([
            pre_pos,                                      # pos (pre-forward)
            pre_vel,                                      # vel (pre-forward)
            qfrc_applied + qfrc_actuator,                 # cmd_eff (post-forward)
            qacc,                                         # acc (post-forward)
            qfrc_applied + qfrc_passive + qfrc_constraint,  # eff (post-forward)
            qfrc_constraint,                              # constr_eff (post-forward)
        ], axis=2)
        return jnp.concat([
            pveaec,                                       # pveaec (6)
            pre_pvesd,                                    # pvesd (5)
            jnp.expand_dims(qfrc_actuator, 2),            # act_eff = real qfrc_actuator (1)
        ], axis=2)


    @override
    def resetWorld(self):
        ret = super().resetWorld()
        # super().resetWorld() restores mjx_model from _original_mjx_model (which has kp=0/kv=0
        # for auto-added actuators). Re-apply the gains for currently controlled joints.
        if hasattr(self, "_auto_added_actids") and len(self._actuated_joint_names) > 0:
            actids = [self._jid_to_actid[self._jname2jid[jn]] for jn in self._actuated_joint_names]
            actid_to_jname = {actids[i]: self._actuated_joint_names[i] for i in range(len(self._actuated_joint_names))}
            self._update_actuator_gains(set(actids), self._actuated_joint_names, actid_to_jname)
        self.reset_joint_impedances_commands()
        return ret


    @override
    def control_period(self):
        return self._sim_step_dt_th


    @override
    @partial(jax.jit, static_argnames=["self"], donate_argnames=("sim_state",))
    def _set_joint_state_data(self, sim_state : SimState, sim_conf : SimConf, 
                              vec_mask_jnp : jnp.ndarray, qpadr_qvadr : jnp.ndarray, 
                              js_pve : jnp.ndarray):
        sim_state = super()._set_joint_state_data(sim_state, sim_conf, vec_mask_jnp, qpadr_qvadr, js_pve)
        return sim_state
