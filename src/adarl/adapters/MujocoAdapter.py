from __future__ import annotations
import os

from adarl.adapters.mujoco_utils import add_arrow_to_renderer, aggregate_models, print_mj_model
from adarl.utils.base_utils import record_region_end, record_region_start, record_time
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch as th
import mujoco
import mujoco.viewer
import time

from typing import Sequence, Any
from typing_extensions import override

from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import compile_xacro_string
from adarl.adapters.mujoco_utils import apply_opt_preset
import copy
import adarl.utils.dbg.ggLog as ggLog
import pprint


from typing import TypeAlias 
mujoco_mjtGeom :            TypeAlias = mujoco.mjtGeom # type: ignore
mujoco_mju_quat2Mat :       TypeAlias = mujoco.mju_quat2Mat # type: ignore
mujoco_mju_mat2Quat :       TypeAlias = mujoco.mju_mat2Quat # type: ignore
mujoco_mjv_initGeom :       TypeAlias = mujoco.mjv_initGeom # type: ignore
mujoco_mjtCatBit :          TypeAlias = mujoco.mjtCatBit # type: ignore
mujoco_mjv_connector :      TypeAlias = mujoco.mjv_connector # type: ignore
mujoco_MjData :             TypeAlias = mujoco.MjData # type: ignore
mujoco__functions :         TypeAlias = mujoco._functions # type: ignore
mujoco_mju_dense2sparse :   TypeAlias = mujoco.mju_dense2sparse # type: ignore
mujoco_MjModel :            TypeAlias = mujoco.MjModel # type: ignore
mujoco_MjSpec :             TypeAlias = mujoco.MjSpec # type: ignore
mujoco_mjtInertiaFromGeom : TypeAlias = mujoco.mjtInertiaFromGeom # type: ignore
mujoco_MjsBody :            TypeAlias = mujoco.MjsBody # type: ignore
mujoco_mjtSensor :          TypeAlias = mujoco.mjtSensor # type: ignore
mujoco_mjtObj :             TypeAlias = mujoco.mjtObj # type: ignore
mujoco_mj_id2name :         TypeAlias = mujoco.mj_id2name # type: ignore
mujoco_mjtTrn :             TypeAlias = mujoco.mjtTrn # type: ignore
mujoco_mj_printModel :      TypeAlias = mujoco.mj_printModel # type: ignore
mujoco_mjtJoint :           TypeAlias = mujoco.mjtJoint # type: ignore
mujoco_mjtIntegrator :      TypeAlias = mujoco.mjtIntegrator # type: ignore
mujoco_mjtDisableBit :      TypeAlias = mujoco.mjtDisableBit # type: ignore
mujoco_mj_resetData :       TypeAlias = mujoco.mj_resetData # type: ignore
mujoco_mj_name2id :         TypeAlias = mujoco.mj_name2id # type: ignore
mujoco_MjvOption :          TypeAlias = mujoco.MjvOption # type: ignore
mujoco_mjtVisFlag :         TypeAlias = mujoco.mjtVisFlag # type: ignore
mujoco_mj_camlight :        TypeAlias = mujoco.mj_camlight # type: ignore
mujoco_mj_forward :         TypeAlias = mujoco.mj_forward # type: ignore

model_element_separator = "#"


def add_compiler_options(urdf_def: str,
                         max_hull_vert: int = 32,
                         discardvisual: bool = False,
                         strippath: bool = False) -> str:
    mujoco_block = ('<mujoco>\n' +
                    f'    <compiler  discardvisual="{str(discardvisual).lower()}" strippath="{str(strippath).lower()}" maxhullvert="{max_hull_vert:d}"/>\n'
                    '</mujoco>')
    return urdf_def.replace("</robot>", mujoco_block + "\n</robot>")


class MujocoAdapter(BaseVecSimulationAdapter, BaseVecJointEffortAdapter):
    """Single-simulation adapter for Mujoco classic.
    This adapter implements the BaseVecSimulationAdapter interface but only supports vec_size == 1.

    WARNING: This adapter should still be considered as a Work In Progress.
    """

    def __init__(self,
                 vec_size: int = 1,
                 sim_step_dt: float = 1/1024,
                 step_length_sec: float = 48/1024,
                 output_th_device: th.device = th.device("cpu"),
                 log_folder: str = "./",
                 show_gui: bool = False,
                 gui_frequency: float = 25.0,
                 opt_preset: str = "mujoco_default",
                 opt_override: dict | None = None,
                 safe_revolute_dof_armature: float = 0.01,
                 revolute_dof_armature_override: float | None = None,
                 revolute_dof_damping_override: float | None = None,
                 revolute_dof_frictionloss_override: float | None = None,
                 disable_builtin_actuators: bool = True,
                 geom_overrides : dict[str,Any] | None = None):
        if vec_size != 1:
            raise ValueError("MujocoAdapter only supports vec_size=1")
        super().__init__(vec_size=vec_size, output_th_device=output_th_device)
        self._sim_step_dt = float(sim_step_dt)
        self._sim_step_dt_th = th.as_tensor(self._sim_step_dt, device=output_th_device)
        self._mj_model: mujoco_MjModel = None
        self._mj_data: mujoco_MjData | None = None
        self._requested_qfrc_applied = np.zeros((0,), dtype=np.float64)
        self._renderer_cache: dict[str, mujoco.Renderer] = {}
        self._jname2jid: dict[tuple[str, str], int] = {}
        self._jid2jname: dict[int, tuple[str, str]] = {}
        self._lname2lid: dict[tuple[str, str], int] = {}
        self._lid2lname: dict[int, tuple[str, str]] = {}
        self._add_ground = True
        self._add_sky = True
        self._uneven_ground = False
        self._discardvisual = False
        self._opt_preset = opt_preset
        self._opt_override = opt_override
        self._safe_revolute_dof_armature = safe_revolute_dof_armature
        self._revolute_dof_armature_override = revolute_dof_armature_override
        self._revolute_dof_damping_override = revolute_dof_damping_override
        self._revolute_dof_frictionloss_override = revolute_dof_frictionloss_override
        self._disable_builtin_actuators = disable_builtin_actuators
        self._enable_rendering = True
        self._stepping = False
        self._step_length_sec = step_length_sec
        self._forward_needed = True
        self._time_since_startup = 0.0
        self._step_stats_len = 1000
        self._log_folder = log_folder
        self._show_gui = show_gui
        self._gui_freq = gui_frequency
        self._viewer = None
        self._viewer_mj_data : mujoco_MjData | None = None
        self._last_gui_update_wtime = 0.0
        self._geom_overrides = geom_overrides

    def _ensure_ready(self):
        if self._mj_model is None or self._mj_data is None:
            raise RuntimeError("Mujoco model not built. Call build_scenario first.")

    @override
    def build_scenario(self, models: Sequence[ModelSpawnDef] = (), **kwargs):
        """Build and setup the environment scenario. Should be called by the environment before startup()."""
        if self._vec_size != 1:
            raise RuntimeError("MujocoAdapter only supports vec_size=1")
        models = list(models)
        if len(models) == 0:
            raise RuntimeError("No models provided to build_scenario")
        scenario_logs_folder = self._log_folder+"/MujocoAdapter/scenario_logs"

        self._mj_model, spec = aggregate_models(models,
                                          add_ground=self._add_ground,
                                          add_sky=self._add_sky,
                                          uneven_ground=self._uneven_ground,
                                          discardvisual=self._discardvisual,
                                          log_folder=scenario_logs_folder,
                                          geom_overrides=self._geom_overrides)
        self._mj_model = apply_opt_preset(self._mj_model, self._opt_preset, self._opt_override)
        
        self._mj_model.opt.timestep = self._sim_step_dt
        if self._disable_builtin_actuators:
            # Disable all built-in actuators; we apply forces/torques directly to the joints in the control step.
            # This matches MjxAdapter: leaving actuators active lets them (e.g. position servos at ctrl=0)
            # fight the impedance effort and pull controlled joints toward their default targets.
            self._mj_model.opt.disableactuator = -1
            self._mj_model.opt.disableflags |= mujoco_mjtDisableBit.mjDSBL_ACTUATION
        self._apply_revolute_dof_overrides()


        os.makedirs(scenario_logs_folder, exist_ok=True)
        with open(scenario_logs_folder+"/mujoco_opt.txt", "w") as text_file:
            text_file.write(str(self._mj_model.opt))
        print_mj_model(self._mj_model, full_dump=True, file=scenario_logs_folder+"/mj_model_full.txt")

        self._mj_data = mujoco_MjData(self._mj_model)
        self._requested_qfrc_applied = np.zeros((self._mj_model.nv,), dtype=np.float64)
        mujoco_mj_forward(self._mj_model, self._mj_data)

        self._renderer_cache.clear()
        self._build_name_maps()
        self._build_renderers()

        self.set_monitored_joints([])
        self.set_monitored_links([])
        self._reset_step_stats(self._step_stats_len)

        ggLog.info(f"Joints:\n{pprint.pformat(self._jname2jid)}")
        ggLog.info(f"Links:\n{pprint.pformat(self._lname2lid)}")
        ggLog.info(f"Cameras:\n{pprint.pformat(self._cname2cid)}")

        self._launch_gui_if_needed()

    def _apply_revolute_dof_overrides(self):
        """Ensure revolute DOFs have nonzero armature/damping/frictionloss and apply the
        configured overrides, matching MjxAdapter so the two backends share the same joint dynamics."""
        safe_damping = 1.0
        safe_frictionloss = 0.2
        for dof_id in range(self._mj_model.nv):
            if self._mj_model.jnt_type[self._mj_model.dof_jntid[dof_id]] != mujoco_mjtJoint.mjJNT_HINGE:
                continue
            if self._mj_model.dof_armature[dof_id] == 0:
                self._mj_model.dof_armature[dof_id] = self._safe_revolute_dof_armature
            if self._revolute_dof_armature_override is not None:
                self._mj_model.dof_armature[dof_id] = self._revolute_dof_armature_override
            if self._mj_model.dof_frictionloss[dof_id] == 0:
                self._mj_model.dof_frictionloss[dof_id] = safe_frictionloss
            if self._revolute_dof_frictionloss_override is not None:
                self._mj_model.dof_frictionloss[dof_id] = self._revolute_dof_frictionloss_override
            if self._mj_model.dof_damping[dof_id] == 0:
                self._mj_model.dof_damping[dof_id] = safe_damping
            if self._revolute_dof_damping_override is not None:
                self._mj_model.dof_damping[dof_id] = self._revolute_dof_damping_override

    def _build_renderers(self):
        self._camera_sizes_wh :dict[str,tuple[int,int]] = {self._cid2cname[cid]:(self._mj_model.cam_resolution[cid][1],self._mj_model.cam_resolution[cid][0]) for cid in self._cid2cname}
        if self._enable_rendering:
            self._render_scene_option = mujoco_MjvOption()
            self._render_scene_option.flags[mujoco_mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            # self._render_scene_option.flags[mujoco_mjtVisFlag.mjVIS_COM] = 1
            # self._render_scene_option.flags[mujoco_mjtVisFlag.mjVIS_TRANSPARENT] = 1
            ggLog.info(f"Making rederer for resolutions: {set(self._camera_sizes_wh.values())}, MUJOCO_GL={os.environ.get('MUJOCO_GL','<not set>')}")
            self._renderers : dict[tuple[int,int],mujoco.Renderer]= {resolution: mujoco.Renderer(self._mj_model,resolution[0],resolution[1])
                                                                        for resolution in set(self._camera_sizes_wh.values())}
        else:
            self._renderers = {}

    @override
    def destroy_scenario(self, **kwargs):
        self._close_gui()
        self._mj_model = None
        self._mj_data = None
        self._renderers = {}
        self._renderers_mj_data = []

    def _launch_gui_if_needed(self):
        """Open the interactive passive viewer window, if show_gui is enabled."""
        if not self._show_gui or self._viewer is not None:
            return
        try:
            # Bind the viewer directly to the live data. resetWorld() resets it in place
            # (rather than reallocating) so this reference stays valid across episodes.
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)
            self._last_gui_update_wtime = 0.0
        except Exception as e:
            ggLog.warn(f"MujocoAdapter: could not open interactive gui ({type(e).__name__}: {e}); disabling show_gui.")
            self._viewer = None
            self._show_gui = False

    def _update_gui(self, force: bool = False):
        """Refresh the viewer window, throttled to gui_frequency."""
        if not self._show_gui or self._viewer is None:
            return
        if not self._viewer.is_running():
            self._close_gui()
            self._show_gui = False
            return
        record_region_start("MujocoAdapter.update_gui()")
        if force or (time.monotonic() - self._last_gui_update_wtime > 1/self._gui_freq):
            self._forward_if_needed()
            record_time("MujocoAdapter.update_gui() forward done")
            self._last_gui_update_wtime = time.monotonic()
            self._viewer.sync(state_only=True)
        record_region_end("MujocoAdapter.update_gui()")

    def _close_gui(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = None
        self._viewer_mj_data = None

    def _build_name_maps(self):
        self._jid2jname : dict[int, tuple[str,str]] = {jid:self._mj_name_to_pair(mujoco_mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid))
                           for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname : dict[int, tuple[str,str]] = {lid:self._mj_name_to_pair(mujoco_mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, lid))
                           for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}
        self._cid2cname : dict[int, str] = {jid:self._mj_name_to_pair(mujoco_mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, jid))[1]
                           for jid in range(self._mj_model.ncam)}
        self._cname2cid = {cn:cid for cid,cn in self._cid2cname.items()}

    @staticmethod
    def _mj_name_to_pair(mjname: str) -> tuple[str, str]:
        if mjname == "world":
            return mjname, mjname
        sep = mjname.find(model_element_separator)
        if mjname.count(model_element_separator) != 1:
            raise RuntimeError(f"Invalid mjName '{mjname}': expected exactly one '{model_element_separator}' separator")
        return mjname[:sep], mjname[sep + len(model_element_separator):]

    @staticmethod
    def _pair_to_mj_name(pair: tuple[str, str]) -> str:
        if pair[0] == "world" or pair[1] == "world":
            return "world"
        return pair[0] + model_element_separator + pair[1]

    @override
    def get_detected_joints(self):
        return list(self._jname2jid.keys())

    @override
    def get_detected_links(self):
        return list(self._lname2lid.keys())

    @override
    def get_detected_cameras(self):
        return list(getattr(self, "_cname2cid", {}).keys())

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str, str]]):
        super().set_monitored_joints(jointsToObserve)
        self._reset_step_stats(self._step_stats_len)
    
    @override
    def set_monitored_links(self, linksToObserve: Sequence[tuple[str, str]]):
        super().set_monitored_links(linksToObserve)
        self._reset_step_stats(self._step_stats_len)

    @override
    def initialize_for_step(self):
        if self._mj_model is None or self._mj_data is None:
            return

    @override
    def step(self) -> float:
        self.initialize_for_step()
        self._reset_step_stats(self._step_stats_len)
        self._stepping = True
        duration = self.run(self._step_length_sec)
        self._stepping = False
        self._compute_stats()
        return duration

    @override
    def getRenderings(self, requestedCameras: list[str], vec_mask: th.Tensor | None = None) -> tuple[list[th.Tensor], th.Tensor]:
        if vec_mask is None:
            vec_mask = th.tensor([1], device=self._out_th_device, dtype=th.bool)
        images: list[th.Tensor] = []
        for cam in requestedCameras:
            width, height = self._camera_sizes_wh[cam]
            if not vec_mask.item():
                images.append(th.empty((0, height, width, 3), dtype=th.uint8, device=self._out_th_device))
            else:
                renderer = self._renderers[width, height]
                cam_id = self._cname2cid[cam]
                renderer.update_scene(self._mj_data, camera=cam_id, scene_option=self._render_scene_option)
                # if self._visualize_xfrc_applied:
                #     for body_id in range(0,self._mj_model.nbody):
                #         if np.linalg.norm(self._mj_data.xfrc_applied[body_id]) != 0.0:
                #             force_vec = self._mj_data.xfrc_applied[body_id,:3]
                #             body_pos = self._mj_data.xipos[body_id]
                #             add_arrow_to_renderer(renderer, body_pos, body_pos+force_vec/10, radius=0.03, rgba=[0.8, 0.1, 0.1, 1])
                img = renderer.render()
                images.append(th.as_tensor(img, device=self._out_th_device).unsqueeze(0))
        times = th.full((1, len(requestedCameras)), fill_value=float(self._mj_data.time), device=self._out_th_device, dtype=self._out_th_float_dtype)
        return images, times

    def _forward_if_needed(self):
        if self._forward_needed:
            mujoco_mj_forward(self._mj_model, self._mj_data)
            self._forward_needed = False

    @override
    def getJointsState(self, requestedJoints: Sequence[tuple[str, str]] | np.ndarray | None = None) -> th.Tensor:
        if requestedJoints is None:
            requestedJoints = self._monitored_joints
        self._forward_if_needed()

        if isinstance(requestedJoints, np.ndarray):
            jids = requestedJoints
        else:
            jids = [self._jname2jid[j] for j in requestedJoints]
        if len(jids) == 0:
            return th.empty((1, 0, 3), device=self._out_th_device, dtype=self._out_th_float_dtype)
        pve = self._read_joints_pve(np.array([jids]))
        return th.as_tensor(pve, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0).view(1, len(jids), 3)

    def _read_joints_pve(self, jids: np.ndarray) -> np.ndarray:
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco_mjtJoint.mjJNT_HINGE) | (jtypes == mujoco_mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        pos = self._mj_data.qpos[qpos_adrs]
        vel = self._mj_data.qvel[qvel_adrs]
        effort = self._mj_data.qfrc_applied[qvel_adrs]
        return np.stack([pos, vel, effort],axis=-1)

    def _read_joints_pvae(self, jids: np.ndarray) -> np.ndarray:
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco_mjtJoint.mjJNT_HINGE) | (jtypes == mujoco_mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        pos = self._mj_data.qpos[qpos_adrs]
        vel = self._mj_data.qvel[qvel_adrs]
        acc = self._mj_data.qacc[qpos_adrs]
        eff = self._mj_data.qfrc_smooth[qvel_adrs] + self._mj_data.qfrc_constraint[qvel_adrs]
        return np.stack([pos, vel, acc, eff], axis=-1)

    def _read_joints_pvaeep(self, jids: np.ndarray) -> np.ndarray:
        """Per-substep joint quantities for the extended step stats, ordered
        [position, velocity, acceleration, commanded_effort, sensed_effort, power]
        (matching MjxAdapter.get_joints_state_step_stats_extended)."""
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco_mjtJoint.mjJNT_HINGE) | (jtypes == mujoco_mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        pos = self._mj_data.qpos[qpos_adrs]
        vel = self._mj_data.qvel[qvel_adrs]
        acc = self._mj_data.qacc[qvel_adrs]
        commanded_effort = self._mj_data.qfrc_applied[qvel_adrs] + self._mj_data.qfrc_actuator[qvel_adrs]
        sensed_effort = commanded_effort + self._mj_data.qfrc_passive[qvel_adrs] + self._mj_data.qfrc_constraint[qvel_adrs]
        power = np.clip(vel * commanded_effort, 0.0, 1e6)
        return np.stack([pos, vel, acc, commanded_effort, sensed_effort, power], axis=-1)

    def _read_joints_pveae(self, jids: np.ndarray) -> np.ndarray:
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco_mjtJoint.mjJNT_HINGE) | (jtypes == mujoco_mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        pos = self._mj_data.qpos[qpos_adrs]
        vel = self._mj_data.qvel[qvel_adrs]
        applied_eff = self._mj_data.qfrc_applied[qvel_adrs]
        acc = self._mj_data.qacc[qpos_adrs]
        sensed_eff = self._mj_data.qfrc_smooth[qvel_adrs] + self._mj_data.qfrc_constraint[qvel_adrs]
        return np.stack([pos, vel, applied_eff, acc, sensed_eff], axis=-1)

    @override
    def getExtendedJointsState(self, requestedJoints: Sequence[tuple[str, str]] | None = None) -> th.Tensor:
        if requestedJoints is None:
            requestedJoints = self._monitored_joints
        if len(requestedJoints) == 0:
            return th.empty((1, 0, 5), device=self._out_th_device, dtype=self._out_th_float_dtype)
        jids = np.array([self._jname2jid[j] for j in requestedJoints], dtype=int)
        vals = self._read_joints_pveae(jids)
        return th.as_tensor(vals, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0)

    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        stats = th.as_tensor(self._joint_step_stats_mmas_j_pvaeep, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0).view(1,4,len(self._monitored_joints),6)
        return stats[:, :, :, :4]

    @override
    def get_joints_state_step_stats_extended(self) -> th.Tensor:
        return th.as_tensor(self._joint_step_stats_mmas_j_pvaeep, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0).view(1,4,len(self._monitored_joints),6)

    @override
    def get_links_state_step_stats(self) -> th.Tensor:
        return th.as_tensor(self._link_step_stats_mmas_l_vels, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0).view(1,4,len(self._monitored_links),6)

    def _set_effort_command(self, jids: np.ndarray, efforts: th.Tensor | np.ndarray, vec_mask: th.Tensor | None = None) -> None:
        if efforts.shape[0] != 1:
            raise ValueError("efforts must have leading dimension 1 for MujocoAdapter")
        if efforts.shape[1] != len(jids):
            raise ValueError(f"efforts has wrong shape {tuple(efforts.shape)}, expected (1,{len(jids)})")
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco_mjtJoint.mjJNT_HINGE) | (jtypes == mujoco_mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        if isinstance(efforts, th.Tensor):
            eff_np = efforts[0].detach().cpu().numpy()
        else:
            eff_np = efforts[0]
        self._requested_qfrc_applied[qvel_adrs] = eff_np

    def _apply_torques_cmd(self):
        # Replace qfrc_applied with requested torques
        self._mj_data.qfrc_applied[:] = 0.0
        if self._requested_qfrc_applied.size > 0:
            self._mj_data.qfrc_applied[:] = self._requested_qfrc_applied

    def _apply_commands(self):
        self._apply_torques_cmd()

    @override
    def getLinksState(self, requestedLinks: Sequence[tuple[str, str]] | np.ndarray | None = None, use_com_pose: bool = False) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        if len(requestedLinks) == 0:
            return th.empty((1, 0, 13), device=self._out_th_device, dtype=self._out_th_float_dtype)
        if isinstance(requestedLinks, np.ndarray):
            lids = requestedLinks
        else:
            lids = np.array([self._lname2lid[l] for l in requestedLinks], dtype=int)
        vals = self._read_links_state(lids, use_com_pose=use_com_pose)
        return th.as_tensor(vals, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0)

    def _read_link_velocities(self, link_names: Sequence[tuple[str, str]]) -> np.ndarray | None:
        if self._mj_model is None or self._mj_data is None:
            return None
        lids = np.array([self._lname2lid[ln] for ln in link_names], dtype=int)
        if len(lids) == 0:
            return np.empty((0, 6), dtype=np.float32)
        cvel = self._mj_data.cvel[lids]
        lin_vel = cvel[:, 3:6]
        ang_vel = cvel[:, 0:3]
        return np.concatenate([lin_vel, ang_vel], axis=-1)

    def _read_links_state(self, lids: np.ndarray, use_com_pose: bool) -> np.ndarray:
        if use_com_pose:
            pos = self._mj_data.xipos[lids]
            ximat = self._mj_data.ximat[lids]
            if ximat.shape[0] == 0:
                quat_xyzw = np.empty((0, 4), dtype=np.float32)
            else:
                quat_xyzw = np.stack([self._mat_to_quat_xyzw(m) for m in ximat], axis=0)
        else:
            pos = self._mj_data.xpos[lids]
            quat_wxyz = self._mj_data.xquat[lids]
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        cvel = self._mj_data.cvel[lids]
        lin_vel = cvel[:, 3:6]
        ang_vel = cvel[:, 0:3]
        return np.concatenate([pos, quat_xyzw, lin_vel, ang_vel], axis=-1)

    @staticmethod
    def _mat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
        # if rotmat.shape == (9,):
        #     rotmat = rotmat.reshape(3, 3)
        quat = mujoco_mju_mat2Quat(rotmat.flatten())
        return quat[[1,2,3,0]]

    @override
    def get_local_link_linear_acceleration(self, requestedLinks: Sequence[tuple[str, str]] | np.ndarray | None = None) -> th.Tensor:
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        if len(requestedLinks) == 0:
            return th.empty((1, 0, 3), device=self._out_th_device, dtype=self._out_th_float_dtype)
        self._ensure_ready()
        if isinstance(requestedLinks, np.ndarray):
            lids = requestedLinks
        else:
            lids = np.array([self._lname2lid[l] for l in requestedLinks], dtype=int)
        self._forward_if_needed()
        # mj_objectAcceleration reads mjData.cacc, which is only filled by mj_rnePostConstraint.
        # cacc is the com-based spatial acceleration with the world acceleration initialized to
        # -gravity, so the result is the proper (accelerometer-style) acceleration of the link.
        mujoco.mj_rnePostConstraint(self._mj_model, self._mj_data)
        accs = np.empty((len(lids), 3), dtype=np.float64)
        res = np.zeros(6, dtype=np.float64)  # [angular_xyz, linear_xyz]
        for i, lid in enumerate(lids):
            # flg_local=1 -> express the 6D acceleration in the link's local frame
            mujoco.mj_objectAcceleration(self._mj_model, self._mj_data, mujoco.mjtObj.mjOBJ_BODY, int(lid), res, 1)
            accs[i] = res[3:6]
        return th.as_tensor(accs, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0)

    @override
    def setJointsStateDirect(self, joint_names: Sequence[tuple[str, str]], joint_states_pve: th.Tensor, vec_mask: th.Tensor | None = None):
        if vec_mask is not None and not vec_mask.item():
            return
        if joint_states_pve.shape[0] != 1:
            raise ValueError("joint_states_pve must have leading dimension 1 for MujocoAdapter")
        jids = np.array([self._jname2jid[jn] for jn in joint_names], dtype=int)
        if joint_states_pve.shape[1] != len(jids) or joint_states_pve.shape[2] != 3:
            raise ValueError(f"joint_states_pve has wrong shape {tuple(joint_states_pve.shape)}, expected (1,{len(jids)},3)")
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco_mjtJoint.mjJNT_HINGE) | (jtypes == mujoco_mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        states_np = joint_states_pve[0].detach().cpu().numpy()
        self._mj_data.qpos[qpos_adrs] = states_np[:, 0]
        self._mj_data.qvel[qvel_adrs] = states_np[:, 1]
        self._mj_data.qfrc_applied[qvel_adrs] = states_np[:, 2]
        mujoco_mj_forward(self._mj_model, self._mj_data)

    def _move_free_links(self, lids: np.ndarray, link_states_pose_vel: np.ndarray):
        jnt_ids = self._mj_model.body_jntadr[lids]
        jtypes = self._mj_model.jnt_type[jnt_ids]
        if not np.all(jtypes == mujoco_mjtJoint.mjJNT_FREE):
            raise NotImplementedError(f"Only free joints are supported; got joint types {jtypes}")

        qpos_adrs = self._mj_model.jnt_qposadr[jnt_ids]
        qvel_adrs = self._mj_model.jnt_dofadr[jnt_ids]

        pos =       link_states_pose_vel[:, 0:3]
        quat_wxyz = link_states_pose_vel[:, 3:7][:, [3, 0, 1, 2]]
        lin_vel =   link_states_pose_vel[:, 7:10]
        ang_vel =   link_states_pose_vel[:, 10:13]

        pos_idx  = qpos_adrs[:, None] + np.arange(3)
        quat_idx = qpos_adrs[:, None] + 3 + np.arange(4)
        # mujoco free-joint qvel layout is [linear_xyz, angular_xyz]
        lin_vel_idx = qvel_adrs[:, None] + np.arange(3)
        ang_vel_idx = qvel_adrs[:, None] + 3 + np.arange(3)

        self._mj_data.qpos[pos_idx] = pos
        self._mj_data.qpos[quat_idx] = quat_wxyz
        self._mj_data.qvel[ang_vel_idx] = ang_vel
        self._mj_data.qvel[lin_vel_idx] = lin_vel

    def _move_fixed_links(self, lids: np.ndarray, link_states_pose_vel: np.ndarray):
        if lids.size == 0:
            return
        pos = link_states_pose_vel[:, 0:3].astype(self._mj_model.body_pos.dtype, copy=False)
        quat_wxyz = link_states_pose_vel[:, 3:7][:, [3, 0, 1, 2]].astype(self._mj_model.body_quat.dtype, copy=False)
        lin_vel = link_states_pose_vel[:, 7:10].astype(self._mj_data.cvel.dtype, copy=False)
        ang_vel = link_states_pose_vel[:, 10:13].astype(self._mj_data.cvel.dtype, copy=False)

        if np.any(lin_vel!=0.0) or np.any(ang_vel!=0.0):
            mask = (lin_vel!=0.0) | (ang_vel!=0.0)
            raise NotImplementedError(f"Setting velocities for fixed bodies is not supported in MujocoAdapter, but links {self.get_monitored_links_ids_names(lids[mask])} have non-zero velocities")

        mocap_ids = self._mj_model.body_mocapid[lids]
        mocap_mask = mocap_ids != -1
        if np.any(mocap_mask):
            mids = mocap_ids[mocap_mask]
            self._mj_data.mocap_pos[mids] = pos[mocap_mask]
            self._mj_data.mocap_quat[mids] = quat_wxyz[mocap_mask]
            if hasattr(self._mj_data, "mocap_velp"):
                self._mj_data.mocap_velp[mids] = lin_vel[mocap_mask]
            if hasattr(self._mj_data, "mocap_velr"):
                self._mj_data.mocap_velr[mids] = ang_vel[mocap_mask]

        static_mask = ~mocap_mask
        if np.any(static_mask):
            lids_static = lids[static_mask]
            self._mj_model.body_pos[lids_static] = pos[static_mask]
            self._mj_model.body_quat[lids_static] = quat_wxyz[static_mask]


    @override
    def setLinksStateDirect(self, link_names: Sequence[tuple[str, str]] | np.ndarray, link_states_pose_vel: th.Tensor, vec_mask: th.Tensor | None = None):
        if vec_mask is not None and not vec_mask.item():
            return
        if link_states_pose_vel.shape[0] != 1:
            raise ValueError("link_states_pose_vel must have leading dimension 1 for MujocoAdapter")
        if isinstance(link_names, np.ndarray):
            lids = link_names
        else:
            lids = np.array([self._lname2lid[ln] for ln in link_names], dtype=int)
        if link_states_pose_vel.shape[1] != len(lids) or link_states_pose_vel.shape[2] != 13:
            raise ValueError(f"link_states_pose_vel has wrong shape {tuple(link_states_pose_vel.shape)}, expected (1,{len(lids)},13)")

        link_states_pose_vel_np = link_states_pose_vel[0].cpu().numpy()

        root_body_ids = self._mj_model.body_rootid[lids]
        if not np.all(root_body_ids == lids):
            nonroot_lids = lids[root_body_ids != lids]
            raise NotImplementedError(f"Only root bodies can have their state set directly; links {self.get_monitored_links_ids_names(nonroot_lids)} are not root bodies")


        jnt_nums = self._mj_model.body_jntnum[lids]
        parent_body_ids = self._mj_model.body_parentid[lids]

        fixed_bodies_mask = (jnt_nums == 0) & (parent_body_ids == 0)
        fixed_bodies = lids[fixed_bodies_mask]
        floating_bodies_mask = (jnt_nums == 1) & (parent_body_ids == 0)
        floating_bodies = lids[floating_bodies_mask]
        unsupported_bodies = lids[(jnt_nums > 1) | (parent_body_ids != 0)]
        if len(unsupported_bodies) > 0:
            raise NotImplementedError(f"Only bodies attached directly to the world with floating or fixed joints can have their state set directly; links {self.get_monitored_links_ids_names(unsupported_bodies)} do not meet this requirement")

        # the links in fixed_bodies are "attached with fixed joints" we alter the model to move them

        self._move_fixed_links(fixed_bodies, link_states_pose_vel_np[fixed_bodies_mask])
        self._move_free_links(floating_bodies, link_states_pose_vel_np[floating_bodies_mask])
        
        mujoco_mj_forward(self._mj_model, self._mj_data)
            

    @override
    def setJointsEffortCommand(self, joint_names: Sequence[tuple[str, str]], efforts: th.Tensor, vec_mask: th.Tensor | None = None) -> None:
        if vec_mask is not None and not vec_mask.item():
            return
        jids = np.array([self._jname2jid[jn] for jn in joint_names], dtype=int)
        self._set_effort_command(jids=jids, efforts=efforts)

    @override
    def setupLight(self):
        raise NotImplementedError("Light setup is not implemented for MujocoAdapter")

    @override
    def spawn_models(self, models: Sequence[ModelSpawnDef]) -> list[str]:
        raise NotImplementedError("Spawning additional models at runtime is not implemented for MujocoAdapter")

    @override
    def delete_model(self, model_name: str):
        raise NotImplementedError("Deleting models at runtime is not implemented for MujocoAdapter")

    def _compute_collision_masks(self,  link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]],
                                        explicit_groups : list[tuple[tuple[str,str],...]] = []) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute geom/body contype-conaffinity bitmasks from a link->colliding-links specification.

        Mirrors MjxAdapter._compute_collision_masks: links are reorganized into a small set of
        collision groups (each a bit position, max 32), each link gets a contype mask (the groups
        it belongs to) and a conaffinity mask (the groups it collides with). Visual geoms
        (contype==0 and conaffinity==0) are left untouched.
        """
        input_collision_groups = [set(lg[1]) for lg in link_group_collisions]

        # Reorganize the links into a small set of groups of links that always collide together
        best_collision_groups : list[set[tuple[str,str]]] = []
        while len(input_collision_groups)>0:
            biggest_common_subgroup = set(input_collision_groups[0])
            for g in input_collision_groups:
                prev_biggest_common_subgroup = biggest_common_subgroup
                biggest_common_subgroup = biggest_common_subgroup.intersection(g)
                if len(biggest_common_subgroup)==0:
                    biggest_common_subgroup = prev_biggest_common_subgroup
            best_collision_groups.append(biggest_common_subgroup)
            input_collision_groups = [(g.difference(biggest_common_subgroup)) for g in input_collision_groups]
            input_collision_groups = [g for g in input_collision_groups if len(g)>0]

        best_collision_groups_set = {tuple(g) for g in best_collision_groups}
        best_collision_groups = [set(g) for g in best_collision_groups_set.union(set(explicit_groups))]
        link_to_group_ids = {} # Which groups each link is part of
        self._linkgroup_to_id : dict[tuple[tuple[str,str],...], int] = {}
        for i,g in enumerate(best_collision_groups):
            g_t = tuple(g)
            self._linkgroup_to_id[g_t] = i
            for l in g:
                link_to_group_ids.setdefault(l, []).append(i)

        link_colliding_groups : dict[tuple[str,str], list[int]] = {} # Which groups each link collides with
        for link,colliding_links in link_group_collisions:
            colliding_links = set(colliding_links)
            for i,g in enumerate(best_collision_groups):
                if g.issubset(colliding_links):
                    link_colliding_groups.setdefault(link, []).append(i)

        if len(best_collision_groups) > 32:
            raise RuntimeError(f"Detected more than 32 separate collision groups. Cannot represent in Mujoco collision masks.")

        nbody = self._mj_model.nbody
        all_links = list(self._lname2lid.keys())
        for l in all_links:
            link_to_group_ids.setdefault(l, [])
            link_colliding_groups.setdefault(l, [])
        for l in list(link_to_group_ids.keys())+list(link_colliding_groups.keys()):
            if l not in all_links:
                raise RuntimeError(f"Link {l} specified in link_group_collisions is not present in the model")

        link_contypes = {}
        link_conaffinity = {}
        for l in all_links:
            if self._lname2lid[l] >= nbody:
                continue  # skip non-body elements
            contype_mask = 0
            for gid in link_to_group_ids[l]:
                contype_mask |= 1<<gid
            link_contypes[l] = contype_mask
            conaffinity_mask = 0
            for gid in link_colliding_groups[l]:
                conaffinity_mask |= 1<<gid
            link_conaffinity[l] = conaffinity_mask

        body_contype : np.ndarray = self._mj_model.body_contype.copy()
        body_conaffinity : np.ndarray = self._mj_model.body_conaffinity.copy()
        geom_contype : np.ndarray = self._mj_model.geom_contype.copy()
        geom_conaffinity : np.ndarray = self._mj_model.geom_conaffinity.copy()
        for lname in all_links:
            body_id = self._lname2lid[lname]
            if body_id >= nbody:
                continue
            aff = link_conaffinity[lname]
            typ = link_contypes[lname]
            body_conaffinity[body_id] = aff
            body_contype[body_id] = typ
            for geom_id in range(self._mj_model.body_geomadr[body_id],
                                 self._mj_model.body_geomadr[body_id]+self._mj_model.body_geomnum[body_id]):
                visual = geom_contype[geom_id]==0 and geom_conaffinity[geom_id]==0
                if not visual:
                    geom_conaffinity[geom_id] = aff
                    geom_contype[geom_id] = typ
        return geom_contype, geom_conaffinity, body_contype, body_conaffinity

    @override
    def set_body_collisions(self, link_group_collisions: list[tuple[tuple[str, str], list[tuple[str, str]]]],
                            explicit_groups: list[tuple[tuple[str, str], ...]] = []):
        self._ensure_ready()
        geom_contype, geom_conaffinity, body_contype, body_conaffinity = self._compute_collision_masks(link_group_collisions, explicit_groups)
        self._mj_model.geom_contype[:] = geom_contype
        self._mj_model.geom_conaffinity[:] = geom_conaffinity
        self._mj_model.body_contype[:] = body_contype
        self._mj_model.body_conaffinity[:] = body_conaffinity
        self._forward_needed = True

    @override
    def set_link_impulses(self, link_ids: Sequence[Any],
                          force_torque_xyzxyz: th.Tensor,
                          durations: th.Tensor, delays: th.Tensor,
                          vec_mask: th.Tensor) -> None:
        raise NotImplementedError("Setting link impulses is not implemented for MujocoAdapter")

    @override
    def sim_step_duration(self) -> th.Tensor:
        return self._sim_step_dt_th

    @override
    def resetWorld(self):
        # Reset in place rather than reallocating, so a gui viewer bound to this MjData keeps
        # following the simulation across episodes (and to avoid a per-episode allocation).
        if self._mj_data is None:
            self._mj_data = mujoco_MjData(self._mj_model)
        else:
            mujoco_mj_resetData(self._mj_model, self._mj_data)
        mujoco_mj_forward(self._mj_model, self._mj_data)
        self._forward_needed = False
        self._requested_qfrc_applied = np.zeros((self._mj_model.nv,), dtype=np.float64)
        self.initialize_for_episode()

    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._time_since_startup 

    def _reset_step_stats(self, max_substeps_count: int):
        self._recorded_stats_substeps = 0
        if max_substeps_count < self._step_length_sec/self._sim_step_dt:
            raise RuntimeError(f"max_substeps_count {max_substeps_count} is less than expected number of substeps per step "
                          f"{int(self._step_length_sec/self._sim_step_dt)}; step stats would be incomplete.")
        self._step_stats_buff_len = max_substeps_count
        self._joint_pvaeep_step_history = np.zeros((max_substeps_count, len(self._monitored_joints), 6), dtype=np.float32)
        self._link_vels_step_history = np.zeros((max_substeps_count, len(self._monitored_links), 6), dtype=np.float32)
        self._joint_step_stats_mmas_j_pvaeep = np.zeros((4, len(self._monitored_joints), 6), dtype=np.float32)
        self._link_step_stats_mmas_l_vels = np.zeros((4, len(self._monitored_links), 6), dtype=np.float32)


    def _update_step_stats(self):
        if not self._stepping:
            return
        if len(self._monitored_joints) > 0:
            joint_pvaeep = self._read_joints_pvaeep(np.array([self._jname2jid[jn] for jn in self._monitored_joints], dtype=int))
            self._joint_pvaeep_step_history[self._recorded_stats_substeps%self._step_stats_buff_len] = joint_pvaeep
        if len(self._monitored_links) > 0:
            link_vels = self._read_link_velocities(self._monitored_links)
            self._link_vels_step_history[self._recorded_stats_substeps%self._step_stats_buff_len] = link_vels
        self._recorded_stats_substeps += 1

    def _compute_stats(self) -> None:
        l = min(self._recorded_stats_substeps, self._step_stats_buff_len)
        if l == 0:
            return
        jmins = self._joint_pvaeep_step_history[:l].min(axis=0)
        jmaxs = self._joint_pvaeep_step_history[:l].max(axis=0)
        javg = self._joint_pvaeep_step_history[:l].mean(axis=0)
        jstd = self._joint_pvaeep_step_history[:l].std(axis=0)
        self._joint_step_stats_mmas_j_pvaeep = np.stack([jmins, jmaxs, javg, jstd], axis=0)
        lmins = self._link_vels_step_history[:l].min(axis=0)
        lmaxs = self._link_vels_step_history[:l].max(axis=0)
        lavg = self._link_vels_step_history[:l].mean(axis=0)
        lstd = self._link_vels_step_history[:l].std(axis=0)
        self._link_step_stats_mmas_l_vels = np.stack([lmins, lmaxs, lavg, lstd], axis=0)

    @override
    def run(self, duration_sec: float):
        target_time = self._mj_data.time + duration_sec
        start_time = self._mj_data.time
        while self._mj_data.time < target_time:
            self._apply_commands()
            mujoco.mj_step(self._mj_model, self._mj_data)  # type: ignore[arg-type]
            self._update_step_stats()
        ran_time = float(self._mj_data.time - start_time)
        self._time_since_startup += ran_time
        self._forward_needed = True
        self._update_gui()
        return ran_time

    @override
    def get_monitored_links_ids(self, link_names: Sequence[tuple[str, str]]):
        return np.array([self._lname2lid[ln] for ln in link_names], dtype=int)

    @override
    def get_monitored_links_ids_names(self, link_ids: Sequence[int] | np.ndarray):
        return [self._lid2lname[lid] for lid in link_ids]

    @override
    def get_monitored_joints_ids(self, joint_names: Sequence[tuple[str, str]]):
        return np.array([self._jname2jid[jn] for jn in joint_names], dtype=int)
