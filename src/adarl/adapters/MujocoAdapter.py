from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch as th
import mujoco

from typing import Sequence, Any
from typing_extensions import override

from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import compile_xacro_string
from adarl.adapters.MjxAdapter import aggregate_models, apply_opt_preset, add_arrow_to_renderer
import copy
import adarl.utils.dbg.ggLog as ggLog
import pprint

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
                 output_th_device: th.device = th.device("cpu")):
        if vec_size != 1:
            raise ValueError("MujocoAdapter only supports vec_size=1")
        super().__init__(vec_size=vec_size, output_th_device=output_th_device)
        self._sim_step_dt = float(sim_step_dt)
        self._sim_step_dt_th = th.as_tensor(self._sim_step_dt, device=output_th_device)
        self._mj_model: mujoco.MjModel = None
        self._mj_data: mujoco.MjData | None = None
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
        self._opt_preset = "mujoco_default"
        self._opt_override = None
        self._enable_rendering = True
        self._stepping = False
        self._step_length_sec = step_length_sec
        self._forward_needed = True
        self._time_since_startup = 0.0
        self._step_stats_len = 1000

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

        self._mj_model, spec = aggregate_models(models,
                                          add_ground=self._add_ground,
                                          add_sky=self._add_sky,
                                          uneven_ground=self._uneven_ground,
                                          discardvisual=self._discardvisual)
        self._mj_model = apply_opt_preset(self._mj_model, self._opt_preset, self._opt_override)
        
        self._mj_model.opt.timestep = self._sim_step_dt
        self._mj_data = mujoco.MjData(self._mj_model)
        self._requested_qfrc_applied = np.zeros((self._mj_model.nv,), dtype=np.float64)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._renderer_cache.clear()
        self._build_name_maps()
        self._build_renderers()

        self.set_monitored_joints([])
        self.set_monitored_links([])
        self._reset_step_stats(self._step_stats_len)

        ggLog.info(f"Joints:\n{pprint.pformat(self._jname2jid)}")
        ggLog.info(f"Links:\n{pprint.pformat(self._lname2lid)}")
        ggLog.info(f"Cameras:\n{pprint.pformat(self._cname2cid)}")

    def _build_renderers(self):
        self._camera_sizes_wh :dict[str,tuple[int,int]] = {self._cid2cname[cid]:(self._mj_model.cam_resolution[cid][1],self._mj_model.cam_resolution[cid][0]) for cid in self._cid2cname}
        if self._enable_rendering:
            self._render_scene_option = mujoco.MjvOption()
            self._render_scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            # self._render_scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1
            # self._render_scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            ggLog.info(f"Making rederer for resolutions: {set(self._camera_sizes_wh.values())}, MUJOCO_GL={os.environ.get('MUJOCO_GL','<not set>')}")
            self._renderers : dict[tuple[int,int],mujoco.Renderer]= {resolution: mujoco.Renderer(self._mj_model,resolution[0],resolution[1])
                                                                        for resolution in set(self._camera_sizes_wh.values())}
        else:
            self._renderers = {}

    @override
    def destroy_scenario(self, **kwargs):
        self._mj_model = None
        self._mj_data = None
        self._renderers = {}
        self._renderers_mj_data = []

    def _build_name_maps(self):
        self._jid2jname : dict[int, tuple[str,str]] = {jid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid))
                           for jid in range(self._mj_model.njnt)}
        self._jname2jid = {jn:jid for jid,jn in self._jid2jname.items()}
        self._lid2lname : dict[int, tuple[str,str]] = {lid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, lid))
                           for lid in range(self._mj_model.nbody)}
        self._lname2lid = {ln:lid for lid,ln in self._lid2lname.items()}
        self._cid2cname : dict[int, str] = {jid:self._mj_name_to_pair(mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, jid))[1]
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
                if self._visualize_xfrc_applied:
                    for body_id in range(0,self._mj_model.nbody):
                        if np.linalg.norm(self._mj_data.xfrc_applied[body_id]) != 0.0:
                            force_vec = self._mj_data.xfrc_applied[body_id,:3]
                            body_pos = self._mj_data.xipos[body_id]
                            add_arrow_to_renderer(renderer, body_pos, body_pos+force_vec/10, radius=0.03, rgba=[0.8, 0.1, 0.1, 1])
                img = renderer.render()
                images.append(th.as_tensor(img, device=self._out_th_device).unsqueeze(0))
        times = th.full((1, len(requestedCameras)), fill_value=float(self._mj_data.time), device=self._out_th_device, dtype=self._out_th_float_dtype)
        return images, times

    def _forward_if_needed(self):
        if self._forward_needed:
            mujoco.mj_forward(self._mj_model, self._mj_data)

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
        if not np.all((jtypes == mujoco.mjtJoint.mjJNT_HINGE) | (jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        pos = self._mj_data.qpos[qpos_adrs]
        vel = self._mj_data.qvel[qvel_adrs]
        effort = self._mj_data.qfrc_applied[qvel_adrs]
        return np.stack([pos, vel, effort],axis=-1)

    def _read_joints_pvae(self, jids: np.ndarray) -> np.ndarray:
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco.mjtJoint.mjJNT_HINGE) | (jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        pos = self._mj_data.qpos[qpos_adrs]
        vel = self._mj_data.qvel[qvel_adrs]
        acc = self._mj_data.qacc[qpos_adrs]
        eff = self._mj_data.qfrc_smooth[qvel_adrs] + self._mj_data.qfrc_constraint[qvel_adrs]
        return np.stack([pos, vel, acc, eff], axis=-1)

    def _read_joints_pveae(self, jids: np.ndarray) -> np.ndarray:
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco.mjtJoint.mjJNT_HINGE) | (jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
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
        return th.as_tensor(self._joint_step_stats_mmas_j_pvea, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0).view(1,4,len(self._monitored_joints),4)

    @override
    def get_links_state_step_stats(self) -> th.Tensor:
        return th.as_tensor(self._link_step_stats_mmas_l_vels, device=self._out_th_device, dtype=self._out_th_float_dtype).unsqueeze(0).view(1,4,len(self._monitored_links),6)

    def _set_effort_command(self, jids: np.ndarray, efforts: th.Tensor | np.ndarray, vec_mask: th.Tensor | None = None) -> None:
        if efforts.shape[0] != 1:
            raise ValueError("efforts must have leading dimension 1 for MujocoAdapter")
        if efforts.shape[1] != len(jids):
            raise ValueError(f"efforts has wrong shape {tuple(efforts.shape)}, expected (1,{len(jids)})")
        jtypes = self._mj_model.jnt_type[jids]
        if not np.all((jtypes == mujoco.mjtJoint.mjJNT_HINGE) | (jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
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
        ggLog.info(f"Getting link states for links: {requestedLinks}")
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
        quat = mujoco.mju_mat2Quat(rotmat.flatten())
        return quat[[1,2,3,0]]

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
        if not np.all((jtypes == mujoco.mjtJoint.mjJNT_HINGE) | (jtypes == mujoco.mjtJoint.mjJNT_SLIDE)):
            raise NotImplementedError(f"Joint types other than HINGE and SLIDE are not supported, but got types {jtypes}")
        qpos_adrs = self._mj_model.jnt_qposadr[jids]
        qvel_adrs = self._mj_model.jnt_dofadr[jids]
        states_np = joint_states_pve[0].detach().cpu().numpy()
        self._mj_data.qpos[qpos_adrs] = states_np[:, 0]
        self._mj_data.qvel[qvel_adrs] = states_np[:, 1]
        self._mj_data.qfrc_applied[qvel_adrs] = states_np[:, 2]
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def _move_free_links(self, lids: np.ndarray, link_states_pose_vel: np.ndarray):
        jnt_ids = self._mj_model.body_jntadr[lids]
        jtypes = self._mj_model.jnt_type[jnt_ids]
        if not np.all(jtypes == mujoco.mjtJoint.mjJNT_FREE):
            raise NotImplementedError(f"Only free joints are supported; got joint types {jtypes}")

        qpos_adrs = self._mj_model.jnt_qposadr[jnt_ids]
        qvel_adrs = self._mj_model.jnt_dofadr[jnt_ids]

        pos =       link_states_pose_vel[:, 0:3]
        quat_wxyz = link_states_pose_vel[:, 3:7][:, [3, 0, 1, 2]]
        lin_vel =   link_states_pose_vel[:, 7:10]
        ang_vel =   link_states_pose_vel[:, 10:13]

        pos_idx  = qpos_adrs[:, None] + np.arange(3)
        quat_idx = qpos_adrs[:, None] + 3 + np.arange(4)
        ang_vel_idx = qvel_adrs[:, None] + np.arange(3)
        lin_vel_idx = qvel_adrs[:, None] + 3 + np.arange(3)

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
        
        mujoco.mj_forward(self._mj_model, self._mj_data)
            

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

    @override
    def set_body_collisions(self, link_group_collisions: list[tuple[tuple[str, str], list[tuple[str, str]]]]):
        raise NotImplementedError("Setting body collisions is not implemented for MujocoAdapter")

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
        self._mj_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)
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
        self._joint_pvea_step_history = np.zeros((max_substeps_count, len(self._monitored_joints), 4), dtype=np.float32)
        self._link_vels_step_history = np.zeros((max_substeps_count, len(self._monitored_links), 6), dtype=np.float32)
        self._joint_step_stats_mmas_j_pvea = np.zeros((4, len(self._monitored_joints), 4), dtype=np.float32)
        self._link_step_stats_mmas_l_vels = np.zeros((4, len(self._monitored_links), 6), dtype=np.float32)


    def _update_step_stats(self):
        if not self._stepping:
            return
        if len(self._monitored_joints) > 0:
            joint_pvea = self._read_joints_pvae(np.array([self._jname2jid[jn] for jn in self._monitored_joints], dtype=int))
            self._joint_pvea_step_history[self._recorded_stats_substeps%self._step_stats_buff_len] = joint_pvea
        if len(self._monitored_links) > 0:
            link_vels = self._read_link_velocities(self._monitored_links)
            self._link_vels_step_history[self._recorded_stats_substeps%self._step_stats_buff_len] = link_vels
        self._recorded_stats_substeps += 1

    def _compute_stats(self) -> np.ndarray:
        l = min(self._recorded_stats_substeps, self._step_stats_buff_len)
        jmins = self._joint_pvea_step_history[:l].min(axis=0)
        jmaxs = self._joint_pvea_step_history[:l].max(axis=0)
        javg = self._joint_pvea_step_history[:l].mean(axis=0)
        jstd = self._joint_pvea_step_history[:l].std(axis=0)
        self._joint_step_stats_mmas_j_pvea = np.stack([jmins, jmaxs, javg, jstd], axis=0)
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
