from __future__ import annotations
"""Vectorized simulation adapter backed by the Genesis physics engine (https://github.com/Genesis-Embodied-AI/Genesis)."""
import os
import tempfile
import numpy as np
import torch as th

from dataclasses import dataclass
from typing import Any, Sequence
from typing_extensions import override

import genesis as gs

from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseVecAdapter import JointProperties, JointType
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.base_utils import compile_xacro_string
from adarl.utils.utils import th_quat_rotate, th_quat_conj
import adarl.utils.dbg.ggLog as ggLog


def ensure_genesis_initialized(th_device: th.device, logging_level: str = "warning"):
    """Initialize genesis if it was not initialized yet. Genesis only supports being initialized once per process."""
    if getattr(gs, "_initialized", False):
        return
    gs.init(backend=gs.gpu if th_device.type == "cuda" else gs.cpu, logging_level=logging_level)


def _quat_xyzw_to_R_th(q: th.Tensor) -> th.Tensor:
    """Convert quaternions of shape (...,4) in xyzw order to rotation matrices of shape (...,3,3)."""
    x, y, z, w = q.unbind(-1)
    return th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                     2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
                    dim=-1).reshape(q.shape[:-1] + (3, 3))


def _pose7_to_T_th(pose_xyz_xyzw: th.Tensor) -> th.Tensor:
    """Convert poses of shape (...,7) (position_xyz + quaternion_xyzw) to 4x4 transforms of shape (...,4,4)."""
    batch_shape = pose_xyz_xyzw.shape[:-1]
    T = th.zeros(batch_shape + (4, 4), device=pose_xyz_xyzw.device, dtype=pose_xyz_xyzw.dtype)
    T[..., :3, :3] = _quat_xyzw_to_R_th(pose_xyz_xyzw[..., 3:7])
    T[..., :3, 3] = pose_xyz_xyzw[..., 0:3]
    T[..., 3, 3] = 1.0
    return T


def _pos_lookat_up_to_T_np(pos, lookat, up) -> np.ndarray:
    """Build a camera transform (-Z forward, +Y up convention, like genesis and mujoco) from pos/lookat/up."""
    pos = np.asarray(pos, dtype=np.float64)
    z = pos - np.asarray(lookat, dtype=np.float64)
    z = z / np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    y = up - np.dot(up, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


def _pos_quat_wxyz_to_T_np(pos, quat_wxyz) -> np.ndarray:
    import mujoco
    R = np.zeros((9,), dtype=np.float64)
    mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _T_np_to_pose7_th(T: np.ndarray, device, dtype=th.float32) -> th.Tensor:
    import mujoco
    quat_wxyz = np.zeros((4,), dtype=np.float64)
    mujoco.mju_mat2Quat(quat_wxyz, T[:3, :3].flatten())
    pose = np.concatenate([T[:3, 3], quat_wxyz[[1, 2, 3, 0]]])
    return th.as_tensor(pose, device=device, dtype=dtype)


@dataclass
class GenesisCameraDef:
    """Definition of a camera to be placed in the simulation. Cameras can also be parsed automatically
    from mjcf models (see GenesisAdapter._parse_mjcf_cameras), these definitions allow to place cameras manually."""
    name: str
    width: int = 320
    height: int = 240
    pose_xyz: tuple[float, float, float] = (0.0, 2.0, 1.0)
    lookat_xyz: tuple[float, float, float] = (0.0, 0.0, 0.5)
    up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float = 60.0
    near: float = 0.05
    far: float = 100.0
    attach_link: tuple[str, str] | None = None
    attach_offset_T: Any | None = None  # 4x4 homogeneous transform from link frame to camera frame


@dataclass
class _ParsedCameraSpec:
    """A camera extracted from an mjcf model definition."""
    cam_name: str
    model_name: str
    body_name: str
    width: int
    height: int
    fovy_deg: float
    T_body_cam: np.ndarray   # camera pose in the parent body frame
    T_world_body: np.ndarray  # parent body pose in the world frame (at the model reference configuration)
    attach_to_entity_link: bool  # True if the model is also loaded as a genesis entity, and the camera should follow its link


@dataclass
class _CameraEntry:
    name: str
    width: int
    height: int
    cam: Any | None = None  # genesis camera, None if rendering is disabled
    virtual_link: tuple[str, str] | None = None
    offset_T: th.Tensor | None = None  # (4,4) transform from the virtual link frame to the camera frame
    attached: bool = False
    dirty: bool = False  # pose was changed since the last render
    link_pose_xyz_xyzw: th.Tensor | None = None  # (vec_size,7) current virtual link pose


@dataclass
class _ImpulsesSpec:
    links_idx: list[int]
    force_torque_xyzxyz: th.Tensor  # (vec_size, n_links, 6)
    start_times: th.Tensor          # (vec_size, n_links)
    end_times: th.Tensor            # (vec_size, n_links)


_gs2adarl_jtype = {gs.JOINT_TYPE.REVOLUTE:  JointType.REVOLUTE,
                   gs.JOINT_TYPE.PRISMATIC: JointType.PRISMATIC,
                   gs.JOINT_TYPE.FIXED:     JointType.FIXED,
                   gs.JOINT_TYPE.FREE:      JointType.FLOATING,
                   gs.JOINT_TYPE.SPHERICAL: JointType.SPHERICAL}


@dataclass
class _LinkSetInfo:
    links_idx: list[int]
    free_pos: list[int]       # positions (within the request) of links with a free base joint
    free_dofs_idx: list[int]  # flattened global dof indices of the free base joints, 6 per free link
    nonfree_pos: list[int]    # positions of links without a free base joint


class GenesisAdapter(BaseVecSimulationAdapter, BaseVecJointEffortAdapter):
    """Adapter for the Genesis simulator (GPU-batched, torch-native).

    Models are loaded from urdf/mjcf definition strings (optionally xacro-compiled). Genesis does not
    namespace names, so joints/links are identified as (model_name, element_name) using the per-model
    name maps built at build_scenario time.

    Cameras can be defined manually with GenesisCameraDef objects, or are parsed automatically from
    mjcf models (mujoco is used for the parsing). Mjcf models containing only cameras (no geoms and
    no joints) are not loaded as genesis entities: their cameras are created as static cameras, and
    each camera body is exposed as a virtual link (model_name, body_name) whose pose can be set with
    setLinksStateDirect() to move the camera (mimicking how camera bodies can be moved in mujoco).
    Cameras parsed from models that do contain geoms are attached to the corresponding entity link
    and follow it. Genesis and mujoco use the same camera frame convention (-Z forward, +Y up), so
    parsed cameras render with the same pose semantics as in the mujoco-based adapters.

    Virtual camera links get synthetic negative link ids, so they can be passed through the id-based
    interface (get_links_ids / getLinksState / setLinksStateDirect) just like physical links, which
    have non-negative genesis ids. They only carry a pose (no velocity/acceleration dynamics).

    Per-model loading options can be passed through ModelSpawnDef.kwargs with the special keys
    'genesis_fixed', 'genesis_merge_fixed_links', 'genesis_scale', 'genesis_gravity_compensation'
    and 'genesis_morph_kwargs'; the remaining kwargs are used as xacro arguments for xacro models.
    """

    def __init__(self,
                 vec_size: int,
                 output_th_device: th.device = th.device("cuda", 0),
                 sim_step_dt: float = 1 / 256,
                 step_length_sec: float = 12 / 256,
                 enable_rendering: bool = False,
                 use_raytracer: bool = False,
                 cameras: Sequence[GenesisCameraDef] = (),
                 add_ground: bool = True,
                 show_gui: bool = False,
                 sim_options_override: dict[str, Any] | None = None,
                 rigid_options_override: dict[str, Any] | None = None,
                 genesis_logging_level: str = "warning",
                 log_folder: str = "./"):
        super().__init__(vec_size=vec_size, output_th_device=output_th_device)
        ensure_genesis_initialized(output_th_device, genesis_logging_level)
        self._sim_step_dt = float(sim_step_dt)
        self._step_length_sec = float(step_length_sec)
        self._enable_rendering = enable_rendering
        self._use_raytracer = use_raytracer
        self._camera_defs = list(cameras)
        self._add_ground = add_ground
        self._show_gui = show_gui
        self._sim_options_override = dict(sim_options_override) if sim_options_override else {}
        self._rigid_options_override = dict(rigid_options_override) if rigid_options_override else {}
        self._log_folder = log_folder
        self._sim_step_dt_th = th.as_tensor(self._sim_step_dt, device=self._out_th_device)

        self._scene = None
        self._entities: dict[str, Any] = {}
        self._joints: dict[tuple[str, str], Any] = {}
        self._links: dict[tuple[str, str], Any] = {}
        self._jname2ids: dict[tuple[str, str], tuple[int, int]] = {}  # name -> (global_dof_idx, global_q_idx), 1-dof joints only
        self._dofidx2jname: dict[int, tuple[str, str]] = {}
        self._linkidx2lname: dict[int, tuple[str, str]] = {}  # genesis link id (>=0) or synthetic camera id (<0) -> name
        self._cameras: dict[str, _CameraEntry] = {}
        self._camera_links: dict[tuple[str, str], _CameraEntry] = {}
        self._camlink_name2id: dict[tuple[str, str], int] = {}  # virtual camera link name -> synthetic negative id
        self._camlink_id2entry: dict[int, _CameraEntry] = {}    # synthetic negative id -> camera entry
        self._joint_ids_cache: dict[tuple, th.Tensor] = {}
        self._link_ids_cache: dict[tuple, th.Tensor] = {}
        self._link_set_info_cache: dict[tuple, _LinkSetInfo] = {}
        self._sim_dev: th.device = th.device("cpu")
        self._sim_time = 0.0
        self._stepping = False
        self._tmp_dir: str | None = None
        self._gravity_vec_w = th.as_tensor(self._sim_options_override.get("gravity", (0.0, 0.0, -9.81)))
        self._monitored_jids = th.empty((0, 2), dtype=th.long)
        self._monitored_lids = th.empty((0,), dtype=th.long)
        self._prev_dofs_vel: th.Tensor | None = None
        self._dofs_acc: th.Tensor | None = None
        self._impulses: _ImpulsesSpec | None = None
        self._reset_step_stats()

    def _rigid_solver(self):
        if self._scene is None:
            raise RuntimeError("Scenario is not built. Call build_scenario() first.")
        return self._scene.rigid_solver

    # =================================================================================
    #                                 scenario building
    # =================================================================================

    @override
    def build_scenario(self, models: Sequence[ModelSpawnDef] = (), **kwargs):
        if self._scene is not None:
            raise RuntimeError("Scenario was already built. Call destroy_scenario() first.")
        camera_defs = kwargs.pop("cameras", None)
        if camera_defs is not None:
            self._camera_defs = list(camera_defs)
        if self._enable_rendering:
            renderer = gs.options.renderers.BatchRenderer(use_rasterizer=not self._use_raytracer)
        else:
            renderer = gs.options.renderers.Rasterizer()
        self._scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self._sim_step_dt,
                                                                 substeps=1,
                                                                 **self._sim_options_override),
                               rigid_options=gs.options.RigidOptions(**self._rigid_options_override),
                               renderer=renderer,
                               show_viewer=self._show_gui,
                               show_FPS=False)
        self._entities = {}
        ground_entity = None
        if self._add_ground:
            ground_entity = self._scene.add_entity(gs.morphs.Plane(), name="ground")
        self._tmp_dir = tempfile.mkdtemp(prefix="GenesisAdapter_models_")
        parsed_cam_specs: list[_ParsedCameraSpec] = []
        for model in models:
            entity, cam_specs = self._add_model(model)
            if entity is not None:
                self._entities[model.name] = entity
            parsed_cam_specs.extend(cam_specs)

        # cameras must be created before scene.build(), their pose/attachment is finalized after build
        cams_to_finalize: list[tuple[Any, _CameraEntry]] = []
        for cd in self._camera_defs:
            entry = _CameraEntry(name=cd.name, width=cd.width, height=cd.height,
                                 virtual_link=(cd.name, f"{cd.name}_link") if cd.attach_link is None else None)
            if self._enable_rendering:
                entry.cam = self._scene.add_camera(res=(cd.width, cd.height),
                                                   pos=cd.pose_xyz,
                                                   lookat=cd.lookat_xyz,
                                                   up=cd.up_xyz,
                                                   fov=cd.fov_deg,
                                                   near=cd.near,
                                                   far=cd.far,
                                                   GUI=False)
            else:
                ggLog.warn(f"GenesisAdapter: camera '{cd.name}' was defined but enable_rendering is False, it will not render.")
            self._cameras[cd.name] = entry
            cams_to_finalize.append((cd, entry))
        for spec in parsed_cam_specs:
            if spec.cam_name in self._cameras:
                ggLog.warn(f"GenesisAdapter: camera '{spec.cam_name}' parsed from model '{spec.model_name}' has the "
                           f"same name as an already defined camera, skipping it.")
                continue
            entry = _CameraEntry(name=spec.cam_name, width=spec.width, height=spec.height,
                                 virtual_link=None if spec.attach_to_entity_link else (spec.model_name, spec.body_name))
            if self._enable_rendering:
                entry.cam = self._scene.add_camera(res=(spec.width, spec.height),
                                                   fov=spec.fovy_deg,
                                                   GUI=False)
            else:
                ggLog.info(f"GenesisAdapter: camera '{spec.cam_name}' parsed from model '{spec.model_name}', but "
                           f"enable_rendering is False, it will not render.")
            self._cameras[spec.cam_name] = entry
            cams_to_finalize.append((spec, entry))

        self._scene.build(n_envs=self._vec_size)

        self._joints = {}
        self._links = {}
        for mname, ent in self._entities.items():
            for j in ent.joints:
                self._joints[(mname, j.name)] = j
            for l in ent.links:
                self._links[(mname, l.name)] = l
        if ground_entity is not None:
            self._links[("ground", "ground_link")] = ground_entity.links[0]
        self._jname2ids = {}
        for jname, j in self._joints.items():
            if j.n_dofs == 1:
                self._jname2ids[jname] = (int(j.dofs_idx[0]), int(j.qs_idx[0]))
        self._dofidx2jname = {ids[0]: jn for jn, ids in self._jname2ids.items()}
        self._linkidx2lname = {int(l.idx): ln for ln, l in self._links.items()}
        self._joint_ids_cache = {}
        self._link_ids_cache = {}
        self._link_set_info_cache = {}

        rs = self._rigid_solver()
        all_vels = rs.get_dofs_velocity()
        self._sim_dev = all_vels.device
        self._prev_dofs_vel = all_vels
        self._dofs_acc = th.zeros_like(all_vels)
        self._gravity_vec_w = self._gravity_vec_w.to(device=self._sim_dev)

        self._camera_links = {}
        for src, entry in cams_to_finalize:
            self._finalize_camera(src, entry)

        # assign synthetic negative ids to the virtual camera links, so they are addressable through
        # the id-based interface alongside the physical links (which have non-negative genesis ids)
        self._camlink_name2id = {}
        self._camlink_id2entry = {}
        for i, (lname, entry) in enumerate(self._camera_links.items()):
            neg_id = -(i + 1)
            self._camlink_name2id[lname] = neg_id
            self._camlink_id2entry[neg_id] = entry
            self._linkidx2lname[neg_id] = lname

        # recompute the monitored sets in terms of the new scene
        self.set_monitored_joints(self._monitored_joints)
        self.set_monitored_links(self._monitored_links)
        ggLog.info(f"GenesisAdapter built scenario: joints={list(self._joints.keys())} links={list(self._links.keys())} "
                   f"cameras={list(self._cameras.keys())} camera_links={list(self._camera_links.keys())}")

    def _finalize_camera(self, src, entry: _CameraEntry):
        """Finalize a camera after scene.build(): set its initial pose, attach it to its link, register its virtual link."""
        eye = th.eye(4, device=self._sim_dev, dtype=th.float32)
        if isinstance(src, GenesisCameraDef):
            if src.attach_link is not None:
                if entry.cam is not None:
                    link = self._links[src.attach_link]
                    offset = np.asarray(src.attach_offset_T, dtype=np.float64) if src.attach_offset_T is not None else np.eye(4)
                    entry.cam.attach(link, offset)
                    entry.attached = True
                return
            entry.offset_T = eye
            T0 = _pos_lookat_up_to_T_np(src.pose_xyz, src.lookat_xyz, src.up_xyz)
            entry.link_pose_xyz_xyzw = _T_np_to_pose7_th(T0, self._sim_dev).expand(self._vec_size, 7).clone()
        else:  # _ParsedCameraSpec
            spec: _ParsedCameraSpec = src
            if spec.attach_to_entity_link:
                if entry.cam is not None:
                    link = self._links.get((spec.model_name, spec.body_name))
                    if link is None:
                        raise RuntimeError(f"Camera '{spec.cam_name}' is attached to body '{spec.body_name}' of model "
                                           f"'{spec.model_name}', but no such link exists in the genesis scene "
                                           f"(was it merged? consider genesis_merge_fixed_links=False).")
                    entry.cam.attach(link, spec.T_body_cam)
                    entry.attached = True
                return
            entry.offset_T = th.as_tensor(spec.T_body_cam, device=self._sim_dev, dtype=th.float32)
            entry.link_pose_xyz_xyzw = _T_np_to_pose7_th(spec.T_world_body, self._sim_dev).expand(self._vec_size, 7).clone()
            if entry.cam is not None:
                T_world_cam = th.as_tensor(spec.T_world_body @ spec.T_body_cam, device=self._sim_dev, dtype=th.float32)
                entry.cam.set_pose(transform=T_world_cam.expand(self._vec_size, 4, 4).contiguous())
                entry.dirty = True
        if entry.virtual_link is not None:
            self._camera_links[entry.virtual_link] = entry

    def _parse_mjcf_cameras(self, model: ModelSpawnDef, mjcf_xml: str) -> tuple[list[_ParsedCameraSpec], bool]:
        """Extract the cameras of an mjcf model using mujoco. Returns the camera specs and whether the
        model is a camera-only model (no geoms and no joints), in which case it should not be loaded
        as a genesis entity."""
        import mujoco
        try:
            m = mujoco.MjModel.from_xml_string(mjcf_xml)
        except Exception as e:
            ggLog.warn(f"GenesisAdapter: could not parse model '{model.name}' with mujoco to extract cameras "
                       f"({type(e).__name__}: {e}). Cameras of this model (if any) will not be available.")
            return [], False
        camera_only = m.ngeom == 0 and m.njnt == 0
        if m.ncam == 0:
            return [], camera_only
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        if model.pose is not None:
            q = model.pose.orientation_xyzw
            T_spawn = _pos_quat_wxyz_to_T_np(model.pose.position, (q[3], q[0], q[1], q[2]))
        else:
            T_spawn = np.eye(4)
        specs = []
        for cid in range(m.ncam):
            cam_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            if cam_name is None:
                cam_name = f"{model.name}_camera{cid}"
            bid = int(m.cam_bodyid[cid])
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            width, height = int(m.cam_resolution[cid][0]), int(m.cam_resolution[cid][1])
            if width <= 1 or height <= 1:
                ggLog.warn(f"GenesisAdapter: camera '{cam_name}' of model '{model.name}' has no resolution set, using 320x240.")
                width, height = 320, 240
            T_body_cam = _pos_quat_wxyz_to_T_np(m.cam_pos[cid], m.cam_quat[cid])
            T_model_body = _pos_quat_wxyz_to_T_np(d.xpos[bid], d.xquat[bid])
            specs.append(_ParsedCameraSpec(cam_name=cam_name,
                                           model_name=model.name,
                                           body_name=body_name,
                                           width=width,
                                           height=height,
                                           fovy_deg=float(m.cam_fovy[cid]),
                                           T_body_cam=T_body_cam,
                                           T_world_body=T_spawn @ T_model_body,
                                           attach_to_entity_link=not camera_only))
        return specs, camera_only

    def _add_model(self, model: ModelSpawnDef) -> tuple[Any, list[_ParsedCameraSpec]]:
        if model.attachment_link is not None:
            raise NotImplementedError("GenesisAdapter does not support ModelSpawnDef.attachment_link")
        if model.definition_string is None:
            raise RuntimeError(f"Model '{model.name}' has no definition_string, GenesisAdapter requires definition strings.")
        fmt = (model.format or "").lower()
        base_fmt = fmt.split(".")[0]
        model_kwargs = dict(model.kwargs) if model.kwargs else {}
        fixed = bool(model_kwargs.pop("genesis_fixed", False))
        merge_fixed_links = bool(model_kwargs.pop("genesis_merge_fixed_links", False))
        scale = float(model_kwargs.pop("genesis_scale", 1.0))
        gravity_compensation = float(model_kwargs.pop("genesis_gravity_compensation", 0.0))
        morph_kwargs = dict(model_kwargs.pop("genesis_morph_kwargs", {}))
        definition = model.definition_string
        if fmt.endswith("xacro"):
            definition = compile_xacro_string(model_definition_string=definition, model_kwargs=model_kwargs)
        pose_kwargs = {}
        if model.pose is not None:
            pose_kwargs["pos"] = tuple(float(v) for v in model.pose.position)
            qx, qy, qz, qw = (float(v) for v in model.pose.orientation_xyzw)
            pose_kwargs["quat"] = (qw, qx, qy, qz)
        material = gs.materials.Rigid(gravity_compensation=gravity_compensation) if gravity_compensation != 0.0 else None
        cam_specs: list[_ParsedCameraSpec] = []
        if base_fmt == "urdf":
            file = os.path.join(self._tmp_dir, f"{model.name}.urdf")
            with open(file, "w") as f:
                f.write(definition)
            morph = gs.morphs.URDF(file=file,
                                   fixed=fixed,
                                   merge_fixed_links=merge_fixed_links,
                                   scale=scale,
                                   **pose_kwargs,
                                   **morph_kwargs)
        elif base_fmt in ("mjcf", "xml"):
            cam_specs, camera_only = self._parse_mjcf_cameras(model, definition)
            if camera_only:
                ggLog.info(f"GenesisAdapter: model '{model.name}' only contains cameras, it will not be loaded as a "
                           f"genesis entity. Its cameras: {[s.cam_name for s in cam_specs]}")
                return None, cam_specs
            file = os.path.join(self._tmp_dir, f"{model.name}.xml")
            with open(file, "w") as f:
                f.write(definition)
            morph = gs.morphs.MJCF(file=file,
                                   scale=scale,
                                   **pose_kwargs,
                                   **morph_kwargs)
        else:
            raise RuntimeError(f"Unsupported model format '{model.format}' for model '{model.name}'")
        return self._scene.add_entity(morph, material=material, name=model.name), cam_specs

    @override
    def destroy_scenario(self, **kwargs):
        if self._scene is not None and hasattr(self._scene, "destroy"):
            try:
                self._scene.destroy()
            except Exception as e:
                ggLog.warn(f"GenesisAdapter: scene destruction failed with {type(e).__name__}: {e}")
        self._scene = None
        self._entities = {}
        self._joints = {}
        self._links = {}
        self._jname2ids = {}
        self._cameras = {}
        self._camera_links = {}
        self._camlink_name2id = {}
        self._camlink_id2entry = {}
        self._joint_ids_cache = {}
        self._link_ids_cache = {}
        self._link_set_info_cache = {}
        self._prev_dofs_vel = None
        self._dofs_acc = None
        self._impulses = None

    @override
    def spawn_models(self, models: Sequence[ModelSpawnDef]) -> list[str]:
        raise NotImplementedError("GenesisAdapter does not support spawning models after build_scenario "
                                  "(the genesis scene is static after build). Include all models in build_scenario.")

    @override
    def delete_model(self, model_name: str):
        raise NotImplementedError("GenesisAdapter does not support deleting models after build_scenario")

    @override
    def setupLight(self):
        ggLog.info("GenesisAdapter.setupLight(): using genesis default lighting")

    # =================================================================================
    #                                  name resolution
    # =================================================================================

    def _to_joint_ids(self, requested) -> th.Tensor:
        if isinstance(requested, th.Tensor):
            return requested
        key = tuple(requested)
        ids = self._joint_ids_cache.get(key)
        if ids is None:
            rows = []
            for jn in key:
                jids = self._jname2ids.get(tuple(jn))
                if jids is None:
                    if tuple(jn) in self._joints:
                        raise RuntimeError(f"Joint {jn} is not a 1-dof joint, only revolute/prismatic joints are supported here")
                    raise RuntimeError(f"Joint {jn} not found. Available joints: {list(self._joints.keys())}")
                rows.append(jids)
            ids = th.as_tensor(rows, dtype=th.long, device=self._sim_dev).view(-1, 2)
            self._joint_ids_cache[key] = ids
        return ids

    def _to_link_ids(self, requested) -> th.Tensor:
        """Resolve link names to ids. Physical links get their non-negative genesis id, virtual
        camera links get their synthetic negative id (see _camlink_name2id)."""
        if isinstance(requested, th.Tensor):
            return requested
        key = tuple(requested)
        ids = self._link_ids_cache.get(key)
        if ids is None:
            rows = []
            for ln in key:
                link = self._links.get(tuple(ln))
                if link is not None:
                    rows.append(int(link.idx))
                elif tuple(ln) in self._camlink_name2id:
                    rows.append(self._camlink_name2id[tuple(ln)])
                else:
                    raise RuntimeError(f"Link {ln} not found. Available links: "
                                       f"{list(self._links.keys()) + list(self._camlink_name2id.keys())}")
            ids = th.as_tensor(rows, dtype=th.long, device=self._sim_dev).view(-1)
            self._link_ids_cache[key] = ids
        return ids

    @override
    def get_joints_ids(self, joint_names):
        return self._to_joint_ids(joint_names)

    @override
    def get_links_ids(self, link_names):
        return self._to_link_ids(link_names)

    @override
    def get_monitored_joints_ids(self, joint_names):
        return self._to_joint_ids(joint_names)

    @override
    def get_monitored_links_ids(self, link_names):
        return self._to_link_ids(link_names)

    @override
    def get_monitored_joints_ids_names(self, joint_ids):
        return [self._dofidx2jname[int(i)] for i in joint_ids[:, 0]]

    @override
    def get_monitored_links_ids_names(self, link_ids):
        return [self._linkidx2lname[int(i)] for i in link_ids]

    @override
    def get_detected_joints(self):
        return list(self._joints.keys())

    @override
    def get_detected_joints_properties(self):
        return {jn: JointProperties(_gs2adarl_jtype[j.type]) for jn, j in self._joints.items()}

    @override
    def get_detected_links(self):
        return list(self._links.keys()) + list(self._camera_links.keys())

    @override
    def get_detected_cameras(self):
        return list(self._cameras.keys())

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str, str]]):
        super().set_monitored_joints(jointsToObserve)
        if self._scene is not None:
            self._monitored_jids = self._to_joint_ids(self._monitored_joints)
            self._reset_step_stats()

    @override
    def set_monitored_links(self, linksToObserve: Sequence[tuple[str, str]]):
        super().set_monitored_links(linksToObserve)
        if self._scene is not None:
            cam = [ln for ln in self._monitored_links if tuple(ln) in self._camlink_name2id]
            if cam:
                raise RuntimeError(f"Virtual camera links cannot be monitored (they have no velocity dynamics): {cam}")
            self._monitored_lids = self._to_link_ids(self._monitored_links)
            self._reset_step_stats()

    # =================================================================================
    #                                   state reading
    # =================================================================================

    def _out(self, t: th.Tensor) -> th.Tensor:
        return t.to(device=self._out_th_device, dtype=self._out_th_float_dtype)

    @override
    def getJointsState(self, requestedJoints=None) -> th.Tensor:
        ids = self._monitored_jids if requestedJoints is None else self._to_joint_ids(requestedJoints)
        if ids.shape[0] == 0:
            return th.empty((self._vec_size, 0, 3), device=self._out_th_device, dtype=self._out_th_float_dtype)
        rs = self._rigid_solver()
        pos = rs.get_qpos(qs_idx=ids[:, 1])
        vel = rs.get_dofs_velocity(dofs_idx=ids[:, 0])
        eff = rs.get_dofs_control_force(dofs_idx=ids[:, 0])
        return self._out(th.stack([pos, vel, eff], dim=2))

    @override
    def getExtendedJointsState(self, requestedJoints=None) -> th.Tensor:
        ids = self._monitored_jids if requestedJoints is None else self._to_joint_ids(requestedJoints)
        if ids.shape[0] == 0:
            return th.empty((self._vec_size, 0, 5), device=self._out_th_device, dtype=self._out_th_float_dtype)
        rs = self._rigid_solver()
        pos = rs.get_qpos(qs_idx=ids[:, 1])
        vel = rs.get_dofs_velocity(dofs_idx=ids[:, 0])
        applied = rs.get_dofs_control_force(dofs_idx=ids[:, 0])
        acc = self._dofs_acc[:, ids[:, 0]]
        measured = rs.get_dofs_force(dofs_idx=ids[:, 0])
        return self._out(th.stack([pos, vel, applied, acc, measured], dim=2))

    def _split_camera_links(self, requestedLinks) -> tuple[list, list[int], list[int]]:
        """Splits a link name request into (names, camera_positions, physical_positions)."""
        names = [tuple(ln) for ln in requestedLinks]
        cam_pos = [i for i, n in enumerate(names) if n in self._camera_links]
        phys_pos = [i for i in range(len(names)) if i not in cam_pos]
        return names, cam_pos, phys_pos

    def _links_state_from_ids(self, ids: th.Tensor, use_com_pose: bool) -> th.Tensor:
        """Read link states given a tensor of link ids that may mix physical ids (>=0) and synthetic
        virtual-camera-link ids (<0). Physical links are read from the solver; virtual camera links
        report only their stored pose (positions 0:7), with zero velocity."""
        n = ids.shape[0]
        if n == 0:
            return th.empty((self._vec_size, 0, 13), device=self._out_th_device, dtype=self._out_th_float_dtype)
        rs = self._rigid_solver()
        real_mask = ids >= 0
        if bool(real_mask.all()):
            pos = rs.get_links_pos(links_idx=ids, ref="link_com" if use_com_pose else "link_origin")
            quat_xyzw = rs.get_links_quat(links_idx=ids)[:, :, [1, 2, 3, 0]]
            lin_vel = rs.get_links_vel(links_idx=ids, ref="link_com")
            ang_vel = rs.get_links_ang(links_idx=ids)
            return self._out(th.cat([pos, quat_xyzw, lin_vel, ang_vel], dim=2))
        out = th.zeros((self._vec_size, n, 13), device=self._out_th_device, dtype=self._out_th_float_dtype)
        if bool(real_mask.any()):
            real_ids = ids[real_mask]
            pos = rs.get_links_pos(links_idx=real_ids, ref="link_com" if use_com_pose else "link_origin")
            quat_xyzw = rs.get_links_quat(links_idx=real_ids)[:, :, [1, 2, 3, 0]]
            lin_vel = rs.get_links_vel(links_idx=real_ids, ref="link_com")
            ang_vel = rs.get_links_ang(links_idx=real_ids)
            out[:, real_mask.nonzero().flatten().to(out.device)] = self._out(th.cat([pos, quat_xyzw, lin_vel, ang_vel], dim=2))
        for p in (~real_mask).nonzero().flatten().tolist():
            entry = self._camlink_id2entry[int(ids[p])]
            out[:, p, 0:7] = self._out(entry.link_pose_xyz_xyzw)
        return out

    @override
    def getLinksState(self, requestedLinks=None, use_com_pose: bool = False) -> th.Tensor:
        ids = self._monitored_lids if requestedLinks is None else self._to_link_ids(requestedLinks)
        return self._links_state_from_ids(ids, use_com_pose)

    @override
    def get_local_link_linear_acceleration(self, requestedLinks=None) -> th.Tensor:
        ids = self._monitored_lids if requestedLinks is None else self._to_link_ids(requestedLinks)
        n = ids.shape[0]
        if n == 0:
            return th.empty((self._vec_size, 0, 3), device=self._out_th_device, dtype=self._out_th_float_dtype)
        rs = self._rigid_solver()
        real_mask = ids >= 0

        def _real_acc(real_ids):
            acc_w = rs.get_links_acc(links_idx=real_ids)  # world-frame classical acceleration
            quat_xyzw = rs.get_links_quat(links_idx=real_ids)[:, :, [1, 2, 3, 0]]
            # proper (accelerometer-style) acceleration, expressed in the link frame
            proper_acc_w = acc_w - self._gravity_vec_w.view(1, 1, 3)
            return self._out(th_quat_rotate(proper_acc_w, th_quat_conj(quat_xyzw)))

        if bool(real_mask.all()):
            return _real_acc(ids)
        # virtual (camera) links have no dynamics: their acceleration is reported as zero
        out = th.zeros((self._vec_size, n, 3), device=self._out_th_device, dtype=self._out_th_float_dtype)
        if bool(real_mask.any()):
            out[:, real_mask.nonzero().flatten().to(out.device)] = _real_acc(ids[real_mask])
        return out

    # =================================================================================
    #                                   state setting
    # =================================================================================

    def _split_by_mask(self, values: th.Tensor, vec_mask: th.Tensor | None):
        """Returns (envs_idx, masked_values): envs_idx is None when all environments are selected."""
        if vec_mask is None or bool(vec_mask.all()):
            return None, values
        envs_idx = vec_mask.nonzero().flatten()
        return envs_idx, values[envs_idx.to(values.device)]

    @override
    def setJointsStateDirect(self, joint_names, joint_states_pve: th.Tensor, vec_mask: th.Tensor | None = None):
        ids = self._to_joint_ids(joint_names)
        if joint_states_pve.dim() != 3 or joint_states_pve.shape[0] != self._vec_size or \
           joint_states_pve.shape[1] != ids.shape[0] or joint_states_pve.shape[2] != 3:
            raise ValueError(f"joint_states_pve has shape {tuple(joint_states_pve.shape)}, "
                             f"expected ({self._vec_size},{ids.shape[0]},3)")
        if ids.shape[0] == 0:
            return
        rs = self._rigid_solver()
        envs_idx, vals = self._split_by_mask(joint_states_pve, vec_mask)
        rs.set_qpos(vals[:, :, 0], qs_idx=ids[:, 1], envs_idx=envs_idx)
        rs.set_dofs_velocity(vals[:, :, 1], dofs_idx=ids[:, 0], envs_idx=envs_idx)
        rs.control_dofs_force(vals[:, :, 2], dofs_idx=ids[:, 0], envs_idx=envs_idx)

    def _get_link_set_info(self, link_names) -> _LinkSetInfo:
        key = tuple(tuple(ln) for ln in link_names)
        info = self._link_set_info_cache.get(key)
        if info is not None:
            return info
        links_idx = []
        free_pos = []
        free_dofs_idx = []
        nonfree_pos = []
        for i, ln in enumerate(key):
            link = self._links.get(ln)
            if link is None:
                raise RuntimeError(f"Link {ln} not found. Available links: {list(self._links.keys())}")
            if link.parent_idx != -1:
                raise NotImplementedError(f"Link {ln} is not a root link, only root links can have their state set directly.")
            links_idx.append(int(link.idx))
            base_joint = link.joints[0] if len(link.joints) > 0 else None
            if base_joint is not None and base_joint.type == gs.JOINT_TYPE.FREE:
                free_pos.append(i)
                free_dofs_idx.extend(int(d) for d in base_joint.dofs_idx)
            else:
                nonfree_pos.append(i)
        info = _LinkSetInfo(links_idx=links_idx, free_pos=free_pos, free_dofs_idx=free_dofs_idx, nonfree_pos=nonfree_pos)
        self._link_set_info_cache[key] = info
        return info

    def _set_camera_links_state(self, names: list, cam_pos: list[int],
                                link_states_pose_vel: th.Tensor, vec_mask: th.Tensor | None):
        if bool((link_states_pose_vel[:, cam_pos, 7:13] != 0.0).any()):
            raise NotImplementedError(f"Cannot set non-zero velocities on virtual camera links {[names[i] for i in cam_pos]}")
        mask = None
        if vec_mask is not None and not bool(vec_mask.all()):
            mask = vec_mask.to(self._sim_dev)
        for i in cam_pos:
            entry = self._camera_links[names[i]]
            pose = link_states_pose_vel[:, i, 0:7].to(device=self._sim_dev, dtype=th.float32)
            if mask is None:
                entry.link_pose_xyz_xyzw.copy_(pose)
            else:
                entry.link_pose_xyz_xyzw[mask] = pose[mask]
            if entry.cam is not None:
                T_link = _pose7_to_T_th(entry.link_pose_xyz_xyzw)
                entry.cam.set_pose(transform=(T_link @ entry.offset_T).contiguous())
                entry.dirty = True

    @override
    def setLinksStateDirect(self, link_names, link_states_pose_vel: th.Tensor, vec_mask: th.Tensor | None = None):
        if isinstance(link_names, (th.Tensor, np.ndarray)):
            link_names = [self._linkidx2lname[int(i)] for i in link_names]
        names, cam_pos, phys_pos = self._split_camera_links(link_names)
        if len(cam_pos) > 0:
            if link_states_pose_vel.dim() != 3 or link_states_pose_vel.shape[1] != len(names):
                raise ValueError(f"link_states_pose_vel has shape {tuple(link_states_pose_vel.shape)}, "
                                 f"expected ({self._vec_size},{len(names)},13)")
            self._set_camera_links_state(names, cam_pos, link_states_pose_vel, vec_mask)
            if len(phys_pos) == 0:
                return
            link_names = [names[i] for i in phys_pos]
            link_states_pose_vel = link_states_pose_vel[:, phys_pos]
        info = self._get_link_set_info(link_names)
        n_links = len(info.links_idx)
        if link_states_pose_vel.dim() != 3 or link_states_pose_vel.shape[0] != self._vec_size or \
           link_states_pose_vel.shape[1] != n_links or link_states_pose_vel.shape[2] != 13:
            raise ValueError(f"link_states_pose_vel has shape {tuple(link_states_pose_vel.shape)}, "
                             f"expected ({self._vec_size},{n_links},13)")
        if n_links == 0:
            return
        rs = self._rigid_solver()
        envs_idx, vals = self._split_by_mask(link_states_pose_vel, vec_mask)
        if len(info.nonfree_pos) > 0:
            nonfree_vels = vals[:, info.nonfree_pos, 7:13]
            if bool((nonfree_vels != 0.0).any()):
                nonfree_names = [link_names[i] for i in info.nonfree_pos]
                raise NotImplementedError(f"Cannot set non-zero velocities on links without a free base joint: {nonfree_names}")
        rs.set_base_links_pos(vals[:, :, 0:3], links_idx=info.links_idx, envs_idx=envs_idx)
        rs.set_base_links_quat(vals[:, :, 3:7][:, :, [3, 0, 1, 2]], links_idx=info.links_idx, envs_idx=envs_idx)
        if len(info.free_pos) > 0:
            free_vels = vals[:, info.free_pos, 7:13]  # (sel, n_free, 6), [lin,ang] matching genesis free-joint dof order
            rs.set_dofs_velocity(free_vels.reshape(free_vels.shape[0], -1), dofs_idx=info.free_dofs_idx, envs_idx=envs_idx)

    # =================================================================================
    #                                     control
    # =================================================================================

    @override
    def setJointsEffortCommand(self, joint_names, efforts: th.Tensor, vec_mask: th.Tensor | None = None) -> None:
        ids = self._to_joint_ids(joint_names)
        if efforts.dim() != 2 or efforts.shape[0] != self._vec_size or efforts.shape[1] != ids.shape[0]:
            raise ValueError(f"efforts has shape {tuple(efforts.shape)}, expected ({self._vec_size},{ids.shape[0]})")
        if ids.shape[0] == 0:
            return
        envs_idx, vals = self._split_by_mask(efforts, vec_mask)
        self._rigid_solver().control_dofs_force(vals, dofs_idx=ids[:, 0], envs_idx=envs_idx)

    @override
    def set_link_impulses(self, link_ids: Sequence[Any],
                          force_torque_xyzxyz: th.Tensor,
                          durations: th.Tensor, delays: th.Tensor,
                          vec_mask: th.Tensor) -> None:
        """Apply forces/torques on a set of links for the given durations, after the given delays.
        Forces and torques are in the world frame, applied at the link CoM (mimicking mujoco's xfrc_applied).
        A new call replaces any previously set impulses."""
        ids = self._to_link_ids(link_ids)
        if bool((ids < 0).any()):
            raise NotImplementedError("Cannot apply impulses to virtual camera links")
        n = ids.shape[0]
        ft = force_torque_xyzxyz.to(device=self._sim_dev, dtype=th.float32).view(self._vec_size, n, 6).clone()
        now = self._sim_time
        start = now + delays.to(device=self._sim_dev, dtype=th.float32).view(self._vec_size, n).clamp(min=0.0)
        end = start + durations.to(device=self._sim_dev, dtype=th.float32).view(self._vec_size, n).clamp(min=0.0)
        if vec_mask is not None:
            end = th.where(vec_mask.to(self._sim_dev).view(-1, 1), end, start)  # zero-duration = inactive
        self._impulses = _ImpulsesSpec(links_idx=ids.tolist(),
                                       force_torque_xyzxyz=ft,
                                       start_times=start,
                                       end_times=end)

    def _apply_active_impulses(self):
        spec = self._impulses
        if spec is None:
            return
        now = self._sim_time
        active = (spec.start_times <= now) & (now < spec.end_times)  # (vec_size, n)
        if not bool(active.any()):
            if now >= float(spec.end_times.max()):
                self._impulses = None
            return
        rs = self._rigid_solver()
        active_f = active.unsqueeze(-1)
        rs.apply_links_external_force(spec.force_torque_xyzxyz[:, :, 0:3] * active_f, links_idx=spec.links_idx, ref="link_com")
        rs.apply_links_external_torque(spec.force_torque_xyzxyz[:, :, 3:6] * active_f, links_idx=spec.links_idx, ref="link_com")

    def _apply_commands(self):
        """Hook called before every simulation substep. Genesis control targets are persistent,
        so the base adapter has nothing to do here; subclasses (e.g. impedance control) override this."""
        pass

    # =================================================================================
    #                                     stepping
    # =================================================================================

    @override
    def initialize_for_step(self):
        pass

    @override
    def step(self) -> float:
        self.initialize_for_step()
        self._reset_step_stats()
        self._stepping = True
        duration = self.run(self._step_length_sec)
        self._stepping = False
        self._compute_step_stats()
        return duration

    @override
    def run(self, duration_sec: float):
        if self._scene is None:
            raise RuntimeError("Scenario is not built. Call build_scenario() first.")
        n_steps = max(1, int(round(duration_sec / self._sim_step_dt)))
        for _ in range(n_steps):
            self._apply_commands()
            self._apply_active_impulses()
            self._scene.step()
            self._sim_time += self._sim_step_dt
            self._update_dofs_acc()
            if self._stepping:
                self._update_step_stats()
        return n_steps * self._sim_step_dt

    def _update_dofs_acc(self):
        vel = self._rigid_solver().get_dofs_velocity()
        self._dofs_acc = (vel - self._prev_dofs_vel) / self._sim_step_dt
        self._prev_dofs_vel = vel

    @override
    def resetWorld(self):
        self._scene.reset()
        rs = self._rigid_solver()
        self._prev_dofs_vel = rs.get_dofs_velocity()
        self._dofs_acc = th.zeros_like(self._prev_dofs_vel)
        self._impulses = None
        self.initialize_for_episode()

    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._sim_time

    @override
    def sim_step_duration(self) -> th.Tensor:
        return self._sim_step_dt_th

    # =================================================================================
    #                                   step statistics
    # =================================================================================

    def _reset_step_stats(self):
        nj = self._monitored_jids.shape[0]
        nl = self._monitored_lids.shape[0]
        dev = self._sim_dev
        self._jstats_min = th.full((self._vec_size, nj, 6), float("inf"), device=dev)
        self._jstats_max = th.full((self._vec_size, nj, 6), float("-inf"), device=dev)
        self._jstats_sum = th.zeros((self._vec_size, nj, 6), device=dev)
        self._jstats_sumsq = th.zeros((self._vec_size, nj, 6), device=dev)
        self._lstats_min = th.full((self._vec_size, nl, 6), float("inf"), device=dev)
        self._lstats_max = th.full((self._vec_size, nl, 6), float("-inf"), device=dev)
        self._lstats_sum = th.zeros((self._vec_size, nl, 6), device=dev)
        self._lstats_sumsq = th.zeros((self._vec_size, nl, 6), device=dev)
        self._stats_substep_count = 0
        self._joint_stats = th.zeros((self._vec_size, 4, nj, 6), device=dev)
        self._link_stats = th.zeros((self._vec_size, 4, nl, 6), device=dev)

    def _read_monitored_joints_pvaeep(self) -> th.Tensor:
        ids = self._monitored_jids
        rs = self._rigid_solver()
        pos = rs.get_qpos(qs_idx=ids[:, 1])
        vel = rs.get_dofs_velocity(dofs_idx=ids[:, 0])
        acc = self._dofs_acc[:, ids[:, 0]]
        commanded = rs.get_dofs_control_force(dofs_idx=ids[:, 0])
        sensed = rs.get_dofs_force(dofs_idx=ids[:, 0])
        power = th.clamp(vel * commanded, 0.0, 1e6)
        return th.stack([pos, vel, acc, commanded, sensed, power], dim=2)

    def _update_step_stats(self):
        if self._monitored_jids.shape[0] > 0:
            x = self._read_monitored_joints_pvaeep()
            self._jstats_min = th.minimum(self._jstats_min, x)
            self._jstats_max = th.maximum(self._jstats_max, x)
            self._jstats_sum += x
            self._jstats_sumsq += x * x
        if self._monitored_lids.shape[0] > 0:
            rs = self._rigid_solver()
            lin = rs.get_links_vel(links_idx=self._monitored_lids, ref="link_com")
            ang = rs.get_links_ang(links_idx=self._monitored_lids)
            x = th.cat([lin, ang], dim=2)
            self._lstats_min = th.minimum(self._lstats_min, x)
            self._lstats_max = th.maximum(self._lstats_max, x)
            self._lstats_sum += x
            self._lstats_sumsq += x * x
        self._stats_substep_count += 1

    def _compute_step_stats(self):
        n = self._stats_substep_count
        if n == 0:
            return
        jmean = self._jstats_sum / n
        jstd = th.sqrt(th.clamp(self._jstats_sumsq / n - jmean * jmean, min=0.0))
        self._joint_stats = th.stack([self._jstats_min, self._jstats_max, jmean, jstd], dim=1)
        lmean = self._lstats_sum / n
        lstd = th.sqrt(th.clamp(self._lstats_sumsq / n - lmean * lmean, min=0.0))
        self._link_stats = th.stack([self._lstats_min, self._lstats_max, lmean, lstd], dim=1)

    @override
    def get_joints_state_step_stats(self) -> th.Tensor:
        return self._out(self._joint_stats[:, :, :, :4])

    @override
    def get_joints_state_step_stats_extended(self) -> th.Tensor:
        return self._out(self._joint_stats)

    @override
    def get_links_state_step_stats(self) -> th.Tensor:
        return self._out(self._link_stats)

    # =================================================================================
    #                                     rendering
    # =================================================================================

    @override
    def getRenderings(self, requestedCameras: list[str],
                      vec_mask: th.Tensor | None = None,
                      out_th_device: th.device | None = None,
                      out: list[th.Tensor] | None = None,
                      depth: bool = False) -> tuple[list[th.Tensor], th.Tensor]:
        if not self._enable_rendering:
            raise RuntimeError("Called getRenderings, but rendering is not enabled. Set enable_rendering=True.")
        if out_th_device is None:
            out_th_device = self._out_th_device
        sel_idx = None
        if vec_mask is not None and not bool(vec_mask.all()):
            sel_idx = vec_mask.nonzero().flatten()
        n_sel = self._vec_size if sel_idx is None else sel_idx.shape[0]
        images: list[th.Tensor] = []
        for i, cam_name in enumerate(requestedCameras):
            entry = self._cameras[cam_name]
            if entry.cam is None:
                raise RuntimeError(f"Camera '{cam_name}' has no renderer (was the adapter built with enable_rendering=False?)")
            if entry.attached:
                entry.cam.move_to_attach()
            rgb_img, depth_img, _, _ = entry.cam.render(rgb=not depth, depth=depth, force_render=entry.dirty)
            entry.dirty = False
            img = depth_img.unsqueeze(-1) if depth else rgb_img  # (vec_size, h, w, c)
            if sel_idx is not None:
                img = img[sel_idx.to(img.device)]
            img = img.to(device=out_th_device, non_blocking=out_th_device.type == "cuda")
            if out is not None:
                out[i].copy_(img)
                img = out[i]
            images.append(img)
        times = th.full((n_sel, len(requestedCameras)), self._sim_time,
                        device=out_th_device, dtype=self._out_th_float_dtype)
        return images, times
