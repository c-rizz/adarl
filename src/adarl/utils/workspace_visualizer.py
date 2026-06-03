#!/usr/bin/env python3
"""
Robot end-effector reachability workspace visualizer.

Samples joint configurations via Sobol quasi-random sequences, computes
forward kinematics with pinocchio (via the Robot helper), optionally
filters self-colliding poses, then displays the resulting EE positions
as an interactive point cloud.

Points are colored by geodesic orientation distance to a target orientation
controlled by sliders in the side panel.  A small EE preview shows the
target orientation as a coordinate frame.

Dependencies: pyvista, pyvistaqt, PyQt5, scipy
  pip install pyvista pyvistaqt PyQt5 scipy

Usage: edit the ``if __name__ == "__main__":`` block and run.
"""

from __future__ import annotations
import os
import sys
import math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.stats.qmc import Sobol

# cv2 ships its own Qt plugins that Qt finds first but cannot load.
# Point QT_QPA_PLATFORM_PLUGIN_PATH to PyQt5's bundled plugins before Qt
# initialises (i.e. before importing any PyQt5 submodule).
try:
    import PyQt5 as _pyqt5
    _plugins = os.path.join(os.path.dirname(_pyqt5.__file__), "Qt5", "plugins")
    if os.path.isdir(_plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _plugins
except ImportError:
    pass

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from PyQt5 import QtWidgets, QtCore
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("pip install pyvista pyvistaqt PyQt5 scipy")
    sys.exit(1)

from adarl.utils.robot_helpers import Robot


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_workspace(
    robot: Robot,
    controlled_joints: list[str],
    fixed_joints: dict[str, float],
    ee_frame: str,
    n_samples: int,
    check_collisions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the joint space and return valid EE poses and their joint configs.

    Returns
    -------
    poses : ndarray, shape (M, 7)
        Each row is [x, y, z, qx, qy, qz, qw] in the robot's base frame.
    joint_configs : ndarray, shape (M, len(controlled_joints))
        Corresponding sampled joint values (same order as controlled_joints).
    """
    limits = robot.get_joint_limits(controlled_joints)
    # limits[jn] is (2,3): row 0 = lower [pos,vel,eff], row 1 = upper
    lowers = np.array([limits[j][0, 0] for j in controlled_joints])
    uppers = np.array([limits[j][1, 0] for j in controlled_joints])

    n_dim = len(controlled_joints)
    m = max(1, math.ceil(math.log2(n_samples)))   # Sobol balance requires 2^m points
    unit_samples = Sobol(d=n_dim, scramble=True).random_base2(m)[:n_samples]  # (N, d) in [0,1)
    joint_samples = unit_samples * (uppers - lowers) + lowers

    # Fix the passive joints once, silently dropping any that don't exist
    known_joints = set(robot.get_joint_names())
    ignored = [k for k in fixed_joints if k not in known_joints]
    if ignored:
        print(f"  Note: ignoring fixed_joints not found in robot: {ignored}")
    fixed_dict = {k: np.asarray([float(v)]) for k, v in fixed_joints.items() if k in known_joints}
    robot.set_joint_pose_by_names(fixed_dict)

    if check_collisions:
        print("  Pre-sampling to detect always-colliding pairs...")
        always_colliding = robot.detect_always_present_collisions(
            moving_joints=controlled_joints,
            fixed_joints_pose=fixed_dict,
            samples=2000,
            threshold=1.0,
        )
        if always_colliding:
            print(f"  Removing {len(always_colliding)} always-colliding pairs: {always_colliding}")
            robot.remove_collision_pairs(always_colliding)
        robot.set_joint_pose_by_names(fixed_dict)  # restore pose after pre-sampling

    results: list[np.ndarray] = []
    configs: list[np.ndarray] = []
    report_step = 5000
    for i, q in enumerate(joint_samples):
        if i % report_step == 0:
            print(f"  {i}/{n_samples}  valid: {len(results)}", end="\r", flush=True)
        jdict = {jn: np.array([float(q[k])]) for k, jn in enumerate(controlled_joints)}
        robot.set_joint_pose_by_names(jdict)
        if check_collisions and robot.has_collisions()[0]:
            continue
        results.append(robot.get_frame_poses_xyzxyzw(frames=[ee_frame])[ee_frame])
        configs.append(q.copy())

    coll_note = f" ({100*len(results)//n_samples}% non-colliding)" if check_collisions else ""
    print(f"\n  Done: {len(results)}/{n_samples} poses kept{coll_note}")

    if not results:
        raise RuntimeError(
            "No valid poses found. Check controlled_joints limits and fixed_joints values."
        )
    return np.array(results, dtype=np.float64), np.array(configs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Robot mesh loading
# ---------------------------------------------------------------------------

def _load_mesh_file(path: str) -> pv.PolyData | None:
    """Load a mesh into PyVista. Falls back to trimesh for DAE files."""
    ext = Path(path).suffix.lower()
    if ext == ".dae":
        try:
            import trimesh
            loaded = trimesh.load(path, force="mesh")
            if isinstance(loaded, trimesh.Scene):
                geoms = list(loaded.geometry.values())
                if not geoms:
                    return None
                loaded = trimesh.util.concatenate(geoms)
            faces = np.hstack([np.full((len(loaded.faces), 1), 3, dtype=np.int64), loaded.faces])
            return pv.PolyData(np.asarray(loaded.vertices, dtype=np.float64), faces.ravel())
        except ImportError:
            pass  # fall through to pv.read
    try:
        return pv.read(path)
    except Exception:
        return None


def load_robot_mesh_pv(
    robot: Robot,
    urdf_string: str,
    controlled_joints: list[str],
    fixed_joints: dict[str, float],
) -> list[pv.PolyData]:
    """
    Return the robot's visual meshes transformed to a neutral pose.

    Neutral pose: uses fixed_joints values for controlled joints when provided
    (i.e. the same reference pose as the rest of the config), otherwise falls
    back to the midpoint of each joint's limits.
    """
    import pinocchio as pin

    try:
        visual_model = pin.buildGeomFromUrdfString(
            robot._model, urdf_string, pin.GeometryType.VISUAL
        )
    except Exception as exc:
        print(f"  Warning: could not build visual geometry model: {exc}")
        return []

    visual_data = pin.GeometryData(visual_model)

    known = set(robot.get_joint_names())
    limits = robot.get_joint_limits(controlled_joints)
    neutral: dict[str, np.ndarray] = {}
    for jn in controlled_joints:
        if jn in fixed_joints:
            neutral[jn] = np.asarray([float(fixed_joints[jn])])
        else:
            lo, hi = limits[jn][0, 0], limits[jn][1, 0]
            neutral[jn] = np.asarray([(lo + hi) / 2.0])
    neutral.update({k: np.asarray([float(v)]) for k, v in fixed_joints.items() if k in known})
    robot.set_joint_pose_by_names(neutral)

    pin.forwardKinematics(robot._model, robot._model_data, robot._joint_position)
    pin.updateGeometryPlacements(robot._model, robot._model_data, visual_model, visual_data)

    meshes: list[pv.PolyData] = []
    n_total = len(visual_model.geometryObjects)
    for i, geom in enumerate(visual_model.geometryObjects):
        path = geom.meshPath
        if not path or not os.path.exists(path):
            continue
        try:
            mesh = _load_mesh_file(path)
            if mesh is None or mesh.n_points == 0:
                continue
            s = np.asarray(geom.meshScale, dtype=np.float64)
            if not np.allclose(s, 1.0):
                mesh.points = mesh.points * s
            T = np.eye(4)
            T[:3, :3] = visual_data.oMg[i].rotation
            T[:3, 3]  = visual_data.oMg[i].translation
            mesh.transform(T, inplace=True)
            meshes.append(mesh)
        except Exception as exc:
            print(f"  Skipping {Path(path).name}: {exc}")

    print(f"  Robot mesh: {len(meshes)}/{n_total} geometry objects loaded")
    return meshes


class _LiveRobotMesh:
    """
    Pre-loads the robot's visual meshes in geometry-local coordinates.
    Calling update() runs FK and overwrites each mesh's points in-place —
    no mesh copies, so it's cheap enough for interactive slider callbacks.
    """

    def __init__(self, robot: Robot, urdf_string: str):
        import pinocchio as pin
        self._robot = robot
        try:
            self._visual_model = pin.buildGeomFromUrdfString(
                robot._model, urdf_string, pin.GeometryType.VISUAL
            )
        except Exception as exc:
            print(f"  Warning: _LiveRobotMesh could not build visual model: {exc}")
            self._visual_model = None
            self._visual_data  = None
            self._local_pts: dict[int, np.ndarray] = {}
            self._meshes: dict[int, pv.PolyData] = {}
            return

        self._visual_data = pin.GeometryData(self._visual_model)
        self._local_pts: dict[int, np.ndarray] = {}
        self._meshes: dict[int, pv.PolyData]   = {}

        for i, geom in enumerate(self._visual_model.geometryObjects):
            path = geom.meshPath
            if not path or not os.path.exists(path):
                continue
            mesh = _load_mesh_file(path)
            if mesh is None or mesh.n_points == 0:
                continue
            pts = mesh.points.astype(np.float64)
            s = np.asarray(geom.meshScale, dtype=np.float64)
            if not np.allclose(s, 1.0):
                pts = pts * s
            self._local_pts[i] = pts
            self._meshes[i]    = mesh          # topology kept; .points overwritten on update

        print(f"  Live robot mesh: {len(self._meshes)} geometry objects pre-loaded")

    @property
    def meshes(self) -> list[pv.PolyData]:
        return list(self._meshes.values())

    def subtree_meshes(self, robot: Robot, root_frame: str) -> list[pv.PolyData]:
        """Return the subset of meshes that belong to the kinematic sub-tree under root_frame."""
        if self._visual_model is None:
            return []
        parent_joint = robot._frame_names_to_parent_joint_names[root_frame]
        subtree_ids  = {robot._joint_name_to_idx[j]
                        for j in robot.get_tree_joint_names_under_joint(parent_joint)}
        return [mesh for geom_idx, mesh in self._meshes.items()
                if self._visual_model.geometryObjects[geom_idx].parentJoint in subtree_ids]

    def update(self, controlled_joints: list[str], q: np.ndarray,
               fixed_dict: dict[str, np.ndarray]) -> None:
        """Set joint config, run FK, update all mesh points in-place."""
        if self._visual_model is None:
            return
        import pinocchio as pin
        self._robot.set_joint_pose_by_names(fixed_dict)
        jdict = {jn: np.array([float(q[k])]) for k, jn in enumerate(controlled_joints)}
        self._robot.set_joint_pose_by_names(jdict)
        self._robot._update_forward_kinematics()
        pin.updateGeometryPlacements(
            self._robot._model, self._robot._model_data,
            self._visual_model, self._visual_data,
        )
        for i, pts_local in self._local_pts.items():
            oMg = self._visual_data.oMg[i]
            self._meshes[i].points = pts_local @ oMg.rotation.T + oMg.translation


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

class _LabeledSlider(QtWidgets.QWidget):
    """Compact horizontal slider with a left label and a right value readout."""

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, label: str, lo: int, hi: int, init: int, suffix: str = ""):
        super().__init__()
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        self._suffix = suffix

        name_lbl = QtWidgets.QLabel(f"{label}:")
        name_lbl.setFixedWidth(48)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(init)

        self._val_lbl = QtWidgets.QLabel(f"{init}{suffix}")
        self._val_lbl.setFixedWidth(48)
        self._val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self._slider.valueChanged.connect(lambda v: self._val_lbl.setText(f"{v}{suffix}"))
        self._slider.valueChanged.connect(self.valueChanged)

        h.addWidget(name_lbl)
        h.addWidget(self._slider)
        h.addWidget(self._val_lbl)

    @property
    def value(self) -> int:
        return self._slider.value()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class WorkspaceViewer(QtWidgets.QMainWindow):
    def __init__(self, poses: np.ndarray, ee_frame: str,
                 live_robot: _LiveRobotMesh | None = None,
                 joint_configs: np.ndarray | None = None,
                 controlled_joints: list[str] | None = None,
                 fixed_dict: dict[str, np.ndarray] | None = None):
        super().__init__()
        self._poses             = poses
        self._ee_frame          = ee_frame
        self._live_robot        = live_robot
        self._joint_configs     = joint_configs
        self._controlled_joints = controlled_joints or []
        self._fixed_dict        = fixed_dict or {}
        self._robot_actors: list = []
        self._syncing           = False
        self.setWindowTitle(f"Workspace Viewer — EE: {ee_frame}  ({len(poses):,} poses)")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ── Left: main view (robot + point cloud) ────────────────────────
        self._main_plot = QtInteractor(parent=self)
        root.addWidget(self._main_plot, stretch=7)

        # ── Right panel ──────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(280)
        pv_layout = QtWidgets.QVBoxLayout(panel)
        root.addWidget(panel, stretch=3)

        pv_layout.addWidget(QtWidgets.QLabel("<b>EE orientation (closest pose)</b>"))

        self._ee_plot = QtInteractor(parent=panel)
        self._ee_plot.setMinimumHeight(260)
        pv_layout.addWidget(self._ee_plot)

        pv_layout.addSpacing(8)
        pv_layout.addWidget(QtWidgets.QLabel("<b>Orientation filter</b>"))

        self._rpy: dict[str, _LabeledSlider] = {}
        for axis in ("Roll", "Pitch", "Yaw"):
            s = _LabeledSlider(axis, -180, 180, 0, "°")
            s.valueChanged.connect(self._on_changed)
            pv_layout.addWidget(s)
            self._rpy[axis] = s

        pv_layout.addSpacing(4)
        self._tol = _LabeledSlider("Tol.", 1, 180, 30, "°")
        self._tol.valueChanged.connect(self._on_changed)
        pv_layout.addWidget(self._tol)

        self._hide_cb = QtWidgets.QCheckBox("Hide out-of-tolerance points")
        self._hide_cb.setChecked(True)
        self._hide_cb.stateChanged.connect(self._on_changed)
        pv_layout.addWidget(self._hide_cb)

        has_robot = live_robot is not None and bool(live_robot.meshes)
        self._robot_cb = QtWidgets.QCheckBox("Show robot mesh")
        self._robot_cb.setChecked(has_robot)
        self._robot_cb.setEnabled(has_robot)
        self._robot_cb.stateChanged.connect(self._update_robot_visibility)
        pv_layout.addWidget(self._robot_cb)

        pv_layout.addStretch()

        self._init_main_plot()
        self._init_ee_plot()
        self._setup_camera_sync()
        self._on_changed()
        # reset_camera AFTER _on_changed so meshes are at their real FK positions
        self._main_plot.reset_camera()
        self._ee_plot.reset_camera()
        self.resize(1400, 800)

    # ── initialisation ───────────────────────────────────────────────────

    def _init_main_plot(self):
        self._main_plot.set_background("black")
        self._main_plot.add_axes()
        cloud = pv.PolyData(self._poses[:, :3].copy())
        cloud["dist"] = np.zeros(len(self._poses), dtype=np.float32)
        self._main_plot.add_mesh(
            cloud, scalars="dist", cmap="coolwarm", clim=[0.0, 1.0],
            point_size=3, render_points_as_spheres=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Orient. dist\n(0=match, 1=tol.)"},
            name="workspace",
        )
        if self._live_robot is not None:
            for i, mesh in enumerate(self._live_robot.meshes):
                actor = self._main_plot.add_mesh(
                    mesh, color="lightgray", opacity=0.35,
                    show_edges=False, name=f"robot_{i}",
                )
                self._robot_actors.append(actor)

    def _init_ee_plot(self):
        self._ee_plot.set_background("#2a2a2a")
        if self._live_robot is not None:
            ee_meshes = self._live_robot.subtree_meshes(
                self._live_robot._robot, self._ee_frame
            )
            for i, mesh in enumerate(ee_meshes):
                self._ee_plot.add_mesh(mesh, color="lightgray", opacity=0.9,
                                       show_edges=False, name=f"ee_{i}")
        self._ee_plot.add_axes()
        self._ee_plot.reset_camera()

    def _setup_camera_sync(self):
        def on_main(obj, evt):
            if self._syncing: return
            self._syncing = True
            self._copy_camera_direction(self._main_plot, self._ee_plot)
            self._syncing = False

        def on_ee(obj, evt):
            if self._syncing: return
            self._syncing = True
            self._copy_camera_direction(self._ee_plot, self._main_plot)
            self._syncing = False

        self._main_plot.camera.AddObserver("ModifiedEvent", on_main)
        self._ee_plot.camera.AddObserver("ModifiedEvent", on_ee)

    # ── helpers ──────────────────────────────────────────────────────────

    def _copy_camera_direction(self, src, dst):
        """Copy view direction and up vector from src to dst, preserving dst focal point + zoom."""
        sp = np.array(src.camera.position)
        sf = np.array(src.camera.focal_point)
        d  = sp - sf
        d_len = np.linalg.norm(d)
        if d_len < 1e-10:
            return
        d_norm = d / d_len
        df = np.array(dst.camera.focal_point)
        dz = np.linalg.norm(np.array(dst.camera.position) - df)
        dst.camera.position  = df + dz * d_norm
        dst.camera.up   = src.camera.up
        dst.render()

    def _target_rotation(self) -> Rotation:
        r = self._rpy["Roll"].value
        p = self._rpy["Pitch"].value
        y = self._rpy["Yaw"].value
        # Intrinsic ZYX: yaw in world frame first, then pitch in yawed frame,
        # then roll in yawed+pitched frame — each slider acts on the local axis.
        return Rotation.from_euler("ZYX", [y, p, r], degrees=True)

    # ── update callbacks ─────────────────────────────────────────────────

    def _update_robot_visibility(self):
        visible = self._robot_cb.isChecked()
        for actor in self._robot_actors:
            actor.SetVisibility(visible)
        self._main_plot.render()

    def _on_changed(self):
        rot      = self._target_rotation()
        q_target = rot.as_quat()
        q_poses  = self._poses[:, 3:]

        dot       = np.clip(np.abs(q_poses @ q_target), 0.0, 1.0)
        angle_rad = 2.0 * np.arccos(dot)
        tol_rad   = np.deg2rad(self._tol.value)
        dist_norm = np.clip(angle_rad / tol_rad, 0.0, 1.0)

        if self._hide_cb.isChecked():
            mask = dist_norm < 1.0
            pts, scal = self._poses[mask, :3], dist_norm[mask]
            if len(pts) == 0:
                pts, scal = np.zeros((1, 3)), np.zeros(1, np.float32)
        else:
            pts, scal = self._poses[:, :3], dist_norm

        cloud = pv.PolyData(pts.astype(np.float64))
        cloud["dist"] = scal.astype(np.float32)
        self._main_plot.add_mesh(
            cloud, scalars="dist", cmap="coolwarm", clim=[0.0, 1.0],
            point_size=3, render_points_as_spheres=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Orient. dist\n(0=match, 1=tol.)"},
            name="workspace",
        )

        closest_q = None
        if self._joint_configs is not None and len(self._joint_configs) > 0:
            closest_q = self._joint_configs[int(np.argmin(angle_rad))]

        if self._live_robot is not None and closest_q is not None:
            self._live_robot.update(self._controlled_joints, closest_q, self._fixed_dict)
            # Re-focus the EE view on the actual EE position
            ee_pos = self._live_robot._robot.get_frame_poses_xyzxyzw(
                frames=[self._ee_frame]
            )[self._ee_frame][:3]
            self._refocus_ee_camera(ee_pos)
            # Draw target-orientation axes at the EE position
            mat = rot.as_matrix()
            scale = 0.08
            for i, (color, name) in enumerate(zip(
                    ("red", "green", "blue"), ("ee_ax_x", "ee_ax_y", "ee_ax_z"))):
                self._ee_plot.add_mesh(
                    pv.Arrow(start=ee_pos, direction=mat[:, i],
                             tip_length=0.30, tip_radius=0.07, shaft_radius=0.03,
                             scale=scale),
                    color=color, name=name,
                )

        self._main_plot.render()
        self._ee_plot.render()

    def _refocus_ee_camera(self, new_focal: np.ndarray):
        """Move the EE camera focal point to new_focal, preserving direction and zoom."""
        cam = self._ee_plot.camera
        old_fp = np.array(cam.focal_point)
        old_pos = np.array(cam.position)
        cam.focal_point = new_focal
        cam.position    = new_focal + (old_pos - old_fp)   # shift by same delta


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_visualizer(
    urdf_string: str,
    controlled_joints: list[str],
    fixed_joints: dict[str, float],
    ee_frame: str,
    n_samples: int = 50_000,
    check_collisions: bool = False,
    urdf_format: str = "urdf",
) -> int:
    """
    Parameters
    ----------
    urdf_path         : Path to a URDF or .xacro file.
    controlled_joints : Joint names that will be sampled.
    fixed_joints      : Remaining joints, held at these values (radians).
    ee_frame          : Frame name used as the end-effector.
    n_samples         : Number of configurations to sample.
    check_collisions  : If True, discard self-colliding configurations.
    urdf_format       : "urdf" or "mjcf".
    """
    from adarl.utils.utils import compile_xacro_string

    text = urdf_string

    robot = Robot(text, robot_description_format=urdf_format)
    print("Joints :", robot.get_joint_names())
    print("Frames :", robot.get_frame_names())

    poses, joint_configs = sample_workspace(
        robot=robot,
        controlled_joints=controlled_joints,
        fixed_joints=fixed_joints,
        ee_frame=ee_frame,
        n_samples=n_samples,
        check_collisions=check_collisions,
    )

    known = set(robot.get_joint_names())
    fixed_dict = {k: np.asarray([float(v)]) for k, v in fixed_joints.items() if k in known}

    print("Loading robot mesh...")
    live_robot = _LiveRobotMesh(robot, text)

    # Re-apply the plugin path fix here: importing robot_helpers triggers a cv2
    # import which resets QT_QPA_PLATFORM_PLUGIN_PATH to cv2's own (incompatible)
    # Qt plugins.  We override it again right before QApplication is created.
    import PyQt5 as _pyqt5
    _plugins = os.path.join(os.path.dirname(_pyqt5.__file__), "Qt5", "plugins")
    if os.path.isdir(_plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _plugins

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = WorkspaceViewer(
        poses, ee_frame,
        live_robot=live_robot,
        joint_configs=joint_configs,
        controlled_joints=controlled_joints,
        fixed_dict=fixed_dict,
    )
    viewer.show()
    return app.exec_()


# ===========================================================================
# CONFIG — edit this block and run the script
# ===========================================================================
if __name__ == "__main__":
    from adarl.utils.utils import compile_xacro_string, pkgutil_get_path
    from pathlib import Path
    model_path = pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf/urdf/centauro.urdf.xacro")
    urdf_string = compile_xacro_string( model_definition_string=Path(model_path).read_text(),
                                        model_kwargs={  "realsense":"false",
                                                        "velodyne" :"false",
                                                        "floating_joint":f"false",
                                                        "sphere_wheel_collision":"true",
                                                        "end_effector_left":"dagana",
                                                        "fixed_base_joint":"true",
                                                        "legs":f"false",
                                                        "dagana_claws_type":"centauro_claws_boxycollision"
                                                        },
                                        extra_pkg_paths={   "centauro_urdf" : pkgutil_get_path("pycentauro","iit-centauro-ros-pkg/centauro_urdf"),
                                                            "dagana_urdf" : pkgutil_get_path("pydagana","iit-dagana-ros-pkg/dagana_urdf")},
                                                    )

    hip_yaw =      0.75
    hip_pitch =    1.25
    knee_pitch =   1.25
    ankle_pitch =  0.0
    ankle_yaw =   -0.75

    sys.exit(run_visualizer(
        urdf_string=urdf_string,   # or .xacro

        # Joints to sample — must match names from robot.get_joint_names()
        controlled_joints=[ "j_arm1_1",
                            "j_arm1_2",
                            "j_arm1_3",
                            "j_arm1_4",
                            "j_arm1_5",
                            "j_arm1_6",
                            "dagana_1_claw_joint"],
        # All other joints, fixed at these values (radians for revolute/prismatic)
        fixed_joints = {  
                "hip_yaw_1" :      -hip_yaw,
                "hip_pitch_1" :    -hip_pitch,
                "knee_pitch_1" :   -knee_pitch,
                "ankle_pitch_1" :  -ankle_pitch,
                "ankle_yaw_1" :    -ankle_yaw,
                "hip_yaw_2" :      hip_yaw,
                "hip_pitch_2" :    hip_pitch,
                "knee_pitch_2" :   knee_pitch,
                "ankle_pitch_2" :  ankle_pitch,
                "ankle_yaw_2" :    ankle_yaw,
                "hip_yaw_3" :      hip_yaw,
                "hip_pitch_3" :    hip_pitch,
                "knee_pitch_3" :   knee_pitch,
                "ankle_pitch_3" :  ankle_pitch,
                "ankle_yaw_3" :    ankle_yaw,
                "hip_yaw_4" :      -hip_yaw,
                "hip_pitch_4" :    -hip_pitch,
                "knee_pitch_4" :   -knee_pitch,
                "ankle_pitch_4" :  -ankle_pitch,
                "ankle_yaw_4" :    -ankle_yaw,
                "torso_yaw" : 0.0,
                "velodyne_joint" : 0,
                "d435_head_joint" : -0.8,
                "j_arm1_1" : 0.52,
                "j_arm1_2" : 0.40,
                "j_arm1_3" : 0.27,
                "j_arm1_4" : -2.00,
                "j_arm1_5" : 0.05,
                "j_arm1_6" : -0.78,
                "j_arm2_1" : 0.52,
                "j_arm2_2" : -0.40,
                "j_arm2_3" : -0.27,
                "j_arm2_4" : -2.00,
                "j_arm2_5" : -0.05,
                "j_arm2_6" : -0.78,
                "j_wheel_1" : 0.0,
                "j_wheel_2" : 0.0,
                "j_wheel_3" : 0.0,
                "j_wheel_4" : 0.0,
                "dagana_1_claw_joint" : 0.3
                },
        # Frame name to use as end-effector — check robot.get_frame_names()
        ee_frame="dagana_1_top_link",

        n_samples=500_000,
        check_collisions=False,   # True = skip self-colliding configs (slower)
    ))
