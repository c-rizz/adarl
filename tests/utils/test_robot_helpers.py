"""Unit tests for adarl.utils.robot_helpers.Robot (pinocchio-backed, CPU only).

Covers single-file construction (URDF + MJCF), forward kinematics, the multi-model merge
(static world and free-floating object via additional_models), part placement, pickling, and
collision detection with added collision boxes.
"""
import pickle

import numpy as np
import pytest

pinocchio = pytest.importorskip("pinocchio")
from adarl.utils.robot_helpers import Robot, ModelDescription


# 1-DOF revolute arm: base collision box at the origin, arm collision box up at z=2 (so base and
# arm never self-collide at the neutral pose).
ROBOT_1DOF = """<robot name="rob">
  <link name="base"><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
  <link name="arm"><collision><origin xyz="0 0 2"/><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
</robot>"""

# single prismatic joint along z, for a deterministic FK check
PRISMATIC = """<robot name="pri">
  <link name="base"/>
  <link name="arm"/>
  <joint name="slide" type="prismatic"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""

# static (jointless) world in MJCF
WORLD_STATIC_MJCF = """<mujoco model="world"><worldbody>
  <body name="table"><geom type="box" size="1 1 0.05"/></body></worldbody></mujoco>"""

# a free-floating object (URDF floating joint) -> free-flyer, +7 nq / +6 nv
FREE_OBJ = """<robot name="freeobj"><link name="fbase"/>
  <link name="obj"><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="obj_float" type="floating"><parent link="fbase"/><child link="obj"/></joint></robot>"""


def nq(robot: Robot) -> int:
    return len(robot.get_joint_pose())


# ------------------------------------------------------------------ construction

def test_single_urdf_construction():
    r = Robot(PRISMATIC, "urdf")
    assert "slide" in r.get_joint_names()
    assert set(["base", "arm"]).issubset(set(r.get_frame_names()))
    assert nq(r) == 1
    assert np.allclose(r.get_joint_pose(), [0.0])          # neutral == 0 for a prismatic joint


def test_mjcf_construction():
    r = Robot(WORLD_STATIC_MJCF, "mjcf")
    assert any("table" in g for g in r.get_geom_names())
    assert nq(r) == 0                                       # a jointless world has no DOF


def test_unsupported_format_raises():
    with pytest.raises(NotImplementedError):
        Robot("<sdf/>", "sdf")


# ------------------------------------------------------------------ forward kinematics

def test_forward_kinematics_prismatic():
    r = Robot(PRISMATIC, "urdf")
    r.set_joint_pose(np.array([0.5]))
    arm = r.get_frame_poses_xyzxyzw(frames=["arm"])["arm"]
    assert np.allclose(arm[:3], [0.0, 0.0, 0.5], atol=1e-6)


# ------------------------------------------------------------------ multi-model merge

def test_merge_static_world_adds_geoms_without_dofs():
    base = Robot(ROBOT_1DOF, "urdf")
    merged = Robot(ROBOT_1DOF, "urdf",
                   additional_models=[ModelDescription(WORLD_STATIC_MJCF, "mjcf")])
    assert nq(merged) == nq(base)                           # static world -> no extra DOF
    assert "j1" in merged.get_joint_names()                 # robot joints preserved
    assert len(merged.get_geom_names()) > len(base.get_geom_names())
    assert any("table" in g for g in merged.get_geom_names())


def test_merge_static_world_placement():
    merged = Robot(ROBOT_1DOF, "urdf", additional_models=[
        ModelDescription(WORLD_STATIC_MJCF, "mjcf",
                         placement_xyz_xyzw=np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]))])
    table = merged.get_frame_poses_xyzxyzw(frames=["table"])["table"]
    assert np.allclose(table[:3], [0.0, 0.0, 0.7], atol=1e-6)


def test_merge_free_floating_adds_dofs_after_robot():
    base = Robot(ROBOT_1DOF, "urdf")
    merged = Robot(ROBOT_1DOF, "urdf", additional_models=[ModelDescription(FREE_OBJ, "urdf")])
    assert nq(merged) == nq(base) + 7                       # free-flyer -> +7 nq
    # robot joint keeps its index at the front; the free-flyer is appended after it
    assert merged._model.joints[merged._joint_name_to_idx["j1"]].idx_q == 0
    props = merged.get_joint_properties(["j1", "obj_float"])
    assert props["j1"]["type"] == Robot.JOINT_TYPES.REVOLUTE
    assert props["obj_float"]["type"] == Robot.JOINT_TYPES.FLOATING
    # neutral config ends with an identity quaternion (valid, not zeros)
    assert np.allclose(merged.get_joint_pose()[-4:], [0.0, 0.0, 0.0, 1.0])


def test_free_floating_object_pose_is_driven_by_q():
    merged = Robot(ROBOT_1DOF, "urdf", additional_models=[ModelDescription(FREE_OBJ, "urdf")])
    merged.set_joint_pose_by_names({"obj_float": np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])})
    obj = merged.get_frame_poses_xyzxyzw(frames=["obj"])["obj"]
    assert np.allclose(obj[:3], [1.0, 2.0, 3.0], atol=1e-6)


# ------------------------------------------------------------------ pickling

def test_pickle_roundtrip_preserves_merged_model():
    merged = Robot(ROBOT_1DOF, "urdf", additional_models=[
        ModelDescription(WORLD_STATIC_MJCF, "mjcf"),
        ModelDescription(FREE_OBJ, "urdf")])
    restored = pickle.loads(pickle.dumps(merged))
    assert nq(restored) == nq(merged)
    assert restored.get_joint_names() == merged.get_joint_names()
    assert restored.get_geom_names() == merged.get_geom_names()
    # FK still works after unpickling
    restored.set_joint_pose_by_names({"obj_float": np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0])})
    obj = restored.get_frame_poses_xyzxyzw(frames=["obj"])["obj"]
    assert np.allclose(obj[:3], [0.0, 0.0, 5.0], atol=1e-6)


# ------------------------------------------------------------------ collision detection

def test_add_collision_box_detects_and_clears_overlap():
    r = Robot(ROBOT_1DOF, "urdf")
    # box overlapping the base collision geom at the origin
    r.add_collision_box(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                        (0.2, 0.2, 0.2), collision_obj_id="probe")
    collisions = r.get_all_collisions()
    assert any("probe" in pair for pair in collisions), f"expected a probe collision, got {collisions}"
    assert r.has_collisions()[0] is True

    # move the probe far away -> no longer colliding with anything
    r.move_collision_object("probe", np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 1.0]))
    collisions = r.get_all_collisions()
    assert not any("probe" in pair for pair in collisions), f"probe should be clear, got {collisions}"
