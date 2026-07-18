"""Unit tests for the quaternion / pose helpers in adarl.utils.utils.

Conventions under test:
  * quaternions are xyzw
  * quat_mul_xyzw(q1, q2) == q1 * q2  (apply q2, then q1)
  * th_quat_rotate(v_xyz, q_xyzw) rotates the vector v by q
"""
import math

import pytest
import torch as th

from adarl.utils.utils import (
    build_pose,
    quat_mul_xyzw,
    th_quat_conj,
    th_quat_rotate,
    th_quat_combine,
    quat_angle_xyzw,
)

IDENTITY = th.tensor([0.0, 0.0, 0.0, 1.0])


def quat_about_z(angle: float) -> th.Tensor:
    return th.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def assert_close(a, b, atol=1e-5):
    a = th.as_tensor(a, dtype=th.float32)
    b = th.as_tensor(b, dtype=th.float32)
    assert th.allclose(a, b, atol=atol), f"{a.tolist()} != {b.tolist()}"


def test_conj_of_identity_is_identity():
    assert_close(th_quat_conj(IDENTITY), IDENTITY)


def test_conj_negates_vector_part_only():
    q = th.tensor([0.1, -0.2, 0.3, 0.9])
    assert_close(th_quat_conj(q), th.tensor([-0.1, 0.2, -0.3, 0.9]))


def test_quat_times_its_conjugate_is_identity():
    q = quat_about_z(0.7)
    r = quat_mul_xyzw(q, th_quat_conj(q))
    assert_close(r[:3], th.zeros(3))
    assert abs(abs(float(r[3])) - 1.0) < 1e-5


def test_multiplying_by_identity_is_noop():
    q = quat_about_z(0.9)
    assert_close(quat_mul_xyzw(IDENTITY, q), q)
    assert_close(quat_mul_xyzw(q, IDENTITY), q)


def test_rotate_by_identity_is_noop():
    v = th.tensor([1.0, 2.0, 3.0])
    assert_close(th_quat_rotate(v, IDENTITY), v)


def test_rotate_x_axis_90deg_about_z_gives_y_axis():
    v = th.tensor([1.0, 0.0, 0.0])
    assert_close(th_quat_rotate(v, quat_about_z(math.pi / 2)), th.tensor([0.0, 1.0, 0.0]))


def test_two_90deg_rotations_compose_to_180deg():
    q90 = quat_about_z(math.pi / 2)
    q180 = quat_mul_xyzw(q90, q90)
    assert_close(th_quat_rotate(th.tensor([1.0, 0.0, 0.0]), q180), th.tensor([-1.0, 0.0, 0.0]))


def test_rotation_preserves_vector_norm():
    v = th.tensor([0.3, -1.2, 2.5])
    rotated = th_quat_rotate(v, quat_about_z(1.1))
    assert_close(rotated.norm(), v.norm())


def test_quat_combine_is_reversed_multiplication():
    a = quat_about_z(0.4)
    b = quat_about_z(-0.9)
    assert_close(th_quat_combine(a, b), quat_mul_xyzw(b, a))


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi - 1e-3])
def test_quat_angle_matches_construction_angle(angle):
    assert abs(float(quat_angle_xyzw(quat_about_z(angle))) - angle) < 1e-4


def test_batched_rotation():
    n = 5
    quats = th.stack([quat_about_z(a) for a in th.linspace(0.0, 1.5, n)])
    vecs = th.randn(n, 3)
    rotated = th_quat_rotate(vecs, quats)
    assert rotated.shape == (n, 3)
    # rotation about z keeps the z component unchanged
    assert_close(rotated[:, 2], vecs[:, 2])


def test_build_pose_fields():
    pose = build_pose(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    assert_close(pose.position, th.tensor([1.0, 2.0, 3.0]))
    assert_close(pose.orientation_xyzw, th.tensor([0.0, 0.0, 0.0, 1.0]))
