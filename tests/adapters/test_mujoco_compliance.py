"""MuJoCo (CPU) adapter compliance.

Runs the full backend-agnostic behavioral suite against MujocoJointImpedanceAdapter and
asserts every check passes. This is the reference green run of the compliance suite.
"""
import pytest

from adapter_compliance import run_vec_adapter_compliance_test
from _backends import build_mujoco_impedance


@pytest.mark.mujoco
@pytest.mark.slow
def test_mujoco_impedance_compliance():
    adapter = build_mujoco_impedance(vec_size=1)
    ctx = run_vec_adapter_compliance_test(adapter)
    assert not ctx.failed, "compliance checks failed:\n  " + "\n  ".join(ctx.failed)
