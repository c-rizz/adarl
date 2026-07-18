"""MJX (JAX / WARP, GPU) adapter compliance.

Runs the same backend-agnostic suite against MjxJointImpedanceAdapter, for both the
``jax`` and ``warp`` implementations. Full compliance passes on both.
"""
import pytest

from adapter_compliance import run_vec_adapter_compliance_test
from _backends import build_mjx_impedance, requires_jax_gpu, requires_mjx_warp


@pytest.mark.mjx
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mjx_impl", [
    pytest.param("jax", marks=requires_jax_gpu),
    pytest.param("warp", marks=requires_mjx_warp),
])
def test_mjx_impedance_compliance(mjx_impl):
    adapter = build_mjx_impedance(vec_size=4, mjx_impl=mjx_impl)
    ctx = run_vec_adapter_compliance_test(adapter)
    assert not ctx.failed, "compliance checks failed:\n  " + "\n  ".join(ctx.failed)
