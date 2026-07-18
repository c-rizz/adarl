"""Unit tests for normalize/unnormalize in adarl.utils.utils.

normalize maps [min, max] -> [-1, 1]; unnormalize is its inverse.
"""
import pytest
import torch as th

from adarl.utils.utils import normalize, unnormalize


@pytest.mark.parametrize("lo,hi", [(-2.0, 2.0), (0.0, 10.0), (-5.0, -1.0)])
def test_normalize_endpoints_and_midpoint(lo, hi):
    assert normalize(lo, lo, hi) == pytest.approx(-1.0)
    assert normalize(hi, lo, hi) == pytest.approx(1.0)
    assert normalize((lo + hi) / 2, lo, hi) == pytest.approx(0.0)


@pytest.mark.parametrize("lo,hi", [(-2.0, 2.0), (0.0, 10.0), (-5.0, -1.0)])
def test_unnormalize_is_inverse_of_normalize(lo, hi):
    for v in (lo, hi, (lo + hi) / 2, lo + 0.37 * (hi - lo)):
        assert unnormalize(normalize(v, lo, hi), lo, hi) == pytest.approx(v)


def test_normalize_tensor_elementwise():
    lo = th.tensor([0.0, -1.0, 10.0])
    hi = th.tensor([1.0, 1.0, 20.0])
    v = th.tensor([0.5, 0.0, 15.0])
    out = normalize(v, lo, hi)
    assert th.allclose(out, th.zeros(3), atol=1e-6)
    assert th.allclose(unnormalize(out, lo, hi), v, atol=1e-6)
