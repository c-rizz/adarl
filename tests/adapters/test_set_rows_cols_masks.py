"""Unit tests for set_rows_cols_masks (the masked sub-block writer used by the MJX adapter).

Regression guard for the bug where a sparse boolean mask combined with full-size vals padded
jnp.nonzero with index 0 — clobbering row 0 (env 0) and misassigning per-row values. This is
what made env-0 impulses vanish after one step in the videos.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
_mjx = pytest.importorskip("adarl.adapters.MjxAdapter")
set_rows_cols_masks = _mjx.set_rows_cols_masks

pytestmark = pytest.mark.mjx


def test_sparse_boolean_mask_leaves_unmasked_rows_untouched():
    # (envs, bodies, 2): only env 2 is selected, at body column 1
    arr = jnp.full((4, 3, 2), -1.0)
    body = jnp.array([1])
    vals = (jnp.arange(4).reshape(4, 1, 1) + 10.0) * jnp.ones((4, 1, 2))  # env i -> 10+i
    out = np.asarray(set_rows_cols_masks(arr, [jnp.array([False, False, True, False]), body], vals))

    assert out[0, 1].tolist() == [-1, -1]        # env 0 NOT clobbered (the historical bug)
    assert out[1, 1].tolist() == [-1, -1]        # other unmasked env untouched
    assert out[3, 1].tolist() == [-1, -1]
    assert out[2, 1].tolist() == [12, 12]        # masked env gets ITS OWN value (no misassignment)
    assert out[2, 0].tolist() == [-1, -1]        # untouched columns stay put
    assert out[2, 2].tolist() == [-1, -1]


def test_effort_like_sparse_mask_writes_only_selected():
    arr = jnp.zeros((4, 5))
    cols = jnp.array([0, 2])
    vals = (jnp.arange(4).reshape(4, 1) + 1.0) * jnp.ones((4, 2))  # env i -> i+1
    out = np.asarray(set_rows_cols_masks(arr, [jnp.array([True, False, True, False]), cols], vals))

    assert out[0].tolist() == [1, 0, 1, 0, 0]
    assert out[1].tolist() == [0, 0, 0, 0, 0]    # unmasked env untouched
    assert out[2].tolist() == [3, 0, 3, 0, 0]
    assert out[3].tolist() == [0, 0, 0, 0, 0]


def test_all_true_mask_is_full_overwrite():
    arr = jnp.zeros((4, 5))
    cols = jnp.array([0, 2])
    vals = (jnp.arange(4).reshape(4, 1) + 1.0) * jnp.ones((4, 2))
    out = np.asarray(set_rows_cols_masks(arr, [jnp.array([True, True, True, True]), cols], vals))

    assert out[:, 0].tolist() == [1, 2, 3, 4]
    assert out[:, 2].tolist() == [1, 2, 3, 4]
    assert out[:, 1].tolist() == [0, 0, 0, 0]    # unselected column untouched


def test_integer_index_masks_do_cross_product_write():
    out = np.asarray(set_rows_cols_masks(jnp.zeros((4, 4)),
                                         [jnp.array([1, 3]), jnp.array([0, 2])],
                                         jnp.ones((2, 2)) * 7))
    assert out[1, 0] == 7 and out[1, 2] == 7 and out[3, 0] == 7 and out[3, 2] == 7
    assert out[0, 0] == 0 and out[2, 2] == 0     # unselected positions untouched
