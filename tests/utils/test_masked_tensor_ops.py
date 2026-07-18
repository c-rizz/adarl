"""Unit tests for the masked row-assignment helpers in adarl.utils.utils.

These are used all over the vectorized adapters/envs to update per-environment state
without incurring CUDA syncs, so their exact semantics matter.
"""
import pytest
import torch as th

from adarl.utils.utils import (
    masked_assign,
    masked_to_masked_assign,
    move_masked_to_start,
    expand_tensor_into_lower_dims,
)


def _rows(*vals):
    return th.tensor([[float(v)] * 3 for v in vals])


def test_masked_assign_inplace_selected_rows():
    orig = _rows(1, 2, 3, 4)
    newv = _rows(10, 20, 30, 40)
    mask = th.tensor([True, False, True, False])
    ret = masked_assign(orig, mask, newv)
    assert ret is orig  # inplace, returns the same tensor
    assert th.equal(orig, _rows(10, 2, 30, 4))


def test_masked_assign_not_inplace_leaves_original():
    orig = _rows(1, 2, 3, 4)
    newv = _rows(10, 20, 30, 40)
    mask = th.tensor([True, True, False, False])
    out = masked_assign(orig, mask, newv, inplace=False)
    assert th.equal(orig, _rows(1, 2, 3, 4))       # untouched
    assert th.equal(out, _rows(10, 20, 3, 4))


def test_masked_assign_scalar_broadcast():
    orig = _rows(1, 2, 3, 4)
    mask = th.tensor([False, True, False, True])
    masked_assign(orig, mask, 0.0)
    assert th.equal(orig, _rows(1, 0, 3, 0))


def test_masked_assign_rejects_wrong_mask_shape():
    orig = _rows(1, 2, 3, 4)
    bad_mask = th.tensor([True, False, True])       # len 3 != 4 rows
    with pytest.raises(RuntimeError):
        masked_assign(orig, bad_mask, 0.0)


def test_expand_row_mask_into_lower_dims():
    mask = th.tensor([True, False, True, False])
    expanded = expand_tensor_into_lower_dims(mask, th.Size([4, 3]))
    assert expanded.shape == (4, 3)
    for col in range(3):
        assert th.equal(expanded[:, col], mask)


def test_move_masked_to_start_preserves_order():
    tensor = _rows(10, 20, 30, 40)
    mask = th.tensor([False, True, False, True])
    out = move_masked_to_start(tensor, mask)
    n_true = int(mask.sum())
    # the selected rows (20, 40) land at the front, in their original order
    assert th.equal(out[:n_true], _rows(20, 40))


def test_masked_to_masked_assign_copies_selected_rows():
    dest = _rows(1, 2, 3, 4)
    dest_mask = th.tensor([True, False, True, False])
    src = _rows(100, 200, 300)
    src_mask = th.tensor([True, True, False])       # 2 selected
    clamped = masked_to_masked_assign(dest, dest_mask, src, src_mask)
    # src rows (100, 200) go into dest rows 0 and 2
    assert th.equal(dest, _rows(100, 2, 200, 4))
    assert th.equal(clamped, dest_mask)


def test_masked_to_masked_assign_clamps_to_available_source_rows():
    dest = _rows(1, 2, 3, 4)
    dest_mask = th.tensor([True, True, True, False])  # wants 3
    src = _rows(100, 200)
    src_mask = th.tensor([True, True])                # only 2 available
    clamped = masked_to_masked_assign(dest, dest_mask, src, src_mask)
    # only the first two dest-selected rows get written
    assert th.equal(clamped, th.tensor([True, True, False, False]))
    assert th.equal(dest[:2], _rows(100, 200))
    assert th.equal(dest[3], th.tensor([4.0, 4.0, 4.0]))
