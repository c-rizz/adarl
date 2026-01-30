# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Any

import torch
import torch._prims_common as utils
from torch import Tensor
from torch._prims_common.wrappers import out_wrapper
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map


DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

# None of these functions are publicly accessible; get at them
# from torch._decomps
__all__: list[str] = []

aten = torch._ops.ops.aten


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# This wraps a decomposition and performs various type promotion logic within it, depending on the strategy provided
# We're currently reusing ELEMENTWISE_TYPE_PROMOTION_KIND, although some of the usages are on non-elementwise ops
# Will need to validate the non-elementwise uses
def type_casts(
    f: Callable,
    type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND,
    compute_dtype_only: bool = False,
    include_non_tensor_args: bool = False,
):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        allowed_types = (
            (Tensor, torch.types._Number) if include_non_tensor_args else (Tensor,)
        )  # type: ignore[arg-type]
        flat_args = [
            x
            for x in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(x, allowed_types)
        ]
        computation_dtype, result_dtype = utils.elementwise_dtypes(
            *flat_args, type_promotion_kind=type_promotion
        )

        # TODO: pretty sure this is not quite right
        def increase_prec(x):
            if isinstance(x, Tensor):
                return x.to(computation_dtype)
            else:
                return x

        def decrease_prec(x):
            if isinstance(x, Tensor):
                return x.to(result_dtype)
            else:
                return x

        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        if compute_dtype_only:
            return r
        else:
            return tree_map(decrease_prec, r)

    return inner


compute_only_pw_cast_for_opmath = partial(
    type_casts,
    type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    compute_dtype_only=True,
)
pw_cast_for_opmath = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
pw_cast_for_opmath_non_tensor_args = partial(
    type_casts,
    type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    include_non_tensor_args=True,
)
pw_cast_for_int_to_real = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)


@pw_cast_for_opmath
@out_wrapper()
def _reflection_pad(a: Tensor, padding: tuple[int, ...]) -> Tensor:
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return middle - 1 - (middle - 1 - dim_idx.abs()).abs()

    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )


@pw_cast_for_opmath
@out_wrapper()
def _replication_pad(a: Tensor, padding: tuple[int, ...]) -> Tensor:
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return torch.clamp(dim_idx, 0, middle - 1)

    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )


def _reflection_or_replication_pad(
    a: Tensor,
    padding: tuple[int, ...],
    idx_fn: Callable[[int, int, int], Tensor],
) -> Tensor:
    dim = len(padding) // 2
    torch._check(
        a.dim() in (dim + 1, dim + 2),
        lambda: f"reflection_pad{dim}d requires {dim + 1}D or {dim + 2}D input",
    )
    inp_shape = a.shape[-dim:]
    nc_dim = a.dim() - dim

    padding_left = [padding[2 * (dim - 1 - i)] for i in range(dim)]
    padding_right = [padding[2 * (dim - 1 - i) + 1] for i in range(dim)]

    result = a
    for i in range(dim):
        idx: list[Any] = [None] * result.dim()
        idx[i + nc_dim] = idx_fn(padding_left[i], inp_shape[i], padding_right[i])
        result = aten._unsafe_index(result, idx)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(result)
    result = result.contiguous(memory_format=memory_format)
    return result


@out_wrapper("grad_input")
def _reflection_pad_backward(grad_output, x, padding):
    dim = len(padding) // 2

    dhw = [h - 1 for h in x.shape[-dim:]]

    padding_left = [padding[2 * (dim - 1 - i)] for i in range(dim)]
    padding_right = [padding[2 * (dim - 1 - i) + 1] for i in range(dim)]

    indices = []
    for i in range(x.ndim):
        view_shape = [1] * x.ndim
        view_shape[i] = -1
        indices.append(torch.arange(x.shape[i], device=x.device).view(view_shape))

    b = indices[:-dim]
    xyz = indices[-dim:]

    def index_range_condition(index_range):
        i, lb, ub = index_range
        return torch.logical_and(i >= lb, i <= ub)

    # Areas after reflection:
    #
    #   top-left    |   top     |   top-right
    # -----------------------------------------
    #   left        |   center  |   right
    # -----------------------------------------
    #   bottom-left |   bottom  |   bottom-right
    #
    # The center area is the original matrix. Other areas are reflections.

    center = [xyz[i] + padding_left[i] for i in range(dim)]
    left_reflect = [padding_left[i] - xyz[i] for i in range(dim)]
    right_reflect = [2 * dhw[i] + padding_left[i] - xyz[i] for i in range(dim)]

    # Accumulate gradients from different areas
    # If some of the padding is negative, center load is not always valid
    range_c = [
        (center[i], 0, dhw[i] + padding_left[i] + padding_right[i]) for i in range(dim)
    ]
    cond = functools.reduce(
        aten.logical_and, [index_range_condition(range_c[i]) for i in range(dim)]
    )
    grad = aten._unsafe_masked_index(grad_output, cond, b + center, 0.0)

    def accumulate(grad, out, index_ranges):
        # If the upper bound is less than the lower bound, we can get rid of one accumulation.
        # This happens when the padding size is zero.
        for i in range(dim):
            upper_less_than_lower = index_ranges[i][2] < index_ranges[i][1]
            if isinstance(upper_less_than_lower, bool) and upper_less_than_lower:
                return grad

        cond = functools.reduce(
            aten.logical_and,
            [index_range_condition(index_range) for index_range in index_ranges],
        )
        g = aten._unsafe_masked_index(grad_output, cond, b + out, 0.0)
        return grad + g

    for area in itertools.product(*[[-1, 0, 1] for _ in range(dim)]):
        if area == tuple([0] * dim):
            # center, this is already done.
            continue

        outs = []
        index_ranges = []

        for i in range(dim):
            if area[i] == 0:
                out = center[i]
                index_range = range_c[i]
            elif area[i] == -1:
                out = left_reflect[i]
                index_range = (xyz[i], 1, padding_left[i])
            elif area[i] == 1:
                out = right_reflect[i]
                index_range = (xyz[i], dhw[i] - padding_right[i], dhw[i] - 1)

            outs.append(out)  # type: ignore[possibly-undefined]
            index_ranges.append(index_range)  # type: ignore[possibly-undefined]

        grad = accumulate(grad, outs, index_ranges)

    return grad


