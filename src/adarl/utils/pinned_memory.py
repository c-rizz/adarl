"""Pinned host tensors that take exactly the memory they need.

th.full(..., pin_memory=True) goes through torch's caching host allocator, which rounds every
allocation up to a power of two: a 33 GiB tensor takes 64 GiB. Here the memory is a plain
allocation pinned in place with cudaHostRegister, and unpinned when the last tensor using it is freed.
"""
from __future__ import annotations
import math
import sys
import torch as th


class RegisteredHostMemory(bytearray):
    """Host memory pinned with cudaHostRegister for as long as this object lives.

    Tensors built on it with th.frombuffer keep it alive, so it is unregistered only after the
    last of them, views included, is freed.
    """
    def __init__(self, nbytes : int):
        self._ptr : int | None = None
        super().__init__(nbytes)
        cudart = th.cuda.cudart()
        ptr = th.frombuffer(self, dtype=th.uint8).data_ptr()
        err = cudart.cudaHostRegister(ptr, nbytes, 0)
        if err != cudart.cudaError.success:
            raise RuntimeError(f"cudaHostRegister of {nbytes} bytes failed: {err}")
        self._ptr = ptr

    def __del__(self):
        # At interpreter shutdown the process is going away anyway, and CUDA may already be torn down
        if getattr(self, "_ptr", None) is not None and not sys.is_finalizing():
            th.cuda.cudart().cudaHostUnregister(self._ptr)
            self._ptr = None


def registered_pinned_full(size : tuple[int,...], fill_value, dtype : th.dtype) -> th.Tensor:
    """Like th.full(size, fill_value, dtype=dtype, pin_memory=True), without the rounding up."""
    numel = math.prod(size)
    if numel == 0:
        return th.empty(size, dtype=dtype, pin_memory=True)
    mem = RegisteredHostMemory(numel*dtype.itemsize)
    t = th.frombuffer(mem, dtype=th.uint8).view(dtype).view(size)
    t.fill_(fill_value)
    return t
