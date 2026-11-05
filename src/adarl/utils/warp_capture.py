"""Keeps Python's garbage collector out of Warp's CUDA graph captures.

The collector can run on any allocation, also while a graph is being captured, and the destructors it calls can
do CUDA calls that are not allowed during a capture. The capture then fails, or just ends up invalidated with
Warp CUDA error 901 at some unrelated later launch, see https://github.com/NVIDIA/warp/issues/2053. Warp fixed
this in 1.19 for its own objects, while mujoco-mjx pins an older version. Captures do not happen just at startup:
the Warp ffi captures a new graph whenever the jax buffers it gets are at new addresses.
"""
from __future__ import annotations
import collections
import gc
import threading
import adarl.utils.dbg.ggLog as ggLog

_lock = threading.Lock()
_active_captures = 0
_gc_was_enabled = False
_captures_count = 0
_postponed_collections = 0
MAX_REPORTED_COLLECTIONS = 10
LOG_EVERY_CAPTURES = 1000


def _gc_off():
    global _active_captures, _gc_was_enabled, _captures_count
    with _lock:
        if _active_captures == 0:
            _gc_was_enabled = gc.isenabled()
            gc.disable()
        _active_captures += 1
        _captures_count += 1
        if _captures_count % LOG_EVERY_CAPTURES == 0:
            ggLog.info(f"Warp graph captures so far: {_captures_count}, with {_postponed_collections} postponed garbage collections")


def _report_garbage(collection : int):
    """Logs what the postponed collection would have destroyed during the capture, and destroys it."""
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        types = collections.Counter(f"{type(o).__module__}.{type(o).__qualname__}" for o in gc.garbage)
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
    ggLog.info(f"Warp capture #{_captures_count}: postponed garbage collection #{collection}, "
               f"it would have collected: {types.most_common(12)}")
    gc.collect()


def _gc_on():
    global _active_captures, _postponed_collections
    with _lock:
        _active_captures -= 1
        if _active_captures > 0 or not _gc_was_enabled:
            return
        gc.enable()
        postponed = gc.get_count()[0] >= gc.get_threshold()[0]
        if postponed:
            _postponed_collections += 1
        collection = _postponed_collections
    if postponed and collection <= MAX_REPORTED_COLLECTIONS:
        _report_garbage(collection)


def install_gc_safe_capture():
    """Make every wp.ScopedCapture of this process, the ones of Warp's jax ffi included, run with the collector off."""
    import warp as wp
    if getattr(wp.ScopedCapture, "_adarl_gc_safe", False):
        return

    class GcSafeScopedCapture(wp.ScopedCapture):
        _adarl_gc_safe = True

        def __enter__(self):
            _gc_off()
            try:
                return super().__enter__()
            except BaseException:
                _gc_on()
                raise

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                _gc_on()

    wp.ScopedCapture = GcSafeScopedCapture
