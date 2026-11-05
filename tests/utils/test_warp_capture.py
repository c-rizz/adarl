"""A garbage collection during a Warp capture must not invalidate it, see adarl.utils.warp_capture."""
import gc
import multiprocessing
import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _capture_with_collection(protected : bool, result):
    import warp as wp
    from adarl.utils.warp_capture import install_gc_safe_capture
    wp.init()
    if protected:
        install_gc_safe_capture()
    gc.collect()
    out = wp.zeros(1, dtype=float, device="cuda:0")
    holder = {"held" : wp.zeros(64, dtype=float, device="cpu", pinned=True)}
    holder["self"] = holder # only the cyclic collector can free it
    del holder
    gc.set_threshold(50) # a collection is due soon after the capture starts
    was_enabled = gc.isenabled()
    try:
        with wp.ScopedCapture(device="cuda:0") as capture:
            garbage_triggers = [[] for _ in range(1000)]
            out.fill_(2.0)
        wp.capture_launch(capture.graph)
        wp.synchronize_device("cuda:0")
        result.put(("ok", float(out.numpy()[0]), was_enabled == gc.isenabled()))
    except Exception as e:
        result.put(("failed", str(e), was_enabled == gc.isenabled()))


def _run(protected : bool):
    ctx = multiprocessing.get_context("spawn") # a failed capture leaves the process' cuda context unusable
    result = ctx.Queue()
    process = ctx.Process(target=_capture_with_collection, args=(protected, result))
    process.start()
    outcome = result.get(timeout=120)
    process.join(timeout=60)
    return outcome


def test_collection_during_capture_is_postponed():
    status, value, gc_state_restored = _run(protected=True)
    assert status == "ok", value
    assert value == 2.0
    assert gc_state_restored
