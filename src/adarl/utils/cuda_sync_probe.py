#!/usr/bin/env python3
"""Probe a torch op for a *hidden* CUDA sync, by timing it behind a busy GPU.

torch.cuda.set_sync_debug_mode("warn") only flags the syncs the runtime explicitly annotates;
some (e.g. ``th.rand(..., generator=cuda_gen)``, kernels that read a device scalar for bookkeeping)
slip through. This probe detects them empirically:

  1. Queue a large "backlog" of GPU work (takes ~backlog_sec on the GPU).
  2. Immediately time the op *on the CPU* (no synchronize around it).
     - No sync  -> the CPU just enqueues the kernel and returns in microseconds.
     - Sync     -> the CPU blocks until the whole backlog drains, so its wall time ~= backlog_sec.
  3. Compare the op's CPU time against two calibration references measured the same way:
     a known no-sync op (x+1) and a known sync op (x.item()).

Run:
    python -m adarl.utils.dbg.cuda_sync_probe                 # runs operation_under_test()
    python -m adarl.utils.dbg.cuda_sync_probe --op rand_gen   # a preset (compare rand_gen vs rand_nogen)
    python -m adarl.utils.dbg.cuda_sync_probe --list          # list presets
    python -m adarl.utils.dbg.cuda_sync_probe --backlog-ms 200 --trials 15

To scrutinize your own op, edit operation_under_test() below (or add a preset to PRESETS).
"""
from __future__ import annotations

import argparse
import time

import torch as th


class ProbeCtx:
    """Scratch objects handed to each op so ops stay one-liners."""
    def __init__(self, device: th.device):
        self.device = device
        self.gen = th.Generator(device=device)
        self.gen.manual_seed(0)
        self.x = th.randn((4096, 64), device=device)          # generic scratch tensor
        self.mask = th.rand((4096,), device=device) > 0.5      # for data-dependent ops



# =========================================================================================
#  EDIT HERE: the single operation you want to scrutinize. It receives a ProbeCtx.
#  (This default mirrors BaseVecEnv._thrand: th.rand with a CUDA generator.)
# =========================================================================================
def operation_under_test(ctx: ProbeCtx):
    # return th.rand(size=(4096, 64), dtype=th.float32, device=ctx.device, generator=ctx.gen) # SYNC
    # return th.as_tensor([1,2,3], device=ctx.device) # SYNC
    # return th.zeros((1024,), device=ctx.device) # NO SYNC
    # return th.empty((1024,), device=ctx.device) # NO SYNC

    # import jax.numpy as jnp
    # import jax
    # import numpy as np
    # return jnp.array(np.empty((1024, 0, 13), dtype=np.float32), device=jax.devices("gpu")[0]) # NO SYNC

    t = th.empty(size=(1024,18), dtype=th.float32).to(device=ctx.device, non_blocking=ctx.device.type=="cuda")
    th.nn.init.trunc_normal_(t, 0, 1, -3, 3, generator=ctx.gen)
    return t
# =========================================================================================


# Known-good calibration references (do NOT edit): one that never syncs, one that always does.
def _ref_nosync(ctx: ProbeCtx):
    return ctx.x.add(1.0)              # pure device op

def _ref_sync(ctx: ProbeCtx):
    return ctx.x[0, 0].item()          # device->host scalar read -> always syncs


PRESETS = {
    "custom":     operation_under_test,
    "rand_gen":   lambda ctx: th.rand((4096, 64), dtype=th.float32, device=ctx.device, generator=ctx.gen),
    "rand_nogen": lambda ctx: th.rand((4096, 64), dtype=th.float32, device=ctx.device),
    "randn_gen":  lambda ctx: th.randn((4096, 64), device=ctx.device, generator=ctx.gen),
    "randint_gen":lambda ctx: th.randint(0, 10, (4096,), device=ctx.device, generator=ctx.gen),
    "item":       _ref_sync,
    "add":        _ref_nosync,
    "cpu":        lambda ctx: ctx.x.cpu(),          # D2H copy -> sync
    "nonzero":    lambda ctx: ctx.mask.nonzero(),   # data-dependent output size -> sync
    "masked":     lambda ctx: ctx.x[ctx.mask],      # boolean index -> sync
}


def _gpu_time_s(fn) -> float:
    """Wall time the GPU spends on fn(), via CUDA events."""
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    th.cuda.synchronize()
    start.record()
    fn()
    end.record()
    th.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def _make_backlog(device: th.device, target_s: float):
    """Return (run_backlog, backlog_seconds): a closure that queues ~target_s of GPU matmuls."""
    n = 2048
    a = th.randn((n, n), device=device)
    b = th.randn((n, n), device=device)

    def run(k: int):
        c = a
        for _ in range(k):
            c = c @ b
        # write into a persistent buffer so the work can't be optimized away
        a.copy_(c)

    iters = 4
    for _ in range(30):  # calibrate iters up to target duration
        t = _gpu_time_s(lambda: run(iters))
        if t >= target_s or iters > 200000:
            break
        iters = max(iters + 1, int(iters * (target_s / max(t, 1e-5))) + 1)
    backlog_s = _gpu_time_s(lambda: run(iters))
    return (lambda: run(iters)), backlog_s


def _cpu_time_under_backlog(op, backlog, ctx: ProbeCtx, trials: int) -> float:
    """Median CPU wall time of op() issued right after queuing the backlog."""
    times = []
    for _ in range(trials):
        th.cuda.synchronize()
        backlog()                       # queue heavy GPU work (returns immediately, async)
        t0 = time.perf_counter()
        op(ctx)                         # the op under test (measured on the CPU)
        dt = time.perf_counter() - t0
        th.cuda.synchronize()           # drain before the next trial
        times.append(dt)
    times.sort()
    return times[len(times) // 2]


def run_probe(op_name: str, op, backlog_ms: float = 120.0, trials: int = 11):
    if not th.cuda.is_available():
        raise SystemExit("No CUDA device available; this probe needs a GPU.")
    device = th.device("cuda")
    ctx = ProbeCtx(device)

    # warm up: JIT/compile, populate the caching allocator (first cudaMalloc can itself sync)
    for fn in (op, _ref_nosync, _ref_sync):
        for _ in range(10):
            fn(ctx)
    th.cuda.synchronize()

    backlog, backlog_s = _make_backlog(device, target_s=backlog_ms / 1000.0)
    backlog(); th.cuda.synchronize()  # warm the backlog kernels

    rows = [
        ("no-sync control  (x + 1)", _ref_nosync),
        ("sync control     (.item())", _ref_sync),
        (f"OP: {op_name}", op),
    ]
    measured = {name: _cpu_time_under_backlog(o, backlog, ctx, trials) for name, o in rows}

    # verdict threshold: a full sync waits ~backlog_s; anything past ~30% of it is clearly waiting.
    thresh = 0.30 * backlog_s
    print(f"\nbacklog GPU time = {backlog_s*1e3:8.2f} ms   (sync -> CPU waits ~this long; "
          f"threshold = {thresh*1e3:.2f} ms)\n")
    print(f"  {'operation':26s} {'cpu time':>11s}  {'% of backlog':>12s}   verdict")
    print(f"  {'-'*26} {'-'*11}  {'-'*12}   {'-'*20}")
    for name, _ in rows:
        t = measured[name]
        pct = 100.0 * t / backlog_s if backlog_s > 0 else 0.0
        verdict = "SYNC" if t > thresh else "no sync"
        print(f"  {name:26s} {t*1e3:8.3f} ms  {pct:11.1f}%   {verdict}")
    print()
    print("Interpretation: the op syncs iff its cpu time tracks the 'sync ref' (~backlog) rather")
    print("than the 'no-sync ref' (~0). Increase --backlog-ms if the two references aren't well")
    print("separated.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="custom", help="preset name (see --list) or 'custom' for operation_under_test()")
    ap.add_argument("--backlog-ms", type=float, default=120.0, help="target GPU backlog duration in ms")
    ap.add_argument("--trials", type=int, default=11, help="number of timed trials (median is reported)")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    args = ap.parse_args()

    if args.list:
        print("presets:", ", ".join(PRESETS))
        raise SystemExit(0)
    if args.op not in PRESETS:
        raise SystemExit(f"unknown --op '{args.op}'. presets: {', '.join(PRESETS)}")

    run_probe(args.op, PRESETS[args.op], backlog_ms=args.backlog_ms, trials=args.trials)
