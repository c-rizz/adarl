#!/usr/bin/env python3
"""Poll GPU memory usage at high frequency and track peak VRAM utilization.

Usable both as a command line tool and as an importable helper.

CLI examples:
    # Monitor all GPUs at 100 Hz until Ctrl-C, then print peak usage
    python -m adarl.utils.vram_monitor

    # Monitor only GPU 0 and 1 at 500 Hz for 30 seconds
    python -m adarl.utils.vram_monitor --gpus 0 1 --hz 500 --duration 30

    # Monitor while a command runs, and report the peak it reached
    python -m adarl.utils.vram_monitor --hz 200 -- python train.py --foo bar

Programmatic use:
    from adarl.utils.vram_monitor import VramMonitor
    with VramMonitor(hz=200) as mon:
        ... do gpu work ...
    print(mon.peak_report())
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

try:
    import adarl.utils.dbg.ggLog as ggLog
except Exception:  # allow standalone use outside a full adarl install
    ggLog = None


def _log_info(msg: str) -> None:
    if ggLog is not None:
        ggLog.info(msg)
    else:
        print(msg, flush=True)


def _log_warn(msg: str) -> None:
    if ggLog is not None:
        ggLog.warn(msg)
    else:
        print(msg, file=sys.stderr, flush=True)


def _bytes_to_mib(b: int) -> float:
    return b / (1024.0 * 1024.0)


class VramMonitor:
    """Background poller that tracks peak per-GPU VRAM usage.

    Polls ``nvmlDeviceGetMemoryInfo`` in a daemon thread at ``hz`` and keeps,
    for each monitored GPU, the maximum ``used`` value ever observed.
    """

    def __init__(self,
                 hz: float = 100.0,
                 gpu_indices: Optional[List[int]] = None):
        if hz <= 0:
            raise ValueError(f"hz must be > 0, got {hz}")
        self._period = 1.0 / hz
        self.hz = hz
        self._requested_indices = gpu_indices

        import pynvml
        self._pynvml = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if gpu_indices is None:
            gpu_indices = list(range(count))
        else:
            for i in gpu_indices:
                if i < 0 or i >= count:
                    raise ValueError(f"GPU index {i} out of range (found {count} GPUs)")
        self._indices = gpu_indices
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}
        self._names = {i: pynvml.nvmlDeviceGetName(self._handles[i]) for i in gpu_indices}
        self._totals = {i: pynvml.nvmlDeviceGetMemoryInfo(self._handles[i]).total
                        for i in gpu_indices}

        self._peak_used: Dict[int, int] = {i: 0 for i in gpu_indices}
        self._last_used: Dict[int, int] = {i: 0 for i in gpu_indices}
        self._sample_count = 0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    # -- sampling ---------------------------------------------------------
    def _sample_once(self) -> None:
        for i in self._indices:
            used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handles[i]).used
            with self._lock:
                self._last_used[i] = used
                if used > self._peak_used[i]:
                    self._peak_used[i] = used
        with self._lock:
            self._sample_count += 1

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop_evt.is_set():
            try:
                self._sample_once()
            except self._pynvml.NVMLError as e:
                _log_warn(f"vram_monitor: NVML sampling error: {e}")
            next_t += self._period
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                self._stop_evt.wait(sleep_t)
            else:
                # We fell behind; resync so we don't busy-spin trying to catch up.
                next_t = time.perf_counter()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "VramMonitor":
        if self._thread is not None:
            raise RuntimeError("VramMonitor already started")
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run,
                                        name="vram_monitor",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "VramMonitor":
        if self._thread is None:
            return self
        self._stop_evt.set()
        self._thread.join()
        self._end_time = time.perf_counter()
        self._thread = None
        return self

    def __enter__(self) -> "VramMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # -- reporting --------------------------------------------------------
    def peak_used(self) -> Dict[int, int]:
        """Return {gpu_index: peak used bytes}."""
        with self._lock:
            return dict(self._peak_used)

    def last_used(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._last_used)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time

    def peak_report(self) -> str:
        peaks = self.peak_used()
        elapsed = self.elapsed()
        samples = self.sample_count
        eff_hz = samples / elapsed if elapsed > 0 else 0.0
        lines = [f"VRAM peak usage (polled ~{self.hz:.0f} Hz, "
                 f"{samples} samples over {elapsed:.1f}s, effective {eff_hz:.0f} Hz):"]
        for i in self._indices:
            total = self._totals[i]
            peak = peaks[i]
            pct = 100.0 * peak / total if total else 0.0
            lines.append(f"  GPU {i} ({self._names[i]}): "
                         f"peak {_bytes_to_mib(peak):.0f} MiB / "
                         f"{_bytes_to_mib(total):.0f} MiB ({pct:.1f}%)")
        return "\n".join(lines)


def _run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Poll GPU VRAM usage at high frequency and report peak utilization.")
    parser.add_argument("--hz", type=float, default=100.0,
                        help="Polling frequency in Hz (default: 100).")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU indices to monitor (default: all).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until "
                             "Ctrl-C or, if given, until the command finishes).")
    parser.add_argument("--print-interval", type=float, default=1.0,
                        help="Seconds between live peak printouts, 0 to disable "
                             "(default: 1.0).")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Optional command to run while monitoring "
                             "(prefix with --). Monitoring stops when it exits.")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    try:
        mon = VramMonitor(hz=args.hz, gpu_indices=args.gpus)
    except Exception as e:
        _log_warn(f"vram_monitor: failed to initialize NVML: {e}")
        return 1

    mon.start()
    _log_info(f"vram_monitor: monitoring GPUs {mon._indices} at {args.hz:.0f} Hz")

    proc: Optional[subprocess.Popen] = None
    if command:
        proc = subprocess.Popen(command)

    rc = 0
    deadline = (mon._start_time + args.duration) if args.duration else None
    last_print = time.perf_counter()
    try:
        while True:
            if proc is not None and proc.poll() is not None:
                rc = proc.returncode
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            now = time.perf_counter()
            if args.print_interval > 0 and (now - last_print) >= args.print_interval:
                peaks = mon.peak_used()
                _log_info("vram_monitor: current peaks -> " + ", ".join(
                    f"GPU{i}: {_bytes_to_mib(peaks[i]):.0f} MiB "
                    f"({100.0 * peaks[i] / mon._totals[i]:.1f}%)"
                    for i in mon._indices))
                last_print = now
            time.sleep(min(0.1, args.print_interval if args.print_interval > 0 else 0.1))
    except KeyboardInterrupt:
        _log_info("vram_monitor: interrupted")
        if proc is not None and proc.poll() is None:
            proc.terminate()
    finally:
        mon.stop()

    _log_info(mon.peak_report())
    return rc


if __name__ == "__main__":
    sys.exit(_run_cli())
