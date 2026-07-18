"""Backend capability detection, skip decorators, and adapter builders.

Imported by the adapter test modules (sibling import; pytest puts this directory on
sys.path). Anything unavailable (no GPU, backend not installed) turns into a clean skip.
"""
from __future__ import annotations

import functools
import tempfile

import pytest
import torch as th


def _log_folder() -> str:
    """A throwaway temp dir for adapter scenario logs, so tests never write into the repo."""
    return tempfile.mkdtemp(prefix="adarl_test_logs_")


# --------------------------------------------------------------------------------------
# capability detection (cached; import/errors -> "not available")
# --------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def has_cuda() -> bool:
    try:
        return bool(th.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def has_jax_gpu() -> bool:
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def has_mjx_warp() -> bool:
    if not has_jax_gpu():
        return False
    try:
        import importlib.util
        from mujoco.mjx._src.io import types  # type: ignore
        return hasattr(types.Impl, "WARP") and importlib.util.find_spec("warp") is not None
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def has_genesis() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("genesis") is not None
    except Exception:
        return False


requires_jax_gpu = pytest.mark.skipif(not has_jax_gpu(), reason="no JAX GPU device available")
requires_mjx_warp = pytest.mark.skipif(not has_mjx_warp(), reason="mjx WARP backend not available")
requires_genesis = pytest.mark.skipif(not has_genesis(), reason="genesis not installed")


# --------------------------------------------------------------------------------------
# adapter builders (kept in one place so tests and future benchmarks agree on config)
# --------------------------------------------------------------------------------------

def build_mujoco_impedance(vec_size: int = 1):
    from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
    return MujocoJointImpedanceAdapter(vec_size=vec_size,
                                       sim_step_dt=1 / 512,
                                       step_length_sec=24 / 512,
                                       output_th_device=th.device("cpu"),
                                       reference_filter_mode="none",
                                       log_folder=_log_folder())


def build_mjx_impedance(vec_size: int = 4, mjx_impl: str = "jax"):
    import jax
    from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
    return MjxJointImpedanceAdapter(vec_size=vec_size,
                                    enable_rendering=False,
                                    jax_device=jax.devices("gpu")[0],
                                    output_th_device=th.device("cuda", 0),
                                    sim_step_dt=1 / 512,
                                    step_length_sec=24 / 512,
                                    realtime_factor=-1,
                                    show_gui=False,
                                    max_joint_impedance_ctrl_torques={('cartpole', 'foot_joint'): 100.0,
                                                                      ('cartpole', 'cartpole_joint'): 100.0},
                                    mjx_impl=mjx_impl,
                                    render_backend="cpu",
                                    reference_filter_mode="none",
                                    log_folder=_log_folder())


def build_genesis_impedance(vec_size: int = 4):
    from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter
    device = th.device("cuda", 0) if has_cuda() else th.device("cpu")
    return GenesisJointImpedanceAdapter(vec_size=vec_size,
                                        output_th_device=device,
                                        sim_step_dt=1 / 256,
                                        step_length_sec=12 / 256,
                                        enable_rendering=False,
                                        reference_filter_mode="none")
