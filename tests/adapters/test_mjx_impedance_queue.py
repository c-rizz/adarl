"""Tests for the MjxJointImpedanceAdapter command queue.

The impedance command queue (``cmds_queue`` / ``cmds_queue_times``, size
``impedance_commands_queue_size``) holds queued commands until they activate; a step consumes/cleans
them. If it fills, ``_insert_cmd_to_queue`` returns ``found_slot=False`` and the dbg_check in
``_add_impedance_command`` fires. These tests characterise the queue:

* under normal per-step commanding it must stay bounded (regression: no leak),
* a delayed command activates only after its delay,
* piling up more distinct future commands than the queue holds (without stepping) fills it and drops
  the excess (the condition that trips the assert).
"""
import numpy as np
import pytest
import torch as th

from adapter_compliance import _spawn_defs, CART_JOINT
from _backends import requires_jax_gpu, requires_mjx_warp, _log_folder

pytestmark = [pytest.mark.mjx, pytest.mark.gpu, pytest.mark.slow]

BAR_LINK = ("cartpole", "bar_link")


def _build(mjx_impl: str, vec_size: int = 2, queue_size: int = 10):
    import jax
    from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
    ad = MjxJointImpedanceAdapter(
        vec_size=vec_size, enable_rendering=False, jax_device=jax.devices("gpu")[0],
        output_th_device=th.device("cuda", 0), sim_step_dt=1 / 512, step_length_sec=24 / 512,
        realtime_factor=-1, show_gui=False,
        max_joint_impedance_ctrl_torques={('cartpole', 'foot_joint'): 100.0, ('cartpole', 'cartpole_joint'): 100.0},
        mjx_impl=mjx_impl, render_backend="cpu", reference_filter_mode="none",
        impedance_commands_queue_size=queue_size, log_folder=_log_folder())
    ad.build_scenario(_spawn_defs())
    ad.set_monitored_joints([CART_JOINT])
    ad.set_monitored_links([BAR_LINK])
    ad.set_impedance_controlled_joints([CART_JOINT])
    ad.startup()
    return ad


def _cmd(ad, p=0.0, v=0.0, e=0.0, kp=0.0, kd=0.0):
    dev = ad.output_th_device()
    return th.as_tensor([p, v, e, kp, kd], device=dev).view(1, 1, 5).expand(ad.vec_size(), 1, 5)


def _occupancy(ad) -> np.ndarray:
    """Number of occupied (finite-time) queue slots, per env."""
    times = np.asarray(ad._sim_state.cmds_queue_times)  # (vec, queue_size)
    return np.isfinite(times).sum(axis=1)


@requires_jax_gpu
@pytest.mark.parametrize("mjx_impl", [
    pytest.param("jax", marks=requires_jax_gpu),
    pytest.param("warp", marks=requires_mjx_warp),
])
def test_queue_stays_bounded_under_normal_stepping(mjx_impl):
    """One command per control step + a step each iteration must NOT accumulate (the regression)."""
    ad = _build(mjx_impl)
    try:
        max_occ = 0
        for _ in range(40):
            ad.set_current_joint_impedance_command(_cmd(ad, kp=100.0, kd=5.0))
            ad.step()
            max_occ = max(max_occ, int(_occupancy(ad).max()))
        assert max_occ <= 3, f"queue grew to {max_occ}/10 under normal per-step commanding (cleanup leak?)"
    finally:
        ad.destroy_scenario()


@requires_jax_gpu
def test_delayed_command_activates_only_after_its_delay():
    ad = _build("jax")
    try:
        step_len = float(ad.step())
        ad.reset_joint_impedances_commands()
        ad.setJointsImpedanceCommand(_cmd(ad, kp=200.0, kd=10.0), delay_sec=2.5 * step_len)
        ad.step()
        kp_early = float(ad.get_current_joint_impedance_command()[0, 0, 3])
        assert abs(kp_early) < 1e-3, f"delayed command active too early: kp={kp_early}"
        for _ in range(3):
            ad.step()
        kp_late = float(ad.get_current_joint_impedance_command()[0, 0, 3])
        assert abs(kp_late - 200.0) < 1e-3, f"delayed command not active after its delay: kp={kp_late}"
    finally:
        ad.destroy_scenario()


@requires_jax_gpu
def test_stepping_drains_the_queue():
    """A queue filled with several soon-to-activate commands drains to a single one after stepping
    (cleanup keeps only the current command). Guards that stepping frees slots."""
    ad = _build("jax", queue_size=6)
    try:
        ad.reset_joint_impedances_commands()
        # several distinct future commands, all due within a few ms (<< step_length 24/512 s)
        for i in range(4):
            ad.setJointsImpedanceCommand(_cmd(ad, kp=float(10 * (i + 1))), delay_sec=0.001 * (i + 1))
        assert int(_occupancy(ad).max()) >= 3, f"expected the queue to fill up, got {int(_occupancy(ad).max())}"
        for _ in range(3):
            ad.step()
        # all activated and cleaned up; only the current command remains
        assert int(_occupancy(ad).max()) <= 1, f"queue did not drain after stepping: {int(_occupancy(ad).max())}"
    finally:
        ad.destroy_scenario()


@requires_jax_gpu
def test_queue_fills_and_drops_when_flooded_without_stepping():
    """More distinct future commands than the queue holds (no step to drain) -> fills, drops excess.

    This is exactly the condition that makes _insert_cmd_to_queue return found_slot=False (the
    line-486 assert). Under normal stepping this should never happen (see the bounded test)."""
    qsize = 4
    ad = _build("jax", queue_size=qsize)
    try:
        ad.reset_joint_impedances_commands()
        for i in range(qsize):  # qsize distinct future commands, no stepping
            ad.setJointsImpedanceCommand(_cmd(ad, kp=float(10 * (i + 1))), delay_sec=float(i + 1))
        assert int(_occupancy(ad).max()) == qsize, f"queue not full: {int(_occupancy(ad).max())}/{qsize}"
        # one more distinct future command cannot be inserted -> dropped, occupancy stays full
        ad.setJointsImpedanceCommand(_cmd(ad, kp=999.0), delay_sec=float(qsize + 1))
        assert int(_occupancy(ad).max()) == qsize, "overflowing command should have been dropped"
    finally:
        ad.destroy_scenario()
