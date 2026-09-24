"""Per-environment run() durations in MjxAdapter.

run(duration_sec) takes one duration per environment. The batch is simulated for the longest one, and an env
that reached its own duration is frozen for the rest of the call by setting its timestep to 0: its state, clock,
step statistics and controller filters stay as they are, and the next run() resumes it. This decouples the
environments: what an env observes must not depend on how long the others had to run (e.g. an adapter whose step
lasts until every env's movement is over).

The per-env clock is an int32 substep counter (time = substeps*dt), which also fixes the float32 clock that used to
stop advancing once its resolution got coarser than dt.
"""
import pytest
import torch as th

from adapter_compliance import _spawn_defs, CART_JOINT, FOOT_JOINT, BAR_LINK, BALL_LINK
from _backends import requires_jax_gpu, requires_mjx_warp, _log_folder

pytestmark = [pytest.mark.mjx, pytest.mark.gpu, pytest.mark.slow]

VEC = 3
DT = 1 / 512
STEP_SUBSTEPS = 24
POLE_START = 0.3    # rad: the pole starts tilted, so it swings
POLE_TARGET = 0.6   # rad: impedance target far from the start, so the reference filter is mid-transient
A, B = 6, 10        # substeps before and during a freeze; the default 20 Hz reference filter settles in ~4


def _build(mjx_impl: str, kind: str):
    import jax
    common = dict(vec_size=VEC, enable_rendering=False, jax_device=jax.devices("gpu")[0],
                  output_th_device=th.device("cuda", 0), sim_step_dt=DT, step_length_sec=STEP_SUBSTEPS * DT,
                  realtime_factor=-1, show_gui=False, mjx_impl=mjx_impl, render_backend="cpu",
                  log_folder=_log_folder())
    if kind == "impedance":
        from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
        adapter = MjxJointImpedanceAdapter(max_joint_impedance_ctrl_torques={CART_JOINT: 100.0, FOOT_JOINT: 100.0},
                                           reference_filter_mode="second_order", **common)
    else:
        from adarl.adapters.MjxAdapter import MjxAdapter
        adapter = MjxAdapter(**common)
    adapter.build_scenario(_spawn_defs())
    adapter.set_monitored_joints([CART_JOINT, FOOT_JOINT])
    adapter.set_monitored_links([BAR_LINK, BALL_LINK])
    if kind == "impedance":
        adapter.set_impedance_controlled_joints([CART_JOINT])
    adapter.startup()
    return adapter


@pytest.fixture(scope="module", params=[
    pytest.param(("jax", "effort"), marks=requires_jax_gpu, id="jax-effort"),
    pytest.param(("warp", "effort"), marks=requires_mjx_warp, id="warp-effort"),
    pytest.param(("jax", "impedance"), marks=requires_jax_gpu, id="jax-impedance"),
    pytest.param(("warp", "impedance"), marks=requires_mjx_warp, id="warp-impedance"),
])
def adapter(request):
    """One adapter per backend and kind for the whole module (building one compiles it); every test starts from
    _reset_scene()."""
    adapter = _build(*request.param)
    try:
        yield adapter
    finally:
        adapter.destroy_scenario()


def _reset_scene(adapter):
    """Same start for every env: the pole tilted, so it swings, and the ball in free flight, clear of contacts.
    With the impedance adapter, a target far from the start, so the reference filter is mid-transient."""
    adapter.resetWorld()
    dev = adapter.output_th_device()
    pole = th.tensor([POLE_START, 0.0, 0.0], device=dev).view(1, 1, 3).expand(VEC, 1, 3)
    adapter.setJointsStateDirect([CART_JOINT], pole)
    ball = th.tensor([0.0, 1.0, 3.0,  0.0, 0.0, 0.0, 1.0,  0.5, 0.0, 0.0,  0.0, 0.0, 0.0], device=dev)
    adapter.setLinksStateDirect([BALL_LINK], ball.view(1, 1, 13).expand(VEC, 1, 13))
    if hasattr(adapter, "setJointsImpedanceCommand"):
        cmd = th.tensor([POLE_TARGET, 0.0, 0.0, 20.0, 1.0], device=dev).view(1, 1, 5).expand(VEC, 1, 5)
        adapter.setJointsImpedanceCommand(cmd, delay_sec=0.0)


def _durations(*substeps) -> th.Tensor:
    return th.tensor(substeps, dtype=th.float64) * DT  # on CPU, as run() expects


def _state(adapter) -> th.Tensor:
    """What a frozen env must keep, per env: joint positions and velocities, and the link states."""
    joints = adapter.getJointsState([CART_JOINT, FOOT_JOINT])[:, :, :2]
    links = adapter.getLinksState([BAR_LINK, BALL_LINK])
    return th.cat([joints.flatten(1), links.flatten(1)], dim=1)


def test_run_returns_what_each_env_simulated(adapter):
    _reset_scene(adapter)
    t0 = adapter.getEnvTimeFromStartup().clone()
    ran = adapter.run(th.tensor([3.4, 0.0, A + B], dtype=th.float64) * DT)  # rounded down to whole substeps
    expected = th.tensor([3, 0, A + B], dtype=th.float32) * DT
    assert ran.device.type == "cpu" and ran.dtype == th.float32
    th.testing.assert_close(ran, expected, rtol=0, atol=0)
    th.testing.assert_close((adapter.getEnvTimeFromStartup() - t0).cpu(), expected, rtol=0, atol=1e-6)
    # a single duration applies to every env
    th.testing.assert_close(adapter.run(A * DT), th.full((VEC,), A * DT), rtol=0, atol=0)


def test_step_returns_per_env_durations(adapter):
    """step() reports how long each env was simulated for, like run(); MjxAdapter.step() runs every env for the
    same step length."""
    _reset_scene(adapter)
    stepped = adapter.step()
    assert stepped.device.type == "cpu" and stepped.dtype == th.float32
    th.testing.assert_close(stepped, th.full((VEC,), STEP_SUBSTEPS * DT), rtol=0, atol=0)


def test_zero_duration_env_does_not_move(adapter):
    """A frozen env keeps its state exactly, velocities included, and its clock stands still."""
    _reset_scene(adapter)
    adapter.run(_durations(A, A, A))  # get everything moving first
    before, t_before = _state(adapter), adapter.getEnvTimeFromStartup().clone()
    adapter.run(_durations(0, B, B))
    after, t_after = _state(adapter), adapter.getEnvTimeFromStartup()
    th.testing.assert_close(after[0], before[0], rtol=0, atol=0)
    assert t_after[0] == t_before[0]
    assert not th.equal(after[1], before[1]), "the other envs should have moved"


def test_frozen_env_resumes_where_it_stopped(adapter):
    """env0 simulates A substeps, is frozen while the others simulate B more, then simulates its own B. env1 does
    the same A+B substeps in one go: both must end in the same state, at the same time. With the impedance adapter
    this also covers the reference filter, which would have settled during the freeze had it kept running."""
    _reset_scene(adapter)
    t0 = adapter.getEnvTimeFromStartup().clone()
    adapter.initialize_for_step()
    adapter.run(_durations(A, A + B, A + B))
    adapter.run(_durations(B, 0, 0))
    state = _state(adapter)
    # Same substeps from the same state; only the constraint solver's warm start differs after the pause, hence a
    # tolerance rather than bit-exactness between env0 and env1
    th.testing.assert_close(state[0], state[1], rtol=0, atol=1e-5)
    th.testing.assert_close(state[2], state[1], rtol=0, atol=0)
    th.testing.assert_close(adapter.getEnvTimeFromStartup() - t0, th.full((VEC,), (A + B) * DT, device=t0.device),
                            rtol=0, atol=1e-6)


def test_frozen_env_stats_ignore_how_long_the_others_ran(adapter):
    """The step statistics are part of the observation. A frozen env must not keep sampling its frozen state while
    the others run, or its averages and deviations would depend on how long they took."""
    _reset_scene(adapter)
    adapter.initialize_for_step()
    adapter.run(_durations(A, A, A))
    alone = adapter.get_joints_state_step_stats()[0].clone()
    _reset_scene(adapter)
    adapter.initialize_for_step()
    adapter.run(_durations(A, A + B, A + B))
    frozen = adapter.get_joints_state_step_stats()[0].clone()
    th.testing.assert_close(frozen, alone, rtol=0, atol=1e-5)


def test_env_that_did_not_run_reports_its_current_state_as_stats(adapter):
    """An env that simulates no substep in a step gets one sample of its current state, instead of the empty
    (+-inf) statistics, which would make its observation non-finite."""
    _reset_scene(adapter)
    adapter.initialize_for_step()
    adapter.run(_durations(0, A, A))
    stats = adapter.get_joints_state_step_stats().clone()  # (vec, [min,max,avg,std], joints, quantities)
    assert th.isfinite(stats).all()
    th.testing.assert_close(stats[0, 0], stats[0, 1], rtol=0, atol=0)  # a single sample: min == max
    assert (stats[0, 3] == 0).all()                                    # and no deviation


def test_clock_keeps_advancing_past_float32_resolution(adapter):
    """The clock is substeps*dt, not a float32 sum of dt: such a sum stops advancing once its resolution is coarser
    than dt, which at dt=1/512 happens at 32768 s (16384 s at dt=1/1024). Kept last: it moves the clock far ahead."""
    import jax.numpy as jnp
    saved = adapter._sim_state.sim_substeps, adapter._sim_state.sim_time
    far = 2**24  # substeps: 32768 s at dt=1/512, where float32 resolution is 2**-8 s, twice dt
    adapter._sim_state = adapter._sim_state.replace_d({"sim_substeps": jnp.full((VEC,), far, dtype=jnp.int32),
                                                       "sim_time": jnp.full((VEC,), far * DT, dtype=jnp.float32)})
    try:
        t0 = adapter.getEnvTimeFromStartup().clone()
        adapter.step()
        advanced = adapter.getEnvTimeFromStartup() - t0
        th.testing.assert_close(advanced, th.full_like(advanced, STEP_SUBSTEPS * DT), rtol=0, atol=2**-8)
    finally:
        adapter._sim_state = adapter._sim_state.replace_d({"sim_substeps": saved[0], "sim_time": saved[1]})
