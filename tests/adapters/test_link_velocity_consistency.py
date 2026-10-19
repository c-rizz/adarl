"""Does the reported link velocity actually describe the link's motion?

Nothing in the suite used to check this, and two separate bugs lived in the gap:

  - the reported linear velocity was mjx's com-based `cvel`, i.e. the velocity of the point
    coinciding with the whole robot's center of mass rather than of the link frame origin. That is
    metres per second of error for a link away from the com, and reverting the fix for it (c67d8ea)
    left every single test green.
  - `vec_body_abs_linvel_xyz` in the locomotion env came back holding the body *position*, because
    the step statistics were read through a torch view of a jax buffer jax was free to recycle.

Comparing against a finite difference of the link's own position catches both, and anything else
that puts the wrong quantity in the velocity slots: a position is nowhere near the derivative of a
position, and a velocity measured about the wrong point is off by omega x r.

The comparison uses the step statistics' *average* velocity rather than the end-of-step velocity,
because that is the like-for-like quantity: both the average and the finite difference cover exactly
the same window, so the check can be tight. The end-of-step velocity is tied to the same statistics
by the bracketing test below.
"""
import pytest
import torch as th

from _backends import build_mujoco_impedance, build_mjx_impedance, requires_jax_gpu, requires_mjx_warp
from adapter_compliance import BAR_LINK, BASE_LINK, CART_JOINT, FOOT_JOINT

# A swinging pole: the bar link sits away from the robot com and moves fast, which is the geometry
# that makes a com-based velocity visibly wrong. Nothing here depends on the backend.
SWING_START_POS = 0.6
SWING_STEPS = 12
# the step statistics are indexed by position among the monitored links, not by link id
MON_LINKS = [BAR_LINK, BASE_LINK]
BAR_STATS_IDX = MON_LINKS.index(BAR_LINK)


def _start(adapter):
    """Bring the adapter up with the cartpole scenario, monitoring the links the checks read."""
    from adapter_compliance import _spawn_defs
    adapter.build_scenario(_spawn_defs())
    adapter.set_monitored_joints([CART_JOINT, FOOT_JOINT])
    adapter.set_monitored_links(MON_LINKS)
    adapter.set_impedance_controlled_joints([CART_JOINT])
    adapter.startup()


def _release_pole(adapter):
    state = th.zeros((adapter.vec_size(), 1, 3), dtype=th.float32)
    state[:, 0, 0] = SWING_START_POS
    adapter.setJointsStateDirect([CART_JOINT], state)


def _check_avg_velocity_matches_motion(adapter):
    """The step's average velocity must equal the step's displacement over its duration."""
    if not hasattr(adapter, "get_links_state_step_stats"):
        pytest.skip("adapter does not expose link step statistics")
    _release_pole(adapter)
    link = BAR_STATS_IDX
    prev_pos = adapter.getLinksState(requestedLinks=[BAR_LINK])[:, 0, 0:3].clone()
    checked = 0
    for i in range(SWING_STEPS):
        t0 = th.as_tensor(adapter.getEnvTimeFromStartup()).reshape(-1)[0].item()
        adapter.step()
        t1 = th.as_tensor(adapter.getEnvTimeFromStartup()).reshape(-1)[0].item()
        pos = adapter.getLinksState(requestedLinks=[BAR_LINK])[:, 0, 0:3].clone()
        avg_vel = adapter.get_links_state_step_stats()[:, 2, link, 0:3]
        dt = t1 - t0
        assert dt > 0, f"step {i}: the clock did not advance (dt={dt})"
        fd = (pos - prev_pos) / dt
        prev_pos = pos
        speed = fd.norm(dim=-1)
        if speed.max().item() < 0.05:
            continue                       # too slow this step to tell anything apart
        checked += 1
        err = (avg_vel - fd).norm(dim=-1)
        # Both sides average over the same substeps, so the only gap is the integrator's own
        # discretisation. Wide enough not to be brittle, far tighter than a wrong reference point.
        tol = 0.15 * speed + 0.02
        assert th.all(err <= tol), (
            f"step {i}: the step's average link velocity {avg_vel.tolist()} does not match the link's "
            f"own displacement: it moved {(pos-prev_pos).tolist()} in {dt}s, i.e. {fd.tolist()} "
            f"(error {err.tolist()} > tolerance {tol.tolist()})")
    assert checked > 0, "the pole never moved fast enough to check the velocity: the setup is broken"


def _check_step_stats_bracket_the_state(adapter):
    """The step's min/avg/max velocity must contain the velocity reported at the end of that step.

    The two come from different buffers, so either can be wrong on its own: this is the path that
    feeds the body absolute velocity in the locomotion env.
    """
    if not hasattr(adapter, "get_links_state_step_stats"):
        pytest.skip("adapter does not expose link step statistics")
    _release_pole(adapter)
    link = BAR_STATS_IDX
    for i in range(SWING_STEPS):
        adapter.step()
        end_vel = adapter.getLinksState(requestedLinks=[BAR_LINK])[:, 0, 7:10]
        stats = adapter.get_links_state_step_stats()   # (V, [min,max,avg,std], L, 6)
        vmin, vmax, vavg = stats[:, 0, link, 0:3], stats[:, 1, link, 0:3], stats[:, 2, link, 0:3]
        slack = 1e-3 + 0.05 * (vmax - vmin).abs()
        assert th.all(vmin - slack <= end_vel) and th.all(end_vel <= vmax + slack), (
            f"step {i}: end-of-step velocity {end_vel.tolist()} is outside the step's own "
            f"[min,max] = [{vmin.tolist()}, {vmax.tolist()}]: the two describe different quantities")
        assert th.all(vmin - slack <= vavg) and th.all(vavg <= vmax + slack), (
            f"step {i}: average {vavg.tolist()} outside [min,max] = [{vmin.tolist()}, {vmax.tolist()}]")


def _check_sim_elements_state_agrees_with_links_state(adapter):
    """The two readers of the link state must return the same thing for the same link.

    getLinksState() and get_sim_elements_state() read the same monitored buffer but index and reorder
    it differently, and the envs do not use the same one: RobotVecEnv's MJX fast path goes through
    get_sim_elements_state (and its [:,7:10] slice is what becomes body_abs_linvel), while everything
    else uses getLinksState. A test of one says nothing about the other, which is how a wrong
    body_abs_linvel coexisted with passing adapter tests.
    """
    if not hasattr(adapter, "get_sim_elements_state"):
        pytest.skip("adapter does not expose get_sim_elements_state")
    _release_pole(adapter)
    link_ids = adapter.get_monitored_links_ids([BAR_LINK])
    for i in range(SWING_STEPS):
        adapter.step()
        from_getter = adapter.getLinksState(requestedLinks=[BAR_LINK])[:, 0, :]
        from_elements = adapter.get_sim_elements_state(link_ids=link_ids).link_state[:, 0, :]
        assert th.allclose(from_getter, from_elements, atol=1e-6), (
            f"step {i}: getLinksState says {from_getter.tolist()} but get_sim_elements_state says "
            f"{from_elements.tolist()} for the same link. The envs read the second one.")


@pytest.mark.mujoco
def test_sim_elements_state_agrees_with_links_state_mujoco():
    adapter = build_mujoco_impedance(vec_size=1)
    _start(adapter)
    try:
        _check_sim_elements_state_agrees_with_links_state(adapter)
    finally:
        adapter.destroy_scenario()


@pytest.mark.gpu
@pytest.mark.mjx
@requires_jax_gpu
@requires_mjx_warp
@pytest.mark.parametrize("mjx_impl", ["warp"])
def test_sim_elements_state_agrees_with_links_state_mjx(mjx_impl):
    adapter = build_mjx_impedance(vec_size=4, mjx_impl=mjx_impl)
    _start(adapter)
    try:
        _check_sim_elements_state_agrees_with_links_state(adapter)
    finally:
        adapter.destroy_scenario()


@pytest.mark.mujoco
def test_link_velocity_matches_finite_difference_mujoco():
    adapter = build_mujoco_impedance(vec_size=1)
    _start(adapter)
    try:
        _check_avg_velocity_matches_motion(adapter)
    finally:
        adapter.destroy_scenario()


@pytest.mark.mujoco
def test_link_step_stats_bracket_the_link_state_mujoco():
    adapter = build_mujoco_impedance(vec_size=1)
    _start(adapter)
    try:
        _check_step_stats_bracket_the_state(adapter)
    finally:
        adapter.destroy_scenario()


@pytest.mark.gpu
@pytest.mark.mjx
@requires_jax_gpu
@requires_mjx_warp
@pytest.mark.parametrize("mjx_impl", ["warp"])
def test_link_velocity_matches_finite_difference_mjx(mjx_impl):
    adapter = build_mjx_impedance(vec_size=4, mjx_impl=mjx_impl)
    _start(adapter)
    try:
        _check_avg_velocity_matches_motion(adapter)
    finally:
        adapter.destroy_scenario()


@pytest.mark.gpu
@pytest.mark.mjx
@requires_jax_gpu
@requires_mjx_warp
@pytest.mark.parametrize("mjx_impl", ["warp"])
def test_link_step_stats_bracket_the_link_state_mjx(mjx_impl):
    adapter = build_mjx_impedance(vec_size=4, mjx_impl=mjx_impl)
    _start(adapter)
    try:
        _check_step_stats_bracket_the_state(adapter)
    finally:
        adapter.destroy_scenario()
