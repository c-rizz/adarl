"""Regression test for moving a FIXED (world-attached) link under the MJX backend.

A fixed link's pose lives in the model (body_pos/body_quat), not in the data. The WARP
backend bakes static geoms' world poses (geom_xpos/geom_xmat) at put_model time and does not
recompute them during forward, so moving a fixed body via setLinksStateDirect used to update
the body xpos but NOT its collision geometry or render BVH -> objects fell through a "moved"
platform under warp while working under jax. MjxAdapter._set_link_poses now recomputes the
moved geoms' world poses; this test guards that fix for both backends.

Setup: drop a free ball onto a fixed platform relocated *under* the ball via
setLinksStateDirect. If collision tracks the moved fixed body, the ball rests on it; if the
platform's collision stayed baked at the origin, the ball falls through.
"""
import numpy as np
import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import requires_jax_gpu, requires_mjx_warp, _log_folder

PLATFORM_MJCF = """
<mujoco><worldbody>
  <body name="platform_link" pos="0 0 0">
    <geom type="box" size="1 1 0.1" mass="1"/>
  </body>
</worldbody></mujoco>"""

BALL_MJCF = """
<mujoco><worldbody>
  <body name="ball_link" pos="0 0 1.0">
    <freejoint/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <geom type="sphere" size="0.1"/>
  </body>
</worldbody></mujoco>"""

PLATFORM = ("platform", "platform_link")
BALL = ("ball", "ball_link")


def _build(mjx_impl: str, vec_size: int):
    import jax
    from adarl.adapters.MjxAdapter import MjxAdapter
    return MjxAdapter(vec_size=vec_size,
                      enable_rendering=False,
                      jax_device=jax.devices("gpu")[0],
                      output_th_device=th.device("cuda", 0),
                      sim_step_dt=1 / 256,
                      step_length_sec=1 / 64,
                      realtime_factor=-1,
                      show_gui=False,
                      add_ground=False,
                      add_sky=False,
                      opt_preset="fast",
                      mjx_impl=mjx_impl,
                      render_backend="cpu",
                      log_folder=_log_folder())


@pytest.mark.mjx
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mjx_impl", [
    pytest.param("jax", marks=requires_jax_gpu),
    pytest.param("warp", marks=requires_mjx_warp),
])
def test_moved_fixed_link_collides(mjx_impl):
    vec_size = 2
    adapter = _build(mjx_impl, vec_size)
    try:
        adapter.build_scenario([
            ModelSpawnDef(name="platform", definition_string=PLATFORM_MJCF, format="mjcf", pose=None, kwargs={}),
            ModelSpawnDef(name="ball", definition_string=BALL_MJCF, format="mjcf", pose=None, kwargs={}),
        ])
        adapter.set_monitored_links([PLATFORM, BALL])

        # move the fixed platform so its top surface is at z=0.5 (center 0.4 + half-thickness 0.1)
        pstate = th.zeros((vec_size, 1, 13))
        pstate[:, 0, 2] = 0.4
        pstate[:, 0, 6] = 1.0
        adapter.setLinksStateDirect([PLATFORM], pstate)
        adapter.startup()

        # kinematic pose of the fixed body must reflect the set body_pos
        plat_z = adapter.getLinksState([PLATFORM])[:, 0, 2].cpu().numpy()
        assert np.allclose(plat_z, 0.4, atol=1e-3), f"platform xpos.z={plat_z.tolist()}"

        for _ in range(120):
            adapter.step()

        ball_z = adapter.getLinksState([BALL])[:, 0, 2].cpu().numpy()
        # rests on the moved platform (top 0.5 + ball radius 0.1 ~ 0.6), NOT fallen through
        assert np.all(ball_z > 0.45), (
            f"ball fell through the moved fixed platform under mjx_impl={mjx_impl}: "
            f"ball_z={ball_z.tolist()} (expected ~0.6)")
    finally:
        adapter.destroy_scenario()
