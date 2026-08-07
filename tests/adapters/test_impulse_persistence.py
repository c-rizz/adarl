"""End-to-end regression for per-env impulse (xfrc) application in MjxAdapter.

Reproduces the exact symptom that surfaced in loco_kyon_ppo env-0 videos: an impulse applied to
one env vanished after a step because the env re-calls set_link_impulses every step with a
(mostly) empty vec_mask, and the buggy masked write clobbered env 0. With vec_size>1 the mask is
sparse, which is the failure condition.
"""
import numpy as np
import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import requires_jax_gpu, _log_folder

BALL_MJCF = """
<mujoco><worldbody>
  <body name="ball_link" pos="0 0 1.0">
    <freejoint/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <geom type="sphere" size="0.1"/>
  </body>
</worldbody></mujoco>"""
BALL = ("ball", "ball_link")


@pytest.mark.mjx
@pytest.mark.gpu
@pytest.mark.slow
@requires_jax_gpu
def test_impulse_persists_and_is_env_isolated():
    import jax
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import MjxAdapter

    vec_size = 2
    ad = MjxAdapter(vec_size=vec_size, enable_rendering=False,
                    jax_device=jax.devices("gpu")[0], output_th_device=th.device("cuda", 0),
                    sim_step_dt=1 / 256, step_length_sec=1 / 64, realtime_factor=-1, show_gui=False,
                    add_ground=False, add_sky=False, opt_preset="fast", mjx_impl="jax",
                    render_backend="cpu", log_folder=_log_folder())
    try:
        ad.build_scenario([ModelSpawnDef(name="ball", definition_string=BALL_MJCF, format="mjcf", pose=None, kwargs={})])
        ad.set_monitored_links([BALL])
        ad.startup()

        ball_bid = ad._lname2lid[BALL]
        link_ids = jnp.array([ball_bid])
        dev = th.device("cuda", 0)

        # Sub-second impulse (0.5 s) on env 0 only (sparse mask). Sub-second duration also guards the
        # start/end-time dtype: if the times were stored as int (truncated to whole seconds), this
        # window would collapse to [0, 0] and never fire. env 1 is given a different force but is
        # masked out, so it must never receive it.
        ft = th.zeros((vec_size, 1, 6), device=dev); ft[0, 0, 0] = 10.0; ft[1, 0, 0] = 99.0
        dur = th.full((vec_size, 1), 0.5, device=dev)
        delay = th.zeros((vec_size, 1), device=dev)
        ad.set_link_impulses(link_ids, ft, dur, delay, th.tensor([True, False], device=dev))

        empty_ft = th.zeros((vec_size, 1, 6), device=dev)
        empty_dur = th.zeros((vec_size, 1), device=dev)
        no_env = th.tensor([False, False], device=dev)
        for step in range(5):
            # Mimic the env: set_link_impulses is called every step, almost always with an empty
            # mask. This must NOT wipe env 0's still-active impulse (the historical bug did).
            ad.set_link_impulses(link_ids, empty_ft, empty_dur, delay, no_env)
            ad.step()
            xfrc_x = np.asarray(ad._sim_state.mjx_data.xfrc_applied[:, ball_bid, 0])
            assert abs(xfrc_x[0] - 10.0) < 1e-3, f"step {step}: env-0 impulse vanished/wrong: {xfrc_x[0]}"
            assert abs(xfrc_x[1]) < 1e-6, f"step {step}: masked-out env-1 received a force: {xfrc_x[1]}"
    finally:
        ad.destroy_scenario()
