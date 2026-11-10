"""Camera timing in MjxAdapter (set_camera_timing): frames captured at a frame rate and returned once their latency
has passed, as from a real camera.

Five cameras with the same pose look down at a bar spinning about z, at a different rate in each env. "live" has no
timing. The scene is advanced one substep at a time and the live image of every substep is kept, so each frame a timed
camera returns can be checked against the live image of the substep its Renderings.times reports.

Two more cameras ride on the bar, one of them timed, and see a mark on the ground turn: they check the frames of a
camera that moves, rendered with the original visuals.
"""
import numpy as np
import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import requires_mjx_warp, _log_folder

pytestmark = [pytest.mark.mjx, pytest.mark.gpu, pytest.mark.slow, requires_mjx_warp]

VEC_SIZE = 3
DT = 1/256
RES = 64
SPIN_RATES = (15.0, 20.0, 25.0) # rad/s about z, one per env
SUBSTEPS = 120
FIXED_PERIOD, FIXED_LATENCY = 4, 3 # substeps
RANDOM_PERIOD, RANDOM_MAX_LATENCY = 2, 10 # substeps
OFFSET_PERIOD = 2 # substeps
OFFSET_RANGE = (4, 16) # substeps, the latency offset an env draws when its frames are reset
OFFSET_JITTER_STD, OFFSET_JITTER_CLAMP = 0.5, 2.0 # substeps and standard deviations, of each frame around the offset
CAMERAS = ("live", "fixed", "in_order", "overtaking", "offset")
TIMED_CAMERAS = CAMERAS[1:]
BAR_CAMERAS = ("bar_live", "bar_timed") # on the bar, the second with the timing of "fixed"

# No gravity (set through opt_override: the aggregated model keeps its own option), so the bar spins in place
SCENE_MJCF = f"""
<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    {"".join(f'<camera name="{c}" pos="0 0 2" xyaxes="1 0 0 0 1 0" resolution="{RES} {RES}"/>' for c in CAMERAS)}
    <geom name="mark" type="box" pos="0.4 0.25 0" size="0.1 0.1 0.02" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    <body name="bar" pos="0 0 0.5">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.1 0.1"/>
      <geom type="box" size="0.6 0.05 0.05" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      {"".join(f'<camera name="{c}" pos="0 0 1.5" xyaxes="1 0 0 0 1 0" resolution="{RES} {RES}"/>' for c in BAR_CAMERAS)}
    </body>
  </worldbody>
</mujoco>"""


def _bar_state(angle : th.Tensor, spin : th.Tensor) -> th.Tensor:
    """Pose and velocity of the bar, shape (envs, 1, 13): at the scene center, turned by angle and spinning about z."""
    state = th.zeros((angle.shape[0], 1, 13))
    state[:, 0, 2] = 0.5
    state[:, 0, 5] = th.sin(angle/2)
    state[:, 0, 6] = th.cos(angle/2)
    state[:, 0, 12] = spin
    return state.to("cuda")


def _render(adapter, cams) -> dict[str, tuple[th.Tensor, th.Tensor]]:
    """Image and capture substep of each env, per camera."""
    r = adapter.get_camera_images([cams[c] for c in CAMERAS], rgb=True)
    return {c : (r.rgb[i].clone(), th.round(r.times[:, i].double()/DT).long().cpu()) for i, c in enumerate(CAMERAS)}


@pytest.fixture(scope="module")
def timed_cameras():
    import jax
    from adarl.adapters.MjxAdapter import MjxAdapter
    adapter = MjxAdapter(vec_size=VEC_SIZE,
                         enable_rendering=True,
                         jax_device=jax.devices("gpu")[0],
                         output_th_device=th.device("cuda", 0),
                         sim_step_dt=DT,
                         step_length_sec=DT,
                         realtime_factor=-1,
                         show_gui=False,
                         add_ground=False,
                         add_sky=False,
                         opt_preset="fast",
                         opt_override={"gravity" : np.zeros(3)},
                         mjx_impl="warp",
                         render_backend="warp",
                         log_folder=_log_folder())
    try:
        adapter.build_scenario([ModelSpawnDef(name="scene", definition_string=SCENE_MJCF,
                                              format="mjcf", pose=None, kwargs={})])
        cams = {c.split("#")[-1] : c for c in adapter.get_detected_cameras()}
        adapter.set_monitored_cameras([cams[c] for c in CAMERAS+BAR_CAMERAS])
        random_latency = ("uniform", (0.0, RANDOM_MAX_LATENCY*DT))
        adapter.set_camera_timing(cams["fixed"], fps=1/(FIXED_PERIOD*DT), latency=FIXED_LATENCY*DT)
        adapter.set_camera_timing(cams["in_order"], fps=1/(RANDOM_PERIOD*DT), latency=random_latency, in_order=True, seed=1)
        adapter.set_camera_timing(cams["overtaking"], fps=1/(RANDOM_PERIOD*DT), latency=random_latency, in_order=False, seed=2)
        adapter.set_camera_timing(cams["offset"], fps=1/(OFFSET_PERIOD*DT),
                                  latency=("normal", (0.0, OFFSET_JITTER_STD*DT, OFFSET_JITTER_CLAMP)),
                                  latency_offset=("uniform", (OFFSET_RANGE[0]*DT, OFFSET_RANGE[1]*DT)),
                                  in_order=False, seed=3)
        adapter.set_camera_timing(cams["bar_timed"], fps=1/(FIXED_PERIOD*DT), latency=FIXED_LATENCY*DT)
        adapter.startup()
        bar = next(link for link in adapter.get_detected_links() if link[1] == "bar")
        adapter.setLinksStateDirect([bar], _bar_state(th.zeros(VEC_SIZE), th.tensor(SPIN_RATES)))
        yield adapter, cams, bar
    finally:
        adapter.destroy_scenario()


@pytest.fixture(scope="module")
def recording(timed_cameras):
    """The renders of every camera at each substep from 0 to SUBSTEPS: recording[n] is at substep n."""
    adapter, cams, _ = timed_cameras
    frames = [_render(adapter, cams)]
    for _ in range(SUBSTEPS):
        adapter.run(DT)
        frames.append(_render(adapter, cams))
    return frames


def test_live_camera_is_current_and_scene_moves(recording):
    for n, frame in enumerate(recording):
        assert th.equal(frame["live"][1], th.full((VEC_SIZE,), n))
    changed = sum(bool((recording[n]["live"][0] != recording[n-1]["live"][0]).flatten(1).any(1).all())
                  for n in range(1, len(recording)))
    assert changed >= 0.9*SUBSTEPS, f"the live image changed in all envs at only {changed} of {SUBSTEPS} substeps"
    red = (recording[-1]["live"][0][..., 0] > recording[-1]["live"][0][..., 2] + 0.05).flatten(1).sum(1)
    assert (red > 0).all(), "the bar is out of view at the end of the recording"


@pytest.mark.parametrize("camera", TIMED_CAMERAS)
def test_frames_are_the_live_image_of_their_capture(recording, camera):
    for n, frame in enumerate(recording):
        images, captures = frame[camera]
        for env in range(VEC_SIZE):
            capture = int(captures[env])
            assert 0 <= capture <= n
            assert th.equal(images[env], recording[capture]["live"][0][env]), \
                f"{camera} at substep {n}, env {env}: the image is not the live one of substep {capture}"


def test_fixed_latency(recording):
    captures = th.stack([frame["fixed"][1] for frame in recording]) # (substeps+1, envs)
    ages = th.arange(len(recording))[:, None] - captures
    settled = slice(FIXED_LATENCY + 2*FIXED_PERIOD, None) # past the frame of the reset and the first capture
    assert ((ages[settled] >= FIXED_LATENCY) & (ages[settled] < FIXED_LATENCY + FIXED_PERIOD)).all()
    for env in range(VEC_SIZE):
        captured = captures[settled, env].unique()
        assert ((captured - captured[0]) % FIXED_PERIOD == 0).all(), f"env {env} captured at {captured.tolist()}"


@pytest.mark.parametrize("camera", ("in_order", "overtaking"))
def test_random_latency_bounds(recording, camera):
    captures = th.stack([frame[camera][1] for frame in recording])
    ages = th.arange(len(recording))[:, None] - captures
    settled = slice(RANDOM_MAX_LATENCY + 2*RANDOM_PERIOD, None)
    assert (ages[settled] >= 0).all() and (ages[settled] <= RANDOM_MAX_LATENCY + RANDOM_PERIOD).all()
    assert (captures[1:] >= captures[:-1]).all(), "a returned frame is older than one returned before it"
    assert captures[settled].unique().numel() > 1


def test_arrival_order(timed_cameras, recording):
    """In order, frames arrive in the order they were captured; overtaking, they sometimes do not."""
    adapter, cams, _ = timed_cameras
    from adarl.adapters.MjxAdapter import _NEVER_SUBSTEP
    overtaken = 0
    for _ in range(60):
        adapter.run(DT)
        for camera in ("in_order", "overtaking"):
            frames = adapter._camera_frames[cams[camera]]
            captures = np.asarray(frames.capture_substep)
            arrivals = np.asarray(frames.available_substep)
            for env in range(VEC_SIZE):
                kept = arrivals[:, env] != _NEVER_SUBSTEP
                in_capture_order = arrivals[kept, env][np.argsort(captures[kept, env])]
                monotonic = bool((np.diff(in_capture_order) >= 0).all())
                if camera == "in_order":
                    assert monotonic, f"env {env}: arrivals {in_capture_order.tolist()} are not in capture order"
                else:
                    overtaken += not monotonic
    assert overtaken > 0, "no frame ever overtook another one"


def test_reset_camera_frames(timed_cameras, recording):
    """Setting a state keeps the frames, reset_camera_frames drops them: the reset state is then a frame that has
    already arrived."""
    adapter, cams, bar = timed_cameras
    reset_env = 1
    mask = th.zeros(VEC_SIZE, dtype=th.bool, device="cuda")
    mask[reset_env] = True
    adapter.setLinksStateDirect([bar], _bar_state(th.full((VEC_SIZE,), 1.0), th.zeros(VEC_SIZE)), vec_mask=mask)
    kept = _render(adapter, cams)
    now = int(kept["live"][1][reset_env])
    assert int(kept["fixed"][1][reset_env]) <= now - FIXED_LATENCY

    adapter.reset_camera_frames(mask)
    reset = _render(adapter, cams)
    for camera in TIMED_CAMERAS:
        assert int(reset[camera][1][reset_env]) == now
        assert th.equal(reset[camera][0][reset_env], reset["live"][0][reset_env])
        for env in (0, 2):
            assert th.equal(reset[camera][1][env], kept[camera][1][env]), "reset_camera_frames changed an env it was not given"

    adapter.run(DT)
    after = _render(adapter, cams)
    assert int(after["fixed"][1][reset_env]) == now, "the first frame after the reset arrived before its latency"


def test_set_camera_timing_after_startup_raises(timed_cameras):
    adapter, cams, _ = timed_cameras
    with pytest.raises(RuntimeError, match="before startup"):
        adapter.set_camera_timing(cams["live"], fps=30.0)


def test_latency_offset(timed_cameras, recording):
    """Each env draws a latency offset when its frames are reset, and the latency of its frames stays around it."""
    adapter, cams, _ = timed_cameras
    from adarl.adapters.MjxAdapter import _NEVER_SUBSTEP
    jitter = OFFSET_JITTER_STD*OFFSET_JITTER_CLAMP + 0.5 + 1e-3 # the latency is rounded to whole substeps
    offsets = []
    for _ in range(2):
        adapter.reset_camera_frames()
        reset_substep = int(_render(adapter, cams)["live"][1][0]) # rendering applies the reset
        offset = np.asarray(adapter._camera_frames[cams["offset"]].latency_offset)/DT
        assert ((offset >= OFFSET_RANGE[0]) & (offset <= OFFSET_RANGE[1])).all(), f"offsets {offset.tolist()}"
        offsets.append(offset)
        latencies = [set() for _ in range(VEC_SIZE)]
        for _ in range(40):
            adapter.run(DT)
            frames = adapter._camera_frames[cams["offset"]]
            captures = np.asarray(frames.capture_substep)
            arrivals = np.asarray(frames.available_substep)
            for env in range(VEC_SIZE):
                kept = (arrivals[:, env] != _NEVER_SUBSTEP) & (captures[:, env] > reset_substep) # not the reset's frame
                latencies[env].update((arrivals[kept, env] - captures[kept, env]).tolist())
        for env in range(VEC_SIZE):
            assert len(latencies[env]) > 1, f"env {env}: every frame had the same latency, {latencies[env]}"
            assert all(abs(latency - offset[env]) <= jitter for latency in latencies[env]), \
                f"env {env}: latencies {sorted(latencies[env])} are not around its offset {offset[env]}"
    assert np.unique(offsets[0]).size == VEC_SIZE, "the envs drew the same offset"
    assert (offsets[0] != offsets[1]).all(), "resetting the frames did not draw the offsets again"


def test_original_visuals_of_the_arrived_frame(timed_cameras, recording):
    """get_original_visuals_camera_images renders the current state, and with apply_camera_timing the newest frame
    that has arrived, the one get_camera_images returns, from where the camera's body was at its capture."""
    adapter, cams, bar = timed_cameras
    live, timed = cams["bar_live"], cams["bar_timed"]
    def on_envs(values : list[float], *shape : int) -> th.Tensor:
        return th.tensor(values, device="cuda").expand(VEC_SIZE, *shape)
    adapter.alter_visuals(link_colors=(adapter.get_links_ids([bar]), on_envs([0.0, 0.0, 1.0], 1, 3)),
                          camera_offset=(timed, on_envs([0.2, 0.0, 0.0], 3), on_envs([0.0, 0.0, 0.0, 1.0], 4)))
    originals = {} # substep -> the current state with the original visuals, from the bar
    checked, older = 0, 0
    for _ in range(FIXED_LATENCY + 4*FIXED_PERIOD):
        adapter.run(DT)
        current = adapter.get_original_visuals_camera_images([live, timed], rgb=True)
        now = int(th.round(current.times[0, 0].double()/DT))
        assert th.equal(current.rgb[1], current.rgb[0]), "without apply_camera_timing the timed camera is not current"
        originals[now] = current.rgb[0].clone()
        altered = adapter.get_camera_images([timed], rgb=True)
        arrived = adapter.get_original_visuals_camera_images([timed], rgb=True, apply_camera_timing=True)
        assert th.equal(arrived.times, altered.times), "not the frame get_camera_images returns"
        assert not th.equal(arrived.rgb[0], altered.rgb[0]), "the visuals were not put back to the original ones"
        for env, capture in enumerate(th.round(arrived.times[:, 0].double()/DT).long().tolist()):
            if capture in originals:
                assert th.equal(arrived.rgb[0][env], originals[capture][env]), \
                    f"substep {now}, env {env}: not the original visuals of substep {capture}"
                checked += 1
                older += capture < now and not th.equal(arrived.rgb[0][env], originals[now][env])
    assert checked >= VEC_SIZE*2*FIXED_PERIOD
    assert older > 0, "no frame older than the current state was rendered"
