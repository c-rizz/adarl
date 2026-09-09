"""Rendering contract tests for MjxAdapter with the WARP render backend (RGB + depth).

Rendering was entirely untested: every other adapter test runs with `enable_rendering=False`
and `render_backend="cpu"`. That gap let two shape/semantics bugs through, both guarded here:

* depth came back **rank 5** `(V,H,W,1,1)`, because `mjx.get_depth` already returns the
  trailing channel dim and `_render_depth_warp` added a second one;
* depth was read from the **wrong buffer** -- `mjx.render` returns `(rgb_data, depth_data)`
  and `_render_warp_depth_jax` passed the RGB buffer to `get_depth`, so every pixel came back
  at the far value regardless of the scene.

The contract pinned here is deliberately backend-specific rather than uniform, because the
warp backend's float output is documented, intended behaviour:

    RGB   -> (V, H, W, 3) float32 in [0,1]      (the CPU backend returns uint8 in [0,255])
    depth -> (V, H, W, 1) float32 metres, saturating at max_depth; 0.0 means "no geometry hit"

The scene is a red box of half-size 0.3 at the origin, viewed head-on from y=-2, so the
nearest visible surface sits at exactly 1.7 m -- which makes depth content checkable, not
just its shape.
"""
import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import requires_mjx_warp, _log_folder


CAM_W, CAM_H = 32, 24
VEC_SIZE = 3
BOX_HALF_SIZE = 0.3
CAM_DIST = 2.0
NEAR_FACE_DEPTH = CAM_DIST - BOX_HALF_SIZE  # 1.7 m from the camera to the box's near face
MAX_DEPTH = 10.0                            # _vec_get_depth's max_depth: depths saturate here
NO_HIT_DEPTH = 0.0                          # pixels that hit no geometry come back as 0.0

SCENE_MJCF = f"""
<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <camera name="cam0" pos="0 {-CAM_DIST} 0.5" xyaxes="1 0 0 0 0 1" resolution="{CAM_W} {CAM_H}"/>
    <body name="box_link" pos="0 0 0.5">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="{BOX_HALF_SIZE} {BOX_HALF_SIZE} {BOX_HALF_SIZE}" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def warp_renderer():
    """One warp adapter for the whole module: building it compiles warp kernels (~10s)."""
    import jax
    from adarl.adapters.MjxAdapter import MjxAdapter
    adapter = MjxAdapter(vec_size=VEC_SIZE,
                         enable_rendering=True,
                         jax_device=jax.devices("gpu")[0],
                         output_th_device=th.device("cuda", 0),
                         sim_step_dt=1 / 256,
                         step_length_sec=1 / 64,
                         realtime_factor=-1,
                         show_gui=False,
                         add_ground=False,
                         add_sky=False,
                         opt_preset="fast",
                         mjx_impl="warp",
                         render_backend="warp",
                         log_folder=_log_folder())
    try:
        adapter.build_scenario([ModelSpawnDef(name="scene", definition_string=SCENE_MJCF,
                                              format="mjcf", pose=None, kwargs={})])
        cameras = adapter.get_detected_cameras()
        adapter.set_monitored_cameras(cameras)
        adapter.startup()
        yield adapter, cameras
    finally:
        adapter.destroy_scenario()


pytestmark = [pytest.mark.mjx, pytest.mark.gpu, pytest.mark.slow, requires_mjx_warp]


# ------------------------------------------------------------------ shape / dtype contract


def test_rgb_shape_and_dtype(warp_renderer):
    adapter, cameras = warp_renderer
    imgs, _ = adapter.getRenderings(cameras, depth=False)
    assert len(imgs) == len(cameras)
    rgb = imgs[0]
    assert rgb.shape == (VEC_SIZE, CAM_H, CAM_W, 3), "RGB must be (V,H,W,3) with H,W from cam_resolution"
    assert rgb.dtype == th.float32                       # warp backend: documented float output
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0, "warp RGB is normalized to [0,1]"


def test_depth_shape_and_dtype(warp_renderer):
    """Regression: depth used to be rank 5, `mjx.get_depth` already supplies the channel dim."""
    adapter, cameras = warp_renderer
    imgs, _ = adapter.getRenderings(cameras, depth=True)
    assert len(imgs) == len(cameras)
    depth = imgs[0]
    assert depth.dim() == 4, f"depth must be rank 4, got rank {depth.dim()} (shape {tuple(depth.shape)})"
    assert depth.shape == (VEC_SIZE, CAM_H, CAM_W, 1)
    assert depth.dtype == th.float32
    assert depth.min() >= 0.0 and depth.max() <= MAX_DEPTH


def test_times_shape(warp_renderer):
    adapter, cameras = warp_renderer
    for depth in (False, True):
        _, times = adapter.getRenderings(cameras, depth=depth)
        assert times.shape == (VEC_SIZE, len(cameras))


# ------------------------------------------------------------------------ content contract


def test_rgb_sees_the_red_box(warp_renderer):
    """A guard that the camera actually images the scene, not just that shapes are right."""
    adapter, cameras = warp_renderer
    rgb, _ = adapter.getRenderings(cameras, depth=False)
    img = rgb[0]
    reddish = (img[..., 0] > img[..., 2] + 0.05).sum().item()
    assert reddish > 0, "camera sees no red pixels: the box is not being rendered"


def test_depth_measures_the_box_distance(warp_renderer):
    """Regression: depth was read from the RGB buffer (`mjx.render` returns (rgb, depth)),
    so every pixel came back at the far value no matter what the camera was pointed at.

    The image is exactly two populations: background (no hit -> 0.0) and the box's flat near
    face, which is head-on so all of its pixels share one known distance.
    """
    adapter, cameras = warp_renderer
    depths, _ = adapter.getRenderings(cameras, depth=True)
    depth = depths[0]
    hits = depth > NO_HIT_DEPTH
    assert hits.any(), (f"no pixel hit any geometry: depth is not reading the scene "
                        f"(min={depth.min().item()}, max={depth.max().item()})")
    assert (~hits).any(), "every pixel hit geometry: the box should not fill the whole frame"
    hit_depths = depth[hits]
    assert hit_depths.min().item() == pytest.approx(NEAR_FACE_DEPTH, abs=0.05), (
        f"nearest depth {hit_depths.min().item():.3f} != expected {NEAR_FACE_DEPTH} m")
    assert hit_depths.max().item() == pytest.approx(NEAR_FACE_DEPTH, abs=0.05), (
        "the box's near face is flat and head-on, so all its pixels must share one depth")


def test_depth_is_consistent_across_envs(warp_renderer):
    """All envs hold the same static scene, so their depth images must agree."""
    adapter, cameras = warp_renderer
    depths, _ = adapter.getRenderings(cameras, depth=True)
    depth = depths[0]
    for v in range(1, VEC_SIZE):
        assert th.equal(depth[0], depth[v]), f"env {v} depth differs from env 0"


# --------------------------------------------------------------------------- vec_mask paths


@pytest.mark.parametrize("depth", [False, True], ids=["rgb", "depth"])
def test_vec_mask_subsets_rows(warp_renderer, depth):
    adapter, cameras = warp_renderer
    mask = th.zeros((VEC_SIZE,), dtype=th.bool)
    mask[0] = mask[VEC_SIZE - 1] = True
    imgs, times = adapter.getRenderings(cameras, vec_mask=mask, depth=depth)
    assert imgs[0].shape[0] == 2, "masked render must return only the selected envs"
    assert times.shape == (2, len(cameras))


@pytest.mark.parametrize("depth", [False, True], ids=["rgb", "depth"])
def test_use_fixed_shapes_keeps_vec_size_and_zeroes_masked(warp_renderer, depth):
    adapter, cameras = warp_renderer
    mask = th.zeros((VEC_SIZE,), dtype=th.bool)
    mask[0] = True
    imgs, _ = adapter.getRenderings(cameras, vec_mask=mask, depth=depth, use_fixed_shapes=True)
    img = imgs[0]
    assert img.shape[0] == VEC_SIZE, "use_fixed_shapes must keep the full vec dimension"
    assert th.count_nonzero(img[1:]) == 0, "masked-out envs must be zeroed"
    assert th.count_nonzero(img[0]) > 0, "the selected env must still hold its render"


# --------------------------------------------------------------------------- the `out=` path


@pytest.mark.parametrize("depth", [False, True], ids=["rgb", "depth"])
def test_out_is_written_in_place(warp_renderer, depth):
    adapter, cameras = warp_renderer
    expected, _ = adapter.getRenderings(cameras, depth=depth)
    out = [th.zeros_like(expected[0])]
    imgs, _ = adapter.getRenderings(cameras, depth=depth, out=out)
    assert imgs[0] is out[0], "getRenderings must hand back the provided out tensor"
    assert th.equal(out[0], expected[0])


@pytest.mark.parametrize("depth", [False, True], ids=["rgb", "depth"])
def test_out_shape_mismatch_raises(warp_renderer, depth):
    adapter, cameras = warp_renderer
    bad = [th.zeros((VEC_SIZE, CAM_H + 1, CAM_W, 3 if not depth else 1),
                    dtype=th.float32, device=th.device("cuda", 0))]
    with pytest.raises(RuntimeError, match="shape"):
        adapter.getRenderings(cameras, depth=depth, out=bad)
