"""MjxAdapter.get_heightfield in its "warp_camera" mode: a downward orthographic depth camera.

The scene is made of axis-aligned boxes, so the height at any point is known exactly, and every map
is compared cell by cell against that. The sensed area spans 8 m in 0.5 m cells, whose centers never
fall on a box edge at yaw 0.
"""
import math

import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import requires_mjx_warp, _log_folder

VEC_SIZE = 3
SIZE_XY = (8.0, 8.0)
RESOLUTION_XY = (16, 16)
CAMERA_Z = 10.0
MAX_DEPTH = 20.0
CRATE_START_Z = 1.0
CRATE_HALF_SIZE = 0.3

# (center x, center y, half x, half y, top z) of the boxes the sensor should see
FLOOR = (0.0, 0.0, 20.0, 20.0, 0.0)
STEP = (2.0, 0.0, 1.0, 1.0, 0.4)
COLLISION_ONLY = (0.0, 2.0, 0.5, 0.5, 0.2)

SCENE_MJCF = f"""
<mujoco>
  <worldbody>
    <body name="floor_link" pos="0 0 -0.1">
      <geom type="box" size="{FLOOR[2]} {FLOOR[3]} 0.1"/>
    </body>
    <body name="step_link" pos="{STEP[0]} {STEP[1]} 0.2">
      <geom type="box" size="{STEP[2]} {STEP[3]} 0.2"/>
    </body>
    <body name="collision_only_link" pos="{COLLISION_ONLY[0]} {COLLISION_ONLY[1]} 0.1">
      <geom type="box" size="{COLLISION_ONLY[2]} {COLLISION_ONLY[3]} 0.1" group="3"/>
    </body>
    <body name="visual_only_link" pos="0 -2 0.3">
      <geom type="box" size="0.5 0.5 0.3" contype="0" conaffinity="0"/>
    </body>
    <body name="robot_link" pos="0 0 1">
      <geom type="box" size="0.5 0.5 0.1"/>
    </body>
    <body name="crate_link" pos="-2 0 {CRATE_START_Z}">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="{CRATE_HALF_SIZE} {CRATE_HALF_SIZE} {CRATE_HALF_SIZE}"/>
    </body>
  </worldbody>
</mujoco>"""

pytestmark = [pytest.mark.mjx, pytest.mark.gpu, pytest.mark.slow, requires_mjx_warp]


def _build_adapter():
    import jax
    from adarl.adapters.MjxAdapter import MjxAdapter
    return MjxAdapter(vec_size=VEC_SIZE,
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
                      mjx_impl="warp",
                      log_folder=_log_folder())


@pytest.fixture(scope="module")
def sensed():
    """One adapter for the whole module: building it compiles warp kernels. Rendering stays off, the
    sensor has to work in a training setup."""
    adapter = _build_adapter()
    adapter.setup_heightfield_sensor(SIZE_XY, RESOLUTION_XY, [("scene", "robot_link")],
                                     {"mode": "warp_camera", "camera_z": CAMERA_Z, "max_depth": MAX_DEPTH})
    try:
        adapter.build_scenario([ModelSpawnDef(name="scene", definition_string=SCENE_MJCF,
                                              format="mjcf", pose=None, kwargs={})])
        adapter.startup()
        yield adapter
    finally:
        adapter.destroy_scenario()


def _cell_centers(center_xy, yaw):
    """(rx, ry, 2) world xy of the cell centers, indexed as get_heightfield is."""
    cell = th.tensor(SIZE_XY, dtype=th.float64)/th.tensor(RESOLUTION_XY, dtype=th.float64)
    local_x = (th.arange(RESOLUTION_XY[0], dtype=th.float64)+0.5)*cell[0] - SIZE_XY[0]/2
    local_y = (th.arange(RESOLUTION_XY[1], dtype=th.float64)+0.5)*cell[1] - SIZE_XY[1]/2
    x, y = th.meshgrid(local_x, local_y, indexing="ij")
    return th.stack([center_xy[0] + math.cos(yaw)*x - math.sin(yaw)*y,
                     center_xy[1] + math.sin(yaw)*x + math.cos(yaw)*y], dim=-1)


def _expected(center_xy, yaw, crate_top):
    points = _cell_centers(center_xy, yaw)
    heights = th.zeros(points.shape[:2], dtype=th.float64)
    crate = (-2.0, 0.0, CRATE_HALF_SIZE, CRATE_HALF_SIZE, crate_top)
    for cx, cy, hx, hy, top in (FLOOR, STEP, COLLISION_ONLY, crate):
        inside = ((points[...,0]-cx).abs() < hx) & ((points[...,1]-cy).abs() < hy)
        heights = th.where(inside, th.full_like(heights, top), heights)
    return heights.to(th.float32)


def _read(adapter, centers, yaws):
    return adapter.get_heightfield(th.tensor(centers, dtype=th.float32, device="cuda"),
                                   th.tensor(yaws, dtype=th.float32, device="cuda")).cpu()


def test_map_matches_the_scene_for_each_env_center_and_yaw(sensed):
    """Also shows the index convention: x along yaw, y 90 degrees counterclockwise of it."""
    centers = [(0.0, 0.0), (2.0, 0.0), (1.0, 0.5)]
    yaws = [0.0, math.pi/2, 0.3]
    heights = _read(sensed, centers, yaws)
    assert heights.shape == (VEC_SIZE,) + RESOLUTION_XY
    assert heights.dtype == th.float32
    crate_top = CRATE_START_Z + CRATE_HALF_SIZE
    for env in range(VEC_SIZE):
        th.testing.assert_close(heights[env], _expected(centers[env], yaws[env], crate_top), atol=1e-3, rtol=0.0)


def test_ignored_link_is_not_seen_and_the_others_are(sensed):
    heights = _read(sensed, [(0.0, 0.0)]*VEC_SIZE, [0.0]*VEC_SIZE)[0]
    points = _cell_centers((0.0, 0.0), 0.0)
    under_robot = (points[...,0].abs() < 0.5) & (points[...,1].abs() < 0.5)
    assert th.all(heights[under_robot] == 0.0), "the robot link is ignored, the floor should show through it"


def test_collision_geometry_counts_and_visual_geometry_does_not(sensed):
    heights = _read(sensed, [(0.0, 0.0)]*VEC_SIZE, [0.0]*VEC_SIZE)[0]
    points = _cell_centers((0.0, 0.0), 0.0)
    on_collision_only = ((points[...,0]-COLLISION_ONLY[0]).abs() < 0.5) & ((points[...,1]-COLLISION_ONLY[1]).abs() < 0.5)
    on_visual_only = (points[...,0].abs() < 0.5) & ((points[...,1]+2.0).abs() < 0.5)
    th.testing.assert_close(heights[on_collision_only], th.full_like(heights[on_collision_only], COLLISION_ONLY[4]), atol=1e-3, rtol=0.0)
    assert th.all(heights[on_visual_only] == 0.0)


def test_nothing_hit_reads_the_lowest_measurable_height(sensed):
    heights = _read(sensed, [(100.0, 100.0)]*VEC_SIZE, [0.0]*VEC_SIZE)
    assert th.all(heights == CAMERA_Z - MAX_DEPTH)


def test_map_follows_moving_bodies(sensed):
    """Last, since it steps the sim: the crate drops onto the floor."""
    top_before = _read(sensed, [(-2.0, 0.0)]*VEC_SIZE, [0.0]*VEC_SIZE)[:, 7:9, 7:9]
    th.testing.assert_close(top_before, th.full_like(top_before, CRATE_START_Z+CRATE_HALF_SIZE), atol=1e-3, rtol=0.0)
    for _ in range(64):
        sensed.step()
    top_after = _read(sensed, [(-2.0, 0.0)]*VEC_SIZE, [0.0]*VEC_SIZE)[:, 7:9, 7:9]
    th.testing.assert_close(top_after, th.full_like(top_after, 2*CRATE_HALF_SIZE), atol=0.02, rtol=0.0)


@pytest.mark.parametrize("size_xy, resolution_xy, options", [
    (SIZE_XY, RESOLUTION_XY, {"mode": "live_rays"}),
    (SIZE_XY, RESOLUTION_XY, {}),
    (SIZE_XY, RESOLUTION_XY, {"mode": "warp_camera", "bad_option": 1}),
    ((8.0, 4.0), (16, 16), {"mode": "warp_camera"}), # not square cells
])
def test_bad_setups_are_refused(size_xy, resolution_xy, options):
    with pytest.raises(ValueError):
        _build_adapter().setup_heightfield_sensor(size_xy, resolution_xy, [], options)


def test_setup_is_refused_after_build_scenario(sensed):
    with pytest.raises(RuntimeError):
        sensed.setup_heightfield_sensor(SIZE_XY, RESOLUTION_XY, [], {"mode": "warp_camera"})


def test_reading_without_a_setup_is_refused():
    with pytest.raises(RuntimeError):
        _build_adapter().get_heightfield(th.zeros((VEC_SIZE, 2), device="cuda"), th.zeros((VEC_SIZE,), device="cuda"))
