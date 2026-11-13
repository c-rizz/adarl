"""MujocoAdapter (mujoco classic): rgb and depth images as MjxAdapter gives them, and the collision filtering of
build_scenario's default_link_group_collisions."""
import math

import numpy as np
import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import _log_folder

pytestmark = [pytest.mark.mujoco]

# A camera 2 m up looking down (image right is +x, image up is +y), a box under it with its top 1.8 m from the camera,
# and a big box 11 m from the camera on the +x side: the -x side of the image sees nothing.
NEAR_DEPTH, FAR_DEPTH = 1.8, 11.0
SCENE = f"""<mujoco>
  <worldbody>
    <camera name="down" pos="0 0 2" fovy="30" resolution="64 48"/>
    <body name="near_box"><geom type="box" pos="0 0 0" size="0.1 0.1 {2-NEAR_DEPTH}" rgba="1 0 0 1"/></body>
    <body name="far_box"><geom type="box" pos="10 0 {2-FAR_DEPTH-1}" size="9.5 20 1" rgba="0 0 1 1"/></body>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def camera_adapter():
    from adarl.adapters.MujocoAdapter import MujocoAdapter
    adapter = MujocoAdapter(vec_size=1, log_folder=_log_folder())
    adapter._add_ground = False
    adapter._add_sky = False
    adapter.build_scenario([ModelSpawnDef(name="scene", definition_string=SCENE, format="mjcf", pose=None, kwargs={})])
    adapter.startup()
    yield adapter
    adapter.destroy_scenario()


def test_depth_as_mjx(camera_adapter):
    renderings = camera_adapter.get_camera_images(["down"], rgb=True, depth=True)
    rgb, depth = renderings.rgb[0], renderings.depth[0]
    assert rgb.shape == (1, 48, 64, 3) and rgb.dtype == th.uint8
    assert depth.shape == (1, 48, 64, 1) and depth.dtype == th.float32
    assert renderings.times.shape == (1, 1)
    d = depth[0, :, :, 0]
    assert d[24, 32].item() == pytest.approx(NEAR_DEPTH, abs=1e-2)   # the near box, under the camera
    assert d[24, 60].item() == pytest.approx(10.0)                    # the far box, beyond 10 m: clamped
    assert d[24, 3].item() == 0.0                                     # nothing hit
    assert rgb[0, 24, 32, 0] > 2*rgb[0, 24, 32, 2]                    # the red near box
    assert rgb[0, 24, 60, 2] > 2*rgb[0, 24, 60, 0]                    # the blue far box
    # depth only, and through the deprecated getRenderings
    only_depth = camera_adapter.get_camera_images(["down"], rgb=False, depth=True)
    assert only_depth.rgb is None and th.equal(only_depth.depth[0], depth)
    images, times = camera_adapter.getRenderings(["down"], depth=True)
    assert th.equal(images[0], depth) and times.shape == (1, 1)
    images, _ = camera_adapter.getRenderings(["down"])
    assert th.equal(images[0], rgb)


def test_masks_and_out(camera_adapter):
    masked = camera_adapter.get_camera_images(["down"], rgb=True, depth=True, vec_mask=th.tensor([False]))
    assert masked.rgb[0].shape == (0, 48, 64, 3) and masked.depth[0].shape == (0, 48, 64, 1)
    assert masked.times.shape == (0, 1)
    fixed = camera_adapter.get_camera_images(["down"], rgb=True, depth=True, vec_mask=th.tensor([False]),
                                             use_fixed_shapes=True)
    assert fixed.depth[0].shape == (1, 48, 64, 1) and fixed.depth[0].abs().sum() == 0
    out_rgb, out_depth = [th.empty((1, 48, 64, 3), dtype=th.uint8)], [th.empty((1, 48, 64, 1), dtype=th.float32)]
    r = camera_adapter.get_camera_images(["down"], rgb=True, depth=True, out_rgb=out_rgb, out_depth=out_depth)
    assert r.rgb[0] is out_rgb[0] and r.depth[0] is out_depth[0]
    assert out_depth[0][0, 24, 32, 0].item() == pytest.approx(NEAR_DEPTH, abs=1e-2)
    with pytest.raises(AttributeError):
        camera_adapter.get_camera_images(["down"], rgb=False, depth=False)


BALL = """<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 0.5">
      <inertial pos="0 0 0" mass="1" diaginertia="0.004 0.004 0.004"/>
      <freejoint name="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>"""


@pytest.mark.parametrize("ball_hits_ground", [True, False])
def test_build_scenario_collisions(ball_hits_ground):
    from adarl.adapters.MujocoAdapter import MujocoAdapter
    adapter = MujocoAdapter(vec_size=1, log_folder=_log_folder(), step_length_sec=0.5)
    collisions = [(("ball", "ball"), [("ground", "ground_link")] if ball_hits_ground else []),
                  ("ground", [("ball", "ball")] if ball_hits_ground else [])]
    adapter.build_scenario([ModelSpawnDef(name="ball", definition_string=BALL, format="mjcf", pose=None, kwargs={})],
                           default_link_group_collisions=collisions, add_ground=True)
    try:
        adapter.startup()
        adapter.set_monitored_links([("ball", "ball")])
        for _ in range(4):
            adapter.step()
        z = adapter.getLinksState()[0, 0, 2].item()
        if ball_hits_ground:
            assert z == pytest.approx(0.1, abs=0.02) # resting on the ground
        else:
            assert z < -0.5                          # fell through it
    finally:
        adapter.destroy_scenario()
