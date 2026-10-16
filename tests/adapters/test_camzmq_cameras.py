"""CamZmqCameras: camera images for real-robot adapters, received with camzmq.

Driven by a real camzmq publisher with its synthetic source, so no camera or robot is needed. Checks that the images
come out as get_camera_images() returns them in simulation (rgb (1,H,W,3) uint8, depth (1,H,W,1) float32 with every
invalid measurement at INVALID_DEPTH), that times are in env time, that camera_info() describes the published camera,
and that a stale frame is refused.
"""
import threading
import time

import pytest
import torch as th

camzmq = pytest.importorskip("camzmq")
from camzmq.sources.synthetic import SyntheticSource  # noqa: E402

from adarl.adapters.camzmq_cameras import CamZmqCameras  # noqa: E402
from adarl.utils.depth_sim import INVALID_DEPTH  # noqa: E402

H, W, FOVY = 30, 48, 70.0
T0 = time.monotonic()


def _env_time() -> float:
    return time.monotonic() - T0


@pytest.fixture
def publisher():
    """Publishes camera "head" at H x W; the synthetic depth has a NaN, +inf and -inf at row 0, columns 0-2."""
    started = []

    def start(rgb="rgb"):
        source = SyntheticSource({"head" : (H, W, FOVY)}, fps=200.0, clock=time.monotonic)
        pub = camzmq.CameraPublisher(source, {"head" : camzmq.PublishConfig(endpoint="tcp://127.0.0.1:*", rgb=rgb)},
                                     metadata_endpoint="tcp://127.0.0.1:*")
        thread = threading.Thread(target=pub.run, daemon=True)
        thread.start()
        started.append((pub, thread))
        return pub

    yield start
    for pub, thread in started:
        pub.close()
        thread.join(timeout=2.0)


def _cameras(pub, **kwargs) -> CamZmqCameras:
    return CamZmqCameras(pub.metadata_endpoint, out_th_device=th.device("cpu"), env_time=_env_time,
                         camera_names={"head_camera" : "head"}, **kwargs)


def test_images_match_the_simulation_contract(publisher):
    cams = _cameras(publisher())
    try:
        r = cams.get_camera_images(["head_camera"], rgb=True, depth=True)
        rgb, depth = r.rgb[0], r.depth[0]
        assert rgb.shape == (1, H, W, 3) and rgb.dtype == th.uint8
        assert depth.shape == (1, H, W, 1) and depth.dtype == th.float32
        assert th.isfinite(depth).all()
        assert (depth[0, 0, :3, 0] == INVALID_DEPTH).all()   # NaN, +inf and -inf all mapped to INVALID_DEPTH
        assert (depth[0, 1:] > 0).all()
        age = _env_time() - r.times[0, 0].item()
        assert r.times.shape == (1, 1) and 0 <= age < 0.1, age
    finally:
        cams.close()


def test_gray_frames_become_three_equal_channels(publisher):
    cams = _cameras(publisher(rgb="gray"))
    try:
        rgb = cams.get_camera_images(["head_camera"], rgb=True).rgb[0]
        assert rgb.shape == (1, H, W, 3)
        assert th.equal(rgb[..., 0], rgb[..., 1]) and th.equal(rgb[..., 0], rgb[..., 2])
    finally:
        cams.close()


def test_vec_mask_and_out_tensors(publisher):
    cams = _cameras(publisher())
    try:
        masked = cams.get_camera_images(["head_camera"], rgb=True, depth=True, vec_mask=th.tensor([False]))
        assert masked.rgb[0].shape[0] == 0 and masked.times.shape == (0, 1)
        fixed = cams.get_camera_images(["head_camera"], depth=True, rgb=False, vec_mask=th.tensor([False]),
                                       use_fixed_shapes=True)
        assert fixed.depth[0].shape == (1, H, W, 1) and (fixed.depth[0] == 0).all()
        out = [th.full((1, H, W, 1), -1.0)]
        r = cams.get_camera_images(["head_camera"], rgb=False, depth=True, out_depth=out)
        assert r.depth[0] is out[0] and (out[0] >= 0).all()
    finally:
        cams.close()


def test_camera_info(publisher):
    """The published camera, by environment name, for callers that check it themselves."""
    cams = _cameras(publisher())
    try:
        info = cams.camera_info("head_camera")
        assert info.name == "head"
        assert (info.height, info.width, info.rgb_channels, info.has_depth) == (H, W, 3, True)
        assert info.fovy_deg == pytest.approx(FOVY, abs=0.01)
        with pytest.raises(KeyError, match="not published"):
            cams.camera_info("tail_camera")
    finally:
        cams.close()


def test_stale_frame_is_refused(publisher):
    pub = publisher()
    cams = _cameras(pub, max_frame_age_sec=0.1)
    try:
        cams.get_camera_images(["head_camera"], rgb=True)
        pub.close() # the publisher stops: the last frame received ages
        time.sleep(0.3)
        with pytest.raises(RuntimeError, match="old"):
            cams.get_camera_images(["head_camera"], rgb=True)
    finally:
        cams.close()


def test_unknown_camera(publisher):
    cams = _cameras(publisher())
    try:
        with pytest.raises(KeyError, match="not published"):
            cams.get_camera_images(["tail_camera"], rgb=True)
    finally:
        cams.close()
