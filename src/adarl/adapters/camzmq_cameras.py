"""Camera images for real-robot adapters, received with camzmq.

The frames arrive at the resolution the policy's simulated camera renders at, so the environment processes them with
the same code as in simulation. This module turns them into what get_camera_images() returns in simulation: rgb as
(vec, H, W, 3) uint8, depth as (vec, H, W, 1) float32 meters with every invalid measurement mapped to INVALID_DEPTH,
and times in the adapter's env time.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch as th

from adarl.adapters.BaseVecAdapter import Renderings
from adarl.utils.depth_sim import INVALID_DEPTH

if TYPE_CHECKING:
    import camzmq


class CamZmqCameras:
    """The latest frame of each camera from a camzmq publisher, as Renderings. Supports vec_size 1 only.

    camera_info() tells how each camera is published (resolution, field of view, ...), so that a caller can check it
    matches the simulated camera: a lens mix-up (e.g. a 2.2 mm instead of a 4 mm ZED X Mini) then shows up there rather
    than as a policy that silently misbehaves. At every request the frame age is checked: a stale image looks
    perfectly valid.
    """

    def __init__(self, metadata_url : str, out_th_device : th.device, env_time : Callable[[], float],
                 camera_names : dict[str, str] | None = None,
                 max_frame_age_sec : float = 0.25,
                 first_frame_timeout_sec : float = 5.0,
                 request_timeout_sec : float = 5.0):
        """
        Parameters
        ----------
        metadata_url : str
            The publisher's metadata endpoint, e.g. "tcp://robot-pc:5700".
        out_th_device : th.device
            Default device of the returned tensors.
        env_time : Callable[[], float]
            The adapter's current env time, which must run at real-time speed on this machine's monotonic clock
            (as StandaloneRealAdapter's does with a realtime factor of 1): frame times are env_time() minus the age.
        camera_names : dict[str, str] | None
            Maps the camera names the environment uses to the names they are published with. Unmapped names are used
            as they are.
        max_frame_age_sec : float
            get_camera_images() raises if the latest frame is older than this.
        first_frame_timeout_sec : float
            How long get_camera_images() waits for a camera's first frame.
        """
        try:
            import camzmq
        except ImportError as e:
            raise ImportError("camera images over ZMQ need the camzmq package (pip install -e camzmq)") from e
        self._out_th_device = out_th_device
        self._env_time = env_time
        self._camera_names = dict(camera_names or {})
        self._max_frame_age_sec = max_frame_age_sec
        self._first_frame_timeout_sec = first_frame_timeout_sec
        self._sub = camzmq.CameraSubscriber(metadata_url, cameras=None, clock=time.monotonic,
                                            request_timeout_s=request_timeout_sec)

    def _remote_name(self, camera : str) -> str:
        return self._camera_names.get(camera, camera)

    def camera_info(self, camera : str) -> camzmq.CameraInfo:
        """How a camera (by environment name) is published: resolution, rgb channels, whether it has depth, the
        intrinsics at the published resolution (and fovy_deg/fovx_deg from them), and the sensor's resolution."""
        remote = self._remote_name(camera)
        if remote not in self._sub.cameras:
            raise KeyError(f"camera {camera!r} (published as {remote!r}) is not published, "
                           f"available: {sorted(self._sub.cameras)}")
        return self._sub.cameras[remote]

    def get_camera_images(self, requestedCameras : Sequence[str], rgb : bool = True, depth : bool = False,
                          vec_mask : th.Tensor | None = None, out_th_device : th.device | None = None,
                          out_rgb : list[th.Tensor] | None = None, out_depth : list[th.Tensor] | None = None,
                          use_fixed_shapes : bool = False) -> Renderings:
        """Same contract as BaseVecAdapter.get_camera_images(), for a single environment."""
        if not rgb and not depth:
            raise AttributeError("get_camera_images: at least one of rgb and depth must be requested")
        device = out_th_device if out_th_device is not None else self._out_th_device
        selected = vec_mask is None or bool(vec_mask.reshape(-1)[0])
        rows = 1 if (selected or use_fixed_shapes) else 0
        rgb_imgs : list[th.Tensor] = []
        depth_imgs : list[th.Tensor] = []
        times = th.zeros((rows, len(requestedCameras)), dtype=th.float32)
        env_now = self._env_time()
        for i, camera in enumerate(requestedCameras):
            info = self.camera_info(camera)
            remote = info.name
            if rgb and info.rgb_channels == 0:
                raise RuntimeError(f"rgb requested for camera {camera!r}, which is published without it")
            if depth and not info.has_depth:
                raise RuntimeError(f"depth requested for camera {camera!r}, which is published without it")
            received = self._sub.latest(remote, timeout_s=self._first_frame_timeout_sec)
            if received is None:
                raise TimeoutError(f"no frame from camera {camera!r} within {self._first_frame_timeout_sec}s")
            frame = received.frame
            age = self._sub.age(remote, frame)
            if age > self._max_frame_age_sec:
                raise RuntimeError(f"the latest frame of camera {camera!r} is {age*1000:.0f} ms old "
                                   f"(max_frame_age_sec={self._max_frame_age_sec}): is its publisher still running?")
            times[:, i] = env_now - age
            if rgb:
                img = frame.rgb if frame.rgb.shape[2] == 3 else np.repeat(frame.rgb, 3, axis=2) # gray -> 3 equal channels
                rgb_imgs.append(self._to_output(img, rows, selected, out_rgb[i] if out_rgb is not None else None, device))
            if depth:
                valid = np.isfinite(frame.depth) & (frame.depth > 0)
                d = np.where(valid, frame.depth, INVALID_DEPTH).astype(np.float32)[..., None]
                depth_imgs.append(self._to_output(d, rows, selected, out_depth[i] if out_depth is not None else None, device))
        return Renderings(rgb=rgb_imgs if rgb else None, depth=depth_imgs if depth else None,
                          times=times.to(device))

    @staticmethod
    def _to_output(img : np.ndarray, rows : int, selected : bool, out : th.Tensor | None, device : th.device) -> th.Tensor:
        t = th.tensor(img).unsqueeze(0)[:rows] # copied out of the received message buffer
        if not selected:
            t = th.zeros_like(t) # only reached with use_fixed_shapes: the masked-out env is zeroed
        if out is not None:
            if out.shape != t.shape:
                raise RuntimeError(f"get_camera_images: out tensor has shape {tuple(out.shape)}, expected {tuple(t.shape)}")
            out.copy_(t)
            return out
        return t.to(device)

    def close(self) -> None:
        self._sub.close()
