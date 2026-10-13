#!/usr/bin/env python3
"""Make rendered depth images look like the output of a passive stereo camera (e.g. a ZED).

A renderer produces exact depth for every pixel; a stereo camera computes it from the disparity
between two views, which fails in characteristic ways. The gap that matters for sim-to-real is
mostly *structural* rather than noise:

- Depth error grows quadratically with range: ``sigma_z = z**2 * sigma_d / (f*B)``. At a 0.8 m
  working distance with f=525 px, B=0.12 m and sigma_d=0.2 px that is only ~2 mm, so the Gaussian
  term is nearly irrelevant for table-top manipulation.
- Disparity is quantized (sub-pixel refinement, typically 1/16 px), so depth is quantized coarsely
  at range and finely up close.
- Textureless or specular surfaces give no match at all: large *blobs* of missing depth, not
  per-pixel salt noise.
- Depth discontinuities produce occlusion shadows (a horizontal band, because the baseline is
  horizontal) and foreground fattening (the matching window dilates the nearer surface).
- The sensor has a minimum and maximum range outside which it returns nothing.

Invalid pixels are represented as ``INVALID_DEPTH`` (0.0), a value a real measurement can never
take since the minimum range is strictly positive. Anything consuming these images must keep
treating 0 as "no measurement": in particular use :func:`resize_depth_masked` rather than a plain
interpolation, or the holes will bleed into their neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch as th
import torch.nn.functional as F

INVALID_DEPTH = 0.0
"""Value marking a pixel with no depth measurement. Zero is unambiguous: a real stereo measurement
is always beyond the sensor's (strictly positive) minimum range."""


@dataclass
class StereoDepthSimParams:
    """Parameters of the simulated stereo camera.

    ``focal_px`` must be expressed at the resolution of the depth image being degraded (the render
    resolution, not the resolution the observation is later resized to), since the disparity noise
    and quantization are defined in pixels. Take ``focal_px`` and ``baseline_m`` from the real
    camera's calibration and rescale the focal length by
    ``render_width/calibration_width``.
    """
    focal_px : float
    """Focal length in pixels, at the resolution of the degraded image."""
    baseline_m : float
    """Stereo baseline in meters (ZED 2: 0.12, ZED Mini: 0.063, ZED X Mini: 0.05)."""
    disparity_noise_px : float = 0.2
    """Std dev of the disparity matching error, in pixels. Produces the z**2 depth error."""
    subpixel_step_px : float = 1.0/16
    """Disparity quantization step, in pixels. 0 disables quantization."""
    min_depth_m : float = 0.3
    """Measurements closer than this are dropped (ZED 2 is ~0.2-0.3 m depending on mode)."""
    max_depth_m : float = 20.0
    """Measurements farther than this are dropped."""
    fattening_px : int = 1
    """Radius of the foreground-fattening dilation, approximating the block-matching window."""
    occlusion_shadow_px : int = 4
    """Maximum width, in pixels, of the occlusion shadow searched for beside depth discontinuities.
    0 disables occlusion shadows."""
    shadow_direction : int = -1
    """Which side of a near surface the occlusion shadow falls on: -1 for the left, +1 for the
    right. Depends on which of the two cameras the depth is referenced to."""
    dropout_rate : float = 0.05
    """Approximate fraction of the image dropped as textureless/specular blobs. 0 disables it."""
    dropout_blob_px : int = 8
    """Characteristic size of the dropout blobs, in pixels."""


# Published specifications of the stereo cameras we model: focal length in pixels at the camera's
# native resolution, that native width, the baseline, and the usable depth range. The ZED X Mini is
# natively HD1200 (1920x1200) with a 50 mm baseline, and comes with one of two lenses.
_STEREO_CAMERAS = {
    "zedxmini2.2mm" : dict(focal_px=737.0,  native_width_px=1920, baseline_m=0.05,
                           min_depth_m=0.10, max_depth_m=8.0),
    "zedxmini4mm"   : dict(focal_px=1274.0, native_width_px=1920, baseline_m=0.05,
                           min_depth_m=0.15, max_depth_m=12.0),
}


def stereo_camera_params(camera : str,
                         render_width_px : int,
                         **overrides) -> StereoDepthSimParams:
    """Parameters approximating a real stereo camera, for depth rendered ``render_width_px`` wide.

    ``camera`` is a key of :data:`_STEREO_CAMERAS`, e.g. ``"zedxmini4mm"``. ``render_width_px`` is
    the render resolution, not the resolution the observation is later resized to. Pixel-valued
    figures are scaled from the camera's native width, the noise and quantization along with the
    focal length: since ``sigma_z = z**2*sigma_d/(f*B)``, scaling only the focal length would make a
    small render look like a far more precise sensor.

    Only the baseline, focal lengths and ranges are published. The matcher-dependent figures
    (disparity noise, window, dropout) are plausible block-matcher values rather than measurements,
    so treat them as a starting point. Any field can be replaced through ``overrides``.
    """
    if camera not in _STEREO_CAMERAS:
        raise ValueError(f"unknown stereo camera '{camera}', expected one of {sorted(_STEREO_CAMERAS)}")
    if render_width_px <= 0:
        raise ValueError(f"render_width_px must be positive, got {render_width_px}")
    spec = _STEREO_CAMERAS[camera]
    scale = render_width_px/spec["native_width_px"]
    params = StereoDepthSimParams(focal_px = spec["focal_px"]*scale,
                                  baseline_m = spec["baseline_m"],
                                  min_depth_m = spec["min_depth_m"],
                                  max_depth_m = spec["max_depth_m"],
                                  disparity_noise_px = 0.2*scale,
                                  subpixel_step_px = (1.0/16)*scale,
                                  # Native-pixel sizes, floored at 1: below a pixel they vanish.
                                  fattening_px = max(1, round(3*scale)),
                                  occlusion_shadow_px = max(1, round(16*scale)),
                                  dropout_rate = 0.05,
                                  dropout_blob_px = max(1, round(64*scale)))
    return replace(params, **overrides) if overrides else params


def stereo_camera_hfov_degrees(camera : str) -> float:
    """Horizontal field of view of a camera in :data:`_STEREO_CAMERAS`, in degrees.

    Derived from the focal length and native width already listed there, so a camera cannot end up
    rendered with a field of view that disagrees with the depth it is given.
    """
    if camera not in _STEREO_CAMERAS:
        raise ValueError(f"unknown stereo camera '{camera}', expected one of {sorted(_STEREO_CAMERAS)}")
    spec = _STEREO_CAMERAS[camera]
    return math.degrees(2*math.atan(spec["native_width_px"]/(2*spec["focal_px"])))


def depth_to_disparity(depth_m : th.Tensor, focal_px : float, baseline_m : float,
                       eps : float = 1e-6) -> th.Tensor:
    """Disparity, in pixels, of a metric depth image: ``d = f*B/z``."""
    return focal_px*baseline_m/depth_m.clamp(min=eps)


def disparity_to_depth(disparity_px : th.Tensor, focal_px : float, baseline_m : float,
                       eps : float = 1e-6) -> th.Tensor:
    """Metric depth of a disparity image: ``z = f*B/d``."""
    return focal_px*baseline_m/disparity_px.clamp(min=eps)


def _occlusion_shadow_mask(disparity_px : th.Tensor, width_px : int, direction : int,
                           eps : float = 1e-6) -> th.Tensor:
    """Pixels hidden from the second camera of a horizontal stereo pair.

    With a horizontal baseline, the pixel at column ``x`` is occluded when some pixel ``k`` columns
    to one side is nearer by more than ``k`` disparities, i.e. when
    ``max_k ( disp[x -+ k] - k ) > disp[x]``. That is a max-plus dilation with a linear structuring
    element, so the shadow beside an edge comes out as wide as the disparity jump, which is the
    actual geometric behaviour rather than a fixed-width band.
    """
    cols = disparity_px.shape[-1]
    padded = F.pad(disparity_px.unsqueeze(1), (width_px, width_px, 0, 0), mode="replicate").squeeze(1)
    cone = disparity_px
    for k in range(1, width_px+1):
        offset = width_px - k*direction
        cone = th.maximum(cone, padded[..., offset:offset+cols] - k)
    return cone > disparity_px + eps


def _dropout_blob_mask(shape : tuple[int,int,int], rate : float, blob_px : int,
                       device : th.device, dtype : th.dtype,
                       generator : th.Generator | None = None) -> th.Tensor:
    """Blobby dropout mask (True = dropped).

    Real stereo holes are patches of untextured or specular surface, not independent pixels, so the
    mask is low-resolution noise smoothly upsampled and thresholded. The realized dropped fraction
    is only approximately ``rate``: upsampling concentrates the uniform noise around 0.5.
    """
    vec, rows, cols = shape
    low_rows = max(1, rows//max(1, blob_px))
    low_cols = max(1, cols//max(1, blob_px))
    noise = th.rand((vec, 1, low_rows, low_cols), device=device, dtype=dtype, generator=generator)
    noise = F.interpolate(noise, size=(rows, cols), mode="bilinear", align_corners=False)
    return noise.squeeze(1) < rate


def stereo_depth_degradation(depth_m : th.Tensor,
                             params : StereoDepthSimParams,
                             generator : th.Generator | None = None) -> th.Tensor:
    """Degrade an exact metric depth image into a passive-stereo-like one.

    Apply this to the metric depth at *render* resolution, before resizing it down to the
    observation resolution: the artifacts are properties of the sensor, not of the resolution the
    policy happens to look at. Resize afterwards with :func:`resize_depth_masked`.

    Parameters
    ----------
    depth_m : th.Tensor
        ``(vec, rows, cols)`` metric depth. Non-positive values are treated as already invalid.
    params : StereoDepthSimParams
        The simulated camera. ``focal_px`` must match the resolution of ``depth_m``.
    generator : th.Generator | None
        Optional generator, for reproducible noise.

    Returns
    -------
    th.Tensor
        ``(vec, rows, cols)`` degraded metric depth, with ``INVALID_DEPTH`` where the camera would
        return no measurement.
    """
    if depth_m.dim() != 3:
        raise ValueError(f"depth_m must be (vec, rows, cols), got {tuple(depth_m.shape)}")
    f_b = params.focal_px*params.baseline_m
    if f_b <= 0:
        raise ValueError(f"focal_px*baseline_m must be positive, got {f_b}")

    # The sensor's range gate, plus whatever the renderer already marked as having no hit.
    valid = (depth_m > 0) & (depth_m >= params.min_depth_m) & (depth_m <= params.max_depth_m)

    # Everything the matcher does happens in disparity space, which is what makes the depth error
    # grow with z**2 and the quantization coarsen with range, without modelling either explicitly.
    disparity = f_b/depth_m.clamp(min=1e-6)
    if params.disparity_noise_px > 0:
        disparity = disparity + th.randn(disparity.shape, device=disparity.device,
                                         dtype=disparity.dtype, generator=generator)*params.disparity_noise_px
    if params.subpixel_step_px > 0:
        disparity = th.round(disparity/params.subpixel_step_px)*params.subpixel_step_px

    if params.occlusion_shadow_px > 0:
        valid = valid & ~_occlusion_shadow_mask(disparity, params.occlusion_shadow_px,
                                                params.shadow_direction)
    if params.fattening_px > 0:
        # The matching window makes the nearer surface win over its neighbours, which dilates it:
        # a max over disparity is a min over depth.
        kernel = 2*params.fattening_px+1
        disparity = F.max_pool2d(disparity.unsqueeze(1), kernel_size=kernel, stride=1,
                                 padding=params.fattening_px).squeeze(1)

    valid = valid & (disparity > 0)
    degraded = f_b/disparity.clamp(min=1e-6)
    # Noise and fattening can push a measurement out of the sensor's range.
    valid = valid & (degraded >= params.min_depth_m) & (degraded <= params.max_depth_m)

    if params.dropout_rate > 0:
        valid = valid & ~_dropout_blob_mask(tuple(depth_m.shape), params.dropout_rate,
                                            params.dropout_blob_px, depth_m.device, depth_m.dtype,
                                            generator=generator)

    return th.where(valid, degraded, th.full_like(degraded, INVALID_DEPTH))


def resize_depth_masked(depth_m : th.Tensor, size : tuple[int,int],
                        mode : str = "bilinear", antialias : bool = True,
                        eps : float = 1e-6) -> th.Tensor:
    """Resize a depth image without blending invalid pixels into the valid ones.

    A plain interpolation averages ``INVALID_DEPTH`` together with real measurements, pulling the
    depth around every hole towards zero and inventing surfaces that are not there. Interpolating
    ``depth*valid`` and ``valid`` separately and dividing gives the validity-weighted mean of the
    valid pixels alone; output pixels with no valid contribution stay invalid.

    Parameters
    ----------
    depth_m : th.Tensor
        ``(vec, rows, cols)`` depth, with ``INVALID_DEPTH`` marking missing measurements.
    size : tuple[int,int]
        Output ``(rows, cols)``.
    mode, antialias
        Passed to ``torch.nn.functional.interpolate``. Antialiasing matters when downsampling by a
        large factor, which is the usual case here.

    Returns
    -------
    th.Tensor
        ``(vec, size[0], size[1])`` resized depth, still using ``INVALID_DEPTH`` for holes.
    """
    if depth_m.dim() != 3:
        raise ValueError(f"depth_m must be (vec, rows, cols), got {tuple(depth_m.shape)}")
    valid = (depth_m > 0).to(depth_m.dtype)
    weighted = (depth_m*valid).unsqueeze(1)
    numerator = F.interpolate(weighted, size=size, mode=mode, align_corners=False, antialias=antialias)
    denominator = F.interpolate(valid.unsqueeze(1), size=size, mode=mode, align_corners=False, antialias=antialias)
    resized = th.where(denominator > eps, numerator/denominator.clamp(min=eps),
                       th.full_like(numerator, INVALID_DEPTH))
    return resized.squeeze(1)
