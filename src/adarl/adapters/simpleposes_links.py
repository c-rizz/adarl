"""Link poses for real-robot adapters, received with simpleposes.

A simpleposes publisher (e.g. xbot2_mujoco's simulator with --pose-link, or a tracker on the real robot) publishes the
pose of each of its links in a reference frame of its own choosing. This module returns them as getLinksPose() does
in simulation, after checking that each link is published in the frame the caller asks for: poses are not composed
between frames. Link velocities are not published.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch as th


class SimplePosesLinks:
    """The latest link poses from a simpleposes publisher. At every request the age of the latest snapshot is
    checked: a stale pose looks perfectly valid."""

    def __init__(self, url : str, out_th_device : th.device,
                 link_names : dict[str, str] | None = None,
                 max_pose_age_sec : float = 0.25,
                 first_pose_timeout_sec : float = 5.0):
        """
        Parameters
        ----------
        url : str
            The publisher's endpoint, e.g. "tcp://robot-pc:5800".
        out_th_device : th.device
            Device of the returned tensors.
        link_names : dict[str, str] | None
            Maps the link names the environment uses (the link part of its (model, link) names) to the names they
            are published with. Unmapped names are used as they are.
        max_pose_age_sec : float
            get_links_pose() raises if the latest snapshot arrived longer ago than this.
        first_pose_timeout_sec : float
            How long get_links_pose() waits for the first snapshot.
        """
        try:
            import simpleposes
        except ImportError as e:
            raise ImportError("link poses over ZMQ need the simpleposes package (pip install -e simpleposes)") from e
        self._url = url
        self._out_th_device = out_th_device
        self._link_names = dict(link_names or {})
        self._max_pose_age_sec = max_pose_age_sec
        self._first_pose_timeout_sec = first_pose_timeout_sec
        self._sub = simpleposes.PoseSubscriber(url, conflate=True, clock=time.monotonic)

    def _remote_name(self, link : tuple[str, str]) -> str:
        return self._link_names.get(link[1], link[1])

    def get_links_pose(self, requestedLinks : Sequence[tuple[str, str]], reference_frame : str) -> th.Tensor:
        """(len(requestedLinks), 7) float32: position_xyz, orientation_xyzw of each link, in reference_frame."""
        snapshot = self._sub.latest(timeout_s=self._first_pose_timeout_sec)
        if snapshot is None:
            raise TimeoutError(f"no link poses received from {self._url} within {self._first_pose_timeout_sec}s")
        age = time.monotonic() - snapshot.t_recv
        if age > self._max_pose_age_sec:
            raise RuntimeError(f"the latest link poses from {self._url} arrived {age:.3f}s ago "
                               f"(max_pose_age_sec={self._max_pose_age_sec})")
        poses = np.empty((len(requestedLinks), 7), dtype=np.float32)
        for i, link in enumerate(requestedLinks):
            remote = self._remote_name(link)
            if remote not in snapshot.links:
                raise KeyError(f"link {link} (published as {remote!r}) is not published at {self._url}, "
                               f"available: {snapshot.links}")
            frame = snapshot.frame(remote)
            if frame != reference_frame:
                raise ValueError(f"link {link} is published in frame {frame!r}, but {reference_frame!r} was requested")
            poses[i] = snapshot.pose(remote)
        return th.as_tensor(poses, device=self._out_th_device)

    def close(self):
        self._sub.close()
