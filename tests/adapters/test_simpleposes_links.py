"""SimplePosesLinks and ZmqXbotAdapter.getLinksPose: link poses for real-robot adapters, received with simpleposes.

Driven by a real simpleposes publisher, so no robot is needed. Checks the poses, the name mapping, that a link
published in another frame than the requested one is refused, that a stale snapshot is refused, and that the xbot
adapters return the poses as getLinksPose() does in simulation.
"""
import threading
import time

import numpy as np
import pytest
import torch as th

simpleposes = pytest.importorskip("simpleposes")

from adarl.adapters.simpleposes_links import SimplePosesLinks  # noqa: E402
from adarl.adapters.VecZmqXbotAdapter import VecZmqXbotAdapter  # noqa: E402
from adarl.adapters.ZmqXbotAdapter import ZmqXbotAdapter  # noqa: E402

PUBLISHED = {"pelvis" : "world", "cube" : "world", "dagana_1_tcp" : "pelvis"}
POSES = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                  [0.8, 0.0, 0.1, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
                  [0.5, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]])
PELVIS = ("kyon", "pelvis")
CUBE = ("cube", "cube_link")
TCP = ("kyon", "dagana_1_tcp")


@pytest.fixture
def publisher():
    """Publishes PUBLISHED with POSES at 200 Hz until stop() (or the end of the test)."""
    pub = simpleposes.PosePublisher(PUBLISHED, "tcp://127.0.0.1:*")
    stop = threading.Event()

    def run(): # the socket is only used by this thread until it stops
        t = 0.0
        while not stop.is_set():
            pub.publish(POSES, t=t)
            t += 0.005
            time.sleep(0.005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def stop_publishing():
        stop.set()
        thread.join(timeout=2.0)

    pub.stop = stop_publishing
    yield pub
    stop_publishing()
    pub.close()


def _links(pub, **kwargs) -> SimplePosesLinks:
    return SimplePosesLinks(pub.endpoint, out_th_device=th.device("cpu"), link_names={"cube_link" : "cube"}, **kwargs)


def test_poses_and_name_mapping(publisher):
    links = _links(publisher)
    try:
        poses = links.get_links_pose([PELVIS, CUBE], reference_frame="world")
        assert poses.shape == (2, 7) and poses.dtype == th.float32
        np.testing.assert_allclose(poses.numpy(), POSES[:2], atol=1e-6)
        tcp = links.get_links_pose([TCP], reference_frame="pelvis")
        np.testing.assert_allclose(tcp.numpy(), POSES[2:], atol=1e-6)
    finally:
        links.close()


def test_refuses_a_link_published_in_another_frame(publisher):
    links = _links(publisher)
    try:
        with pytest.raises(ValueError, match="published in frame 'pelvis', but 'world' was requested"):
            links.get_links_pose([PELVIS, TCP], reference_frame="world")
    finally:
        links.close()


def test_refuses_an_unpublished_link(publisher):
    links = _links(publisher)
    try:
        with pytest.raises(KeyError, match="'nope'"):
            links.get_links_pose([("kyon", "nope")], reference_frame="world")
    finally:
        links.close()


def test_refuses_a_stale_snapshot(publisher):
    links = _links(publisher, max_pose_age_sec=0.1)
    try:
        links.get_links_pose([PELVIS], reference_frame="world")
        publisher.stop()
        time.sleep(0.3)
        with pytest.raises(RuntimeError, match="arrived .* ago"):
            links.get_links_pose([PELVIS], reference_frame="world")
    finally:
        links.close()


def test_times_out_without_a_publisher():
    links = SimplePosesLinks("tcp://127.0.0.1:9", out_th_device=th.device("cpu"), first_pose_timeout_sec=0.2)
    try:
        with pytest.raises(TimeoutError):
            links.get_links_pose([PELVIS], reference_frame="world")
    finally:
        links.close()


def _xbot_adapter(links : SimplePosesLinks | None) -> ZmqXbotAdapter:
    """A ZmqXbotAdapter with only what getLinksPose() uses: the constructor would connect to XBot."""
    adapter = object.__new__(ZmqXbotAdapter)
    adapter._links = links
    adapter._monitored_links = [PELVIS, CUBE]
    return adapter


def test_xbot_adapters_return_the_poses(publisher):
    links = _links(publisher)
    try:
        adapter = _xbot_adapter(links)
        poses = adapter.getLinksPose([PELVIS, CUBE])
        np.testing.assert_allclose(poses[CUBE].position.numpy(), POSES[1, :3], atol=1e-6)
        np.testing.assert_allclose(poses[CUBE].orientation_xyzw.numpy(), POSES[1, 3:], atol=1e-6)
        with pytest.raises(ValueError, match="use_com_pose"):
            adapter.getLinksPose([PELVIS], use_com_pose=True)
        with pytest.raises(RuntimeError, match="getLinksPose"):
            adapter.getLinksState([PELVIS])

        vec = object.__new__(VecZmqXbotAdapter)
        vec._sub_adapter = adapter
        vec._out_th_device = th.device("cpu")
        vec._out_th_float_dtype = th.float32
        vec.sub_adapter = lambda: adapter
        monitored = vec.getLinksPose()
        assert monitored.shape == (1, 2, 7)
        np.testing.assert_allclose(monitored[0].numpy(), POSES[:2], atol=1e-6)
        assert vec.getLinksPose([]).shape == (1, 0, 7)
    finally:
        links.close()


def test_xbot_adapter_without_simpleposes_url():
    with pytest.raises(RuntimeError, match="simpleposes_url"):
        _xbot_adapter(None).getLinksPose([PELVIS])
