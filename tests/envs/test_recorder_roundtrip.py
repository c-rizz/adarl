"""What the recorder saves must be what happened, step by step.

The recorder captures one env's slice of the observations and infos and only writes them out when the
episode ends. It used to store those slices as plain views (`tensor[env_idx]`), which is correct only
as long as every single thing it captures happens to be freshly allocated each step. Anything handed
to it as a view into a reused buffer was recorded as that buffer's *final* contents, repeated for
every step of the episode, with no error anywhere -- which is the hardest kind of wrong data to
notice, because the file looks perfectly well formed.

The env below hands the recorder exactly that: an info entry which is a view of a buffer it rewrites
in place on every step. The recorded series must be 1,2,3,... and not N,N,N,...
"""
import glob
import os
import tempfile
import zipfile

import numpy as np
import pytest
import torch as th

from adarl.envs.vec.EnvRunner import EnvRunner
from adarl.envs.vec.EnvRunnerRecorderWrapper import EnvRunnerRecorderWrapper
from test_episode_bookkeeping import CounterVecEnv

EP_STEPS = 6


class InPlaceInfoVecEnv(CounterVecEnv):
    """A counter env whose info is a *view* of a buffer it overwrites every step."""

    def __init__(self, num_envs: int, max_ep_steps: th.Tensor):
        super().__init__(num_envs=num_envs, max_ep_steps=max_ep_steps)
        self._reused_buffer = th.zeros((num_envs,), dtype=th.float32)

    def post_step(self):
        # rewritten in place, exactly like the per-episode buffers the envs keep around
        self._reused_buffer.copy_(self._ep_step_counter.to(th.float32))

    def get_infos(self, state, labels=None):
        return {"counter": self._reused_buffer}   # deliberately not a copy


def _read_recorded_counter(folder: str) -> np.ndarray:
    h5py = pytest.importorskip("h5py")
    files = glob.glob(os.path.join(folder, "**", "*_info.hdf5.h5zip"), recursive=True)
    assert files, f"the recorder wrote no info file into {folder}"
    with zipfile.ZipFile(sorted(files)[0]) as zf:
        name = [n for n in zf.namelist() if n.endswith((".hdf5", ".h5"))][0]
        extracted = zf.extract(name, tempfile.mkdtemp())
    with h5py.File(extracted, "r") as f:
        assert "counter" in f, f"'counter' missing from the recording, got {list(f.keys())}"
        return np.asarray(f["counter"]).reshape(-1)


@pytest.fixture
def _session_run_info():
    """The recorder reads a couple of run_info keys when it saves; provide and restore them."""
    import adarl.utils.session as session
    import types
    run_info = getattr(session.default_session, "run_info", None)
    if run_info is None:
        pytest.skip("no default session to attach run_info to")
    added = []
    for key, value in (("run_id", "recorder_roundtrip"),
                       ("collected_episodes", types.SimpleNamespace(value=0)),
                       ("collected_steps", types.SimpleNamespace(value=0))):
        if key not in run_info:
            run_info[key] = value
            added.append(key)
    yield run_info
    for key in added:
        del run_info[key]


def test_record_step_snapshots_what_it_is_given(tmp_path):
    """The capture itself must copy, independently of whether the caller happened to hand it copies.

    Driven through the runner, the recorder currently receives freshly built infos, so a view would
    survive and this hazard is invisible end to end. This calls the capture directly with a view of a
    buffer that is then overwritten, which is the contract that has to hold for the recording to mean
    anything: what was captured at step k must still read as step k's values afterwards.
    """
    env = CounterVecEnv(num_envs=2, max_ep_steps=th.full((2,), EP_STEPS, dtype=th.long))
    runner = EnvRunner(env, quiet=True, autoreset=True)
    recorder = EnvRunnerRecorderWrapper(runner=runner, fps=10.0, outFolder=str(tmp_path),
                                        env_index=0, record_video=False, use_global_ep_count=False,
                                        saveFrequency_ep=1, vec_obs_keys=["count"])
    reused = th.zeros((2,), dtype=th.float32)
    obs = {"count": th.zeros((2, 1), dtype=th.float32)}
    reused.fill_(7.0)
    recorder._record_step(obs, th.zeros((2, 1)), {"counter": reused}, th.zeros((2,)),
                          th.zeros((2,), dtype=th.bool), th.zeros((2,), dtype=th.bool))
    reused.fill_(99.0)          # the env moves on and rewrites its buffer
    captured = recorder._infoBuffer[-1]["counter"]
    assert float(th.as_tensor(captured).reshape(-1)[0]) == 7.0, (
        f"the recorder captured {captured}, which followed the buffer to 99.0: it stored a live view "
        f"instead of a snapshot, so a recording reflects the end of the episode at every step")
    runner.close()


def test_recorded_info_is_a_snapshot_not_a_live_view(tmp_path, _session_run_info):
    env = InPlaceInfoVecEnv(num_envs=2, max_ep_steps=th.full((2,), EP_STEPS, dtype=th.long))
    runner = EnvRunner(env, quiet=True, autoreset=True)
    recorder = EnvRunnerRecorderWrapper(runner=runner, fps=10.0, outFolder=str(tmp_path),
                                        env_index=0, record_video=False, use_global_ep_count=False,
                                        saveFrequency_ep=1, vec_obs_keys=["count"])
    recorder.reset()
    action = th.zeros((2, 1), dtype=th.float32)
    for _ in range(EP_STEPS + 1):   # one full episode, plus the step that closes it out
        recorder.step(action)
    recorder.close()

    recorded = _read_recorded_counter(str(tmp_path))
    assert len(set(recorded.tolist())) > 1, (
        f"every recorded step has the same value ({recorded.tolist()}): the recorder kept a view of a "
        f"buffer the env rewrites in place, so the whole episode reads as its final state")
    # the counter is written after each step, so consecutive steps must differ by one
    diffs = np.diff(recorded[:EP_STEPS])
    assert np.all(diffs == 1), (
        f"recorded counter {recorded.tolist()} is not the per-step sequence the env produced "
        f"(differences {diffs.tolist()}, expected all 1)")
