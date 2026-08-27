"""Tests for ObsConverter's passthrough-components support.

Passthrough keys are vector components kept *out* of the concatenated vector part and served
separately, so a consumer (the autoencoders) can route them around the encoder and append them
to the latent. ObsConverter only decides the layout; what to do with them is the caller's choice.

The critical property guarded here is that with no passthrough keys everything behaves exactly
as before -- ObsConverter is shared by the feature extractors and the AE stack.
"""
import numpy as np
import pytest
import torch as th

import adarl.utils.spaces as spaces
from adarl.utils.ObsConverter import ObsConverter


BATCH = 5


def _box(n):
    return spaces.ThBox(low=-np.ones(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))


def _img_box(shape=(3, 8, 8)):
    return spaces.ThBox(low=np.zeros(shape, dtype=np.float32), high=np.ones(shape, dtype=np.float32))


def _space(with_image=True):
    d = {"proprio": _box(4), "extrinsic": _box(2)}
    if with_image:
        d["camera"] = _img_box()
    return spaces.ThDict(d)


def _obs(with_image=True, traj=None):
    o = {"proprio": th.arange(BATCH * 4, dtype=th.float32).view(BATCH, 4),
         "extrinsic": th.arange(BATCH * 2, dtype=th.float32).view(BATCH, 2) + 100}
    if with_image:
        o["camera"] = th.zeros(BATCH, 3, 8, 8)
    if traj is not None:
        o = {k: v.unsqueeze(1).repeat(1, traj, *([1] * (v.dim() - 1))) for k, v in o.items()}
    return o


# ------------------------------------------------------- default behaviour must not change


def test_without_passthrough_nothing_changes():
    c = ObsConverter(_space())
    obs = _obs()
    assert c.vector_part_size() == 6
    assert c.passthrough_part_size() == 0
    assert not c.has_passthrough_part()
    assert c.getVectorPart(obs).shape == (BATCH, 6)


def test_without_passthrough_roundtrip_is_complete():
    c = ObsConverter(_space())
    obs = _obs()
    rebuilt = c.buildDictObs(c.getVectorPart(obs), c.getImgPart(obs))
    assert set(rebuilt.keys()) == {"proprio", "extrinsic", "camera"}
    assert th.equal(rebuilt["proprio"], obs["proprio"])
    assert th.equal(rebuilt["extrinsic"], obs["extrinsic"])


def test_empty_passthrough_part_keeps_batch_dims():
    c = ObsConverter(_space())
    assert c.getPassthroughPart(_obs()).shape == (BATCH, 0)


# --------------------------------------------------------------------------- splitting out


def test_passthrough_is_removed_from_the_vector_part():
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    obs = _obs()
    assert c.vector_part_size() == 2 and c.passthrough_part_size() == 4
    assert c.has_passthrough_part()
    assert th.equal(c.getVectorPart(obs), obs["extrinsic"])
    assert th.equal(c.getPassthroughPart(obs), obs["proprio"])


def test_passthrough_keys_are_reported():
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    assert c.passthrough_keys() == ("proprio",)


def test_multiple_passthrough_keys_keep_relative_order():
    sp = spaces.ThDict({"a": _box(1), "b": _box(2), "c": _box(3)})
    c = ObsConverter(sp, passthrough_keys=["c", "a"])  # declared order must not matter
    obs = {"a": th.ones(BATCH, 1), "b": th.ones(BATCH, 2) * 2, "c": th.ones(BATCH, 3) * 3}
    assert th.equal(c.getVectorPart(obs), obs["b"])
    # kept in the space's own order (a before c), not the order they were requested in
    assert th.equal(c.getPassthroughPart(obs), th.cat([obs["a"], obs["c"]], dim=-1))


def test_passthrough_limits_match_the_components():
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    lo, hi = c.getPassthroughPartLimits()
    assert lo.shape == (4,) and hi.shape == (4,)
    assert np.allclose(lo, -1) and np.allclose(hi, 1)


@pytest.mark.parametrize("traj", [None, 3])
def test_works_for_batched_and_trajectory_observations(traj):
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    obs = _obs(traj=traj)
    lead = (BATCH,) if traj is None else (BATCH, traj)
    assert c.getVectorPart(obs).shape == lead + (2,)
    assert c.getPassthroughPart(obs).shape == lead + (4,)


def test_works_without_an_image_part():
    c = ObsConverter(_space(with_image=False), passthrough_keys=["proprio"])
    obs = _obs(with_image=False)
    assert th.equal(c.getPassthroughPart(obs), obs["proprio"])


# -------------------------------------------------------------------------------- rebuild


def test_rebuild_with_passthrough_restores_every_component():
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    obs = _obs()
    rebuilt = c.buildDictObs(c.getVectorPart(obs), c.getImgPart(obs),
                             passthroughPart_batch=c.getPassthroughPart(obs))
    assert set(rebuilt.keys()) == {"proprio", "extrinsic", "camera"}
    assert th.equal(rebuilt["proprio"], obs["proprio"])
    assert th.equal(rebuilt["extrinsic"], obs["extrinsic"])


def test_rebuild_without_passthrough_omits_those_components():
    """The AE does not reconstruct passthrough components -- they bypass it entirely."""
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    obs = _obs()
    rebuilt = c.buildDictObs(c.getVectorPart(obs), c.getImgPart(obs))
    assert set(rebuilt.keys()) == {"extrinsic", "camera"}


def test_to_standard_tensors_covers_passthrough_components():
    c = ObsConverter(_space(), passthrough_keys=["proprio"])
    obs = {k: v.numpy() for k, v in _obs().items()}
    out = c.to_standard_tensors(obs, device=th.device("cpu"))
    assert isinstance(out["proprio"], th.Tensor)


# --------------------------------------------------------------------------------- errors


def test_image_component_cannot_be_a_passthrough():
    with pytest.raises(RuntimeError, match="not vector components"):
        ObsConverter(_space(), passthrough_keys=["camera"])


def test_unknown_passthrough_key_is_rejected():
    with pytest.raises(RuntimeError, match="not vector components"):
        ObsConverter(_space(), passthrough_keys=["nope"])
