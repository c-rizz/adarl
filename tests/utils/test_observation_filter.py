"""Unit tests for adarl.utils.ObservationFilter.

Covers space analysis (vector vs image), the default vec/img filters, user-defined filters
(subset + concatenation), the derived spaces (shape, limits, labels), error cases, and that the
hot-path `get()` survives torch.compile, since it is meant to run inside policy code.
"""
import numpy as np
import pytest
import torch as th

import adarl.utils.spaces as spaces
from adarl.utils.ObservationFilter import Filter, ObservationFilter


VEC_SIZES = {"intrinsic": 3, "extrinsic": 2, "proprio": 4}
IMG_CHW = (3, 8, 8)


def _box(size, low=-1.0, high=1.0, labels=None):
    return spaces.ThBox(low=np.full((size,), low, dtype=np.float32),
                        high=np.full((size,), high, dtype=np.float32),
                        shape=(size,), dtype=np.float32, labels=labels)


def _img_box(shape=IMG_CHW):
    return spaces.ThBox(low=np.zeros(shape, dtype=np.float32),
                        high=np.ones(shape, dtype=np.float32),
                        shape=shape, dtype=np.float32)


def _obs_space(with_image=True):
    d = {k: _box(v) for k, v in VEC_SIZES.items()}
    if with_image:
        d["camera"] = _img_box()
    return spaces.ThDict(d)


def _obs(batch=5, with_image=True):
    o = {k: th.arange(batch * v, dtype=th.float32).view(batch, v) for k, v in VEC_SIZES.items()}
    if with_image:
        o["camera"] = th.zeros((batch,) + IMG_CHW)
    return o


# ------------------------------------------------------------------ space analysis / defaults


def test_splits_vectors_and_images():
    f = ObservationFilter(_obs_space())
    # gymnasium.spaces.Dict sorts its keys, so this is alphabetical rather than insertion order
    assert f.vector_keys() == ("extrinsic", "intrinsic", "proprio")
    assert f.image_keys() == ("camera",)
    assert f.is_image("camera") and not f.is_image("intrinsic")
    assert f.image_shape_chw("camera") == IMG_CHW


def test_two_dimensional_image_gets_a_channel_dim():
    sp = spaces.ThDict({"gray": _img_box((8, 8)), "vec": _box(2)})
    f = ObservationFilter(sp)
    assert f.image_keys() == ("gray",)
    assert f.image_shape_chw("gray") == (1, 8, 8)


def test_default_filters_exist_and_behave():
    f = ObservationFilter(_obs_space())
    assert set(f.filter_names()) == {"vec", "img"}
    assert f.is_concat("vec") and not f.is_concat("img")

    obs = _obs()
    vec = f.get("vec", obs)
    assert isinstance(vec, th.Tensor)
    assert vec.shape == (5, sum(VEC_SIZES.values()))

    img = f.get("img", obs)
    assert isinstance(img, dict) and set(img.keys()) == {"camera"}


def test_defaults_are_skipped_when_no_such_component():
    f = ObservationFilter(_obs_space(with_image=False))
    assert f.filter_names() == ("vec",)
    assert "img" not in f


def test_defaults_can_be_disabled_and_renamed():
    assert ObservationFilter(_obs_space(), default_filters=False).filter_names() == ()
    f = ObservationFilter(_obs_space(), vec_filter_name="V", img_filter_name="I")
    assert set(f.filter_names()) == {"V", "I"}


def test_user_filter_overrides_a_default():
    f = ObservationFilter(_obs_space(), [Filter("vec", ["proprio"], concat=True)])
    assert f.keys("vec") == ("proprio",)
    assert f.get("vec", _obs()).shape == (5, VEC_SIZES["proprio"])


# ------------------------------------------------------------------------------ user filters


def test_concat_filter_selects_and_concatenates_in_declared_order():
    f = ObservationFilter(_obs_space(), [Filter("privileged", ["extrinsic", "intrinsic"], concat=True)])
    obs = _obs()
    got = f.get("privileged", obs)
    assert th.equal(got, th.cat([obs["extrinsic"], obs["intrinsic"]], dim=-1))  # declared order, not dict order
    assert got.shape == (5, VEC_SIZES["extrinsic"] + VEC_SIZES["intrinsic"])


def test_non_concat_filter_returns_dict_subset():
    f = ObservationFilter(_obs_space(), [Filter("some", ["intrinsic", "camera"])])
    obs = _obs()
    got = f.get("some", obs)
    assert isinstance(got, dict) and set(got.keys()) == {"intrinsic", "camera"}
    assert got["intrinsic"] is obs["intrinsic"]  # a subset, not a copy


def test_keys_none_means_everything():
    f = ObservationFilter(_obs_space(), [Filter("all", None)], default_filters=False)
    assert f.keys("all") == ("camera", "extrinsic", "intrinsic", "proprio")  # sorted by gym's Dict


@pytest.mark.parametrize("shape", [(7,), (5, 7), (2, 5, 7)])
def test_concat_handles_unbatched_batched_and_trajectory_shapes(shape):
    """Concatenation is along the last dim, so any leading dims pass through untouched."""
    sp = spaces.ThDict({"a": _box(7), "b": _box(7)})
    f = ObservationFilter(sp)
    obs = {"a": th.zeros(shape), "b": th.ones(shape)}
    assert f.get("vec", obs).shape == shape[:-1] + (14,)


# ----------------------------------------------------------------------------------- spaces


def test_concat_space_shape_and_limits():
    sp = spaces.ThDict({"a": _box(2, low=-3.0, high=4.0), "b": _box(3, low=-1.0, high=1.0)})
    f = ObservationFilter(sp)
    space = f.get_space("vec")
    assert isinstance(space, spaces.ThBox)
    assert space.shape == (5,)
    assert np.allclose(space.low, [-3, -3, -1, -1, -1])
    assert np.allclose(space.high, [4, 4, 1, 1, 1])


def test_concat_space_labels_are_prefixed_with_the_component():
    sp = spaces.ThDict({"a": _box(2, labels=np.array(["x", "y"], dtype=object)),
                        "b": _box(1, labels=np.array(["z"], dtype=object))})
    f = ObservationFilter(sp)
    assert list(f.get_space("vec").labels) == ["a.x", "a.y", "b.z"]


def test_non_concat_space_is_a_dict_of_the_original_subspaces():
    sp = _obs_space()
    f = ObservationFilter(sp, [Filter("some", ["intrinsic", "camera"])])
    space = f.get_space("some")
    assert isinstance(space, spaces.ThDict)
    assert set(space.spaces.keys()) == {"intrinsic", "camera"}
    assert space.spaces["camera"] is sp.spaces["camera"]


def test_space_matches_what_get_returns():
    f = ObservationFilter(_obs_space(), [Filter("privileged", ["intrinsic", "extrinsic"], concat=True)])
    got = f.get("privileged", _obs(batch=1))
    assert got.shape[1:] == f.get_space("privileged").shape


# ---------------------------------------------------------------------------------- errors


def test_rejects_non_dict_space():
    with pytest.raises(ValueError, match="must be a Dict"):
        ObservationFilter(_box(3))


def test_rejects_nested_dict_space():
    sp = spaces.ThDict({"a": _box(2), "nested": spaces.ThDict({"b": _box(2)})})
    with pytest.raises(ValueError, match="nested dicts are not"):
        ObservationFilter(sp)


def test_unknown_key_fails_at_init_and_lists_available():
    with pytest.raises(ValueError, match="unknown observation components"):
        ObservationFilter(_obs_space(), [Filter("bad", ["nope"])])


def test_concatenating_an_image_is_rejected():
    with pytest.raises(ValueError, match="Only vector components can be concatenated"):
        ObservationFilter(_obs_space(), [Filter("bad", ["intrinsic", "camera"], concat=True)])


def test_concatenating_mixed_dtypes_is_rejected():
    sp = spaces.ThDict({"a": _box(2),
                        "b": spaces.ThBox(low=np.zeros((2,), dtype=np.float64),
                                          high=np.ones((2,), dtype=np.float64),
                                          shape=(2,), dtype=np.float64)})
    with pytest.raises(ValueError, match="different dtypes"):
        ObservationFilter(sp, [Filter("bad", ["a", "b"], concat=True)])


def test_duplicated_and_empty_key_lists_are_rejected():
    with pytest.raises(ValueError, match="duplicated"):
        ObservationFilter(_obs_space(), [Filter("bad", ["intrinsic", "intrinsic"])])
    with pytest.raises(ValueError, match="selects no observation component"):
        ObservationFilter(_obs_space(), [Filter("bad", [])])


def test_unknown_filter_name_lists_the_defined_ones():
    f = ObservationFilter(_obs_space())
    with pytest.raises(KeyError, match="No filter named"):
        f.get("nope", _obs())


# --------------------------------------------------------------------------- torch.compile


def test_get_compiles_fullgraph():
    """`get` runs in the policy hot path, so it must capture without graph breaks."""
    f = ObservationFilter(_obs_space(), [Filter("privileged", ["intrinsic", "extrinsic"], concat=True)])
    obs = _obs()

    def run(o):
        return f.get("privileged", o)

    compiled = th.compile(run, fullgraph=True, backend="eager")
    assert th.equal(compiled(obs), run(obs))
