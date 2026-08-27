"""Unit tests for the rolling history buffer in adarl.utils.vec_state_helper.ThBoxStateHelper.

`ThBoxStateHelper.update()` keeps a per-environment ring of the last `history_length`
states, implemented as `state.roll(1, dims=1)` with the newest sample written to index 0
(vec_state_helper.py, `update()`). This buffer feeds the per-step state update that
RobotVecEnv wants to run under `torch.compile` -- the `@th.compile(...)` on
`_post_step_optimized` is currently commented out with the note that it "seems to cause
issues with the rolling buffer used in states with history".

These tests pin down (a) the rolling semantics in eager mode and (b) that the update and
the history-truncating observe path capture as a *single* graph under
`torch.compile(fullgraph=True)` -- i.e. no graph break -- and produce identical results to
eager, so the compile-time regression can't silently come back.
"""
import numpy as np
import pytest
import torch as th

from adarl.utils.vec_state_helper import ThBoxStateHelper

requires_cuda = pytest.mark.skipif(not th.cuda.is_available(), reason="needs a CUDA GPU")


VEC_SIZE = 4
FIELDS = ["a", "b"]
FIELD_SIZE = [3]
HISTORY = 3


def _make_helper_on(device, history_length=HISTORY, obs_history_length=1):
    """A small helper on `device`: 2 fields of size 3, vec_size 4, `history_length` deep."""
    return ThBoxStateHelper(
        fields_minmax={f: [-10.0, 10.0] for f in FIELDS},
        dtype=th.float32,
        th_device=th.device(device),
        field_size=FIELD_SIZE,
        vec_size=VEC_SIZE,
        field_names=FIELDS,
        history_length=history_length,
        observation_definitions=ThBoxStateHelper.SimpleObsDef(
            obs_history_length=obs_history_length
        ),
    )


def _make_helper(history_length=HISTORY, obs_history_length=1):
    """CPU helper (the fast lane default)."""
    return _make_helper_on("cpu", history_length, obs_history_length)


def _sample_on(device, step):
    """A distinct instantaneous state per `step`, shape (vec_size, fields_num, *field_shape).

    Value encodes both the step and the per-env/field/subfield position so any mis-shift,
    transpose or broadcast is visible.
    """
    fields_num = len(FIELDS)
    base = th.arange(VEC_SIZE * fields_num * FIELD_SIZE[0], dtype=th.float32, device=device)
    base = base.view(VEC_SIZE, fields_num, FIELD_SIZE[0]) * 0.01
    return base + float(step)


def _sample(step):
    return _sample_on("cpu", step)


# --------------------------------------------------------------------------- eager


def test_reset_fills_whole_history():
    h = _make_helper()
    state = h.reset_state(0.0)
    assert state.shape == (VEC_SIZE, HISTORY, len(FIELDS), FIELD_SIZE[0])
    assert th.count_nonzero(state) == 0


def test_update_writes_newest_to_index_zero():
    h = _make_helper()
    state = h.reset_state(0.0)
    s0 = _sample(1)
    state = h.update(s0, state, inplace=False)
    assert th.equal(state[:, 0], s0)          # newest at front
    assert th.count_nonzero(state[:, 1:]) == 0  # rest still the reset value


def test_update_rolls_and_drops_oldest():
    h = _make_helper(history_length=3)
    state = h.reset_state(0.0)
    samples = [_sample(k) for k in range(1, 5)]  # push 4 into a depth-3 history
    for s in samples:
        state = h.update(s, state, inplace=False)
    # newest..oldest along the history dim; the very first push has fallen off
    assert th.equal(state[:, 0], samples[3])
    assert th.equal(state[:, 1], samples[2])
    assert th.equal(state[:, 2], samples[1])


def test_update_inplace_mutates_and_returns_same_tensor():
    h = _make_helper()
    state = h.reset_state(0.0)
    ret = h.update(_sample(1), state, inplace=True)
    assert ret is state
    assert th.equal(state[:, 0], _sample(1))


def test_update_accepts_mapping_input():
    h = _make_helper()
    state = h.reset_state(0.0)
    tens = _sample(1)
    mapping = {"a": tens[:, 0], "b": tens[:, 1]}
    state = h.update(mapping, state, inplace=False)
    assert th.equal(state[:, 0], tens)


# ------------------------------------------------------------------- torch.compile


def _run_sequence(update_fn, h, steps=4):
    """Push `steps` distinct samples through `update_fn(sample, state)` (inplace=False)."""
    state = h.reset_state(0.0)
    for k in range(1, steps + 1):
        state = update_fn(_sample(k), state)
    return state


def test_update_compiles_fullgraph_no_break():
    """The rolling update must capture as one graph -- fullgraph=True errors on any break.

    Uses backend="eager" so this asserts the dynamo graph-break behaviour (the historical
    failure mode) without needing a C toolchain.
    """
    h = _make_helper()

    def upd(sample, state):
        return h.update(sample, state, inplace=False)

    eager = _run_sequence(upd, h)
    compiled_fn = th.compile(upd, fullgraph=True, backend="eager")
    got = _run_sequence(compiled_fn, h)
    assert th.equal(got, eager)


def test_update_inplace_compiles_fullgraph():
    """The in-place variant (state.copy_ of the rolled buffer) also captures fullgraph."""
    h = _make_helper()

    def upd(sample, state):
        h.update(sample, state, inplace=True)
        return state

    compiled_fn = th.compile(upd, fullgraph=True, backend="eager")
    state = h.reset_state(0.0)
    for k in range(1, 4):
        state = compiled_fn(_sample(k), state)
    assert th.equal(state[:, 0], _sample(3))
    assert th.equal(state[:, 1], _sample(2))
    assert th.equal(state[:, 2], _sample(1))


def test_observe_history_slice_compiles_fullgraph():
    """observe() truncates the history dim (state[:, :obs_history_length, ...]); check it
    too captures fullgraph, since it runs in the same per-step compiled region."""
    h = _make_helper(history_length=3, obs_history_length=2)

    def step(sample, state):
        state = h.update(sample, state, inplace=False)
        return h.observe(state)

    eager_state = h.reset_state(0.0)
    compiled = th.compile(step, fullgraph=True, backend="eager")
    for k in range(1, 4):
        obs_c = compiled(_sample(k), eager_state)
        eager_state = h.update(_sample(k), eager_state, inplace=False)
        obs_e = h.observe(eager_state)
    assert obs_c.shape[1] == 2  # only the two most recent history steps observed
    assert th.equal(obs_c, obs_e)


def test_update_compiles_max_autotune_cpu_when_available():
    """Best-effort CPU codegen with `mode="max-autotune"` (the mode RobotVecEnv's commented
    `@th.compile` on `_post_step_optimized` requests). NOTE: on CPU this exercises autotuned
    codegen only -- `max-autotune`'s cudagraphs half is CUDA-only, so this does NOT cover the
    cudagraph hazard; the GPU tests below do. Skips where no compiler toolchain exists."""
    h = _make_helper()

    def upd(sample, state):
        return h.update(sample, state, inplace=False)

    eager = _run_sequence(upd, h)
    try:
        compiled_fn = th.compile(upd, mode="max-autotune", fullgraph=True)
        got = _run_sequence(compiled_fn, h)
    except Exception as e:  # noqa: BLE001 -- environmental (missing gcc/triton), not a logic failure
        pytest.skip(f"max-autotune backend unavailable in this environment: {type(e).__name__}: {e}")
    assert th.equal(got, eager)


# ---------------------------------------------- torch.compile cudagraphs (GPU only)
#
# `mode="max-autotune"` turns on cudagraphs on CUDA. Cudagraphs reuse *static* input/output
# buffers, so feeding a compiled call's output straight back in as the next call's `state`
# argument -- exactly the rolling-history pattern `state = update(sample, state)` -- makes the
# graph read a buffer a later run has already overwritten. This is the concrete failure behind
# the commented-out `@th.compile(mode="max-autotune", ...)` at RobotVecEnv `_post_step_optimized`.


@requires_cuda
def test_rolling_feedback_under_cudagraphs_is_unsafe():
    """The naive `state = compiled(sample, state)` loop must NOT be silently trusted under
    max-autotune on CUDA -- documents the RobotVecEnv:2069 hazard. The cudagraph static-buffer
    aliasing has two faces, and this asserts *either*:
      - it raises the "output of CUDAGraphs ... overwritten" RuntimeError (torch's tree checker
        catches it -- what current torch does), OR
      - it silently returns data that diverges from eager (the pre-checker symptom: wrong values,
        which is what led to the non-finite states originally seen).
    Both are "unsafe"; only the *cloned*-output form (next test) is trustworthy.

    If a future torch makes naive feedback silently CORRECT, this fails loudly -- a signal that
    the hazard behind the commented-out `@th.compile` may be gone; revisit that note."""
    dev = th.device("cuda")
    h = _make_helper_on(dev)

    def upd(sample, state):
        return h.update(sample, state, inplace=False)

    # eager reference
    ref = []
    st = h.reset_state(0.0)
    for k in range(1, 6):
        st = upd(_sample_on(dev, k), st)
        ref.append(st.clone())

    compiled_fn = th.compile(upd, mode="max-autotune", fullgraph=True)
    st = h.reset_state(0.0)
    try:
        diverged = False
        for k in range(1, 6):
            st = compiled_fn(_sample_on(dev, k), st)  # feed the static output back in
            if not th.equal(st, ref[k - 1]):
                diverged = True
    except RuntimeError as e:
        assert "CUDAGraph" in str(e), f"unexpected RuntimeError: {e}"
        return  # caught form -- unsafe as expected
    assert diverged, ("naive rolling feedback was silently correct under max-autotune cudagraphs; "
                      "the RobotVecEnv:2069 hazard may no longer apply on this torch")


@requires_cuda
def test_rolling_feedback_under_cudagraphs_is_correct_when_output_cloned():
    """Cloning the compiled output before feeding it back detaches it from the cudagraph's
    static pool -- the update then compiles under max-autotune *and* matches eager step for
    step. (`cudagraph_mark_step_begin()` alone is NOT enough; the fed-back input is the one
    overwritten -- it must be cloned.)"""
    dev = th.device("cuda")
    h = _make_helper_on(dev)

    def upd(sample, state):
        return h.update(sample, state, inplace=False)

    # eager reference
    ref = []
    st = h.reset_state(0.0)
    for k in range(1, 5):
        st = upd(_sample_on(dev, k), st)
        ref.append(st.clone())

    compiled_fn = th.compile(upd, mode="max-autotune", fullgraph=True)
    st = h.reset_state(0.0)
    for k in range(1, 5):
        st = compiled_fn(_sample_on(dev, k), st).clone()
        assert th.equal(st, ref[k - 1])
