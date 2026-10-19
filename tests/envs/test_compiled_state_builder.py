"""Aliased views as inputs of a cudagraph-compiled function.

This is the shape of a bug that cost a day. RobotVecEnv._get_adapter_data_raw_mjx read a link's
13-vector and handed its slices on as separate fields:

    pos    = bodystate[:, 0:3]      quat   = bodystate[:, 3:7]
    linvel = bodystate[:, 7:10]     angvel = bodystate[:, 10:13]

all views of one tensor. _get_new_instantaneous_state, which consumes them, is compiled with
th_compile_ext(copy_outs=True, mode="max-autotune", fullgraph=True) -- i.e. with cudagraphs. In a real
training run on torch 2.10 the compiled function returned the *position* for the linear velocity, the
angular velocity and the height alike: the aliased inputs collapsed onto the base tensor's offset-0
view. Velocities read as metres of position, the recorded episodes were nonsense, and nothing caught
it: the adapter was correct, every uncompiled replay was correct, and no test went through the
compiled path.

It was a torch bug, not an adarl one: **torch 2.10 miscompiles that function**, reading the aliased
views wrongly, and **torch 2.14 does not**. Two fixes, either alone sufficient: clone the slices so
no two inputs alias (what _get_adapter_data_raw_mjx now does), or upgrade torch. Both were confirmed
by runs on the affected workstation that were identical apart from the change.

These tests pin the contract the clone relies on. They do NOT reproduce the failure on torch >= 2.14,
which is the whole point: the aliased case is only miscompiled by the older version, so a green run
here says the platform honours the contract, not that the env code is safe without the clone. Anyone
seeing the aliased test fail has found the 2.10 behaviour again.
"""
import dataclasses

import pytest
import torch as th

from adarl.utils.utils import th_compile_ext

VEC = 8


@dataclasses.dataclass
class _AdapterDataLike:
    """Shaped like RobotAdapterData: several tensors, in the real case aliasing one base."""
    pos_xy: th.Tensor
    ground_dist: th.Tensor
    quat: th.Tensor
    linvel: th.Tensor
    angvel: th.Tensor


class _BuilderLike:
    """A method, like RobotVecEnv._get_new_instantaneous_state, slicing per component into a dict."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @th_compile_ext(copy_outs=True, mode="max-autotune", fullgraph=True)
    def build(self, data: _AdapterDataLike):
        n = self.num_envs
        return {"extrinsic": {
                    "linvel_x": data.linvel[:, 0].view(n, 1), "linvel_y": data.linvel[:, 1].view(n, 1),
                    "linvel_z": data.linvel[:, 2].view(n, 1),
                    "angvel_x": data.angvel[:, 0].view(n, 1), "angvel_y": data.angvel[:, 1].view(n, 1),
                    "angvel_z": data.angvel[:, 2].view(n, 1),
                    "pos_x": data.pos_xy[:, 0].view(n, 1), "pos_y": data.pos_xy[:, 1].view(n, 1),
                    "height": data.ground_dist.view(n, 1),
                    "quat_w": data.quat[:, 3].view(n, 1)}}


@th_compile_ext(copy_outs=True, mode="max-autotune", fullgraph=True)
def _fan_out(a: th.Tensor, b: th.Tensor, c: th.Tensor, d: th.Tensor):
    """Scatter several inputs into a dict, the way the state builder does."""
    return {"a": a * 1.0, "b": b * 1.0, "c": c * 1.0, "d": d * 1.0}


def _reference(base: th.Tensor):
    return {"a": base[:, 0:3].clone(), "b": base[:, 3:7].clone(),
            "c": base[:, 7:10].clone(), "d": base[:, 10:13].clone()}


def _run_and_compare(inputs, expected, label):
    out = None
    # several calls: cudagraphs only engage after warmup, which is why short replays looked clean
    for i in range(6):
        out = _fan_out(inputs["a"], inputs["b"], inputs["c"], inputs["d"])
        th.cuda.synchronize()
        for k in expected:
            assert th.allclose(out[k].cpu(), expected[k].cpu(), atol=1e-6), (
                f"{label}, call {i}: output '{k}' is {out[k][0].tolist()} but its input was "
                f"{expected[k][0].tolist()}. A compiled function returned something other than what "
                f"it was given, so every value derived from it downstream is wrong.")


@pytest.mark.gpu
@pytest.mark.skipif(not th.cuda.is_available(), reason="needs CUDA for cudagraphs")
def test_compiled_function_returns_its_unaliased_inputs():
    """The baseline: separate tensors must survive the compiled call."""
    base = th.arange(VEC * 13, dtype=th.float32, device="cuda").reshape(VEC, 13)
    expected = _reference(base)
    inputs = {k: v.clone() for k, v in expected.items()}   # no aliasing at all
    _run_and_compare(inputs, expected, "unaliased inputs")


def _run_builder(base, alias: bool):
    """Feed the builder either views of one tensor (the real case) or private copies."""
    builder = _BuilderLike(VEC)
    take = (lambda t: t) if alias else (lambda t: t.clone())
    data = _AdapterDataLike(pos_xy=take(base[:, 0:2]), ground_dist=take(base[:, 2]),
                            quat=take(base[:, 3:7]), linvel=take(base[:, 7:10]),
                            angvel=take(base[:, 10:13]))
    out = None
    for _ in range(6):   # cudagraphs engage only after warmup
        out = builder.build(data)
        th.cuda.synchronize()
    ex = out["extrinsic"]
    return {k: float(v.reshape(-1)[0]) for k, v in ex.items()}


@pytest.mark.gpu
@pytest.mark.skipif(not th.cuda.is_available(), reason="needs CUDA for cudagraphs")
def test_builder_with_aliased_dataclass_fields_matches_its_inputs():
    """The real call site's shape: a dataclass of aliased views, sliced per component in a method.

    RobotVecEnv._get_adapter_data_raw_mjx used to pass views of one 13-vector this way, and the
    compiled builder returned the position for the velocities and the height. Neither this test nor
    the simpler one reproduces that, despite mirroring the structure -- dataclass input, bound method,
    per-component slicing inside the graph, nested dict out -- so the trigger is still unidentified.
    """
    base = th.arange(VEC * 13, dtype=th.float32, device="cuda").reshape(VEC, 13) * 0.01
    want = {"linvel_x": float(base[0, 7]), "linvel_y": float(base[0, 8]), "linvel_z": float(base[0, 9]),
            "angvel_x": float(base[0, 10]), "angvel_y": float(base[0, 11]), "angvel_z": float(base[0, 12]),
            "pos_x": float(base[0, 0]), "pos_y": float(base[0, 1]), "height": float(base[0, 2]),
            "quat_w": float(base[0, 6])}
    got = _run_builder(base, alias=True)
    assert got == pytest.approx(want, abs=1e-6), (
        f"with aliased dataclass fields the builder returned {got} instead of {want}; "
        f"cloning the fields gives {_run_builder(base, alias=False)}")


@pytest.mark.gpu
@pytest.mark.skipif(not th.cuda.is_available(), reason="needs CUDA for cudagraphs")
def test_compiled_function_returns_its_aliased_inputs():
    """The case that failed: every input a view of one base tensor.

    Known to pass on the affected machine, so aliasing alone is not the trigger. Kept because the
    contract is worth stating: code that hands slices of one tensor to a compiled function relies on
    it.
    """
    base = th.arange(VEC * 13, dtype=th.float32, device="cuda").reshape(VEC, 13)
    expected = _reference(base)
    inputs = {"a": base[:, 0:3], "b": base[:, 3:7],
              "c": base[:, 7:10], "d": base[:, 10:13]}      # views, all sharing one storage
    assert inputs["a"].data_ptr() == base.data_ptr(), "the first view must sit at the base's offset 0"
    _run_and_compare(inputs, expected, "aliased views of one tensor")
