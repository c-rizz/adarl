"""Adapter-agnostic compliance test for vectorized simulation adapters.

Runs a series of behavioral checks against any BaseVecSimulationAdapter, with additional
checks for joint-impedance adapters (BaseVecJointImpedanceAdapter). The checks use the
cartpole+ball models and validate the adapter interface contract: shapes/devices, direct
state setting (masked and unmasked), basic physics sanity (gravity, ballistic motion),
time bookkeeping, impedance command semantics (immediate, queued, delayed, masked),
step statistics and world resetting.

Usage:
    python test_vec_adapter_compliance.py --backend genesis --vec-size 4
    python test_vec_adapter_compliance.py --backend mujoco
    python test_vec_adapter_compliance.py --backend all
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import torch as th

import adarl.utils.utils
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter

CART_JOINT = ("cartpole", "cartpole_joint")
FOOT_JOINT = ("cartpole", "foot_joint")
BAR_LINK = ("cartpole", "bar_link")
BASE_LINK = ("cartpole", "base_link")
BALL_LINK = ("ball", "ball")


class ComplianceContext:
    def __init__(self, adapter: BaseVecSimulationAdapter):
        self.adapter = adapter
        self.vec = adapter.vec_size()
        self.dev = adapter.output_th_device()
        self.passed: list[str] = []
        self.skipped: list[str] = []
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, msg: str = ""):
        if condition:
            self.passed.append(name)
            print(f"  [PASS] {name}")
        else:
            self.failed.append(f"{name}: {msg}")
            print(f"  [FAIL] {name}: {msg}")

    def skip(self, name: str, reason: str):
        self.skipped.append(f"{name}: {reason}")
        print(f"  [SKIP] {name}: {reason}")

    def thtens(self, data):
        return th.as_tensor(data, device=self.dev, dtype=th.float32)


def _spawn_defs() -> list[ModelSpawnDef]:
    return [ModelSpawnDef(name="cartpole",
                          definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl", "models/cartpole_v0.urdf.xacro")).read_text(),
                          format="urdf.xacro",
                          pose=None,
                          kwargs={}),
            ModelSpawnDef(name="ball",
                          definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl", "models/ball.urdf")).read_text(),
                          format="urdf",
                          pose=None,
                          kwargs={})]


def _set_pole_state(ctx: ComplianceContext, pos: float, vel: float = 0.0, vec_mask: th.Tensor | None = None):
    state = ctx.thtens([pos, vel, 0.0]).view(1, 1, 3).expand(ctx.vec, 1, 3)
    ctx.adapter.setJointsStateDirect([CART_JOINT], state, vec_mask=vec_mask)


def _pole_pos(ctx: ComplianceContext) -> th.Tensor:
    return ctx.adapter.getJointsState([CART_JOINT])[:, 0, 0]


# =====================================================================================
#                                  generic checks
# =====================================================================================

def check_introspection(ctx: ComplianceContext):
    joints = [tuple(j) for j in ctx.adapter.get_detected_joints()]
    links = [tuple(l) for l in ctx.adapter.get_detected_links()]
    ctx.check("introspection.joints", CART_JOINT in joints and FOOT_JOINT in joints,
              f"expected {CART_JOINT} and {FOOT_JOINT} in {joints}")
    ctx.check("introspection.links", BAR_LINK in links and BALL_LINK in links,
              f"expected {BAR_LINK} and {BALL_LINK} in {links}")


def check_state_shapes(ctx: ComplianceContext):
    js = ctx.adapter.getJointsState([CART_JOINT, FOOT_JOINT])
    ctx.check("shapes.joints_state", tuple(js.shape) == (ctx.vec, 2, 3), f"got {tuple(js.shape)}")
    ls = ctx.adapter.getLinksState([BAR_LINK, BALL_LINK])
    ctx.check("shapes.links_state", tuple(ls.shape) == (ctx.vec, 2, 13), f"got {tuple(ls.shape)}")
    quat_norms = ls[:, :, 3:7].norm(dim=-1)
    ctx.check("shapes.quat_normalized", bool(((quat_norms - 1.0).abs() < 1e-3).all()),
              f"quaternion norms: {quat_norms.flatten().tolist()}")
    js_mon = ctx.adapter.getJointsState()
    ctx.check("shapes.monitored_joints_state", tuple(js_mon.shape) == (ctx.vec, 2, 3), f"got {tuple(js_mon.shape)}")
    try:
        ejs = ctx.adapter.getExtendedJointsState([CART_JOINT])
        ctx.check("shapes.extended_joints_state", tuple(ejs.shape) == (ctx.vec, 1, 5), f"got {tuple(ejs.shape)}")
    except NotImplementedError:
        ctx.skip("shapes.extended_joints_state", "not implemented by this adapter")


def check_set_joint_state(ctx: ComplianceContext):
    _set_pole_state(ctx, 0.3)
    pos = _pole_pos(ctx)
    ctx.check("set_joint_state.readback", bool(((pos - 0.3).abs() < 5e-3).all()), f"got {pos.tolist()}")
    if ctx.vec > 1:
        mask = th.zeros((ctx.vec,), dtype=th.bool, device=ctx.dev)
        mask[::2] = True
        _set_pole_state(ctx, 0.1, vec_mask=mask)
        pos = _pole_pos(ctx)
        ok_masked = bool(((pos[mask] - 0.1).abs() < 5e-3).all())
        ok_unmasked = bool(((pos[~mask] - 0.3).abs() < 5e-3).all())
        ctx.check("set_joint_state.masked", ok_masked and ok_unmasked, f"got {pos.tolist()}")
    else:
        ctx.skip("set_joint_state.masked", "vec_size==1")


def check_set_link_state_and_gravity(ctx: ComplianceContext):
    state = ctx.thtens([0.0, 3.0, 5.0,
                        0.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 13).expand(ctx.vec, 1, 13)
    ctx.adapter.setLinksStateDirect([BALL_LINK], state)
    ls = ctx.adapter.getLinksState([BALL_LINK])[:, 0]
    ctx.check("set_link_state.readback_pose", bool(((ls[:, 0:3] - ctx.thtens([0., 3., 5.])).abs() < 1e-2).all()),
              f"got positions {ls[:, 0:3].tolist()}")
    ctx.check("set_link_state.readback_vel", bool(((ls[:, 7:10] - ctx.thtens([1., 0., 0.])).abs() < 1e-2).all()),
              f"got velocities {ls[:, 7:10].tolist()}")
    elapsed = 0.0
    while elapsed < 0.25:
        elapsed += ctx.adapter.step()
    ls = ctx.adapter.getLinksState([BALL_LINK])[:, 0]
    expected_vz = -9.81 * elapsed
    ctx.check("gravity.ballistic_vz", bool(((ls[:, 9] - expected_vz).abs() < 0.15 * abs(expected_vz) + 0.05).all()),
              f"after {elapsed:.3f}s expected vz~{expected_vz:.3f}, got {ls[:, 9].tolist()}")
    ctx.check("gravity.ballistic_vx", bool(((ls[:, 7] - 1.0).abs() < 0.15).all()),
              f"expected vx~1.0, got {ls[:, 7].tolist()}")


def check_time(ctx: ComplianceContext):
    t0 = float(ctx.adapter.getEnvTimeFromStartup())
    stepped = 0.0
    for _ in range(5):
        stepped += ctx.adapter.step()
    t1 = float(ctx.adapter.getEnvTimeFromStartup())
    ctx.check("time.step_advances_env_time", abs((t1 - t0) - stepped) < 1e-5,
              f"env time advanced {t1-t0:.6f}, steps returned {stepped:.6f}")
    ctx.check("time.step_duration_positive", stepped > 0.0, f"got {stepped}")


def check_effort(ctx: ComplianceContext):
    if not isinstance(ctx.adapter, BaseVecJointEffortAdapter) or isinstance(ctx.adapter, BaseVecJointImpedanceAdapter):
        ctx.skip("effort.moves_joint", "not a pure effort adapter")
        return
    _set_pole_state(ctx, 0.0)
    ctx.adapter.setJointsEffortCommand([CART_JOINT], th.full((ctx.vec, 1), 2.0, device=ctx.dev))
    elapsed = 0.0
    while elapsed < 0.2:
        elapsed += ctx.adapter.step()
    vel = ctx.adapter.getJointsState([CART_JOINT])[:, 0, 1]
    ctx.check("effort.moves_joint", bool((vel > 0.01).all()), f"velocities after +2Nm: {vel.tolist()}")
    ctx.adapter.setJointsEffortCommand([CART_JOINT], th.zeros((ctx.vec, 1), device=ctx.dev))


# =====================================================================================
#                                 impedance checks
# =====================================================================================

def _jimp_cmd(ctx: ComplianceContext, p=0.0, v=0.0, e=0.0, kp=0.0, kd=0.0) -> th.Tensor:
    return ctx.thtens([p, v, e, kp, kd]).view(1, 1, 5).expand(ctx.vec, 1, 5)


def check_impedance(ctx: ComplianceContext):
    adapter = ctx.adapter
    if not isinstance(adapter, BaseVecJointImpedanceAdapter):
        ctx.skip("impedance.*", "not an impedance adapter")
        return
    step_len = float(adapter.step())

    adapter.set_impedance_controlled_joints([CART_JOINT])
    ctx.check("impedance.controlled_joints_roundtrip",
              [tuple(j) for j in adapter.get_impedance_controlled_joints()] == [CART_JOINT],
              f"got {adapter.get_impedance_controlled_joints()}")

    cp = adapter.control_period()
    ctx.check("impedance.control_period", isinstance(cp, th.Tensor) and float(cp) > 0.0, f"got {cp}")

    # command roundtrip
    cmd = _jimp_cmd(ctx, p=0.1, v=0.2, e=0.3, kp=40.0, kd=5.0)
    adapter.set_current_joint_impedance_command(cmd)
    got = adapter.get_current_joint_impedance_command()
    ctx.check("impedance.current_command_roundtrip", bool(((got - cmd.to(got.device)).abs() < 1e-4).all()),
              f"sent {cmd[0,0].tolist()}, got {got[0,0].tolist()}")

    # position holding: perturb the pole, command it back to zero
    adapter.reset_joint_impedances_commands()
    _set_pole_state(ctx, 0.35)
    adapter.set_current_joint_impedance_command(_jimp_cmd(ctx, kp=200.0, kd=10.0))
    elapsed = 0.0
    while elapsed < 1.5:
        elapsed += adapter.step()
    pos = _pole_pos(ctx)
    ctx.check("impedance.position_hold", bool((pos.abs() < 0.15).all()),
              f"pole at {pos.tolist()} after {elapsed:.2f}s of kp=200 hold towards 0")

    # effort feedforward through the impedance command
    adapter.reset_joint_impedances_commands()
    _set_pole_state(ctx, 0.0)
    adapter.set_current_joint_impedance_command(_jimp_cmd(ctx, e=2.0))
    elapsed = 0.0
    while elapsed < 0.25:
        elapsed += adapter.step()
    pos = _pole_pos(ctx)
    ctx.check("impedance.effort_feedforward", bool((pos > 0.005).all()),
              f"pole at {pos.tolist()} after {elapsed:.2f}s of +2Nm feedforward")

    # delayed command
    adapter.reset_joint_impedances_commands()
    _set_pole_state(ctx, 0.35)
    delay = 2.5 * step_len
    adapter.setJointsImpedanceCommand(_jimp_cmd(ctx, kp=200.0, kd=10.0), delay_sec=delay)
    adapter.step()
    got = adapter.get_current_joint_impedance_command()
    ctx.check("impedance.delayed_command_not_yet_applied", bool((got[:, 0, 3].abs() < 1e-4).all()),
              f"kp after 1 step (delay {delay:.4f}s): {got[:, 0, 3].tolist()}")
    for _ in range(3):
        adapter.step()
    got = adapter.get_current_joint_impedance_command()
    ctx.check("impedance.delayed_command_applied", bool(((got[:, 0, 3] - 200.0).abs() < 1e-3).all()),
              f"kp after 4 steps: {got[:, 0, 3].tolist()}")

    # masked command
    if ctx.vec > 1:
        adapter.reset_joint_impedances_commands()
        mask = th.zeros((ctx.vec,), dtype=th.bool, device=ctx.dev)
        mask[::2] = True
        adapter.setJointsImpedanceCommand(_jimp_cmd(ctx, kp=123.0), vec_mask=mask)
        adapter.step()
        got = adapter.get_current_joint_impedance_command()
        kp = got[:, 0, 3]
        mask_out = mask.to(kp.device)
        ok = bool(((kp[mask_out] - 123.0).abs() < 1e-3).all()) and bool((kp[~mask_out].abs() < 1e-3).all())
        ctx.check("impedance.masked_command", ok, f"kp per env: {kp.tolist()}, mask: {mask.tolist()}")
    else:
        ctx.skip("impedance.masked_command", "vec_size==1")

    # reference filter setting
    try:
        adapter.set_reference_filter(th.full((ctx.vec,), 20.0, device=ctx.dev))
        ctx.check("impedance.set_reference_filter", True)
    except NotImplementedError:
        ctx.skip("impedance.set_reference_filter", "not implemented by this adapter")

    adapter.reset_joint_impedances_commands()


# =====================================================================================
#                              stats and reset checks
# =====================================================================================

def check_step_stats(ctx: ComplianceContext):
    try:
        ctx.adapter.step()
        stats = ctx.adapter.get_joints_state_step_stats()
        ok_shape = tuple(stats.shape) == (ctx.vec, 4, 2, 4)
        ctx.check("stats.joints_shape", ok_shape, f"got {tuple(stats.shape)}")
        if ok_shape:
            jmin, jmax, javg = stats[:, 0], stats[:, 1], stats[:, 2]
            ctx.check("stats.joints_minavgmax", bool((jmin <= javg + 1e-5).all() and (javg <= jmax + 1e-5).all()),
                      "expected min <= avg <= max")
        lstats = ctx.adapter.get_links_state_step_stats()
        ctx.check("stats.links_shape", tuple(lstats.shape) == (ctx.vec, 4, 2, 6), f"got {tuple(lstats.shape)}")
    except NotImplementedError:
        ctx.skip("stats.*", "not implemented by this adapter")


def check_reset_world(ctx: ComplianceContext):
    pre = ctx.adapter.getLinksState([BALL_LINK])[:, 0, 0:3]
    ctx.adapter.resetWorld()
    post = ctx.adapter.getLinksState([BALL_LINK])[:, 0, 0:3]
    moved_back = bool((post[:, 1].abs() < 0.5).all())  # the ball was sent to y=3.0 earlier, reset should bring it back
    ctx.check("reset.world", moved_back, f"ball position before reset {pre.tolist()}, after {post.tolist()}")


# =====================================================================================
#                                      runner
# =====================================================================================

def run_vec_adapter_compliance_test(adapter: BaseVecSimulationAdapter) -> ComplianceContext:
    ctx = ComplianceContext(adapter)
    print(f"=== Compliance test for {type(adapter).__name__} (vec_size={ctx.vec}, device={ctx.dev}) ===")
    adapter.build_scenario(_spawn_defs())
    adapter.set_monitored_joints([CART_JOINT, FOOT_JOINT])
    adapter.set_monitored_links([BAR_LINK, BASE_LINK])
    adapter.startup()

    check_introspection(ctx)
    check_state_shapes(ctx)
    check_set_joint_state(ctx)
    check_set_link_state_and_gravity(ctx)
    check_time(ctx)
    check_effort(ctx)
    check_impedance(ctx)
    check_step_stats(ctx)
    check_reset_world(ctx)

    adapter.destroy_scenario()
    print(f"=== {type(adapter).__name__}: {len(ctx.passed)} passed, {len(ctx.skipped)} skipped, {len(ctx.failed)} failed ===")
    for f in ctx.failed:
        print(f"    FAILED: {f}")
    return ctx


def _build_genesis_adapter(vec_size: int):
    from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter
    device = th.device("cuda", 0) if th.cuda.is_available() else th.device("cpu")
    return GenesisJointImpedanceAdapter(vec_size=vec_size,
                                        output_th_device=device,
                                        sim_step_dt=1 / 256,
                                        step_length_sec=12 / 256,
                                        enable_rendering=False,
                                        reference_filter_mode="none")


def _build_mujoco_adapter():
    from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
    return MujocoJointImpedanceAdapter(vec_size=1,
                                       sim_step_dt=1 / 512,
                                       step_length_sec=24 / 512,
                                       output_th_device=th.device("cpu"),
                                       reference_filter_mode="none")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="all", choices=["genesis", "mujoco", "all"])
    parser.add_argument("--vec-size", type=int, default=4)
    args = parser.parse_args()
    results = []
    if args.backend in ("mujoco", "all"):
        results.append(run_vec_adapter_compliance_test(_build_mujoco_adapter()))
    if args.backend in ("genesis", "all"):
        results.append(run_vec_adapter_compliance_test(_build_genesis_adapter(args.vec_size)))
    if any(len(r.failed) > 0 for r in results):
        raise SystemExit(1)
