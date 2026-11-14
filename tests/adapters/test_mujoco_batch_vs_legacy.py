"""TEMPORARY: MujocoAdapter on mujoco.batch steps exactly as the single-MjData MujocoAdapter it replaced.

Drives MujocoJointImpedanceAdapter and MujocoJointImpedanceAdapterLegacy (the pre-batch copies) with the same
calls and compares everything they return after every step, with exact equality. Delete together with
MujocoAdapterLegacy.py and MujocoJointImpedanceAdapterLegacy.py once the batch version is accepted."""
from pathlib import Path

import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import _log_folder

pytestmark = [pytest.mark.mujoco, pytest.mark.slow]

# A three-joint arm on a fixed base, a free cube on a fixed table, a mocap marker and a camera
ARM_SCENE = """<mujoco>
  <worldbody>
    <camera name="cam" pos="1.5 -1.5 1.2" xyaxes="1 1 0 -0.3 0.3 0.9" fovy="50" resolution="48 36"/>
    <body name="table" pos="0.6 0 0.1">
      <geom name="table_geom" type="box" size="0.3 0.3 0.1" rgba=".6 .4 .2 1"/>
    </body>
    <body name="marker" mocap="true" pos="0 0.5 0.5">
      <geom name="marker_geom" type="sphere" size="0.03" contype="0" conaffinity="0" rgba="0 1 0 1"/>
    </body>
    <body name="cube" pos="0.45 0 0.25">
      <inertial pos="0 0 0" mass="0.1" diaginertia="6e-5 6e-5 6e-5"/>
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.03 0.03 0.03" rgba="1 0 0 1"/>
    </body>
    <body name="base" pos="0 0 0.3">
      <geom name="base_geom" type="cylinder" size="0.05 0.05"/>
      <body name="link1">
        <inertial pos="0.15 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body name="link2" pos="0.3 0 0">
          <inertial pos="0.15 0 0" mass="0.8" diaginertia="0.008 0.008 0.008"/>
          <joint name="j2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
          <body name="link3" pos="0.3 0 0">
            <inertial pos="0.07 0 0" mass="0.3" diaginertia="0.002 0.002 0.002"/>
            <joint name="j3" type="hinge" axis="0 1 0" range="-2 2"/>
            <geom name="tip_geom" type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def _l(name):
    return ("scene", name)


def _fix_legacy_pveae(adapter):
    """The legacy getExtendedJointsState read qacc at the joints' qpos addresses instead of their dof addresses: the
    wrong dof (or out of bounds) once a free joint precedes them. The batch version reads the dof addresses; give
    the legacy one the same fix, so that the rest of what it returns is compared."""
    import numpy as np
    def read_joints_pveae(jids):
        m, d = adapter._mj_model, adapter._mj_data
        qpos_adrs, qvel_adrs = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
        return np.stack([d.qpos[qpos_adrs], d.qvel[qvel_adrs], d.qfrc_applied[qvel_adrs], d.qacc[qvel_adrs],
                         d.qfrc_smooth[qvel_adrs] + d.qfrc_constraint[qvel_adrs]], axis=-1)
    adapter._read_joints_pveae = read_joints_pveae


def _build(cls, models, imp_joints, monitored_links, collision_pairs, link_group_collisions, **kwargs):
    adapter = cls(vec_size=1,
                  sim_step_dt=1/1024,
                  step_length_sec=24/1024,
                  log_folder=_log_folder(),
                  reference_filter_mode="second_order",
                  reference_filter_cutoff_frequency=30.0,
                  joint_state_filter_decimation_time=0.01,
                  use_gains_filter=True,
                  **kwargs)
    adapter.set_monitored_collision_pairs(collision_pairs)
    adapter.build_scenario(models, default_link_group_collisions=link_group_collisions, add_ground=True)
    adapter.startup()
    adapter.set_monitored_joints(imp_joints)
    adapter.set_monitored_links(monitored_links)
    adapter.set_impedance_controlled_joints(imp_joints)
    if cls.__name__.endswith("Legacy"):
        _fix_legacy_pveae(adapter)
    return adapter


class _Comparison:
    def __init__(self):
        self.mismatches : list[str] = []
        self.compared = 0

    def __call__(self, step, name, new, legacy):
        self.compared += 1
        if isinstance(new, th.Tensor):
            same = isinstance(legacy, th.Tensor) and new.shape == legacy.shape and new.dtype == legacy.dtype and th.equal(new, legacy)
            if not same:
                detail = (f"shapes {tuple(new.shape)}/{tuple(legacy.shape)}, dtypes {new.dtype}/{legacy.dtype}"
                          if not isinstance(legacy, th.Tensor) or new.shape != legacy.shape or new.dtype != legacy.dtype
                          else f"max |diff| {(new.double()-legacy.double()).abs().max().item():.3e}")
                self.mismatches.append(f"step {step}: {name}: {detail}")
        elif new != legacy:
            self.mismatches.append(f"step {step}: {name}: {new} != {legacy}")


def _run_and_compare(adapters, imp_joints, monitored_links, init_joint_states, link_states, link_names,
                     num_steps, kp, kd, pos_spread, cameras=()):
    """Drive both adapters with the same calls; compare everything they return after every step."""
    compare = _Comparison()
    gen = th.Generator().manual_seed(0)
    nj = len(imp_joints)

    def both(fn):
        return [fn(a) for a in adapters]

    def init():
        for a in adapters:
            a.resetWorld()
            a.setJointsStateDirect(imp_joints, init_joint_states.clone())
            if link_states is not None:
                a.setLinksStateDirect(link_names, link_states.clone())

    init()
    for step in range(num_steps):
        cmd = th.zeros((1, nj, 5))
        cmd[0, :, 0] = init_joint_states[0, :, 0] + (th.rand((nj,), generator=gen)*2-1)*pos_spread
        cmd[0, :, 3] = kp
        cmd[0, :, 4] = kd
        delay = float(th.rand((), generator=gen))*0.03
        if step == 3:
            both(lambda a: a.set_current_joint_impedance_command(cmd.clone()))
        else:
            both(lambda a: a.setJointsImpedanceCommand(cmd.clone(), delay_sec=delay))
        if step == 12:
            both(lambda a: a.set_reference_filter(th.tensor([10.0])))
        if step == num_steps//2:
            init()
        durations = both(lambda a: a.step())
        checks = [("step duration", lambda a: durations[adapters.index(a)]),
                  ("getJointsState", lambda a: a.getJointsState()),
                  ("getExtendedJointsState", lambda a: a.getExtendedJointsState()),
                  ("getLinksState", lambda a: a.getLinksState()),
                  ("joint step stats", lambda a: a.get_joints_state_step_stats_extended()),
                  ("link step stats", lambda a: a.get_links_state_step_stats()),
                  ("getEnvTimeFromStartup", lambda a: a.getEnvTimeFromStartup()),
                  ("check_colliding_links", lambda a: a.check_colliding_links()),
                  ("link accelerations", lambda a: a.get_local_link_linear_acceleration()),
                  ("current impedance command", lambda a: a.get_current_joint_impedance_command())]
        if cameras and step % 5 == 0:
            checks.append(("rgb", lambda a: a.get_camera_images(list(cameras), rgb=True, depth=True).rgb[0]))
            checks.append(("depth", lambda a: a.get_camera_images(list(cameras), rgb=True, depth=True).depth[0]))
        for name, fn in checks:
            new, legacy = both(fn)
            compare(step, name, new, legacy)
    return compare


def test_arm_scene_identical():
    from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
    from adarl.adapters.MujocoJointImpedanceAdapterLegacy import MujocoJointImpedanceAdapterLegacy
    models = [ModelSpawnDef(name="scene", definition_string=ARM_SCENE, format="mjcf", pose=None, kwargs={})]
    imp_joints = [_l("j1"), _l("j2"), _l("j3")]
    arm = [_l("link1"), _l("link2"), _l("link3")]
    ground, cube, table = ("ground", "ground_link"), _l("cube"), _l("table")
    collision_pairs = [(_l("link3"), cube), (cube, table), (cube, ground), (_l("link2"), table)]
    link_group_collisions = ([(l, [cube, table, ground]) for l in arm] +
                             [(cube, arm + [table, ground]), (table, arm + [cube]), (ground, arm + [cube])])
    adapters = [_build(cls, models, imp_joints, arm + [cube], collision_pairs, link_group_collisions)
                for cls in (MujocoJointImpedanceAdapter, MujocoJointImpedanceAdapterLegacy)]
    try:
        init_joint_states = th.tensor([[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]]])
        # the free cube, the fixed table (moved in the model) and the mocap marker
        link_names = [cube, table, _l("marker")]
        link_states = th.zeros((1, 3, 13))
        link_states[0, :, 6] = 1.0 # identity quaternions, xyzw
        link_states[0, 0, 0:3] = th.tensor([0.45, 0.02, 0.25])
        link_states[0, 0, 7:13] = th.tensor([0.0, 0.1, 0.0, 0.0, 0.0, 0.5])
        link_states[0, 1, 0:3] = th.tensor([0.62, 0.0, 0.1])
        link_states[0, 2, 0:3] = th.tensor([0.1, 0.4, 0.6])
        compare = _run_and_compare(adapters, imp_joints, arm + [cube], init_joint_states, link_states, link_names,
                                   num_steps=40, kp=60.0, kd=4.0, pos_spread=1.0, cameras=("cam",))
        # the cube is hit: contacts and collisions are exercised
        collided = adapters[1].check_colliding_links()
        assert compare.compared > 400
        assert not compare.mismatches, "\n".join(compare.mismatches[:20])
    finally:
        for a in adapters:
            a.destroy_scenario()


def test_kyon_grasp_scene_identical():
    try:
        import pydescriptions  # noqa: F401
        import yaml
        from adarl.utils.utils import pkgutil_get_path
        from adarl.adapters.mjcf_export import load_scene
        scene_file = Path(pkgutil_get_path("adarl_envs", "ros/kyon_grasp_scene.yaml"))
        scene = yaml.safe_load(scene_file.read_text())
        # the environments' own description checkouts, with no download
        scene["pydescriptions"] = {"packages" : list(scene["pydescriptions"]["packages"]
                                                     if isinstance(scene["pydescriptions"], dict) else scene["pydescriptions"]),
                                   "root" : "pkg:adarl_envs/descriptions"}
        models, _, options, _ = load_scene(scene, scene_file.parent)
    except Exception as e:
        pytest.skip(f"the kyon grasp scene is not available: {type(e).__name__}: {e}")
    from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
    from adarl.adapters.MujocoJointImpedanceAdapterLegacy import MujocoJointImpedanceAdapterLegacy
    physics = dict(opt_preset=options["opt_preset"],
                   revolute_dof_armature_override=options["revolute_dof_armature_override"],
                   revolute_dof_damping_override=options["revolute_dof_damping_override"],
                   revolute_dof_frictionloss_override=options["revolute_dof_frictionloss_override"],
                   safe_revolute_dof_armature=options["safe_revolute_dof_armature"])
    probe = MujocoJointImpedanceAdapter(vec_size=1, log_folder=_log_folder())
    probe.build_scenario(models)
    import mujoco
    imp_joints = [j for j in probe.get_detected_joints()
                  if j[0] == "kyon" and probe._mj_model.jnt_type[probe._jname2jid[j]] == mujoco.mjtJoint.mjJNT_HINGE]
    probe.destroy_scenario()
    feet = [("kyon", f"contact_{i}") for i in range(1, 5)]
    links = [("kyon", "pelvis")] + feet + [("cube", "cube")]
    ground = ("ground", "ground_link")
    collision_pairs = [(f, ground) for f in feet] + [(("cube", "cube"), ground)]
    link_group_collisions = [("kyon", [ground, ("cube", "cube")]), (("cube", "cube"), ["kyon", ground]),
                             (ground, ["kyon", ("cube", "cube")])]
    adapters = [_build(cls, models, imp_joints, links, collision_pairs, link_group_collisions, **physics)
                for cls in (MujocoJointImpedanceAdapter, MujocoJointImpedanceAdapterLegacy)]
    try:
        init_joint_states = th.zeros((1, len(imp_joints), 3))
        compare = _run_and_compare(adapters, imp_joints, links, init_joint_states, None, None,
                                   num_steps=16, kp=300.0, kd=10.0, pos_spread=0.3)
        assert compare.compared > 150
        assert not compare.mismatches, "\n".join(compare.mismatches[:20])
    finally:
        for a in adapters:
            a.destroy_scenario()
