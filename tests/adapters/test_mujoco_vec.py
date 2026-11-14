"""MujocoAdapter with several environments (mujoco.batch): each environment steps exactly as a single-environment
adapter given that environment's inputs, with per-environment states, masks, efforts and durations; images have the
per-environment shapes."""
import numpy as np
import pytest
import torch as th

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from _backends import _log_folder

pytestmark = [pytest.mark.mujoco]

VEC = 3
SCENE = """<mujoco>
  <worldbody>
    <camera name="cam" pos="1.2 -1.2 1.0" xyaxes="1 1 0 -0.3 0.3 0.9" fovy="50" resolution="32 24"/>
    <body name="cube" pos="0.4 0 0.1">
      <inertial pos="0 0 0" mass="0.1" diaginertia="6e-5 6e-5 6e-5"/>
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.03 0.03 0.03" rgba="1 0 0 1"/>
    </body>
    <body name="marker" mocap="true" pos="0 0.5 0.5">
      <geom name="marker_geom" type="sphere" size="0.03" contype="0" conaffinity="0"/>
    </body>
    <body name="base" pos="0 0 0.3">
      <geom name="base_geom" type="cylinder" size="0.05 0.05"/>
      <body name="link1">
        <inertial pos="0.15 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body name="link2" pos="0.3 0 0">
          <inertial pos="0.15 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
JOINTS = [("scene", "j1"), ("scene", "j2")]
LINKS = [("scene", "link2"), ("scene", "cube")]
PAIRS = [(("scene", "cube"), ("ground", "ground_link")), (("scene", "link2"), ("scene", "cube"))]


def _adapter(vec_size, **kwargs):
    from adarl.adapters.MujocoAdapter import MujocoAdapter
    a = MujocoAdapter(vec_size=vec_size, sim_step_dt=1/1024, step_length_sec=16/1024, log_folder=_log_folder(), **kwargs)
    a.set_monitored_collision_pairs(PAIRS)
    a.build_scenario([ModelSpawnDef(name="scene", definition_string=SCENE, format="mjcf", pose=None, kwargs={})])
    a.startup()
    a.set_monitored_joints(JOINTS)
    a.set_monitored_links(LINKS)
    return a


def _inputs(env):
    g = th.Generator().manual_seed(env)
    joint_states = th.cat([th.rand((1, 2, 1), generator=g)*2-1, th.zeros((1, 2, 2))], dim=2)
    cube = th.zeros((1, 1, 13))
    cube[0, 0, 0:3] = th.tensor([0.4, 0.1*env, 0.2])
    cube[0, 0, 6] = 1.0
    marker = th.zeros((1, 1, 13))
    marker[0, 0, 0:3] = th.tensor([0.1*env, 0.4, 0.6])
    marker[0, 0, 6] = 1.0
    efforts = [th.rand((1, 2), generator=g)*4-2 for _ in range(6)]
    return joint_states, th.cat([cube, marker], dim=1), efforts


def _outputs(a):
    return {"joints" : a.getJointsState(),
            "extended" : a.getExtendedJointsState(),
            "links" : a.getLinksState(),
            "links com" : a.getLinksState(use_com_pose=True),
            "joint stats" : a.get_joints_state_step_stats_extended(),
            "link stats" : a.get_links_state_step_stats(),
            "time" : a.getEnvTimeFromStartup(),
            "collisions" : a.check_colliding_links(),
            "accelerations" : a.get_local_link_linear_acceleration()}
    # Images are not compared: MuJoCo's OpenGL renderer does not give the same image of the same state every time
    # (it depends on what the renderer drew before, and varies between runs), with or without the batch.


def test_envs_step_as_single_env_adapters():
    batched = _adapter(VEC, sim_threads=2)
    singles = [_adapter(1) for _ in range(VEC)]
    try:
        inputs = [_inputs(env) for env in range(VEC)]
        cube_and_marker = [("scene", "cube"), ("scene", "marker")]
        batched.resetWorld()
        batched.setJointsStateDirect(JOINTS, th.cat([i[0] for i in inputs]))
        batched.setLinksStateDirect(cube_and_marker, th.cat([i[1] for i in inputs]))
        for env, s in enumerate(singles):
            s.resetWorld()
            s.setJointsStateDirect(JOINTS, inputs[env][0])
            s.setLinksStateDirect(cube_and_marker, inputs[env][1])
        for k in range(6):
            batched.setJointsEffortCommand(JOINTS, th.cat([i[2][k] for i in inputs]))
            for env, s in enumerate(singles):
                s.setJointsEffortCommand(JOINTS, inputs[env][2][k])
            durations = batched.step()
            single_durations = th.cat([s.step() for s in singles])
            assert th.equal(durations, single_durations)
            outs = _outputs(batched)
            single_outs = [_outputs(s) for s in singles]
            for name, value in outs.items():
                expected = th.cat([so[name] for so in single_outs])
                assert th.equal(value, expected), f"step {k}: {name} differs"
    finally:
        for a in [batched] + singles:
            a.destroy_scenario()


def _impedance_adapter(vec_size):
    from adarl.adapters.MujocoJointImpedanceAdapter import MujocoJointImpedanceAdapter
    a = MujocoJointImpedanceAdapter(vec_size=vec_size, sim_step_dt=1/1024, step_length_sec=16/1024, log_folder=_log_folder(),
                                    reference_filter_mode="second_order", reference_filter_cutoff_frequency=30.0,
                                    joint_state_filter_decimation_time=0.01, use_gains_filter=True,
                                    sim_threads=2 if vec_size > 1 else None)
    a.set_monitored_collision_pairs(PAIRS)
    a.build_scenario([ModelSpawnDef(name="scene", definition_string=SCENE, format="mjcf", pose=None, kwargs={})])
    a.startup()
    a.set_monitored_joints(JOINTS)
    a.set_monitored_links(LINKS)
    a.set_impedance_controlled_joints(JOINTS)
    return a


def test_impedance_envs_as_single_env_adapters():
    """Per-environment commands, delays, filter cutoffs (one turned off), masks and durations: each environment of
    the batch controls and steps exactly as a single-environment adapter given the same calls."""
    batched = _impedance_adapter(VEC)
    singles = [_impedance_adapter(1) for _ in range(VEC)]
    gen = th.Generator().manual_seed(1)
    try:
        batched.resetWorld()
        for s in singles:
            s.resetWorld()
        cutoffs = th.tensor([15.0, 0.0, 40.0]) # env 1 without reference filter
        batched.set_reference_filter(cutoffs)
        for env, s in enumerate(singles):
            s.set_reference_filter(cutoffs[env:env+1])
        for k in range(10):
            cmd = th.zeros((VEC, 2, 5))
            cmd[:, :, 0] = th.rand((VEC, 2), generator=gen)*2-1
            cmd[:, :, 3] = 40.0 + 20*th.rand((VEC, 1), generator=gen)
            cmd[:, :, 4] = 3.0
            delays = th.rand((VEC,), generator=gen)*0.03
            mask = th.rand((VEC,), generator=gen) > 0.3   # some environments get no command this step
            if k == 4:
                batched.set_current_joint_impedance_command(cmd, vec_mask=mask)
            else:
                batched.setJointsImpedanceCommand(cmd, delay_sec=delays, vec_mask=mask)
            for env, s in enumerate(singles):
                if not mask[env]:
                    continue
                if k == 4:
                    s.set_current_joint_impedance_command(cmd[env:env+1])
                else:
                    s.setJointsImpedanceCommand(cmd[env:env+1], delay_sec=delays[env:env+1])
            if k % 3 == 2: # per-environment durations: env 2 waits, env 0 runs half
                durations = th.tensor([8/1024, 16/1024, 0.0])
                ran = batched.run(durations)
                single_ran = th.cat([s.run(float(durations[env])) for env, s in enumerate(singles)])
            else:
                ran = batched.step()
                single_ran = th.cat([s.step() for s in singles])
            assert th.equal(ran, single_ran)
            outs = _outputs(batched)
            outs["command"] = batched.get_current_joint_impedance_command()
            single_outs = [{**_outputs(s), "command" : s.get_current_joint_impedance_command()} for s in singles]
            for name, value in outs.items():
                expected = th.cat([so[name] for so in single_outs])
                assert th.equal(value, expected), f"step {k}: {name} differs"
    finally:
        for a in [batched] + singles:
            a.destroy_scenario()


def test_masks_and_per_env_durations():
    a = _adapter(VEC)
    try:
        a.resetWorld()
        before = a.getJointsState().clone()
        # set the joints of env 1 only
        states = th.zeros((VEC, 2, 3))
        states[:, :, 0] = 0.5
        a.setJointsStateDirect(JOINTS, states, vec_mask=th.tensor([False, True, False]))
        after = a.getJointsState()
        assert th.equal(after[[0, 2]], before[[0, 2]]) and th.allclose(after[1, :, 0], th.tensor([0.5, 0.5]))
        # per-environment durations: each env runs for its own time and the others wait
        durations = a.run(th.tensor([4/1024, 8/1024, 0.0]))
        assert th.allclose(durations, th.tensor([4/1024, 8/1024, 0.0]))
        assert th.allclose(a.getEnvTimeFromStartup(), th.tensor([4/1024, 8/1024, 0.0]))
        # masked rendering: only the selected environments, unless the shapes are fixed
        r = a.get_camera_images(["cam"], rgb=True, depth=True, vec_mask=th.tensor([True, False, True]))
        assert r.rgb[0].shape == (2, 24, 32, 3) and r.times.shape == (2, 1)
        r = a.get_camera_images(["cam"], rgb=True, depth=True, vec_mask=th.tensor([True, False, True]),
                                use_fixed_shapes=True)
        assert r.rgb[0].shape == (3, 24, 32, 3) and r.rgb[0][1].sum() == 0 and r.rgb[0][0].sum() > 0
        # efforts of env 2 only
        a.setJointsEffortCommand(JOINTS, th.ones((VEC, 2)), vec_mask=th.tensor([False, False, True]))
        assert np.array_equal(a._requested_qfrc_applied[:, a._mj_model.jnt_dofadr[[a._jname2jid[j] for j in JOINTS]]],
                              np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))
        # a fixed body is shared by all the environments: it can only move in all of them, to the same pose
        base = th.zeros((VEC, 1, 13))
        base[:, 0, 6] = 1.0
        base[:, 0, 2] = 0.3
        a.setLinksStateDirect([("scene", "base")], base)
        base[1, 0, 2] = 0.4
        with pytest.raises(NotImplementedError, match="fixed bodies"):
            a.setLinksStateDirect([("scene", "base")], base)
    finally:
        a.destroy_scenario()
