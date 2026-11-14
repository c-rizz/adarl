"""Per-joint revolute dof overrides: a float for all joints, or a dict with a None-key default."""
import mujoco
import pytest

from adarl.adapters.mujoco_utils import apply_dof_overrides, apply_dof_overrides_to_spec, resolve_dof_override

MODEL = """<mujoco>
  <worldbody>
    <body name="base">
      <geom type="sphere" size="0.1"/>
      <body name="b1"><joint name="robot#j1" type="hinge" armature="1" damping="1" frictionloss="1"/><geom type="sphere" size="0.05"/></body>
      <body name="b2"><joint name="robot#j2" type="hinge" armature="1" damping="1" frictionloss="1"/><geom type="sphere" size="0.05"/></body>
      <body name="b3"><joint name="other#j1" type="hinge" armature="1" damping="1" frictionloss="1"/><geom type="sphere" size="0.05"/></body>
    </body>
  </worldbody>
</mujoco>"""


def test_resolve():
    assert resolve_dof_override(None, "robot#j1") is None
    assert resolve_dof_override(2.0, "robot#j1") == 2.0
    d = {("robot", "j1"): 3.0, "j2": 4.0, "other#j1": 5.0, None: 9.0}
    assert resolve_dof_override(d, "robot#j1") == 3.0  # (model, joint)
    assert resolve_dof_override(d, "robot#j2") == 4.0  # bare joint name
    assert resolve_dof_override(d, "other#j1") == 5.0  # full name wins over the default
    assert resolve_dof_override(d, "robot#j9") == 9.0  # default
    assert resolve_dof_override({"default": 7.0}, "robot#j9") == 7.0
    assert resolve_dof_override({("robot", "j1"): 3.0}, "robot#j2") is None  # no default: no override
    assert resolve_dof_override({None: 9.0, "j1": None}, "robot#j1") is None  # None value: explicitly no override


@pytest.mark.parametrize("on_spec", [True, False])
def test_per_joint_overrides(on_spec):
    kwargs = dict(revolute_dof_armature_override={("robot", "j1"): 0.5, None: 0.25},
                  revolute_dof_damping_override={("robot", "j1"): 2.0},
                  revolute_dof_frictionloss_override=3.0)
    spec = mujoco.MjSpec.from_string(MODEL)
    if on_spec:
        model = apply_dof_overrides_to_spec(spec, **kwargs).compile()
    else:
        model = apply_dof_overrides(spec.compile(), **kwargs)
    armature = [model.dof_armature[model.jnt_dofadr[model.joint(n).id]] for n in ("robot#j1", "robot#j2", "other#j1")]
    damping = [model.dof_damping[model.jnt_dofadr[model.joint(n).id]] for n in ("robot#j1", "robot#j2", "other#j1")]
    frictionloss = [model.dof_frictionloss[model.jnt_dofadr[model.joint(n).id]] for n in ("robot#j1", "robot#j2", "other#j1")]
    assert armature == pytest.approx([0.5, 0.25, 0.25])
    assert damping == pytest.approx([2.0, 1.0, 1.0])  # only robot#j1 listed, no default: others untouched
    assert frictionloss == pytest.approx([3.0, 3.0, 3.0])
