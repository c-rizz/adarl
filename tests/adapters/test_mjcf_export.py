"""mjcf_export: aggregate_models' output written as MJCF, with models left unprefixed and collision filtering."""
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from adarl.adapters.mjcf_export import export_mjcf, load_scene
from adarl.adapters.mujoco_utils import aggregate_models

# A robot with its own default classes, as a URDF converted by MuJoCo has, and a visual-only geom
ROBOT = """<mujoco>
  <default>
    <geom rgba="1 0 0 1"/>
    <default class="link"><geom size="0.05"/></default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <geom name="badge" type="sphere" size="0.01" contype="0" conaffinity="0"/>
      <body name="arm" pos="0.2 0 0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom class="link" type="sphere"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

CUBE = """<mujoco>
  <worldbody>
    <body name="cube" pos="0.5 0 0.1">
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.0001 0.0001 0.0001"/>
      <freejoint name="free"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>"""

TOOL = """<mujoco>
  <worldbody>
    <body name="tool">
      <inertial pos="0 0 0" mass="0.05" diaginertia="0.0001 0.0001 0.0001"/>
      <geom type="capsule" size="0.01 0.05"/>
    </body>
  </worldbody>
</mujoco>"""


def _scene(robot_prefix=False):
    return {"models" : [{"name" : "robot", "definition" : ROBOT, "format" : "mjcf", "prefix" : robot_prefix},
                        {"name" : "cube", "definition" : CUBE, "format" : "mjcf"},
                        {"name" : "tool", "definition" : TOOL, "format" : "mjcf", "attachment_link" : ["robot", "arm"]}],
            "options" : {"add_ground" : True}}


@pytest.mark.parametrize("robot_prefix", [False, True])
def test_export_reloads_as_the_aggregated_model(robot_prefix):
    scene = _scene(robot_prefix)
    models, unprefixed, options, _ = load_scene(scene, Path("."))
    aggregated, _ = aggregate_models(models, unprefixed_models=frozenset(unprefixed), **options)
    reloaded = mujoco.MjModel.from_xml_string(export_mjcf(scene, Path(".")))
    for field in ("nq", "nv", "njnt", "nbody", "ngeom", "nlight"):
        assert getattr(reloaded, field) == getattr(aggregated, field), field
    for field in ("body_mass", "body_pos", "geom_size", "geom_rgba", "geom_contype", "geom_conaffinity", "jnt_range",
                  "dof_damping", "dof_armature"):
        np.testing.assert_allclose(getattr(reloaded, field), getattr(aggregated, field), err_msg=field)
    bodies = {reloaded.body(i).name for i in range(reloaded.nbody)}
    joints = {reloaded.joint(i).name for i in range(reloaded.njnt)}
    prefix = "robot#" if robot_prefix else ""
    assert {prefix + "base", prefix + "arm", "cube#cube", "tool#tool", "ground#ground_link"} <= bodies
    assert {prefix + "shoulder", "cube#free"} == joints
    # the tool hangs from the arm, and the robot's default classes still apply
    assert reloaded.body_parentid[reloaded.body("tool#tool").id] == reloaded.body(prefix + "arm").id
    sphere = reloaded.geom(reloaded.body_geomadr[reloaded.body(prefix + "arm").id])
    assert sphere.size[0] == pytest.approx(0.05) and tuple(sphere.rgba) == (1, 0, 0, 1)


def _can_collide(model, body_a, body_b):
    """Whether, by MuJoCo's contype/conaffinity rule, some geom of body_a can collide with some geom of body_b."""
    def geoms(body):
        b = model.body(body).id
        return range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b])
    return any((model.geom_contype[g1] & model.geom_conaffinity[g2]) or (model.geom_contype[g2] & model.geom_conaffinity[g1])
               for g1 in geoms(body_a) for g2 in geoms(body_b))


@pytest.mark.parametrize("robot_prefix", [False, True])
def test_collisions(robot_prefix):
    prefix = "robot#" if robot_prefix else ""
    scene = _scene(robot_prefix)
    # Links named both ways: [model, link] and a bare model name for all its links (ground, tool)
    scene["collisions"] = [{"link" : ["robot", "arm"], "collides_with" : [["cube", "cube"]]},
                           {"link" : "tool", "collides_with" : [["cube", "cube"], "ground"]},
                           {"link" : ["cube", "cube"], "collides_with" : ["ground", ["robot", "arm"], "tool"]},
                           {"link" : "ground", "collides_with" : [["cube", "cube"], "tool"]}]
    model = mujoco.MjModel.from_xml_string(export_mjcf(scene, Path(".")))
    arm, base, cube, tool, ground = prefix + "arm", prefix + "base", "cube#cube", "tool#tool", "ground#ground_link"
    assert _can_collide(model, arm, cube) and _can_collide(model, cube, ground) and _can_collide(model, tool, ground)
    assert _can_collide(model, tool, cube)
    assert not _can_collide(model, arm, ground) and not _can_collide(model, arm, tool)
    for other in (arm, cube, tool, ground): # not listed: collides with nothing
        assert not _can_collide(model, base, other)
    badge = model.geom(prefix + "badge").id # visual geoms stay visual
    assert model.geom_contype[badge] == 0 and model.geom_conaffinity[badge] == 0
    # without collisions the masks are the models' own
    plain = mujoco.MjModel.from_xml_string(export_mjcf(_scene(robot_prefix), Path(".")))
    assert _can_collide(plain, arm, ground) and _can_collide(plain, base, cube)


def test_collisions_errors():
    scene = _scene()
    scene["collisions"] = [{"link" : ["robot", "nope"], "collides_with" : ["ground"]}]
    with pytest.raises(RuntimeError, match="not present"):
        export_mjcf(scene, Path("."))
    scene = _scene()
    scene["models"][1]["prefix"] = False # two unprefixed models: their links cannot be told apart
    scene["collisions"] = [{"link" : ["robot", "arm"], "collides_with" : ["ground"]}]
    with pytest.raises(ValueError, match="at most one unprefixed model"):
        export_mjcf(scene, Path("."))
    for bad in ([{"link" : ["robot", "arm"]}], [{"link" : ["robot", "arm", "x"], "collides_with" : []}],
                [{"link" : "robot", "collides_with" : "ground"}], {"link" : "robot"}):
        with pytest.raises(ValueError):
            load_scene({**_scene(), "collisions" : bad}, Path("."))


@pytest.fixture
def fake_pydescriptions(tmp_path, monkeypatch):
    """A pydescriptions module resolving two packages to directories of tmp_path: 'parts' holds a box xacro that
    'robots' includes with $(find parts)."""
    import types
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "box.xacro").write_text("""<mujoco xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="box" params="size"><geom type="box" size="${size} ${size} ${size}"/></xacro:macro>
</mujoco>""")
    (tmp_path / "robots" / "mjcf").mkdir(parents=True)
    (tmp_path / "robots" / "mjcf" / "boxbot.mjcf.xacro").write_text("""<mujoco xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="size" default="0.1"/>
  <xacro:include filename="$(find parts)/box.xacro"/>
  <worldbody><body name="boxbot">
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <xacro:box size="$(arg size)"/>
  </body></worldbody>
</mujoco>""")
    calls = []
    def resolve(packages, root=None, refs=None):
        calls.append((list(packages), root, refs))
        return {p : str(tmp_path / p) for p in packages}
    monkeypatch.setitem(sys.modules, "pydescriptions", types.SimpleNamespace(resolve=resolve))
    return tmp_path, calls


def test_pydescriptions(fake_pydescriptions):
    tmp_path, calls = fake_pydescriptions
    model = {"name" : "boxbot", "file" : "pydescriptions://robots/mjcf/boxbot.mjcf.xacro", "kwargs" : {"size" : 0.2}}
    scene = {"pydescriptions" : ["robots", "parts"], "models" : [model], "options" : {"add_ground" : False}}
    reloaded = mujoco.MjModel.from_xml_string(export_mjcf(scene, tmp_path))
    assert reloaded.geom_size[reloaded.body_geomadr[reloaded.body("boxbot#boxbot").id]][0] == pytest.approx(0.2)
    assert calls[-1] == (["robots", "parts"], None, None) # the list form uses the cache
    # the mapping form passes root (resolved like a path) and refs
    scene["pydescriptions"] = {"packages" : ["robots", "parts"], "root" : "ws", "refs" : {"repo" : "branch"}}
    export_mjcf(scene, tmp_path)
    assert calls[-1] == (["robots", "parts"], tmp_path / "ws", {"repo" : "branch"})
    # a model's own pkg_paths take precedence over the resolved packages
    (tmp_path / "other_parts").mkdir()
    (tmp_path / "other_parts" / "box.xacro").write_text((tmp_path / "parts" / "box.xacro").read_text()
                                                         .replace('type="box"', 'type="sphere"'))
    scene["pydescriptions"] = ["robots", "parts"]
    model["pkg_paths"] = {"parts" : "other_parts"}
    reloaded = mujoco.MjModel.from_xml_string(export_mjcf(scene, tmp_path))
    assert reloaded.geom_type[0] == mujoco.mjtGeom.mjGEOM_SPHERE


def test_pydescriptions_errors(fake_pydescriptions, monkeypatch):
    tmp_path, _ = fake_pydescriptions
    model = {"name" : "boxbot", "file" : "pydescriptions://robots/mjcf/boxbot.mjcf.xacro"}
    with pytest.raises(ValueError, match="not in the scene's pydescriptions list"):
        load_scene({"pydescriptions" : ["parts"], "models" : [model]}, tmp_path)
    with pytest.raises(ValueError, match="not in the scene's pydescriptions list"):
        load_scene({"models" : [model]}, tmp_path)
    for bad in ({"packages" : []}, {"packages" : ["robots"], "rot" : "x"}, "robots"):
        with pytest.raises(ValueError):
            load_scene({"pydescriptions" : bad, "models" : [model]}, tmp_path)
    monkeypatch.setitem(sys.modules, "pydescriptions", None) # not installed
    with pytest.raises(ImportError, match="not installed"):
        load_scene({"pydescriptions" : ["robots"], "models" : [model]}, tmp_path)


def test_scene_errors():
    with pytest.raises(ValueError, match="no models"):
        load_scene({"models" : []}, Path("."))
    with pytest.raises(ValueError, match="Unknown scene keys"):
        load_scene({**_scene(), "colisions" : []}, Path("."))
    with pytest.raises(ValueError, match="unknown keys"):
        load_scene({"models" : [{"name" : "a", "definition" : CUBE, "format" : "mjcf", "pos" : 1}]}, Path("."))
    with pytest.raises(ValueError, match="exactly one"):
        load_scene({"models" : [{"name" : "a", "format" : "mjcf"}]}, Path("."))
    with pytest.raises(ValueError, match="needs a 'format'"):
        load_scene({"models" : [{"name" : "a", "definition" : CUBE}]}, Path("."))
    with pytest.raises(ValueError, match="xacro formats"):
        load_scene({"models" : [{"name" : "a", "definition" : CUBE, "format" : "mjcf", "kwargs" : {"x" : 1}}]}, Path("."))


def test_cli_writes_only_the_mjcf_to_stdout(tmp_path):
    import yaml
    scene_file = tmp_path / "scene.yaml"
    (tmp_path / "cube.xml").write_text(CUBE)
    scene_file.write_text(yaml.safe_dump({"models" : [{"name" : "robot", "definition" : ROBOT, "format" : "mjcf",
                                                       "prefix" : False},
                                                      {"name" : "cube", "file" : "cube.xml"}], # relative, format inferred
                                          "options" : {"add_ground" : False, "add_sky" : False},
                                          "collisions" : [{"link" : ["robot", "arm"], "collides_with" : ["cube"]},
                                                          {"link" : "cube", "collides_with" : [["robot", "arm"]]}]}))
    out = subprocess.run([sys.executable, "-m", "adarl.adapters.mjcf_export", str(scene_file)],
                         capture_output=True, text=True, check=True, cwd=tmp_path.parent)
    model = mujoco.MjModel.from_xml_string(out.stdout)
    assert _can_collide(model, "arm", "cube#cube") and not _can_collide(model, "base", "cube#cube")
