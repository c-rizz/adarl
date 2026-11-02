"""alter_model: the ratio convention, and keeping the model's mass-dependent constants consistent.

Ratios are multiplicative, the same on every adapter: a ratio of 1 leaves the model unchanged (environments start from
ratios of 1 and randomize around them), a ratio of 2 doubles exactly the altered parameters. MjxAdapter is the
reference; GenesisAdapter used to compute old * (1 + ratio), doubling every randomized parameter even with no
randomization.

A mass alteration also scales the inertia (same geometry, uniform density), and on MJX the constants MuJoCo derives
from the mass matrix at compile time are recomputed as mj_setConst does: body_invweight0 and dof_invweight0, which set
the softness of every constraint, and body_subtreemass. Without that, contacts stop being mass-independent: a body
twice as heavy sinks twice as deep into the ground.
"""
import copy

import numpy as np
import pytest
import torch as th

from adapter_compliance import _spawn_defs, CART_JOINT, FOOT_JOINT, BAR_LINK, BALL_LINK
from _backends import requires_jax_gpu, requires_mjx_warp, requires_genesis, build_genesis_impedance, _log_folder

VEC = 2
LINKS = [BAR_LINK, BALL_LINK]
JOINTS = [CART_JOINT, FOOT_JOINT]
DERIVED = ("body_invweight0", "dof_invweight0", "body_subtreemass")
"""MJX constants recomputed in float32 from the mass matrix: they match MuJoCo's to ~0.5%, not to the last bit."""


def _build_mjx(mjx_impl : str = "jax", vec_size : int = VEC):
    import jax
    from adarl.adapters.MjxAdapter import MjxAdapter
    return MjxAdapter(vec_size=vec_size, enable_rendering=False, jax_device=jax.devices("gpu")[0],
                      output_th_device=th.device("cuda", 0), sim_step_dt=1/512, step_length_sec=24/512,
                      realtime_factor=-1, show_gui=False, mjx_impl=mjx_impl, render_backend="cpu",
                      log_folder=_log_folder())


def _params(adapter, backend : str) -> dict[str, np.ndarray]:
    """The parameters alter_model changes, for every link/geom/dof and environment."""
    if backend == "mjx":
        m = adapter._sim_state.mjx_model
        return {k : np.array(getattr(m, k)) for k in
                ("body_mass", "body_inertia", "geom_friction", "dof_armature", "dof_damping", "dof_frictionloss",
                 "body_ipos", "body_iquat") + DERIVED}
    rs = adapter._rigid_solver()
    return {"mass" : rs.get_links_inertial_mass().cpu().numpy(),
            "friction_ratio" : rs.get_geoms_friction_ratio().cpu().numpy(),
            "armature" : rs.get_dofs_armature().cpu().numpy(),
            "damping" : rs.get_dofs_damping().cpu().numpy(),
            "frictionloss" : rs.get_dofs_frictionloss().cpu().numpy(),
            "com_shift" : rs.get_links_COM_shift().cpu().numpy(),
            "links_invweight" : rs.get_links_invweight().cpu().numpy(),
            "dofs_invweight" : rs.get_dofs_invweight().cpu().numpy()}


def _alter(adapter, backend : str, ratio : float) -> None:
    link_ids = adapter.get_links_ids(LINKS)
    joint_ids = adapter.get_joints_ids(JOINTS)
    full = lambda *shape: th.full((VEC,) + shape, ratio)
    kwargs = dict(link_masses=(link_ids, full(len(LINKS))),
                  link_frictions=(link_ids, full(len(LINKS), 3)),
                  joint_armature_ratios=(joint_ids, full(len(JOINTS))),
                  joint_damping_ratios=(joint_ids, full(len(JOINTS))),
                  joint_frictionloss_ratios=(joint_ids, full(len(JOINTS))),
                  com_position_diffs=(link_ids, th.zeros((VEC, len(LINKS), 3))))
    if backend == "mjx": # genesis does not support CoM orientation
        kwargs["com_quatxyzw_diffs"] = (link_ids, th.tensor([0.0, 0.0, 0.0, 1.0]).expand(VEC, len(LINKS), 4).clone())
    adapter.alter_model(**kwargs)


@pytest.fixture(scope="module", params=[pytest.param("mjx", marks=[requires_jax_gpu, pytest.mark.mjx]),
                                        pytest.param("genesis", marks=[requires_genesis, pytest.mark.genesis])])
def scene(request):
    adapter = _build_mjx() if request.param == "mjx" else build_genesis_impedance(vec_size=VEC)
    adapter.build_scenario(_spawn_defs())
    adapter.startup()
    try:
        yield adapter, request.param
    finally:
        adapter.destroy_scenario()


pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_DERIVED_ALL = DERIVED + ("links_invweight", "dofs_invweight")


def test_ratio_one_leaves_the_model_unchanged(scene):
    adapter, backend = scene
    before = _params(adapter, backend)
    _alter(adapter, backend, 1.0)
    after = _params(adapter, backend)
    for name in before:
        rtol = 5e-3 if name in _DERIVED_ALL else 1e-6 # derived constants are recomputed, in float32 on MJX
        np.testing.assert_allclose(after[name], before[name], rtol=rtol, atol=0, err_msg=f"{backend}: {name} changed")


def test_ratio_two_doubles_the_altered_parameters(scene):
    adapter, backend = scene
    _alter(adapter, backend, 1.0) # the nominal model (alter_model resets previous alterations first)
    before = _params(adapter, backend)
    _alter(adapter, backend, 2.0)
    after = _params(adapter, backend)
    for name in before:
        if name in ("body_ipos", "body_iquat", "com_shift") + _DERIVED_ALL:
            continue # altered additively / by composition with neutral values here, or derived (tested below)
        old, new = before[name], after[name]
        doubled = np.isclose(new, 2*old, rtol=1e-6) & (old != 0)
        unchanged = np.isclose(new, old, rtol=1e-6)
        assert (doubled | unchanged).all(), f"{backend}: {name} has values neither doubled nor unchanged"
        assert doubled.any(), f"{backend}: no {name} value was doubled"


def test_doubled_free_body_halves_its_inverse_weights(scene):
    """For a free rigid body the inverse weight is exactly [1/m, mean(1/I)] at its center of mass: doubling its mass
    and inertia halves both. The ball is a free body."""
    adapter, backend = scene
    _alter(adapter, backend, 1.0)
    before = _params(adapter, backend)
    _alter(adapter, backend, 2.0)
    after = _params(adapter, backend)
    name = "body_invweight0" if backend == "mjx" else "links_invweight"
    ball_id = adapter.get_links_ids([BALL_LINK]) # a jax array on MJX, a torch tensor on genesis
    ball = int(np.asarray(ball_id.cpu() if isinstance(ball_id, th.Tensor) else ball_id)[0])
    b_old, b_new = before[name], after[name]
    b_old, b_new = (b_old[:, ball], b_new[:, ball]) if b_old.shape[0] == VEC else (b_old[ball].T, b_new[ball].T)
    np.testing.assert_allclose(b_new, b_old/2, rtol=5e-3, err_msg=f"{backend}: ball {name}")


def test_doubled_free_body_halves_its_dof_inverse_weights(scene):
    """The dofs' inverse weights are only updated by recomputing them from the mass matrix: doubling the ball's mass
    and inertia must halve exactly the 6 of its free joint, and leave every other dof's unchanged."""
    adapter, backend = scene
    name = "dof_invweight0" if backend == "mjx" else "dofs_invweight"
    ball_ids = adapter.get_links_ids([BALL_LINK])
    adapter.alter_model(link_masses=(ball_ids, th.ones((VEC, 1))))
    before = _params(adapter, backend)[name]
    adapter.alter_model(link_masses=(ball_ids, th.full((VEC, 1), 2.0)))
    after = _params(adapter, backend)[name]
    halved = np.isclose(after, before/2, rtol=5e-3) & (before != 0)
    unchanged = np.isclose(after, before, rtol=5e-3)
    assert (halved | unchanged).all(), f"{backend}: {name} values neither halved nor unchanged:\n{before}\n{after}"
    assert (halved.sum(axis=-1) == 6).all(), f"{backend}: expected the ball's 6 dofs halved, got {halved.sum(axis=-1)}"


# --------------------------------------------------------------- MJX: exactness and physics, jax and warp


@pytest.fixture(scope="module", params=[pytest.param("jax", marks=requires_jax_gpu),
                                        pytest.param("warp", marks=requires_mjx_warp)])
def mjx_scene(request):
    adapter = _build_mjx(request.param, vec_size=3)
    adapter.build_scenario(_spawn_defs())
    adapter.set_monitored_links([BALL_LINK])
    adapter.startup()
    try:
        yield adapter
    finally:
        adapter.destroy_scenario()


@pytest.mark.mjx
def test_invweight0_matches_mujoco_set_const(mjx_scene):
    """Per-environment masses, inertias and armatures: the recomputed constants equal what mj_setConst computes for a
    CPU model with the same parameters."""
    import mujoco
    adapter = mjx_scene
    link_ids, joint_ids = adapter.get_links_ids(LINKS), adapter.get_joints_ids(JOINTS)
    adapter.alter_model(link_masses=(link_ids, th.tensor([[1.0, 1.0], [0.5, 3.0], [2.0, 0.7]])),
                        joint_armature_ratios=(joint_ids, th.tensor([[1.0, 1.0], [3.0, 0.2], [0.5, 2.0]])))
    m_batch = adapter._sim_state.mjx_model
    for env in range(3):
        cpu = copy.deepcopy(adapter._mj_model)
        for name in ("body_mass", "body_inertia", "dof_armature"):
            getattr(cpu, name)[:] = np.array(getattr(m_batch, name)[env])
        mujoco.mj_setConst(cpu, mujoco.MjData(cpu))
        for name in DERIVED:
            np.testing.assert_allclose(np.array(getattr(m_batch, name)[env]), getattr(cpu, name), rtol=5e-3,
                                       atol=1e-6, err_msg=f"env {env}: {name}")


@pytest.mark.mjx
def test_contact_penetration_does_not_depend_on_mass(mjx_scene):
    """MuJoCo's soft contacts are designed to behave the same whatever the masses: a ball resting on the ground sinks
    equally deep at its nominal mass, at twice it and at half it. With the nominal inverse weights left in place, the
    heavier ball sank twice as deep and the lighter one half as deep."""
    adapter = mjx_scene
    ball_ids = adapter.get_links_ids([BALL_LINK])
    adapter.alter_model(link_masses=(ball_ids, th.tensor([[1.0], [2.0], [0.5]])))
    dev = adapter.output_th_device()
    radius = 0.5
    # on the ground plane, clear of the cartpole (its rail runs along x)
    rest = th.tensor([0.0, 3.0, radius, 0.0, 0.0, 0.0, 1.0] + [0.0]*6, device=dev).view(1, 1, 13).expand(3, 1, 13)
    adapter.setLinksStateDirect([BALL_LINK], rest.clone())
    for _ in range(40): # ~1.9 s: settled
        adapter.step()
    penetration = (radius - adapter.getLinksState([BALL_LINK])[:, 0, 2]).cpu().numpy()
    assert penetration[0] > 0, f"nominal ball not in contact: penetration {penetration}"
    np.testing.assert_allclose(penetration[1:], penetration[0], rtol=0.05,
                               err_msg=f"penetration at mass x1, x2, x0.5: {penetration}")


def test_link_scales_unsupported_on_genesis(scene):
    adapter, backend = scene
    if backend != "genesis":
        pytest.skip("only genesis lacks per-environment geometry")
    with pytest.raises(NotImplementedError):
        adapter.alter_model(link_scales=(adapter.get_links_ids([BALL_LINK]), th.ones((VEC, 1, 3))))


# --------------------------------------------------------------- MJX: per-axis link scaling (link_scales)

BOX_HALF = np.array([0.05, 0.03, 0.02])
BOX_MASS = 0.5
SIDE_HALF = np.array([0.01, 0.02, 0.005])
SIDE_POS = np.array([0.08, 0.0, 0.0])
TILTED_MOMENTS = np.array([0.003, 0.002, 0.004])
TILTED_IQUAT = np.array([np.cos(np.pi/8), 0.0, 0.0, np.sin(np.pi/8)]) # wxyz, 45 degrees about z
SCALE_CAM_SIZE = 64
SCALES = th.tensor([[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]], [[0.5, 1.0, 2.0]]])


def _box_inertia(mass : float, half : np.ndarray) -> np.ndarray:
    a, b, c = half
    return mass/3*np.array([b*b + c*c, a*a + c*c, a*a + b*b])


_fmt = lambda a: " ".join(str(float(v)) for v in a)
SCALE_MJCF = f"""
<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <camera name="top" pos="0 0 0.6" xyaxes="1 0 0 0 1 0" fovy="30" resolution="{SCALE_CAM_SIZE} {SCALE_CAM_SIZE}"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <inertial pos="0 0 0" mass="{BOX_MASS}" diaginertia="{_fmt(_box_inertia(BOX_MASS, BOX_HALF))}"/>
      <geom name="box_geom" type="box" size="{_fmt(BOX_HALF)}" rgba="1 0 0 1"/>
      <geom name="side_geom" type="box" size="{_fmt(SIDE_HALF)}" pos="{_fmt(SIDE_POS)}" quat="0.70710678 0 0 0.70710678"
            contype="0" conaffinity="0" rgba="0 1 0 1"/>
    </body>
    <body name="tilted" pos="0.6 0.6 0.3">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" quat="{_fmt(TILTED_IQUAT)}" diaginertia="{_fmt(TILTED_MOMENTS)}"/>
      <geom type="box" size="0.04 0.04 0.04" rgba="0 0 1 1"/>
    </body>
    <body name="sphere" pos="-0.6 0.6 0.3">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
      <geom type="sphere" size="0.05" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>"""
"""A box with a second, offset geom turned 90 degrees about z (its x axis along the body's y axis), a body whose
inertial frame is turned 45 degrees about z, and a sphere, which cannot be scaled per axis."""


@pytest.fixture(scope="module", params=[pytest.param("jax", marks=requires_jax_gpu),
                                        pytest.param("warp", marks=requires_mjx_warp)])
def scale_scene(request):
    import jax
    from adarl.adapters.MjxAdapter import MjxAdapter
    from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
    warp = request.param == "warp"
    adapter = MjxAdapter(vec_size=3, enable_rendering=warp, jax_device=jax.devices("gpu")[0],
                         output_th_device=th.device("cuda", 0), sim_step_dt=1/512, step_length_sec=24/512,
                         realtime_factor=-1, show_gui=False, mjx_impl=request.param,
                         render_backend="warp" if warp else "cpu", log_folder=_log_folder())
    adapter.build_scenario([ModelSpawnDef(name="scaled", definition_string=SCALE_MJCF, format="mjcf",
                                          pose=None, kwargs={})])
    links = {tuple(l)[1] : tuple(l) for l in adapter.get_detected_links()}
    adapter.set_monitored_links([links["box"]])
    cameras = [c for c in adapter.get_detected_cameras() if "top" in str(c)]
    if warp:
        adapter.set_monitored_cameras(cameras)
    adapter.startup()
    try:
        yield adapter, links, cameras, request.param
    finally:
        adapter.destroy_scenario()


def _body_and_geoms(adapter, link) -> tuple[int, np.ndarray]:
    body = int(np.asarray(adapter.get_links_ids([link]))[0])
    return body, np.nonzero(adapter._mj_model.geom_bodyid == body)[0]


def _quat2mat(quat : np.ndarray) -> np.ndarray:
    import mujoco
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


@pytest.mark.mjx
def test_unit_scale_leaves_the_model_unchanged(scale_scene):
    adapter, links, _, _ = scale_scene
    fields = ("geom_size", "geom_pos", "geom_rbound", "geom_aabb", "body_inertia", "body_iquat", "body_ipos", "body_mass")
    adapter.alter_model(link_masses=(adapter.get_links_ids([links["box"]]), th.ones((3, 1)))) # the nominal model
    before = {f : np.array(getattr(adapter._sim_state.mjx_model, f)) for f in fields}
    adapter.alter_model(link_scales=(adapter.get_links_ids([links["box"], links["tilted"]]), th.ones((3, 2, 3))))
    for f in fields:
        np.testing.assert_allclose(np.array(getattr(adapter._sim_state.mjx_model, f)), before[f], rtol=1e-5, atol=1e-7,
                                   err_msg=f"{f} changed")


@pytest.mark.mjx
def test_scaled_box_geometry_and_inertia(scale_scene):
    """Sizes, positions and bounds follow the scale along the body axes, also for a geom turned relative to the body;
    the inertia is the scaled box's at the same mass."""
    adapter, links, _, _ = scale_scene
    adapter.alter_model(link_scales=(adapter.get_links_ids([links["box"]]), SCALES))
    m = adapter._sim_state.mjx_model
    body, (box_geom, side_geom) = _body_and_geoms(adapter, links["box"])
    for env in range(3):
        s = SCALES[env, 0].numpy()
        half = BOX_HALF*s
        np.testing.assert_allclose(m.geom_size[env, box_geom], half, rtol=1e-6)
        np.testing.assert_allclose(m.geom_rbound[env, box_geom], np.linalg.norm(half), rtol=1e-6)
        np.testing.assert_allclose(np.array(m.geom_aabb[env, box_geom]).reshape(2, 3)[1], half, rtol=1e-6)
        # turned 90 degrees about z: the side geom's x axis lies along the body's y axis and vice versa
        np.testing.assert_allclose(m.geom_size[env, side_geom], SIDE_HALF*s[[1, 0, 2]], rtol=1e-6)
        np.testing.assert_allclose(m.geom_pos[env, side_geom], SIDE_POS*s, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(m.body_inertia[env, body], _box_inertia(BOX_MASS, half), rtol=1e-5)
        np.testing.assert_allclose(m.body_mass[env, body], BOX_MASS, rtol=1e-6)


@pytest.mark.mjx
def test_scaled_inertia_off_the_scaling_axes(scale_scene):
    """An inertial frame turned relative to the scaling axes: the stretched inertia tensor is not diagonal in it, the
    moments and the frame are both recomputed, and together they give the stretched tensor."""
    adapter, links, _, _ = scale_scene
    adapter.alter_model(link_scales=(adapter.get_links_ids([links["tilted"]]), SCALES))
    m = adapter._sim_state.mjx_model
    body, _ = _body_and_geoms(adapter, links["tilted"])
    rot = _quat2mat(TILTED_IQUAT)
    sigma = rot @ np.diag(TILTED_MOMENTS.sum()/2 - TILTED_MOMENTS) @ rot.T
    for env in range(3):
        s = SCALES[env, 0].numpy()
        sigma_s = np.diag(s) @ sigma @ np.diag(s)
        expected = np.trace(sigma_s)*np.eye(3) - sigma_s
        new_rot = _quat2mat(np.array(m.body_iquat[env, body]))
        got = new_rot @ np.diag(np.array(m.body_inertia[env, body])) @ new_rot.T
        np.testing.assert_allclose(got, expected, atol=1e-7, err_msg=f"env {env}")
        if not np.allclose(s, s[0]):
            assert not np.allclose(np.abs(np.array(m.body_iquat[env, body])), np.abs(TILTED_IQUAT), atol=1e-3), \
                f"env {env}: the inertial frame should have turned"


@pytest.mark.mjx
def test_invweight0_matches_mujoco_set_const_after_scaling(scale_scene):
    import mujoco
    adapter, links, _, _ = scale_scene
    adapter.alter_model(link_scales=(adapter.get_links_ids([links["box"], links["tilted"]]),
                                     th.cat([SCALES, SCALES.flip(0)], dim=1)))
    m_batch = adapter._sim_state.mjx_model
    for env in range(3):
        cpu = copy.deepcopy(adapter._mj_model)
        for name in ("body_mass", "body_inertia", "body_iquat", "body_ipos", "dof_armature"):
            getattr(cpu, name)[:] = np.array(getattr(m_batch, name)[env])
        mujoco.mj_setConst(cpu, mujoco.MjData(cpu))
        for name in DERIVED:
            np.testing.assert_allclose(np.array(getattr(m_batch, name)[env]), getattr(cpu, name), rtol=5e-3,
                                       atol=1e-6, err_msg=f"env {env}: {name}")


@pytest.mark.mjx
def test_scaled_box_rests_at_its_scaled_height(scale_scene):
    """Collision uses each environment's own geometry: the box settles on the ground at its scaled half-height."""
    adapter, links, _, _ = scale_scene
    adapter.alter_model(link_scales=(adapter.get_links_ids([links["box"]]), SCALES))
    half_z = BOX_HALF[2]*SCALES[:, 0, 2].numpy()
    rest = th.zeros((3, 1, 13))
    rest[:, 0, 2] = th.as_tensor(half_z) + 0.002
    rest[:, 0, 6] = 1.0
    adapter.setLinksStateDirect([links["box"]], rest.to(adapter.output_th_device()))
    for _ in range(40): # ~1.9 s: settled
        adapter.step()
    z = adapter.getLinksState([links["box"]])[:, 0, 2].cpu().numpy()
    np.testing.assert_allclose(z, half_z, atol=1e-3, err_msg=f"resting heights {z}, half-heights {half_z}")


@pytest.mark.mjx
def test_scaled_box_is_rendered_scaled(scale_scene):
    """The warp raytracer draws each environment's own geometry: the box's footprint seen from above scales with
    sx*sy (and with the inverse square of the distance of its top face)."""
    adapter, links, cameras, impl = scale_scene
    if impl != "warp":
        pytest.skip("rendering is tested with the warp render backend")
    adapter.alter_model(link_scales=(adapter.get_links_ids([links["box"]]), SCALES))
    half_z = BOX_HALF[2]*SCALES[:, 0, 2].numpy()
    rest = th.zeros((3, 1, 13))
    rest[:, 0, 2] = th.as_tensor(half_z)
    rest[:, 0, 6] = 1.0
    adapter.setLinksStateDirect([links["box"]], rest.to(adapter.output_th_device()))
    rgb, _ = adapter.getRenderings(cameras, depth=False)
    img = rgb[0].float()
    red = (img[..., 0] > 0.5) & (img[..., 1] < 0.3) & (img[..., 2] < 0.3)
    area = red.sum(dim=(1, 2)).float().cpu().numpy()
    assert area[0] > 100, f"box not visible: {area} red pixels"
    top_dist = 0.6 - 2*half_z
    s = SCALES[:, 0].numpy()
    expected = s[:, 0]*s[:, 1]*(top_dist[0]/top_dist)**2
    np.testing.assert_allclose(area/area[0], expected, rtol=0.15, err_msg=f"red pixels {area}")


@pytest.mark.mjx
def test_unscalable_links_are_rejected(scale_scene):
    adapter, links, _, _ = scale_scene
    with pytest.raises(ValueError, match="SPHERE"):
        adapter.alter_model(link_scales=(adapter.get_links_ids([links["sphere"]]), th.ones((3, 1, 3))))
