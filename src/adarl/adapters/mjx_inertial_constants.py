"""Recomputing MuJoCo's mass-dependent model constants for MJX models altered per environment.

MuJoCo computes body_invweight0 and dof_invweight0 once, at compile time (mj_setConst), from the mass matrix in the
reference configuration qpos0. They set the softness of every constraint (contacts, friction, joint limits): a
constraint's regularization is proportional to the inverse weight of what it acts on, which is what makes contact
dynamics independent of the masses involved. When the inertial parameters of a model are changed after compilation
(e.g. domain randomization of link masses), they have to be recomputed, otherwise a heavier body gets a softer
contact (it sinks deeper and responds more slowly) and a lighter one a stiffer contact.

MJX has no set_const, and its mass-matrix functions only accept JAX-implementation models. This module keeps a
JAX-implementation copy of the model reduced to its kinematic tree and inertia, which also serves models simulated
with the Warp implementation, and recomputes the two constants following mj_setConst (engine_setconst.c) exactly.

It also computes the inertia of a body whose geometry is stretched per axis (scale_body_inertia).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import smooth, support

# Fields the constants depend on, copied from the simulated model, and per environment replaced at recomputation
INERTIAL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "dof_armature")

PRINCIPAL_AXES_RTOL = 1e-5
"""Relative off-diagonal magnitude below which a scaled inertia is taken as diagonal in the original inertial frame,
which is then kept as it is."""


@jax.tree_util.register_dataclass
@dataclass
class InertialConstantsModel:
    model : mjx.Model
    """JAX-implementation model with the simulated model's kinematic tree and inertia, without constraints."""
    data : mjx.Data
    """A data of model, whose qpos is set to qpos0 at each recomputation."""
    body_static : jax.Array
    """(nbody,) bool: bodies welded to the world, including the world body, whose inverse weights are 0."""
    body_simple2 : jax.Array
    """(nbody,) bool: moving bodies MuJoCo flags as simple (body_simple == 2: no rotations, diagonal inertia), whose
    inverse weights are 1/mass and 0, as mj_setConst sets them."""
    dof_average : jax.Array
    """(nv, nv): averages diag(M^-1) over the dofs of each joint, per component group as mj_setConst does (the 3
    translations and the 3 rotations of a free joint, the 3 rotations of a ball joint)."""
    dof_simple2_body : jax.Array
    """(nv,) int: for the dofs of simple bodies the body, whose 1/mass is the dof's inverse weight; -1 otherwise."""


def build_inertial_constants_model(spec : mujoco.MjSpec, mj_model : mujoco.MjModel,
                                   device : jax.Device | None = None) -> InertialConstantsModel:
    """Build the model recomputations run on, from the spec mj_model was compiled from.

    Everything not needed to compute the mass matrix and the body jacobians in qpos0 is removed (sensors, actuators,
    tendons, equalities, contact pairs and excludes; geoms do not collide), so that the JAX implementation accepts
    models it could not simulate (e.g. with the contact sensors used by the Warp implementation). The inertial and
    kinematic parameters are then copied from mj_model, which may have been modified after compilation.
    """
    spec = spec.copy()
    for elements in (spec.sensors, spec.actuators, spec.equalities, spec.pairs, spec.excludes, spec.tendons):
        for element in list(elements):
            spec.delete(element)
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    m = spec.compile()
    for name in ("nbody", "njnt", "nq", "nv"):
        if getattr(m, name) != getattr(mj_model, name):
            raise RuntimeError(f"stripped model has {name}={getattr(m, name)}, the simulated one {getattr(mj_model, name)}")
    for name in ("body_parentid", "jnt_type", "jnt_bodyid", "jnt_dofadr", "dof_bodyid"):
        if not np.array_equal(getattr(m, name), getattr(mj_model, name)):
            raise RuntimeError(f"stripped model has a different {name} than the simulated one")
    for name in INERTIAL_FIELDS + ("qpos0", "jnt_pos", "jnt_axis"):
        getattr(m, name)[:] = getattr(mj_model, name)
    # options the JAX implementation supports; they do not affect the mass matrix
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    m.opt.enableflags = 0
    model = mjx.put_model(m, device=device, impl="jax")
    data = mjx.make_data(model)

    nv = mj_model.nv
    simple = mj_model.body_simple
    body_static = mj_model.body_weldid == 0
    body_simple2 = (simple == 2) & ~body_static
    dof_average = np.zeros((nv, nv), dtype=np.float32)
    dof_simple2_body = np.full((nv,), -1, dtype=np.int32)
    for j in range(mj_model.njnt):
        adr, jtype, body = int(mj_model.jnt_dofadr[j]), mj_model.jnt_type[j], int(mj_model.jnt_bodyid[j])
        if simple[body] == 2:
            dof_simple2_body[adr] = body
            continue
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            groups = [range(adr, adr+3), range(adr+3, adr+6)]
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            groups = [range(adr, adr+3)]
        else:
            groups = [range(adr, adr+1)]
        for group in groups:
            idx = np.array(group)
            dof_average[np.ix_(idx, idx)] = 1.0/len(idx)
    put = lambda a: jax.device_put(jnp.asarray(a), device)
    return InertialConstantsModel(model=model, data=data, body_static=put(body_static), body_simple2=put(body_simple2),
                                  dof_average=put(dof_average), dof_simple2_body=put(dof_simple2_body))


def compute_subtreemass(body_parentid : np.ndarray, body_mass : jax.Array) -> jax.Array:
    """body_subtreemass, the mass of each body plus its descendants, for body_mass of shape (..., nbody), as
    mj_setConst computes it. The Warp implementation divides by it to compute subtree centers of mass."""
    subtreemass = body_mass
    for i in range(len(body_parentid) - 1, 0, -1):
        subtreemass = subtreemass.at[..., int(body_parentid[i])].add(subtreemass[..., i])
    return subtreemass


def compute_invweight0(icm : InertialConstantsModel, body_mass : jax.Array, body_inertia : jax.Array,
                       body_ipos : jax.Array, body_iquat : jax.Array, body_pos : jax.Array, body_quat : jax.Array,
                       dof_armature : jax.Array) -> tuple[jax.Array, jax.Array]:
    """body_invweight0 (nbody, 2) and dof_invweight0 (nv,) of the model with the given inertial parameters (one
    environment: vmap it for a batch), as mj_setConst computes them.

    body_invweight0: the means of the translational and of the rotational diagonal of J*M^-1*J' at the body's center
    of mass, with J its jacobian, in qpos0. dof_invweight0: the diagonal of M^-1, averaged per joint component group.
    """
    m = icm.model.replace(body_mass=body_mass, body_inertia=body_inertia, body_ipos=body_ipos, body_iquat=body_iquat,
                          body_pos=body_pos, body_quat=body_quat, dof_armature=dof_armature)
    nbody, nv = m.nbody, m.nv
    inv_mass = 1.0/jnp.maximum(mujoco.mjMINVAL, body_mass)
    if nv == 0:
        body_iw = jnp.zeros((nbody, 2), dtype=body_mass.dtype)
        dof_iw = jnp.zeros((0,), dtype=body_mass.dtype)
    else:
        d = icm.data.replace(qpos=m.qpos0)
        d = smooth.kinematics(m, d)
        d = smooth.com_pos(m, d)
        d = smooth.crb(m, d)
        # M = L L': with X = L^-1 J, the diagonal of J' M^-1 J is the squared column norms of X. Better conditioned
        # than inverting M, which matters in float32 (light links next to heavy ones)
        chol = jnp.linalg.cholesky(support.full_m(m, d))
        solve = lambda x: jax.scipy.linalg.solve_triangular(chol, x, lower=True)

        def body_invweight(b):
            jacp, jacr = support.jac(m, d, d.xipos[b], b) # (nv, 3) each
            return jnp.stack([jnp.sum(jnp.square(solve(jacp)))/3, jnp.sum(jnp.square(solve(jacr)))/3])

        body_iw = jax.vmap(body_invweight)(jnp.arange(nbody))
        minv_diag = jnp.sum(jnp.square(solve(jnp.eye(nv, dtype=body_mass.dtype))), axis=0)
        dof_iw = icm.dof_average @ minv_diag
        dof_iw = jnp.where(icm.dof_simple2_body >= 0, inv_mass[jnp.maximum(icm.dof_simple2_body, 0)], dof_iw)
    body_iw = jnp.where(icm.body_simple2[:, None], jnp.stack([inv_mass, jnp.zeros_like(inv_mass)], axis=-1), body_iw)
    body_iw = jnp.where(icm.body_static[:, None], 0.0, body_iw)
    return body_iw, dof_iw


def _mat_to_quat(rot : jax.Array) -> jax.Array:
    """Unit quaternions (w, x, y, z) of rotation matrices rot (..., 3, 3), by Shepperd's method.

    Row k of the 4x4 candidates is 4*q_k*q: the row with the largest diagonal entry 4*q_k^2 is the best conditioned.
    """
    r00, r11, r22 = rot[..., 0, 0], rot[..., 1, 1], rot[..., 2, 2]
    r01, r02, r10 = rot[..., 0, 1], rot[..., 0, 2], rot[..., 1, 0]
    r12, r20, r21 = rot[..., 1, 2], rot[..., 2, 0], rot[..., 2, 1]
    candidates = jnp.stack([jnp.stack([1 + r00 + r11 + r22, r21 - r12, r02 - r20, r10 - r01], axis=-1),
                            jnp.stack([r21 - r12, 1 + r00 - r11 - r22, r01 + r10, r02 + r20], axis=-1),
                            jnp.stack([r02 - r20, r01 + r10, 1 - r00 + r11 - r22, r12 + r21], axis=-1),
                            jnp.stack([r10 - r01, r02 + r20, r12 + r21, 1 - r00 - r11 + r22], axis=-1)], axis=-2)
    best = jnp.argmax(jnp.diagonal(candidates, axis1=-2, axis2=-1), axis=-1)
    quat = jnp.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    return quat/jnp.linalg.norm(quat, axis=-1, keepdims=True)


def scale_body_inertia(body_inertia : jax.Array, body_iquat : jax.Array,
                       scale : jax.Array) -> tuple[jax.Array, jax.Array]:
    """Principal moments (..., 3) and inertial frame (..., 4, wxyz) of bodies whose mass distribution is stretched by
    scale (..., 3) along the body frame's axes, at constant mass and uniform density.

    body_inertia and body_iquat are the bodies' principal moments and inertial frame. Stretching by S = diag(scale)
    maps the second moment of the mass distribution, Sigma = sum(m r r'), to S Sigma S, and the inertia is
    trace(Sigma) I - Sigma. The result is diagonalized again: the inertial frame stays body_iquat when its axes are
    aligned with the scaling axes or the scaling is uniform, otherwise it rotates.
    """
    rot = jnp.vectorize(mjx_math.quat_to_mat, signature="(4)->(3,3)")(body_iquat)
    rot_t = jnp.swapaxes(rot, -1, -2)
    sigma = jnp.sum(body_inertia, axis=-1, keepdims=True)/2 - body_inertia # principal second moments
    sigma_body = (rot*sigma[..., None, :]) @ rot_t
    sigma_body = scale[..., :, None]*sigma_body*scale[..., None, :]
    eye = jnp.eye(3, dtype=body_inertia.dtype)
    inertia_body = jnp.trace(sigma_body, axis1=-2, axis2=-1)[..., None, None]*eye - sigma_body
    inertia = rot_t @ inertia_body @ rot # in the original inertial frame
    diag = jnp.diagonal(inertia, axis1=-2, axis2=-1)
    off_diag = jnp.max(jnp.abs(inertia - diag[..., None]*eye), axis=(-2, -1))
    aligned = off_diag <= PRINCIPAL_AXES_RTOL*jnp.sum(diag, axis=-1)
    moments, axes = jnp.linalg.eigh(inertia)
    axes = axes.at[..., :, 2].multiply(jnp.sign(jnp.linalg.det(axes))[..., None]) # a rotation, not a reflection
    new_iquat = _mat_to_quat(rot @ axes)
    return (jnp.where(aligned[..., None], diag, moments),
            jnp.where(aligned[..., None], body_iquat, new_iquat))
