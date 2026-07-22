
import mujoco
from typing import Any, TypeAlias

from mujoco import mjx
from mujoco.mjx._src.io import _get_contact
import numpy as np
from packaging.version import Version

from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
import adarl.adapters.mujoco_utils as mjutils
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import compile_xacro_string 
import dataclasses
import jax
import pathlib
import os

mjtGeom :             TypeAlias = mujoco.mjtGeom # type: ignore
mju_quat2Mat :        TypeAlias = mujoco.mju_quat2Mat # type: ignore
mjv_initGeom :        TypeAlias = mujoco.mjv_initGeom # type: ignore
mjtCatBit :           TypeAlias = mujoco.mjtCatBit # type: ignore
mjv_connector :       TypeAlias = mujoco.mjv_connector # type: ignore
MjData :              TypeAlias = mujoco.MjData # type: ignore
_functions :          TypeAlias = mujoco._functions # type: ignore
_mju_dense2sparse :   TypeAlias = mujoco.mju_dense2sparse # type: ignore
_MjModel :            TypeAlias = mujoco.MjModel # type: ignore
_MjSpec :             TypeAlias = mujoco.MjSpec # type: ignore
_mjtInertiaFromGeom : TypeAlias = mujoco.mjtInertiaFromGeom # type: ignore
_MjsBody :            TypeAlias = mujoco.MjsBody # type: ignore
_mjtSensor :          TypeAlias = mujoco.mjtSensor # type: ignore
_mjtObj :             TypeAlias = mujoco.mjtObj # type: ignore
_mj_id2name :         TypeAlias = mujoco.mj_id2name # type: ignore
_mjtTrn :             TypeAlias = mujoco.mjtTrn # type: ignore
_mj_printModel :      TypeAlias = mujoco.mj_printModel # type: ignore
_mjtJoint :           TypeAlias = mujoco.mjtJoint # type: ignore
jax_Device :          TypeAlias = jax.Device # type: ignore
_mjtIntegrator :      TypeAlias = mujoco.mjtIntegrator # type: ignore
_mjtDisableBit :      TypeAlias = mujoco.mjtDisableBit # type: ignore
_mj_resetData :       TypeAlias = mujoco.mj_resetData # type: ignore
_mj_name2id :         TypeAlias = mujoco.mj_name2id # type: ignore
_MjvOption :          TypeAlias = mujoco.MjvOption # type: ignore
_mjtVisFlag :         TypeAlias = mujoco.mjtVisFlag # type: ignore
_mj_camlight :        TypeAlias = mujoco.mj_camlight # type: ignore


def path2tstr(path):
    return tuple([n.name for n in path])


def tree_nbytes(tree: Any) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def log_largest_dataclass_fields(obj: Any, name: str, top_k: int = 8):
    try:
        obj_fields = dataclasses.fields(obj)
    except TypeError:
        ggLog.info(f"Cannot summarize fields for {name}: object is not a dataclass")
        return

    field_infos: list[tuple[str, int, str, str]] = []
    total_nbytes = 0
    for field in obj_fields:
        value = getattr(obj, field.name)
        nbytes = tree_nbytes(value)
        total_nbytes += nbytes
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        shape_str = str(tuple(shape)) if shape is not None else "-"
        dtype_str = str(dtype) if dtype is not None else "-"
        field_infos.append((field.name, nbytes, shape_str, dtype_str))

    field_infos = [fi for fi in field_infos if fi[1] > 0]
    field_infos.sort(key=lambda x: x[1], reverse=True)
    if not field_infos:
        ggLog.info(f"No array-backed fields found for {name}")
        return

    ggLog.info(f"Largest {min(top_k, len(field_infos))} fields in {name}:")
    denom = max(total_nbytes, 1)
    for field_name, nbytes, shape_str, dtype_str in field_infos[:top_k]:
        ggLog.info(
            f"  {field_name:<24} {nbytes/1024**2:9.3f} MB"
            f"  ({100*nbytes/denom:5.1f}%)"
            f"  shape={shape_str}"
            f"  dtype={dtype_str}"
        )


def add_compiler_options(urdf_def : str,
                         max_hull_vert : int = 32,
                         discardvisual : bool = False,
                         strippath : bool = False):

    mujoco_block = ('<mujoco>\n'+
                    f'    <compiler  discardvisual="{str(discardvisual).lower()}" strippath="{str(strippath).lower()}" maxhullvert="{max_hull_vert:d}" inertiafromgeom="false"/>\n'
                    '</mujoco>')
    return urdf_def.replace("</robot>",mujoco_block+"\n</robot>")


# Mapping from the geom-type names accepted in <adarl><override type="..."/> to MuJoCo geom types.
adarl_geom_type_map = {
    "sphere":    mjutils.mjtGeom.mjGEOM_SPHERE,
    "capsule":   mjutils.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mjutils.mjtGeom.mjGEOM_ELLIPSOID,
    "cylinder":  mjutils.mjtGeom.mjGEOM_CYLINDER,
    "box":       mjutils.mjtGeom.mjGEOM_BOX,
}


def extract_adarl_directives(urdf_def : str):
    """Parse and strip <adarl> extension blocks from a URDF string.

    An <adarl> element can be placed inside a <visual> or <collision> (either as a direct
    child or nested in its <geometry>) to express things plain URDF cannot. Its child
    elements are "directives" applied to the corresponding MuJoCo geom after parsing but
    before compilation. Supported:

        <override type="ellipsoid" size="0.1 0.2 0.1"/>
            Replace the geom shape and/or size. `type` is one of sphere, capsule,
            ellipsoid, cylinder, box. `size` is the space-separated MuJoCo size for that
            type (e.g. 3 radii for an ellipsoid). Either attribute may be omitted.

    A URDF geom takes its MuJoCo name from the `name` of its <visual>/<collision>, so any
    tagged element without a name is given an auto-generated, stable one (so its geom can
    be found later). Note a directive only takes effect if the geom survives compilation
    (e.g. a visual-only geom is dropped when discardvisual is on).

    Returns
    -------
    cleaned_urdf : str
        The URDF with every <adarl> element removed (parseable by MuJoCo).
    overrides : dict[str, list[tuple[str, dict[str, str]]]]
        Maps geom name -> list of (directive_tag, attributes) to apply to that geom.
    """
    import lxml.etree as etree
    def localname(tag):
        if isinstance(tag, str) and "}" in tag:
            return tag.rsplit("}", 1)[1]
        return tag
    root = etree.fromstring(urdf_def.encode("utf-8"))
    overrides : dict[str, list[tuple[str, dict[str,str]]]] = {}
    auto_idx = 0
    for elem in root.iter():
        if localname(elem.tag) not in ("visual", "collision"):
            continue
        geometry = next((c for c in elem if localname(c.tag) == "geometry"), None)
        # <adarl> may be a direct child of <visual>/<collision> or nested inside its <geometry>.
        containers = [elem] + ([geometry] if geometry is not None else [])
        adarl_pairs = [(cont, c) for cont in containers for c in cont if localname(c.tag) == "adarl"]
        if not adarl_pairs:
            continue
        name = elem.get("name")
        if name is None:
            link = elem.getparent()
            link_name = link.get("name") if link is not None else "link"
            name = f"__adarl_{link_name}_{localname(elem.tag)}_{auto_idx}"
            auto_idx += 1
            elem.set("name", name)
        directives = overrides.setdefault(name, [])
        for container, adarl_elem in adarl_pairs:
            for directive in adarl_elem:
                if not isinstance(directive.tag, str): # skip comments / processing instructions
                    continue
                directives.append((localname(directive.tag), dict(directive.attrib)))
            container.remove(adarl_elem)
    return etree.tostring(root, encoding="unicode"), overrides


def apply_adarl_directive(geom, directive_tag : str, attrib : dict[str, str]):
    """Apply a single <adarl> directive (see extract_adarl_directives) to an MjsGeom."""
    if directive_tag == "override":
        type_str = attrib.get("type")
        if type_str is not None:
            if type_str not in adarl_geom_type_map:
                raise RuntimeError(f"<adarl> override: unknown geom type '{type_str}', "
                                   f"expected one of {sorted(adarl_geom_type_map)}")
            geom.type = adarl_geom_type_map[type_str]
        size_str = attrib.get("size")
        if size_str is not None:
            vals = [float(v) for v in size_str.split()]
            if not 1 <= len(vals) <= 3:
                raise RuntimeError(f"<adarl> override: 'size' must have 1 to 3 values, got {len(vals)}")
            size = np.zeros(3, dtype=np.float64)
            size[:len(vals)] = vals
            geom.size[:] = size
    else:
        raise RuntimeError(f"<adarl>: unknown directive '<{directive_tag}>'")


def add_geom_to_renderer(renderer : mujoco.Renderer,
                         geom_type : mjutils.mjtGeom,
                         size_xyz : np.ndarray,
                         pos_xyz  : np.ndarray,
                         quat_xyzw  : np.ndarray,
                         rgba : np.ndarray):
    if renderer.scene.ngeom == renderer.scene.maxgeom:
        raise RuntimeError(f"Cannot add geom ngeom == maxgeom == {renderer.scene.ngeom}")
    ggLog.info(f"adding geom {dict( geom_type=geom_type, size_xyz=size_xyz, pos_xyz=pos_xyz, quat_xyzw=quat_xyzw, rgba=rgba)}")
    quat_xyzw = quat_xyzw.astype(np.float64)
    orient_mat = np.empty((9,),dtype=quat_xyzw.dtype)
    mjutils.mju_quat2Mat(orient_mat, quat_xyzw[...,[3,0,1,2]])
    mjutils.mjv_initGeom(geom=renderer.scene.geoms[renderer.scene.ngeom],
                        type=geom_type,
                        size=size_xyz,
                        pos=pos_xyz,
                        mat=orient_mat,
                        rgba=rgba)
    renderer.scene.ngeom += 1


def add_arrow_to_renderer(renderer, from_, to, radius=0.03, rgba=[0.2, 0.2, 0.6, 1]):
  """Add an arrow to the scene."""
  scene = renderer.scene
  scene.geoms[scene.ngeom].category = mjutils.mjtCatBit.mjCAT_STATIC
  mjutils.mjv_initGeom(
      geom=scene.geoms[scene.ngeom],
      type=mjutils.mjtGeom.mjGEOM_ARROW,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(rgba).astype(np.float32),
  )
  mjutils.mjv_connector(
      geom=scene.geoms[scene.ngeom],
      type=mjutils.mjtGeom.mjGEOM_ARROW,
      width=radius,
      from_=from_,
      to=to,
  )
  scene.ngeom += 1


@jax.jit
def get_renderdata_dict(jax_data : mjx.Data):
    """ Copy the position and orientation data from a jax mjx.Data into a dict. Just to avoid copying all fields when only these are needed."""
    d = {
        'xpos' : jax_data.xpos,
        'xquat' : jax_data.xquat,
        'geom_xpos' : jax_data.geom_xpos,
        'geom_xmat' : jax_data.geom_xmat,
        'site_xpos' : jax_data.site_xpos,
        'site_xmat' : jax_data.site_xmat,
        'xipos' : jax_data.xipos,
        'ximat' : jax_data.ximat,
        'xfrc_applied' : jax_data.xfrc_applied,
    }
    # The native contact struct only exists on the JAX/C backends. The warp backend stores
    # contacts as flattened contact__* fields and mjx itself does not extract them into MjData
    # (mjx._src.io._get_data_into_warp skips 'contact'), so there is nothing to copy for rendering.
    # hasattr is evaluated at trace time on the static pytree type, so this branch is static.
    if hasattr(jax_data._impl, "contact"):
        d['contact'] = jax_data._impl.contact
    return d


def get_renderdata_into(
    cpu_data: list[mjutils.MjData],
    jax_data
):
    """ Copy the data needed for rendering from a jax mjx.Data into a list of mujoco_MjData.
        Just to avoid copying all fields when only these are needed."""
    poses = jax.device_get(get_renderdata_dict(jax_data))
    # contact is absent on the warp backend (see get_renderdata_dict); skip contact rendering there.
    contact = poses.get('contact')
    for i in range(len(cpu_data)):
        cdata = cpu_data[i]
        cdata.xpos = poses['xpos'][i]
        cdata.xquat = poses['xquat'][i]
        cdata.geom_xpos = poses['geom_xpos'][i]
        cdata.geom_xmat = poses['geom_xmat'][i].reshape((-1,9))
        cdata.site_xpos = poses['site_xpos'][i]
        cdata.site_xmat = poses['site_xmat'][i].reshape((-1,9))
        cdata.xipos = poses['xipos'][i]
        cdata.ximat = poses['ximat'][i].reshape((-1,9))
        cdata.xfrc_applied = poses['xfrc_applied'][i]

        if contact is not None:
            # Copy active contacts so that mjVIS_CONTACTPOINT etc. can be rendered.
            # Active contacts are those with dist <= 0, same criterion as mjx.get_data_into.
            contact_i = jax.tree_util.tree_map(lambda x, i=i: x[i], contact)
            ncon = int((contact_i.dist <= 0).sum())
            if ncon != cdata.ncon or cdata.nefc != 0:
                mjutils._functions._realloc_con_efc(cdata, ncon=ncon, nefc=0)  # pylint: disable=protected-access
            _get_contact(cdata.contact, contact_i)
            # efc_address would index into an efc array we don't populate; invalidate it.
            if cdata.contact.efc_address.size:
                cdata.contact.efc_address[:] = -1


def get_data_into(
    result: mjutils.MjData | list[mjutils.MjData],
    m,
    d,
    exclude : list[str] = []
):

  if Version(mujoco.__version__) >= Version("3.3.6"):
      return mjx.get_data_into(result, m, d)

  # Copy of get_data_into from mjx, with an exclude argument added, as some useless fields were causing issues
  """Gets mjx.Data from a device into an existing mujoco_MjData or list."""
  batched = isinstance(result, list)
  if batched and len(d.qpos.shape) < 2:
    raise ValueError('dst is a list, but d is not batched.')
  if not batched and len(d.qpos.shape) >= 2:
    raise ValueError('dst is a an MjData, but d is batched.')

  from mujoco.mjx._src.io import types, _get_contact, support
  d = jax.device_get(d)

  batch_size = d.qpos.shape[0] if batched else 1

  dof_i, dof_j = [], []
  for i in range(m.nv):
    j = i
    while j > -1:
      dof_i.append(i)
      dof_j.append(j)
      j = m.dof_parentid[j]

  for i in range(batch_size):
    d_i = jax.tree_util.tree_map(lambda x, i=i: x[i], d) if batched else d
    result_i = result[i] if batched else result
    ncon = (d_i.contact.dist <= 0).sum()
    efc_active = (d_i.efc_J != 0).any(axis=1)
    nefc = int(efc_active.sum())
    result_i.nJ = nefc * m.nv
    if ncon != result_i.ncon or nefc != result_i.nefc:
      mjutils._functions._realloc_con_efc(result_i, ncon=ncon, nefc=nefc)  # pylint: disable=protected-access
    result_i.efc_J_rownnz[:] = np.repeat(m.nv, nefc)
    result_i.efc_J_rowadr[:] = np.arange(0, nefc * m.nv, m.nv)
    result_i.efc_J_colind[:] = np.tile(np.arange(m.nv), nefc)

    for field in types.Data.fields():
      restricted_to = field.metadata.get('restricted_to')
      if restricted_to == 'mjx':
        continue
      if field.name in exclude:
          continue

      if field.name == 'contact':
        _get_contact(result_i.contact, d_i.contact)
        # efc_address must be updated because rows were deleted above:
        efc_map = np.cumsum(efc_active) - 1
        result_i.contact.efc_address[:] = efc_map[result_i.contact.efc_address]
        continue

      # MuJoCo actuator_moment is sparse, MJX uses a dense representation.
      if field.name == 'actuator_moment':
        moment_rownnz = np.zeros(m.nu, dtype=np.int32)
        moment_rowadr = np.zeros(m.nu, dtype=np.int32)
        moment_colind = np.zeros(m.nJmom, dtype=np.int32)
        actuator_moment = np.zeros(m.nJmom)
        if m.nu:
          mjutils._mju_dense2sparse(
              actuator_moment,
              d_i.actuator_moment,
              moment_rownnz,
              moment_rowadr,
              moment_colind,
          )
        result_i.moment_rownnz[:] = moment_rownnz
        result_i.moment_rowadr[:] = moment_rowadr
        result_i.moment_colind[:] = moment_colind
        result_i.actuator_moment[:] = actuator_moment
        continue

      value = getattr(d_i, field.name)

      if field.name in ('nefc', 'ncon'):
        value = {'nefc': nefc, 'ncon': ncon}[field.name]
      elif field.name.endswith('xmat') or field.name == 'ximat':
        value = value.reshape((-1, 9))
      elif field.name.startswith('efc_'):
        value = value[efc_active]
        if field.name == 'efc_J':
          value = value.reshape(-1)
      elif field.name == 'qM' and not support.is_sparse(m):
        value = value[dof_i, dof_j]
      elif field.name == 'qLD' and not support.is_sparse(m):
        value = value[dof_i, dof_j]
      elif field.name == 'qLDiagInv' and not support.is_sparse(m):
        value = np.ones(m.nv)

      if isinstance(value, np.ndarray) and value.shape:
        if restricted_to in ('mujoco', 'mjx'):
          continue  # don't copy fields that are mujoco-only or MJX-only
        else:
          # print(f"copying {field.name}")
          getattr(result_i, field.name)[:] = value
      else:
        setattr(result_i, field.name, value)


model_element_separator = "#"


def aggregate_models(models : list[ModelSpawnDef],
                     add_ground : bool,
                     add_sky : bool,
                     discardvisual : bool,
                     log_folder : str | None,
                     uneven_ground : bool = False,
                     geom_overrides : dict[str, dict[str, Any]] | None = None,
                     contact_pairs : list[tuple[str, str]] | None = None,
                     opt_preset : str | None = None,
                     opt_overrides : dict[str, Any] | None = None,
                     revolute_dof_armature_override : float | None = None,
                     revolute_dof_damping_override : float | None = None,
                     revolute_dof_frictionloss_override : float | None = None,
                     safe_revolute_dof_armature = 0.01,
                     safe_revolute_dof_damping = 1.0,
                     safe_revolute_dof_frictionloss = 0.2):
    
    """Aggregates multiple models into a single MjSpec and MjModel.

    Merges all the provided models into a single MjSpec, optionally adds ground/sky, applies
    overrides, and compiles the result into an MjModel.

    Parameters
    ----------
    models : list[ModelSpawnDef]
        The models to merge into the scenario. Each is attached to the merged spec, with its
        elements prefixed by `<modelname>#` (model_element_separator).
    add_ground : bool
        If True, add a ground plane geom (named "floor"), a checker ground material and a
        directional light to the scenario.
    add_sky : bool
        If True, add a gradient skybox texture to the scenario.
    uneven_ground : bool
        If True, replace the flat ground plane with a heightfield ("uneven_ground") for
        uneven terrain. Ignored unless add_ground/add_sky trigger ground generation.
    discardvisual : bool
        If True, visual-only geoms are discarded at compile time (passed through to each
        model's compiler options). Discarded geoms cannot be targeted by geom_overrides.
    log_folder : str | None
        If not None, the aggregated model is dumped as XML to `<log_folder>/aggregated_model.xml`.
    geom_overrides : dict[str, dict[str, Any]] | None
        Optional per-geom field overrides applied to the merged spec before compile.
        Maps geom name -> {field_name: value}, e.g. {"robot#foot": {"solimp": [0.9, 0.95, 0.001, 0.5, 2]}}.
        Note that attached geoms are prefixed with `<modelname>#` (model_element_separator).
    contact_pairs : list[tuple[str, str]] | None
        Optional list of (body_a, body_b) body-name pairs to monitor for contact. One
        mjSENS_CONTACT sensor named `__pair_<i>__` is injected per pair (warp backend only;
        the JAX backend reads contacts directly). Body names must exist in the merged spec.
    opt_preset : str | None
        Name of a solver/integrator preset to apply to the spec options (see
        apply_opt_preset_to_spec / _apply_opt_preset_to_opt). None leaves the MuJoCo defaults.
    opt_overrides : dict[str, Any] | None
        Optional per-field overrides for the spec options, applied on top of opt_preset.
    revolute_dof_armature_override : float | None
        If not None, force this armature on every hinge-joint DOF (see apply_dof_overrides_to_spec).
    revolute_dof_damping_override : float | None
        If not None, force this damping on every hinge-joint DOF.
    revolute_dof_frictionloss_override : float | None
        If not None, force this frictionloss on every hinge-joint DOF.
    safe_revolute_dof_armature : float
        Fallback armature applied to any hinge-joint DOF that has zero armature (avoids
        instability). Used only when the value would otherwise be zero.
    safe_revolute_dof_damping : float
        Fallback damping applied to any hinge-joint DOF that has zero damping.
    safe_revolute_dof_frictionloss : float
        Fallback frictionloss applied to any hinge-joint DOF that has zero frictionloss.

    Returns
    -------
    tuple[mjutils._MjModel, mjutils._MjSpec]
        The compiled MjModel and the MjSpec it was compiled from.
    """
    ggLog.info(f"MjxAdapter building scenario")
    if add_ground or add_sky:
        n="\n"
        ground_geoms = []
        assets = []
        uneven_ground = False
        if add_ground:
            ground_geoms.append('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin="0.0" pos="0 0 0"/>')
            assets.append('<texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />')
            assets.append('<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />')
        if uneven_ground:
            ground_geoms.append('<geom name="uneven_ground" type="hfield" hfield="uneven_ground" material="groundplane" friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin="0.0" pos="0 0 0"/>')
            assets.append('<hfield name="uneven_ground" nrow="128" ncol="128" size="10 10 10 10" />')
        if add_sky:
            assets.append('<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />')
        models.append(ModelSpawnDef(name="ground",
                                    definition_string=f"""
                                    <mujoco>
                                        <compiler angle="radian"/>
                                        <asset>
                                            {n.join(assets)}
                                        </asset>
                                        <worldbody>
                                            <body name="ground_link">
                                                <light  pos="0 0 1"
                                                        dir="0.3 0.3 -1"
                                                        type="directional"
                                                        ambient="0.2 0.2 0.2"
                                                        diffuse="0.7 0.7 0.7"
                                                        specular="0.5 0.5 0.5"
                                                        castshadow="true"/>
                                                {n.join(ground_geoms)}
                                            </body>
                                        </worldbody>
                                    </mujoco>""",
                                    format="mjcf",
                                    pose=None,
                                    kwargs={}))

    specs : list[tuple[str, mjutils._MjSpec, tuple[str,str] | None]] = []
    # <adarl> in-URDF overrides, keyed by post-attach geom name (<modelname>#<geomname>).
    adarl_overrides : dict[str, list[tuple[str, dict[str,str]]]] = {}
    ggLog.info(f"Spawning models: {[model.name for model in models]}")
    for model in models:
        mformat = model.format.strip().lower()
        if mformat[-6:] == ".xacro":
            def_string = compile_xacro_string( model_definition_string=model.definition_string,
                                                            model_kwargs=model.kwargs)
            mformat = mformat[:-6]  # remove .xacro suffix for later processing
        elif mformat in ("urdf","mjcf"):
            def_string = model.definition_string
            if def_string is None:
                raise RuntimeError(f"Model '{model.name}' has format '{model.format}' but no definition_string")
        else:
            raise RuntimeError(f"Unsupported model format '{model.format}' for model '{model.name}'")
        if mformat in ("urdf","urdf.xacro"):
            def_string, model_adarl = extract_adarl_directives(def_string)
            ggLog.info(f"Extracted <adarl> directives for model '{model.name}': {model_adarl}")
            for geom_name, directives in model_adarl.items():
                adarl_overrides[model.name + model_element_separator + geom_name] = directives
            def_string = add_compiler_options(def_string, discardvisual=discardvisual)
        if log_folder is not None:
            pathlib.Path(log_folder).mkdir(parents=True, exist_ok=True)
            with open(f"{log_folder}/model_{model.name}.{mformat}", "w") as f:
                f.write(def_string)
        mjSpec = mjutils._MjSpec.from_string(def_string)
        # mjSpec.compiler.discardvisual = False
        mjSpec.compiler.degree = False
        mjSpec.compiler.inertiafromgeom = mjutils._mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
        specs.append((model.name, mjSpec, model.attachment_link))
        if model.pose is not None:
            raise NotImplementedError(f"Error adding model '{model.name}' ModelSpawnDef.pose is not supported yet")
    big_speck = mjutils._MjSpec()

    world_frame = big_speck.worldbody.add_frame()
    big_speck.compiler.degree = False
    big_speck.compiler.inertiafromgeom = mjutils._mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    # big_speck.compiler.discardvisual = False
    for mname, spec, attachment_link in specs:
        ggLog.info(f"Attaching model '{mname}' to '{attachment_link}'")
        if model_element_separator in mname:
            raise RuntimeError(f"Cannot have models with '#' in their name (this character is used internally). Found model named {mname}")
        # add all th bodies that are direct childern of worldbody
        body = spec.worldbody.first_body()
        # spec.compiler.discardvisual = False
        if spec.compiler.degree:
            raise NotImplementedError(f"model {mname} uses degrees instead of radians.")
        while body is not None:
            if attachment_link is None:
                world_frame.attach_body(body, mname+model_element_separator, "")
            else:
                parentbody : mjutils._MjsBody = big_speck.body(model_element_separator.join(attachment_link))
                f = parentbody.add_frame()
                f.attach_body(body, mname+model_element_separator, "")
                # ggLog.info(f"Attaching body '{mname}';'{body.name}' to '{attachment_link}'")
            body = spec.worldbody.next_body(body)

    if geom_overrides:
        geoms_by_name = {g.name: g for g in big_speck.geoms}
        for geom_name, field_overrides in geom_overrides.items():
            if geom_name not in geoms_by_name:
                raise RuntimeError(f"geom_overrides: no geom named '{geom_name}' in the merged spec. "
                                   f"Available geoms: {sorted(geoms_by_name.keys())}")
            geom = geoms_by_name[geom_name]
            for field, new_value in field_overrides.items():
                if not hasattr(geom, field):
                    raise RuntimeError(f"geom_overrides: geom '{geom_name}' has no field '{field}'")
                if type(new_value) != type(getattr(geom, field)):
                    raise RuntimeError(f"geom_overrides: geom '{geom_name}' field '{field}' has type {type(getattr(geom, field))} but got value of type {type(new_value)}")
                setattr(geom, field, new_value)
                ggLog.info(f"geom_overrides: set {geom_name}.{field} = {new_value}")

    if adarl_overrides:
        geoms_by_name = {g.name: g for g in big_speck.geoms}
        for geom_name, directives in adarl_overrides.items():
            if geom_name not in geoms_by_name:
                raise RuntimeError(f"<adarl>: no geom named '{geom_name}' in the merged spec "
                                   f"(it may have been discarded, e.g. a visual geom with discardvisual on). "
                                   f"Available geoms: {sorted(geoms_by_name.keys())}")
            geom = geoms_by_name[geom_name]
            for directive_tag, attrib in directives:
                apply_adarl_directive(geom, directive_tag, attrib)
                ggLog.info(f"<adarl>: applied <{directive_tag} {attrib}> to geom '{geom_name}'")

    # Inject one mjSENS_CONTACT sensor per monitored body pair (warp backend only — JAX reads
    # mjx_data.contact directly). Sensors are named __pair_<i>__ so adrs can be recovered
    # post-compile by name. intprm encoding: [num, reduce_code, data_field_bitmask] where
    # data="found" maps to bitmask 1.
    if contact_pairs:
        for i, (body_a, body_b) in enumerate(contact_pairs):
            if big_speck.body(body_a) is None:
                raise RuntimeError(f"contact_pairs[{i}]: no body named '{body_a}' in the merged spec")
            if big_speck.body(body_b) is None:
                raise RuntimeError(f"contact_pairs[{i}]: no body named '{body_b}' in the merged spec")
            sens = big_speck.add_sensor()
            sens.name = f"__pair_{i}__"
            sens.type = mjutils._mjtSensor.mjSENS_CONTACT
            sens.objtype = mjutils._mjtObj.mjOBJ_BODY
            sens.objname = body_a
            sens.reftype = mjutils._mjtObj.mjOBJ_BODY
            sens.refname = body_b
            sens.intprm = [1, 0, 1]

    # big_speck.compiler.discardvisual = False
    big_speck.memory = 50*1024*1024 #allocate 50mb for arena (this becomes mjmodel.narena and mjdata.narena)
    big_speck = apply_opt_preset_to_spec(big_speck, preset_name=opt_preset, opt_override=opt_overrides)
    big_speck = apply_dof_overrides_to_spec(big_speck, 
                                            revolute_dof_armature_override,
                                            revolute_dof_damping_override,
                                            revolute_dof_frictionloss_override,
                                            safe_revolute_dof_armature,
                                            safe_revolute_dof_damping,
                                            safe_revolute_dof_frictionloss)
    mj_model = big_speck.compile()
    if log_folder is not None:
        with open(log_folder+"/aggregated_model.xml", "w") as text_file:
                text_file.write(big_speck.to_xml())
    return mj_model, big_speck

def apply_dof_overrides(mj_model : mjutils._MjModel,
                        revolute_dof_armature_override : float | None = None,
                        revolute_dof_damping_override : float | None = None,
                        revolute_dof_frictionloss_override : float | None = None,
                        safe_revolute_dof_armature = 0.01,
                        safe_revolute_dof_damping = 1.0,
                        safe_revolute_dof_frictionloss = 0.2):
    for dof_id in range(mj_model.nv):
        joint_type = mj_model.jnt_type[mj_model.dof_jntid[dof_id]]
        if joint_type == mjutils._mjtJoint.mjJNT_HINGE:
            if mj_model.dof_armature[dof_id] == 0:
                ggLog.warn(f"Revolute dof {dof_id} has zero armature. Setting it to {safe_revolute_dof_armature}. Override with MjxAdapter constructor argument 'revolute_dof_armature_override'.")
                mj_model.dof_armature[dof_id] = safe_revolute_dof_armature
            if revolute_dof_armature_override is not None:
                ggLog.info(f"Overriding revolute dof {dof_id} armature to {revolute_dof_armature_override} (was {mj_model.dof_armature[dof_id]}), due to MjxAdapter constructor argument 'revolute_dof_armature_override'.")
                mj_model.dof_armature[dof_id] = revolute_dof_armature_override

            if mj_model.dof_frictionloss[dof_id] == 0:
                ggLog.warn(f"Revolute dof {dof_id} has zero frictionloss. Setting it to {safe_revolute_dof_frictionloss}.")
                mj_model.dof_frictionloss[dof_id] = safe_revolute_dof_frictionloss
            if revolute_dof_frictionloss_override is not None:
                ggLog.info(f"Overriding revolute dof {dof_id} frictionloss to {revolute_dof_frictionloss_override} (was {mj_model.dof_frictionloss[dof_id]}), due to MjxAdapter constructor argument 'revolute_dof_frictionloss_override'.")
                mj_model.dof_frictionloss[dof_id] = revolute_dof_frictionloss_override

            if mj_model.dof_damping[dof_id] == 0:
                ggLog.warn(f"Revolute dof {dof_id} has zero damping. Setting it to {safe_revolute_dof_damping}.")
                mj_model.dof_damping[dof_id] = safe_revolute_dof_damping
            if revolute_dof_damping_override is not None:
                ggLog.info(f"Overriding revolute dof {dof_id} damping to {revolute_dof_damping_override} (was {mj_model.dof_damping[dof_id]}), due to MjxAdapter constructor argument 'revolute_dof_damping_override'.")
                mj_model.dof_damping[dof_id] = revolute_dof_damping_override
    return mj_model

def apply_dof_overrides_to_spec(spec : mjutils._MjSpec,
                                revolute_dof_armature_override : float | None = None,
                                revolute_dof_damping_override : float | None = None,
                                revolute_dof_frictionloss_override : float | None = None,
                                safe_revolute_dof_armature = 0.01,
                                safe_revolute_dof_damping = 1.0,
                                safe_revolute_dof_frictionloss = 0.2):
    """Same as apply_dof_overrides, but operates on an (uncompiled) MjSpec instead of a compiled MjModel.

    A hinge joint has exactly one DOF, so the spec joint's scalar armature/damping/frictionloss
    fields correspond directly to the per-DOF fields of the compiled model.
    """
    for joint in spec.joints:
        if joint.type == mjutils._mjtJoint.mjJNT_HINGE:
            if joint.armature == 0:
                ggLog.warn(f"Revolute joint '{joint.name}' has zero armature. Setting it to {safe_revolute_dof_armature}. Override with MjxAdapter constructor argument 'revolute_dof_armature_override'.")
                joint.armature = safe_revolute_dof_armature
            if revolute_dof_armature_override is not None:
                ggLog.info(f"Overriding revolute joint '{joint.name}' armature to {revolute_dof_armature_override} (was {joint.armature}), due to MjxAdapter constructor argument 'revolute_dof_armature_override'.")
                joint.armature = revolute_dof_armature_override

            if joint.frictionloss == 0:
                ggLog.warn(f"Revolute joint '{joint.name}' has zero frictionloss. Setting it to {safe_revolute_dof_frictionloss}.")
                joint.frictionloss = safe_revolute_dof_frictionloss
            if revolute_dof_frictionloss_override is not None:
                ggLog.info(f"Overriding revolute joint '{joint.name}' frictionloss to {revolute_dof_frictionloss_override} (was {joint.frictionloss}), due to MjxAdapter constructor argument 'revolute_dof_frictionloss_override'.")
                joint.frictionloss = revolute_dof_frictionloss_override

            # In mujoco >=3.9 the spec joint 'damping' is a per-DOF vector (shape (3,)) to
            # support ball/free joints; for a hinge only index 0 is meaningful. Older versions
            # expose it as a scalar, so handle both.
            damping_is_vec = getattr(joint.damping, "ndim", 0) > 0
            if (joint.damping[0] if damping_is_vec else joint.damping) == 0:
                ggLog.warn(f"Revolute joint '{joint.name}' has zero damping. Setting it to {safe_revolute_dof_damping}.")
                if damping_is_vec:
                    joint.damping[0] = safe_revolute_dof_damping
                else:
                    joint.damping = safe_revolute_dof_damping
            if revolute_dof_damping_override is not None:
                old_damping = joint.damping[0] if damping_is_vec else joint.damping
                ggLog.info(f"Overriding revolute joint '{joint.name}' damping to {revolute_dof_damping_override} (was {old_damping}), due to MjxAdapter constructor argument 'revolute_dof_damping_override'.")
                if damping_is_vec:
                    joint.damping[0] = revolute_dof_damping_override
                else:
                    joint.damping = revolute_dof_damping_override
    return spec

def _apply_opt_preset_to_opt(opt, preset_name : str | None, opt_override : dict[str,Any] | None,
                             opt_override_enableflags : dict[str,bool] | None = None) -> None:
    """Apply a solver/integrator preset and overrides onto an options object in-place.

    `opt` may be either a compiled model's options (mj_model.opt) or a spec's options
    (spec.option) -- both expose the same fields, so the logic is shared by
    apply_opt_preset (acts on an MjModel) and apply_opt_preset_to_spec (acts on an MjSpec).
    """
    good_impratio = 1.0
    mjINT_EULER = mjutils._mjtIntegrator.mjINT_EULER # type: ignore
    mjDSBL_EULERDAMP = mjutils._mjtDisableBit.mjDSBL_EULERDAMP # type: ignore
    mjtEnableBit = mujoco.mjtEnableBit # type: ignore
    # good_cone = mujoco.mjtCone.mjCONE_PYRAMIDAL #ELLIPTIC
    if preset_name is None or preset_name == "mujoco_default":
        pass
    elif preset_name == "fastest":
        # Inspired from barkour example
        opt.integrator = mjINT_EULER
        opt.iterations = 1 # constraint solver iterations
        opt.ls_iterations = 5 # doc: "Ensures that at most iterations times ls_iterations linesearch iterations are performed during each constraint solve"
        opt.disableflags |= mjDSBL_EULERDAMP
        opt.impratio = good_impratio # see comment above
    elif preset_name == "faster":
        opt.integrator = mjINT_EULER
        opt.iterations = 3
        opt.ls_iterations = 3
        opt.disableflags |= mjDSBL_EULERDAMP
        opt.noslip_iterations = 0 #3 # may cause instability (https://mujoco.readthedocs.io/en/latest/modeling.html#solver-settings)
        opt.impratio = good_impratio # see comment above
    elif preset_name == "fast":
        opt.integrator = mjINT_EULER
        opt.iterations = 10
        opt.ls_iterations = 5
        # opt.disableflags |= mujoco_mjtDisableBit.mjDSBL_EULERDAMP
        opt.impratio = good_impratio # see comment above
    elif preset_name == "medium":
        opt.integrator = mjINT_EULER
        opt.iterations = 20
        opt.ls_iterations = 5
        # opt.disableflags |= mujoco_mjtDisableBit.mjDSBL_EULERDAMP
        opt.impratio = good_impratio # see comment above
    elif preset_name == "slow":
        opt.integrator = mjINT_EULER
        opt.iterations = 30
        opt.ls_iterations = 5
        # opt.disableflags |= mujoco_mjtDisableBit.mjDSBL_EULERDAMP
        opt.impratio = good_impratio # see comment above
    elif preset_name == "slower":
        opt.integrator = mjINT_EULER
        opt.iterations = 50
        opt.ls_iterations = 5
        # opt.disableflags |= mujoco_mjtDisableBit.mjDSBL_EULERDAMP
        opt.impratio = good_impratio # see comment above
    else:
        raise RuntimeError(f"Unknown opt preset '{preset_name}'")
    ggLog.info(f"opt_override = {opt_override}, opt_override_enableflags = {opt_override_enableflags}")
    if opt_override is not None:
        for k,v in opt_override.items():
            if not hasattr(opt, k):
                raise RuntimeError(f"opt_override: no field named '{k}' in mujoco_MjOption")
            setattr(opt,k,v)
    if opt_override_enableflags is not None:
        for f,v in opt_override_enableflags.items():
            if v:
                opt.enableflags |= getattr(mjtEnableBit, f)
            else:
                opt.enableflags &= ~getattr(mjtEnableBit, f)
    # opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE


def apply_opt_preset(mj_model : mjutils._MjModel, preset_name : str | None, opt_override : dict[str,Any] | None,
                     opt_override_enableflags : dict[str,bool] | None = None) -> mjutils._MjModel:
    """Apply a solver/integrator preset and overrides onto a compiled MjModel (in-place)."""
    _apply_opt_preset_to_opt(mj_model.opt, preset_name, opt_override, opt_override_enableflags)
    return mj_model


def apply_opt_preset_to_spec(spec : mjutils._MjSpec, preset_name : str | None, opt_override : dict[str,Any] | None,
                             opt_override_enableflags : dict[str,bool] | None = None) -> mjutils._MjSpec:
    """Apply a solver/integrator preset and overrides onto an MjSpec (in-place).

    Spec-level counterpart of apply_opt_preset: it writes to spec.option, so the settings
    are baked in at compile time rather than mutated on an already-compiled model. Useful
    when you still need to compile (e.g. for MJX) and want the options set on the spec.
    """
    _apply_opt_preset_to_opt(spec.option, preset_name, opt_override, opt_override_enableflags)
    return spec


def format_mj_model(mj_model : mjutils._MjModel, *, full_dump : bool = False) -> str:
    """Return a structured human-readable description of a mujoco_MjModel.

    `print(mj_model)` is uninformative and `mujoco_mj_printModel` produces a wall
    of low-level array dumps. This walks the main tables (bodies / joints /
    dofs / actuators / sensors / cameras / geoms / sites) showing names, indices,
    and the most useful per-element fields.

    Parameters
    ----------
    full_dump : bool
        If True, append the full mujoco_mj_printModel output (very long).
    """
    M = mj_model
    out : list[str] = []

    def name_of(objtype : int, oid : int, default : str = "<unnamed>") -> str:
        n = mjutils._mj_id2name(M, objtype, int(oid))
        return n if n is not None else default

    def fmt_arr(a) -> str:
        return np.array2string(np.asarray(a), precision=3, suppress_small=True, separator=" ")

    mjtJoint = mjutils._mjtJoint # type: ignore
    jtype_map = {
        int(mjtJoint.mjJNT_FREE):  "FREE",
        int(mjtJoint.mjJNT_BALL):  "BALL",
        int(mjtJoint.mjJNT_SLIDE): "SLIDE",
        int(mjtJoint.mjJNT_HINGE): "HINGE",
    }

    out.append("=== MjModel ===")
    out.append(f"  nq={M.nq}  nv={M.nv}  na={M.na}  nu={M.nu}")
    out.append(f"  nbody={M.nbody}  njnt={M.njnt}  ngeom={M.ngeom}  nsite={M.nsite}")
    out.append(f"  nsensor={M.nsensor}  ncam={M.ncam}  nlight={M.nlight}  neq={M.neq}  ntendon={M.ntendon}")
    out.append(f"  nmesh={M.nmesh}  nhfield={M.nhfield}  ntex={M.ntex}  nmat={M.nmat}")
    out.append(f"  opt: timestep={M.opt.timestep}  integrator={int(M.opt.integrator)}  "
               f"iterations={M.opt.iterations}  ls_iterations={M.opt.ls_iterations}  "
               f"impratio={M.opt.impratio}")

    mjtObj = mjutils._mjtObj # type: ignore
    out.append("")
    out.append("=== Bodies ===")
    for bid in range(M.nbody):
        pid = int(M.body_parentid[bid])
        pname = name_of(mjtObj.mjOBJ_BODY, pid) if pid != bid else "<root>"
        out.append(f"  [{bid:3d}] {name_of(mjtObj.mjOBJ_BODY, bid):30s} "
                   f"parent={pname:30s} mass={float(M.body_mass[bid]):10.4g}  "
                   f"pos={fmt_arr(M.body_pos[bid])}  ipos={fmt_arr(M.body_ipos[bid])}")

    out.append("")
    out.append("=== Joints ===")
    for jid in range(M.njnt):
        jtype = jtype_map.get(int(M.jnt_type[jid]), str(int(M.jnt_type[jid])))
        body  = name_of(mjtObj.mjOBJ_BODY, int(M.jnt_bodyid[jid]))
        out.append(f"  [{jid:3d}] {name_of(mjtObj.mjOBJ_JOINT, jid):30s} "
                   f"type={jtype:5s}  body={body:30s} "
                   f"qpos[{int(M.jnt_qposadr[jid]):3d}] dof[{int(M.jnt_dofadr[jid]):3d}]  "
                   f"limited={bool(M.jnt_limited[jid])}  range={fmt_arr(M.jnt_range[jid])}")

    out.append("")
    out.append("=== DoFs ===")
    for did in range(M.nv):
        jid = int(M.dof_jntid[did])
        jname = name_of(mjtObj.mjOBJ_JOINT, jid)
        jtype = jtype_map.get(int(M.jnt_type[jid]), str(int(M.jnt_type[jid])))
        out.append(f"  [{did:3d}] {jname:30s} {jtype:5s}  "
                   f"damping={float(M.dof_damping[did]):10.4g}  "
                   f"armature={float(M.dof_armature[did]):10.4g}  "
                   f"frictionloss={float(M.dof_frictionloss[did]):10.4g}")

    if M.nu > 0:
        out.append("")
        out.append("=== Actuators ===")
        for aid in range(M.nu):
            trntype = int(M.actuator_trntype[aid])
            trnid   = int(M.actuator_trnid[aid, 0])
            if trntype == int(mjutils._mjtTrn.mjTRN_JOINT):
                target = "joint:" + name_of(mjtObj.mjOBJ_JOINT, trnid)
            else:
                target = f"trntype={trntype} trnid={trnid}"
            out.append(f"  [{aid:3d}] {name_of(mjtObj.mjOBJ_ACTUATOR, aid):30s} "
                       f"target={target:38s} ctrlrange={fmt_arr(M.actuator_ctrlrange[aid])}  "
                       f"forcerange={fmt_arr(M.actuator_forcerange[aid])}  "
                       f"gear={fmt_arr(M.actuator_gear[aid])}")

    if M.nsensor > 0:
        out.append("")
        out.append("=== Sensors ===")
        for sid in range(M.nsensor):
            objtype = int(M.sensor_objtype[sid])
            objid   = int(M.sensor_objid[sid])
            obj = name_of(objtype, objid) if objtype != 0 else "-"
            out.append(f"  [{sid:3d}] {name_of(mjtObj.mjOBJ_SENSOR, sid):30s} "
                       f"type={int(M.sensor_type[sid]):3d}  adr={int(M.sensor_adr[sid]):4d}  "
                       f"dim={int(M.sensor_dim[sid]):2d}  obj={obj}")

    if M.ncam > 0:
        out.append("")
        out.append("=== Cameras ===")
        for cid in range(M.ncam):
            body = name_of(mjtObj.mjOBJ_BODY, int(M.cam_bodyid[cid]))
            out.append(f"  [{cid:3d}] {name_of(mjtObj.mjOBJ_CAMERA, cid):30s} "
                       f"body={body:30s} mode={int(M.cam_mode[cid])}  "
                       f"pos={fmt_arr(M.cam_pos[cid])}  fovy={float(M.cam_fovy[cid]):.2f}")

    if M.ngeom > 0:
        out.append("")
        out.append("=== Geoms ===")
        for gid in range(M.ngeom):
            body = name_of(mjtObj.mjOBJ_BODY, int(M.geom_bodyid[gid]))
            out.append(f"  [{gid:3d}] {name_of(mjtObj.mjOBJ_GEOM, gid):30s} "
                       f"type={int(M.geom_type[gid]):2d}  body={body:30s} "
                       f"size={fmt_arr(M.geom_size[gid])}  pos={fmt_arr(M.geom_pos[gid])}  "
                       f"contype={int(M.geom_contype[gid])} conaffinity={int(M.geom_conaffinity[gid])}")

    if M.nsite > 0:
        out.append("")
        out.append("=== Sites ===")
        for sid in range(M.nsite):
            body = name_of(mjtObj.mjOBJ_BODY, int(M.site_bodyid[sid]))
            out.append(f"  [{sid:3d}] {name_of(mjtObj.mjOBJ_SITE, sid):30s} "
                       f"body={body:30s} pos={fmt_arr(M.site_pos[sid])}")

    out.append("")
    out.append("=== opt ===")
    out.append(str(M.opt))

    if full_dump:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.close()
        try:
            mjutils._mj_printModel(M, tmp.name)
            with open(tmp.name, "r") as f:
                out.append("")
                out.append("=== mj_printModel (verbose) ===")
                out.append(f.read())
        finally:
            os.unlink(tmp.name)

    return "\n".join(out)


def print_mj_model(mj_model : mjutils._MjModel, *, full_dump : bool = False, file : str | None = None) -> None:
    """Pretty-print a mujoco_MjModel. See `format_mj_model` for options.

    If `file` is given, the text is written to that path instead of stdout.
    """
    text = format_mj_model(mj_model, full_dump=full_dump)
    if file is not None:
        with open(file, "w") as f:
            f.write(text)
    else:
        print(text)
