"""Aggregate models with adarl's aggregate_models and write the resulting MJCF, e.g. to feed another simulator.

    adarl adapters.mjcf_export scene.yaml [-o out.xml]
    python -m adarl.adapters.mjcf_export scene.yaml [-o out.xml]

Without -o the MJCF goes to stdout (everything else goes to stderr), so the command can be used where a command
that prints a model is expected. The scene file:

    pydescriptions: [kyon_urdf, zed_description]  # optional: description packages resolved with pydescriptions
    models:
      - name: kyon
        file: pydescriptions://kyon_urdf/urdf/kyon.urdf.xacro   # or pkg:<python package>/<path>, or a path
        format: urdf.xacro       # urdf, mjcf, urdf.xacro or mjcf.xacro; inferred from the file name if omitted
        kwargs: {legs: "false"}  # xacro arguments
        pkg_paths: {kyon_urdf: /path/to/kyon_urdf}   # where xacro finds packages ($(find ...), package://)
        prefix: false            # keep the model's own names instead of prefixing them with "kyon#"
      - name: cube
        file: pkg:adarl_envs/models/cube.urdf.xacro
        attachment_link: [kyon, base_link]   # optional: attach to this body of another model instead of the world
      - name: marker
        definition: "<mujoco>...</mujoco>"   # inline, instead of file
        format: mjcf
    options:                     # keyword arguments of aggregate_models
      add_ground: true
      opt_preset: ...
    collisions:                  # optional: what collides with what, as build_scenario's default_link_group_collisions
      - link: [kyon, arm1_7]     # a link [model, link], or a model name for all its links
        collides_with: [ground, cube, [kyon, pelvis]]

Relative file paths, also in pkg_paths, are relative to the scene file, and pkg:<package>[/<path>] is a path inside
an installed python package. Mesh paths in the output are those of the models, which xacro resolves to absolute
paths, so they must exist where the MJCF is loaded.

The packages listed in pydescriptions are resolved once with pydescriptions.resolve, as the environments do, and
given to the xacro of every model as package paths (a model's own pkg_paths take precedence).
pydescriptions://<package>/<path> is a path inside one of them, usable wherever a path is. The list form uses
pydescriptions' cache, at the pinned commits. To use workspace checkouts instead, e.g. the ones the environments of
an editable adarl_envs use, give a mapping:

    pydescriptions:
      packages: [kyon_urdf, zed_description]
      root: pkg:adarl_envs/descriptions      # pydescriptions.resolve's root
      refs: {iit-kyon-ros-pkg: my-branch}    # optional, needs root

With collisions, the contype and conaffinity of the geoms are set as MjxAdapter sets them: each link collides only
with the links it is listed with, and a link that is not listed collides with nothing. Visual geoms (contype and
conaffinity 0) are left as they are. The links of an unprefixed model are named by its model name too, which needs
it to be the only unprefixed model.
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_FORMAT_SUFFIXES = {".urdf.xacro" : "urdf.xacro", ".mjcf.xacro" : "mjcf.xacro", ".xml.xacro" : "mjcf.xacro",
                    ".urdf" : "urdf", ".mjcf" : "mjcf", ".xml" : "mjcf"}
_SCENE_KEYS = {"pydescriptions", "models", "options", "collisions"}
_PYDESCRIPTIONS_KEYS = {"packages", "root", "refs"}
_PYDESCRIPTIONS_SCHEME = "pydescriptions://"
_MODEL_KEYS = {"name", "file", "definition", "format", "kwargs", "pkg_paths", "prefix", "attachment_link"}
_DEFAULT_OPTIONS = {"add_ground" : True, "add_sky" : True, "discardvisual" : False, "log_folder" : None}

LinkSpec = tuple[str, str] | str
LinkGroupCollisions = list[tuple[LinkSpec, list[LinkSpec]]]


def _resolve_file(file : str, base_dir : Path, descriptions : dict[str, str] = {}) -> Path:
    if file.startswith(_PYDESCRIPTIONS_SCHEME):
        package, _, resource = file[len(_PYDESCRIPTIONS_SCHEME):].partition("/")
        if package not in descriptions:
            raise ValueError(f"'{file}': package '{package}' is not in the scene's pydescriptions list "
                             f"({sorted(descriptions)})")
        return Path(descriptions[package])/resource
    if file.startswith("pkg:"):
        from adarl.utils.utils import pkgutil_get_path
        package, _, resource = file[len("pkg:"):].partition("/")
        if not package:
            raise ValueError(f"'{file}' is not in the form pkg:<package>[/<path>]")
        return Path(pkgutil_get_path(package, resource or None))
    path = Path(file).expanduser()
    return path if path.is_absolute() else base_dir / path


def _infer_format(file : str) -> str:
    for suffix, fmt in _FORMAT_SUFFIXES.items():
        if file.endswith(suffix):
            return fmt
    raise ValueError(f"Cannot infer the format of '{file}', give it with 'format'")


def _load_pydescriptions(entry : Any, base_dir : Path) -> dict[str, str]:
    """package -> path of the description packages listed in a scene's pydescriptions entry."""
    if isinstance(entry, list):
        entry = {"packages" : entry}
    if not isinstance(entry, dict):
        raise ValueError(f"pydescriptions must be a list of packages or a mapping with keys "
                         f"{sorted(_PYDESCRIPTIONS_KEYS)}, got {entry}")
    unknown = set(entry) - _PYDESCRIPTIONS_KEYS
    if unknown:
        raise ValueError(f"pydescriptions: unknown keys {sorted(unknown)}, expected some of {sorted(_PYDESCRIPTIONS_KEYS)}")
    packages = entry.get("packages")
    if not isinstance(packages, list) or not packages or not all(isinstance(p, str) for p in packages):
        raise ValueError(f"pydescriptions: packages must be a non-empty list of package names, got {packages}")
    try:
        import pydescriptions
    except ImportError as e:
        raise ImportError("The scene lists pydescriptions packages, but pydescriptions is not installed") from e
    root = entry.get("root")
    return pydescriptions.resolve(packages,
                                  root = None if root is None else _resolve_file(str(root), base_dir),
                                  refs = entry.get("refs"))


def _link_spec(spec : Any, where : str) -> LinkSpec:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (str(spec[0]), str(spec[1]))
    raise ValueError(f"{where}: a link is [model, link] or a model name, got {spec}")


def _load_collisions(entries : Any) -> LinkGroupCollisions:
    if not isinstance(entries, list):
        raise ValueError(f"collisions must be a list of {{link, collides_with}} entries, got {entries}")
    collisions = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"link", "collides_with"}:
            raise ValueError(f"collisions[{i}] must have exactly the keys 'link' and 'collides_with', got {entry}")
        if not isinstance(entry["collides_with"], list):
            raise ValueError(f"collisions[{i}]: collides_with must be a list, got {entry['collides_with']}")
        collisions.append((_link_spec(entry["link"], f"collisions[{i}].link"),
                           [_link_spec(c, f"collisions[{i}].collides_with") for c in entry["collides_with"]]))
    return collisions


def load_scene(scene : dict[str, Any], base_dir : Path):
    """The ModelSpawnDefs, the names of the models to leave unprefixed, the aggregate_models keyword arguments and
    the collisions (None if not given) described by a scene (the content of a scene file, see the module docstring)."""
    from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
    from adarl.utils.utils import compile_xacro_string
    unknown = set(scene) - _SCENE_KEYS
    if unknown:
        raise ValueError(f"Unknown scene keys {sorted(unknown)}, expected some of {sorted(_SCENE_KEYS)}")
    if not scene.get("models"):
        raise ValueError("The scene has no models")
    descriptions = _load_pydescriptions(scene["pydescriptions"], base_dir) if scene.get("pydescriptions") else {}
    models = []
    unprefixed = set()
    for i, m in enumerate(scene["models"]):
        unknown = set(m) - _MODEL_KEYS
        if unknown:
            raise ValueError(f"Model {i}: unknown keys {sorted(unknown)}, expected some of {sorted(_MODEL_KEYS)}")
        if "name" not in m:
            raise ValueError(f"Model {i} has no name")
        if ("file" in m) == ("definition" in m):
            raise ValueError(f"Model '{m['name']}' needs exactly one of 'file' and 'definition'")
        if "file" in m:
            definition = _resolve_file(m["file"], base_dir, descriptions).read_text()
            fmt = m.get("format") or _infer_format(m["file"])
        else:
            definition = m["definition"]
            if "format" not in m:
                raise ValueError(f"Model '{m['name']}' has an inline definition, it needs a 'format'")
            fmt = m["format"]
        fmt = fmt.strip().lower()
        kwargs = {str(k) : str(v) for k, v in (m.get("kwargs") or {}).items()}
        if fmt.endswith(".xacro"): # compiled here, as aggregate_models cannot be given package paths
            pkg_paths = {**descriptions,
                         **{str(k) : str(_resolve_file(str(v), base_dir, descriptions))
                            for k, v in (m.get("pkg_paths") or {}).items()}}
            definition = compile_xacro_string(model_definition_string = definition,
                                              model_kwargs = kwargs,
                                              extra_pkg_paths = pkg_paths)
            fmt = fmt[:-len(".xacro")]
        elif m.get("kwargs") or m.get("pkg_paths"):
            raise ValueError(f"Model '{m['name']}': kwargs and pkg_paths are for xacro formats, the format is {fmt}")
        attachment_link = m.get("attachment_link")
        if attachment_link is not None:
            if len(attachment_link) != 2:
                raise ValueError(f"Model '{m['name']}': attachment_link must be [model, link], got {attachment_link}")
            attachment_link = (str(attachment_link[0]), str(attachment_link[1]))
        models.append(ModelSpawnDef(name = m["name"],
                                    definition_string = definition,
                                    format = fmt,
                                    pose = None,
                                    kwargs = {},
                                    attachment_link = attachment_link))
        if not m.get("prefix", True):
            unprefixed.add(m["name"])
    options = {**_DEFAULT_OPTIONS, **(scene.get("options") or {})}
    for key in ("ground_rgba", "orthographic_camera"):
        if options.get(key) is not None:
            options[key] = tuple(tuple(v) if isinstance(v, list) else v for v in options[key])
    collisions = _load_collisions(scene["collisions"]) if scene.get("collisions") is not None else None
    return models, unprefixed, options, collisions


def model_links(mj_model, unprefixed : set[str]) -> dict[tuple[str, str], int]:
    """The bodies of an aggregated model as (model_name, link_name) -> body id. The bodies of an unprefixed model
    get its name if it is the only unprefixed model; with more than one there is no telling them apart."""
    from adarl.adapters.mujoco_utils import model_element_separator
    links = {}
    for body_id in range(mj_model.nbody):
        name = mj_model.body(body_id).name
        if name == "world":
            link = ("world", "world")
        elif name == "":
            continue # cannot be named
        elif model_element_separator in name:
            model, _, link_name = name.partition(model_element_separator)
            link = (model, link_name)
        elif len(unprefixed) == 1:
            link = (next(iter(unprefixed)), name)
        else:
            raise ValueError(f"Body '{name}' has no model prefix and there are {len(unprefixed)} unprefixed models "
                             f"({sorted(unprefixed)}): collisions need at most one unprefixed model")
        links[link] = body_id
    return links


def apply_collisions(spec, mj_model, unprefixed : set[str], collisions : LinkGroupCollisions):
    """Set the contype and conaffinity of the geoms of spec (compiled into mj_model) as MjxAdapter does for
    default_link_group_collisions."""
    from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
    from adarl.adapters.mujoco_utils import compute_collision_masks
    links = model_links(mj_model, unprefixed)
    expanded, _ = BaseVecSimulationAdapter.expand_link_group_collisions(collisions, links.keys())
    geom_contype, geom_conaffinity, _, _, _ = compute_collision_masks(mj_model, links, expanded)
    for geom in spec.geoms:
        geom.contype = int(geom_contype[geom.id])
        geom.conaffinity = int(geom_conaffinity[geom.id])


def export_mjcf(scene : dict[str, Any], base_dir : Path) -> str:
    """The MJCF of a scene, aggregated by aggregate_models."""
    from adarl.adapters.mujoco_utils import aggregate_models
    models, unprefixed, options, collisions = load_scene(scene, base_dir)
    mj_model, spec = aggregate_models(models, unprefixed_models = frozenset(unprefixed), **options)
    if collisions is not None:
        apply_collisions(spec, mj_model, unprefixed, collisions)
    return spec.to_xml()


@contextlib.contextmanager
def _stdout_to_stderr():
    """Send everything written to stdout, also by native code, to stderr."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main():
    p = argparse.ArgumentParser(description="Aggregate models with adarl's aggregate_models and write the MJCF.")
    p.add_argument("scene", help="scene yaml file: models, aggregate_models options and collisions")
    p.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    args = p.parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    scene = yaml.safe_load(scene_path.read_text()) or {}
    with _stdout_to_stderr():
        mjcf = export_mjcf(scene, scene_path.parent)
    if args.output is None:
        sys.stdout.write(mjcf)
        sys.stdout.flush()
    else:
        Path(args.output).write_text(mjcf)
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
