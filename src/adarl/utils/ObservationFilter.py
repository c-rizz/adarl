"""Named filtering of dictionary observations into vector/image groups.

An :class:`ObservationFilter` is built once from a ``Dict`` observation space plus a list of
:class:`Filter` definitions, and then answers two questions cheaply:

    obs_filter.get("privileged", obs)   -> the filtered observation
    obs_filter.get_space("privileged")  -> the matching observation space

Each filter names a subset of the *top-level* keys of the observation. With ``concat=True`` the
selected (vector) components are concatenated into a single tensor, and the corresponding space is
a single :class:`~adarl.utils.spaces.ThBox` whose limits and labels are the concatenation of the
components' ones. With ``concat=False`` the filter yields a plain dict subset and a ``ThDict``.

Two filters are registered by default (unless a user filter takes their name, or there are no keys
of that kind): ``"vec"``, all vector components concatenated, and ``"img"``, all image components
as a dict.

Example
-------
    obs_filter = ObservationFilter(obs_space,
                                   [Filter("privileged", ["intrinsic", "extrinsic"], concat=True)])
    privileged = obs_filter.get("privileged", obs)      # a single concatenated tensor
    policy_in  = obs_filter.get("vec", obs)             # default filter
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch as th
from gymnasium import spaces as gym_spaces

import adarl.utils.spaces as spaces
from adarl.utils.utils import numpy_to_torch_dtype_dict


@dataclass(frozen=True)
class Filter:
    """Definition of a named observation filter.

    Parameters
    ----------
    name : str
        Name used to request this filter in :meth:`ObservationFilter.get`.
    keys : Sequence[str] | None
        Top-level observation keys this filter selects, in the order they should be concatenated.
        ``None`` means "every key of the observation", in the order of the observation space --
        beware that ``gymnasium.spaces.Dict`` sorts its keys alphabetically, so list the keys
        explicitly whenever the concatenation layout matters.
    concat : bool
        If True the selected components are concatenated into a single tensor along the last
        dimension. Only possible for vector (1-dimensional) components.
    """
    name: str
    keys: Sequence[str] | None = None
    concat: bool = False


class ObservationFilter:

    class KeyInfo:
        """Analysis of one top-level component of the observation space."""

        def __init__(self, key: str, space: gym_spaces.Box):
            self.key = key
            self.space = space
            self.is_img = ObservationFilter._is_img(space)
            self.shape: tuple[int, ...] = (ObservationFilter._img_shape_chw(space) if self.is_img
                                           else (int(space.shape[0]),))
            self.dtype = space.dtype
            self.dtype_th: th.dtype = numpy_to_torch_dtype_dict[np.dtype(space.dtype).type]

        @property
        def size(self) -> int:
            """Number of elements of a vector component."""
            return self.shape[0]

        def __repr__(self):
            return f"KeyInfo({self.key}, is_img={self.is_img}, shape={self.shape}, dtype={self.dtype})"

    class ResolvedFilter:
        """A :class:`Filter` resolved against a concrete observation space."""

        def __init__(self, name: str, keys: tuple[str, ...], concat: bool, space):
            self.name = name
            self.keys = keys
            self.concat = concat
            self.space = space

        def __repr__(self):
            return f"ResolvedFilter({self.name}, keys={self.keys}, concat={self.concat})"

    @staticmethod
    def _is_img(box_space: gym_spaces.Box) -> bool:
        """Whether a component is an image (2- or 3-dimensional) rather than a vector."""
        ndims = len(box_space.shape)
        if ndims == 1:
            return False
        elif ndims in (2, 3):
            return True
        else:
            raise RuntimeError(f"Unexpected observation shape {box_space.shape}, "
                               f"expected 1 dimension (vector) or 2/3 dimensions (image)")

    @staticmethod
    def _img_shape_chw(box_space: gym_spaces.Box) -> tuple[int, int, int]:
        """Image shape as (channels, height, width), adding the channel dim for 2D images."""
        shape = box_space.shape
        if len(shape) == 2:
            return (1, int(shape[0]), int(shape[1]))
        elif len(shape) == 3:
            return (int(shape[0]), int(shape[1]), int(shape[2]))
        else:
            raise RuntimeError(f"Unexpected image observation shape {shape}")

    def __init__(self,
                 observation_space: gym_spaces.Dict,
                 filters: Sequence[Filter] = (),
                 default_filters: bool = True,
                 vec_filter_name: str = "vec",
                 img_filter_name: str = "img"):
        """Build the filters for a dictionary observation space.

        Parameters
        ----------
        observation_space : gym_spaces.Dict
            Space of the observations to be filtered. Must be a flat dict of Box components.
        filters : Sequence[Filter]
            User-defined filters. A filter named like a default one replaces it.
        default_filters : bool
            Whether to also define the default ``vec``/``img`` filters, by default True.
        vec_filter_name : str
            Name of the default all-vectors filter, by default "vec".
        img_filter_name : str
            Name of the default all-images filter, by default "img".
        """
        if not isinstance(observation_space, gym_spaces.Dict):
            raise ValueError(f"observation_space must be a Dict space, but it is a "
                             f"{type(observation_space)}. If it is not, wrap it into one.")
        self._observation_space = observation_space

        self._key_infos: dict[str, ObservationFilter.KeyInfo] = {}
        for key, subspace in observation_space.spaces.items():
            if not isinstance(subspace, gym_spaces.Box):
                raise ValueError(f"Observation component '{key}' is a {type(subspace)}, but only "
                                 f"Box components are supported (nested dicts are not).")
            self._key_infos[key] = ObservationFilter.KeyInfo(key, subspace)

        self._vector_keys = tuple(k for k, i in self._key_infos.items() if not i.is_img)
        self._image_keys = tuple(k for k, i in self._key_infos.items() if i.is_img)

        filter_defs: dict[str, Filter] = {}
        if default_filters:
            # Only define a default if there is anything of that kind, so that an empty default
            # never shadows a meaningful error.
            if len(self._vector_keys) > 0:
                filter_defs[vec_filter_name] = Filter(vec_filter_name, self._vector_keys, concat=True)
            if len(self._image_keys) > 0:
                filter_defs[img_filter_name] = Filter(img_filter_name, self._image_keys, concat=False)
        for f in filters:
            filter_defs[f.name] = f # user filters override the defaults

        self._filters: dict[str, ObservationFilter.ResolvedFilter] = {
            name: self._resolve_filter(f) for name, f in filter_defs.items()}

    def _resolve_filter(self, f: Filter) -> ObservationFilter.ResolvedFilter:
        keys = tuple(self._key_infos.keys()) if f.keys is None else tuple(f.keys)
        unknown = [k for k in keys if k not in self._key_infos]
        if len(unknown) > 0:
            raise ValueError(f"Filter '{f.name}' requests unknown observation components {unknown}. "
                             f"Available components are {list(self._key_infos.keys())}")
        duplicates = [k for k in set(keys) if keys.count(k) > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Filter '{f.name}' requests duplicated components {duplicates}")
        if len(keys) == 0:
            raise ValueError(f"Filter '{f.name}' selects no observation component")
        space = self._build_concat_space(f.name, keys) if f.concat else self._build_dict_space(keys)
        return ObservationFilter.ResolvedFilter(f.name, keys, f.concat, space)

    def _build_dict_space(self, keys: Sequence[str]) -> spaces.ThDict:
        return spaces.ThDict({k: self._key_infos[k].space for k in keys})

    def _build_concat_space(self, name: str, keys: Sequence[str]) -> spaces.ThBox:
        infos = [self._key_infos[k] for k in keys]
        img_keys = [i.key for i in infos if i.is_img]
        if len(img_keys) > 0:
            raise ValueError(f"Filter '{name}' has concat=True but selects image components "
                             f"{img_keys}. Only vector components can be concatenated.")
        dtypes = {i.dtype_th for i in infos}
        if len(dtypes) > 1:
            raise ValueError(f"Filter '{name}' concatenates components of different dtypes: "
                             f"{ {i.key: i.dtype_th for i in infos} }")
        lows = np.concatenate([np.asarray(i.space.low).reshape(-1) for i in infos])
        highs = np.concatenate([np.asarray(i.space.high).reshape(-1) for i in infos])
        labels = np.concatenate([self._component_labels(i) for i in infos])
        devices = {i.space.th_device for i in infos if isinstance(i.space, spaces.ThBox)}
        th_device = next(iter(devices)) if len(devices) == 1 else th.device("cpu")
        return spaces.ThBox(low=lows, high=highs, shape=(int(lows.shape[0]),),
                            dtype=infos[0].dtype, torch_device=th_device, labels=labels)

    @staticmethod
    def _component_labels(info: ObservationFilter.KeyInfo) -> np.ndarray:
        """Labels of a vector component, prefixed with the component name to stay unambiguous."""
        if isinstance(info.space, spaces.ThBox):
            sub_labels = np.asarray(info.space.labels).reshape(-1)
        else:
            sub_labels = np.array([str(i) for i in range(info.size)], dtype=object)
        return np.array([f"{info.key}.{l}" for l in sub_labels], dtype=object)

    def get(self, name: str, obs: Mapping[str, th.Tensor]) -> th.Tensor | dict[str, th.Tensor]:
        """Apply a filter to an observation.

        Returns a single concatenated tensor if the filter has ``concat=True``, otherwise a dict
        holding the selected components. Concatenation happens along the last dimension, so
        unbatched, batched and batch-of-trajectories observations are all handled the same way.
        """
        f = self._get_filter(name)
        if f.concat:
            return th.cat([obs[k] for k in f.keys], dim=-1)
        else:
            return {k: obs[k] for k in f.keys}

    def get_space(self, name: str) -> spaces.ThBox | spaces.ThDict:
        """Space of the observations returned by :meth:`get` for this filter."""
        return self._get_filter(name).space

    def _get_filter(self, name: str) -> ObservationFilter.ResolvedFilter:
        f = self._filters.get(name)
        if f is None:
            raise KeyError(f"No filter named '{name}'. Defined filters are {self.filter_names()}")
        return f

    def filter_names(self) -> tuple[str, ...]:
        return tuple(self._filters.keys())

    def keys(self, name: str) -> tuple[str, ...]:
        """Observation components selected by this filter, in concatenation order."""
        return self._get_filter(name).keys

    def is_concat(self, name: str) -> bool:
        return self._get_filter(name).concat

    def vector_keys(self) -> tuple[str, ...]:
        return self._vector_keys

    def image_keys(self) -> tuple[str, ...]:
        return self._image_keys

    def is_image(self, key: str) -> bool:
        return self._key_infos[key].is_img

    def image_shape_chw(self, key: str) -> tuple[int, ...]:
        """Shape of an image component, as (channels, height, width)."""
        info = self._key_infos[key]
        if not info.is_img:
            raise ValueError(f"Component '{key}' is not an image")
        return info.shape

    def observation_space(self) -> gym_spaces.Dict:
        return self._observation_space

    def __contains__(self, name: str) -> bool:
        return name in self._filters

    def __repr__(self):
        return (f"ObservationFilter(filters={ {n: (f.keys, f.concat) for n, f in self._filters.items()} })")
