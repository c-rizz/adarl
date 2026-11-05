from __future__ import annotations
import gymnasium as gym
from typing import Any, SupportsFloat, Sequence, Union, Tuple, Dict, overload
from matplotlib.pylab import Generator
from numpy.typing import NDArray
import numpy as np
import torch as th
gym_spaces = gym.spaces
from copy import deepcopy
from gymnasium.vector.utils import batch_space
from adarl.utils.utils import torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict
import adarl.utils.dbg.ggLog as ggLog
from collections import OrderedDict
import numpy.typing as npt

class ThBox(gym.spaces.Box):
    def __init__(   self,
                    low: SupportsFloat | NDArray[Any] | th.Tensor,
                    high: SupportsFloat | NDArray[Any] | th.Tensor,
                    shape: Sequence[int] | None = None,
                    dtype: type[np.floating[Any]] | type[np.integer[Any]] | th.dtype | str = np.float32,
                    seed: int | None = None,
                    torch_device : th.device = th.device("cpu"),
                    labels : npt.NDArray[np.object_] | None = None,
                    generator : th.Generator | None = None,
                    default_value : th.Tensor | None = None):
        """ Box space, like the openai gym one, but based on torch Tensors, and with some additional functionality.

        Parameters
        ----------
        low : SupportsFloat | NDArray[Any] | th.Tensor
            Lower limit of the space, will be broadcasted up to the shape of the space
        high : SupportsFloat | NDArray[Any] | th.Tensor
            Higher limit of the space
        shape : Sequence[int] | None, optional
            Shape of the space, if None, it will be determined from low and high
        dtype : type[np.floating[Any]] | type[np.integer[Any]] | th.dtype | str, optional
            dtype of the space, by default np.float32
        seed : int | None, optional
            Seed or generator for the underlying gym class, by default None
        torch_device : th.device, optional
            Torch device to be used, by default th.device("cpu")
        labels : th.Tensor | np.ndarray | None, optional
            Names of the fields of this space, by default None
        generator : th.Generator | None, optional
            Torch generator used by the sample() function, by default None
        default_value : th.Tensor | None, optional
            Default value, useful for example for initializing a policy network, if it is None (low+high)/2 is used, by default None

        Raises
        ------
        RuntimeError
            _description_
        """
        self.th_device = torch_device
        self._seed = seed
        if generator is None:
            generator = th.Generator(device=self.th_device)
            if seed is not None:
                generator.manual_seed(seed)
        elif seed is not None:
            raise RuntimeError("Cannot provide both a generator and a seed")
        self._th_rng = generator
        if isinstance(low,th.Tensor):
            low = low.cpu().numpy()
        if isinstance(high,th.Tensor):
            high = high.cpu().numpy()
        if isinstance(dtype,th.dtype):
            numpy_dtype = torch_to_numpy_dtype_dict[dtype]
            torch_dtype = dtype
        else:
            numpy_dtype = dtype
            torch_dtype = numpy_to_torch_dtype_dict[dtype]
        self.torch_dtype_str = str(torch_dtype).split(".")[1] # yaml cannot save this directly as str-based __reduce__ (used by dtypes) is not supported by yaml, see https://github.com/pytorch/pytorch/issues/78720
        super().__init__(low=low,high=high,shape=shape,dtype=numpy_dtype,seed=seed)
        if labels is None:
            labels = np.array([f"{i}" for i in range(np.prod(self.shape, dtype=int))], dtype=object)
        self.labels : npt.NDArray[np.object_] = labels
        del self._np_random # disable the numpy rng, we don't use it and it is annoying to pickle through numpy 2.0/1.x
        self._high_th = th.as_tensor(self.high, device=self.th_device)
        self._low_th = th.as_tensor(self.low, device=self.th_device)
        if default_value is not None:
            self.zero_action = default_value.expand_as(self._high_th)
        else:
            self.zero_action = (self._high_th+self._low_th)/2
        self.zero_action = self.zero_action.to(self.th_device)
    
    @property
    def torch_dtype(self) -> th.dtype:
        return getattr(th,self.torch_dtype_str)

    def sample(self):
        # ggLog.info(f"Sampling ThBox, rng state = {hash_tensor(self._rng.get_state()) if self._rng is not None else None}")
        # import traceback
        # traceback.print_stack()
        # only works for uniform
        dtype = self.torch_dtype
        if dtype.is_floating_point:
            r = th.rand(self._high_th.size(),
                        device=self.th_device,
                        generator=self._th_rng,
                        dtype=dtype)
        elif self.dtype not in [th.int8, th.int16, th.int32, th.int64, th.uint8, th.uint16, th.uint32, th.uint64]:
            r = th.rand(self._high_th.size(),
                        device=self.th_device,
                        generator=self._th_rng)
            r = (r*(self._high_th-self._low_th)+self._low_th).to(dtype)
        return r*(self._high_th-self._low_th)+self._low_th
        # return th.as_tensor(super().sample(), device = self._th_device) # does not use the torch rng

    @staticmethod
    def _all_equal(a : th.Tensor, b : th.Tensor) -> bool:
        """Elementwise equality of a and b (broadcasted), with nans counted as equal to nans."""
        eq = a == b
        if a.is_floating_point():
            eq |= a.isnan() & b.isnan()
        return bool(eq.all())

    @staticmethod
    def _compact(a : np.ndarray | th.Tensor) -> float | int | list:
        """The single value of a if all its elements are equal, the nested list of its elements otherwise."""
        t = th.as_tensor(a).cpu()
        if t.numel() > 0 and ThBox._all_equal(t, t.flatten()[0]):
            return t.flatten()[0].item()
        return t.tolist()

    @staticmethod
    def _restore_rng(rng_device : th.device | str, rng_state : th.Tensor) -> th.Generator:
        rng = th.Generator(device=rng_device)
        # A space loaded onto another device type (e.g. saved on cuda, loaded on the cpu) gets a
        # generator whose state has another format: it keeps its default seed instead.
        if rng_state.numel() == rng.get_state().numel():
            rng.set_state(rng_state)
        else:
            ggLog.warn(f"ThBox: the saved rng state does not fit a generator on {rng_device} "
                       f"(it was saved from another device type), the space's rng keeps its default seed.")
        return rng

    def __getstate__(self):
        # Saved as the constructor arguments that rebuild the space, with the arrays whose elements are all
        # equal saved as a single value: saving the attributes stored every array several times over in full
        # (bounds, bounded masks, default value, labels), which for an image space is hundreds of KB.
        # Arrays are saved as lists to avoid issues with numpy 2.0/1.x
        n = int(np.prod(self.shape, dtype=int))
        labels = self.labels.tolist() if isinstance(self.labels, np.ndarray) else self.labels
        if labels == [f"{i}" for i in range(n)]:
            labels = None
        midpoint = (self._high_th+self._low_th)/2
        if (self.zero_action.dtype == midpoint.dtype and self.zero_action.shape == midpoint.shape
                and self._all_equal(self.zero_action, midpoint.to(self.zero_action.device))):
            default_value = None
        else:
            default_value = {"value" : self._compact(self.zero_action),
                             "dtype" : str(self.zero_action.dtype).split(".")[1]}
        return {"_thbox_state_format" : 2,
                "low" : self._compact(self.low),
                "high" : self._compact(self.high),
                "shape" : list(self.shape),
                "torch_dtype_str" : self.torch_dtype_str,
                "th_device" : self.th_device,
                "seed" : self._seed,
                "labels" : labels,
                "default_value" : default_value,
                "_th_rng_state" : self._th_rng.get_state().cpu().numpy().tobytes(),
                "_th_rng_device" : self._th_rng.device}

    def __setstate__(self, state):
        if state.get("_thbox_state_format") == 2:
            torch_dtype = getattr(th, state["torch_dtype_str"])
            numpy_dtype = torch_to_numpy_dtype_dict[torch_dtype]
            low, high = state["low"], state["high"]
            default_value = state["default_value"]
            if default_value is not None:
                default_value = th.as_tensor(default_value["value"], dtype=getattr(th, default_value["dtype"]))
            rng = self._restore_rng(state["_th_rng_device"], th.frombuffer(bytearray(state["_th_rng_state"]), dtype=th.uint8))
            self.__init__(low = np.asarray(low, dtype=numpy_dtype) if isinstance(low, list) else low,
                          high = np.asarray(high, dtype=numpy_dtype) if isinstance(high, list) else high,
                          shape = tuple(state["shape"]),
                          dtype = torch_dtype,
                          torch_device = state["th_device"],
                          labels = np.array(state["labels"], dtype=object) if state["labels"] is not None else None,
                          generator = rng,
                          default_value = default_value)
            self._seed = state["seed"] # set after __init__, which does not accept both a seed and a generator
            return
        # Older format: the attributes themselves
        if "_th_rng" not in state:
            state["_th_rng"] = self._restore_rng(state["_th_rng_device"], th.ByteTensor(state["_th_rng_state"]))
        self.__dict__.update(state)
        self.dtype = torch_to_numpy_dtype_dict[getattr(th,self.torch_dtype_str)]
        if isinstance(self.high,(list,int,float)):
            self.bounded_above = th.as_tensor(self.bounded_above, dtype=th.bool).cpu().numpy().astype(np.bool_)
            self.bounded_below = th.as_tensor(self.bounded_below, dtype=th.bool).cpu().numpy().astype(np.bool_)
            self.high = th.as_tensor(self.high, dtype=self.torch_dtype).cpu().numpy()
            self.low = th.as_tensor(self.low, dtype=self.torch_dtype).cpu().numpy()
            self._high_th = th.as_tensor(self._high_th, dtype=self.torch_dtype).to(self.th_device)
            self._low_th = th.as_tensor(self._low_th, dtype=self.torch_dtype).to(self.th_device)
            self.zero_action = th.as_tensor(self.zero_action, dtype=self.torch_dtype).to(self.th_device)
        else:
            # for backward compat
            self.bounded_above = th.as_tensor(self.bounded_above).cpu().numpy().astype(np.bool_)
            self.bounded_below = th.as_tensor(self.bounded_below).cpu().numpy().astype(np.bool_)
            self.high = th.as_tensor(self.high).cpu().numpy()
            self.low = th.as_tensor(self.low).cpu().numpy()
        if isinstance(self.labels,list):
            self.labels = np.array(self.labels, dtype=object)

    def seed(self, seed: int | None = None) -> list[int]:
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        if seed is not None:
            self._th_rng.manual_seed(seed)
        return super().seed(seed)


def get_space_labels(space : gym_spaces.Dict | ThBox):
    if isinstance(space, ThBox):
        return space.labels
    elif isinstance(space, gym_spaces.Dict):
        return {k: get_space_labels(space.spaces[k]) for k in space.spaces}
    elif isinstance(space, gym_spaces.Box):
        return np.array([f"{i}" for i in range(np.prod(space.shape, dtype=int))], dtype=object)
    else:
        raise NotImplemented(f"Cannot get labels from space of type {type(space)}")
    
@batch_space.register(ThBox)
def batch_space_box(space, n=1):
    # ggLog.info(f"batching space {space.low.shape}")
    low, high = np.broadcast_to(space.low, (n,)+space.low.shape), np.broadcast_to(space.high, (n,)+space.high.shape)
    # ggLog.info(f"batched lims (memsize={low.nbytes/1024/1024} MB)")
    # repeats = tuple([n] + [1] * space.low.ndim)
    # low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return ThBox(low=low, high=high, dtype=space.dtype, seed=space._seed)


class ThDict(gym_spaces.Dict):
    """
    Wrap gymnasium.spaces.Dict to avoid using numpy random generator, which is not pickleable across numpy 2.0/1.x

    """

    def __init__(self, spaces: None | dict[str, gym_spaces.Space] | Sequence[tuple[str, gym_spaces.Space]] = None, seed: dict | int | Generator | None = None, **spaces_kwargs: gym.Space):
        super().__init__(spaces, seed, **spaces_kwargs)
        del self._np_random
        self._th_rng = th.Generator(device=th.device("cpu"))
        self._seed = seed
        if seed is not None:
            thseed = self._np_random.integers(np.iinfo(np.int32).max, size=1)[0]
            self._th_rng.manual_seed(int(thseed))

    def seed(self, seed: dict[str, Any] | int | None = None) -> list[int]:
        if isinstance(seed, int):
            seeds = [seed]
            self._th_rng.manual_seed(seed)
            subseeds = th.randint(0, np.iinfo(np.int32).max, (len(self.spaces),), generator=self._th_rng,
                                  device=th.device("cpu")).tolist()
            for subspace, subseed in zip(self.spaces.values(), subseeds):
                seeds += subspace.seed(int(subseed))
            return seeds
        else:
            return super().seed(seed)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_np_random",None)
        return state
    
@batch_space.register(ThDict)
def batch_space_dict(space, n=1):
    return ThDict(
        OrderedDict(
            [
                (key, batch_space(subspace, n=n))
                for (key, subspace) in space.spaces.items()
            ]
        ),
        seed=space._seed,
    )

@overload
def get_obs_shape(
    observation_space: gym_spaces.Dict,
) -> Dict[str, Tuple[int, ...]]:
    ...

@overload
def get_obs_shape(
    observation_space: gym_spaces.Box,
) -> Tuple[int, ...]:
    ...

def get_obs_shape(
    observation_space: gym_spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, gym_spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, gym_spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, gym_spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, gym_spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, gym_spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
    
def get_1d_space_size(space : gym.Space) -> int:
    """ Returns the size of a 1D space. Useful to get the size of a one-dimensional action spae, or reward space.

    Parameters
    ----------
    space : gym.Space
        The space to evaluate

    Returns
    -------
    int
        The size of the space if it is 1D, otherwise raises an error

    """
    if isinstance(space, gym.spaces.Box):
        if len(space.shape) == 0:
            rewards_num = 1
        elif len(space.shape) == 1:
            rewards_num = space.shape[0]
        else:
            raise RuntimeError(f"AsyncProcessExperienceCollector: unsupported space shape {space.shape}, dimensionality can only be 0 or 1.")
    else:
        raise RuntimeError(f"AsyncProcessExperienceCollector: unsupported space type {space}")
    return rewards_num


def check_space_labels_mismatches(space1 : gym.spaces.Space, space2 : gym.spaces.Space, path : str = "") -> list[str]:
    """Recursively compares the labels of the labeled (e.g. ThBox) leaves of two spaces.
    Only leaves that have labels in both spaces are compared; Dict and Tuple spaces are
    traversed through their common keys/indices.

    Returns
    -------
    list[str]
        A description of each mismatch found, empty if the labels all match.
    """
    mismatches = []
    if isinstance(space1, gym.spaces.Dict) and isinstance(space2, gym.spaces.Dict):
        for k in space1.spaces.keys():
            if k in space2.spaces:
                mismatches += check_space_labels_mismatches(space1.spaces[k], space2.spaces[k], f"{path}/{k}")
    elif isinstance(space1, gym.spaces.Tuple) and isinstance(space2, gym.spaces.Tuple):
        for i, (s1, s2) in enumerate(zip(space1.spaces, space2.spaces)):
            mismatches += check_space_labels_mismatches(s1, s2, f"{path}/{i}")
    else:
        labels1 = getattr(space1, "labels", None)
        labels2 = getattr(space2, "labels", None)
        if labels1 is not None and labels2 is not None:
            labels1 = np.asarray(labels1, dtype=object)
            labels2 = np.asarray(labels2, dtype=object)
            if not np.array_equal(labels1, labels2):
                l1 = labels1.reshape(-1).tolist()
                l2 = labels2.reshape(-1).tolist()
                if len(l1) == len(l2):
                    diffs = [f"[{i}] {a} != {b}" for i,(a,b) in enumerate(zip(l1,l2)) if a != b]
                    mismatches.append(f"{path or '/'}: {len(diffs)} of {len(l1)} labels differ: " + "; ".join(diffs))
                else:
                    only1 = [l for l in l1 if l not in l2]
                    only2 = [l for l in l2 if l not in l1]
                    mismatches.append(f"{path or '/'}: label counts differ ({len(l1)} != {len(l2)}), "
                                      f"only in first: {only1}, only in second: {only2}")
    return mismatches