from __future__ import annotations
import gymnasium as gym
from typing import Any, SupportsFloat, Sequence, Union, Tuple, Dict, overload
from matplotlib.pylab import Generator
from numpy.typing import NDArray
import numpy as np
import torch as th
gym_spaces = gym.spaces
from copy import deepcopy
from gymnasium.vector.utils.spaces import batch_space
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
        self._th_device = torch_device
        self._seed = seed
        if generator is None:
            generator = th.Generator(device=self._th_device)
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
        self._high_th = th.as_tensor(self.high, device=self._th_device)
        self._low_th = th.as_tensor(self.low, device=self._th_device)
        if default_value is not None:
            self.zero_action = default_value.expand_as(self._high_th)
        else:
            self.zero_action = (self._high_th+self._low_th)/2
    

    def sample(self):
        # ggLog.info(f"Sampling ThBox, rng state = {hash_tensor(self._rng.get_state()) if self._rng is not None else None}")
        # import traceback
        # traceback.print_stack()
        # only works for uniform
        dtype : th.dtype = getattr(th,self.torch_dtype_str)
        if dtype.is_floating_point:
            r = th.rand(self._high_th.size(),
                        device=self._th_device,
                        generator=self._th_rng,
                        dtype=dtype)
        elif self.dtype not in [th.int8, th.int16, th.int32, th.int64, th.uint8, th.uint16, th.uint32, th.uint64]:
            r = th.rand(self._high_th.size(),
                        device=self._th_device,
                        generator=self._th_rng)
            r = (r*(self._high_th-self._low_th)+self._low_th).to(dtype)
        return r*(self._high_th-self._low_th)+self._low_th
        # return th.as_tensor(super().sample(), device = self._th_device) # does not use the torch rng

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_np_random",None)
        # serialize ndarrays as torch tensors to avoid issues with numpy 2.0/1.x
        state["bounded_above"] = th.as_tensor(self.bounded_above*1, dtype=th.bool)
        state["bounded_below"] = th.as_tensor(self.bounded_below*1, dtype=th.bool)
        state["high"] = th.as_tensor(self.high)
        state["low"] = th.as_tensor(self.low)
        if isinstance(self.labels,np.ndarray):
            state["labels"] = self.labels.tolist()
        state.pop("dtype", None)
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.bounded_above = self.bounded_above.cpu().numpy().astype(np.bool_)
        self.bounded_below = self.bounded_below.cpu().numpy().astype(np.bool_)
        self.high = self.high.cpu().numpy()
        self.low = self.low.cpu().numpy()
        self.dtype = torch_to_numpy_dtype_dict[getattr(th,self.torch_dtype_str)]
        if isinstance(self.labels,list):
            self.labels = np.array(self.labels, dtype=object)

def get_space_labels(space : gym_spaces.Dict | ThBox):
    if isinstance(space, ThBox):
        return space.labels
    elif isinstance(space, gym_spaces.Dict):
        return {k: get_space_labels(space.spaces[k]) for k in space.spaces}
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
            subseeds = th.randint(0, np.iinfo(np.int32).max, (len(self.spaces),), generator=self._th_rng).tolist()
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