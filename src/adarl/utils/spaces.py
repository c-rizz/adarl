from __future__ import annotations
import gymnasium as gym
from typing import Any, SupportsFloat, Sequence
from numpy.typing import NDArray
import numpy as np
import torch as th
from adarl.utils.utils import torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict
gym_spaces = gym.spaces
from copy import deepcopy
from gymnasium.vector.utils.spaces import batch_space
import adarl.utils.dbg.ggLog as ggLog
class ThBox(gym.spaces.Box):
    def __init__(   self,
                    low: SupportsFloat | NDArray[Any] | th.Tensor,
                    high: SupportsFloat | NDArray[Any] | th.Tensor,
                    shape: Sequence[int] | None = None,
                    dtype: type[np.floating[Any]] | type[np.integer[Any]] | th.dtype = np.float32,
                    seed: int | np.random.Generator | None = None,
                    torch_device : th.device = th.device("cpu"),
                    labels : th.Tensor | None = None):
        self._th_device = torch_device
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
        # self.torch_dtype = torch_dtype # yaml cannot save this for some reason, see https://github.com/pytorch/pytorch/issues/78720
        self.labels = labels
        super().__init__(low=low,high=high,shape=shape,dtype=numpy_dtype,seed=seed)

    def sample(self):
        return th.as_tensor(super().sample(), device = self._th_device)
    

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
    return ThBox(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))