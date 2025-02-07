from __future__ import annotations
import gymnasium as gym
from typing import Any, SupportsFloat, Sequence
from numpy.typing import NDArray
import numpy as np
import torch as th
gym_spaces = gym.spaces
from copy import deepcopy
from gymnasium.vector.utils.spaces import batch_space
from adarl.utils.utils import torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict
import adarl.utils.dbg.ggLog as ggLog
class ThBox(gym.spaces.Box):
    def __init__(   self,
                    low: SupportsFloat | NDArray[Any] | th.Tensor,
                    high: SupportsFloat | NDArray[Any] | th.Tensor,
                    shape: Sequence[int] | None = None,
                    dtype: type[np.floating[Any]] | type[np.integer[Any]] | th.dtype | str = np.float32,
                    seed: int | np.random.Generator | None = None,
                    torch_device : th.device = th.device("cpu"),
                    labels : th.Tensor | None = None,
                    generator : th.Generator | None = None):
        self._th_device = torch_device
        self._rng = generator
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
        self.labels = labels
        super().__init__(low=low,high=high,shape=shape,dtype=numpy_dtype,seed=seed)
        self._high_th = th.as_tensor(self.high, device=self._th_device)
        self._low_th = th.as_tensor(self.low, device=self._th_device)

    def sample(self):
        # ggLog.info(f"Sampling ThBox, rng state = {hash_tensor(self._rng.get_state()) if self._rng is not None else None}")
        # import traceback
        # traceback.print_stack()
        # only works for uniform
        r = th.rand(self._high_th.size(),
                    device=self._th_device,
                    generator=self._rng,
                    dtype=getattr(th,self.torch_dtype_str))
        return r*(self._high_th-self._low_th)+self._low_th
        # return th.as_tensor(super().sample(), device = self._th_device) # does not use the torch rng
    
    

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