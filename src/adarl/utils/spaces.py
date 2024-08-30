from __future__ import annotations
import gymnasium as gym
from typing import Any, SupportsFloat, Sequence
from numpy.typing import NDArray
import numpy as np
import torch as th
from adarl.utils.utils import torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict
gym_spaces = gym.spaces

class ThBox(gym.spaces.Box):
    def __init__(   self,
                    low: SupportsFloat | NDArray[Any] | th.Tensor,
                    high: SupportsFloat | NDArray[Any] | th.Tensor,
                    shape: Sequence[int] | None = None,
                    dtype: type[np.floating[Any]] | type[np.integer[Any]] | th.dtype = np.float32,
                    seed: int | np.random.Generator | None = None,
                    torch_device : th.device = th.device("cpu")):
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
        super().__init__(low=low,high=high,shape=shape,dtype=numpy_dtype,seed=seed)
    def sample(self):
        return th.as_tensor(super().sample(), device = self._th_device)