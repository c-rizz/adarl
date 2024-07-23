import gymnasium as gym
from typing import Any, SupportsFloat, Sequence
from numpy.typing import NDArray
import numpy as np
import torch as th
gym_spaces = gym.spaces

class ThBox(gym.spaces.Box):
    def __init__(   self,
                    low: SupportsFloat | NDArray[Any],
                    high: SupportsFloat | NDArray[Any],
                    shape: Sequence[int] | None = None,
                    dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
                    seed: int | np.random.Generator | None = None,
                    torch_device : th.device = th.device("cpu")):
        self._th_device = torch_device
        super().__init__(low=low,high=high,shape=shape,dtype=dtype,seed=seed)
    def sample(self):
        return th.as_tensor(super().sample(), device = self._th_device)