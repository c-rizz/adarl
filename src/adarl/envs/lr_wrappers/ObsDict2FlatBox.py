
import numpy as np
import adarl.utils
from collections import OrderedDict

from adarl.envs.lr_wrappers.LrWrapper import LrWrapper
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.spaces as spaces
import torch as th

def dict2box(dict_space):
    dtype_type = None
    low = []
    high = []
    size = 0
    for subspace_name, subspace in dict_space.spaces.items():
        if type(subspace) == spaces.gym_spaces.Dict:
            subspace = dict2box(subspace)
        if type(subspace) == spaces.gym_spaces.Box:
            low.extend(subspace.low.flatten())
            high.extend(subspace.high.flatten())
            size += np.prod(subspace.shape)
            if dtype_type == None:
                dtype_type = subspace.dtype
            if dtype_type != subspace.dtype:
                raise AttributeError(f"dtypes must be all the same but there's a {dtype_type} and a {subspace.dtype}")
            
        else:
            raise AttributeError(f"Unsupported space {subspace_name}:{subspace}")
    return spaces.gym_spaces.Box(low = np.array(low),
                          high = np.array(high),
                          dtype = dtype_type,
                          shape = (size,))



class ObsDict2FlatBox(LrWrapper):

    def __init__(self,
                 env : adarl.envs.BaseEnv,
                 key : str = "obs",
                 torch_observations = False):
        self._torch_observations = torch_observations
        super().__init__(env=env)

        if type(env.observation_space)!=spaces.gym_spaces.Dict:
            raise AttributeError("Input env observation_space is not a dict")
        
        self.observation_space = dict2box(env.observation_space)
        self.action_space = env.action_space
        self.metadata = env.metadata
        ggLog.info(f"observation_space = {self.observation_space}")


    def _dict2box_obs(self, obs, obs_space):
        if self._torch_observations: 
            ret = th.empty(size=obs_space.shape, dtype = obs_space.dtype)
        else:
            ret = np.empty(shape=obs_space.shape, dtype = obs_space.dtype)
        pos = 0
        for key, value in obs.items():
            if isinstance(value, dict) or isinstance(value, OrderedDict):
                value = self._dict2box_obs(value, obs_space)
            if type(value) == np.ndarray:
                subobs_size = np.prod(value.shape)
                ret[pos:pos+subobs_size] = value.flatten()
                pos+=subobs_size
            if type(value) == th.Tensor:
                subobs_size = np.prod(value.shape)
                ret[pos:pos+subobs_size] = value.flatten()
                pos+=subobs_size
            else:
                raise AttributeError(f"Unexpected type {type(value)}")
        return ret

    def getObservation(self, state):
        obs = self.env.getObservation(state)
        obs =  self._dict2box_obs(obs, self.observation_space)
        return obs
