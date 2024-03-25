import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np

from lr_gym.envs.lr_wrappers.LrWrapper import LrWrapper
from lr_gym.envs.BaseEnv import BaseEnv
import lr_gym.utils.spaces as spaces
from typing import Tuple, Optional
import torch as th
from lr_gym.utils.utils import torch_to_numpy_dtype_dict

class ImgFormatWrapper(LrWrapper):
    
    def __init__(self,  env : BaseEnv,
                        img_dict_key : Optional[str] = None,
                        input_channel_order : str = "chw",
                        output_channel_order : str = "chw",
                        output_dtype : th.dtype = th.float32):
        super().__init__(env)
        self._input_channel_order = input_channel_order 
        self._output_channel_order = output_channel_order  
        self._output_dtype  = output_dtype
        self._img_dict_key = img_dict_key
        assert ''.join(sorted(output_channel_order)) == "chw", "output_channel_order must be a permutation of 'chw'"

    
        if isinstance(env.observation_space, spaces.gym_spaces.Dict):
            if self._img_dict_key is None:
                raise RuntimeError(f"Observatiopn space is a dict but no img_dict_key was provided")
            sub_img_space = env.observation_space[self._img_dict_key]
        else:
            sub_img_space = env.observation_space

        self._input_dtype = sub_img_space.dtype

        input_img_shape : Tuple[int,...] = sub_img_space.shape #type:ignore
            
        if len(input_img_shape) == 3:
            assert ''.join(sorted(input_channel_order)) == "chw", "input_channel_order must be a permutation of 'chw' if image has 3 dimensions"
            self._input_img_height = input_img_shape[input_channel_order.find("h")]
            self._input_img_width = input_img_shape[input_channel_order.find("w")]
            self._input_img_channels = input_img_shape[input_channel_order.find("c")]
        elif len(input_img_shape) == 2:
            assert ''.join(sorted(input_channel_order)) == "hw", "input_channel_order must be a permutation of 'hw' if image has 2 dimensions"
            self._input_img_height = input_img_shape[input_channel_order.find("h")]
            self._input_img_width = input_img_shape[input_channel_order.find("w")]
            self._input_img_channels = 1
        else:
            raise RuntimeError("Unexpected image shape ",sub_img_space)
        # ggLog.info(f"img space = {sub_img_space}")



        self._output_shape = [input_img_shape[input_channel_order.find(d)] for d in output_channel_order]
        self._output_shape = tuple(self._output_shape)
        self._output_img_height = self._input_img_height
        self._output_img_width = self._input_img_width
        self._output_img_channels = self._input_img_channels
        
        if self._output_dtype == th.float32:
            low = 0
            high = 1
        elif self._output_dtype == th.uint8:
            low = 0
            high = 255
        else:
            raise RuntimeError(f"Unsupported env observation space dtype {sub_img_space.dtype}")
        output_dtype_np : np.dtype = torch_to_numpy_dtype_dict[output_dtype] #type: ignore
        output_img_obs_space =  spaces.gym_spaces.Box(low=low, high=high,
                                        shape=self._output_shape,
                                        dtype=output_dtype_np) #type: ignore
        
        if isinstance(env.observation_space, spaces.gym_spaces.Dict):
            obs_dict = {}
            for k in env.observation_space.spaces.keys():
                obs_dict[k] = env.observation_space[k]
            obs_dict[self._img_dict_key] = output_img_obs_space
            self.observation_space = spaces.gym_spaces.Dict(obs_dict)

        else:
            self.observation_space = output_img_obs_space


    def _convert_frame(self, obs) -> th.Tensor:
        ggLog.info(f"got obs: {obs}")
        # import traceback
        # traceback.print_stack()
        if self._img_dict_key is None:
            img = obs
        else:
            img = obs[self._img_dict_key]

        if self._input_channel_order != self._output_channel_order:
            if len(self._input_channel_order) == 2:
                img.unsqueeze(0)
                input_channel_order = "c"+self._input_channel_order
            else:
                input_channel_order = self._input_channel_order
            permutation = [-1,-1,-1]
            for i in range(3):
                permutation[i] = input_channel_order.find(self._output_channel_order[i])
            img = th.permute(img, permutation)
        
        input_dtype_np : np.dtype = torch_to_numpy_dtype_dict[img.dtype] #type: ignore
        output_dtype_np : np.dtype = torch_to_numpy_dtype_dict[img.dtype] #type: ignore
        if self._output_dtype != img.dtype:
            if self._output_dtype==th.float32 and np.issubdtype(input_dtype_np, np.integer):
                img = img.to(dtype=th.float32)/np.iinfo(input_dtype_np).max 
            elif self._output_dtype==th.uint8 and np.issubdtype(input_dtype_np, np.floating):
                img = (img*np.iinfo(output_dtype_np).max).to(dtype=th.uint8)
            else:
                raise NotImplementedError(f"Unsuppored output_dtype {self._output_dtype} and input_dtype {input_dtype_np}")

        # ggLog.info(f"img max = {np.amax(img)}")
        if self._img_dict_key is None:
            obs = img
        else:
            obs[self._img_dict_key] = img
        return obs


    def getObservation(self, state):
        # ggLog.info("Converting Frame")
        ggLog.info(f"ImgFormatWrapper getting obs from state {state}")
        obs = self.env.getObservation(state)
        ggLog.info(f"ImgFormatWrapper got obs {obs}")
        obs = self._convert_frame(obs)
        return obs


