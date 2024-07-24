import adarl.utils.dbg.ggLog as ggLog
import numpy as np

from adarl.envs.lr_wrappers.LrWrapper import LrWrapper
from adarl.envs.BaseEnv import BaseEnv
import adarl.utils.spaces as spaces
from typing import Tuple, Optional, Literal
import torch as th
import copy
from adarl.utils.utils import torch_to_numpy_dtype_dict
import torchvision.transforms.functional

class ImgFormatWrapper(LrWrapper):
    
    def __init__(self,  env : BaseEnv,
                        img_dict_key : Optional[str] = None,
                        input_channel_order : str = "chw",
                        output_channel_order : str = "chw",
                        output_dtype : th.dtype = th.float32,
                        input_device : th.device | None = None,
                        output_device : th.device | None = None,
                        input_type : type | None = None,
                        grayscale_output : bool | None = None):
        super().__init__(env)
        self._input_channel_order = input_channel_order 
        self._output_channel_order = output_channel_order  
        self._output_dtype  = output_dtype
        self._img_dict_key = img_dict_key
        self._output_device = output_device
        self._input_device = input_device
        self._input_type = input_type
        assert ''.join(sorted(output_channel_order)) == "chw", "output_channel_order must be a permutation of 'chw'"
        if self._input_device is not None and self._input_device.type == "cuda" and self._input_device.index is None:
            self._input_device = th.device("cuda:0")
    
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
        if grayscale_output is None:
            self._grayscale_output = self._input_img_channels == 1
        else:
            self._grayscale_output = grayscale_output
            

        self._output_img_height = self._input_img_height
        self._output_img_width = self._input_img_width
        self._output_img_channels = 1 if self._grayscale_output else self._input_img_channels
        self._output_shape = [0,0,0]
        self._output_shape[self._output_channel_order.find("c")] = self._output_img_channels
        self._output_shape[self._output_channel_order.find("h")] = self._output_img_height
        self._output_shape[self._output_channel_order.find("w")] = self._output_img_width
        self._output_shape = tuple(self._output_shape)
        
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
        # import traceback
        # traceback.print_stack()
        if self._img_dict_key is None:
            img = obs
        else:
            img = obs[self._img_dict_key]
        # ggLog.info(f"ImgFormatWrapper input = {img.size()}")
        if self._input_type is not None and not isinstance(img, self._input_type):
            raise RuntimeError(f"input image is of type '{type(img)}' instead of the required type '{self._input_type}'")
        img = th.as_tensor(img) # if it's a numpy array make it a cpu tensor
        if self._input_device is not None:
            if self._input_device.type != img.device.type or (img.device.type=="cuda" and img.device.index!=self._input_device.index):
                raise RuntimeError(f"input image is on device '{img.device}' instead of the required device '{self._input_device}'")
        img = th.as_tensor(img, device=self._output_device)

        if self._input_channel_order != self._output_channel_order:
            if len(self._input_channel_order) == 2:
                img.unsqueeze(0)
                input_channel_order = "c"+self._input_channel_order
            else:
                input_channel_order = self._input_channel_order            
            img = th.permute(img, [input_channel_order.find(self._output_channel_order[i]) for i in range(3)])
        
        input_dtype_np : np.dtype = torch_to_numpy_dtype_dict[img.dtype] #type: ignore
        output_dtype_np : np.dtype = torch_to_numpy_dtype_dict[img.dtype] #type: ignore
        if self._output_dtype != img.dtype:
            if self._output_dtype==th.float32 and np.issubdtype(input_dtype_np, np.integer):
                img = img.to(dtype=th.float32)/np.iinfo(input_dtype_np).max 
            elif self._output_dtype==th.uint8 and np.issubdtype(input_dtype_np, np.floating):
                img = (img*np.iinfo(output_dtype_np).max).to(dtype=th.uint8)
            else:
                raise NotImplementedError(f"Unsuppored output_dtype {self._output_dtype} and input_dtype {input_dtype_np}")

        if self._grayscale_output:
            if self._input_img_channels == 1:
                pass
            elif self._input_img_channels == 3:
                if self._output_channel_order != "chw":
                    img = th.permute(img, [self._output_channel_order.find(d) for d in "chw"])
                img = torchvision.transforms.functional.rgb_to_grayscale(img)
                if self._output_channel_order != "chw":
                    img = th.permute(img, ["chw".find(d) for d in self._output_channel_order])
            else:
                raise RuntimeError(f"Cannot convert to grayscale image with {self._input_img_channels} channels")
        # ggLog.info(f"img max = {np.amax(img)}")
        if self._img_dict_key is None:
            obs = img
        else:
            obs = copy.deepcopy(obs)
            obs[self._img_dict_key] = img
        # ggLog.info(f"ImgFormatWrapper output = {img.size()}")
        return obs


    def getObservation(self, state):
        # ggLog.info("Converting Frame")
        # ggLog.info(f"ImgFormatWrapper getting obs from state {state}")
        obs = self.env.getObservation(state)
        # ggLog.info(f"ImgFormatWrapper got obs {obs}")
        obs = self._convert_frame(obs)
        return obs


