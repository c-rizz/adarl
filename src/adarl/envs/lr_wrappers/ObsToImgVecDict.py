#!/usr/bin/env python3


from adarl.envs.lr_wrappers.LrWrapper import LrWrapper
from adarl.envs.BaseEnv import BaseEnv
import adarl.utils.spaces as spaces
from adarl.utils.ObsConverter import ObsConverter
import torch as th
import numpy as np
import adarl.utils.dbg.ggLog as ggLog


class ObsToImgVecDict(LrWrapper):

    def __init__(self,
                 env : BaseEnv):
        if not isinstance(env.observation_space, spaces.gym_spaces.Dict):
            raise AttributeError(f"observation_shape must be a gym.spaces.Dict, if it is not, just wrap it to be one")
        obs_space : spaces.gym_spaces.Dict = env.observation_space # type: ignore
        super().__init__(env=env)
        self._obs_converter = ObsConverter(obs_space, hide_achieved_goal=False)
        self._vec_key = "vec"
        self._img_key = "img"
        obss = {}
        if self._obs_converter.hasVectorPart():
            l,h = self._obs_converter.getVectorPartLimits()
            obss[self._vec_key] = spaces.gym_spaces.Box(low=l,high=h,
                                                        shape=(self._obs_converter.vector_part_size(),),
                                                        dtype=np.float32) # Can I use a torch dtype?
        if self._obs_converter.has_image_part():
            img_shape = self._obs_converter.imageSizeCHW()
            l,h = self._obs_converter.getImgPixelRange()
            obss[self._img_key] = spaces.gym_spaces.Box(low=l,high=h,
                                                        shape=img_shape,
                                                        dtype=self._obs_converter.getImgDtype()) # Can I use a torch dtype?
        self.observation_space = spaces.gym_spaces.Dict(obss)
        self.action_space = env.action_space
        self.metadata = env.metadata
        ggLog.info(f"ObsToImgVecDict.observation_space: {self.observation_space}")

    def getObservation(self, state):
        obs = self.env.getObservation(state)
        obs_batch = {k:v.unsqueeze(0) for k,v in obs.items()}
        return {self._img_key : self._obs_converter.getImgPart(observation_batch=obs_batch).squeeze(0),
                self._vec_key : self._obs_converter.getVectorPart(observation_batch=obs_batch).squeeze(0)}