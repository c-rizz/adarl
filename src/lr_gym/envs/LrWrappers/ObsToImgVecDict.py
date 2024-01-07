#!/usr/bin/env python3


from lr_gym.envs.LrWrapper import LrWrapper
from lr_gym.envs.BaseEnv import BaseEnv
import lr_gym.utils.spaces as spaces
from lr_gym.utils.ObsConverter import ObsConverter
import torch as th
import numpy as np
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
                                                        shape=(self._obs_converter.vectorPartSize(),),
                                                        dtype=np.float32) # Can I use a torch dtype?
        if self._obs_converter.hasImagePart():
            img_shape = self._obs_converter.imageSizeCHW()
            l,h = self._obs_converter.getImgPixelRange()
            obss[self._vec_key] = spaces.gym_spaces.Box(low=l,high=h,
                                                        shape=img_shape,
                                                        dtype=self._obs_converter.getImgDtype()) # Can I use a torch dtype?
        self.observation_space = spaces.gym_spaces.Dict(obss)
        self.action_space = env.action_space
        self.metadata = env.metadata

    def getObservation(self, state):
        obs = self.env.getObservation(state)
        obs_batch = {k:v.unsqueeze(0) for k,v in obs.items()}
        return {self._img_key : self._obs_converter.getImgPart(observation_batch=obs_batch).squeeze(0),
                self._vec_key : self._obs_converter.getVectorPart(observation_batch=obs_batch).squeeze(0)}