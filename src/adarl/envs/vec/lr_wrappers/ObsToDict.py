#!/usr/bin/env python3
from __future__ import annotations
from adarl.envs.vec.lr_wrappers.lr_vec_wrapper import LrVecWrapper
from adarl.envs.vec.BaseVecEnv import BaseVecEnv, Observation
import adarl.utils.spaces as spaces
from typing import Mapping, TypeVar
import torch as th


class ObsToDict(LrVecWrapper[Observation]):

    def __init__(self,
                 env : BaseVecEnv[Observation],
                 key : str = "obs"):
        self._key = key
        super().__init__(env=env)
        self.single_observation_space = spaces.gym_spaces.Dict({key: self.single_observation_space})
        self.vec_observation_space = spaces.gym_spaces.Dict({key: self.vec_observation_space})

    def get_observations(self, states: Mapping[str | tuple[str, ...], th.Tensor]) -> dict[str,Observation]:
        return {self._key: self.env.get_observations(states)}