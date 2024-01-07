from typing import Dict, Tuple, Union
import gymnasium as gym
import numpy as np
import torch as th
import copy
from numpy.typing import DTypeLike

class ObsConverter:

    class SpaceInfo():
        def __init__(self, indexes, is_img, shape, dtype, obs_space):
            self.indexes = indexes
            self.is_img = is_img
            self.shape = shape
            self.dtype = dtype
            self.obs_space = obs_space

        def __repr__(self):
            return f"[{self.indexes}, {self.is_img}, {self.shape}, {self.dtype}]"

    @staticmethod
    def _is_img(box_space):
        obs_shape = box_space.shape
        if len(obs_shape)==1:
            return False
        elif len(obs_shape)==2 or len(obs_shape)==3:
            return True
        else:
            raise NotImplementedError(f"Unexpected obs_shape {obs_shape}")

    @staticmethod
    def _img_shape_chw(box_space):
        obs_shape = box_space.shape
        if len(obs_shape)==2:
            # 1-channel image
            return (1, obs_shape[0], obs_shape[1])
        elif len(obs_shape)==3:
            # multi-channel image
            return (obs_shape[0], obs_shape[1], obs_shape[2])
        else:
            raise NotImplementedError(f"Unexpected image obs_shape {obs_shape}")

    @staticmethod
    def _get_space_info(obs_space):
        """Generates a list of all the sub-observations definitions.
        Each of them with its corresponding key and the space shape and type."""
        if isinstance(obs_space, gym.spaces.Box):
            is_img = ObsConverter._is_img(obs_space)
            if is_img:
                shape = ObsConverter._img_shape_chw(obs_space)
            else:
                shape = (obs_space.shape[0],)
            dtype = obs_space.dtype
            return [ObsConverter.SpaceInfo([], is_img, shape, dtype, obs_space)]
        elif isinstance(obs_space, gym.spaces.Dict):
            ret = []
            for sub_obs_key, sub_obs_space in obs_space.spaces.items():
                sub_info = ObsConverter._get_space_info(sub_obs_space)
                for si in sub_info:
                    ret.append(ObsConverter.SpaceInfo([sub_obs_key]+si.indexes, si.is_img, si.shape, si.dtype, si.obs_space))
            return ret
        else:
            raise NotImplementedError(f"Unexpected obs_space {obs_space}")

    @staticmethod
    def _get_space_structure(obs_space : gym.spaces.Space):
        """Generates a nested-dict structure like the one of the original observations.
            None values are placed in place of the observation data."""
        if isinstance(obs_space, gym.spaces.Box):
            return None
        elif isinstance(obs_space, gym.spaces.Dict):
            ret = {}
            for sub_obs_key, sub_obs_space in obs_space.spaces.items():
                ret[sub_obs_key] = ObsConverter._get_space_structure(sub_obs_space)
            return ret
        else:
            raise NotImplementedError(f"Unexpected obs_space {obs_space}")

    @staticmethod
    def _get_sub_obs(obs, indexes):
        for i in indexes:
            obs = obs[i]
        return obs

    def __init__(self, observation_shape : gym.spaces.Dict, hide_achieved_goal : bool = True):
        self._original_obs_space = observation_shape
        
        if not isinstance(observation_shape, gym.spaces.Dict):
            raise AttributeError(f"observation_shape must be a gym.spaces.Dict, if it is not, just wrap it to be one")

        self._original_obs_space_structure = self._get_space_structure(observation_shape)
        self._hide_achieved_goal = hide_achieved_goal
        obs_elements = self._get_space_info(observation_shape)
        # ggLog.info(f"observation_shape analysis:")
        # for idxs, val in enumerate(self._obs_elements):
        #     ggLog.info(f"{idxs}: {val}")

        # Look for the vector components and define what will be their order once concatenated
        self._vec_part_idxs = []
        self._vec_parts_sizes = []
        self._vectorPartSize = 0
        self._space_infos  ={}
        for space_info in obs_elements:
            if not space_info.is_img:
                if space_info.indexes[0] != "achieved_goal" and space_info.indexes[0] != "desired_goal":
                    self._vec_part_idxs.append(space_info.indexes)
                    self._vectorPartSize += space_info.shape[0]
                    self._vec_parts_sizes.append(space_info.shape[0])
                    self._space_infos[tuple(space_info.indexes)] = space_info
        for space_info in obs_elements: #keep desired goal at the end
            if not space_info.is_img:
                if space_info.indexes[0] == "desired_goal":
                    self._vec_part_idxs.append(space_info.indexes)
                    self._vectorPartSize += space_info.shape[0]
                    self._vec_parts_sizes.append(space_info.shape[0])
                    self._space_infos[tuple(space_info.indexes)] = space_info
        if not self._hide_achieved_goal:  # if it must be visible add also achieved goal
            for space_info in obs_elements:
                if not space_info.is_img:
                    if space_info.indexes[0] == "achieved_goal":
                        self._vec_part_idxs.append(space_info.indexes)
                        self._vectorPartSize += space_info.shape[0]
                        self._vec_parts_sizes.append(space_info.shape[0])
                        self._space_infos[tuple(space_info.indexes)] = space_info

        
        self._img_part_indexes = None
        for space_info in obs_elements:
            if space_info.is_img:
                if self._img_part_indexes is not None:
                    raise NotImplementedError(f"observation has more than one image, currently not supported: {obs_elements}")
                self._img_part_indexes = space_info.indexes
                self._image_channels = space_info.shape[0]
                self._image_height = space_info.shape[1] # type: ignore
                self._image_width = space_info.shape[2] # type: ignore
                self._img_dtype = space_info.dtype
                self._space_infos[tuple(space_info.indexes)] = space_info
        if self._img_part_indexes is None: # If no image os present
            self._image_channels = 0
            self._image_height = 0
            self._image_width = 0
            self._img_dtype = np.uint8

        if self._img_dtype == np.uint8:
            self._img_pixel_range = (0,255)
        elif self._img_dtype == np.float32:
            self._img_pixel_range = (0,1)
        else:
            raise NotImplemented(f"Unsupported image dtype {self._img_dtype}")

    def vectorPartSize(self) -> int:
        return int(self._vectorPartSize)

    def imageSizeCHW(self) -> Tuple[int,int,int]:
        return (self._image_channels, self._image_height, self._image_width)

    def hasImagePart(self):
        return self._img_part_indexes is None

    def hasVectorPart(self):
        return self._vectorPartSize > 0
    
    def _getVectorPart(self, observation_batch : Dict[str,th.Tensor]) -> th.Tensor:

        # ggLog.info(f"getVectorPart("+str([f"{k} : {subobs_batch.size()}" for k, subobs_batch in observation_batch.items()]))
        if self._vectorPartSize == 0:
            img_part = self.getImgPart(observation_batch)
            if len(img_part.size())<4:
                batch_size = 1
                traj_size = None # No trajectory dimension
            elif len(img_part.size())==4:
                batch_size = img_part.size()[0]
                traj_size = None # No trajectory dimension
            elif len(img_part.size())==5:
                batch_size = img_part.size()[0]
                traj_size = img_part.size()[1]
            else:
                raise RuntimeError(f"Unexpected image part dimensionality. img_part.size() = {img_part.size()}")
            if traj_size is None: 
                return th.empty(size=(batch_size,0)).to(img_part.device) #same batch size as the image part
            else:
                return th.empty(size=(batch_size,traj_size,0)).to(img_part.device) #same batch size as the image part
        vec = []
        for idxs in self._vec_part_idxs:
            subvec = self._get_sub_obs(observation_batch, idxs)
            # ggLog.info(f"subvec.size() = {subvec.size()}")
            vec.append(subvec)
        if len(vec[0].size())==2: # batch, no trajectories
            return th.cat(vec,dim=1)
        elif len(vec[0].size())==3: # batch of trajectories
            return th.cat(vec,dim=2)
        else:
            raise RuntimeError(f"Unexpected vec dimensionality: vec[0].size() = {vec[0].size()}")


    def getVectorPart(self, observation_batch : Union[Dict[str,th.Tensor],Dict[str,np.ndarray]]):
        first_obs = next(iter(observation_batch.values()))
        if isinstance(first_obs,np.ndarray):
            observation_batch_th = {k:th.as_tensor(v) for k,v in observation_batch.items()}
            return self._getVectorPart(observation_batch_th).numpy()
        elif isinstance(first_obs,th.Tensor):
            return self._getVectorPart(observation_batch) # type: ignore
        else:
            raise AttributeError(f"unexpected observation type {type(first_obs)}")
        
    def getVectorPartLimits(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._vectorPartSize == 0:
            return np.empty((0,)),np.empty((0,))
        lows = []
        highs = []
        for idxs in self._vec_part_idxs:
            lows.append(self._space_infos[tuple(idxs)].obs_space.low)
            highs.append(self._space_infos[tuple(idxs)].obs_space.high)
        return np.concatenate(lows),np.concatenate(highs)


    def _getImgPart(self, observation_batch : Dict[str,th.Tensor]):
        # ggLog.info(f"getImgPart("+str([f"{k} : {subobs_batch.size()}" for k, subobs_batch in observation_batch.items()]))        
        if self._img_part_indexes is None:
            vecPart = self.getVectorPart(observation_batch)
            if len(vecPart.size())==1:
                batch_size = 1
                traj_size = None # No trajectory dimension
            elif len(vecPart.size())==2:
                batch_size = vecPart.size()[0]
                traj_size = None # No trajectory dimension
            elif len(vecPart.size())==3:
                batch_size = vecPart.size()[0]
                traj_size = vecPart.size()[1]
            else:
                raise RuntimeError(f"Unexpected vector part dimensionality. vec_part.size() = {vecPart.size()}")
            # ggLog.info(f"No image, batch_size = {batch_size}, traj_size = {traj_size}")
            if traj_size is None:
                return th.empty(size=(batch_size,)+self.imageSizeCHW()).to(vecPart.device)
            else:
                return th.empty(size=(batch_size,traj_size)+self.imageSizeCHW()).to(vecPart.device)
        return self._get_sub_obs(observation_batch, self._img_part_indexes)
    

    def getImgPart(self, observation_batch : Union[Dict[str,th.Tensor],Dict[str,np.ndarray]]):
        first_obs = next(iter(observation_batch.values()))
        if isinstance(first_obs,np.ndarray):
            observation_batch_th = {k:th.as_tensor(v) for k,v in observation_batch.items()}
            return self._getImgPart(observation_batch_th).numpy()
        elif isinstance(first_obs,th.Tensor):
            return self._getImgPart(observation_batch) # type: ignore
        else:
            raise AttributeError(f"unexpected observation type {type(first_obs)}")

    def buildDictObs(self, vectorPart_batch : th.Tensor, imgPart_batch : th.Tensor):
        obs = copy.deepcopy(self._original_obs_space_structure)
        pos = 0
        for i in range(len(self._vec_part_idxs)):
            idxs = self._vec_part_idxs[i]
            vec_size = self._vec_parts_sizes[i]
            self._get_sub_obs(obs, idxs[:-1])[idxs[-1]] = vectorPart_batch[:, pos:pos+vec_size]
            pos+=vec_size
        if self._img_part_indexes is not None:
            self._get_sub_obs(obs, self._img_part_indexes[:-1])[self._img_part_indexes[-1]] = imgPart_batch
        return obs

    def getImgDtype(self) -> DTypeLike:
        return self._img_dtype

    def getImgPixelRange(self):
        return self._img_pixel_range
    
    def to_standard_tensors(self, obs_batch, device):        
        for idxs in self._vec_part_idxs:
            self._get_sub_obs(obs_batch, idxs[:-1])[idxs[-1]] = th.as_tensor(self._get_sub_obs(obs_batch, idxs), device = device)
        if self._img_part_indexes is not None:
            self._get_sub_obs(obs_batch, self._img_part_indexes[:-1])[self._img_part_indexes[-1]] = th.as_tensor(self._get_sub_obs(obs_batch, self._img_part_indexes), device = device)
            while len(self._get_sub_obs(obs_batch, self._img_part_indexes).size())<4:
                self._get_sub_obs(obs_batch, self._img_part_indexes[:-1])[self._img_part_indexes[-1]] = self._get_sub_obs(obs_batch, self._img_part_indexes).unsqueeze(dim = 0)
        return obs_batch
