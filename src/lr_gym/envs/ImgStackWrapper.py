import gym
import cv2
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
from lr_gym.envs.LrWrapper import LrWrapper

class ImgStackWrapper(LrWrapper):

    def __init__(self,  env : gym.Env,
                        frame_stacking_size : int,
                        img_dict_key = None,
                        action_repeat : int = -1):
        super().__init__(env)
        self._img_dict_key = img_dict_key
        self._frame_stacking_size = frame_stacking_size
        self._actionToDo = None

        if self._img_dict_key is None:
            sub_img_space = env.observation_space
        else:
            sub_img_space = env.observation_space[img_dict_key]

        self._nonstacked_channels = sub_img_space.shape[0]
            
        if len(sub_img_space.shape) == 3:
            self._input_img_height = sub_img_space.shape[1]
            self._input_img_width = sub_img_space.shape[2]
            self._input_img_channels = sub_img_space.shape[0]
        elif len(sub_img_space.shape) == 2:
            self._input_img_height = sub_img_space.shape[0]
            self._input_img_width = sub_img_space.shape[1]
            self._input_img_channels = 1
        else:
            raise RuntimeError("Unexpected image shape ",sub_img_space)
        # ggLog.info(f"img space = {sub_img_space}")

        self._output_img_channels = self._input_img_channels
        self._output_img_height = self._input_img_height
        self._output_img_width = self._input_img_width
        
        if sub_img_space.dtype == np.float32:
            low = 0
            high = 1
        elif sub_img_space.dtype == np.uint8:
            low = 0
            high = 255
        else:
            raise RuntimeError(f"Unsupported env observation space dtype {sub_img_space.dtype}")
        img_obs_space =  gym.spaces.Box(low=low, high=high,
                                        shape=(self._output_img_channels*self._frame_stacking_size , self._output_img_height, self._output_img_width),
                                        dtype=sub_img_space.dtype)
        if self._img_dict_key is None:
            self.observation_space = img_obs_space
        else:
            obs_dict = {}
            for k in env.observation_space.spaces.keys():
                obs_dict[k] = env.observation_space[k]
            obs_dict[self._img_dict_key] = img_obs_space
            self.observation_space = gym.spaces.Dict(obs_dict)

        stack_shape = [self._output_img_channels*self._frame_stacking_size, self._output_img_height, self._output_img_height]
        self._stackedImg = np.empty(shape=stack_shape, dtype=sub_img_space.dtype)
        self._framesBuffer = [None]*self._frame_stacking_size
        if action_repeat == -1:
            self._action_repeat = self._frame_stacking_size
        else:
            self._action_repeat = action_repeat
        # print("observation_space =", self.observation_space)
        # print("observation_space.dtype =", self.observation_space.dtype)

    def _preproc_frame(self, img):
        # ggLog.info(f"preproc input shape = {img.shape}")
        img = np.squeeze(img)
        if len(img.shape) == 3 and img.shape[2] == 3: # RGB with HWC shape
            img = np.transpose(img, (2,0,1)) # convert channel ordering from HWC to CHW
            ggLog.warn("ImgStackGymWrapper: do not rely on ImgStackGymWrapper to dimension ordering, use ImgFormatWrapper")
        elif len(img.shape) != 2 and len(img.shape) != 3:
            raise RuntimeError(f"Unexpected image shape {img.shape}")
        
        # print("ImgStack: ",img.shape)
        return img

    def _fill_observation(self, obs):
        for i in range(self._frame_stacking_size):
            self._stackedImg[i*self._output_img_channels:(i+1)*self._output_img_channels] = self._framesBuffer[i]
        if self._img_dict_key is None:
            obs = self._stackedImg
        else:
            obs[self._img_dict_key] = self._stackedImg
        return obs

    def _pushFrame(self, frame):
        for i in range(len(self._framesBuffer)-1):
            self._framesBuffer[i]=self._framesBuffer[i+1]
        self._framesBuffer[-1] = frame

    def submitAction(self, action) -> None:
        self._actionToDo = action
        self.env.submitAction(self._actionToDo)

    def performStep(self):
        self._rewardSum = 0
        for i in range(self._action_repeat):
            previousState = self.env.getState()
            self.env.performStep()
            state = self.env.getState()
            obs = self.env.getObservation(state)
            if self._img_dict_key is None:
                img = obs
            else:
                img = obs[self._img_dict_key]
            img = self._preproc_frame(img)
            self._pushFrame(img)
            self._rewardSum += self.env.computeReward(previousState, state, self._actionToDo)
            self.env.submitAction(self._actionToDo)
        self._previousState = self._lastState
        self._lastState = state

    def getObservation(self, state):
        obs = self.env.getObservation(state)
        
        for i in range(self._frame_stacking_size):
            self._stackedImg[i*self._output_img_channels:(i+1)*self._output_img_channels] = self._framesBuffer[i]
        if self._img_dict_key is None:
            obs = self._stackedImg
        else:
            obs[self._img_dict_key] = self._stackedImg

        return obs

    def performReset(self):

        self.env.performReset()

        obs = self.env.getObservation(self.env.getState())
        if self._img_dict_key is None:
            img = obs
        else:
            img = obs[self._img_dict_key]
        img = self._preproc_frame(img)
        for _ in range(self._frame_stacking_size):
            self._pushFrame(img)
        self._previousState = self.env.getState()
        self._lastState = self.env.getState()

    def computeReward(self, previousState, state, action) -> float:
        if not (state is self._lastState and action is self._actionToDo and previousState is self._previousState):
            raise RuntimeError("GymToLr.computeReward is only valid if used for the last executed step. And it looks like you tried using it for something else.")
        return self._rewardSum
