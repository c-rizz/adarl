import adarl.utils.dbg.ggLog as ggLog
import numpy as np
from adarl.envs.lr_wrappers.LrWrapper import LrWrapper
from adarl.envs.BaseEnv import BaseEnv
import adarl.utils.spaces as spaces
import torch as th
import copy
from typing import Optional, Dict
from typing_extensions import override
from collections import deque

class ImgStackWrapper(LrWrapper):

    def __init__(self,  env : BaseEnv,
                        frame_stacking_size : int,
                        img_dict_key = None,
                        action_repeat : int = -1):
        super().__init__(env)
        self._img_dict_key = img_dict_key
        self._frame_stacking_size = frame_stacking_size
        self._actionToDo = th.tensor([])

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
        img_obs_space =  spaces.gym_spaces.Box(low=low, high=high,
                                        shape=(self._output_img_channels*self._frame_stacking_size , self._output_img_height, self._output_img_width),
                                        dtype=sub_img_space.dtype)
        if self._img_dict_key is None:
            self.observation_space = img_obs_space
        else:
            obs_dict = {}
            for k in env.observation_space.spaces.keys():
                obs_dict[k] = env.observation_space[k]
            obs_dict[self._img_dict_key] = img_obs_space
            self.observation_space = spaces.gym_spaces.Dict(obs_dict)

        stack_shape = [self._output_img_channels*self._frame_stacking_size, self._output_img_height, self._output_img_height]
        self._stackedImg = np.empty(shape=stack_shape, dtype=sub_img_space.dtype)
        self._framesBuffer = [None]*self._frame_stacking_size
        if action_repeat == -1:
            self._action_repeat = self._frame_stacking_size
        else:
            self._action_repeat = action_repeat
        self._last_states = deque(maxlen=self._action_repeat)

        self.state_space = spaces.gym_spaces.Tuple([self.env.state_space]*self._action_repeat)
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

    def submitAction(self, action : th.Tensor) -> None:
        self._actionToDo = action
        self.env.submitAction(self._actionToDo)

    def performStep(self):
        for i in range(self._action_repeat):
            # previousState = self.env.getState()
            self.env.performStep()
            state = self.env.getState()
            obs = self.env.getObservation(state)
            self._last_states.append(state)
            if self._img_dict_key is None:
                img = obs
            else:
                img = obs[self._img_dict_key]
            self._pushFrame(self._preproc_frame(img))
            self.env.submitAction(self._actionToDo)
        # ggLog.info(f"performed step: self._last_states = {self._last_states}")

    def getObservation(self, state):
        observations = [self.env.getObservation(substate) for substate in state[-3:]]
        if isinstance(self.env.observation_space, spaces.gym_spaces.Box):
            obs = th.cat(observations) #type: ignore
        elif isinstance(self.env.observation_space, spaces.gym_spaces.Dict):
            obs = copy.deepcopy(observations[0])
            # ggLog.info(f"observations[0][self._img_dict_key].size() = {observations[0][self._img_dict_key].size()}")
            obs[self._img_dict_key] = th.cat([obs[self._img_dict_key] for obs in observations])
            # ggLog.info(f"obs[self._img_dict_key].size() = {obs[self._img_dict_key].size()}")
        else:
            NotImplementedError(f"Unsupported observation space {self.env.observation_space}")

        return obs

    @override
    def performReset(self, options = {}):
        self.env.performReset(options)
        state = self.env.getState()
        self._last_states.extend([copy.deepcopy(state) for _ in range(self._action_repeat)])


    def getState(self):
        return list(self._last_states)

    def computeReward(self, previousState, state, action, env_conf = None, sub_rewards : Optional[Dict[str,th.Tensor]] = None) -> float:
        
        tot_reward = 0.0
        tot_sub_rewards = {}
        # n= '\n'
        # ggLog.info(f"state = {n.join([str(s['internal_info']) for s in state])}")
        # ggLog.info(f"previousState = {n.join([str(s['internal_info']) for s in previousState])}")
        states = previousState[-1:]+state
        for i in range(self._action_repeat):
            sub_sub_rewards = {}
            sub_reward = self.env.computeReward(previousState=states[i],
                                                state=states[i+1],
                                                action=action,
                                                env_conf=env_conf,
                                                sub_rewards=sub_sub_rewards)
            tot_reward += sub_reward
            for k in sub_sub_rewards:
                tot_sub_rewards[k] = sub_sub_rewards[k] + self._sub_rewards_sum.get(k,0)
        if sub_rewards is not None:
            sub_rewards.update(tot_sub_rewards)
        return tot_reward

    def reachedTerminalState(self, previousState, state) -> th.Tensor:
        r = None
        for i in range(len(state)):
            st = self.env.reachedTerminalState(previousState=previousState[i], state=state[i])
            if r is None:
                r = st
            else:
                r |= st
        return r #type: ignore