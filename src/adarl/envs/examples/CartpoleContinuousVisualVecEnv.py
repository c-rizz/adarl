#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""



import adarl.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any
import adarl.utils.dbg.ggLog as ggLog
import random

from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
import adarl
from adarl.utils.utils import Pose, build_pose, JointState, to_string_tensor
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
import torch as th
from adarl.utils.spaces import ThBox, gym_spaces
from adarl.utils.tensor_trees import space_from_tree
from typing_extensions import override
from typing import Generic
from pathlib import Path
import adarl.utils.utils
from torchvision.transforms.functional import rgb_to_grayscale, resize
import time
from adarl.envs.examples.CartpoleContinuousVecEnv import CartpoleContinuousVecEnv

class CartpoleContinuousVisualVecEnv(ControlledVecEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""



    def __init__(   self,
                    adapter : BaseVecJointEffortAdapter,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    step_duration_sec : float = 0.05,
                    startSimulation : bool = True,
                    wall_sim_speed = False,
                    seed = 1,
                    th_device : th.device = th.device("cpu"),
                    task : str = "balance",
                    img_obs : bool = False,
                    img_obs_resolution : int = 64,
                    img_obs_frame_stacking_size : int = 3,
                    sparse_reward = True):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        environmentController : BaseAdapter
            Specifies which simulator controller to use. By default it connects to Gazebo


        """

        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._renderingEnabled = render
        self._ui_camera_name = "simple_camera"
        self._task = task
        self._img_obs = img_obs
        self._img_obs_resolution = img_obs_resolution
        self._img_obs_frame_stacking_size = img_obs_frame_stacking_size
        self._sparse_reward = sparse_reward
        self._upright_hinge_threshold = 3.14159/180*2
        
        self._CART_POS = 0
        self._CART_VEL = 1
        self._POLE_SIN = 2
        self._POLE_COS = 3
        self._POLE_VEL = 4
        self._TIMESTEP = 5

        state_vec_max = th.as_tensor([  2.5 * 2, # cart position
                                        np.finfo(np.float32).max, # cart velocity
                                        1.0, # pole angle sin
                                        1.0, # pole angle cos
                                        np.finfo(np.float32).max, # pole joint velocity
                                        float("+inf")], # timestep
                                    device=th_device)
        state_vec_labels = ["cart_pos","cart_vel","pole_pos","pole_vel","timestep"]
        vec_state_space = ThBox(-state_vec_max,state_vec_max,
                                labels=to_string_tensor(state_vec_labels),
                                torch_device=th_device)
        if img_obs:
            img_state_space = ThBox(low=0, high=255,
                                    shape=(self._img_obs_frame_stacking_size, self._img_obs_resolution, self._img_obs_resolution),
                                    dtype=th.uint8,
                                    torch_device=th_device)
            single_observation_space = img_state_space
        else:
            single_observation_space = ThBox(-state_vec_max[:5],state_vec_max[:5],
                                             labels=to_string_tensor(state_vec_labels[:-1]),
                                             torch_device=th_device)
        
        states_dict = {"vec" : vec_state_space}
        if img_obs:
            states_dict["img"] = single_observation_space
        state_space = gym_spaces.Dict(states_dict)

        act_max = np.array([1.0])
        super(CartpoleContinuousVecEnv).__init__(th_device=th_device,
                         seed=seed,
                         obs_dtype=th.float32,
                         single_action_space = ThBox(-act_max,act_max, torch_device=th_device),
                         single_observation_space = single_observation_space,
                         single_state_space=state_space,
                         single_reward_space=ThBox(low=float("-inf"),high=float("+inf"), shape=tuple(), torch_device=th_device),
                         info_space=None,
                         step_duration_sec=step_duration_sec,
                         adapter=adapter)
        example_labels : dict[str,th.Tensor] = {}
        example_state = {k:th.as_tensor((s.low+s.high)/2).to(device=th_device).unsqueeze(0) for k,s in states_dict.items()}
        example_infos = self.get_infos(example_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__

        self._build()
        self._adapter.startup()
        self.initialize_episodes()
        self._success = False


    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        super()._initialize_episodes(vec_mask=vec_mask, options=options)
        if self._img_obs:
            sub_step_imgs_vec_hw : list[th.Tensor] = [None,None,None]
            for i in range(self._img_obs_frame_stacking_size):
                imgs_vec_chw, times = self._adapter.getRenderings([self._ui_camera_name])
                # ggLog.info(f"imgs_vec_chw[0].shape = {imgs_vec_chw[0].shape}")
                imgs_vec_hw = self.reshape_imgs(imgs_vec_chw=imgs_vec_chw[0].permute(0,3,1,2))
                sub_step_imgs_vec_hw[i] = imgs_vec_hw
            self._stacked_img = th.stack(sub_step_imgs_vec_hw,dim=1)

    @override
    def get_observations(self, state) -> dict[Any, th.Tensor]:
        if not self._img_obs:
            return state["vec"][:,:-1]
        else:
            return state["img"]

    @override
    def get_states(self) -> th.Tensor:
        state = super().get_states()
        if self._img_obs:
            state["img"] = self._stacked_img
        # ggLog.info(f"state = {state}")
        return state

    def _get_spawn_defs(self):
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            cam_file = "models/simple_camera.mjcf.xacro"
        else:            
            cam_file = "models/simple_camera.sdf.xacro"
        camera_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                    name="simple_camera",
                                    pose=None,
                                    format="sdf.xacro",
                                    kwargs={"camera_width":426,
                                            "camera_height":240,
                                            "frame_rate":1/self._intendedStepLength_sec})
        cartpole_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
                                                        name="cartpole_v0",
                                                        pose=None,
                                                        format="urdf.xacro",
                                                        kwargs={"use_collisions" : "false"})
        return [cartpole_def, camera_def]    

    def reshape_imgs(self, imgs_vec_chw : th.Tensor) -> th.Tensor:
        top    = int( 74/360*imgs_vec_chw.shape[2])
        bottom = int(320/360*imgs_vec_chw.shape[2])
        left   = int(0*imgs_vec_chw.shape[3])
        right  = int(1*imgs_vec_chw.shape[3])
        imgs_vec_chw  = imgs_vec_chw[:,:,top:bottom,left:right]
        imgs_grey_vec_hw = rgb_to_grayscale(imgs_vec_chw)
        imgs_grey_vec_hw = resize(imgs_grey_vec_hw, [self._img_obs_resolution, self._img_obs_resolution])
        return imgs_grey_vec_hw.view(self.num_envs, self._img_obs_resolution, self._img_obs_resolution)

    def step(self):
        if not self._img_obs:
            r =  super().step()
            # ggLog.info(f"state = {self.get_states()}")
            return r
        # Modified version for image stacking
        self.pre_step()
        estimated_step_duration_sec = 0.0
        t0 = time.monotonic()
        self._adapter.initialize_for_step()
        sub_step_imgs_vec_hw : list[th.Tensor] = [None,None,None]
        substep_len = self._intendedStepLength_sec/self._img_obs_frame_stacking_size
        for i in range(self._img_obs_frame_stacking_size):
            estimated_step_duration_sec += self._adapter.run(substep_len)
            imgs_vec_hw, times = self._adapter.getRenderings([self._ui_camera_name])
            imgs_vec_hw = self.reshape_imgs(imgs_vec_chw=imgs_vec_hw[0].permute(0,3,1,2).to(device=self._th_device, non_blocking=self._th_device.type=="cuda"))
            sub_step_imgs_vec_hw[i] = imgs_vec_hw
        self._stacked_img = th.stack(sub_step_imgs_vec_hw,dim=1)
        t1 = time.monotonic()
        th.add(self._ep_step_counter,1,out=self._ep_step_counter)
        self._tot_step_counter+=1
        self._estimated_env_times += estimated_step_duration_sec
        if abs(estimated_step_duration_sec - self._intendedStepLength_sec) > self._step_precision_tolerance:
            ggLog.warn(f"Step duration is different than intended: {estimated_step_duration_sec} != {self._intendedStepLength_sec}")
        self.post_step()
        tf = time.monotonic()
