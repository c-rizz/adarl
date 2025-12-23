#!/usr/bin/env python3
from __future__ import annotations
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
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
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

class CartpoleContinuousVisualVecEnv(CartpoleContinuousVecEnv):
    def __init__(   self,
                    adapter : BaseVecJointImpedanceAdapter | BaseVecJointEffortAdapter,
                    render : bool = False,
                    step_duration_sec : float = 0.05,
                    wall_sim_speed = False,
                    seed = 1,
                    th_device : th.device = th.device("cpu"),
                    task : str = "balance",
                    img_obs : bool = False,
                    img_obs_resolution : int = 64,
                    img_obs_frame_stacking_size : int = 3,
                    sparse_reward = True,
                    max_episode_steps : int = 1000,
                    terminate_on_rail_distance = False,
                    terminate_on_pole_angle = True,
                    use_gym_inverted_pendulum_model = False):
        


        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._renderingEnabled = render
        self._ui_camera_name = "simple_camera"
        self._task = task
        self._sparse_reward = sparse_reward
        self._terminate_on_rail_distance = th.as_tensor(terminate_on_rail_distance, device=th_device)
        self._terminate_on_pole_angle = th.as_tensor(terminate_on_pole_angle, device=th_device)
        self._gym_inverted_pendulum = use_gym_inverted_pendulum_model
        self._upright_hinge_threshold = 0.2 # like gym's InvertedPendulum
        self._max_cart_dist = 2
        self._init_noise_scale = 0.01
        self._force_range = 3.0 * 100 if use_gym_inverted_pendulum_model else 50.0
        self._img_obs = img_obs
        self._img_obs_resolution = img_obs_resolution
        self._img_obs_frame_stacking_size = img_obs_frame_stacking_size
        
        base_state_space, base_single_observation_space, base_reward_space = self._build_spaces(th_device)

        if img_obs:
            img_state_space = ThBox(low=0, high=255,
                                    shape=(self._img_obs_frame_stacking_size, self._img_obs_resolution, self._img_obs_resolution),
                                    dtype=th.uint8,
                                    torch_device=th_device)
            single_observation_space = img_state_space
            single_state_space = gym_spaces.Dict(base_state_space.spaces | {"img": img_state_space})
        else:
            single_observation_space = base_single_observation_space
            single_state_space = base_state_space


        act_max = np.array([1.0])
        super(CartpoleContinuousVecEnv,self).__init__(th_device=th_device,
                                                seed=seed,
                                                obs_dtype=th.float32,
                                                single_action_space = ThBox(-act_max,act_max, torch_device=th_device),
                                                single_observation_space = single_observation_space,
                                                single_state_space=single_state_space,
                                                single_reward_space=base_reward_space,
                                                info_space=None,
                                                step_duration_sec=step_duration_sec,
                                                adapter=adapter,
                                                max_episode_steps=max_episode_steps)
        example_labels : dict[str,th.Tensor] = {}
        example_state = {k:th.as_tensor((s.low+s.high)/2).to(device=th_device).unsqueeze(0).repeat(self.num_envs, *([1]*len(s.shape)) ) for k,s in single_state_space.spaces.items()}
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
            sub_step_imgs_vec_hw : list[th.Tensor] = [None,None,None]  #type: ignore
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
    def get_states(self) -> dict[str, th.Tensor]:
        state = super().get_states()
        if self._img_obs:
            state["img"] = self._stacked_img
        # ggLog.info(f"state = {state}")
        return state

    # def _get_spawn_defs(self):
    #     base_defs = super()._get_spawn_defs()
    #     if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
    #         cam_file = "models/simple_camera.mjcf.xacro"
    #     else:            
    #         cam_file = "models/simple_camera.sdf.xacro"
    #     camera_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
    #                                 name="simple_camera",
    #                                 pose=None,
    #                                 format="sdf.xacro",
    #                                 kwargs={"camera_width":426,
    #                                         "camera_height":240,
    #                                         "frame_rate":1/self._intendedStepLength_sec})
    #     cartpole_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
    #                                                     name="cartpole_v0",
    #                                                     pose=None,
    #                                                     format="urdf.xacro",
    #                                                     kwargs={"use_collisions" : "false"})
    #     return base_defs + [cartpole_def, camera_def]

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
        sub_step_imgs_vec_hw : list[th.Tensor] = [None,None,None] #type: ignore
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

    def get_infos(self,states, labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        vstates = states["vec"]
        step_count = states["vec"][:,self._TIMESTEP]
        sub_rewards : dict[str,th.Tensor] = {}
        reward = self.compute_rewards(states, sub_rewards)
        info =  {"pole_angle" : th.atan2(vstates[:,self._POLE_SIN],vstates[:,self._POLE_COS]),
                "cart_pos" : vstates[:,self._CART_POS],
                "step_count" : step_count,
                "reward" : reward,
                "jstats_avg" : states["jstats"][:,2,:,0],
                "lstats_avg" : states["lstats"][:,2,:,0],
                "lvels" : states["lvels"].reshape(self.num_envs,2*3),
                "hinge_vel" : vstates[:,self._POLE_VEL],
                "slider_vel" : vstates[:,self._CART_VEL]}
        info["obs"] = self.get_observations(states)
        if labels is not None:
            labels["vecobs"] = self.single_observation_space.labels
        info.update({"reward_"+k:v for k,v in sub_rewards.items()})
        return info