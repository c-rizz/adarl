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
from adarl.utils.dbg.dbg_checks import dbg_check_size

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
                    use_gym_inverted_pendulum_model = False,
                    camera_offset_xyz = (.0,.0,.0)):

        self._img_obs = img_obs
        self._img_obs_resolution = img_obs_resolution
        self._img_obs_frame_stacking_size = img_obs_frame_stacking_size
        self._lowres_camera_name = "lowres_camera"
        self._lowres_camera_link_name = (self._lowres_camera_name, "simple_camera_link")

        super().__init__(   adapter=adapter,
                            render=render,
                            step_duration_sec=step_duration_sec,
                            wall_sim_speed=wall_sim_speed,
                            seed=seed,
                            th_device=th_device,
                            task=task,
                            sparse_reward=sparse_reward,
                            max_episode_steps=max_episode_steps,
                            terminate_on_rail_distance=terminate_on_rail_distance,
                            terminate_on_pole_angle=terminate_on_pole_angle,
                            use_gym_inverted_pendulum_model=use_gym_inverted_pendulum_model,
                            camera_offset_xyz=camera_offset_xyz)

    def _build_spaces(self, th_device):
        base_state_space, base_single_observation_space, base_reward_space = super()._build_spaces(th_device)

        if self._img_obs:
            img_state_space = ThBox(low=0, high=255,
                                    shape=(self._img_obs_frame_stacking_size, self._img_obs_resolution, self._img_obs_resolution),
                                    dtype=th.uint8,
                                    torch_device=th_device)
            single_observation_space = img_state_space
            single_state_space = gym_spaces.Dict(base_state_space.spaces | {"img": img_state_space})
        else:
            single_observation_space = base_single_observation_space
            single_state_space = base_state_space
        return single_state_space, single_observation_space, base_reward_space

    @override
    def _get_spawn_defs(self):
        r = super()._get_spawn_defs()
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            cam_file = "models/simple_camera.mjcf.xacro"
        else:            
            cam_file = "models/simple_camera.sdf.xacro"
        # We make the camera the smallest resolution that can fit the final cropped and resized image without losing quality
        self._camera_crop_tblr = [74/360, 320/360, 0.0, 1.0]  # top, bottom, left, right crop ratios
        aspect = 426/240 # more or less 16/9
        cam_height = self._img_obs_resolution #*1/(self._camera_crop_tblr[1]-self._camera_crop_tblr[0])
        self._lowres_cam_resolution_hw = (cam_height, int(cam_height*aspect))
        lowres_cam_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                    name=self._lowres_camera_name,
                                    pose=None,
                                    format="sdf.xacro",
                                    kwargs={"camera_width":self._lowres_cam_resolution_hw[1],
                                            "camera_height":self._lowres_cam_resolution_hw[0],
                                            "frame_rate":1/self._intendedStepLength_sec,
                                            "camera_name": self._lowres_camera_name})
        r.append(lowres_cam_def)
        return r

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        super()._initialize_episodes(vec_mask=vec_mask, options=options)
        if self._img_obs:
            sub_step_imgs_vec_hw : list[th.Tensor] = [None,None,None]  #type: ignore
            for i in range(self._img_obs_frame_stacking_size):
                imgs_vec_chw, times = self._adapter.getRenderings([self._lowres_camera_name])
                # ggLog.info(f"imgs_vec_chw[0].shape = {imgs_vec_chw[0].shape}")
                imgs_vec_hw = self.reshape_imgs(imgs_vec_chw=imgs_vec_chw[0].permute(0,3,1,2))
                sub_step_imgs_vec_hw[i] = imgs_vec_hw
            self._stacked_img = th.cat(sub_step_imgs_vec_hw,dim=1).to(device=self._th_device, non_blocking=self._th_device.type == "cuda")
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._adapter.setLinksStateDirect([self._lowres_camera_link_name],
                                            link_states_pose_vel=th.as_tensor(self._camera_pose + [0,0,0,0,0,0]).expand(self.num_envs, 1, 13),
                                            vec_mask=vec_mask)
        else:
            raise NotImplementedError()

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
        return state

    @th.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
    def reshape_imgs(self, imgs_vec_chw : th.Tensor) -> th.Tensor:
        top    = int(self._camera_crop_tblr[0]*imgs_vec_chw.shape[2])
        bottom = int(self._camera_crop_tblr[1]*imgs_vec_chw.shape[2])
        left   = int(self._camera_crop_tblr[2]*imgs_vec_chw.shape[3])
        right  = int(self._camera_crop_tblr[3]*imgs_vec_chw.shape[3])
        imgs_vec_chw  = imgs_vec_chw[:,:,top:bottom,left:right]
        imgs_grey_vec_hw = rgb_to_grayscale(imgs_vec_chw)
        imgs_grey_vec_hw = resize(imgs_grey_vec_hw, [self._img_obs_resolution, self._img_obs_resolution])
        # imgs_grey_vec_hw = imgs_vec_chw[:,0]
        dbg_check_size(imgs_grey_vec_hw, (self.num_envs, 1, self._img_obs_resolution, self._img_obs_resolution))
        return imgs_grey_vec_hw

    @override
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
        cam_h, cam_w = self._lowres_cam_resolution_hw
        obs_h = obs_w = self._img_obs_resolution
        frames = self._img_obs_frame_stacking_size
        substep_len = self._intendedStepLength_sec/frames
        tot_run_time = 0.0
        tot_render_time = 0.0
        tot_reshape_time = 0.0

        all_renderings_fvhwc = th.empty((frames, self.num_envs, cam_h, cam_w, 3),
                                  dtype=th.uint8,
                                  device=th.device("cpu"))
        for i in range(frames):
            t0_sub = time.monotonic()
            estimated_step_duration_sec += self._adapter.run(substep_len)
            t1_sub = time.monotonic()
            self._adapter.getRenderings([   self._lowres_camera_name],
                                        out=[all_renderings_fvhwc[i]])
            t2_sub = time.monotonic()
            tot_run_time += (t1_sub - t0_sub)
            tot_render_time += (t2_sub - t1_sub)
        t_pre_reshape = time.monotonic()
        frames = self.reshape_imgs(imgs_vec_chw=all_renderings_fvhwc.permute(0,1,4,2,3).view(-1,3,cam_h,cam_w)).view((frames, self.num_envs, obs_h, obs_w))
        tot_reshape_time = time.monotonic() - t_pre_reshape
        self._stacked_img = frames.permute(1,0,2,3).to(device=self._th_device, non_blocking=self._th_device.type == "cuda")
        t1 = time.monotonic()
        th.add(self._ep_step_counter,1,out=self._ep_step_counter)
        self._tot_step_counter+=1
        self._estimated_env_times += estimated_step_duration_sec
        if abs(estimated_step_duration_sec - self._intendedStepLength_sec) > self._step_precision_tolerance:
            ggLog.warn(f"Step duration is different than intended: {estimated_step_duration_sec} != {self._intendedStepLength_sec}")
        self.post_step()
        tf = time.monotonic()
        # ggLog.info(f"Step timing: total={tf - t0:.4f}s, tpost={tf-t1:.4f}s, run={tot_run_time:.4f}s, render={tot_render_time:.4f}s, reshape={tot_reshape_time:.4f}s, stime={estimated_step_duration_sec:.6f}s, rt_single={estimated_step_duration_sec/(tf - t0):.2f}x")

