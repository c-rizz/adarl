from __future__ import annotations
import adarl.utils.utils
import gymnasium as gym
import cv2
import os
import time
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
from vidgear.gears import WriteGear
import math
import adarl.utils.session
from adarl.utils.utils import puttext_cv, masked_assign
from typing import Callable, Optional, Any
import h5py
import lzma
import pickle
from adarl.utils.tensor_trees import flatten_tensor_tree, map_tensor_tree, stack_tensor_tree, is_all_bounded, is_all_finite, TensorTree
import torch as th
import adarl.utils.tensor_trees as tt
from typing_extensions import override
from adarl.envs.vec.EnvRunnerInterface import EnvRunnerInterface, ObsType
from adarl.envs.vec.EnvRunnerWrapper import EnvRunnerWrapper
import adarl.utils.dbg.dbg_img
import adarl.utils.dbg.dbg_img as dbg_img 
from adarl.utils.spaces import get_space_labels
from adarl.envs.vec.BaseVecEnv import BaseVecEnv
import hdf5plot.save

class EnvRunnerRecorderWrapper(EnvRunnerWrapper[ObsType]):
    def __init__(self,  runner : EnvRunnerInterface[ObsType],
                        fps : float,
                        outFolder : str,
                        env_index : int,
                        saveBestEpisodes = False, 
                        saveFrequency_ep = 1,
                        vec_obs_key = None,
                        overlay_text_func : Optional[Callable[[Any,Any,Any,Any,Any,Any],str]] = None,
                        overlay_text_xy = (0.05,0.05),
                        overlay_text_height = 0.04,
                        overlay_text_color_rgb = (20,20,255),
                        use_global_ep_count = True,
                        record_infoobs = True,
                        record_video = True,
                        publish : bool = False,
                        stream : bool = False):
        super().__init__(runner=runner)
        if stream:
            adarl.utils.dbg.dbg_img.helper.enable_web_dbg(True)
        self._publish_imgs = publish
        self._env_idx = env_index
        self._use_global_ep_count = use_global_ep_count
        self._outFps = fps
        self._frameRepeat = 1
        if fps < 30:
            self._frameRepeat = int(math.ceil(30/fps))
            self._outFps = fps*self._frameRepeat
        self._imgBuffer = []
        self._record_infoobs = record_infoobs
        self._record_video = record_video
        self._vecBuffer = {"vecobs":[], "action":[], "reward":[], "terminated":[], "truncated":[]}
        self._infoBuffer = []
        self._outFolder = outFolder
        self._saveBestEpisodes = saveBestEpisodes
        self._saveFrequency_ep = saveFrequency_ep
        self._bestReward = float("-inf")
        self._tot_vstep_counter = 0
        self._ep_counts = th.zeros((self.num_envs,), device=runner.th_device, dtype=th.long)
        self._ep_rewards = th.zeros((self.num_envs,), device=runner.th_device, dtype=th.float32)
        self._ep_step_counts = th.zeros((self.num_envs,), device=runner.th_device, dtype=th.long)
        self._vec_obs_key = vec_obs_key
        self._has_vec_obs = False
        # if self._vec_obs_key is None and isinstance(self.vec_observation_space, gym.spaces.Dict):
        #     if len(self.vec_observation_space.spaces) == 1:
        #         onlykey = next(iter(self.vec_observation_space.spaces.keys()))
        #         if isinstance(self.vec_observation_space.spaces[onlykey], gym.spaces.Box):
        #             self._vec_obs_key = onlykey
            # else:
            #     raise RuntimeError(f"No vec_obs_key was provided and the observation space is"
            #                        f" a dict of more than 1 element. (env.observation_space = {env.observation_space})")

        self._overlay_text_func = overlay_text_func
        self._overlay_text_xy = overlay_text_xy
        self._overlay_text_height = overlay_text_height
        self._overlay_text_color_bgr = overlay_text_color_rgb[2],overlay_text_color_rgb[1],overlay_text_color_rgb[0]
        self._last_saved_ep = float("-inf")
        self._saved_eps_count = 0
        self._saved_best_eps_count = 0
        self._stored_frames = 0

        space_labels = get_space_labels(self._runner.info_space)
        self._info_labels_np = map_tensor_tree(space_labels, lambda t: t.detach().cpu().numpy() if t is not None else None)
        # self._vecobs_labels = get_space_labels(self._runner.single_observation_space)
        # # ggLog.info(f"self._vecobs_labels = {self._vecobs_labels}")
        # if self._vec_obs_key is not None:
        #     self._vecobs_labels = self._vecobs_labels[self._vec_obs_key]
        # self._vecobs_labels = flatten_tensor_tree(self._vecobs_labels)
        # self._vecobs_labels = map_tensor_tree(self._vecobs_labels, lambda t: th.unsqueeze(t,0) if t is not None else None) # for back compatibility
        # # self._obs_labels = map_tensor_tree(self._obs_labels, lambda t: th.unsqueeze(t,0) if t is not None else None) # for back compatibility
        # # ggLog.info(f"obs labels = {self._vecobs_labels}")

        self.add_on_ep_end_callback(self._on_ep_end)

        os.makedirs(self._outFolder, exist_ok=True)
        os.makedirs(self._outFolder+"/best", exist_ok=True)

    def _record_step(self, obs, action, info, reward, terminated, truncated):
        if self._record_video:
            imgs = self._runner.get_ui_renderings()[0]
        else:
            imgs = None
        obs, action, info, reward, terminated, truncated, img = map_tensor_tree((obs, action, info, reward, terminated, truncated, imgs),
                                                                           lambda tensor: tensor[self._env_idx] if tensor is not None else None)
        if self._record_video and self._publish_imgs:
            dbg_img.helper.publishDbgImg("render", img_callback=lambda: img)
        if self._vec_obs_key is not None:
            vecobs = obs[self._vec_obs_key]
        else:
            vecobs = obs
        if isinstance(action, th.Tensor):
            action = action.cpu()
        action = np.array(action)
        self._update_buffers(img, vecobs, action, reward, terminated, truncated, info)
        # ggLog.info(f"recorded step: action = {action}, stored_steps = {self._stored_steps}")

    @override
    def step(self, actions):
        # ggLog.info(f"rec.step()")
        self._ep_step_counts += 1
        self._tot_vstep_counter += 1
        vstep_ret_tuple =  self._runner.step(actions)
        self._ep_rewards += vstep_ret_tuple[2]
        ep_count = adarl.utils.session.default_session.run_info["collected_episodes"].value if self._use_global_ep_count else  self._ep_counts[self._env_idx]
        if self._may_episode_be_saved(ep_count):
            # ggLog.info(f"Recording step (ep_count={ep_count}, freq={self._saveFrequency_ep}), pub={self._publish_imgs}, sbest={self._saveBestEpisodes}")
            (consequent_observations, next_start_observations,
                rewards, terminateds, truncateds, consequent_infos,
                next_start_infos, reinit_dones) = vstep_ret_tuple
            
            # if reinit_dones:
            #     # Then, a reset just happened; action, terminated and truncated are invalid
            #     actions, terminateds, truncateds = None,None,None

            # If the episode just started (i.e. reinit_done == True), we must save next_start instead of consequent, as
            # the buffers have just been flushed in the _on_ep_end callback.
            # In all other cases next_start==consequent
            # So we can simply always save next_start
            self._record_step(next_start_observations, actions, next_start_infos, rewards, terminateds, truncateds)
        return vstep_ret_tuple

    @override
    def reset(self, seed = None, options = {}) -> tuple[ObsType, TensorTree[th.Tensor]]:
        # ggLog.info(f"rec.reset()")
        obss, infos = super().reset(seed=seed, options=options)
        ep_count = adarl.utils.session.default_session.run_info["collected_episodes"].value if self._use_global_ep_count else  self._ep_counts[self._env_idx]
        if self._may_episode_be_saved(ep_count):
            self._record_step(obs = obss, action = None, info = infos, reward=None, terminated=None, truncated=None)
        return obss, infos

    def _update_buffers(self, img, vecobs, action, reward, terminated, truncated, info):
        self._imgBuffer.append(img)
        self._update_vecbuffer(vecobs, action, reward, terminated, truncated)
        self._infoBuffer.append(info)
        self._stored_frames += 1

    def _update_vecbuffer(self, vecobs, action, reward, terminated, truncated):
        if self._has_vec_obs:
            self._vecBuffer["vecobs"].append(vecobs)
        if self._stored_frames > 0: # at step 0 these are invalid
            self._vecBuffer["action"].append(action)
            self._vecBuffer["reward"].append(reward)
            self._vecBuffer["terminated"].append(terminated)
            self._vecBuffer["truncated"].append(truncated)


    def _writeVideo(self, outFilename : str, imgs : list[th.Tensor | None], vecs, infos):
        if len(imgs)>0:
            # ggLog.info(f"RecorderGymWrapper: {len(imgs)} frames: "+outFilename)
            #outFile = self._outVideoFile+str(self._episodeCounter).zfill(9)
            npimgs = [t.cpu().numpy() if t is not None else None for t in imgs]
            if not outFilename.endswith(".mp4"):
                outFilename+=".mp4"
            in_resolution_wh = None
            goodImg = None
            for npimg in npimgs:
                if npimg is not None:
                    goodImg = npimg
                    break
            if goodImg is None:
                ggLog.warn("RecorderGymWrapper: No valid images in framebuffer, will not write video")
                return
            in_resolution_wh = (goodImg.shape[1], goodImg.shape[0]) # npimgs are hwc
            height = in_resolution_wh[1]
            minheight = 360
            if height<minheight:
                height = minheight
            out_resolution_wh = [int(height/in_resolution_wh[1]*in_resolution_wh[0]), height]
            if out_resolution_wh[0] % 2 != 0:
                out_resolution_wh[0] += 1
            # ggLog.info(f"in_size = {goodImg.shape}")
            # ggLog.info(f"out_resolution_wh = {out_resolution_wh}")
            
            output_params = {   "-c:v": "libx264",
                                "-crf": 23,
                                "-profile:v":
                                "baseline",
                                "-input_framerate":self._outFps,
                                "-disable_force_termination" : True,
                                "-level" : 3.0,
                                "-pix_fmt" : "yuv420p"} 
            writer = WriteGear(output=outFilename, logging=False, **output_params)
            for i in range(len(npimgs)):
                npimg = npimgs[i]
                info = infos[i]
                if npimg is None:
                    npimg = np.zeros_like(goodImg)
                npimg = cv2.resize(npimg,dsize=out_resolution_wh,interpolation=cv2.INTER_NEAREST)
                npimg = self._preproc_frame(npimg)
                if self._overlay_text_func is not None:
                    if self._has_vec_obs:
                        vecobs = vecs["vecobs"][i]
                    else:
                        vecobs = None
                    if i == 0:
                        action, reward, terminated, truncated = None, None, None, None
                    else:
                        action = vecs["action"][i-1]
                        reward = vecs["reward"][i-1]
                        terminated = vecs["terminated"][i-1]
                        truncated = vecs["truncated"][i-1]
                    text = self._overlay_text_func(vecobs, action, reward, terminated, truncated, info)
                    puttext_cv(npimg, text,
                                origin = (int(npimg.shape[1]*self._overlay_text_xy[0]), int(npimg.shape[0]*self._overlay_text_xy[1])),
                                rowheight = int(npimg.shape[0]*self._overlay_text_height),
                                fontScale = 1.0,
                                color = self._overlay_text_color_bgr)
                for _ in range(self._frameRepeat):
                    writer.write(npimg)
            writer.close()
        
    def _write_infobuffer(self, out_filename, infobuffer):
        infos = tt.map_tensor_tree(infobuffer, lambda t: th.as_tensor(t).detach())
        infos = tt.stack_tensor_tree(infos)
        infos_np = tt.map_tensor_tree(infos, lambda t: t.cpu().numpy())
        hdf5plot.save.save_dict(out_filename+".hdf5", infos_np, self._info_labels_np)

        
    def _write_vecbuffer(self, out_filename, vecbuffer, vecbuffer_labels={}):
        out_filename += ".hdf5"
        i = 0
        # ggLog.info(f"writing buffer {vecbuffer}")
        with h5py.File(out_filename, "w") as f:
            # loop through obs, action, reward, terminated, truncation
            for k,v in vecbuffer.items():
                if k == "vecobs" and not self._has_vec_obs:
                    continue
                # ggLog.info(f"{self._vec_obs_key} writing subbuffer {k}:{v}")
                try:
                    # we now have a list of observations (or actions, rewards, ...), make the list into batched obs
                    v = map_tensor_tree(v, lambda t: th.as_tensor(t).detach().cpu()) # make it a tensor if it isnt
                    v = stack_tensor_tree(src_trees=v)
                    v = flatten_tensor_tree(v) # flatten in case we have complex observations
                    for sk,sv in v.items():
                        f.create_dataset(f"{k}.{sk}", data=sv)
                    # if self._vecobs_labels is not None:
                    #     for sk,sv in self._vecobs_labels.items():
                    #         f.create_dataset(f"{k}.{sk}_labels", data=sv)
                except TypeError as e:
                    raise RuntimeError(f"Error saving {k}, type={type(v)}, exception={adarl.utils.utils.exc_to_str(e)}")


    def _saveLastEpisode(self, filename : str):
        if len(self._imgBuffer) > 1:
            if self._has_vec_obs:
                for i in range(len(self._vecBuffer)):
                    vecobs = self._vecBuffer["vecobs"][i]
                    self._vecBuffer["vecobs"][i] = map_tensor_tree(flatten_tensor_tree(vecobs),
                                                                lambda l: vecobs if isinstance(vecobs, np.ndarray) else l.cpu().numpy())
            if self._record_video:
                self._writeVideo(filename,self._imgBuffer, self._vecBuffer, self._infoBuffer)
            if self._record_infoobs:
                self._write_vecbuffer(filename,self._vecBuffer)
                self._write_infobuffer(filename+"_info",self._infoBuffer)
            self._saved_eps_count += 1
        
        

    def _preproc_frame(self, img_hwc):
        # ggLog.info(f"raw frame shape = {img_whc.shape}")
        if img_hwc.dtype == np.float32:
            img_hwc = (img_hwc*255).astype(dtype=np.uint8, copy=False)

        if len(img_hwc.shape) not in [2,3] or img_hwc.shape[2] not in [1,3] or img_hwc.dtype != np.uint8:
            raise RuntimeError(f"Unsupported image format, dtpye={img_hwc.dtype}, shape={img_hwc.shape}")
        
        if len(img_hwc.shape) == 2:
            img_hwc = np.expand_dims(img_hwc,axis=2)
        if img_hwc.shape[2] == 1:
            img_hwc = np.repeat(img_hwc,3,axis=2)
        
        # ggLog.info(f"Preproc frame shape = {img_whc.shape}")

        if img_hwc.shape[1]<256:
            shape_wh = (256,int(256/img_hwc.shape[0]*img_hwc.shape[1]))
            img_hwc = cv2.resize(img_hwc,dsize=shape_wh,interpolation=cv2.INTER_NEAREST)

        # bgr -> rgb
        img_hwc[:,:,[0,1,2]] = img_hwc[:,:,[2,1,0]]
        
        return img_hwc

    def _may_episode_be_saved(self, ep_count):
        return self._saveBestEpisodes or (self._saveFrequency_ep>0 and ep_count % self._saveFrequency_ep == 0) or self._publish_imgs

    def _on_ep_end(self,    envs_ended_mask : th.Tensor,
                            last_observations : ObsType,
                            last_actions : th.Tensor | None,
                            last_infos : TensorTree[th.Tensor],
                            last_rewards : th.Tensor,
                            last_terminateds : th.Tensor, 
                            last_truncateds : th.Tensor):
        # ggLog.info(f"rec._on_ep_end()")
        ep_count = adarl.utils.session.default_session.run_info["collected_episodes"].value if self._use_global_ep_count else  self._ep_counts[self._env_idx]
        run_id = adarl.utils.session.default_session.run_info["run_id"]
        if self._may_episode_be_saved(ep_count) and envs_ended_mask[self._env_idx] and self._stored_frames > 1:
            # Episode with at least a full step finishing
            if self._stored_frames!=self._ep_step_counts[self._env_idx]+1:
                # The if is needed to distinguish between an autoreset and a normal reset
                # If the two counters are different we are in the middle of a step, we are in an autoreset
                # If they are the same we are in a reset triggered from the outside
                # In this case do not save the last obs/action/etcetera, they have already been saved in the last step
                self._record_step(last_observations, last_actions, last_infos, last_rewards, last_terminateds, last_truncateds)
            step_count = adarl.utils.session.default_session.run_info["collected_steps"].value if self._use_global_ep_count else  self._tot_vstep_counter*self.num_envs
            fname = f"ep_{run_id}_{self._saved_eps_count}_{ep_count:09d}_{step_count:010d}_{self._ep_rewards[self._env_idx]:09.9g}"
            if self._saveBestEpisodes and self._ep_rewards[self._env_idx] > self._bestReward:
                if self._saveBestEpisodes:
                    self._saveLastEpisode(f"{self._outFolder}/best/{fname}")            
                    self._saved_best_eps_count += 1
            if self._saveFrequency_ep==1 or (self._saveFrequency_ep>0 and ep_count - self._last_saved_ep >= self._saveFrequency_ep):
                self._saveLastEpisode(f"{self._outFolder}/{fname}")
                self._last_saved_ep = ep_count

        if self._saveBestEpisodes and self._ep_rewards[self._env_idx]>self._bestReward and envs_ended_mask[self._env_idx]:
            self._bestReward = self._ep_rewards[self._env_idx]
        self._ep_rewards[envs_ended_mask] = 0.0
        self._ep_step_counts[envs_ended_mask] = 0
        self._ep_counts[envs_ended_mask] = 0

        if envs_ended_mask[self._env_idx]:
            self._imgBuffer = []
            self._vecBuffer = {"vecobs":[], "action":[], "reward":[], "terminated":[], "truncated":[]}
            self._infoBuffer = []
            self._stored_frames = 0
        

    def close(self):
        # ggLog.info(f"self._outFolder = {self._outFolder}")
        # self._saveLastEpisode(self._outFolder+(f"/ep_{self._episodeCounter}".zfill(6)+f"_{self._epReward}.mp4"))
        ep_count = adarl.utils.session.default_session.run_info["collected_episodes"].value if self._use_global_ep_count else  self._ep_counts[self._env_idx]
        run_id = adarl.utils.session.default_session.run_info["run_id"]
        step_count = adarl.utils.session.default_session.run_info["collected_steps"].value if self._use_global_ep_count else  self._tot_vstep_counter*self.num_envs
        fname = f"ep_{run_id}_{ep_count:09d}_{step_count:010d}_{self._ep_rewards[self._env_idx]:09.9g}_{self._saved_eps_count}"
        self._saveLastEpisode(f"{self._outFolder}/{fname}")
        return self._runner.close()

    # def setSaveAllEpisodes(self, enable : bool, disable_after_one_episode : bool = False):
    #     self._saveAllEpisodes = enable
    #     self._disableAfterEp = disable_after_one_episode


    def get_base_env(self) -> BaseVecEnv[ObsType]:
        """Get the underlying adarl base environment

        Returns
        -------
        BaseEnv
            The adarl.BaseEnv object.
        """
        return self._runner.get_base_env()