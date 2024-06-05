import gymnasium as gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
from vidgear.gears import WriteGear
import math
import lr_gym.utils.session
from lr_gym.utils.utils import puttext_cv
from typing import Callable, Optional, Any
import h5py
import lzma
import pickle
from lr_gym.utils.tensor_trees import flatten_tensor_tree
import torch as th

class RecorderGymWrapper(gym.Wrapper):
    """Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.
    .. note::
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """
    def __init__(self, env : gym.Env, fps : float, outFolder : str,
                        saveBestEpisodes = False, 
                        saveFrequency_ep = 1,
                        vec_obs_key = None,
                        overlay_text_func : Optional[Callable[[Any,Any,Any,Any,Any,Any],str]] = None,
                        overlay_text_xy = (0.05,0.05),
                        overlay_text_height = 0.04,
                        overlay_text_color_rgb = (20,20,255),
                        use_global_ep_count = True,
                        only_video = False):
        super().__init__(env)
        self._use_global_ep_count = use_global_ep_count
        self._outFps = fps
        self._frameRepeat = 1
        if fps < 30:
            self._frameRepeat = int(math.ceil(30/fps))
            self._outFps = fps*self._frameRepeat
        self._frameBuffer = []
        self._only_video = only_video
        self._vecBuffer = {"vecobs":[], "action":[], "reward":[], "terminated":[], "truncated":[]}
        self._infoBuffer = []
        self._episodeCounter = 0
        self._outFolder = outFolder
        ggLog.info(f"outFolder = {outFolder}")
        self._saveBestEpisodes = saveBestEpisodes
        self._saveFrequency_ep = saveFrequency_ep
        self._bestReward = float("-inf")
        self._epReward = 0.0
        self._epStepCount = 0
        self._vec_obs_key = vec_obs_key
        self._overlay_text_func = overlay_text_func
        self._overlay_text_xy = overlay_text_xy
        self._overlay_text_height = overlay_text_height
        self._overlay_text_color_bgr = overlay_text_color_rgb[2],overlay_text_color_rgb[1],overlay_text_color_rgb[0]
        self._last_saved_ep = float("-inf")
        self._saved_eps_count = 0
        self._saved_best_eps_count = 0

        os.makedirs(self._outFolder, exist_ok=True)
        os.makedirs(self._outFolder+"/best", exist_ok=True)

    def step(self, action):
        obs, reward, terminated, truncated, info =  self.env.step(action)
        if self._may_episode_be_saved(self._episodeCounter):
            img = self.render()
            if img is not None:
                self._frameBuffer.append(img)
            else:
                self._frameBuffer.append(None)
            if self._vec_obs_key is not None:
                vecobs = np.array(obs[self._vec_obs_key])
            else:
                vecobs = flatten_tensor_tree(obs)
            if isinstance(action, th.Tensor):
                action = action.cpu()
            action = np.array(action)
            self._update_vecbuffer(vecobs, action, reward, terminated, truncated)
            self._infoBuffer.append(info)
            # else:
            #     self._vecBuffer.append([obs, rew, done, info])
        self._epReward += reward
        self._epStepCount += 1
        return obs, reward, terminated, truncated, info

    def _update_vecbuffer(self, vecobs, action, reward, terminated, truncated):
        self._vecBuffer["vecobs"].append(vecobs)
        if action is not None: self._vecBuffer["action"].append(action)
        if reward is not None: self._vecBuffer["reward"].append(reward)
        if terminated is not None: self._vecBuffer["terminated"].append(terminated)
        if truncated is not None: self._vecBuffer["truncated"].append(truncated)


    def _writeVideo(self, outFilename : str, imgs, vecs, infos):
        if len(imgs)>0:
            ggLog.info(f"RecorderGymWrapper: {len(imgs)} frames: "+outFilename)
            #outFile = self._outVideoFile+str(self._episodeCounter).zfill(9)
            if not outFilename.endswith(".mp4"):
                outFilename+=".mp4"
            in_resolution_wh = None
            goodImg = None
            for npimg in imgs:
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
                
            output_params = {   "-c:v": "libx264",
                                "-crf": 23,
                                "-profile:v":
                                "baseline",
                                "-input_framerate":self._outFps,
                                "-disable_force_termination" : True,
                                "-level" : 3.0,
                                "-pix_fmt" : "yuv420p"} 
            writer = WriteGear(output=outFilename, logging=False, **output_params)
            for i in range(len(imgs)):
                npimg = imgs[i]
                info = infos[i]
                if npimg is None:
                    npimg = np.zeros_like(goodImg)
                npimg = cv2.resize(npimg,dsize=out_resolution_wh,interpolation=cv2.INTER_NEAREST)
                npimg = self._preproc_frame(npimg)
                if self._overlay_text_func is not None:
                    vecobs = vecs["vecobs"][i]
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
        out_filename += ".xz"
        with lzma.open(out_filename, "wb") as f:
            pickle.dump(infobuffer, f)
        
    def _write_vecbuffer(self, out_filename, vecbuffer):
        out_filename += ".hdf5"
        # ggLog.info(f"writing buffer {vecbuffer}")
        with h5py.File(out_filename, "w") as f:
            for k,v in vecbuffer.items():
                # ggLog.info(f"writing subbuffer {v}")
                if isinstance(v,dict):
                    for sk,sv in vecbuffer.items():
                        f.create_dataset(f"{k}.{sk}", data=sv)
                else:
                    f.create_dataset(k, data=v)


    def _saveLastEpisode(self, filename : str):
        self._writeVideo(filename,self._frameBuffer, self._vecBuffer, self._infoBuffer)
        if not self._only_video:
            self._write_vecbuffer(filename,self._vecBuffer)
            self._write_infobuffer(filename+"_info",self._infoBuffer)
        

    def _preproc_frame(self, img_hwc):
        # ggLog.info(f"raw frame shape = {img_whc.shape}")
        if img_hwc.dtype == np.float32:
            img_hwc = (img_hwc*255).astype(dtype=np.uint8, copy=False)

        if len(img_hwc.shape) == 2:
            img_hwc = np.expand_dims(img_hwc,axis=2)
        if img_hwc.shape[2] == 1:
            img_hwc = np.repeat(img_hwc,3,axis=2)
        
        if img_hwc.shape[2] != 3 or img_hwc.dtype != np.uint8:
            raise RuntimeError(f"Unsupported image format, dtpye={img_hwc.dtype}, shape={img_hwc.shape}")
        # ggLog.info(f"Preproc frame shape = {img_whc.shape}")

        if img_hwc.shape[1]<256:
            shape_wh = (256,int(256/img_hwc.shape[0]*img_hwc.shape[1]))
            img_hwc = cv2.resize(img_hwc,dsize=shape_wh,interpolation=cv2.INTER_NEAREST)

        # bgr -> rgb
        img_hwc[:,:,[0,1,2]] = img_hwc[:,:,[2,1,0]]
        
        return img_hwc

    def _may_episode_be_saved(self, ep_count):
        return self._saveFrequency_ep==1 or (self._saveBestEpisodes or (self._saveFrequency_ep>0 and ep_count % self._saveFrequency_ep == 0))

    def reset(self, **kwargs):
        if self._epStepCount > 0:
            ep_count = lr_gym.utils.session.default_session.run_info["collected_episodes"] if self._use_global_ep_count else  self._episodeCounter
            fname = f"ep_{self._saved_best_eps_count:09d}_{ep_count:09d}_{self._epReward:09.9g}"
            if self._epReward > self._bestReward:
                self._bestReward = self._epReward
                if self._saveBestEpisodes:
                    self._saveLastEpisode(f"{self._outFolder}/best/{fname}")            
                    self._saved_best_eps_count += 1
            if self._saveFrequency_ep==1 or (self._saveFrequency_ep>0 and ep_count - self._last_saved_ep >= self._saveFrequency_ep):
                self._saveLastEpisode(f"{self._outFolder}/{fname}")
                self._last_saved_ep = ep_count
                self._saved_eps_count += 1


        obs, info = self.env.reset(**kwargs)

        if self._epReward>self._bestReward:
            self._bestReward = self._epReward
        if self._epStepCount>0:
            self._episodeCounter += 1
        self._epReward = 0.0
        self._epStepCount = 0        
        self._frameBuffer = []
        self._vecBuffer = {"vecobs":[], "action":[], "reward":[], "terminated":[], "truncated":[]}
        self._infoBuffer = []

        if self._vec_obs_key is not None:
            vecobs = np.array(obs[self._vec_obs_key])
        else:
            vecobs = flatten_tensor_tree(obs)
        self._update_vecbuffer(vecobs, None, None, None, None)
        self._infoBuffer.append(info)
        
        ep_count = lr_gym.utils.session.default_session.run_info["collected_episodes"] if self._use_global_ep_count else  self._episodeCounter
        if self._may_episode_be_saved(ep_count):
            img = self.render()
            if img is not None:
                self._frameBuffer.append(img)
        return obs, info

    def close(self):
        # ggLog.info(f"self._outFolder = {self._outFolder}")
        # self._saveLastEpisode(self._outFolder+(f"/ep_{self._episodeCounter}".zfill(6)+f"_{self._epReward}.mp4"))
        ep_count = lr_gym.utils.session.default_session.run_info["collected_episodes"] if self._use_global_ep_count else  self._episodeCounter
        fname = f"ep_{self._saved_best_eps_count:09d}_{ep_count:09d}_{self._epReward:09.9g}"
        self._saveLastEpisode(f"{self._outFolder}/{fname}")
        return self.env.close()

    def setSaveAllEpisodes(self, enable : bool, disable_after_one_episode : bool = False):
        self._saveAllEpisodes = enable
        self._disableAfterEp = disable_after_one_episode
