import gymnasium as gym
import cv2
import os
import time
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
from vidgear.gears import WriteGear
import math
from lr_gym.utils.utils import puttext_cv
from typing import Callable, Optional, Any

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
                        overlay_text_func : Optional[Callable[[Any,Any,Any,Any,Any,Any],str]] = None):
        super().__init__(env)
        self._outFps = fps
        self._frameRepeat = 1
        if fps < 30:
            self._frameRepeat = int(math.ceil(30/fps))
            self._outFps = fps*self._frameRepeat
        self._frameBuffer = []
        self._vecBuffer = []
        self._infoBuffer = []
        self._episodeCounter = 0
        self._outFolder = outFolder
        self._saveBestEpisodes = saveBestEpisodes
        self._saveFrequency_ep = saveFrequency_ep
        self._bestReward = float("-inf")
        self._epReward = 0.0
        self._epStepCount = 0
        self._vec_obs_key = vec_obs_key
        self._overlay_text_func = overlay_text_func
        self._last_saved_ep = float("-inf")
        try:
            os.makedirs(self._outFolder)
        except FileExistsError:
            pass
        try:
            os.makedirs(self._outFolder+"/best")
        except FileExistsError:
            pass

    def step(self, action):
        obs, rew, terminated, truncated, info =  self.env.step(action)
        if self._may_episode_be_saved(self._episodeCounter):
            img = self.render()
            if img is not None:
                self._frameBuffer.append(img)
            else:
                self._frameBuffer.append(None)
            if self._vec_obs_key is not None:
                vecobs = np.array(obs[self._vec_obs_key])
            else:
                vecobs = None
            action = np.array(action)
            self._vecBuffer.append([vecobs, action, rew, terminated, truncated])
            self._infoBuffer.append(info)
            # else:
            #     self._vecBuffer.append([obs, rew, done, info])
        self._epReward += rew
        self._epStepCount += 1
        return obs, rew, terminated, truncated, info



    def _writeVideo(self, outFilename : str, imgs, vecs, infos):
        if len(imgs)>0:
            ggLog.info(f"RecorderGymWrapper saving {len(imgs)} frames video to "+outFilename)
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
                vec, info = vecs[i], infos[i]
                if npimg is None:
                    npimg = np.zeros_like(goodImg)
                npimg = cv2.resize(npimg,dsize=out_resolution_wh,interpolation=cv2.INTER_NEAREST)
                npimg = self._preproc_frame(npimg)
                if self._overlay_text_func is not None:
                    vecobs, action, rew, terminated, truncated = vec
                    text = self._overlay_text_func(vecobs, action, rew, terminated, truncated, info)
                    puttext_cv(npimg, text,
                                origin = (int(npimg.shape[1]*0.1), int(npimg.shape[0]*0.1)),
                                rowheight = int(npimg.shape[0]*0.05),
                                fontScale = 1.0,
                                color = (20,20,255))
                for _ in range(self._frameRepeat):
                    writer.write(npimg)
            writer.close()

    def _writeVecs(self, outFilename : str, vecs):
        with open(outFilename+".txt", 'a') as f:
            for vec in vecs:
                f.write(f"{vec}\n")

    def _saveLastEpisode(self, filename : str):
        self._writeVideo(filename,self._frameBuffer, self._vecBuffer, self._infoBuffer)
        self._writeVecs(filename,self._vecBuffer)
        self._writeVecs(filename+"_info",self._infoBuffer)
        

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
        return self._saveBestEpisodes or (self._saveFrequency_ep>0 and ep_count % self._saveFrequency_ep == 0)

    def reset(self, **kwargs):
        if self._epStepCount > 0:
            if self._epReward > self._bestReward:
                self._bestReward = self._epReward
                if self._saveBestEpisodes:
                    self._saveLastEpisode(self._outFolder+"/best/ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")            
            if self._saveFrequency_ep>0 and self._episodeCounter - self._last_saved_ep >= self._saveFrequency_ep:
                self._saveLastEpisode(self._outFolder+"/ep_"+(f"{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
                self._last_saved_ep = self._episodeCounter


        obs, info = self.env.reset(**kwargs)

        if self._epReward>self._bestReward:
            self._bestReward = self._epReward
        if self._epStepCount>0:
            self._episodeCounter += 1
        self._epReward = 0.0
        self._epStepCount = 0        
        self._frameBuffer = []
        self._vecBuffer = []
        self._infoBuffer = []
        if self._vec_obs_key is not None:
            vecobs = obs[self._vec_obs_key]
        else:
            vecobs = None
        self._vecBuffer.append([vecobs, None, None, None, None])
        self._infoBuffer.append(info)
        
        # else:
        #     self._vecBuffer.append([obs, None, None, None])
        if self._may_episode_be_saved(self._episodeCounter):
            img = self.render()
            if img is not None:
                self._frameBuffer.append(img)
        return obs, info

    def close(self):
        self._saveLastEpisode(self._outFolder+(f"/ep_{self._episodeCounter}").zfill(6)+f"_{self._epReward}.mp4")
        return self.env.close()

    def setSaveAllEpisodes(self, enable : bool, disable_after_one_episode : bool = False):
        self._saveAllEpisodes = enable
        self._disableAfterEp = disable_after_one_episode
