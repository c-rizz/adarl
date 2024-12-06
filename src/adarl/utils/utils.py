from __future__ import annotations

import numpy as np
import time
import cv2
import collections
from typing import List, Tuple, Callable, Dict, Union, Optional, Any, Optional, Literal, TypeVar, Sequence
import os
import quaternion
import datetime
import tqdm
import random
import multiprocessing
import csv
import sys
import importlib
import traceback

import adarl.utils.dbg.ggLog as ggLog
import traceback
import xacro
import torch as th
import subprocess
import re
from typing import TypedDict
import gymnasium as gym

name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":  (np.uint8,  1),
    "bayer_bggr8":  (np.uint8,  1),
    "bayer_gbrg8":  (np.uint8,  1),
    "bayer_grbg8":  (np.uint8,  1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}


numpy_to_torch_dtype_dict = {
    bool          : th.bool,
    np.uint8      : th.uint8,
    np.int8       : th.int8,
    np.int16      : th.int16,
    np.int32      : th.int32,
    np.int64      : th.int64,
    np.float16    : th.float16,
    np.float32    : th.float32,
    np.float64    : th.float64,
    np.complex64  : th.complex64,
    np.complex128 : th.complex128
}
numpy_to_torch_dtype_dict.update({np.dtype(npd):td for npd,td in numpy_to_torch_dtype_dict.items()})

torch_to_numpy_dtype_dict = {v:k for k,v in numpy_to_torch_dtype_dict.items()}

class AverageKeeper:
    def __init__(self, bufferSize = 100):
        self._bufferSize = bufferSize
        self.reset()

    def addValue(self, newValue):
        self._buffer.append(newValue)
        self._all_time_sum += newValue
        self._all_time_count += 1
        self._avg = float(sum(self._buffer))/len(self._buffer)
        self._all_time_avg = self._all_time_sum/self._all_time_count

    def getAverage(self, all_time : bool = False):
        if all_time:
            return self._all_time_avg
        else:
            return self._avg

    def reset(self):
        self._buffer = collections.deque(maxlen=self._bufferSize)
        self._avg = 0.0
        self._all_time_sum = 0.0
        self._all_time_count = 0.0
        self._all_time_avg = 0.0

    def __enter__(self):
        self._t0 = time.monotonic()
    
    def __exit__(self, exc_type, exc_val, exc_t):
        self.addValue(time.monotonic()-self._t0)



from dataclasses import dataclass


T = TypeVar('T')
@dataclass
class Pose:
    position : th.Tensor
    orientation_xyzw : th.Tensor

    def array_xyz_xyzw(self, type : type[T] = th.Tensor) -> T:
        tensor = th.concat([self.position,self.orientation_xyzw])
        if type == th.Tensor:
            return tensor
        elif type == np.ndarray:
            return tensor.cpu().numpy()
        elif type == list:
            return tensor.cpu().tolist()
        elif type == tuple:
            return tuple(tensor.cpu().to())

def build_pose(x,y,z, qx,qy,qz,qw, th_device=None) -> Pose:
    # return {"position" : th.tensor([x,y,z], device=th_device),
    #         "orientation_xyzw" : th.tensor([qx,qy,qz,qw], device=th_device)}
    return Pose(position = th.tensor([x,y,z], device=th_device),
                orientation_xyzw = th.tensor([qx,qy,qz,qw], device=th_device))

@dataclass
class JointStateArray:
    joints_pve : th.Tensor

    def __init__(self, joints_pve : th.Tensor):
        """

        Parameters
        ----------
        joint_dof_pve : th.Tensor
            A tensor in the shape (joints_number, 3). The first dimension represents the joint index,
            then dimension 1 contains position, velocity and effort, in this order.
        """
        self.joints_pve = joints_pve

    @property
    def positions(self):
        return self.joints_pve[:,0]
    
    @property
    def velocities(self):
        return self.joints_pve[:,1]
    
    @property
    def efforts(self):
        return self.joints_pve[:,2]

@dataclass
class JointState:
    position : th.Tensor
    rate : th.Tensor
    effort : th.Tensor
    
    def __init__(self, position : Union[th.Tensor, List[float], float],
                       rate : Union[th.Tensor, List[float], float],
                       effort : Union[th.Tensor, List[float], float]):
        self.position = th.as_tensor(position).view(-1)
        self.rate = th.as_tensor(rate).view(-1)
        self.effort = th.as_tensor(effort).view(-1)

    
@dataclass
class LinkState:
    pose : Pose
    pos_velocity_xyz : th.Tensor
    ang_velocity_xyz : th.Tensor

    def __init__(self, position_xyz : th.Tensor | tuple, orientation_xyzw : th.Tensor | tuple ,
                    pos_com_velocity_xyz : th.Tensor | tuple, ang_velocity_xyz : th.Tensor | tuple):
        if isinstance(position_xyz, tuple):
            position_xyz = th.as_tensor(position_xyz)
        if isinstance(orientation_xyzw, tuple):
            orientation_xyzw = th.as_tensor(orientation_xyzw)
        if isinstance(pos_com_velocity_xyz, tuple):
            pos_com_velocity_xyz = th.as_tensor(pos_com_velocity_xyz)
        if isinstance(ang_velocity_xyz, tuple):
            ang_velocity_xyz = th.as_tensor(ang_velocity_xyz)
        self.pose = build_pose(position_xyz[0],position_xyz[1],position_xyz[2], orientation_xyzw[0],orientation_xyzw[1],orientation_xyzw[2],orientation_xyzw[3])
        self.pos_velocity_xyz = pos_com_velocity_xyz
        self.ang_velocity_xyz = ang_velocity_xyz





def createSymlink(src, dst):
    try:
        os.symlink(src, dst)
    except FileExistsError:
        try:
            os.unlink(dst)
            time.sleep(random.random()*10) #TODO: have a better way to avoid race conditions
            os.symlink(src, dst)
        except:
            pass



    

def puttext_cv(img, string, origin, rowheight, fontScale = 0.5, color = (255,255,255)):
    for i, line in enumerate(string.split('\n')):
        cv2.putText(img,
                    text = line,
                    org=(origin[0],int(origin[1]+rowheight*i)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale = fontScale,
                    color = color)





def evaluatePolicy(env,
                   model,
                   episodes : int,
                   on_ep_done_callback : Callable[[float, int,int],Any] | None = None,
                   predict_func : Optional[Callable[[Any], Tuple[Any,Any]]] = None,
                   progress_bar : bool = False,
                   images_return = None,
                   obs_return = None):
    with th.no_grad():
        if predict_func is None:
            predict_func_ = model.predict
        else:
            predict_func_ = predict_func
        rewards = np.empty((episodes,), dtype = np.float32)
        steps = np.empty((episodes,), dtype = np.int32)
        wallDurations = np.empty((episodes,), dtype = np.float32)
        predictWallDurations = np.empty((episodes,), dtype = np.float32)
        totDuration=0.0
        successes = 0.0
        #frames = []
        #do an average over a bunch of episodes
        if not progress_bar:
            maybe_tqdm = lambda x:x
        else:
            maybe_tqdm = tqdm.tqdm
        for episode in maybe_tqdm(range(0,episodes)):
            frame = 0
            episodeReward = 0
            terminated = False
            truncated = False
            predDurations = []
            t0 = time.monotonic()
            # ggLog.info("Env resetting...")
            obs, info = env.reset()
            # ggLog.info("Env resetted")
            if images_return is not None:
                images_return.append([])
            if obs_return is not None:
                obs_return.append([])
            while not (terminated or truncated):
                t0_pred = time.monotonic()
                # ggLog.info("Predicting")
                if images_return is not None:
                    images_return[-1].append(env.render())
                if obs_return is not None:
                    obs_return[-1].append(obs)
                action, _states = predict_func_(obs)
                predDurations.append(time.monotonic()-t0_pred)
                # ggLog.info("Stepping")
                obs, stepReward, terminated, truncated, info = env.step(action)
                frame+=1
                episodeReward += stepReward
                # ggLog.info(f"Step reward = {stepReward}")
            rewards[episode]=episodeReward
            if "success" in info.keys():
                if info["success"]:
                    ggLog.info(f"Success {successes} ratio = {successes/(episode+1)}")
                    successes += 1
            steps[episode]=frame
            wallDurations[episode]=time.monotonic() - t0
            predictWallDurations[episode]=sum(predDurations)
            if on_ep_done_callback is not None:
                on_ep_done_callback(episodeReward=episodeReward, steps=frame, episode=episode)
            ggLog.debug("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
        eval_results = {"reward_mean" : np.mean(rewards),
                        "reward_std" : np.std(rewards),
                        "steps_mean" : np.mean(steps),
                        "steps_std" : np.std(steps),
                        "success_ratio" : successes/episodes,
                        "wall_duration_mean" : np.mean(wallDurations),
                        "wall_duration_std" : np.std(wallDurations),
                        "predict_wall_duration_mean" : np.mean(predictWallDurations),
                        "predict_wall_duration_std" : np.std(predictWallDurations)}
    return eval_results


def evaluatePolicyVec(vec_env : gym.vector.VectorEnv,
                   model : th.nn.Module,
                   episodes : int,
                   on_ep_done_callback : Callable[[float, int,int],Any] | None = None,
                   predict_func : Optional[Callable[[Any], Tuple[Any,Any]]] = None,
                   progress_bar : bool = False,
                   images_return = None,
                   obs_return = None,
                   extra_info_stats : list[str] = [],
                   deterministic : bool = False):
    with th.no_grad():
        is_training = model.training
        model.eval()
        if predict_func is None:
            predict_func_ = model.predict
        else:
            predict_func_ = predict_func
        buffsizes = episodes+vec_env.num_envs # may collect at most num_env excess episodes
        rewards = np.empty((buffsizes,), dtype = np.float32)
        durations_steps = np.empty((buffsizes,), dtype = np.int32)
        extra_stats = {k:np.empty((buffsizes,), dtype = np.float32) for k in extra_info_stats}
        successes = np.empty((buffsizes,), dtype = np.int32)
        collected_eps = 0
        collected_steps = 0
        #frames = []
        #do an average over a bunch of episodes
        if not progress_bar:
            maybe_tqdm = lambda x:x
        else:
            maybe_tqdm = tqdm.tqdm

        running_rews = [0] * vec_env.num_envs
        running_durations = [0] * vec_env.num_envs
        if obs_return is not None:
            running_obss = [[] for i in range(vec_env.num_envs)]
        t0 = time.monotonic()
        obss, infos = vec_env.reset()
        while collected_eps < episodes:
            acts, _states = predict_func_(obss, deterministic = deterministic)
            obss, rews, terms, truncs, infos = vec_env.step(acts)
            collected_steps += vec_env.num_envs
            for i in range(vec_env.num_envs):
                running_rews[i] += rews[i]
                running_durations[i] += 1
                if obs_return is not None:
                    running_obss[i].append(obss[i])
                if terms[i] or truncs[i]:
                    rewards[collected_eps] = running_rews[i]
                    durations_steps[collected_eps] = running_durations[i]
                    for k in extra_stats:
                        extra_stats[k][collected_eps] = infos[k][i]
                    if obs_return is not None:
                        obs_return.append(running_obss[i])
                    if on_ep_done_callback is not None:
                        on_ep_done_callback(episodeReward=running_rews[i], steps=running_durations[i], episode=collected_eps)
                    if "success" in infos.keys():
                        successes[collected_eps] = 1 if infos["success"][i] else 0
                    running_durations[i] = 0
                    running_rews[i] = 0
                    if obs_return is not None:
                        running_obss[i] = []
                    collected_eps += 1
        tf = time.monotonic()
        eval_results = {"reward_mean" : np.mean(rewards[:episodes]),
                        "reward_std" : np.std(rewards[:episodes]),
                        "steps_mean" : np.mean(durations_steps[:episodes]),
                        "steps_std" : np.std(durations_steps[:episodes]),
                        "success_ratio" : sum(successes[:episodes])/episodes,
                        "fps" : collected_steps/(tf-t0),
                        "collected_steps" : collected_steps,
                        "collected_episodes" : collected_eps}
        eval_results.update({f"{k}_mean":np.mean(v[:episodes]) for k,v in extra_stats.items()})
        eval_results.update({f"{k}_std":np.std(v[:episodes]) for k,v in extra_stats.items()})
        model.train(is_training)
    return eval_results

def fileGlobToList(fileGlobStr : str):
    """Convert a file path glob (i.e. a file path ending with *) to a list of files

    Parameters
    ----------
    fileGlobStr : str
        a string representing a path, possibly with an asterisk at the end

    Returns
    -------
    List
        A list of files
    """
    if fileGlobStr.endswith("*"):
        folderName = os.path.dirname(fileGlobStr)
        fileNamePrefix = os.path.basename(fileGlobStr)[:-1]
        files = []
        for f in os.listdir(folderName):
            if f.startswith(fileNamePrefix):
                files.append(f)
        files = sorted(files, key = lambda x: int(x.split("_")[-2]))
        fileList = [folderName+"/"+f for f in files]
        numEpisodes = 1
    else:
        fileList = [fileGlobStr]
    return fileList



def evaluateSavedModels(files : List[str], evaluator : Callable[[str],Dict[str,Union[float,int,str]]], maxProcs = int(multiprocessing.cpu_count()/2), args = []):
    # file paths should be in the format ".../<__file__>/<run_id>/checkpoints/<model.zip>"
    loaded_run_id = files[0].split("/")[-2]
    run_id = "eval_of_"+loaded_run_id+"_at_"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    folderName = os.getcwd()+"/"+os.path.basename(__file__)+"/eval/"+run_id
    os.makedirs(folderName)
    csvfilename = folderName+"/evaluation.csv"
    with open(csvfilename,"w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ",")
        neverWroteToCsv = True

        processes = maxProcs
        # print(f"Using {processes} parallel evaluators")
        argss = [[file,*args] for file in files]
        with multiprocessing.Pool(processes) as p:
            eval_results = p.map(evaluator, argss)

        for i in range(len(argss)):
            eval_results[i]["file"] = argss[i][0]

        if neverWroteToCsv:
            csvwriter.writerow(eval_results[0].keys())
            neverWroteToCsv = False
        for eval_results in eval_results:
            csvwriter.writerow(eval_results.values())
            csvfile.flush()



class RequestFailError(Exception):
    def __init__(self, message, partialResult):            
        super().__init__(message)
        self.partialResult = partialResult



def pkgutil_get_path(package, resource = None)  -> str:
    """ Modified version from pkgutil.get_data """

    spec = importlib.util.find_spec(package)
    if spec is None:
        raise FileNotFoundError(f"Could not spec for package for ({package}, {resource})")
    # loader = spec.loader
    # if loader is None or not hasattr(loader, 'get_data'):
    #     return None # If this happens, maybe __init__.py is missing?
    # XXX needs test
    mod = (sys.modules.get(package) or
           importlib._bootstrap._load(spec))
    if mod is None or not hasattr(mod, '__file__'):
        raise FileNotFoundError(f"Could not __file__ for package for ({package}, {resource})")
    
    if resource is None:
        return os.path.dirname(mod.__file__)

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.join(*parts)
    return resource_name.replace("//","/")

def exc_to_str(exception):
    # return '\n'.join(traceback.format_exception(etype=type(exception), value=exception, tb=exception.__traceback__))
    return '\n'.join(traceback.format_exception(exception, value=exception, tb=exception.__traceback__))












# # TODO: move these in adarl_ros

def ros1_image_to_numpy(rosMsg) -> np.ndarray:
    """Extracts an numpy/opencv image from a ros sensor_msgs image

    Parameters
    ----------
    rosMsg : sensor_msgs.msg.Image
        The ros image message

    Returns
    -------
    np.ndarray
        The numpy array contaning the image. Compatible with opencv

    Raises
    -------
    TypeError
        If the input image encoding is not supported

    """
    import sensor_msgs
    import sensor_msgs.msg

    if rosMsg.encoding not in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(rosMsg.encoding))

    dtype_class, channels = name_to_dtypes[rosMsg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder('>' if rosMsg.is_bigendian else '<')
    shape = (rosMsg.height, rosMsg.width, channels)

    data = np.frombuffer(rosMsg.data, dtype=dtype).reshape(shape)
    data.strides = (
        rosMsg.step,
        dtype.itemsize * channels,
        dtype.itemsize
    )

    if not np.isfinite(data).all():
        ggLog.warn(f"ros1_image_to_numpy(): nan detected in image")



    # opencv uses bgr instead of rgb
    # probably should be done also for other encodings
    if rosMsg.encoding == "rgb8":
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    if channels == 1:
        data = data[...,0]
    return data

def numpyImg_to_ros1(img : np.ndarray):
    """
    """
    import sensor_msgs.msg
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rosMsg = sensor_msgs.msg.Image()
    rosMsg.data = img.tobytes()
    rosMsg.step = img.strides[0]
    rosMsg.is_bigendian = (img.dtype.byteorder == '>')
    rosMsg.height = img.shape[0]
    rosMsg.width = img.shape[1]

    if img.shape[2] == 3:
        rosMsg.encoding = "rgb8"
    elif img.shape[2] == 1:
        rosMsg.encoding = "mono8"
    else:
        raise RuntimeError(f"unable to determine image type, shape = {img.shape}")
    return rosMsg


def buildRos1PoseStamped(position_xyz, orientation_xyzw, frame_id):
    import geometry_msgs.msg

    pose = geometry_msgs.msg.PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = position_xyz[0]
    pose.pose.position.y = position_xyz[1]
    pose.pose.position.z = position_xyz[2]
    pose.pose.orientation.x = orientation_xyzw[0]
    pose.pose.orientation.y = orientation_xyzw[1]
    pose.pose.orientation.z = orientation_xyzw[2]
    pose.pose.orientation.w = orientation_xyzw[3]
    return pose



class MoveFailError(Exception):
    def __init__(self, message):            
        super().__init__(message)

def string_find(string : str, keywords : list[str], reverse = False):
    for k in keywords:
        if reverse:
            pos = string.rfind(k)
        else:
            pos = string.find(k)
        if pos != -1:
            return pos
    return -1

def find_string_limits(text, pos):
    """ Assuming pos indicates a character in a string in text (string meaning a substringh delimited by quotes),
      this function fninds the position of the delimiters, i.e. the quotes"""
    start = string_find(text[:pos+1], ["\"","'"], reverse=True)
    end = string_find(text[start+1:], [text[start]])+start+1
    return start, end


def _fix_urdf_subst_find_paths(urdf_string : str):
    done = False
    pos = 0
    while not done:
        subst_start = urdf_string.find("$(", pos)
        # ggLog.info(f"Got match at {subst_start} : {urdf_string[subst_start:subst_start+20]}...")
        if subst_start != -1:
            subst_end = urdf_string.find(")",subst_start)
            subst_inner = urdf_string[subst_start+2:subst_end] # Get the part inside the parentheses 
            parts = [p for p in subst_inner.split(" ") if len(p)>0]
            # ggLog.info(f"Got parts {parts}")
            if parts[0] == "find":
                pkg_name = parts[1]
                # ggLog.info(f"Found $(find {pkg_name})")
                import rospkg
                try:
                    pkg_path = os.path.abspath(rospkg.RosPack().get_path(pkg_name)) # get ROS package
                except rospkg.common.ResourceNotFound as e:
                    pkg_path = pkgutil_get_path(pkg_name) # get generic python package
                full_subst = urdf_string[subst_start:subst_end+1]
                # ggLog.info(f"Replacing {full_subst} with {[pkg_path]}")
                urdf_string = urdf_string.replace(full_subst,pkg_path) # could be done more efficiently...
                pos = subst_start+len(pkg_path)
            else:
                pos = subst_start+1
        else:
            done = True
    return urdf_string


def _fix_urdf_package_paths(urdf_string : str):
    done = False
    pos = 0
    while not done:
        keyword = "package://"
        keyword_start = urdf_string.find(keyword, pos)
        # ggLog.info(f"Got match at {keyword_start} : {urdf_string[keyword_start:keyword_start+20]}...")
        if keyword_start != -1:
            path_start, path_end = find_string_limits(urdf_string, keyword_start)
            original_path = urdf_string[path_start+1:path_end]
            # ggLog.info(f"Resolving path in [{path_start},{path_end}]: '{original_path}'")
            split_path = original_path.split("/")
            pkg_name = split_path[2]
            import rospkg
            try:
                pkg_path = os.path.abspath(rospkg.RosPack().get_path(pkg_name))
            except rospkg.common.ResourceNotFound as e:
                pkg_path = pkgutil_get_path(pkg_name) # get egenric python package
            abs_path = pkg_path+"/"+"/".join(split_path[3:])
            # ggLog.info(f"pkg_path: {pkg_path}")
            urdf_string = urdf_string.replace(original_path,abs_path) # could be done more efficiently...
            pos = path_start+len(abs_path)
            # ggLog.info(f"Fixed to {urdf_string[path_start-5:path_start+len(abs_path)+5]}")
        else:
            done = True
    return urdf_string

def _fix_urdf_ros_paths(urdf_string):
    urdf_string = _fix_urdf_package_paths(urdf_string)
    urdf_string = _fix_urdf_subst_find_paths(urdf_string)
    return urdf_string

def compile_xacro_string(model_definition_string, model_kwargs = None):
    xacro_args = {"output":None, "just_deps":False, "xacro_ns":True, "verbosity":1}
    mappings = {}
    if model_kwargs is not None:
        mappings.update(model_kwargs) #mappings should be in the form {'from':'to'}
    mappings = {k:str(v) for k,v in mappings.items()}
    # ggLog.info(f"Xacro args = {xacro_args}")
    # ggLog.info(f"Input xacro: \n{model_definition_string}")
    model_definition_string = _fix_urdf_ros_paths(model_definition_string)
    doc = xacro.parse(model_definition_string)
    xacro.process_doc(doc, mappings = mappings, **xacro_args)
    model_definition_string = doc.toprettyxml(indent='  ', encoding="utf-8").decode('UTF-8')
    return model_definition_string

import adarl.adapters.BaseAdapter
def getBlocking(getterFunction : Callable, blocking_timeout_sec : float, env_controller : adarl.adapters.BaseAdapter.BaseAdapter, step_duration_sec : float = 0.1) -> Dict[Tuple[str,str],Any]:
    call_time = time.monotonic()
    last_warn_time = call_time
    while True:
        gottenStuff, missingStuff = getterFunction()
        if len(missingStuff)==0:
            return gottenStuff
        else:
            t = time.monotonic()
            if t-call_time >= blocking_timeout_sec:
                raise RequestFailError(message=f"Failed to get data {missingStuff}. Got {gottenStuff}",
                                    partialResult=gottenStuff)
            else:
                if t - last_warn_time > 0.1:
                    last_warn_time = t
                    ggLog.warn(f"Waiting for {missingStuff} since {t-call_time:.2f}s got {gottenStuff.keys()}")
                env_controller.run(step_duration_sec)




def isinstance_noimport(obj, class_names):
    if isinstance(class_names, str):
        class_names = [class_names]
    # return type(obj).__name__ in class_names
    for superclass in type(obj).__mro__:
        if superclass.__name__ in class_names:
            return True
    return False





def pyTorch_makeDeterministic(seed):
    """ Make pytorch as deterministic as possible.
        Still, DOES NOT ENSURE REPRODUCIBILTY ACROSS DIFFERENT TORCH/CUDA BUILDS AND
        HARDWARE ARCHITECTURES
    """
    import torch as th
    th.manual_seed(seed)
    np.random.seed(seed)
    # print(f"Seed set to {seed}")
    # time.sleep(10)
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)
    # Following may make things better, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def getBestGpu(seed ):
    import torch as th
    gpus_mem_info = []
    for i in range(th.cuda.device_count()):
        prevDev = th.cuda.current_device()
        th.cuda.set_device(th.device(type="cuda", index=i))
        gpus_mem_info.append(th.cuda.mem_get_info()) #Returns [free, total]
        th.cuda.set_device(prevDev)
        # print(f"Got {gpus_mem_info[-1]}")

    bestRatio = 0
    bestGpu = None
    ratios = [0.0]*len(gpus_mem_info)
    for i in range(len(gpus_mem_info)):
        tot = gpus_mem_info[i][1]
        free = gpus_mem_info[i][0]
        ratio = free/tot
        ratios[i] = ratio
        if ratio > bestRatio:
            bestRatio = ratio
            bestGpu = i

    # Look for the gpus that are within 10% of the best one
    candidates = []
    for i in range(len(gpus_mem_info)):
        if ratios[i] - bestRatio < 0.1:
            candidates.append(i)
    
    chosen_one = candidates[seed%len(candidates)]

    ggLog.info(f"Choosing GPU {chosen_one} with {ratios[chosen_one]*100}% free memory. Candidates were {[f'{i}:{ratios[i]*100}%' for i in candidates]}, seed was {seed}")
    return chosen_one


def torch_selectBestGpu(seed = 0):
    import torch as th
    bestGpu = getBestGpu(seed = seed)
    th.cuda.set_device(bestGpu)
    return th.device('cuda:'+str(bestGpu))


def obs_to_tensor(obs) -> Union[th.Tensor, Dict[Any, th.Tensor]]:
    if isinstance(obs, dict):
        return {k:obs_to_tensor(v) for k,v in obs.items()}
    else:
        return th.as_tensor(obs)
    
@th.jit.script
def vector_projection(v1,v2):
    """Project v1 onto the direction of v2
    """
    return th.dot(v1,v2/v2.norm())*v2/v2.norm()




def quaternionDistance(q1 : quaternion.quaternion, q2 : quaternion.quaternion ):
    """ Returns the minimum angle that separates two orientations.
    """
    # q1a = quaternion.as_float_array(q1)
    # q2a = quaternion.as_float_array(q2)
    #
    # return np.arccos(2*np.square(np.inner(q1a,q2a)) - 1)
    return quaternion.rotation_intrinsic_distance(q1,q2)

def buildQuaternion(x,y,z,w):
    return quaternion.quaternion(w,x,y,z)

def quaternion_xyzw_from_rotmat(rotmat : np.ndarray | th.Tensor):
    if isinstance(rotmat, th.Tensor):
        rotmat_np = rotmat.cpu().numpy()
    else:
        rotmat_np = rotmat
    quat_xyzw = quaternion.as_float_array(quaternion.from_rotation_matrix(rotmat_np))[...,[1,2,3,0]]
    if isinstance(rotmat, th.Tensor):
        return rotmat.new(quat_xyzw)
    else:
        return quat_xyzw

def ros_rpy_to_quaternion_xyzw(rpy):
    roll  = quaternion.from_rotation_vector([rpy[0], 0,      0])
    pitch = quaternion.from_rotation_vector([0,      rpy[1], 0])
    yaw   = quaternion.from_rotation_vector([0,      0,      rpy[2 ]])
    # On fixed axes:
    # First rotate around x (roll)
    # Then rotate around y (pitch)
    # Then rotate around z (yaw)
    q = yaw*pitch*roll
    return q.x, q.y, q.z, q.w



def quat_conjugate(quaternion_xyzw : np.ndarray | th.Tensor):
    if isinstance(quaternion_xyzw, th.Tensor):
        quaternion_xyzw = quaternion_xyzw.cpu()
    q = quaternion.from_float_array(quaternion_xyzw[...,[3,0,1,2]])
    q = q.conjugate()
    return quaternion.as_float_array(q)[...,[1,2,3,0]]

def quat_rotate_np(vector_xyz : np.ndarray | th.Tensor, quaternion_xyzw : np.ndarray | th.Tensor):
    if isinstance(quaternion_xyzw, th.Tensor):
        quaternion_xyzw = quaternion_xyzw.cpu().numpy()
    if isinstance(vector_xyz, th.Tensor):
        vector_xyz = vector_xyz.cpu().numpy()
    if len(quaternion_xyzw.shape)<2:
        quaternion_xyzw = np.expand_dims(quaternion_xyzw, axis=0)
        nonvec = True
    else:
        nonvec = False
    q = quaternion.from_float_array(quaternion_xyzw[:,[3,0,1,2]])
    qv = quaternion.from_vector_part(vector_xyz)
    r = quaternion.as_vector_part(q*qv*q.conjugate())
    if nonvec:
        return r[0]
    else:
        return r
    


def th_quat_combine(q_applied_first_xyzw : th.Tensor, q_applied_second_xyzw : th.Tensor):
    return quat_mul_xyzw(q_applied_second_xyzw,q_applied_first_xyzw)

def quat_mul_xyzw_np(q1_xyzw : np.ndarray, q2_xyzw : np.ndarray):
    quat_mul_xyzw(th.as_tensor(q1_xyzw),
                    th_quat_conj(th.as_tensor(q2_xyzw))).cpu().numpy()
@th.jit.script
def quat_mul_xyzw(q1_xyzw : th.Tensor, q2_xyzw : th.Tensor):
    r1 = q1_xyzw[...,3]
    v1 = q1_xyzw[...,0:3]
    r2 = q2_xyzw[...,3]
    v2 = q2_xyzw[...,0:3]
    q = th.empty_like(q1_xyzw)
    q[...,3] = r1*r2 - th.dot(v1,v2)
    q[...,0:3] = r1*v2 + r2*v1 + th.cross(v1,v2)
    return q

@th.jit.script
def th_quat_conj(q_xyzw : th.Tensor) -> th.Tensor:
    """Gives the inverse rotation of q, usually denoted q^-1 or q'. Note that q*q' = 1
    """
    return q_xyzw*th.tensor([-1.0,-1.0,-1.0,1.0], device=q_xyzw.device)

@th.jit.script
def th_quat_rotate(vector_xyz : th.Tensor, quaternion_xyzw : th.Tensor):
    vector_xyzw = th.cat([vector_xyz, th.zeros_like(vector_xyz[...,0].unsqueeze(-1))], dim=-1)
    return quat_mul_xyzw(quaternion_xyzw, quat_mul_xyzw(vector_xyzw, th_quat_conj(quaternion_xyzw)))[...,0:3]

@th.jit.script
def quat_swing_twist_decomposition_xyzw(quat_xyzw : th.Tensor, axis_xyz : th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Decomposes the quaternion into a rotation around
    the axis (twist), and a rotation perpendicular to
    the axis (swing).

    Parameters
    ----------
    quat_wxyz : th.Tensor
        Quaternion rotation
    axis_xyz : th.Tensor
        Axis

    Returns
    -------
    Tuple[th.Tensor, th.Tensor]
        swing, twist
    """
    quat_axis = quat_xyzw[0:3]
    twist_xyzw = th.empty(size=(4,), device=quat_axis.device)
    twist_xyzw[0:3] = vector_projection(quat_axis, axis_xyz)
    twist_xyzw[3] = quat_xyzw[3]
    twist_xyzw = twist_xyzw/twist_xyzw.norm()
    swing_xyzw = quat_mul_xyzw(quat_xyzw,th_quat_conj(twist_xyzw))
    return swing_xyzw, twist_xyzw

@th.jit.script
def quat_angle_xyzw(q_xyzw : th.Tensor) -> th.Tensor:
    """Angle of the angle-axis representation of the quaternion

    Parameters
    ----------
    q_wxyz : th.Tensor
        _description_

    Returns
    -------
    th.Tensor
        _description_
    """
    return 2*th.atan2(th.norm(q_xyzw[0:3]),q_xyzw[3])


def imgToCvIntRgb(img_chw_rgb : Union[th.Tensor, np.ndarray], min_val = -1, max_val = 1) -> np.ndarray:
    if isinstance(img_chw_rgb, np.ndarray):
        imgTorch = th.as_tensor(img_chw_rgb)
    else:
        imgTorch = img_chw_rgb
    if len(imgTorch.size())==2:
        imgTorch = imgTorch.unsqueeze(0)
    if imgTorch.size()[0] not in [1,3,4]:
        imgTorch = imgTorch.permute(2,0,1) # hwc to chw

    channels = imgTorch.size()[0]
    if channels == 1:
        imgTorch = imgTorch.repeat((3,1,1))
    elif channels == 3:
        imgTorch = imgTorch
    else:
        raise AttributeError(f"Unsupported image shape {imgTorch.size()}")
    
    if imgTorch.dtype in (th.float32, th.float64):
        imgTorch = (imgTorch + (-min_val))/(max_val-min_val) * 255
        imgTorch = imgTorch.to(dtype=th.uint8)
    elif imgTorch.dtype == th.uint8:
        pass
    else:
        raise AttributeError(f"Unsupported image dtype {imgTorch.dtype}")

    imgTorch = imgTorch[[2,1,0]] # rgb to bgr
    imgTorch = imgTorch.permute(1,2,0)
    imgCv = imgTorch.cpu().numpy()
    return imgCv

def cpuinfo():
    command = "cat /proc/cpuinfo"
    all_info = subprocess.check_output(command, shell=True).decode().strip()
    for line in all_info.split("\n"):
        if "model name" in line:
            return re.sub( ".*model name.*:", "", line,1)
    return None


def quintic_pos(t, duration, pos_range, offset):
    t = t/duration
    # nicely shaped quintic curve goes from 0 to 1 for x going from 0 to 1
    # zero first and second derivativeat 0 and 1
    # max derivative at 0.5
    # max second derivative at 0.5 +- (sqrt(3)/6)
    pos = pos_range*t*t*t*(6*t*t - 15*t +10) + offset
    return pos

def quintic_vel(t, duration, pos_range):
    b = 1/duration
    vel = 30*pos_range*b*b*b*t*t*(b*b*t*t-2*b*t+1)
    return vel

def quintic_acc(t, duration, pos_range):
    b = 1/duration
    vel = 30*pos_range*b*b*b*t*t*(b*b*t*t-2*b*t+1)
    return vel

def quintic_tpva(t, duration, pos_range, offset):
    s = (t,
         quintic_pos(t, duration, pos_range, offset),
         quintic_vel(t, duration, pos_range),
         quintic_acc(t, duration, pos_range))
    return s

def compute_quintic(p0 : float, pf : float, max_vel : float, max_acc : float):
    offset = p0
    pos_range = pf-p0
    duration_vel_lim = 15*pos_range/(max_vel*8)
    duration_acc_lim = np.sqrt(10*pos_range)/(np.power(3,0.25)*np.sqrt(max_acc))
    duration = max(duration_vel_lim, duration_acc_lim)
    return duration, pos_range, offset

def build_quintic_trajectory(p0 : float, v0 : float, pf : float, ctrl_freq_hz : float, max_vel : float, max_acc : float):
    # TODO: implement v0 usage, maybe somehow scaling a shifting the quintic
    duration, pos_range, offset = compute_quintic(p0 = p0, pf=pf, max_vel=max_vel, max_acc=max_acc)
    samples_num = int(duration*ctrl_freq_hz+1)
    traj_tpva = np.zeros(shape=(samples_num, 4), dtype=np.float32)
    for i in range(samples_num):
        t = i*1/ctrl_freq_hz
        traj_tpva[i] = quintic_tpva(t, duration, pos_range, offset)
    traj_tpva[-1] = duration, pf, 0, 0
    return traj_tpva


def build_1D_vramp_trajectory(t0 : float, p0 : float, v0 : float, pf : float, ctrl_freq_hz : float, max_vel : float, max_acc : float) -> np.ndarray:
    """Generate a 1-dimensional trajectory, using a quintic (6x^51-15x^4+10x^3) position trajectory
    and determining velocity and acceleration consequently. The trajectory will be scaled to respect 
    the max_vel and max_acc arguments.
    Usage of the v0 initial velocity is not implemented yet.

    Parameters
    ----------
    t0 : float
        Start time
    p0 : float
        Start position
    v0 : float
        Start velocity (currently unused)
    pf : float
        End position
    ctrl_freq_hz : float
        Determines how many samples to generate.
    max_vel : float
        Maximum velocity to plan for.
    max_acc : float
        Maximum acceleration to plan for.

    Returns
    -------
    np.ndarray
        Numpy array containing the trajectory samples in the form (time, position, velocity, acceleration)

    """
    # ggLog.info(f"build_1D_vramp_traj_samples("+ f"t0 = {t0}\n"
    #                                             f"p0 = {p0}\n"
    #                                             f"v0 = {v0}\n"
    #                                             f"pf = {pf}\n"
    #                                             f"ctrl_freq_hz = {ctrl_freq_hz}\n"
    #                                             f"max_vel = {max_vel}\n"
    #                                             f"max_acc = {max_acc}\n"
    #                                             ")")

    d = abs(pf-p0)
    if pf<p0:
        v0 = -v0 # direction was flipped, so flip the velocity
    trajectory_tpva = build_quintic_trajectory(0,v0,d,ctrl_freq_hz,max_vel, max_acc)
    if pf < p0:
        trajectory_tpva = [(t,-p,-v,-a) for t,p,v,a in trajectory_tpva]
    trajectory_tpva = [(t+t0,p+p0,v,a) for t,p,v,a in trajectory_tpva]
    trajectory_tpva = np.array(trajectory_tpva, dtype = np.float64)

    traj_max_vel = np.max(np.abs(trajectory_tpva[:,2]))
    if traj_max_vel > max_vel:
        raise RuntimeError(f"Error computing trajectory, max_vel exceeded. traj_max_vel = {traj_max_vel} > {max_vel}")    
    traj_max_acc = np.max(np.abs(trajectory_tpva[:,3]))
    if traj_max_acc > max_acc:
        raise RuntimeError(f"Error computing trajectory, max_acc exceeded. traj_max_acc = {traj_max_acc} > {max_acc}")

    return trajectory_tpva


def randn_like(t : th.Tensor, mu : th.Tensor, std : th.Tensor, generator  : th.Generator):
    return th.randn(size=t.size(),
                    generator=generator,
                    dtype=t.dtype,
                    device=t.device)*std + mu

def randn_from_mustd(mu_std : th.Tensor, generator  : th.Generator,
                     squash_sigma : float = -1.0,
                     size : Sequence[int] | None = None):
    if size is None:
        size = mu_std[0].size()
    noise =  th.randn(size=size,
                    generator=generator,
                    dtype=mu_std.dtype,
                    device=mu_std.device)
    if squash_sigma > 0:
        noise = th.tanh(noise/(squash_sigma))*squash_sigma
    return noise*mu_std[1] + mu_std[0]

def to_string_tensor(strings : list[str] | np.ndarray, max_string_len : int = 32):
    return th.as_tensor([list(n.encode("utf-8").ljust(max_string_len)[:max_string_len]) for n in strings], dtype=th.uint8) # ugly, but simple

_T = TypeVar('_T', float, th.Tensor)

def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1

printed_dbg_check_msg = False
def dbg_check(is_check_passed : Callable[[],bool|th.Tensor], build_msg : Callable[[],str]):
    from adarl.utils.session import default_session
    if default_session.debug_level>0:
        global printed_dbg_check_msg
        if not printed_dbg_check_msg:
            ggLog.warn(f"dbg_check is enabled")
            printed_dbg_check_msg = True
        if not is_check_passed():
            raise RuntimeError(build_msg())
    
def dbg_check_finite(tensor_tree, min = float("-inf"), max = float("+inf")):
    from adarl.utils.tensor_trees import is_all_finite, is_all_bounded, flatten_tensor_tree, map_tensor_tree, is_leaf_finite
    dbg_check(is_check_passed=lambda: is_all_finite(tensor_tree), 
              build_msg=lambda: f"Non-finite values in tensor tree: isfinite = {map_tensor_tree(flatten_tensor_tree(tensor_tree), is_leaf_finite)}")
    if min != float("-inf") or max != float("+inf"):
        dbg_check(is_check_passed=lambda: is_all_bounded(tensor_tree, min=th.as_tensor(min),max=th.as_tensor(max)), 
              build_msg=lambda: f"out of bounds values in tensor tree: {tensor_tree}")
    
def dbg_check_size(tensor : th.Tensor, size : Sequence[int], msg : str = ""):
    dbg_check(is_check_passed=lambda: tensor.size()==size, 
              build_msg=lambda: f"Unexpected tensor size: {tensor.size()} instead of {size}. "+msg)
        