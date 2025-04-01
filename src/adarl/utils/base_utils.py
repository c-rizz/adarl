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
import pathlib

import adarl.utils.dbg.ggLog as ggLog
import traceback
import xacro
import inspect

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



import subprocess
import re
from typing import TypedDict, Mapping

class AverageKeeper:
    def __init__(self, bufferSize = 100):
        self._bufferSize = bufferSize
        self.reset()

    def addValue(self, newValue):
        self._buffer.append(newValue)
        self._last_added = newValue
        self._all_time_sum += newValue
        self._all_time_count += 1
        self._avg = float(sum(self._buffer))/len(self._buffer)
        self._all_time_avg = self._all_time_sum/self._all_time_count

    def getAverage(self, all_time : bool = False):
        if all_time:
            return self._all_time_avg
        else:
            return self._avg
        
    def getLast(self):
        return self._last_added

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
    if hasattr(spec,"submodule_search_locations"):
        if len(spec.submodule_search_locations)>0:
            submodule_paths = spec.submodule_search_locations
    else:
        # loader = spec.loader
        # if loader is None or not hasattr(loader, 'get_data'):
        #     return None # If this happens, maybe __init__.py is missing?
        # XXX needs test
        mod = (sys.modules.get(package) or importlib._bootstrap._load(spec))
        if mod is None or not hasattr(mod, '__file__'):
            raise FileNotFoundError(f"Could not __file__ for package for ({package}, {resource})")        
        submodule_paths = [os.path.dirname(mod.__file__)]
    if resource is None:
        if len(spec.submodule_search_locations)>1:
                ggLog.warn(f"package '{package}' has multiple submodule paths ({submodule_paths}). Will just use the first.")
        return submodule_paths[0]

    for submodule_path in submodule_paths:
        parts = resource.split('/')
        parts.insert(0, submodule_path)
        resource_name = os.path.join(*parts)
        resource_name = resource_name.replace("//","/")
        if pathlib.Path(resource_name).exists():
            return resource_name
    raise FileNotFoundError(f"resource {resource} not found in package {package}, submodule_paths = {submodule_paths}")

def exc_to_str(exception):
    # return '\n'.join(traceback.format_exception(etype=type(exception), value=exception, tb=exception.__traceback__))
    return '\n'.join(traceback.format_exception(exception, value=exception, tb=exception.__traceback__))

def get_caller_info(depth : int = 1, width : int = 1, inline = True):
    frame = inspect.currentframe()
    r = []
    for i in range(width):
        for i in range(depth+1+i):
            frame = frame.f_back  
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        r.append(f"{filename, lineno}")
    if inline:
        return ",".join(r)
    else:
        return "\n".join(r)










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


def _find_pkg(pkg_name, extra_pkg_paths : dict[str,str] = {}):
    if pkg_name in extra_pkg_paths:
        return extra_pkg_paths[pkg_name]
    try:
        import rospkg
        try:
            pkg_path = os.path.abspath(rospkg.RosPack().get_path(pkg_name))
        except rospkg.common.ResourceNotFound as e:
            pkg_path = pkgutil_get_path(pkg_name) # get generic python package
    except ImportError as e:
        ggLog.warn(f"Could not import rospkg")
        pkg_path = pkgutil_get_path(pkg_name) # get generic python package
    return pkg_path


def _fix_urdf_subst_find_paths(urdf_string : str, extra_pkg_paths : dict[str,str] = {}):
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
                # ggLog.info(f"Found $(find {pkg_name})")
                pkg_path = _find_pkg(pkg_name=parts[1], extra_pkg_paths=extra_pkg_paths)
                full_subst = urdf_string[subst_start:subst_end+1]
                # ggLog.info(f"Replacing {full_subst} with {[pkg_path]}")
                urdf_string = urdf_string.replace(full_subst,pkg_path) # could be done more efficiently...
                pos = subst_start+len(pkg_path)
            else:
                ggLog.warn(f"Skipping xacro subst: {urdf_string[subst_start:subst_end+1]}")
                pos = subst_start+1
        else:
            done = True
    return urdf_string


def _fix_urdf_package_paths(urdf_string : str, extra_pkg_paths : dict[str,str] = {}):
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
            pkg_path = _find_pkg(pkg_name=split_path[2], extra_pkg_paths=extra_pkg_paths)
            abs_path = pkg_path+"/"+"/".join(split_path[3:])
            # ggLog.info(f"pkg_path: {pkg_path}")
            urdf_string = urdf_string.replace(original_path,abs_path) # could be done more efficiently...
            pos = path_start+len(abs_path)
            # ggLog.info(f"Fixed to {urdf_string[path_start-5:path_start+len(abs_path)+5]}")
        else:
            done = True
    return urdf_string

def _fix_urdf_ros_paths(urdf_string, extra_pkg_paths : dict[str,str] = {}):
    urdf_string = _fix_urdf_package_paths(urdf_string, extra_pkg_paths)
    # urdf_string = _fix_urdf_subst_find_paths(urdf_string, extra_pkg_paths)
    # ggLog.info(f"fixed xacro = {urdf_string}")
    return urdf_string

def compile_xacro_string(model_definition_string, model_kwargs = None, extra_pkg_paths : dict[str,str] = {}):
    xacro_args = {"output":None, "just_deps":False, "xacro_ns":True, "verbosity":1}
    mappings = {}
    if model_kwargs is not None:
        mappings.update(model_kwargs) #mappings should be in the form {'from':'to'}
    mappings = {k:str(v) for k,v in mappings.items()}
    # ggLog.info(f"Xacro args = {xacro_args}")
    # ggLog.info(f"Input xacro: \n{model_definition_string}")
    doc = xacro.parse(model_definition_string)
    xacro.process_doc(doc, mappings = mappings, extra_find_pkgs=extra_pkg_paths, **xacro_args)
    model_definition_string = doc.toprettyxml(indent='  ', encoding="utf-8").decode('UTF-8')
    model_definition_string = _fix_urdf_ros_paths(model_definition_string, extra_pkg_paths=extra_pkg_paths)
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




































































































































