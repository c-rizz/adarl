import typing
from nptyping import NDArray
import numpy as np
import time
import cv2
import collections
from typing import List, Tuple, Callable, Dict, Union, Optional, Any
import os
import quaternion
import signal
import datetime
import inspect
import shutil
import tqdm
from pathlib import Path
import random
import multiprocessing
import csv
import sys
import yaml
import importlib
import traceback

import lr_gym.utils.dbg.ggLog as ggLog
import traceback
import xacro
import faulthandler
import subprocess
import threading

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




class JointState:
    # These are lists because the joint may have multiple DOF
    # position = []
    # rate = []
    # effort = []

    def __init__(self, position : List[float], rate : List[float], effort : List[float]):
        self.position = position
        self.rate = rate
        self.effort = effort

    def __str__(self):
        return "JointState("+str(self.position)+","+str(self.rate)+","+str(self.effort)+")"

    def __repr__(self):
        return self.__str__()

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
        self._avg = 0
        self._all_time_sum = 0
        self._all_time_count = 0
        self._all_time_avg = 0

    def __enter__(self):
        self._t0 = time.monotonic()
    
    def __exit__(self, exc_type, exc_val, exc_t):
        self.addValue(time.monotonic()-self._t0)




def quaternionDistance(q1 : quaternion.quaternion ,q2 : quaternion.quaternion ):
    """ Returns the minimum angle that separates two orientations.

    Parameters
    ----------
    q1 : quaternion.quaternion
        Description of parameter `q1`.
    q2 : quaternion.quaternion
        Description of parameter `q2`.

    Returns
    -------
    type
        Description of returned object.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    # q1a = quaternion.as_float_array(q1)
    # q2a = quaternion.as_float_array(q2)
    #
    # return np.arccos(2*np.square(np.inner(q1a,q2a)) - 1)
    return quaternion.rotation_intrinsic_distance(q1,q2)

def buildQuaternion(x,y,z,w):
    return quaternion.quaternion(w,x,y,z)

class Pose:
    position : NDArray[(3,), np.float32]
    orientation : np.quaternion

    def __init__(self, x,y,z, qx,qy,qz,qw):
        self.position = np.array([x,y,z])
        self.orientation = buildQuaternion(x=qx,y=qy,z=qz,w=qw)

    def __str__(self):
        return f"[{self.position[0],self.position[1],self.position[2],self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w}]"

    def getPoseStamped(self, frame_id : str):
        return buildRos1PoseStamped(self.position, np.array([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w]), frame_id=frame_id)

    def getListXyzXyzw(self):
        return [self.position[0],self.position[1],self.position[2],self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w]
        
    def __repr__(self):
        return self.__str__()

class LinkState:
    pose : Pose = None
    pos_velocity_xyz : Tuple[float, float, float] = None
    ang_velocity_xyz : Tuple[float, float, float] = None

    def __init__(self, position_xyz : Tuple[float, float, float], orientation_xyzw : Tuple[float, float, float, float],
                    pos_velocity_xyz : Tuple[float, float, float], ang_velocity_xyz : Tuple[float, float, float]):
        self.pose = Pose(position_xyz[0],position_xyz[1],position_xyz[2], orientation_xyzw[0],orientation_xyzw[1],orientation_xyzw[2],orientation_xyzw[3])
        self.pos_velocity_xyz = pos_velocity_xyz
        self.ang_velocity_xyz = ang_velocity_xyz

    def __str__(self):
        return "LinkState("+str(self.pose)+","+str(self.pos_velocity_xyz)+","+str(self.ang_velocity_xyz)+")"

    def __repr__(self):
        return self.__str__()



_is_shutting_down = False

def is_shutting_down():
    return _is_shutting_down

def shutdown():
    print(f"Shutting down")
    global _is_shutting_down
    _is_shutting_down = True




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


wandb_enabled = False

def is_wandb_enabled():
    return wandb_enabled

def setupLoggingForRun(file : str, currentframe = None, folderName : Optional[str] = None, use_wandb : bool = True, experiment_name : Optional[str] = None, run_id : Optional[str] = None, comment = ""):
    """Sets up a logging output folder for a training run.
        It creates the folder, saves the current main script file for reference

    Parameters
    ----------
    file : str
        Path of the main script, the file wil be copied in the log folder
    frame : [type]
        Current frame from the main method, use inspect.currentframe() to get it. It will be used to save the
        call parameters.

    Returns
    -------
    str
        The logging folder to be used
    """
    if folderName is None:
        if run_id is None:
            run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        script_out_folder = os.getcwd()+"/"+os.path.basename(file)
        folderName = script_out_folder+"/"+run_id
        os.makedirs(folderName, exist_ok=True)
    else:
        os.makedirs(folderName, exist_ok=True)
        script_out_folder = str(Path(folderName).parent.absolute())

    createSymlink(src = folderName, dst = script_out_folder+"/latest")
    shutil.copyfile(file, folderName+"/main_script")
    if currentframe is not None:
        args, _, _, values = inspect.getargvalues(currentframe)
    else:
        args, values = ([],{})
    # inputargs = [(i, values[i]) for i in args]
    # with open(folderName+"/input_args.txt", "w") as input_args_file:
    #     print(str(inputargs), file=input_args_file)
    args_yaml_file = folderName+"/input_args.yaml"
    with open(args_yaml_file, "w") as input_args_yamlfile:
        yaml.dump(values,input_args_yamlfile, default_flow_style=None)

    ggLog.info(f"values = {values}")

    if "modelFile" in values:
        if values["modelFile"] is not None and type(values["modelFile"]) == str:
            model_dir = os.path.dirname(values["modelFile"])
            if os.path.isdir(model_dir):
                parent = Path(model_dir).parent.absolute()
                diff = subprocess.run(['diff', args_yaml_file, f"{parent}/input_args.yaml"], stdout=subprocess.PIPE).stdout.decode('utf-8')
                ggLog.info(f"Args comparison with loaded model:\n{diff}")
            else:
                ggLog.info("modelFile is not a file")

    if use_wandb:
        global wandb_enabled
        import wandb
        if experiment_name is None:
            experiment_name = os.path.basename(file)
        try:
            wandb.init( project=experiment_name,
                        config = values,
                        name = run_id,
                        monitor_gym = False, # Do not save openai gym videos
                        save_code = True, # Save run code
                        sync_tensorboard = True, # Save tensorboard stuff,
                        notes = comment
                        )
            wandb_enabled = True
        except wandb.sdk.wandb_manager.ManagerConnectionError as e:
            ggLog.error(f"Wandb connection failed: {exc_to_str(e)}")

    return folderName


from lr_gym.utils.sigint_handler import setupSigintHandler
    

def lr_gym_shutdown():
    shutdown()
    if is_wandb_enabled():
        import wandb
        wandb.finish()
    t0 = time.monotonic()
    timeout = 60
    if threading.current_thread() == threading.main_thread():
        active_threads = threading.enumerate()

        while len(active_threads)>1 and time.monotonic() - t0 < timeout:
            ggLog.warn(f"lr_gym shutting down: waiting for active threads {active_threads}")
            time.sleep(10)

        # for t in threading.enumerate():
        #     if t != threading.main_thread():
        #         terminate the thread???

        # if len(active_threads)>1: # If we just exit() the process will wait for subthreads and not terminate
        #     ggLog.error(f"Trying to shutdown but there are still threads running after {timeout} seconds. Self-terminating")
        #     ggLog.error("Sending SIGTERM to myself")
        #     os.kill(os.getpid(), signal.SIGTERM)
        #     time.sleep(60)
        #     ggLog.error("Still alive, sending SIGKILL to myself")
        #     os.kill(os.getpid(), signal.SIGKILL)
        #     time.sleep(10)
        #     ggLog.error("Still alive after SIGKILL!")




def evaluatePolicy(env, model, episodes : int, on_ep_done_callback = None, predict_func : Callable[[Any], Tuple[Any,Any]] = None, progress_bar : bool = True, images_return = None, obs_return = None):
    if predict_func is None:
        predict_func = model.predict
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
        done = False
        predDurations = []
        t0 = time.monotonic()
        # ggLog.info("Env resetting...")
        obs = env.reset()
        # ggLog.info("Env resetted")
        if images_return is not None:
            images_return.append([])
        if obs_return is not None:
            obs_return.append([])
        while not done:
            t0_pred = time.monotonic()
            # ggLog.info("Predicting")
            if images_return is not None:
                images_return[-1].append(env.render())
            if obs_return is not None:
                obs_return[-1].append(obs)
            action, _states = predict_func(obs)
            predDurations.append(time.monotonic()-t0_pred)
            # ggLog.info("Stepping")
            obs, stepReward, done, info = env.step(action)
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
        print(f"Using {processes} parallel evaluators")
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



def pkgutil_get_path(package, resource = None):
    """ Modified version from pkgutil.get_data """

    spec = importlib.util.find_spec(package)
    if spec is None:
        return None
    loader = spec.loader
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    # XXX needs test
    mod = (sys.modules.get(package) or
           importlib._bootstrap._load(spec))
    if mod is None or not hasattr(mod, '__file__'):
        return None
    
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












# # TODO: move these in lr_gym_ros

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


def compile_xacro_string(model_definition_string, model_kwargs = None):
    xacro_args = {"output":None, "just_deps":False, "xacro_ns":True, "verbosity":1}
    mappings = {}
    if model_kwargs is not None:
        mappings.update(model_kwargs) #mappings should be in the form {'from':'to'}
    mappings = {k:str(v) for k,v in mappings.items()}
    # ggLog.info(f"Xacro args = {xacro_args}")
    # ggLog.info(f"Input xacro: \n{model_definition_string}")
    doc = xacro.parse(model_definition_string)
    xacro.process_doc(doc, mappings = mappings, **xacro_args)
    model_definition_string = doc.toprettyxml(indent='  ', encoding="utf-8").decode('UTF-8')
    return model_definition_string

def getBlocking(getterFunction : Callable, blocking_timeout_sec : float, env_controller : EnvironmentError, step_duration_sec : float = 0.1) -> Dict[Tuple[str,str],Any]:
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
                env_controller.freerun(step_duration_sec)


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


def isinstance_noimport(obj, class_names):
    if isinstance(class_names, str):
        class_names = [class_names]
    return type(obj).__name__ in class_names





def lr_gym_startup( main_file_path : str,
                    currentframe = None,
                    using_pytorch : bool = True,
                    folderName : Optional[str] = None,
                    seed = None,
                    experiment_name : Optional[str] = None,
                    run_id : Optional[str] = None,
                    debug = False,
                    run_comment = "",
                    use_wandb = True) -> str:
    if isinstance(debug, bool):
        if debug:
            debug_level = 1
        else:
            debug_level = 0
    else:
        debug_level = debug
    faulthandler.enable() # enable handlers for SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL
    logFolder = setupLoggingForRun(main_file_path, currentframe, folderName=folderName, experiment_name=experiment_name, run_id=run_id, comment=run_comment, use_wandb=use_wandb)
    ggLog.addLogFile(logFolder+"/gglog.log")
    if seed is None:
        raise AttributeError("You must specify the run seed")
    ggLog.setId(str(seed))
    np.set_printoptions(edgeitems=10,linewidth=180)
    setupSigintHandler()
    if using_pytorch:
        import torch as th
        pyTorch_makeDeterministic(seed)
        th.autograd.set_detect_anomaly(debug_level >= 0) # Detect NaNs
        th.distributions.Distribution.set_default_validate_args(debug_level >= 1) # do not check distribution args validity (it leads to cuda syncs)
        if th.cuda.is_available():
            ggLog.info(f"CUDA AVAILABLE: device = {th.cuda.get_device_name()}")
        else:
            ggLog.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"+
                        "                  NO CUDA AVAILABLE!\n"+
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"+
                        "Will continue in 10 sec...")
            time.sleep(10)
    return logFolder

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
    ratios = [None]*len(gpus_mem_info)
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
