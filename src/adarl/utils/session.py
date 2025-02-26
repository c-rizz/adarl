from typing import Optional, List, Union, Tuple
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
from adarl.utils.utils import pyTorch_makeDeterministic, createSymlink, exc_to_str

import datetime
import threading
import time
import os
import shutil
import inspect
from pathlib import Path
import yaml
import subprocess
import faulthandler
import adarl.utils.utils
import multiprocessing
import multiprocessing.pool
import random
import atexit
from adarl.utils.wandb_wrapper import wandb_init
import adarl.utils.mp_helper as mp_helper
import adarl.utils.wandb_wrapper as wandb_wrapper
import signal
faulthandler.enable() # enable handlers for SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL
import dataclasses 
import socket
import cpuinfo
import warnings
import traceback

original_showwarning = None
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    original_showwarning(message, category, filename, lineno, file=file, line=line)
    traceback.print_stack()  # Print full Python stack trace

def override_warning_func():
    global original_showwarning
    original_showwarning = warnings.showwarning
    warnings.showwarning = custom_showwarning

class Session():
    def __init__(self):

        self._is_shutting_down = False
        self._is_wandb_enabled = False
        self._wandb_wrapper = wandb_wrapper.default_wrapper
        self._id = f"{int(time.monotonic()*1000000)}_{int(random.random()*1000000000)}"
        self._initialized = False
        # ggLog.info(f"Created session {self._id}")

    def reapply_globals(self):
        wandb_wrapper.default_wrapper = self._wandb_wrapper

    def initialize(self,main_file_path : str,
                        currentframe = None,
                        using_pytorch : bool = True,
                        folderName : Optional[str] = None,
                        seed = None,
                        experiment_name : Optional[str] = None,
                        run_id : Optional[str] = None,
                        debug : Union[bool, int]  = False,
                        run_comment = "",
                        use_wandb = True):
        
        self._initialized = True
        self._is_wandb_enabled = use_wandb
        if isinstance(debug, bool):
            if debug:
                debug_level = 1
            else:
                debug_level = 0
        else:
            debug_level = debug
        self.debug_level = debug_level
        # self._manager = multiprocessing.Manager()
        # self.run_info = self._manager.dict()
        # ggLog.info(f"Initializing session {self} in process {os.getpid()}")
        self.run_info = {}
        self.run_info["comment"] = run_comment
        self.run_info["experiment_name"] = experiment_name
        self.run_info["run_id"] = run_id
        self.run_info["start_time_monotonic"] = time.monotonic()
        self.run_info["start_time"] = time.time()
        self.run_info["collected_episodes"] = mp_helper.get_context().Value("i",0)
        self.run_info["collected_steps"] = mp_helper.get_context().Value("i",0)
        self.run_info["train_iterations"] = mp_helper.get_context().Value("i",0)
        self.run_info["seed"] = seed
        self.run_info["hostname"] = socket.gethostname()
        self.run_info["cpu"] = cpuinfo.get_cpu_info()["brand_raw"]
        self.run_info["gpu"] = ""
        self._logFolder = self._setupLoggingForRun(main_file_path,
                                                   currentframe,
                                                   folderName=folderName,
                                                   experiment_name=experiment_name,
                                                   run_id=run_id,
                                                   comment=run_comment,
                                                   use_wandb=use_wandb)
        ggLog.addLogFile(self._logFolder+"/gglog.log")
        if seed is None:
            raise AttributeError("You must specify the run seed")
        ggLog.setId(str(seed))
        np.set_printoptions(edgeitems=10,linewidth=180)
        from adarl.utils.sigint_handler import setupSigintHandler
        setupSigintHandler()
        if using_pytorch:
            import torch as th
            self.run_info["gpu"] = th.cuda.get_device_name()
            th.set_printoptions(linewidth=160)
            pyTorch_makeDeterministic(seed)
            if debug_level>0:
                if debug_level>1:
                    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
                if debug_level>2:
                    warnings.simplefilter("always")
                override_warning_func()
                th.cuda.set_sync_debug_mode("warn")
            th.autograd.set_detect_anomaly(debug_level >= 2) # type: ignore
            th.distributions.Distribution.set_default_validate_args(debug_level >= 2) # do not check distribution args validity (it leads to cuda syncs)
            if th.cuda.is_available():
                ggLog.info(f"CUDA AVAILABLE: device = {th.cuda.get_device_name()}")
            else:
                ggLog.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"+
                            "                  NO CUDA AVAILABLE!\n"+
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"+
                            "Will continue in 10 sec...")
                time.sleep(10)


    def _setupLoggingForRun(self,   file : str,
                                    currentframe = None,
                                    folderName : Optional[str] = None,
                                    use_wandb : bool = True,
                                    experiment_name : Optional[str] = None,
                                    run_id : Optional[str] = None,
                                    comment = ""):
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
            script_out_folder = os.getcwd()+"/lrg_exps/"+os.path.basename(file)
            folderName = script_out_folder+"/"+run_id
            os.makedirs(folderName, exist_ok=True)
        else:
            os.makedirs(folderName, exist_ok=True)
            script_out_folder = str(Path(folderName).parent.absolute())

        createSymlink(src = folderName, dst = script_out_folder+"/latest")
        shutil.copyfile(file, folderName+"/main_script.py")
        if currentframe is not None:
            args, _, _, config = inspect.getargvalues(currentframe)
        else:
            args, config = ([],{})

        has_torch = False
        cuda_available = False
        cuda_device_name = None
        try:
            import torch as th
            cuda_available = th.cuda.is_available() 
            if cuda_available:
                cuda_device_name = th.cuda.get_device_name()
        except ImportError as e:
            pass
        config["has_torch"] = has_torch
        config["cuda_available"] = cuda_available
        config["cuda_device_name"] = cuda_device_name
        config["cpu_name"] = adarl.utils.utils.cpuinfo()
        

        # inputargs = [(i, values[i]) for i in args]
        # with open(folderName+"/input_args.txt", "w") as input_args_file:
        #     print(str(inputargs), file=input_args_file)
        args_yaml_file = folderName+"/input_args.yaml"
        with open(args_yaml_file, "w") as input_args_yamlfile:
            yaml.dump(config,input_args_yamlfile, default_flow_style=None)
        # ggLog.info(f"values = {values}")

        if "modelFile" in config:
            if config["modelFile"] is not None and type(config["modelFile"]) == str:
                model_dir = os.path.dirname(config["modelFile"])
                if os.path.isdir(model_dir):
                    parent = Path(model_dir).parent.absolute()
                    diff = subprocess.run(['diff', args_yaml_file, f"{parent}/input_args.yaml"], stdout=subprocess.PIPE).stdout.decode('utf-8')
                    ggLog.info(f"Args comparison with loaded model:\n{diff}")
                else:
                    ggLog.info("modelFile is not a file")

        if use_wandb:
            import wandb
            if experiment_name is None:
                experiment_name = os.path.basename(file)
            try:
                ggLog.info(f"Starting run with experiment name '{experiment_name}', run id {run_id}")
                config_s = "\n".join([str(t) for t in config.items()])
                ggLog.info(f"config = {config_s}")
                config = {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v for k,v in config.items()} # dataclasses have some issue with json serialization
                wandb_init( project=experiment_name,
                            config = config,
                            name = f"{run_id}_{comment.strip().replace(' ','_')}",
                            monitor_gym = False, # Do not save openai gym videos
                            save_code = True, # Save run code
                            sync_tensorboard = True, # Save tensorboard stuff,
                            notes = comment
                            )
            except wandb.sdk.wandb_manager.ManagerConnectionError as e: # type: ignore
                ggLog.error(f"Wandb connection failed: {exc_to_str(e)}")

        return folderName

    def log_folder(self) -> str:
        return self._logFolder

    def is_shutting_down(self):
        return self._is_shutting_down
    
    def mark_shutting_down(self):
        self._is_shutting_down = True

    def shutdown(self):
        self.mark_shutting_down()

        if self.is_wandb_enabled():
            import wandb
            wandb.finish()
            ggLog.info(f"Told wandb to finish.")
            
        t0 = time.monotonic()
        timeout = 30
        if threading.current_thread() == threading.main_thread():
            active_threads = [t for t in threading.enumerate() if t.name!="MainThread" and not t.isDaemon()]
            elapsed = 0
            while len(active_threads)>0 and elapsed < timeout:
                elapsed = time.monotonic() - t0
                ggLog.warn(f"Session shutting down: [{int(timeout - elapsed)}] waiting for active threads {active_threads}")
                time.sleep(1)
                active_threads = [t for t in threading.enumerate() if t.name!="MainThread"]
                all_daemonic = True
                for t in active_threads: all_daemonic = all_daemonic and t.isDaemon()
                if all_daemonic:
                    break
        if len(active_threads)>1:
            ggLog.warn(f"Session shutting down: still have active threads {active_threads}")

        all_children_terminated = False
        t0_chterm = time.monotonic()
        while not all_children_terminated and time.monotonic() < t0_chterm+timeout:
            child_procs : list[multiprocessing.Process] = mp_helper.get_context().active_children()
            child_procs = [p for p in child_procs if not p.daemon and p.is_alive()] # exclude daemonic process
            all_children_terminated = len(child_procs)==0
            if not all_children_terminated:
                sig = signal.SIGINT if time.monotonic() < timeout + t0_chterm - 10 else signal.SIGKILL
                ggLog.warn(f"Session is shutting down, but still have {len(child_procs)} child processes. Sending signal {sig} to all")
                ggLog.warn(f"Procs are: {child_procs}")
                for p in child_procs:
                    os.kill(p.pid, sig)
                for p in child_procs:
                    p.join(timeout = min(5,max(0,t0_chterm+timeout-time.monotonic())))
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

    def is_wandb_enabled(self):
        return self._is_wandb_enabled
    
    # def __del__(self):
    #     print(f"Session {self._id} destroyed")

def set_current_session(session : Session):
    global default_session
    # ggLog.info(f"Overwriting session {default_session._id} with {session._id} in process {os.getpid()}")
    default_session = session
    default_session.reapply_globals()

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp_helper.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)







 # Initialize with a minimal setup. Processes that do not initialize properly will use this.
default_session : Session = Session()
atexit.register(lambda: default_session.mark_shutting_down())

def adarl_shutdown(session : Session):
    session.shutdown()



def adarl_startup( main_file_path : str,
                    currentframe = None,
                    using_pytorch : bool = True,
                    folderName : Optional[str] = None,
                    seed = None,
                    experiment_name : Optional[str] = None,
                    run_id : Optional[str] = None,
                    debug : Union[bool, int]  = False,
                    run_comment = "",
                    use_wandb = True) -> Tuple[str, Session]:
    global default_session
    # default_session = Session()
    default_session.initialize( main_file_path = main_file_path,
                                currentframe = currentframe,
                                using_pytorch = using_pytorch,
                                folderName = folderName,
                                seed = seed,
                                experiment_name = experiment_name,
                                run_id = run_id,
                                debug = debug,
                                run_comment = run_comment,
                                use_wandb = use_wandb)
    return default_session.log_folder(), default_session


















def runFunction_wrapper(seed,
                        folderName,
                        runFunction,
                        resumeModelFile,
                        run_id,
                        run_args,
                        start_adarl,
                        launch_file_path,
                        debug_level):
    try:
        seedFolder = folderName+f"/seed_{seed}"
        experiment_name = os.path.basename(launch_file_path)
        if start_adarl:
            folderName, session = adarl_startup(launch_file_path,
                                        inspect.currentframe(),
                                        folderName = seedFolder,
                                        seed = seed,
                                        experiment_name = experiment_name,
                                        run_id = run_id,
                                        debug = debug_level,
                                        run_comment=run_args["comment"])

        ggLog.info(f"Starting run with seed {seed}:\n"
                   f"Out folder = {seedFolder}\n"
                   f"Run id = {run_id}")
        os.makedirs(folderName,exist_ok=True)
        adarl.utils.utils.createSymlink(src = folderName, dst = str(Path(folderName).parent.absolute())+"/latest")
        # time.sleep(seed)
        # if resumeModelFile is not None:
        #     os.makedirs(seedFolder,exist_ok=True)            
        return runFunction(seed=seed, folderName=seedFolder, resumeModelFile=resumeModelFile, run_id=run_id, args = run_args)
    except Exception as e:
        ggLog.error(f"Run failed with exception: {adarl.utils.utils.exc_to_str(e)}")
        return None

def runFunction_wrapper_arg_kwargs(args, kwargs):
    return runFunction_wrapper(*args,**kwargs)

def detectFolderArgs(resumeFolder):
    ret = {}

    for seedDir in os.listdir(resumeFolder):
        if seedDir.startswith("seed_") and os.path.isdir(resumeFolder+"/"+seedDir) and not os.path.islink(resumeFolder+"/"+seedDir):
            seed = int(seedDir.split("_")[-1])
            checkpointsDir = resumeFolder+"/"+seedDir+"/checkpoints"
            checkpoints = os.listdir(checkpointsDir)
            replay_buffers = [file for file in checkpoints if file.startswith("model_checkpoint_replay_buffer_")]
            model_to_reload = None
            if len(replay_buffers)>2:
                raise RuntimeError("Could not select replay buffer, there should be at most 2")
            elif len(replay_buffers)==0:
                # raise RuntimeError("No Replay Buffer found")
                model_checkpoint_files = [file for file in checkpoints if file.startswith("model_checkpoint_")]
                newest_checkpoint_steps = max([int(file.split("_")[2]) for file in model_checkpoint_files])
                model_to_reload = checkpointsDir+"/model_checkpoint_"+str(newest_checkpoint_steps)+"_steps"
            else:
                if len(replay_buffers)==2:
                    rp_steps = [int(rp.split("_")[4]) for rp in replay_buffers]
                    if rp_steps[0] > rp_steps[1]: # Choose the older one, if there's two then the second one is potentially half-saved
                        replay_buffer = replay_buffers[1]
                    else:
                        replay_buffer = replay_buffers[0]
                elif len(replay_buffers)==1:
                    replay_buffer = replay_buffers[0]
                else:
                    raise RuntimeError()
                rp_buff_ep = replay_buffer.split("_")[4]
                rp_buff_step = replay_buffer.split("_")[5]
                for file in checkpoints:
                    if file.startswith("model_checkpoint_") and file.split("_")[2] == rp_buff_ep and file.split("_")[3] == rp_buff_step:
                        model_to_reload = checkpointsDir+"/"+file[:-4] # remove extension
                # model_to_reload = checkpointsDir+f"/model_checkpoint_{rp_buff_ep}_{???}_{rp_buff_step}_steps"

            ret[seed] = {"modelToLoad" : model_to_reload}
    
    return ret

# def rebuild_run_folder(newFolderName, seed, resumeFolder):
#     seedFolder = newFolderName+f"/seed_{seed}"
#     os.makedirs(seedFolder,exist_ok=True)
#     os.makedirs(seedFolder+"/feature_extractor",exist_ok=True)
#     copyfile(src=resumeFolder+f"/seed_{seed}/GymEnvWrapper_log.csv", dst=seedFolder+"/GymEnvWrapper_log.csv")
#     copyfile(src=resumeFolder+f"/seed_{seed}/feature_extractor/simple_feature_extractor_log.csv", dst=seedFolder+"/feature_extractor/simple_feature_extractor_log.csv")

# def evaluator(args): 
#     file = args[0]
#     evalFunction = args[1]
#     folderName = args[2]
#     steps = int(re.sub("[^0-9]", "", os.path.basename(file)))
#     return evalFunction(fileToLoad=file, folderName=folderName+"/runs/"+str(steps))
    
def launchRun(runFunction,
            seedsNum,
            seedsOffset, 
            maxProcs,
            launchFilePath,
            seeds : Optional[List[int]] = None,
            resumeFolder : Optional[str] = None,
            pretrainedModelFile : Optional[str] = None,
            args = {},
            pkgs_to_save = ["adarl"],
            start_adarl : bool = True,
            debug_level = 0):
    experiment_name = os.path.basename(launchFilePath)
    script_out_folder = os.getcwd()+"/lrg_exps/"+experiment_name
    done = False
    tries = 0
    folderName = ""
    launch_id = ""
    while not done:
        try:
            launch_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            folderName = script_out_folder+"/"+launch_id
            os.makedirs(folderName)
            done = True
        except FileExistsError as e:
            # This may happen if two runs start at the same time
            # Use a random backoff time and retry
            time.sleep(random.random()*10)
            tries += 1
            if tries > 10:
                raise e
    for pkg in pkgs_to_save:
        pkg_path = adarl.utils.utils.pkgutil_get_path(pkg,"")
        if pkg_path is None:
            raise RuntimeError(f"Failed to get path for package {pkg}")
        shutil.copytree(pkg_path, folderName+"/"+pkg)
    args["launch_id"] = launch_id #Unique for each launch, even between different seeds, this way they can be grouped together
    

    num_processes = maxProcs
    if resumeFolder is None:
        if seeds == None:
            seeds = [x+seedsOffset for x in range(seedsNum)]
        if pretrainedModelFile is not None:
            pretrainedModelFile = os.path.abspath(pretrainedModelFile)
        argss = [{"seed" : seed,
                  "folderName" : folderName,
                  "runFunction" : runFunction,
                  "resumeModelFile" : pretrainedModelFile,
                  "run_id" : launch_id+"_"+str(seed),
                  "run_args" : args,
                  "start_adarl" : start_adarl,
                  "launch_file_path" : launchFilePath,
                  "debug_level" : debug_level} for seed in seeds]
    else:
        resumeFolder = os.path.abspath(resumeFolder)
        ggLog.info(f"Resuming run from folder {resumeFolder}")
        if pretrainedModelFile is not None:
            raise AttributeError("Incompatible arguments, cannot specify both pretrainedModelFile and resumeFolder")
        if seeds is not None:
            raise AttributeError("Incompatible arguments, cannot specify both seeds and resumeFolder")
        detected_args = detectFolderArgs(resumeFolder)
        # for seed in det_args.keys():
        #     rebuild_run_folder(folderName, seed, resumeFolder)
        argss = [{"seed" : seed,
                  "folderName" : folderName,
                  "runFunction" : runFunction,
                  "resumeModelFile" : detected_args[seed]["modelToLoad"],
                  "run_id" : launch_id+"_"+str(seed),
                  "run_args" : args,
                  "start_adarl" : start_adarl,
                  "launch_file_path" : launchFilePath,
                  "debug_level" : debug_level} for seed in detected_args]

    ggLog.info(f"Will launch {argss} using {num_processes} processes") 

    num_processes = min(num_processes, len(argss))

    if len(argss) == 1 or num_processes==1:
        run_results = []
        for args in argss:
            r = runFunction_wrapper(**args)
            run_results.append(r)
    else:
        from adarl.utils.sigint_handler import setupSigintHandler, launch_halt_waiter
        setupSigintHandler()
        launch_halt_waiter()
        with NestablePool(num_processes, maxtasksperchild=1) as p:
            # run_results = p.starmap(runFunction_wrapper, argss)
            # run_results = p.starmap(lambda f,kwargs: f(**kwargs), [([],kwargs) for kwargs in argss])
            run_results = p.starmap(runFunction_wrapper_arg_kwargs, [([],kwargs) for kwargs in argss])
    
    ggLog.info(f"All runs finished. Results:\n"+"\n".join([str(run_result) for run_result in run_results]))
            
    default_session.shutdown() # tell whatever may be running to stop
