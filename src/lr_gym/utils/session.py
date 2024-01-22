from typing import Optional, List, Union
import lr_gym.utils.dbg.ggLog as ggLog
import numpy as np
import wandb.sdk
from lr_gym.utils.utils import pyTorch_makeDeterministic, createSymlink, exc_to_str

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
import re
import lr_gym.utils.utils
import multiprocessing
import random


_is_shutting_down = False

def is_shutting_down():
    return _is_shutting_down

def shutdown():
    """Mark the current session as shutting-down.
    """
    # print(f"Shutting down")
    global _is_shutting_down
    _is_shutting_down = True


from lr_gym.utils.sigint_handler import setupSigintHandler

wandb_enabled = False

def is_wandb_enabled():
    return wandb_enabled

def _setupLoggingForRun(file : str, currentframe = None, folderName : Optional[str] = None, use_wandb : bool = True, experiment_name : Optional[str] = None, run_id : Optional[str] = None, comment = ""):
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

    # ggLog.info(f"values = {values}")

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
            ggLog.info(f"Staring run with experiment name '{experiment_name}', run id {run_id}")
            wandb.init( project=experiment_name,
                        config = values,
                        name = run_id,
                        monitor_gym = False, # Do not save openai gym videos
                        save_code = True, # Save run code
                        sync_tensorboard = True, # Save tensorboard stuff,
                        notes = comment
                        )
            wandb_enabled = True
        except wandb.sdk.wandb_manager.ManagerConnectionError as e: # type: ignore
            ggLog.error(f"Wandb connection failed: {exc_to_str(e)}")

    return folderName



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


def lr_gym_startup( main_file_path : str,
                    currentframe = None,
                    using_pytorch : bool = True,
                    folderName : Optional[str] = None,
                    seed = None,
                    experiment_name : Optional[str] = None,
                    run_id : Optional[str] = None,
                    debug : Union[bool, int]  = False,
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
    logFolder = _setupLoggingForRun(main_file_path, currentframe, folderName=folderName, experiment_name=experiment_name, run_id=run_id, comment=run_comment, use_wandb=use_wandb)
    ggLog.addLogFile(logFolder+"/gglog.log")
    if seed is None:
        raise AttributeError("You must specify the run seed")
    ggLog.setId(str(seed))
    np.set_printoptions(edgeitems=10,linewidth=180)
    setupSigintHandler()
    if using_pytorch:
        import torch as th
        pyTorch_makeDeterministic(seed)
        if debug_level>0:
            th.cuda.set_sync_debug_mode("warn")
        th.autograd.set_detect_anomaly(debug_level >= 0) # type: ignore
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


















def runFunction_wrapper(seed,
                        folderName,
                        runFunction,
                        resumeModelFile,
                        run_id,
                        run_args,
                        start_lr_gym,
                        launch_file_path,
                        debug_level):
    try:
        seedFolder = folderName+f"/seed_{seed}"
        experiment_name = os.path.basename(launch_file_path)
        if start_lr_gym:
            folderName = lr_gym_startup(launch_file_path,
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
        lr_gym.utils.utils.createSymlink(src = folderName, dst = str(Path(folderName).parent.absolute())+"/latest")
        # time.sleep(seed)
        # if resumeModelFile is not None:
        #     os.makedirs(seedFolder,exist_ok=True)            
        return runFunction(seed=seed, folderName=seedFolder, resumeModelFile=resumeModelFile, run_id=run_id, args = run_args)
    except Exception as e:
        ggLog.error(f"Run failed with exception: {lr_gym.utils.utils.exc_to_str(e)}")
        return None

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
            pkgs_to_save = ["lr_gym"],
            start_lr_gym : bool = True,
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
        shutil.copytree(lr_gym.utils.utils.pkgutil_get_path(pkg,""), folderName+"/"+pkg)
    args["launch_id"] = launch_id #Unique for each launch, even between different seeds, this way they can be grouped together
    

    num_processes = maxProcs
    if resumeFolder is None:
        if seeds == None:
            seeds = [x+seedsOffset for x in range(seedsNum)]
        if pretrainedModelFile is not None:
            pretrainedModelFile = os.path.abspath(pretrainedModelFile)
        argss = [(seed,
                  folderName,
                  runFunction,
                  pretrainedModelFile,
                  launch_id+"_"+str(seed),
                  args,
                  start_lr_gym,
                  launchFilePath,
                  debug_level) for seed in seeds]
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
        argss = [(seed,
                  folderName,
                  runFunction,
                  detected_args[seed]["modelToLoad"],
                  launch_id+"_"+str(seed),
                  args,
                  start_lr_gym,
                  launchFilePath,
                  debug_level) for seed in detected_args]

    ggLog.info(f"Will launch {argss} using {num_processes} processes") 

    num_processes = min(num_processes, len(argss))

    if len(argss) == 1 or num_processes==1:
        run_results = []
        for args in argss:
            r = runFunction_wrapper(*args)
            run_results.append(r)
    else:
        from lr_gym.utils.sigint_handler import setupSigintHandler, launch_halt_waiter
        setupSigintHandler()
        launch_halt_waiter()
        with multiprocessing.Pool(num_processes, maxtasksperchild=1) as p:
            run_results = p.starmap(runFunction_wrapper, argss)
    
    ggLog.info(f"All runs finished. Results:\n"+"\n".join(str(run_results)))
            
    shutdown() # tell whatever may be running to stop
