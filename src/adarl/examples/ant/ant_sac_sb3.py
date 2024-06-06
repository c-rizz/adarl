#!/usr/bin/env python3

import time
import tqdm
import inspect
import numpy as np
import os

os.environ["MUJOCO_GL"]="egl"


import adarl.utils.dbg.ggLog as ggLog
import gymnasium as gym
import adarl.utils.utils
from adarl.envs.GymToLr import GymToLr
from adarl.envs.RecorderGymWrapper import RecorderGymWrapper
from adarl.envs.GymEnvWrapper import GymEnvWrapper
from adarl.utils.buffers import ThDReplayBuffer
from adarl.envs.ObsToDict import ObsToDict
import torch as th
import adarl.utils.session
from typing import Tuple
from stable_baselines3.common.vec_env import SubprocVecEnv
from adarl.envs.VecEnvLogger import VecEnvLogger
from stable_baselines3 import SAC
from adarl.utils.sb3_callbacks import EvalCallback_ep
from wandb.integration.sb3 import WandbCallback
from adarl.utils.sb3_callbacks import SigintHaltCallback

def build_ant(seed,logFolder) -> Tuple[gym.Env, float]:

    
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    stepLength_sec = 0.05
    
    base_gym_env = gym.make('Ant-v4', render_mode="rgb_array", terminate_when_unhealthy=False)
    base_gym_env_step_len_sec = base_gym_env.unwrapped.dt
    # TODO: what's the gym control frequency?
    stepLength_sec = base_gym_env_step_len_sec
    lrenv = GymToLr(openaiGym_env=base_gym_env, stepSimDuration_sec=stepLength_sec)
    lrenv = ObsToDict(env=lrenv)
    gym_env = GymEnvWrapper(lrenv)
    # env = RecorderGymWrapper(env=gym_env, fps = 1/stepLength_sec, outFolder=logFolder+"/videos/RecorderGymWrapper", saveFrequency_ep=50)
    #setup seeds for reproducibility
    # env.seed(RANDOM_SEED)
    gym_env.action_space.seed(seed)
    gym_env.reset(seed=seed)
    return gym_env, 1/stepLength_sec


def runFunction(seed, folderName, resumeModelFile, run_id, args):
   
    device = th.device("cuda:0")
    
    seed = 20200401
    parallel_envs = 4

    builders = [(lambda i: (lambda: build_ant(seed = seed*100000+i, logFolder=folderName)[0]))(i) for i in range(parallel_envs)]
    env = SubprocVecEnv(builders, start_method = "forkserver")
    env = VecEnvLogger(env)
    
    eval_env, targetFps = build_ant(seed = seed*100000000, logFolder=folderName+"/eval")
    eval_recEnv = RecorderGymWrapper(eval_env,
                                fps = targetFps, outFolder = folderName+"/eval/videos/RecorderGymWrapper",
                                saveBestEpisodes = True,
                                saveFrequency_ep = 1)
    eval_env = eval_recEnv
    

    model = SAC("MultiInputPolicy", env, verbose=1,
                    batch_size=8192,
                    buffer_size=10_000_000,
                    gamma=0.99,
                    learning_rate=0.005,
                    ent_coef="auto",
                    learning_starts=10000,
                    tau=0.005,
                    gradient_steps=100,
                    train_freq=(1000,"step"),
                    target_entropy="auto",
                    seed = seed,
                    device=device,
                    policy_kwargs=dict(net_arch=[256,256]),
                    replay_buffer_class = ThDReplayBuffer,
                    replay_buffer_kwargs = {"storage_torch_device" : device},
                    tensorboard_log=folderName+f"/tensorboard")

    callbacks = []
    callbacks.append(EvalCallback_ep(eval_env, best_model_save_path=folderName+"/eval/EvalCallback",
                                    log_path=folderName+"/eval/EvalCallback", eval_freq_ep=50,
                                    deterministic=False, render=False, verbose=True,
                                    n_eval_episodes = 1))
    callbacks += [WandbCallback( gradient_save_freq=100,
                                            model_save_path=None,
                                            verbose=2),
                            SigintHaltCallback()]
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1_000_000,
                callback=callbacks)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    res = adarl.utils.utils.evaluatePolicy(env = eval_env, model = None, episodes = 10, predict_func=model.predict)
    print(f"Summary:\n{res}")



if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from adarl.utils.session import launchRun

    ap = argparse.ArgumentParser()
    # ap.add_argument("--evaluate", default=None, type=str, help="Load and evaluate model file")
    ap.add_argument("--resumeFolder", default=None, type=str, help="Resume an entire run composed of multiple seeds")
    ap.add_argument("--seedsNum", default=1, type=int, help="Number of seeds to test with")
    # ap.add_argument("--seeds", nargs="+", required=False, type=int, help="Seeds to use")
    # ap.add_argument("--no_rb_checkpoint", default=False, action='store_true', help="Do not save replay buffer checkpoints")
    # ap.add_argument("--robot_pc_ip", default=None, type=str, help="Ip of the pc connected to the robot (which runs the control, using its rt kernel)")
    ap.add_argument("--seedsOffset", default=0, type=int, help="Offset the used seeds by this amount")
    # ap.add_argument("--xvfb", default=False, action='store_true', help="Run with xvfb")
    ap.add_argument("--maxProcs", default=int(multiprocessing.cpu_count()/2), type=int, help="Maximum number of parallel runs")
    # ap.add_argument("--offline", default=False, action='store_true', help="Train offline")
    # group = ap.add_mutually_exclusive_group()
    # group.add_argument("--gazebo",     default=False, action='store_true',     help="Use gazebo classic env")
    # group.add_argument("--gz",         default=False, action='store_true',         help="Use ignition gazebo env")
    # group.add_argument("--simplified", default=False, action='store_true', help="Use simplified pybullet env")
    # group.add_argument("--real", default=False, action='store_true', help="Run on real robot")
    ap.add_argument("--comment", required = True, type=str, help="Comment explaining what this run is about")

    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    # if args["real"] and args["maxProcs"]>0:
    #     raise AttributeError("Cannot run multiple processes in the real")


    # if args["simplified"]:
    #     mode = "simplified"
    # elif args["gz"]:
    #     mode = "gz"
    # elif args["gazebo"]:
    #     mode = "gazebo_classic"
    # else:
    #     raise RuntimeError("No mode was specified, use either --gazebo --gz or --simplified")

    action_repeat = 4
    ep_duration = int(1000/action_repeat)
    parallel_envs = 1


    
    launchRun(  seedsNum=args["seedsNum"],
                seedsOffset=args["seedsOffset"],
                runFunction=runFunction,
                maxProcs=args["maxProcs"],
                launchFilePath=__file__,
                resumeFolder = args["resumeFolder"],
                args = args,
                debug_level = -10)