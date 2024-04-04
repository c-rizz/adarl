#!/usr/bin/env python3

import time
import tqdm
import inspect
import numpy as np
import os

os.environ["MUJOCO_GL"]="egl"


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import lr_gym.utils.dbg.ggLog as ggLog
import gymnasium as gym
import lr_gym.utils.utils
from stable_baselines3.td3.policies import MultiInputPolicy
from lr_gym.envs.GymToLr import GymToLr
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import datetime
from lr_gym.utils.buffers import ThDictReplayBuffer
from lr_gym.envs.ObsToDict import ObsToDict
import torch as th
import lr_gym.utils.session

def build_hafcheetah(seed,logFolder) -> gym.Env:

    
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    stepLength_sec = 0.05
    
    base_gym_env = gym.make('HalfCheetah-v4', render_mode="rgb_array")
    base_gym_env_step_len_sec = base_gym_env.unwrapped.dt
    # TODO: what's the gym control frequency?
    stepLength_sec = base_gym_env_step_len_sec
    lrenv = GymToLr(openaiGym_env=base_gym_env, stepSimDuration_sec=stepLength_sec)
    lrenv = ObsToDict(env=lrenv)
    gym_env = GymEnvWrapper(lrenv)
    env = RecorderGymWrapper(env=gym_env, fps = 1/stepLength_sec, outFolder=logFolder+"/videos/RecorderGymWrapper", saveFrequency_ep=50)
    #setup seeds for reproducibility
    # env.seed(RANDOM_SEED)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


def runFunction(seed, folderName, resumeModelFile, run_id, args):
    learning_rate = 0.001
    buffer_size = 200000
    learning_starts = 10000
    batch_size = 100
    tau = 0.005
    gamma = 0.98
    train_freq = (1, "episode")
    gradient_steps = -1
    policy_kwargs = {"net_arch":[400,300]}
    noise_std = 0.0
    device = th.device("cuda:0")
    
    seed = 20200401
    env = build_hafcheetah(seed = seed, logFolder=folderName)


    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))


    # replay_buffer_class = ThDictReplayBuffer
    # replay_buffer_kwargs = {"storage_torch_device" : th.device("cuda:0")}
    model = TD3(policy=MultiInputPolicy,
                env=env,
                verbose=1,
                batch_size=batch_size,
                buffer_size=buffer_size,
                gamma=gamma,
                gradient_steps=gradient_steps,
                tau=tau,
                learning_rate=learning_rate,
                learning_starts=learning_starts,
                policy_kwargs=policy_kwargs,
                train_freq=train_freq,
                seed = seed,
                device=device,
                action_noise=action_noise) #,
                # replay_buffer_class=replay_buffer_class,
                # replay_buffer_kwargs=replay_buffer_kwargs)

    
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1_000_000)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    res = lr_gym.utils.utils.evaluatePolicy(env = env, model = None, episodes = 10, predict_func=model.predict)
    print(f"Summary:\n{res}")



if __name__ == "__main__":

    import os
    import argparse
    import multiprocessing
    from lr_gym.utils.session import launchRun

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