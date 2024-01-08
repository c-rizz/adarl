#!/usr/bin/env python3

import time
import inspect
import numpy as np

from stable_baselines3.td3.policies import MlpPolicy, MultiInputPolicy
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils

from lr_gym.envs.GymToLr import GymToLr
from lr_gym.envs.LrWrappers.ObsToDict import ObsToDict
import os
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper
from lr_gym.utils.sb3_buffers import ThDictReplayBuffer
from lr_gym.envs.NestedDictFlattenerGymWrapper import NestedDictFlattenerGymWrapper
from lr_gym.envs.LrWrappers.ObsToImgVecDict import ObsToImgVecDict

def runFunction(seed, folderName, resumeModelFile, run_id, args):
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """
   
    folderName = lr_gym.utils.session.lr_gym_startup(__file__, inspect.currentframe(), seed=seed)

    # dmenv = suite.load("cheetah",
    #                     "run",
    #                     task_kwargs={'random': RANDOM_SEED},
    #                     visualize_reward=False)
    ggLog.info("Building env...")
    os.environ["MUJOCO_GL"] = "egl"
    import dmc2gym.wrappers
    targetFps = 100
    env = dmc2gym.make(domain_name='cheetah', task_name='run', seed=seed, frame_skip = 2) # dmc2gym.wrappers.DMCWrapper(env=dmenv,task_kwargs = {'random' : RANDOM_SEED})
    print(f"dmc2gym gave {env}")
    # env = EnvCompatibility(old_env=env, render_mode="rgb_array")
    env = GymToLr(openaiGym_env = env, stepSimDuration_sec = 1/targetFps)
    # env = ObsToDict(env)
    env = ObsToImgVecDict(env)
    #env = ObsDict2FlatBox(env)
    env = GymEnvWrapper(env, episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    env = RecorderGymWrapper(env,
                             fps = targetFps, outFolder = folderName+"/videos/RecorderGymWrapper",
                             saveBestEpisodes = False,
                             saveFrequency_ep = 50)
    ggLog.info("Built")

    #setup seeds for reproducibility
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # model = SAC(MlpPolicy, env, verbose=1,
    #             buffer_size=20000,
    #             batch_size = 64,
    #             learning_rate=0.0025,
    #             policy_kwargs=dict(net_arch=[32,32]),
    #             target_entropy = 0.9)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3( MultiInputPolicy, env, action_noise=action_noise, verbose=1, batch_size=100,
                    buffer_size=1000000, gamma=0.99, gradient_steps=1000,
                    learning_rate=0.0005, learning_starts=5000, policy_kwargs=dict(net_arch=[64,64]), train_freq=1000,
                    seed = seed, device = "cuda",
                    replay_buffer_class = ThDictReplayBuffer,
                    replay_buffer_kwargs = {"storage_torch_device":lr_gym.utils.utils.torch_selectBestGpu()})

    
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1000000)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    # res = lr_gym.utils.utils.evaluatePolicy(env = eval_env, model = None, episodes = 10, predict_func=model.predict)
    # print(f"Summary:\n{res}")

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
