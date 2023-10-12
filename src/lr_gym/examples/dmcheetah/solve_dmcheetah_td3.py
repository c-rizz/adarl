#!/usr/bin/env python3

import torch as th
import time
import tqdm
import inspect
import numpy as np
from nptyping import NDArray

from stable_baselines3.td3.policies import MlpPolicy, MultiInputPolicy
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym.envs.ObsDict2FlatBox import ObsDict2FlatBox
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils

from lr_gym.envs.GymToLr import GymToLr
from lr_gym.envs.ObsToDict import ObsToDict
import os
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper
from autoencoding_rl.buffers import GenericHerReplayBuffer, RandomHoldoutBuffer, ThDictReplayBuffer_updatable, ThDictReplayBuffer


def main(obsNoise : NDArray[(4,),np.float32]) -> None: 
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    RANDOM_SEED=0
    
    folderName = lr_gym.utils.utils.lr_gym_startup(__file__, inspect.currentframe(), seed=RANDOM_SEED)

    # dmenv = suite.load("cheetah",
    #                     "run",
    #                     task_kwargs={'random': RANDOM_SEED},
    #                     visualize_reward=False)
    ggLog.info("Building env...")
    os.environ["MUJOCO_GL"] = "egl"
    import dmc2gym.wrappers
    targetFps = 100
    env = dmc2gym.make(domain_name='cheetah', task_name='run', seed=RANDOM_SEED, frame_skip = 2) # dmc2gym.wrappers.DMCWrapper(env=dmenv,task_kwargs = {'random' : RANDOM_SEED})
    env = GymToLr(openaiGym_env = env, stepSimDuration_sec = 1/targetFps)
    #env = ObsToDict(env)
    #env = ObsDict2FlatBox(env)
    env = GymEnvWrapper(env, episodeInfoLogFile = folderName+"/GymEnvWrapper_log.csv")
    env = RecorderGymWrapper(env,
                             fps = targetFps, outFolder = folderName+"/videos/RecorderGymWrapper",
                             saveBestEpisodes = False,
                             saveFrequency_ep = 50)
    ggLog.info("Built")

    #setup seeds for reproducibility
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

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
                    seed = RANDOM_SEED, device = "cuda",
                    replay_buffer_class = ThDictReplayBuffer,
                    replay_buffer_kwargs = {"storage_torch_device":lr_gym.utils.utils.torch_selectBestGpu()})

    
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=1000000)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    ggLog.info("Computing average reward...")
    t_preVal = time.time()
    rewards=[]
    totFrames=0
    totDuration=0
    #frames = []
    #do an average over a bunch of episodes
    for episode in tqdm.tqdm(range(0,50)):
        frame = 0
        episodeReward = 0
        done = False
        obs = env.reset()
        t0 = time.time()
        while not done:
            #ggLog.info("Episode "+str(episode)+" frame "+str(frame))
            action, _states = model.predict(obs)
            obs, stepReward, done, info = env.step(action)
            #frames.append(env.render("rgb_array"))
            #time.sleep(0.016)
            frame+=1
            episodeReward += stepReward
        rewards.append(episodeReward)
        totFrames +=frame
        totDuration += time.time() - t0
        #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    avgReward = sum(rewards)/len(rewards)
    duration_val = time.time() - t_preVal
    ggLog.info("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")
    ggLog.info("Average reward = "+str(avgReward))

if __name__ == "__main__":
    n = None    
    main(n)
