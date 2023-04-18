#!/usr/bin/env python3

import time
import tqdm
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN
from lr_gym.envs.CartpoleEnv import CartpoleEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.dbg.ggLog as ggLog
import argparse
import lr_gym
import inspect
import datetime
import os
import cv2

def main() -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    logFolder = lr_gym.utils.utils.lr_gym_startup(__file__,
                                                    inspect.currentframe(),
                                                    folderName = f"solve_cartpole_env/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                                    seed = 0,
                                                    experiment_name = None,
                                                    run_id = None)
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    stepLength_sec = 0.05


    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=False, default="GzController", type=str, help="label to put on y axis")
    ap.add_argument("--saveimages", default=False, action='store_true', help="Do not center the window averaging")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    stepLength_sec = 0.05
    if args["controller"] == "GzController":
        from lr_gym_ros2.env_controllers.GzController import GzController
        env_controller = GzController(stepLength_sec=stepLength_sec)
    elif args["controller"] == "GazeboController":
        from lr_gym_ros.envControllers.GazeboController import GazeboController
        env_controller = GazeboController(stepLength_sec=stepLength_sec)
    elif args["controller"] == "PyBulletController":
        from lr_gym.env_controllers.PyBulletController import PyBulletController
        env_controller = PyBulletController(stepLength_sec=stepLength_sec)
    else:
        print(f"Requested unknown controller '{args['controller']}'")
        exit(0)

    env = GymEnvWrapper(CartpoleEnv(render=False,
                                    startSimulation = True,
                                    stepLength_sec = stepLength_sec,
                                    environmentController = env_controller),
                        episodeInfoLogFile = logFolder+"/GymEnvWrapper_log.csv")
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env._max_episode_steps = 500 #limit episode length

    model = DQN(MlpPolicy, env, verbose=1, seed=RANDOM_SEED, learning_starts=100,
                policy_kwargs=dict(net_arch=[64,64]), learning_rate = 0.0025, train_freq=1,
                target_update_interval=500) # , n_cpu_tf_sess=1) #seed=RANDOM_SEED, n_cpu_tf_sess=1 are needed to get deterministic results
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=25000)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


    # ggLog.info("Computing average reward...")
    # t_preVal = time.time()
    # rewards=[]
    # totFrames=0
    # totDuration=0
    # #frames = []
    # #do an average over a bunch of episodes
    # for episode in tqdm.tqdm(range(0,50)):
    #     frame = 0
    #     episodeReward = 0
    #     done = False
    #     obs = env.reset()
    #     t0 = time.time()
    #     while not done:
    #         #ggLog.info("Episode "+str(episode)+" frame "+str(frame))
    #         action, _states = model.predict(obs)
    #         obs, stepReward, done, info = env.step(action)
    #         #frames.append(env.render("rgb_array"))
    #         #time.sleep(0.016)
    #         frame+=1
    #         episodeReward += stepReward
    #     rewards.append(episodeReward)
    #     totFrames +=frame
    #     totDuration += time.time() - t0
    #     #print("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    # avgReward = sum(rewards)/len(rewards)
    # duration_val = time.time() - t_preVal
    # ggLog.info("Computed average reward. Took "+str(duration_val)+" seconds ("+str(totFrames/totDuration)+" fps).")
    # ggLog.info("Average rewar = "+str(avgReward))



    images = [] if args["saveimages"] else None
    obs = []
    t0 = time.monotonic()
    res = lr_gym.utils.utils.evaluatePolicy(env = env, model = None, episodes = 10, predict_func=model.predict,
                                            images_return = images, obs_return=obs)
    t1 = time.monotonic()

    eps = len(obs)
    steps = sum([len(i) for i in obs])
    print(f"Summary:\n{res}")
    print(f"Got {eps} episodes, {steps} steps")
    print(f"took {t1-t0}s, {steps/(t1-t0)} fps, sim/real = {stepLength_sec*steps/(t1-t0):.2f}x")

    if args["saveimages"] and images is not None:
        imagesOutFolder = "./solve_cartpole_sb3/"+str(int(time.time()))
        print(f"Saving images to {imagesOutFolder}...")
        os.makedirs(imagesOutFolder, exist_ok=True)
        episode = -1
        for ep in images:
            episode+=1
            frame = -1
            for img in ep:
                frame+=1
                # img = img[:,:,0]
                # img = np.transpose(img, (1,2,0))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                r = cv2.imwrite(imagesOutFolder+"/frame-"+str(episode)+"-"+str(frame)+".png",img_bgr)
                if not r:
                    print("couldn't save image")

if __name__ == "__main__":
    main()
