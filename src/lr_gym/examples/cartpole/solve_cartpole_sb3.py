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
from lr_gym.envs.GymToLr import GymToLr
from lr_gym.envs.RecorderGymWrapper import RecorderGymWrapper

def main() -> None:
    """Solves the gazebo cartpole environment using the DQN implementation by stable-baselines.

    It does not use the rendering at all, it learns from the joint states.
    The DQN hyperparameters have not been tuned to make this efficient.

    Returns
    -------
    None

    """

    logFolder, session = lr_gym.utils.session.lr_gym_startup(__file__,
                                                    inspect.currentframe(),
                                                    folderName = os.path.basename(__file__)+f"/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                                    seed = 0,
                                                    experiment_name = None,
                                                    run_id = None)
    #logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    #rospy.init_node('solve_dqn_stable_baselines', anonymous=True, log_level=rospy.WARN)
    #env = gym.make('CartPoleStayUp-v0')
    stepLength_sec = 0.05


    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=False, default="gym", type=str, help="Cartpole environment to use")
    ap.add_argument("--saveimages", default=False, action='store_true', help="Do not center the window averaging")
    ap.add_argument("--gymenv", default=True, action='store_true', help="Do not center the window averaging")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    mode = args["mode"].lower().strip()

    if mode == "gymenv":
        import gymnasium as gym
        base_gym_env = gym.make('CartPole-v1', render_mode="rgb_array")
        base_gym_env_step_len_sec = base_gym_env.tau
        # TODO: what's the gym control frequency?
        stepLength_sec = base_gym_env_step_len_sec
        lrenv = GymToLr(openaiGym_env=base_gym_env, stepSimDuration_sec=stepLength_sec)
    else:
        if mode == "GzController":
            from lr_gym_ros2.env_controllers.GzController import GzController
            env_controller = GzController(stepLength_sec=stepLength_sec)
        elif mode == "GazeboController":
            from lr_gym_ros.envControllers.GazeboController import GazeboController
            env_controller = GazeboController(stepLength_sec=stepLength_sec)
        elif mode == "PyBulletController":
            from lr_gym.env_controllers.PyBulletController import PyBulletController
            env_controller = PyBulletController(stepLength_sec=stepLength_sec)
        else:
            print(f"Requested unknown controller '{args['controller']}'")
            exit(0)
        lrenv = CartpoleEnv(startSimulation=True,
                            environmentController = env_controller,
                            render=render)
    gym_env = GymEnvWrapper(lrenv)
    env = RecorderGymWrapper(env=gym_env, fps = 1/stepLength_sec, outFolder=logFolder+"/videos/RecorderGymWrapper", saveFrequency_ep=50)
    #setup seeds for reproducibility
    RANDOM_SEED=20200401
    # env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env.reset(seed=RANDOM_SEED)
    # env._max_episode_steps = 500 #limit episode length

    model = DQN(MlpPolicy, env, verbose=1, seed=RANDOM_SEED, learning_starts=100,
                policy_kwargs=dict(net_arch=[64,64]), learning_rate = 0.0025, train_freq=1,
                target_update_interval=500) # , n_cpu_tf_sess=1) #seed=RANDOM_SEED, n_cpu_tf_sess=1 are needed to get deterministic results
    ggLog.info("Learning...")
    t_preLearn = time.time()
    model.learn(total_timesteps=25000)
    duration_learn = time.time() - t_preLearn
    ggLog.info("Learned. Took "+str(duration_learn)+" seconds.")


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


if __name__ == "__main__":
    main()
