#!/usr/bin/env python3

from adarl.envs.CartpoleEnv import CartpoleEnv
from adarl.envs.GymEnvWrapper import GymEnvWrapper
import adarl.utils.utils
import time
import cv2
import os

import argparse
import gymnasium as gym


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=False, default="GzController", type=str, help="label to put on y axis")
    ap.add_argument("--episodes", required=False, default=10, type=int, help="label to put on y axis")
    ap.add_argument("--saveimages", default=False, action='store_true', help="Do not center the window averaging")
    ap.add_argument("--gymenv", default=False, action='store_true', help="Use directly the Gymnasium env")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    stepLength_sec = 0.05
    render = args["saveimages"]

    if args["gymenv"]:
        import gymnasium as gym
        env = gym.make('CartPole-v1')
    else:
        if args["controller"] == "GzController":
            from adarl_ros2.adapters.GzController import GzController
            env_controller = GzController(stepLength_sec=stepLength_sec)
        elif args["controller"] == "GazeboAdapter":
            from adarl_ros.adapters.GazeboAdapter import GazeboAdapter
            env_controller = GazeboAdapter(stepLength_sec=stepLength_sec)
        elif args["controller"] == "PyBulletAdapter":
            from adarl.adapters.PyBulletAdapter import PyBulletAdapter
            env_controller = PyBulletAdapter(stepLength_sec=stepLength_sec)
        else:
            print(f"Requested unknown controller '{args['controller']}'")
            exit(0)

        env = GymEnvWrapper(CartpoleEnv(startSimulation=True,
                                        environmentController = env_controller,
                                        render=render))

    images = [] if render else None
    obs = []
    t0 = time.monotonic()
    def policy(obs):
        return 1 if obs[3] > 0 else 0, 0
    res = adarl.utils.utils.evaluatePolicy(env = env, model = None, episodes = args["episodes"], predict_func=policy,
                                            images_return = images, obs_return=obs)
    t1 = time.monotonic()

    eps = len(obs)
    steps = sum([len(i) for i in obs])
    print(f"Summary:\n{res}")
    print(f"Got {eps} episodes, {steps} steps")
    print(f"took {t1-t0}s, {steps/(t1-t0)} fps, sim/real = {stepLength_sec*steps/(t1-t0):.2f}x")
    if images is not None:
        print(f"images[0][0].shape = {images[0][0].shape}")
        print(f"images[0][0].dtype = {images[0][0].dtype}")
        # print(f"images[0][0] = {images[0][0]}")
    newline = "\n"
    print(f"final obss = {newline.join([str(ep[-1]) for ep in obs])}")

    if args["saveimages"] and images is not None:
        imagesOutFolder = "./run_cartpole/"+str(int(time.time()))
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