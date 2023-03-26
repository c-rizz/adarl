#!/usr/bin/env python3

from lr_gym.envs.CartpoleEnv import CartpoleEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
import lr_gym.utils.utils
import time
import cv2
import os

import argparse



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=False, default="GzRosController", type=str, help="label to put on y axis")
    ap.add_argument("--saveimages", default=False, action='store_true', help="Do not center the window averaging")
    ap.set_defaults(feature=True)
    args = vars(ap.parse_args())

    stepLength_sec = 0.05

    if args["controller"] == "GzRosController":
        from lr_gym_ros2.env_controllers.GzRosController import GzRosController
        env_controller = GzRosController(stepLength_sec=stepLength_sec)
    elif args["controller"] == "GazeboController":
        from lr_gym_ros.envControllers.GazeboController import GazeboController
        env_controller = GazeboController(stepLength_sec=stepLength_sec)
    else:
        print(f"Requested unknown controller '{args['controller']}'")
        exit(0)

    render = True
    env = GymEnvWrapper(CartpoleEnv(startSimulation=True,
                                    environmentController = env_controller,
                                    render=render))

    images = [] if render else None
    obs = []
    t0 = time.monotonic()
    def policy(obs):
        return 1 if obs[3] > 0 else 0, 0
    res = lr_gym.utils.utils.evaluatePolicy(env = env, model = None, episodes = 10, predict_func=policy,
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