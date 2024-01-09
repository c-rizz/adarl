#!/usr/bin/env python3

import os
os.environ["MUJOCO_GL"] = "egl"

#@title Import MuJoCo, MJX, and Brax


from datetime import datetime
import functools
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
from ml_collections import config_dict
import mujoco
from mujoco import mjx


def run_classic_mujoco(duration, framerate, mj_model, mj_data, scene_option, renderer):
    print("------- RUN MUJOCO CASSIC -------")
    print(mj_data.qpos, type(mj_data.qpos))

    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    while mj_data.time < duration:
        mujoco.mj_step(mj_model, mj_data)
        if len(frames) < mj_data.time * framerate:
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    print(mj_data.qpos, type(mj_data.qpos))
    
    # Simulate and display video.
    # media.show_video(frames, fps=framerate)
            
def run_mjx(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer):
    print("------- RUN MJX -------")
    print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())
    jit_step = jax.jit(mjx.step)

    frames = []
    while mjx_data.time < duration:
        mjx_data = jit_step(mjx_model, mjx_data)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(mj_model, mjx_data)
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    # media.show_video(frames, fps=framerate)
            
def run_mjx_batched(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer):
    print("------- RUN MJX  -------")
    print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())
    frames = []
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 4096)
    data_batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

    jit_step = jax.vmap(mjx.step, in_axes=(None, 0))


    frames = []
    while mjx_data.time < duration:
        data_batch = jit_step(mjx_model, data_batch)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(mj_model, data_batch)
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    print(data_batch.qpos)


def build_sim():
    xml = """
    <mujoco>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
        <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
    </worldbody>
    </mujoco>
    """

    # Make model, data, and renderer
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    return mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option



def main():

    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)

    mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option = build_sim()
    run_classic_mujoco(duration, framerate, mj_model, mj_data, scene_option, renderer)

    mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option = build_sim()
    run_mjx(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer)

    mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option = build_sim()
    run_mjx_batched(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer)



if __name__ == "__main__":
    main()