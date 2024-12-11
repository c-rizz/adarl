#!/usr/bin/env python3

import os
os.environ["MUJOCO_GL"] = "egl"

import jax
import mujoco
from mujoco import mjx
import time
from tqdm import tqdm

import adarl.utils.utils
from pathlib import Path
import jax

def run_classic_mujoco(duration, framerate, mj_model, mj_data, scene_option, renderer):
    t0 = time.monotonic()
    print("------- RUN MUJOCO CASSIC -------")
    print(f" gravity = {mj_model.opt.gravity}")
    print(f" qpos = {mj_data.qpos}")
    mj_data.qpos[1] = 1 # to put it out of balance
    # print(f" qpos = {mj_data.qpos}")

    render_count = 0
    for i in tqdm(range(int(duration/mj_model.opt.timestep))):
        mujoco.mj_step(mj_model, mj_data)
        print(f"time = {mj_data.time} qpos = {mj_data.qpos}")
        if renderer is not None and render_count < mj_data.time * framerate:
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            render_count+=1
        # print(f"mj_data.time = {mj_data.time}")
        # print(f"mj_model.opt.timestep*i = {mj_model.opt.timestep*(i+1)}")
    tf = time.monotonic()

    # print(f" qpos = {mj_data.qpos}")
    
    # Simulate and display video.
    # media.show_video(frames, fps=framerate)
    return tf-t0
            
def run_mjx(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer):
    print("------- RUN MJX -------")
    print(f" qpos = {mjx_data.qpos}")
    jit_step = jax.jit(mjx.step)
    _ = jit_step(mjx_model, mjx_data) # trigger jit compile

    t0 = time.monotonic()
    render_count = 0
    for i in tqdm(range(int(duration/mj_model.opt.timestep))):
        mjx_data = jit_step(mjx_model, mjx_data)
        if renderer is not None and render_count < mj_model.opt.timestep*(i+1)*framerate:
            mj_data = mjx.get_data(mj_model, mjx_data)
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
        render_count+=1
    tf = time.monotonic()
    # print(f" qpos = {mjx_data.qpos}")

    # media.show_video(frames, fps=framerate)
    return tf-t0
            
def run_mjx_batched(duration, framerate, mj_model, mjx_model, mjx_data : mjx.Data, scene_option, renderer, numenvs):
    print(f"------- RUN MJX ({numenvs}) -------")
    print(f" qpos = {mjx_data.qpos}")
    frames = []
    # rng = jax.random.PRNGKey(0)
    # print(f"rng.shape {rng.shape}")
    # rng = jax.random.split(rng, numenvs)
    # print(f"rng.shape {rng.shape}")
    # data_batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))) )(rng)
    # # data_batch = jax.vmap(lambda _: mjx_data.replace(qpos=jax.numpy.array([0.0])))(rng) #ugly!
    # print(f"mjx_data.qpos.shape {mjx_data.qpos.shape}")    
    qpos_shape = mjx_data.qpos.shape
    # leaves = jax.tree_util.tree_leaves(mjx_data)
    # print(f"original:   has {len(leaves)} leaves: {[leave.shape if isinstance(leave, jax.Array) else leave for leave in leaves]}")
    data_batch = jax.vmap(lambda: mjx_data.replace(qpos=jax.numpy.zeros(shape=qpos_shape)), in_axes=None, out_axes=0, axis_size=numenvs)() 
    # leaves = jax.tree_util.tree_leaves(data_batch)
    # print(f"vmapped:   has {len(leaves)} leaves: {[leave.shape if isinstance(leave, jax.Array) else leave for leave in leaves]}")
    # print(f"data_batch.qpos.shape {data_batch.qpos.shape}")

    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    _ = jit_step(mjx_model, data_batch) # trigger jit compile

    def take_batch_element(batch, e):
        mjx_data = jax.tree_map(lambda l: l[e], batch)
        return mjx.get_data(mj_model, mjx_data)
    jit_take_batch_element = jax.jit(take_batch_element)
    t0 = time.monotonic()
    render_count = 0
    for i in tqdm(range(int(duration/mj_model.opt.timestep))):
        data_batch = jit_step(mjx_model, data_batch)
        # print(f"{data_batch.qpos}")
        if renderer is not None and render_count < mj_model.opt.timestep*(i+1)*framerate:
            mj_data_batch = mjx.get_data(mj_model, data_batch)[0]
            # print(f"mj_data_batch = {mj_data_batch}")
            for e in range(numenvs):
                mj_data = jax.tree_map(lambda l: l[e], mj_data_batch)
                renderer.update_scene(mj_data, scene_option=scene_option)
                pixels = renderer.render()
                # print(f"rendering {e}")
                # def f(l):
                #     print(f"type(l) = {type(l)}, l.shape = {l.shape}, l = {l}")
                #     return l[e]
                # env_mjx_data = jit_take_batch_element(data_batch)
                # env_mj_data = mjx.get_data(mj_model, env_mjx_data)
                # env_mj_data = jit_take_batch_element(data_batch)
                # renderer.update_scene(env_mj_data, scene_option=scene_option)
                # pixels = renderer.render()
            render_count+=1
    tf = time.monotonic()

    # print(f"data_batch.qpos.shape {data_batch.qpos.shape}")
    # print(f" qpos = {data_batch.qpos}")
    return tf-t0


def build_sim(render):
    # xml = """
    # <mujoco>
    # <worldbody>
    #     <light name="top" pos="0 0 1"/>
    #     <body name="box_and_sphere" euler="0 0 -30">
    #     <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
    #     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    #     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    #     </body>
    # </worldbody>
    # </mujoco>
    # """
    model_definition_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
                                                                        model_kwargs={})
    # model_definition_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_rig_simple.urdf.xacro")).read_text(),
    #                                                                     model_kwargs={"use_cylinders" : False})
    xml = model_definition_string
    # Make model, data, and renderer
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    if render:
        renderer = mujoco.Renderer(mj_model)
    else:
        renderer = None
    mujoco.mj_resetData(mj_model, mj_data)

    print(f"Model has {mj_model.njnt} joints")
    print(f"Found joints {[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in range(mj_model.njnt)]}")
    print(f"Found links {[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, lid) for lid in range(mj_model.nbody)]}")

    mjx_model = mjx.put_model(mj_model, device = jax.devices("gpu")[0])
    mjx_data = mjx.put_data(mj_model, mj_data, device = jax.devices("gpu")[0])

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    return mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option



def main():

    duration = 3  # (seconds)
    framerate = 60  # (Hz)
    render = False

    mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option = build_sim(render)
    ep_frames = duration/mj_model.opt.timestep
    d_mjc = run_classic_mujoco(duration, framerate, mj_model, mj_data, scene_option, renderer)
    dt = mj_model.opt.timestep
    print(f"mj_model.opt.timestep = {mj_model.opt.timestep}")
    print(f"MJ classic took {d_mjc:6f}s, {ep_frames/d_mjc} fps, {ep_frames*dt/d_mjc} sim/real")

    mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option = build_sim(render)
    d_mjx1 = run_mjx(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer)
    print(f"MJX took {d_mjx1:6f}s, {ep_frames/d_mjx1} fps, {ep_frames*dt/d_mjx1} sim/real")

    for n in [1,2,10,100,1000,5000,10000,100000]:
        mj_model, mj_data, renderer, mjx_model, mjx_data, scene_option = build_sim(render)
        d_mjx = run_mjx_batched(duration, framerate, mj_model, mjx_model, mjx_data, scene_option, renderer, numenvs=n)
        print(f"MJX({n}) took {d_mjx:6f}s ({d_mjc/(d_mjx/n):6f}x higher throughput than mjc, {(ep_frames*n)/d_mjx} fps, {ep_frames*n*dt/d_mjx} sim/real, single fps = {ep_frames/d_mjx})")



if __name__ == "__main__":
    main()