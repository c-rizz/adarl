from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from pathlib import Path
import adarl.utils.utils
import os
import time
import torch as th
import cv2
from adarl.utils.utils import imgToCvIntRgb


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_FLAGS"]="--xla_gpu_triton_gemm_any=true"
import jax
from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter
from adarl.adapters.MjxAdapter import MjxAdapter

def test_sim_adapter(adapter : BaseVecSimulationAdapter, render : bool, print_state : bool):
    adapter.build_scenario([ModelSpawnDef( name="cartpole",
                                           definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
                                           format="urdf.xacro",
                                           pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                                           kwargs={}),
                            ModelSpawnDef( name="ball",
                                           definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/ball.urdf")).read_text(),
                                           format="urdf",
                                           pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                                           kwargs={}), #"0.707 -0.707 0 0"}),
                            # ModelSpawnDef( name="ball2",
                            #                definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/ball.urdf")).read_text(),
                            #                format="urdf",
                            #                pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                            #                kwargs={}), #"0.707 -0.707 0 0"}),
                            ModelSpawnDef( name="camera",
                                           definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.mjcf.xacro")).read_text(),
                                           format="mjcf.xacro",
                                           pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                                           kwargs={"camera_width":480,
                                                   "camera_height":int(480*9/16),
                                                   "position_xyz":"0 2 0.5",
                                                   "orientation_wxyz":"0.0 0.0 0.707 0.707"}), #"0.707 -0.707 0 0"}),
                            # ModelSpawnDef( name="ground",
                            #                definition_string="""<mujoco>
                            #                                         <compiler angle="radian"/>
                            #                                         <asset>
                            #                                             <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
                            #                                             <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
                            #                                             <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
                            #                                         </asset>
                            #                                         <worldbody>
                            #                                             <body name="ground_link">
                            #                                                 <light pos="0 0 10" dir="0 0 -1" directional="true" />
                            #                                                 <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin="0.0" />
                            #                                             </body>
                            #                                         </worldbody>
                            #                                     </mujoco>""",
                            #                format="mjcf",
                            #                pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                            #                kwargs={})
                                           ])
    
    # model_file = adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf")
    # adapter.build_scenario([ModelSpawnDef( name="cube",
    #                                        definition_string=Path(model_file).read_text(),
    #                                        format="urdf",
    #                                        pose=adarl.utils.utils.build_pose(0,0,10,0,0,0,1),
    #                                        kwargs={})])
    
    vsize = adapter.vec_size()
    print(f"detected joints = {adapter.detected_joints()}")
    print(f"detected links = {adapter.detected_links()}")
    print(f"detected cameras = {adapter.detected_cameras()}")
    n = "\n"
    adapter.set_monitored_joints([("cartpole","cartpole_joint"),("cartpole","foot_joint")])
    adapter.set_monitored_links([("cartpole","bar_link"),("cartpole","base_link"),("camera","simple_camera_link")])
    # adapter.setLinksStateDirect([("camera","simple_camera_link")],th.as_tensor([0.0, 20.0, 1.0,
    #                                                                             0.0, 0.0, 0.0, 1.0,
    #                                                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).repeat((vsize,1,13)))
    adapter.setLinksStateDirect([("ball","ball")],th.as_tensor([0.0, 3.0, 1.0,
                                                                0.0, 0.0, 0.0, 1.0,
                                                                1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).repeat((vsize,1,13)))
    # adapter.set_monitored_joints([])
    # adapter.set_monitored_links([("cube","cube_link2")])
    adapter.startup()

    print(f"joint states = {adapter.getJointsState()}")
    print(f"link states = {adapter.getLinksState()}")
    # print(f"revolute joint state = {adapter.getJointsState([('cartpole','foot_joint')])}")
    # print(f"cart link state = {adapter.getLinksState([('cartpole','base_link')])}")

    adapter.setJointsStateDirect([("cartpole","cartpole_joint")], th.cat([th.randn(size=(vsize,1,1), device = "cuda")*0.05,
                                                                          th.zeros(size=(vsize,1,2), device = "cuda")], dim=2))
    print(f"joint states = {adapter.getJointsState()}")

    os.makedirs("test_sim_adapter", exist_ok=True)
    if isinstance(adapter, BaseVecJointImpedanceAdapter):
        adapter.set_impedance_controlled_joints([("cartpole","cartpole_joint")])
        adapter.set_current_joint_impedance_command(th.as_tensor([[[0.0,0.0,0.0,200.0,10.0]]], device = "cuda"))
    t0 = time.monotonic()
    for step in range(100):
        t1 = time.monotonic()
        if isinstance(adapter,BaseVecJointEffortAdapter) and not isinstance(adapter,BaseVecJointImpedanceAdapter):
            a = 50.0*adapter.getJointsState([('cartpole','cartpole_joint')])[:,:,0]
            # print(f"commanding {a}")
            adapter.setJointsEffortCommand([("cartpole","foot_joint")], a)
        if step == 1:
            t0 = time.monotonic() # exclude the first round, it may have some jit compiles in it
        dt = adapter.step()
        if render:
            img = adapter.getRenderings(["simple_camera"])[0][0][0]
            img = imgToCvIntRgb(img, min_val=0, max_val=1)
            r = cv2.imwrite(f"test_sim_adapter/frame-{step}.png",img)
            if not r:
                print("couldn't save image")
        # time.sleep(100/1024)
        # print(f"stepped of {dt}s")

        # print(f"joint states = {adapter.getJointsState()}")
        stime = adapter.getEnvTimeFromStartup()
        tf = time.monotonic()
        wtime = tf - t0
        print(f"[{step}] t = {stime:.3f} "
              f"    # tot single rt = {stime/wtime:.5f} "
              f"    # tot vec rt = {(stime * adapter.vec_size())/wtime:.5f} "
              f"    # inst single rt {dt/(tf-t1):.5f}")
        if print_state:
            js = adapter.getJointsState([('cartpole','cartpole_joint'),('cartpole','foot_joint')])[:,:]
            print(  f"revolute joint state = {js[:,0,:]}\n"
                    f"linear   joint state = {js[:,1,:]}")
        # print(f"link states = {adapter.getLinksState(use_com_pose=False)}")

        # print(f"link states = {adapter.getLinksState()}")
        # print(f"cart link state = {adapter.getLinksState([('cartpole','base_link')])}")

    adapter.destroy_scenario()

render = False
test_sim_adapter(MjxJointImpedanceAdapter(  vec_size=1000,
                                            enable_rendering=render,
                                            jax_device=jax.devices("gpu")[0],
                                            output_th_device=th.device("cuda",0),
                                            sim_step_dt=1/512,
                                            step_length_sec=50/1024,
                                            realtime_factor=-1,
                                            show_gui=False,
                                            max_joint_impedance_ctrl_torques = {('cartpole','foot_joint') : 100.0,
                                                                                ('cartpole','cartpole_joint') : 100.0}),
                                            render = render,
                                            print_state = False)
# test_sim_adapter(MjxAdapter(vec_size=10000,
#                             enable_rendering=render,
#                             jax_device=jax.devices("gpu")[0],
#                             output_th_device=th.device("cuda",0),
#                             sim_step_dt=1/512,
#                             step_length_sec=50/1024,
#                             realtime_factor=-1,
#                             show_gui=False),
#                     render = render,
#                     print_state = False)

