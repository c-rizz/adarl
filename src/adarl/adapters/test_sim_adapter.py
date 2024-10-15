from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from pathlib import Path
import adarl.utils.utils
import os
import time
import torch as th
import cv2
from adarl.utils.utils import imgToCvIntRgb

def test_sim_adapter(adapter : BaseVecSimulationAdapter):
    adapter.build_scenario([ModelSpawnDef( name="cartpole",
                                           definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
                                           format="urdf.xacro",
                                           pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                                           kwargs={}),
                            ModelSpawnDef( name="camera",
                                           definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/simple_camera.mjcf.xacro")).read_text(),
                                           format="mjcf.xacro",
                                           pose=adarl.utils.utils.build_pose(0,0,0,0,0,0,1),
                                           kwargs={"camera_width":480,
                                                   "camera_height":int(480*9/16)})])
    
    # model_file = adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf")
    # adapter.build_scenario([ModelSpawnDef( name="cube",
    #                                        definition_string=Path(model_file).read_text(),
    #                                        format="urdf",
    #                                        pose=adarl.utils.utils.build_pose(0,0,10,0,0,0,1),
    #                                        kwargs={})])
    
    print(f"detected joints = {adapter.detected_joints()}")
    print(f"detected links = {adapter.detected_links()}")
    print(f"detected cameras = {adapter.detected_cameras()}")
    n = "\n"
    adapter.set_monitored_joints([("cartpole","cartpole_joint"),("cartpole","foot_joint")])
    adapter.set_monitored_links([("cartpole","bar_link"),("cartpole","base_link"),("camera","simple_camera_link")])
    # adapter.setLinksStateDirect([("camera","simple_camera_link")],th.as_tensor([0.0, 2.0, 1.0,
    #                                                                             0.0, 0.0, 0.0, 1.0,
    #                                                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).repeat((10,1,13)))
    # adapter.set_monitored_joints([])
    # adapter.set_monitored_links([("cube","cube_link2")])
    adapter.startup()

    print(f"joint states = {adapter.getJointsState()}")
    # print(f"revolute joint state = {adapter.getJointsState([('cartpole','foot_joint')])}")
    print(f"link states = {adapter.getLinksState()}")
    # print(f"cart link state = {adapter.getLinksState([('cartpole','base_link')])}")

    adapter.setJointsStateDirect([("cartpole","cartpole_joint")], th.cat([th.randn(size=(10,1,1), device = "cuda")*0.01,
                                                                          th.zeros(size=(10,1,2), device = "cuda")], dim=2))
    print(f"joint states = {adapter.getJointsState()}")

    os.makedirs("test_sim_adapter", exist_ok=True)
    for t in range(1000):
        dt = adapter.step()
        img = adapter.getRenderings(["simple_camera"])[0][0][0]
        print(f"img = {img}")
        img = imgToCvIntRgb(img, min_val=0, max_val=1)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        r = cv2.imwrite(f"test_sim_adapter/frame-{t}.png",img_bgr)
        if not r:
            print("couldn't save image")
        # time.sleep(100/1024)
        # print(f"stepped of {dt}s")

        # print(f"joint states = {adapter.getJointsState()}")
        print(f"t = {adapter.getEnvTimeFromStartup():.3f} revolute joint pose = {adapter.getJointsState([('cartpole','cartpole_joint')])[:,:,0]}")

        # print(f"link states = {adapter.getLinksState()}")
        # print(f"cart link state = {adapter.getLinksState([('cartpole','base_link')])}")
        adapter.getRenderings

    adapter.destroy_scenario()


from adarl.adapters.MjxAdapter import MjxAdapter
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import jax
test_sim_adapter(MjxAdapter(vec_size=10,
                            enable_rendering=True,
                            jax_device=jax.devices("gpu")[0],
                            sim_step_dt=2/1024,
                            step_length_sec=100/1024,
                            realtime_factor=-1))

