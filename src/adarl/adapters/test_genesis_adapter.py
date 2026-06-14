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

from adarl.adapters.GenesisAdapter import GenesisAdapter, GenesisCameraDef
from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter


def test_sim_adapter(adapter: BaseVecSimulationAdapter, render: bool, print_state: bool):
    adapter.build_scenario([ModelSpawnDef(name="cartpole",
                                          definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl", "models/cartpole_v0.urdf.xacro")).read_text(),
                                          format="urdf.xacro",
                                          pose=None,
                                          kwargs={}),
                            ModelSpawnDef(name="ball",
                                          definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl", "models/ball.urdf")).read_text(),
                                          format="urdf",
                                          pose=None,
                                          kwargs={})])
    vsize = adapter.vec_size()
    dev = adapter.output_th_device()
    print(f"detected joints = {adapter.get_detected_joints()}")
    print(f"detected links = {adapter.get_detected_links()}")
    print(f"detected cameras = {adapter.get_detected_cameras()}")
    adapter.set_monitored_joints([("cartpole", "cartpole_joint"), ("cartpole", "foot_joint")])
    adapter.set_monitored_links([("cartpole", "bar_link"), ("cartpole", "base_link")])
    adapter.setLinksStateDirect([("ball", "ball")], th.as_tensor([0.0, 3.0, 1.0,
                                                                  0.0, 0.0, 0.0, 1.0,
                                                                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                 device=dev).view(1, 1, 13).expand(vsize, 1, 13))
    adapter.startup()

    print(f"joint states = {adapter.getJointsState()}")
    print(f"link states = {adapter.getLinksState()}")

    adapter.setJointsStateDirect([("cartpole", "cartpole_joint")], th.cat([th.randn(size=(vsize, 1, 1), device=dev) * 0.05,
                                                                           th.zeros(size=(vsize, 1, 2), device=dev)], dim=2))
    print(f"joint states = {adapter.getJointsState()}")

    os.makedirs("test_sim_adapter", exist_ok=True)
    if isinstance(adapter, BaseVecJointImpedanceAdapter):
        adapter.set_impedance_controlled_joints([("cartpole", "cartpole_joint")])
        adapter.set_current_joint_impedance_command(th.as_tensor([[[0.0, 0.0, 0.0, 200.0, 10.0]]], device=dev).expand(vsize, 1, 5))
    t0 = time.monotonic()
    for step in range(100):
        t1 = time.monotonic()
        if isinstance(adapter, BaseVecJointEffortAdapter) and not isinstance(adapter, BaseVecJointImpedanceAdapter):
            a = 50.0 * adapter.getJointsState([("cartpole", "cartpole_joint")])[:, :, 0]
            adapter.setJointsEffortCommand([("cartpole", "foot_joint")], a)
        if step == 1:
            t0 = time.monotonic()  # exclude the first round from the timing stats
        dt = adapter.step()
        if render:
            img = adapter.getRenderings(["simple_camera"])[0][0][0]
            img = imgToCvIntRgb(img, min_val=0, max_val=255)
            r = cv2.imwrite(f"test_sim_adapter/frame-{step}.png", img)
            if not r:
                print("couldn't save image")
        stime = adapter.getEnvTimeFromStartup()
        tf = time.monotonic()
        wtime = tf - t0
        print(f"[{step}] t = {stime:.3f} "
              f"    # tot single rt = {stime/wtime:.5f} "
              f"    # tot vec rt = {(stime * adapter.vec_size())/wtime:.5f} "
              f"    # inst single rt {dt/(tf-t1):.5f}")
        if print_state:
            js = adapter.getJointsState([("cartpole", "cartpole_joint"), ("cartpole", "foot_joint")])[:, :]
            print(f"revolute joint state = {js[:,0,:]}\n"
                  f"linear   joint state = {js[:,1,:]}")
            print(f"joint step stats = {adapter.get_joints_state_step_stats()[0]}")

    adapter.destroy_scenario()


if __name__ == "__main__":
    render = True
    device = th.device("cuda", 0) if th.cuda.is_available() else th.device("cpu")
    test_sim_adapter(GenesisJointImpedanceAdapter(vec_size=4,
                                                  output_th_device=device,
                                                  sim_step_dt=1 / 256,
                                                  step_length_sec=12 / 256,
                                                  enable_rendering=render,
                                                  cameras=[GenesisCameraDef(name="simple_camera",
                                                                            width=480,
                                                                            height=270,
                                                                            pose_xyz=(0.0, 2.0, 0.7),
                                                                            lookat_xyz=(0.0, 0.0, 0.4),
                                                                            fov_deg=60.0)],
                                                  max_joint_impedance_ctrl_torques={("cartpole", "foot_joint"): 100.0,
                                                                                    ("cartpole", "cartpole_joint"): 100.0}),
                     render=render,
                     print_state=False)
