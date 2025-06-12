#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any
import adarl.utils.dbg.ggLog as ggLog

from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
import adarl
from adarl.utils.utils import Pose, build_pose, JointState, to_string_tensor
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
# from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
import torch as th
from adarl.utils.spaces import ThBox, gym_spaces
from adarl.utils.tensor_trees import space_from_tree
from typing_extensions import override
from pathlib import Path
import adarl.utils.utils

class CartpoleContinuousVecEnv(ControlledVecEnv):

    def __init__(   self,
                    adapter : BaseVecJointImpedanceAdapter, # could be made into BaseVecJointEffortAdapter, but need to use setJointEffortCommand
                    max_episode_steps : int = 500,
                    render : bool = False,
                    step_duration_sec : float = 0.05,
                    wall_sim_speed = False,
                    seed = 1,
                    th_device : th.device = th.device("cpu"),
                    task : str = "balance",
                    sparse_reward = True):
        """
        """

        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._renderingEnabled = render
        self._ui_camera_name = "simple_camera"
        self._task = task
        self._sparse_reward = sparse_reward
        self._upright_hinge_threshold = 3.14159/180*2
        self._adapter : BaseVecJointImpedanceAdapter
        
        self._CART_POS = 0
        self._CART_VEL = 1
        self._POLE_SIN = 2
        self._POLE_COS = 3
        self._POLE_VEL = 4
        self._TIMESTEP = 5

        state_vec_max = th.as_tensor([  2.5 * 2, # cart position
                                        np.finfo(np.float32).max, # cart velocity
                                        1.0, # pole angle sin
                                        1.0, # pole angle cos
                                        np.finfo(np.float32).max, # pole joint velocity
                                        float("+inf")], # timestep
                                    device=th_device)
        state_vec_labels = ["cart_pos","cart_vel","pole_pos","pole_vel","timestep"]
        vec_state_space = ThBox(-state_vec_max,state_vec_max,
                                labels=to_string_tensor(state_vec_labels),
                                torch_device=th_device)
        single_observation_space = ThBox(-state_vec_max[:5],state_vec_max[:5],
                                             labels=to_string_tensor(state_vec_labels[:-1]),
                                             torch_device=th_device)
        
        states_dict = {"vec" : vec_state_space}
        state_space = gym_spaces.Dict(states_dict) #type: ignore : gym Dict space uses dict instead of Mapping

        act_max = np.array([1.0])
        super().__init__(th_device=th_device,
                         seed=seed,
                         obs_dtype=th.float32,
                         single_action_space = ThBox(-act_max,act_max, torch_device=th_device),
                         single_observation_space = single_observation_space,
                         single_state_space=state_space,
                         single_reward_space=ThBox(low=float("-inf"),high=float("+inf"), shape=tuple(), torch_device=th_device),
                         info_space=None, #type: ignore : Will be set later
                         step_duration_sec=step_duration_sec,
                         adapter=adapter,
                         max_episode_steps=max_episode_steps)
        example_labels : dict[str,th.Tensor] = {}
        example_state = {k:th.as_tensor((s.low+s.high)/2).to(device=th_device).unsqueeze(0) for k,s in states_dict.items()}
        example_infos = self.get_infos(example_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__

        self._build()
        self._adapter.startup()
        self.initialize_episodes()
        self._success = False

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        # self._adapter.setJointsEffortCommand(   joint_names = (("cartpole_v0","foot_joint"),), 
        #                                         efforts = (actions*20).expand(self.num_envs, 1, 1))
        # ggLog.info(f"Submitting actions {actions}")
        jimp_cmd = self._thzeros((self.num_envs,1,5))
        jimp_cmd[:,:,2] = th.clamp(actions, -1, 1)*50
        self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_cmd)
        # ggLog.info(f"Sending cmd {actions}")

    def post_step(self):
        # ggLog.info(f"Step {self.get_ep_step_counter()}")
        return super().post_step()
    
    @override
    def are_states_terminal(self, states : dict[str,th.Tensor]) -> th.Tensor:
        if self._task == "balance":
            maxCartDist = 2
            maxPoleAngle = 0.261791667 #15 degrees
            vstates = states["vec"]            
            pole_angle = th.atan2(states["vec"][:,self._POLE_SIN],states["vec"][:,self._POLE_COS])
            return th.logical_or(th.abs(vstates[:,self._CART_POS]) > maxCartDist, th.abs(pole_angle) > maxPoleAngle)
        else:
            return th.zeros((self.num_envs,), dtype=th.bool, device=self._th_device)
    
    @override
    def are_states_timedout(self, states : dict[str,th.Tensor]) -> th.Tensor:
        return states["vec"][:,self._TIMESTEP] >= self.get_max_episode_steps()

    @override
    def compute_rewards(self,   states : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        if not self._sparse_reward:
            pole_angle = th.atan2(states["vec"][:,self._POLE_SIN],states["vec"][:,self._POLE_COS])
            up_reward = 1-th.abs(pole_angle/th.pi)
            sub_rewards_return["up_reward"] = up_reward
            return up_reward
        else:
            if self._task == "balance":
                health_reward = th.ones((self.num_envs,), device=self._th_device, dtype=th.float32)
                sub_rewards_return["health"] = health_reward
                return health_reward
            elif self._task == "swingup":
                pole_angle = th.atan2(states["vec"][:,self._POLE_SIN],states["vec"][:,self._POLE_COS])
                upright = pole_angle < self._upright_hinge_threshold
                sub_rewards_return["upright"] = upright
                return upright
            else:
                raise RuntimeError(f"unknown task {self._task}")

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        # ggLog.info(f"initializing eps {vec_mask}")
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            if self._task == "balance":
                joint_states_pve=th.normal(mean=th.zeros((self.num_envs,2,3), device=self._th_device, dtype=th.float32),
                                            std=th.as_tensor([0.02, 0.0, 0.0], device=self._th_device, dtype=th.float32).expand(self.num_envs, 2, 3),
                                            generator=self._rng)
            elif self._task == "swingup":
                joint_states_pve=self._thrandn((self.num_envs,2,3))*self._thtens([0.1, 0.0, 0.0])+self._thtens([[0.0, 0.0, 0.0],[th.pi, 0.0, 0.0]])
                # joint_states_pve=self._thrand((self.num_envs,2,3))*th.as_tensor([2*th.pi, 0.0, 0.0])                
            self._adapter.setJointsStateDirect( joint_names=(("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")),
                                                joint_states_pve=joint_states_pve)
        else:
            raise NotImplementedError()
        self._adapter.setLinksStateDirect([("simple_camera", "simple_camera_link")],
                                          link_states_pose_vel=th.as_tensor([0.0,-3.0,0.3,0.0,0.0,0.707,0.707,0,0,0,0,0,0]).expand(self.num_envs, 1, 13))
        self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = self._thzeros((self.num_envs,1,5)))

    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        # if th.any(vec_mask[1:]):
        #     raise RuntimeError(f"Can only render env #0 (because the camera can only be at one position across all sims)")
        try:
            imgs, times = self._adapter.getRenderings([self._ui_camera_name], vec_mask=vec_mask)
            return imgs, times
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return [], th.empty((0,))


    @override
    def get_observations(self, state) -> dict[Any, th.Tensor]:
        return state["vec"][:,:-1]

    @override
    def get_states(self) -> dict[str,th.Tensor]:
        jstate_vec_j_pve : th.Tensor = self._adapter.getJointsState()
        vec_state = th.stack([
            jstate_vec_j_pve[:,0,0], # cart_pos
            jstate_vec_j_pve[:,0,0], # cart_vel
            th.sin(jstate_vec_j_pve[:,1,0]), # hinge_sin
            th.cos(jstate_vec_j_pve[:,1,0]), # hinge_cos
            jstate_vec_j_pve[:,1,1], # hinge_vel
            self.get_ep_step_counter() #step
        ], dim = 1)
        state = {"vec" : vec_state}
        return state

    def _get_spawn_defs(self):
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            cam_file = "models/simple_camera.mjcf.xacro"
        else:            
            cam_file = "models/simple_camera.sdf.xacro"
        camera_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                    name="simple_camera",
                                    pose=None,
                                    format="sdf.xacro",
                                    kwargs={"camera_width":426,
                                            "camera_height":240,
                                            "frame_rate":1/self._intendedStepLength_sec})
        cartpole_def = ModelSpawnDef(definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text(),
                                                        name="cartpole_v0",
                                                        pose=None,
                                                        format="urdf.xacro",
                                                        kwargs={"use_collisions" : "false"})
        return [cartpole_def, camera_def]
    
    @override
    def _build(self):
        envCtrlName = type(self._adapter).__name__
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            self._adapter.build_scenario(models =self._get_spawn_defs())
        elif isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
            if adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("PyBulletJointImpedanceAdapter")):
                self._adapter.build_scenario(models = self._get_spawn_defs())
            elif adarl.utils.utils.isinstance_noimport(self._adapter.sub_adapter(), ("RosXbotAdapter", "RosXbotGazeboAdapter")):
                self._adapter.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path( "adarl_envs",
                                                                                                            "gazebo/all_gazebo_xbot.launch"),
                                            launch_file_args={"gui":"false"})
            else:
                raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        
        self._adapter.set_monitored_joints([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        self._adapter.set_impedance_controlled_joints([("cartpole_v0","foot_joint")])
        # if self._renderingEnabled:
        #     self._adapter.set_monitored_cameras(["camera"])

    @override
    def close(self):
        self._adapter.destroy_scenario()

    @override
    def get_infos(self,states, labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        vstates = states["vec"]
        return {"pole_angle" : th.atan2(vstates[:,2],vstates[:,3])}
