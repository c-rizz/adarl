#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""



import adarl.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any
import adarl.utils.dbg.ggLog as ggLog
import random

from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
import adarl
from adarl.utils.utils import Pose, build_pose, JointState
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
import torch as th
from adarl.utils.spaces import ThBox
from adarl.utils.tensor_trees import space_from_tree
from typing_extensions import override
from typing import Generic
from pathlib import Path
import adarl.utils.utils

class CartpoleContinuousVecEnv(ControlledVecEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""



    def __init__(   self,
                    adapter : BaseVecJointEffortAdapter,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    step_duration_sec : float = 0.05,
                    startSimulation : bool = True,
                    wall_sim_speed = False,
                    seed = 1,
                    th_device : th.device = th.device("cpu")):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        environmentController : BaseAdapter
            Specifies which simulator controller to use. By default it connects to Gazebo


        """

        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._renderingEnabled = render
        self._ui_camera_name = "simple_camera"
        obs_max = np.array([2.5 * 2,
                            np.finfo(np.float32).max,
                            0.7 * 2,
                            np.finfo(np.float32).max])
        state_max = np.array([2.5 * 2, # cart position
                            np.finfo(np.float32).max, # cart velocity
                            0.7 * 2, # pole angle
                            np.finfo(np.float32).max, # pole joint velocity
                            float("+inf")]) # timestep
        act_max = np.array([1.0])
        super().__init__(th_device=th_device,
                         seed=seed,
                         obs_dtype=th.float32,
                         single_action_space = ThBox(-act_max,act_max),
                         single_observation_space = ThBox(-obs_max, obs_max),
                         single_state_space=ThBox(-state_max, state_max),
                         single_reward_space=ThBox(low=float("-inf"),high=float("+inf"), shape=tuple(), torch_device=th_device),
                         info_space=None,
                         step_duration_sec=step_duration_sec,
                         adapter=adapter)
        example_labels : dict[str,th.Tensor] = {}
        example_infos = self.get_infos(self.single_state_space.low, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__

        self._adapter.startup()
        self._success = False

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        # self._adapter.setJointsEffortCommand(   joint_names = (("cartpole_v0","foot_joint"),), 
        #                                         efforts = (actions*20).expand(self.num_envs, 1, 1))
        jimp_cmd = self._thzeros((self.num_envs,1,5))
        jimp_cmd[:,:,2] = th.clamp(actions, -1, 1)*20
        self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_cmd)
        # ggLog.info(f"Sending cmd {actions}")

    def post_step(self):
        # ggLog.info(f"Step {self.get_ep_step_counter()}")
        return super().post_step()
    
    @override
    def are_states_terminal(self, states : th.Tensor) -> th.Tensor:
        maxCartDist = 2
        maxPoleAngle = 0.261791667 #15 degrees
        return th.logical_or(th.abs(states[:,0]) > maxCartDist, th.abs(states[:,2]) > maxPoleAngle)
    
    @override
    def are_states_timedout(self, states : th.Tensor) -> th.Tensor:
        return states[:,4] > self.get_max_episode_steps()

    @override
    def compute_rewards(self,   state : th.Tensor,
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        health_reward = th.ones((self.num_envs,), device=self._th_device, dtype=th.float32)
        if sub_rewards_return is not None:
            sub_rewards_return["health"] = health_reward
        return health_reward

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        # ggLog.info(f"initializing eps {vec_mask}")
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            self._adapter.setJointsStateDirect(joint_names=(("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")),
                                               joint_states_pve=th.normal(mean=th.zeros((self.num_envs,2,3), device=self._th_device, dtype=th.float32),
                                                                          std=th.as_tensor([0.1, 0.0, 0.0], device=self._th_device, dtype=th.float32).expand(self.num_envs, 2, 3),
                                                                          generator=self._rng))
        else:
            raise NotImplementedError()
        self._adapter.setLinksStateDirect([("simple_camera", "simple_camera_link")],
                                          link_states_pose_vel=th.as_tensor([0.0,-2.5,0.7,0.0,0.0,0.707,0.707,0,0,0,0,0,0]).expand(self.num_envs, 1, 13))
        self._adapter.setJointsEffortCommand(   joint_names = (("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")), 
                                                efforts = self._thzeros((self.num_envs,2)))
        

    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        if th.any(vec_mask[1:]):
            raise RuntimeError(f"Can only render env #0 (because the camera can only be at one position across all sims)")
        try:
            imgs, times = self._adapter.getRenderings([self._ui_camera_name], vec_mask=vec_mask)
            return imgs, times
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {adarl.utils.utils.exc_to_str(e)}")
            return [], th.empty((0,))


    @override
    def get_observations(self, state) -> dict[Any, th.Tensor]:
        return state[:,:-1]


    @override
    def get_states(self) -> th.Tensor:
        jstate_vec_j_pve : th.Tensor = self._adapter.getJointsState()
        jstate = jstate_vec_j_pve[:,:,:2].flatten(start_dim=1)
        state = th.cat([jstate, self.get_ep_step_counter().unsqueeze(1)], dim = 1)
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
                                                        kwargs={})
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
        ggLog.info(f"set monitored joints")
        self._adapter.set_impedance_controlled_joints([("cartpole_v0","foot_joint")])
        # if self._renderingEnabled:
        #     self._adapter.set_monitored_cameras(["camera"])

    @override
    def close(self):
        self._adapter.destroy_scenario()

    @override
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> th.Tensor:
        return {}