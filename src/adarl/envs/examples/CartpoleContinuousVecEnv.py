#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Any
import adarl.utils.dbg.ggLog as ggLog

from adarl.envs.vec.ControlledVecEnv import ControlledVecEnv
import adarl
from adarl.utils.utils import Pose, build_pose, JointState, to_string_tensor, quat_xyzw_between_vecs_py, quat_mul_xyzw
from adarl.adapters.BaseVecSimulationAdapter import BaseVecSimulationAdapter, ModelSpawnDef
# from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.BaseVecJointImpedanceAdapter import BaseVecJointImpedanceAdapter
from adarl.adapters.BaseVecJointEffortAdapter import BaseVecJointEffortAdapter
from adarl.adapters.VecSimJointImpedanceAdapterWrapper import VecSimJointImpedanceAdapterWrapper
import torch as th
from adarl.utils.spaces import ThBox, gym_spaces
from adarl.utils.tensor_trees import space_from_tree
from typing_extensions import override
from pathlib import Path
import adarl.utils.utils
from adarl.utils.dbg.dbg_checks import dbg_check_finite, dbg_check

gym_inverted_pendulum_model = """
<mujoco model="inverted pendulum">
	<compiler inertiafromgeom="true"/>
	<default>
		<joint armature="0" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
		<tendon/>
		<motor ctrlrange="-3 3"/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
	<size nstack="3000"/>
	<worldbody>
		<!--geom name="ground" type="plane" pos="0 0 0" /-->
		<geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
		<body name="cart" pos="0 0 0">
			<joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
			<geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
			<body name="pole" pos="0 0 0">
				<joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" type="hinge"/>
				<geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
				<!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
			</body>
		</body>
	</worldbody>
	<actuator>
		<motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
	</actuator>
</mujoco>"""
class CartpoleContinuousVecEnv(ControlledVecEnv):

    def __init__(   self,
                    adapter : BaseVecJointEffortAdapter | BaseVecJointImpedanceAdapter,
                    max_episode_steps : int = 500,
                    render : bool = False,
                    step_duration_sec : float = 0.05,
                    wall_sim_speed = False,
                    seed = 1,
                    th_device : th.device = th.device("cpu"),
                    task : str = "balance",
                    sparse_reward = True,
                    terminate_on_rail_distance = False,
                    terminate_on_pole_angle = True,
                    use_gym_inverted_pendulum_model = False,
                    camera_offset_xyz = (.0,.0,.0)):
        print(f"Creating CartpoleContinuousVecEnv with camera_offset_xyz {camera_offset_xyz}")
        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._renderingEnabled = render
        self._gym_inverted_pendulum = use_gym_inverted_pendulum_model
        self._task = task
        self._sparse_reward = sparse_reward
        self._terminate_on_rail_distance = th.as_tensor(terminate_on_rail_distance, device=th_device)
        self._terminate_on_pole_angle = th.as_tensor(terminate_on_pole_angle, device=th_device)

        self._ui_camera_name = "simple_camera"
        self._camera_link_name= ("simple_camera", "simple_camera_link")
        cam_dist = 3.0 if self._gym_inverted_pendulum else 3.5
        self._camera_pose = [0.,-cam_dist,0.3,0.,0.,0.707,0.707]
        self._apply_camera_offset(camera_offset_xyz)
        self._upright_hinge_threshold = 0.2 # like gym's InvertedPendulum
        self._max_cart_dist = 2
        self._init_noise_scale = 0.01
        self._force_range = 3.0 * 100 if use_gym_inverted_pendulum_model else 50.0
        self._adapter : BaseVecJointImpedanceAdapter | BaseVecJointEffortAdapter
        
        single_state_space, single_observation_space, single_reward_space = self._build_spaces(th_device)

        act_max = np.array([1.0])
        super().__init__(th_device=th_device,
                         seed=seed,
                         obs_dtype=th.float32,
                         single_action_space = ThBox(-act_max,act_max, torch_device=th_device),
                         single_observation_space = single_observation_space,
                         single_state_space=single_state_space,
                         single_reward_space=single_reward_space,
                         info_space=None, #type: ignore : Will be set later
                         step_duration_sec=step_duration_sec,
                         adapter=adapter,
                         max_episode_steps=max_episode_steps)
        example_labels : dict[str,th.Tensor] = {}
        example_state = {k:th.as_tensor((s.low+s.high)/2).to(device=th_device).unsqueeze(0).repeat(self.num_envs, *([1]*len(s.shape)) ) for k,s in single_state_space.spaces.items()}
        example_infos = self.get_infos(example_state, example_labels)
        self.info_space = space_from_tree(example_infos, example_labels) # needs to be done afer super()__init__

        self._build()
        self._adapter.startup()
        self.initialize_episodes()
        self._success = False

    def _build_spaces(self, th_device):
        self._CART_POS = 0
        self._CART_VEL = 1
        self._POLE_SIN = 2
        self._POLE_COS = 3
        self._POLE_VEL = 4
        self._TIMESTEP = 5

        state_vec_max = th.as_tensor([  2.5, # cart position
                                        np.finfo(np.float32).max, # cart velocity
                                        1.0, # pole angle sin
                                        1.0, # pole angle cos
                                        np.finfo(np.float32).max, # pole joint velocity
                                        float("+inf")], # timestep
                                    device=th_device)
        state_vec_labels = ["cart_pos","cart_vel","pole_pos_sin", "pole_pos_cos","pole_vel","timestep"]
        single_vec_state_space = ThBox(-state_vec_max,state_vec_max,
                                labels=to_string_tensor(state_vec_labels),
                                torch_device=th_device)
        single_observation_space = ThBox(-state_vec_max[:5],state_vec_max[:5],
                                             labels=single_vec_state_space.labels[:-1],
                                             torch_device=th_device)
        states_dict = {"vec" : single_vec_state_space,
                       "jstats" : ThBox(low=float("-inf"),
                                    high=float("+inf"),
                                    shape=(4,2,5),
                                    torch_device=th_device),
                       "lstats" : ThBox(low=float("-inf"),
                                    high=float("+inf"),
                                    shape=(4,2,6),
                                    torch_device=th_device),
                       "lvels" : ThBox(low=float("-inf"),
                                       high=float("+inf"),
                                       shape=(2,3),
                                       torch_device=th_device)}
        state_space = gym_spaces.Dict(states_dict) #type: ignore : gym Dict space uses dict instead of Mapping


        if self._task == "center_2r":
            reward_space = ThBox(low=th.as_tensor([0.0, 0.0], device=th_device),
                                 high=th.as_tensor([float("+inf"), float("+inf")], device=th_device),
                                 shape=(2,),
                                 torch_device=th_device)
        else:
            reward_space = ThBox(low=float("-inf"),high=float("+inf"), shape=tuple(), torch_device=th_device)

        return state_space, single_observation_space, reward_space
    
    def _apply_camera_offset(self, offset_xyz : tuple[float,float,float]):
        if offset_xyz == (0.0, 0.0, 0.0):
            return # avoid unnecessary numerical differences
        original_cam_pose = th.as_tensor(self._camera_pose)
        self._camera_pose[0] += offset_xyz[0]
        self._camera_pose[1] += offset_xyz[1]
        self._camera_pose[2] += offset_xyz[2]

        # We do as if we rotated around the origin
        correction_quat = quat_xyzw_between_vecs_py(original_cam_pose[0:3], th.as_tensor(self._camera_pose[0:3]))
        new_cam_quat = quat_mul_xyzw(correction_quat, original_cam_pose[3:7])
        self._camera_pose[3:7] = new_cam_quat.tolist()

    @override
    def submit_actions(self, actions : th.Tensor) -> None:
        dbg_check_finite(actions, async_assert=True)
        # ggLog.info(f"Submitting actions {actions}")
        force_command = th.clamp(actions, -1, 1)*self._force_range
        # ggLog.info(f"Applying force command {force_command}")
        # self._adapter.setJointsEffortCommand(   joint_names = (self._rail_joint,), 
        #                                         efforts = force_command.expand(self.num_envs, 1))
        if isinstance(self._adapter, BaseVecJointImpedanceAdapter):            
            jimp_cmd = self._thzeros((self.num_envs,1,5))
            jimp_cmd[:,:,2] = force_command
            self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_cmd)
        elif isinstance(self._adapter, BaseVecJointEffortAdapter):
            self._adapter.setJointsEffortCommand(   joint_names = (self._rail_joint,), 
                                                    efforts = force_command.expand(self.num_envs, 1))
        else:
            raise RuntimeError(f"Unsupported adapter type {type(self._adapter)}")
        dbg_check_finite(self._adapter.getJointsState(), async_assert=True)
        # ggLog.info(f"Action submitted")

    def pre_step(self):
        # ggLog.info(f"Pre-step")
        dbg_check_finite(self._adapter.getJointsState(), async_assert=True)
        return super().pre_step()
    
    def post_step(self):
        # ggLog.info(f"Post-step")
        dbg_check_finite(self._adapter.getJointsState(), async_assert=True)
        # ggLog.info(f"Step {self.get_ep_step_counter()}")
        return super().post_step()
    
    @override
    def are_states_terminal(self, states : dict[str,th.Tensor]) -> th.Tensor:
        if self._task == "balance":
            
            vstates = states["vec"]            
            pole_angle = th.atan2(states["vec"][:,self._POLE_SIN],states["vec"][:,self._POLE_COS])
            cart_pos = vstates[:,self._CART_POS]

            too_far = th.logical_and(th.abs(cart_pos) > self._max_cart_dist, self._terminate_on_rail_distance)
            too_slanted = th.logical_and(th.abs(pole_angle) > self._upright_hinge_threshold, self._terminate_on_pole_angle)

            return th.logical_or(too_far, too_slanted)
        else:
            return th.zeros((self.num_envs,), dtype=th.bool, device=self._th_device)
    
    @override
    def are_states_timedout(self, states : dict[str,th.Tensor]) -> th.Tensor:
        return states["vec"][:,self._TIMESTEP] >= self.get_max_episode_steps()

    @override
    def compute_rewards(self,   states : dict[str,th.Tensor],
                                sub_rewards_return : dict[str,th.Tensor] = {}) -> th.Tensor:
        
        pole_angle = th.abs(th.atan2(states["vec"][:,self._POLE_SIN],states["vec"][:,self._POLE_COS]))
        cart_pos = states["vec"][:,self._CART_POS]
        centering_reward = 0.1*(1-th.clamp(th.abs(cart_pos)/2, min=0, max=1))
        upright_reward = 1-th.abs(pole_angle/th.pi)
        is_upright = (pole_angle < self._upright_hinge_threshold)*1
        is_centered = (th.abs(cart_pos) < 0.05)*1
        sub_rewards_return["upright"] = upright_reward
        sub_rewards_return["centering"] = centering_reward
        sub_rewards_return["is_upright"]  = is_upright
        sub_rewards_return["is_centered"] = is_centered
        if self._task.startswith("center"):
            if self._task == "center_2r":
                if self._sparse_reward:
                    return th.stack([is_centered, is_upright], dim=1)
                else:    
                    return th.stack([centering_reward, upright_reward], dim=1)
            elif self._task == "center_1r":
                if self._sparse_reward:
                    return is_centered*is_upright
                else:
                    return centering_reward + upright_reward        
        elif self._task == "balance":
            if self._sparse_reward:
                return is_upright
            else:
                return upright_reward
        elif self._task == "swingup":
            if self._sparse_reward:
                return is_upright
            else:
                return upright_reward
        else:
            raise RuntimeError(f"unknown task {self._task}")

    @override
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options = {}) -> None:
        # ggLog.info(f"initializing eps {vec_mask}")
        if isinstance(self._adapter, BaseVecSimulationAdapter):
            if self._task == "balance" or self._task == "center_2r" or self._task == "center_1r":
                joint_states_pve = self._thzeros((self.num_envs,2,3))
                joint_states_pve[:,:,0].uniform_(-self._init_noise_scale, self._init_noise_scale, generator=self._rng) # cart pos
                # joint_states_pve=th.normal(mean=th.zeros((self.num_envs,2,3),
                #                                          device=self._th_device, dtype=th.float32),
                #                             std=th.as_tensor([self._init_noise_scale, 0.0, 0.0],
                #                                              device=self._th_device, dtype=th.float32).expand(self.num_envs, 2, 3),
                #                             generator=self._rng)                
            elif self._task == "swingup":
                joint_states_pve=self._thrandn((self.num_envs,2,3))*self._thtens([0.1, 0.0, 0.0])+self._thtens([[0.0, 0.0, 0.0],[th.pi, 0.0, 0.0]])
                # joint_states_pve=self._thrand((self.num_envs,2,3))*th.as_tensor([2*th.pi, 0.0, 0.0])           
            else:
                raise NotImplementedError()     
            self._adapter.setJointsStateDirect( joint_names=(self._rail_joint, self._hinge_joint),
                                                joint_states_pve=joint_states_pve,
                                                vec_mask=vec_mask)
            self._adapter.setLinksStateDirect([self._camera_link_name],
                                            link_states_pose_vel=th.as_tensor(self._camera_pose + [0,0,0,0,0,0]).expand(self.num_envs, 1, 13),
                                            vec_mask=vec_mask)
        else:
            raise NotImplementedError()
        if isinstance(self._adapter, BaseVecJointImpedanceAdapter):
            self._adapter.reset_joint_impedances_commands()
            start_command = self._thzeros((self.num_envs,1,5))
            self._adapter.setJointsImpedanceCommand(joint_impedances_pvesd = start_command, vec_mask=None)
            self._adapter.set_current_joint_impedance_command(joint_impedances_pvesd = start_command, vec_mask=None)
        elif isinstance(self._adapter, BaseVecJointEffortAdapter):
            self._adapter.setJointsEffortCommand(   joint_names = (self._rail_joint,),
                                                    efforts = self._thzeros((self.num_envs, 1)),
                                                    vec_mask=vec_mask)
        else:
            raise RuntimeError(f"Unsupported adapter type {type(self._adapter)}")

    @override
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        # if th.any(vec_mask[1:]):
        #     raise RuntimeError(f"Can only render env #0 (because the camera can only be at one position across all sims)")
        try:
            imgs, times = self._adapter.getRenderings([self._ui_camera_name], vec_mask=vec_mask)
            # ggLog.info(f"got renderings with shapes {[img.shape for img in imgs]} and times {times}")
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
        hinge_pos = jstate_vec_j_pve[:,1,0]
        vec_state = th.stack([
            jstate_vec_j_pve[:,0,0], # cart_pos
            jstate_vec_j_pve[:,0,1], # cart_vel
            th.sin(hinge_pos), # hinge_sin
            th.cos(hinge_pos), # hinge_cos
            jstate_vec_j_pve[:,1,1], # hinge_vel
            self.get_ep_step_counter() #step
        ], dim = 1)
        state = {"vec" : vec_state}
        dbg_check(lambda: th.isfinite(state["vec"]).all(),
                  lambda: f"Non-finite values in state vec: {state['vec']}")
        joint_step_stats = self._adapter.get_joints_state_step_stats_extended()
        link_step_stats = self._adapter.get_links_state_step_stats()
        state["jstats"] = joint_step_stats
        state["lstats"] = link_step_stats
        lstate = self._adapter.getLinksState()
        state["lvels"] = lstate[:,:,7:10]
        return state

    def _get_spawn_defs(self):
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            cam_file = "models/simple_camera.mjcf.xacro"
        else:            
            cam_file = "models/simple_camera.sdf.xacro"
        
        if self._gym_inverted_pendulum:
            cartpole_model_string = gym_inverted_pendulum_model
            model_format = "mjcf"
            self._rail_joint = ("cartpole_v0","slider")
            self._hinge_joint = ("cartpole_v0","hinge")
            self._cart_link = ("cartpole_v0","cart")
            self._pole_link = ("cartpole_v0","pole")
        else:
            cartpole_model_string = Path(adarl.utils.utils.pkgutil_get_path("adarl","models/cartpole_v0.urdf.xacro")).read_text()
            model_format = "urdf.xacro"
            self._rail_joint = ("cartpole_v0","foot_joint")
            self._hinge_joint = ("cartpole_v0","cartpole_joint")
            self._cart_link = ("cartpole_v0","base_link")
            self._pole_link = ("cartpole_v0","bar_link")

        camera_def = ModelSpawnDef( definition_string=Path(adarl.utils.utils.pkgutil_get_path("adarl",cam_file)).read_text(),
                                    name="simple_camera",
                                    pose=None,
                                    format="sdf.xacro",
                                    kwargs={"camera_width":426,
                                            "camera_height":240,
                                            "frame_rate":1/self._intendedStepLength_sec})
        cartpole_def = ModelSpawnDef(definition_string=cartpole_model_string,
                                        name="cartpole_v0",
                                        pose=None,
                                        format=model_format,
                                        kwargs={"use_collisions" : "false"})
        return [cartpole_def, camera_def]
    
    @override
    def _build(self):
        envCtrlName = type(self._adapter).__name__
        if adarl.utils.utils.isinstance_noimport(self._adapter, "MjxAdapter"):
            self._adapter.build_scenario(models =self._get_spawn_defs())
        elif isinstance(self._adapter, VecSimJointImpedanceAdapterWrapper):
            for subadapter in self._adapter.sub_adapters():
                if adarl.utils.utils.isinstance_noimport(subadapter, ("PyBulletJointImpedanceAdapter")):
                    self._adapter.build_scenario(models = self._get_spawn_defs())
                elif adarl.utils.utils.isinstance_noimport(subadapter, ("RosXbotAdapter", "RosXbotGazeboAdapter")):
                    self._adapter.build_scenario(launch_file_pkg_and_path = adarl.utils.utils.pkgutil_get_path( "adarl_envs",
                                                                                                                "gazebo/all_gazebo_xbot.launch"),
                                                launch_file_args={"gui":"false"})
                else:
                    raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        else:
            raise NotImplementedError("Adapter "+envCtrlName+" is not supported")
        
        
        self._adapter.set_monitored_joints([self._rail_joint, self._hinge_joint])
        self._adapter.set_monitored_links([self._cart_link, self._pole_link])
        if isinstance(self._adapter, BaseVecJointImpedanceAdapter):
            self._adapter.set_impedance_controlled_joints([self._rail_joint])
        # if self._renderingEnabled:
        #     self._adapter.set_monitored_cameras(["camera"])

    @override
    def close(self):
        self._adapter.destroy_scenario()

    @override
    def get_infos(self,states, labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        vstates = states["vec"]
        step_count = states["vec"][:,self._TIMESTEP]
        sub_rewards : dict[str,th.Tensor] = {}
        reward = self.compute_rewards(states, sub_rewards)
        info =  {"pole_angle" : th.atan2(vstates[:,self._POLE_SIN],vstates[:,self._POLE_COS]),
                "cart_pos" : vstates[:,self._CART_POS],
                "step_count" : step_count,
                "reward" : reward,
                "jstats_avg" : states["jstats"][:,2,:,0],
                "lstats_avg" : states["lstats"][:,2,:,0],
                "lvels" : states["lvels"].reshape(self.num_envs,2*3),
                "hinge_vel" : vstates[:,self._POLE_VEL],
                "slider_vel" : vstates[:,self._CART_VEL]}
        info["obs"] = self.get_observations(states)
        if labels is not None:
            labels["obs"] = self.single_observation_space.labels
        info.update({"reward_"+k:v for k,v in sub_rewards.items()})
        return info