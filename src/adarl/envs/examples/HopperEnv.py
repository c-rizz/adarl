#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""

import adarl.utils.spaces as spaces
import numpy as np
from typing import Tuple, Dict, Any

from adarl.envs.ControlledEnv import ControlledEnv
from adarl.adapters.BaseAdapter import BaseAdapter
from adarl.adapters.PyBulletAdapter import PyBulletAdapter
#import tf2_py
import adarl.utils
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import Pose, build_pose
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter

class HopperEnv(ControlledEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    """

    
    metadata = {'render.modes': ['rgb_array']}


    POS_Z_OBS = 0
    TARGET_DIRECTION_COSINE_OBS = 1
    TARGET_DIRECTION_SINE_OBS = 2
    VEL_X_OBS = 3
    VEL_Y_OBS = 4
    VEL_Z_OBS = 5
    TORSO_ROLL_OBS = 6
    TORSO_PITCH_OBS = 7
    TORSO_THIGH_JOINT_POS_OBS = 8
    TORSO_THIGH_JOINT_VEL_OBS = 9
    THIGH_LEG_JOINT_POS_OBS = 10
    THIGH_LEG_JOINT_VEL_OBS = 11
    LEG_FOOT_JOINT_POS_OBS = 12
    LEG_FOOT_JOINT_VEL_OBS = 13
    CONTACT_OBS = 14
    AVG_X_POS = 15
    PREV_AVG_X_POS = 16

    MAX_TORQUE = 75

    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    simulatorController : BaseAdapter = None,
                    startSimulation : bool = True,
                    simulationBackend : str = "gazebo",
                    useMjcfFile : bool = False,
                    seed = 1):
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
        simulatorController : BaseAdapter
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """

        self._envSeed = seed
        self._useMjcfFile = useMjcfFile
        self._spawned = False
        action_high = np.array([1, 1, 1])        
        # Observations are:
        #  (pos_z, torso_thigh_joint_pos, thigh_leg_joint_pos, leg_foot_joint_pos, vel_x, vel_y, vel_z, torso_thigh_joint_vel, thigh_leg_joint_vel, leg_foot_joint_vel)
        obs_high = np.full((15), 100.0, dtype=np.float32)
        state_high = np.full((17), 100.0, dtype=np.float32)

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = simulatorController,
                         startSimulation = startSimulation,
                         action_space = spaces.gym_spaces.Box(low=-action_high, high=action_high, dtype=np.float32),
                         observation_space = spaces.gym_spaces.Box(-obs_high, obs_high),
                         state_space=spaces.gym_spaces.Box(-state_high, state_high))

        #print("HopperEnv: action_space = "+str(self.action_space))
        #print("HopperEnv: action_space = "+str(self.action_space))
        self._adapter.set_monitored_joints([("hopper","torso_to_thigh"),
                                                        ("hopper","thigh_to_leg"),
                                                        ("hopper","leg_to_foot"),
                                                        ("hopper","torso_pitch_joint")])

        self._adapter.set_monitored_links([("hopper","torso"),("hopper","thigh"),("hopper","leg"),("hopper","foot")])

        self._stepLength_sec = stepLength_sec
        self._renderingEnabled = render
        self._success = False
        self._success_ratio_avglen = 50
        self._successes = [1]*self._success_ratio_avglen
        self._tot_episodes = 0
        self._success_ratio = 0
        if self._renderingEnabled:
            self._adapter.set_monitored_cameras(["camera"])

        self._adapter.startup()

    def submitAction(self, action : np.ndarray) -> None:
        super().submitAction(action)

        if action.size!=3:
            raise AttributeError("Action must have length 3, but action="+str(action))

        unnormalizedAction = (  float(np.clip(action[0],-1,1))*self.MAX_TORQUE,
                                float(np.clip(action[1],-1,1))*self.MAX_TORQUE,
                                float(np.clip(action[2],-1,1))*self.MAX_TORQUE)
        self._adapter.setJointsEffortCommand([ ("hopper","torso_to_thigh",unnormalizedAction[0]),
                                                    ("hopper","thigh_to_leg",unnormalizedAction[1]),
                                                    ("hopper","leg_to_foot",unnormalizedAction[2])])


    def reachedTerminalState(self, previousState : Tuple[float,float,float,float,float,float,float,float,float,float],
                         state : Tuple[float,float,float,float,float,float,float,float,float,float]) -> bool:
        done = False
        if super().reachedTerminalState(previousState, state):
            # ggLog.info("Episode terminated: superclass reasons")
            done |= True

        if state[self.AVG_X_POS]<-0.5 or state[self.POS_Z_OBS] <= -0.45 or abs(state[self.TORSO_PITCH_OBS]) >= 1.0 :
            # ggLog.info("Episode terminated: terminal hopper state")
            done |= True
        #rospy.loginfo("height = {:1.4f}".format(torso_z_displacement)+"\t pitch = {:1.4f}".format(torso_pitch)+"\t done = "+str(done))
        if done:
            self._success = state[self.AVG_X_POS] > 5
            self._successes[self._tot_episodes % self._success_ratio_avglen] = self._success
            self._success_ratio = sum(self._successes)/len(self._successes)
        return done


    def computeReward( self,
                        previousState : Tuple[float,float,float,float,float,float,float,float,float,float],
                        state : Tuple[float,float,float,float,float,float,float,float,float,float],
                        action : Tuple[float,float,float],
                        env_conf = None) -> float:
        if not self.reachedTerminalState(previousState, state):
            speed = (state[15] - state[16])/self._stepLength_sec
            # print("Speed: "+str(speed))
            return 1 + 2*speed - 0.003*(action[0]*action[0] + action[1]*action[1] + action[2]*action[2]) # should be more or less the same as openai's hopper_v3
        else:
            return -1


    def initializeEpisode(self) -> None:
        if not self._spawned and isinstance(self._adapter, BaseSimulationAdapter):
            self._adapter.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("adarl","models/hopper_v1.urdf.xacro"),
                                                    model_name="hopper",
                                                    pose=build_pose(0,0,0,0,0,0,1),
                                                    model_kwargs={"camera_width":"213","camera_height":"120"})
            self._spawned = True
        self._adapter.setJointsEffortCommand([  ("hopper","torso_to_thigh",0),
                                                       ("hopper","thigh_to_leg",0),
                                                       ("hopper","leg_to_foot",0)])



    def getObservation(self, state) -> np.ndarray:
        return state[0:-2]

    def getState(self) -> np.ndarray:
        """Get an observation of the environment.

        Returns
        -------
        np.ndarray
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        jointStates = self._adapter.getJointsState([("hopper","torso_to_thigh"),
                                                                  ("hopper","thigh_to_leg"),
                                                                  ("hopper","leg_to_foot"),
                                                                  ("hopper","torso_pitch_joint")])
        linksState = self._adapter.getLinksState([("hopper","torso"),
                                                                ("hopper","thigh"),
                                                                ("hopper","leg"),
                                                                ("hopper","foot")])

        #print("type linksState[()]= "+ str(type(linksState[("hopper","torso")])))
        avg_pos_x = (   linksState[("hopper","torso")].pose.position[0] +
                        linksState[("hopper","thigh")].pose.position[0] +
                        linksState[("hopper","leg")].pose.position[0]   +
                        linksState[("hopper","foot")].pose.position[0])/4

        torso_pose = linksState[("hopper","torso")].pose

        if self._actionsCounter == 0:
            self._initial_torso_z = torso_pose.position[2]
            self._previousAvgPosX = avg_pos_x

        # print("frame "+str(self._framesCounter)+"\t initial_torso_z = "+str(self._initial_torso_z)+"\t torso_z = "+str(linksState[("hopper","torso")].pose.position.z))
        # time.sleep(1)
        #avg_vel_x = (avg_pos_x - self._previousAvgPosX)/self._stepLength_sec
        #(r,p,y) = tf.transformations.euler_from_quaternion([torso_pose.orientation.x, torso_pose.orientation.y, torso_pose.orientation.z, torso_pose.orientation.w])
        #print("torsoState = ",torsoState)
        #print("jointStates ",jointStates)
        state = np.array(  [linksState[("hopper","torso")].pose.position[2] - self._initial_torso_z,
                            1, # for pybullet consistency
                            0, # for pybullet consistency
                            linksState[("hopper","torso")].ang_velocity_xyz[0] * 0.3, #0.3 is just to be consistent with pybullet
                            linksState[("hopper","torso")].ang_velocity_xyz[1] * 0.3, #this will always be zero
                            linksState[("hopper","torso")].ang_velocity_xyz[2] * 0.3,
                            0, # roll of torso,for pybullet consistency
                            jointStates[("hopper","torso_pitch_joint")].position[0],
                            jointStates[("hopper","torso_to_thigh")].position[0],
                            jointStates[("hopper","torso_to_thigh")].rate[0],
                            jointStates[("hopper","thigh_to_leg")].position[0],
                            jointStates[("hopper","thigh_to_leg")].rate[0],
                            jointStates[("hopper","leg_to_foot")].position[0],
                            jointStates[("hopper","leg_to_foot")].rate[0],
                            0, # should be it touching the ground or not, not used
                            avg_pos_x,
                            self._previousAvgPosX])
        # print("avg_vel_x = "+str(avg_vel_x))
        # s = ""
        # for oi in state:
        #     s+=" {:0.4f}".format(oi)
        # print("satte = " +s)
        # time.sleep(1)
        self._previousAvgPosX = avg_pos_x

        return state

    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        npArrImg, t = self._adapter.getRenderings(["camera"])["camera"]
        # npArrImg = adarl.utils.utils.ros1_image_to_numpy(imgMsg)
        # t = imgMsg.header.stamp.to_sec()
        return (npArrImg,t)


    def build(self, backend : str = "gazebo"):
        # if backend == "gazebo":
        #     worldpath = "\"$(find adarl_ros)/worlds/ground_plane_world_plugin.world\""
        #     self._adapter.build_scenario(launch_file_pkg_and_path=("adarl_ros","/launch/gazebo_server.launch"),
        #                                                 launch_file_args={  "gui":"false",
        #                                                                     "paused":"true",
        #                                                                     "physics_engine":"ode",
        #                                                                     "limit_sim_speed":"true",
        #                                                                     "world_name":worldpath,
        #                                                                     "gazebo_seed":f"{self._envSeed}",
        #                                                                     "wall_sim_speed":"false"})
        #     # time.sleep(10)
        if isinstance(self._adapter, PyBulletAdapter):
            if self._useMjcfFile:
                self._adapter.build_scenario(adarl.utils.utils.pkgutil_get_path("adarl","models/hopper_mjcf_pybullet.xml"), format = "mjcf")
            else:
                self._adapter.build_scenario(adarl.utils.utils.pkgutil_get_path("adarl","models/hopper_v1.urdf"), format = "urdf")
            self._spawned = True
        else:
            raise NotImplementedError("Backend "+backend+" not supported")

    def _destroy(self):
        self._adapter.destroy_scenario()

    def getInfo(self,state=None) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        i["success"] = self._success
        i["success_ratio"] = self._success_ratio
        i["x_position"] = state[self.AVG_X_POS]
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i



    def performReset(self, options = {}):
        super().performReset()
        self._tot_episodes += 1
        self._success = False