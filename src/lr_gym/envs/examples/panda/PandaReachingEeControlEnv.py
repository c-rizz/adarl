#!/usr/bin/env python3

import numpy as np
from typing import Tuple
import quaternion
import lr_gym.utils.spaces as spaces
from lr_gym.envs.ControlledEnv import ControlledEnv
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym


class PandaReachingEeControlEnv(ControlledEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """

    action_space_high = np.array([  1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1])
    action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high) # 3D translatiomn vector, maximum 10cm


    observation_space_high = np.array([ np.finfo(np.float32).max, # end-effector x position
                                        np.finfo(np.float32).max, # end-effector y position
                                        np.finfo(np.float32).max, # end-effector z position
                                        np.finfo(np.float32).max, # end-effector roll position
                                        np.finfo(np.float32).max, # end-effector pitch position
                                        np.finfo(np.float32).max, # end-effector yaw position
                                        np.finfo(np.float32).max, # joint 1 position
                                        np.finfo(np.float32).max, # joint 2 position
                                        np.finfo(np.float32).max, # joint 3 position
                                        np.finfo(np.float32).max, # joint 4 position
                                        np.finfo(np.float32).max, # joint 5 position
                                        np.finfo(np.float32).max, # joint 6 position
                                        np.finfo(np.float32).max, # joint 7 position
                                        np.finfo(np.float32).max, # flag indicating action fails (zero if there were no fails in last step)
                                        ])

    observation_space = spaces.gym_spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    goalPose : Tuple[float,float,float,float,float,float,float] = (0,0,0, 0,0,0,0),
                    maxStepsPerEpisode : int = 500,
                    goalTolerancePosition : float = 0.05,
                    goalToleranceOrientation_rad : float = 0.0175*5,
                    operatingArea = np.array([[-1, -1, 0], [1, 1, 1.5]]),
                    startSimulation : bool = True,
                    backend="gazebo",
                    environmentController = None,
                    real_robot_ip : str = None):
        """Short summary.

        Parameters
        ----------
        goalPose : Tuple[float,float,float,float,float,float,float]
            end-effector pose to reach (x,y,z, qx,qy,qz,qw)
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        goalTolerancePosition : float
            Position tolerance under which the goal is considered reached, in meters
        goalToleranceOrientation_rad : float
            Orientation tolerance under which the goal is considered reached, in radiants


        """


        self._real_robot_ip = real_robot_ip

        if environmentController is None:                
            raise AttributeError("You must specify environmentController")
        else:
            self._environmentController = environmentController

        super().__init__(   maxStepsPerEpisode = maxStepsPerEpisode,
                            startSimulation = startSimulation,
                            environmentController=self._environmentController,
                            simulationBackend=backend)

        self._camera_name = "simple_camera"
        self._environmentController.setCamerasToObserve([self._camera_name])


        self._initialJointPose = {  ("panda","panda_joint1") : 0.0,
                                    ("panda","panda_joint2") : -3.14159/4,
                                    ("panda","panda_joint3") : 0.0,
                                    ("panda","panda_joint4") : -3.14159/2,
                                    ("panda","panda_joint5") : 0.0,
                                    ("panda","panda_joint6") : 3.14159/4,
                                    ("panda","panda_joint7") : 0.0}

        self._environmentController.setJointsToObserve( [("panda","panda_joint1"),
                                                        ("panda","panda_joint2"),
                                                        ("panda","panda_joint3"),
                                                        ("panda","panda_joint4"),
                                                        ("panda","panda_joint5"),
                                                        ("panda","panda_joint6"),
                                                        ("panda","panda_joint7")])


        self._environmentController.setLinksToObserve( [("panda","panda_link1"),
                                                        ("panda","panda_link2"),
                                                        ("panda","panda_link3"),
                                                        ("panda","panda_link4"),
                                                        ("panda","panda_link5"),
                                                        ("panda","panda_link6"),
                                                        ("panda","panda_link7"),
                                                        ("panda","panda_link8")])

        self._goalPose = goalPose
        self._goalTolerancePosition = goalTolerancePosition
        self._goalToleranceOrientation_rad = goalToleranceOrientation_rad
        self._lastMoveFailed = False
        self._maxPositionChange = 0.1
        self._maxOrientationChange = 45.0/180*3.14159 # 5 degrees

        self._environmentController.startController()

        self._operatingArea = operatingArea #min xyz, max xyz




    def submitAction(self, action : np.typing.NDArray[(6,), np.float32]) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float, float, float, float]
            Relative end-effector movement in cartesian space. It is normalized to the max movement distance, i.e.
            this funciont shoult receive values in the [-1,1] range, which are then converted to the proper
            value range.

        """
        super().submitAction(action)
        #print("received action "+str(action))
        clippedAction = np.clip(np.array(action, dtype=np.float32),-1,1)
        action_xyz = clippedAction[0:3]*self._maxPositionChange
        action_rpy  = clippedAction[3:6]*self._maxOrientationChange
        action_quat = quaternion.from_euler_angles(action_rpy)
        #print("dist action_quat "+str(quaternion.rotation_intrinsic_distance(action_quat,      quaternion.from_euler_angles(0,0,0))))

        currentPose = self.getState()[0:6]
        currentPose_xyz = currentPose[0:3]
        currentPose_rpy = currentPose[3:6]
        currentPose_quat = quaternion.from_euler_angles(currentPose_rpy)

        absolute_xyz = currentPose_xyz + action_xyz
        absolute_quat = action_quat * currentPose_quat
        absolute_quat_arr = np.array([absolute_quat.x, absolute_quat.y, absolute_quat.z, absolute_quat.w])
        unnorm_action = np.concatenate([absolute_xyz, absolute_quat_arr])
        print("moving to "+str(unnorm_action))

        self._environmentController.setCartesianPoseCommand(linkPoses = {("panda","panda_link8") : unnorm_action},
                                                            do_cartesian = False,
                                                            velocity_scaling = 0.5,
                                                            acceleration_scaling = 0.5)
        #rospy.loginfo("Moving Ee of "+str(clippedAction))



    def performStep(self) -> None:
        """Short summary.

        Returns
        -------
        None
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        super().performStep()
        if self._checkGoalReached(self.getState()):
            ggLog.info("Goal Reached")


    def _getDist2goal(self, state : np.typing.NDArray[(15,), np.float32]):
        position = state[0:3]
        orientation_quat = quaternion.from_euler_angles(state[3:6])

        position_dist2goal = np.linalg.norm(position - self._goalPose[0:3])
        goalQuat = quaternion.from_float_array([self._goalPose[6],self._goalPose[3],self._goalPose[4],self._goalPose[5]])
        # print("orientation_quat =",orientation_quat)
        # print("goal_quat =",goalQuat)
        orientation_dist2goal = quaternion.rotation_intrinsic_distance(orientation_quat,goalQuat)

        return position_dist2goal, orientation_dist2goal



    def _checkGoalReached(self,state):
        #print("getting distance for state ",state)
        position_dist2goal, orientation_dist2goal = self._getDist2goal(state)
        #print(position_dist2goal,",",orientation_dist2goal)
        return position_dist2goal < self._goalTolerancePosition and orientation_dist2goal < self._goalToleranceOrientation_rad




    def checkEpisodeEnded(self, previousState : np.typing.NDArray[(15,), np.float32], state : np.typing.NDArray[(15,), np.float32]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True

        if not(np.all(state[0:3] >= self._operatingArea[0]) and np.all(state[0:3] <= self._operatingArea[1])):
            return True
        return False


    def computeReward(self, previousState : np.typing.NDArray[(15,), np.float32], state : np.typing.NDArray[(15,), np.float32], action : int, env_conf = None) -> float:

        posDist, minAngleDist = self._getDist2goal(state)
        mixedDistance = np.linalg.norm([posDist,minAngleDist])

        reward = 1-mixedDistance + 1/(1/100 + mixedDistance)
        if np.isnan(reward):
            raise RuntimeError("Reward is nan! mixedDistance="+str(mixedDistance))

        #rospy.loginfo("Computed reward {:.04f}".format(reward)+"   Distance = "+str(posDist_new))
        return reward


    def initializeEpisode(self) -> None:
        self._environmentController.addCollisionBox(pose_xyz_xyzw = (0,0,-0.501,0,0,0,1),
                                                    size_xyz = (10,10,1),
                                                    attach_link = None,
                                                    reference_frame = "world",
                                                    attach_ignored_links = None)
        moved = False                                                    
        for i in range(5):
            try:
                self._environmentController.moveToJointPoseSync(jointPositions=self._initialJointPose, velocity_scaling=0.9, acceleration_scaling=0.9)
                moved = True
                break
            except Exception as e:
                ggLog.error("Initialization move failed. exception = "+lr_gym.utils.utils.exc_to_str(e))
                #self._actionsFailsInLastStepCounter+=1
                self._environmentController.freerun(duration_sec=1.0)
                ggLog.error(f"Retrying Initialization (i = {i}).")
        if not moved:
            ggLog.error("Failed to move to episode initialization joint pose.")

    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> np.typing.NDArray[(15,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        np.typing.NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        eePose = self._environmentController.getLinksState(requestedLinks=[("panda","panda_link8")])[("panda","panda_link8")].pose
        jointStates = self._environmentController.getJointsState([("panda","panda_joint1"),
                                                                 ("panda","panda_joint2"),
                                                                 ("panda","panda_joint3"),
                                                                 ("panda","panda_joint4"),
                                                                 ("panda","panda_joint5"),
                                                                 ("panda","panda_joint6"),
                                                                 ("panda","panda_joint7")])


        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))





        state = [   eePose.position[0],
                    eePose.position[1],
                    eePose.position[2],
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    jointStates[("panda","panda_joint1")].position[0],
                    jointStates[("panda","panda_joint2")].position[0],
                    jointStates[("panda","panda_joint3")].position[0],
                    jointStates[("panda","panda_joint4")].position[0],
                    jointStates[("panda","panda_joint5")].position[0],
                    jointStates[("panda","panda_joint6")].position[0],
                    jointStates[("panda","panda_joint7")].position[0],
                    self._environmentController.actionsFailsInLastStep()]

        return np.array(state,dtype=np.float32)

    def buildSimulation(self, backend : str = "gazebo"):
        if backend == "gazebo":
            self._environmentController.build_scenario(launch_file_pkg_and_path=("lr_gym_ros","/launch/launch_panda_moveit.launch"),
                                                        launch_file_args={  "gui":"false",
                                                                            "load_gripper":"false"})
        elif backend == "real":
            self._environmentController.build_scenario(launch_file_pkg_and_path=("lr_gym_ros","/launch/launch_panda_moveit.launch"),
                                                        launch_file_args={  "robot_ip":self._real_robot_ip,
                                                                            "simulated":"false",
                                                                            "control_mode":"position"},
                                                        basePort = 11311,
                                                        ros_master_ip = self._real_robot_pc_ip)
        elif backend == "ros2_gz":
            self._environmentController.build_scenario(launch_file_pkg_and_path=("lr_gym_ros2","/launch/gz_panda_cam.launch.xml"),
                                                        launch_file_args={  "use_gui":"false"})
        elif backend == "gz":
            self._environmentController.build_scenario( sdf_file=("lr_gym_ros2","/worlds/empty_cams.sdf"),
                                                        launch_file_pkg_and_path=("lr_gym_ros2","/launch/gz_panda_cam.launch.xml"),
                                                        launch_file_args={  "use_gui":"false", "launch_gazebo" : "false"})
        else:
            raise NotImplementedError("Backend '"+backend+"' not supported")

    def _destroySimulation(self):
        self._environmentController.destroy_scenario()

    def getInfo(self,state=None):
        return {}

    def getUiRendering(self):

        img, t = self._environmentController.getRenderings([self._camera_name])[self._camera_name]
        if img is None:
            npImg = None
            time = -1
            ggLog.warn("Could not get ui image, returning None")
        else:
            npImg = img
            time = t
        return npImg, time
