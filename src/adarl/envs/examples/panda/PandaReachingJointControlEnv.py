#!/usr/bin/env python3

import numpy as np
from typing import Tuple
import quaternion
import adarl.utils.spaces as spaces
from adarl.envs.ControlledEnv import ControlledEnv
import adarl.utils.dbg.ggLog as ggLog


class PandaReachingJointControlEnv(ControlledEnv):
    """This class represents and environment in which a Panda arm is controlled with Moveit to reach a goal pose.

    As moveit_commander is not working with python3 this environment relies on an intermediate ROS node for sending moveit commands.
    """

    action_space_high = np.array([  1,
                                    1,
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
            self._adapter = environmentController

        super().__init__(   maxStepsPerEpisode = maxStepsPerEpisode,
                            startSimulation = startSimulation,
                            environmentController=self._adapter,
                            simulationBackend=backend)

        self._camera_name = "simple_camera"
        self._adapter.set_monitored_cameras([self._camera_name])

        self._adapter.set_monitored_joints( [("panda","panda_joint1"),
                                                        ("panda","panda_joint2"),
                                                        ("panda","panda_joint3"),
                                                        ("panda","panda_joint4"),
                                                        ("panda","panda_joint5"),
                                                        ("panda","panda_joint6"),
                                                        ("panda","panda_joint7")])


        self._adapter.set_monitored_links( [("panda","panda_link1"),
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

        self._adapter.startup()

        self._operatingArea = operatingArea #min xyz, max xyz




    def submitAction(self, action_joints : np.typing.NDArray[(7,), np.float32]) -> None:
        """Plan and execute moveit movement without blocking.

        Parameters
        ----------
        action : Tuple[float, float, float, float, float, float, float]
            Relative end-effector movement in joint space. It is normalized to the max movement distance, i.e.
            this funciont shoult receive values in the [-1,1] range, which are then converted to the proper
            value range.

        """
        super().submitAction(action_joints)
        #print("received action "+str(action))
        clippedAction = np.clip(np.array(action_joints, dtype=np.float32),-1,1)
        rel_joint_move = clippedAction*self._maxPositionChange
        
        currentJointPose = self.getState()[6:6+7]

        absolute_req_jpose = currentJointPose + rel_joint_move

        self._adapter.setJointsPositionCommand(jointPositions = {("panda",f"panda_joint{i+1}") : absolute_req_jpose[i] for i in range(7)})
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




    def reachedTerminalState(self, previousState : np.typing.NDArray[(15,), np.float32], state : np.typing.NDArray[(15,), np.float32]) -> bool:
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
        pass

    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> np.typing.NDArray[(15,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        np.typing.NDArray[(15,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        eePose = self._adapter.getLinksState(requestedLinks=[("panda","panda_link8")])[("panda","panda_link8")].pose
        jointStates = self._adapter.getJointsState([("panda","panda_joint1"),
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
                    self._adapter.actionsFailsInLastStep()]

        return np.array(state,dtype=np.float32)

    def build(self, backend : str = "gazebo"):
        if backend == "gazebo":
            self._adapter.build_scenario(launch_file_pkg_and_path=("adarl_ros","/launch/launch_panda_moveit.launch"),
                                                        launch_file_args={  "gui":"false",
                                                                            "load_gripper":"false"})
        elif backend == "real":
            self._adapter.build_scenario(launch_file_pkg_and_path=("adarl_ros","/launch/launch_panda_moveit.launch"),
                                                        launch_file_args={  "robot_ip":self._real_robot_ip,
                                                                            "simulated":"false",
                                                                            "control_mode":"position"},
                                                        basePort = 11311,
                                                        ros_master_ip = self._real_robot_pc_ip)
        elif backend == "gz":
            self._adapter.build_scenario(launch_file_pkg_and_path=("adarl_ros2","/launch/gz_panda_cam.launch.xml"),
                                                        launch_file_args={  "use_gui":"false"})
        else:
            raise NotImplementedError("Backend '"+backend+"' not supported")

    def _destroy(self):
        self._adapter.destroy_scenario()

    def getInfo(self,state=None):
        return {}

    def getUiRendering(self):

        img, t = self._adapter.getRenderings([self._camera_name])[self._camera_name]
        if img is None:
            npImg = None
            time = -1
            ggLog.warn("Could not get ui image, returning None")
        else:
            npImg = img
            time = t
        return npImg, time
