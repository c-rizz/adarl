#!/usr/bin/env python3
"""This file implements PandaEffortKeepPoseEnvironment."""


import numpy as np
from typing import Tuple
import quaternion
import adarl.utils.spaces as spaces
from adarl.envs.ControlledEnv import ControlledEnv
import adarl



class PandaEffortBaseEnv(ControlledEnv):
    """This class represents an environment in which a Panda arm is controlled with torque control to keep an end-effector pose."""

    action_space_high = np.array([  1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1])
    action_space = spaces.gym_spaces.Box(-action_space_high,action_space_high) # 7 joints, torque controlled, normalized




    # From franka documentation
    joint_position_limits = np.array([  [ 2.8973, -2.8973],
                                        [ 1.7628, -1.7628],
                                        [ 2.8973, -2.8973],
                                        [-0.0698, -3.0718],
                                        [ 2.8973, -2.8973],
                                        [ 3.7525, -0.0175],
                                        [ 2.8973, -2.8973]])



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
                                        np.finfo(np.float32).max, # joint 1 velocity
                                        np.finfo(np.float32).max, # joint 2 velocity
                                        np.finfo(np.float32).max, # joint 3 velocity
                                        np.finfo(np.float32).max, # joint 4 velocity
                                        np.finfo(np.float32).max, # joint 5 velocity
                                        np.finfo(np.float32).max, # joint 6 velocity
                                        np.finfo(np.float32).max, # joint 7 velocity
                                        ])
    observation_space = spaces.gym_spaces.Box(-observation_space_high, observation_space_high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    maxTorques : Tuple[float, float, float, float, float, float, float] = [87, 87, 87, 87, 12, 12, 12],
                    environmentController : adarl.adapters.BaseAdapter = None,
                    forced_ros_master_uri : str = None,
                    stepLength_sec : float = 0.01,
                    startSimulation : bool = False):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        render : bool
            Perform rendering at each timestep
            Disable this if you don't need the rendering
        maxTorques : Tuple[float, float, float, float, float, float, float]
            Maximum torque that can be applied ot each joint
        environmentController : adarl.adapters.BaseAdapter
            The contorller to be used to interface with the environment. If left to None an EffortRosControlAdapter will be used.

        """


        if environmentController is None:
            # environmentController = EffortRosControlAdapter(
            #                  effortControllersInfos = { "panda_arm_effort_effort_compensated_controller" : ("panda_arm_effort_effort_compensated_controller",
            #                                                                                                 "panda",
            #                                                                                                 ("panda_joint1",
            #                                                                                                  "panda_joint2",
            #                                                                                                  "panda_joint3",
            #                                                                                                  "panda_joint4",
            #                                                                                                  "panda_joint5",
            #                                                                                                  "panda_joint6",
            #                                                                                                  "panda_joint7"))},
            #                  trajectoryControllersInfos = {"panda_arm_effort_trajectory_controller" :   ("panda_arm_effort_trajectory_controller",
            #                                                                                              "panda",
            #                                                                                              ("panda_joint1",
            #                                                                                               "panda_joint2",
            #                                                                                               "panda_joint3",
            #                                                                                               "panda_joint4",
            #                                                                                               "panda_joint5",
            #                                                                                               "panda_joint6",
            #                                                                                               "panda_joint7"))},
            #                  initialJointPositions = [  ("panda","panda_joint1", 0),
            #                                             ("panda","panda_joint2", 0),
            #                                             ("panda","panda_joint3", 0),
            #                                             ("panda","panda_joint4", -1),
            #                                             ("panda","panda_joint5", 0),
            #                                             ("panda","panda_joint6", 1),
            #                                             ("panda","panda_joint7", 0)],
            #                  stepLength_sec = stepLength_sec,
            #                  forced_ros_master_uri = forced_ros_master_uri)
            raise AttributeError("You must specify an environmentController")
        if not isinstance(environmentController, adarl.adapters.BaseJointEffortAdapter):
            raise AttributeError("environmentController must be a BaseJointEffortAdapter")

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         environmentController = environmentController,
                         startSimulation = startSimulation)




        self._maxTorques = np.array(maxTorques)
        self._renderingEnabled = render
        if self._renderingEnabled:
            self._adapter.set_monitored_cameras(["camera"]) #TODO: fix the camera topic

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
                                                       # ("panda","panda_link8"),
                                                       ])

        self._adapter.startup()





    def submitAction(self, action : Tuple[float, float, float, float, float, float, float]) -> None:
        """Send the joint torque command.

        Parameters
        ----------
        action : Tuple[float, float, float, float, float, float, float]
            torque control command

        """
        super().submitAction(action)
        clippedAction = np.clip(np.array(action, dtype=np.float32),-self._maxTorques,self._maxTorques)
        torques = [normalizedTorque*maxTorque for normalizedTorque,maxTorque in zip(clippedAction,self._maxTorques)]
        jointTorques = [("panda","panda_joint"+str(i+1),torques[i]) for i in range(7)]
        self._adapter.setJointsEffortCommand(jointTorques)

    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> np.typing.NDArray[(20,), np.float32]:
        """Get an observation of the environment.

        Returns
        -------
        np.typing.NDArray[(20,), np.float32]
            numpy ndarray. The content of each field is specified at the self.observation_space_high definition

        """

        jointStates = self._adapter.getJointsState([  ("panda","panda_joint1"),
                                                                    ("panda","panda_joint2"),
                                                                    ("panda","panda_joint3"),
                                                                    ("panda","panda_joint4"),
                                                                    ("panda","panda_joint5"),
                                                                    ("panda","panda_joint6"),
                                                                    ("panda","panda_joint7")])

        eePose = self._adapter.getLinksState([("panda","panda_link7")])[("panda","panda_link7")].pose

        quat = quaternion.from_float_array([eePose.orientation.w,eePose.orientation.x,eePose.orientation.y,eePose.orientation.z])
        eeOrientation_rpy = quaternion.as_euler_angles(quat)

        #print("got ee pose "+str(eePose))





        state = [   eePose.position[0],
                    eePose.position[1],
                    eePose.position[2],
                    eeOrientation_rpy[0],
                    eeOrientation_rpy[1],
                    eeOrientation_rpy[2],
                    jointStates["panda","panda_joint1"].position[0],
                    jointStates["panda","panda_joint2"].position[0],
                    jointStates["panda","panda_joint3"].position[0],
                    jointStates["panda","panda_joint4"].position[0],
                    jointStates["panda","panda_joint5"].position[0],
                    jointStates["panda","panda_joint6"].position[0],
                    jointStates["panda","panda_joint7"].position[0],
                    jointStates["panda","panda_joint1"].rate[0],
                    jointStates["panda","panda_joint2"].rate[0],
                    jointStates["panda","panda_joint3"].rate[0],
                    jointStates["panda","panda_joint4"].rate[0],
                    jointStates["panda","panda_joint5"].rate[0],
                    jointStates["panda","panda_joint6"].rate[0],
                    jointStates["panda","panda_joint7"].rate[0]] # No unrecoverable failure states
        #print("Got state = ",state)
        return np.array(state,dtype=np.float32)



    def build(self, backend : str = "gazebo"):
        if backend != "gazebo":
            raise NotImplementedError("Backend "+backend+" not supported")

        self._adapter.build_scenario(launch_file_pkg_and_path=("adarl_ros","/launch/launch_panda.launch"),
                                                    launch_file_args={  "gui":"false",
                                                                        "gazebo_seed":f"{self._envSeed}",
                                                                        "load_gripper":"false"})

    def _destroy(self):
        self._adapter.destroy_scenario()

    def _normalizedJointPositions(self, state):
        jnt_positions = np.array([state[i] for i in range(6,6+7)])

        max_limits = self.joint_position_limits[:,0]
        min_limits = self.joint_position_limits[:,1]
        jnt_ranges = max_limits-min_limits

        normalized_positions = (jnt_positions-min_limits)/jnt_ranges
        return normalized_positions

    def reachedTerminalState(self, previousState, state) -> bool:
        r = super().reachedTerminalState(previousState, state)
        if r:
            self.submitAction([0,0,0,0,0,0,0])
        return r
