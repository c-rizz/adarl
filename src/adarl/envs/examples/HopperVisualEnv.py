#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""

import numpy as np
from typing import Tuple, Any, Dict
import cv2

from adarl.envs.HopperEnv import HopperEnv
from adarl.adapters.BaseAdapter import BaseAdapter
#import tf2_py
import adarl.utils
import adarl.utils.dbg.ggLog as ggLog


class HopperVisualEnv(HopperEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup.

    """

    RobotState = np.typing.NDArray[(15,), np.float32]
    ImgObservation = np.typing.NDArray[(Any,Any,Any),np.float32]
    State = Tuple[RobotState, ImgObservation]
    
    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    simulatorController : BaseAdapter = None,
                    startSimulation : bool = True,
                    simulationBackend : str = "gazebo",
                    obs_img_height_width : Tuple[int,int] = (64,64),
                    frame_stacking_size : int = 3,
                    imgEncoding : str = "float",
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
        self._stepLength_sec = stepLength_sec
        #aspect = 426/160.0
        self._obs_img_height = obs_img_height_width[0]
        self._obs_img_width = obs_img_height_width[1]
        self._frame_stacking_size = frame_stacking_size
        self._imgEncoding = imgEncoding
        self._success = False
        self._success_ratio_avglen = 50
        self._successes = [1]*self._success_ratio_avglen
        self._tot_episodes = 0
        self._success_ratio = 0
        super(HopperEnv, self).__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = simulatorController,
                         startSimulation = startSimulation,
                         simulationBackend = simulationBackend)

        if imgEncoding == "float":
            self.observation_space = spaces.gym_spaces.Box(low=0, high=1,
                                                    shape=(self._frame_stacking_size, self._obs_img_height, self._obs_img_width),
                                                    dtype=np.float32)
        elif imgEncoding == "int":
            self.observation_space = spaces.gym_spaces.Box(low=0, high=255,
                                                    shape=(self._frame_stacking_size, self._obs_img_height, self._obs_img_width),
                                                    dtype=np.uint8)
        else:
            raise AttributeError(f"Unsupported imgEncoding '{imgEncoding}' requested, it can be either 'int' or 'float'")
        
        self._stackedImg = np.zeros(shape=(self._frame_stacking_size,self._obs_img_height, self._obs_img_width), dtype=np.float32)
        

        #print("HopperEnv: action_space = "+str(self.action_space))
        #print("HopperEnv: action_space = "+str(self.action_space))
        self._adapter.set_monitored_joints([("hopper","torso_to_thigh"),
                                                        ("hopper","thigh_to_leg"),
                                                        ("hopper","leg_to_foot"),
                                                        ("hopper","torso_pitch_joint")])
        self._adapter.set_monitored_links([("hopper","torso"),("hopper","thigh"),("hopper","leg"),("hopper","foot")])
        self._adapter.set_monitored_cameras(["camera"])

        self._adapter.startup()



    def reachedTerminalState(self, previousState : State,
                                state : State) -> bool:
        robotState : RobotState = state[0]
        prevRobotState : RobotState = previousState[0]
        return super().reachedTerminalState(prevRobotState, robotState)



    def computeReward(  self,
                        previousState : State,
                        state : State,
                        action : Tuple[float,float,float],
                        env_conf = None) -> float:
        robotState : RobotState = state[0]
        prevRobotState : RobotState = previousState[0]

        if not self.reachedTerminalState(previousState, state):
            speed = (robotState[15] - robotState[16])/self._stepLength_sec
            # print("Speed: "+str(speed))
            return 1 + 2*speed - 0.003*(action[0]*action[0] + action[1]*action[1] + action[2]*action[2]) # should be more or less the same as openai's hopper_v3
        else:
            return -1


    def getObservation(self, state) -> ImgObservation:
        return state[1]



    def getState(self) -> State:
        robotState = super().getState()
        imgObservation = np.copy(self._stackedImg)
        return (robotState, imgObservation)

    def initializeEpisode(self) -> None:
        if not self._spawned and isinstance(self._adapter, BaseSimulationAdapter):
            simCamHeight = int(64*(self._obs_img_height/64))
            simCamWidth = int(64*16/9*(self._obs_img_height/64))
            self._adapter.spawn_model(model_file=adarl.utils.utils.pkgutil_get_path("adarl","models/hopper_v1.urdf.xacro"),
                                                    model_name="hopper",
                                                    pose=build_pose(0,0,0,0,0,0,1),
                                                    model_kwargs={"camera_width":str(simCamWidth),"camera_height":str(simCamHeight)})
            self._spawned = True
        self._adapter.setJointsEffortCommand([("hopper","torso_to_thigh",0),
                                                            ("hopper","thigh_to_leg",0),
                                                            ("hopper","leg_to_foot",0)])

    def buildSimulation(self, backend : str = "gazebo"):

        if backend == "gazebo":
            worldpath = "\"$(find adarl_ros)/worlds/ground_plane_world_plugin.world\""
            self._adapter.build_scenario(launch_file_pkg_and_path=("adarl_ros","/launch/gazebo_server.launch"),
                                                        launch_file_args={  "gui":"false",
                                                                            "paused":"true",
                                                                            "physics_engine":"ode",
                                                                            "limit_sim_speed":"true",
                                                                            "world_name":worldpath,
                                                                            "gazebo_seed":f"{self._envSeed}",
                                                                            "wall_sim_speed":"false"})

        else:
            raise NotImplementedError("Backend "+backend+" not supported")


    def _reshapeFrame(self, frame):
        npArrImage = adarl.utils.utils.ros1_image_to_numpy(frame)
        # ggLog.info("Received image of shape "+str(npArrImage.shape))
        npArrImage = cv2.cvtColor(npArrImage, cv2.COLOR_BGR2GRAY)
        
        og_width = npArrImage.shape[1]
        og_height = npArrImage.shape[0]
        npArrImage = npArrImage[0:int(220.0/240*og_height), int(100/426.0*og_width):int(326/426.0*og_width)] #crop bottom 90px , left 100px, right 100px
        # print("shape",npArrImage.shape)
        #imgHeight = npArrImage.shape[0]
        #imgWidth = npArrImage.shape[1]
        #npArrImage = npArrImage[int(imgHeight*0/240.0):int(imgHeight*160/240.0),:] #crop top and bottom, it's an ndarray, it's fast
        npArrImage = cv2.resize(npArrImage, dsize = (self._obs_img_width, self._obs_img_height), interpolation = cv2.INTER_LINEAR)
        npArrImage = np.reshape(npArrImage, (self._obs_img_height, self._obs_img_width))
        if self._imgEncoding == "float":
            npArrImage = np.float32(npArrImage / 255) 
        elif self._imgEncoding == "int":
            npArrImage = np.uint8(npArrImage)
        else:
            raise RuntimeError(f"Unknown img encoding {self._imgEncoding}")
        
        #print("npArrImage.shape = "+str(npArrImage.shape))
        return npArrImage



    def performStep(self) -> None:
        for i in range(self._frame_stacking_size):
            #ggLog.info(f"Stepping {i}")
            super(HopperEnv, self).performStep()
            self._adapter.step()
            img, t = self._adapter.getRenderings(["camera"])["camera"]
            if img is None:
                ggLog.error("No camera image received. Observation will contain and empty image.")
                img = np.empty([self._obs_img_height, self._obs_img_width,3])
            img = self._reshapeFrame(img)
            self._stackedImg[i] = img
            self._estimatedSimTime += self._stepLength_sec



    def performReset(self, options = {}):
        #ggLog.info("PerformReset")
        super().performReset()
        self._adapter.resetWorld()
        self.initializeEpisode()
        img, t = self._adapter.getRenderings(["camera"])["camera"]
        if img is None:
            ggLog.error("No camera image received. Observation will contain and empty image.")
            img = np.empty([self._obs_img_height, self._obs_img_width,3])
        img = self._reshapeFrame(img)
        for i in range(self._frame_stacking_size):
            self._stackedImg[i] = img

    def getInfo(self,state=None) -> Dict[Any,Any]:
        return super().getInfo(state=state[0])
