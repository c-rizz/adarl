#!/usr/bin/env python3
"""
Class implementing Gazebo-based gym cartpole environment.

Based on ControlledEnv
"""



import gym
import numpy as np
from typing import Tuple, Dict, Any
import lr_gym.utils.dbg.ggLog as ggLog
import random

from lr_gym.envs.ControlledEnv import ControlledEnv
import lr_gym
from lr_gym.utils.utils import Pose, JointState
from lr_gym.env_controllers.SimulatedEnvController import SimulatedEnvController

class CartpoleEnv(ControlledEnv):
    """This class implements an OpenAI-gym environment with Gazebo, representing the classic cart-pole setup."""

    high = np.array([   2.5 * 2,
                        np.finfo(np.float32).max,
                        0.7 * 2,
                        np.finfo(np.float32).max])

    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(-high, high)
    metadata = {'render.modes': ['rgb_array']}

    def __init__(   self,
                    maxStepsPerEpisode : int = 500,
                    render : bool = False,
                    stepLength_sec : float = 0.05,
                    environmentController = None,
                    startSimulation : bool = True,
                    wall_sim_speed = False,
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
        environmentController : EnvironmentController
            Specifies which simulator controller to use. By default it connects to Gazebo


        """

        self._spawned = False
        self._wall_sim_speed = wall_sim_speed
        self._renderingEnabled = render
        self.seed(seed)
        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         stepLength_sec = stepLength_sec,
                         environmentController = environmentController,
                         startSimulation = startSimulation,
                         simulationBackend = "gazebo")

        self._environmentController.setJointsToObserve([("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        if self._renderingEnabled:
            self._environmentController.setCamerasToObserve(["camera"])

        self._environmentController.startController()
        self._success = False

    def submitAction(self, action : int) -> None:
        super().submitAction(action)
        if action == 0: #left
            direction = -1
        elif action == 1:
            direction = 1
        else:
            raise AttributeError("Invalid action (it's "+str(action)+")")

        self._environmentController.setJointsEffortCommand(jointTorques = [("cartpole_v0","foot_joint", direction * 20)])



    def checkEpisodeEnded(self, previousState : Tuple[float,float,float,float, np.ndarray], state : Tuple[float,float,float,float, np.ndarray]) -> bool:
        if super().checkEpisodeEnded(previousState, state):
            return True
        cartPosition = state[0]
        poleAngle = state[2]

        maxCartDist = 2
        maxPoleAngle = 0.261791667 #15 degrees

        if cartPosition < -maxCartDist or cartPosition > maxCartDist   or   maxPoleAngle < -poleAngle or poleAngle > maxPoleAngle:
            done = True
        else:
            done = False

        self._success = self._actionsCounter>=self._maxStepsPerEpisode

        return done


    def computeReward(self, previousState : Tuple[float,float,float,float], state : Tuple[float,float,float,float], action : int) -> float:
        return 1


    def initializeEpisode(self) -> None:
        # ggLog.info(f"Initializing isinstance(self._environmentController, SimulatedEnvController) = {isinstance(self._environmentController, SimulatedEnvController)}")
        
        if isinstance(self._environmentController, SimulatedEnvController):
            self._environmentController.setJointsStateDirect({("cartpole_v0","foot_joint"): JointState(position = [0.1*random.random()-0.05], rate=[0], effort=[0]),
                                                              ("cartpole_v0","cartpole_joint"): JointState(position = [0.1*random.random()-0.05], rate=[0], effort=[0])})
        self._environmentController.setJointsEffortCommand([("cartpole_v0","foot_joint",0),("cartpole_v0","cartpole_joint",0)])


    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        try:
            img, t = self._environmentController.getRenderings([self._rendering_cam_name])[0]
            # return imgs[0]
            # img = self._environmentController.getRenderings(["box::simple_camera_link::simple_camera"])[0]
            npImg = img
            if img is None:
                time = -1
            else:
                time = t
            return npImg, time
        except Exception as e:
            ggLog.warn(f"Exception getting ui image: {lr_gym.utils.utils.exc_to_str(e)}")
            return None, 0


    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> Tuple[float,float,float,float]:
        """Get an observation of the environment.

        Returns
        -------
        Tuple[float,float,float,float]
            A tuple containing: (cart position in meters, carts speed in meters/second, pole angle in radiants, pole speed in rad/s)

        """


        #t0 = time.monotonic()
        states = self._environmentController.getJointsState(requestedJoints=[("cartpole_v0","foot_joint"),("cartpole_v0","cartpole_joint")])
        #print("states['foot_joint'] = "+str(states["foot_joint"]))
        #print("Got joint state "+str(states))
        #t1 = time.monotonic()
        #rospy.loginfo("observation gathering took "+str(t1-t0)+"s")

        state = ( states[("cartpole_v0","foot_joint")].position[0],
                  states[("cartpole_v0","foot_joint")].rate[0],
                  states[("cartpole_v0","cartpole_joint")].position[0],
                  states[("cartpole_v0","cartpole_joint")].rate[0])

        #print(state)

        return np.array(state)


    def buildSimulation(self, backend):
        # ggLog.info("Building env")
        envCtrlName = type(self._environmentController).__name__
        if envCtrlName in ["GazeboController", "GazeboControllerNoPlugin"]:
            # ggLog.info(f"sim_img_width  = {sim_img_width}")
            # ggLog.info(f"sim_img_height = {sim_img_height}")
            if not self._renderingEnabled:
                worldpath = "\"$(find lr_gym_ros)/worlds/ground_plane_world_plugin.world\""
            else:
                worldpath = "\"$(find lr_gym_ros)/worlds/fixed_camera_world_plugin.world\""
            self._environmentController.build_scenario( launch_file_pkg_and_path=("lr_gym_ros","/launch/gazebo_server.launch"),
                                                        launch_file_args={  "gui":"false",
                                                                            "paused":"true",
                                                                            "physics_engine":"bullet",
                                                                            "limit_sim_speed":"false",
                                                                            "world_name":worldpath,
                                                                            "gazebo_seed":f"{self._envSeed}",
                                                                            "wall_sim_speed":f"{self._wall_sim_speed}"})
            self._rendering_cam_name = "camera"
        elif envCtrlName == "GzController":
            self._environmentController.build_scenario(sdf_file = ("lr_gym_ros2","/worlds/empty_cams.sdf"))
            # self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/simple_camera.sdf.xacro"),
            #                                         model_name=None,
            #                                         pose=Pose(0,2,0.5,0,0.0,-0.707,0.707),
            #                                         model_kwargs={"camera_width":"1920","camera_height":"1080","frame_rate":1/self._intendedStepLength_sec},
            #                                         model_format="sdf.xacro")
            self._rendering_cam_name = "simple_camera"
        elif envCtrlName == "PyBulletController":
            self._environmentController.build_scenario(None)
        else:
            raise NotImplementedError("environmentController "+envCtrlName+" not supported")

        if not self._spawned and isinstance(self._environmentController, SimulatedEnvController):
            if type(self._environmentController).__name__ == "GzController":
                cartpole_model_name = None
                cam_model_name = None
            else:
                cartpole_model_name = "cartpole_v0"
                cam_model_name = "simple_camera"
            cartpole_pose = Pose(0,0,0,0,0,0,1)
            name = self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cartpole_v0.urdf.xacro"),
                                                            model_name=cartpole_model_name,
                                                            pose=cartpole_pose,
                                                            # model_kwargs={"camera_width":"213","camera_height":"120"},
                                                            model_format="urdf.xacro")
            self._spawned = True
            self._environmentController.spawn_model(model_file=lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/simple_camera.sdf.xacro"),
                                                    model_name=cam_model_name,
                                                    pose=Pose(0,2,0.5, 0.0,0.0,-0.707,0.707),
                                                    # pose=Pose(0,5,0.5, 0.0,0.0,0.0,1.0),
                                                    model_kwargs={"camera_width":"256","camera_height":"144","frame_rate":1/self._intendedStepLength_sec},
                                                    model_format="sdf.xacro")
            self._rendering_cam_name = "simple_camera"
            ggLog.info(f"Model spawned with name {name}")





    def _destroySimulation(self):
        self._environmentController.destroy_scenario()

    def getInfo(self,state=None) -> Dict[Any,Any]:
        i = super().getInfo(state=state)
        i["success"] = self._success
        # ggLog.info(f"Setting success_ratio to {i['success_ratio']}")
        return i