#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

from lr_gym.envs.BaseEnv import BaseEnv

class ControlledEnv(BaseEnv):
    """This is a base-class for implementing OpenAI-gym environments using environment controllers derived from EnvironmentController.

    It implements part of the methods defined in BaseEnv relying on an EnvironmentController
    (not all methods are available on non-simulated EnvironmentControllers like RosEnvController, at least for now).

    The idea is that environments created from this will be able to run on different simulators simply by using specifying
    environmentController objects in the constructor

    You can extend this class with a sub-class to implement specific environments.
    """

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self,
                 maxStepsPerEpisode : int = 500,
                 stepLength_sec : float = -1,
                 environmentController = None,
                 startSimulation : bool = False,
                simulationBackend : str = None,
                 is_timelimited : bool = True):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times
        stepLength_sec : float
            Duration in seconds of each simulation step. Lower values will lead to
            slower simulation. This value should be kept higher than the gazebo
            max_step_size parameter.
        environmentController : EnvironmentController
            Specifies which simulator controller to use. By default it connects to Gazebo

        Raises
        -------
        rospy.ROSException
            In cause it fails to find the required ROS services
        ROSInterruptException
            In case it gets interrupted while waiting for ROS servics

        """


        if environmentController is None:
            raise AttributeError("You must specify environmentController")
        self._environmentController = environmentController
        self._estimatedSimTime = 0.0 # Estimated from the results of each environmentController.step()
        self._intendedStepLength_sec = stepLength_sec

        super().__init__(maxStepsPerEpisode = maxStepsPerEpisode,
                         startSimulation = startSimulation,
                         simulationBackend = simulationBackend,
                         is_timelimited=is_timelimited)





    def performStep(self) -> None:
        super().performStep()
        estimatedStepDuration_sec = 0
        while True: # Do at least one step, then check if we need more
            estimatedStepDuration_sec += self._environmentController.step()
            if estimatedStepDuration_sec >= self._intendedStepLength_sec:
                break
        self._estimatedSimTime += estimatedStepDuration_sec



    def performReset(self):
        super().performReset()
        self._environmentController.resetWorld()
        self._estimatedSimTime = 0.0
        self.initializeEpisode()


    def getSimTimeFromEpStart(self):
        return self._environmentController.getEnvSimTimeFromStart()
