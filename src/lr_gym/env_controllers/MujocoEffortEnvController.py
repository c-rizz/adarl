
from abc import ABC, abstractmethod
from lr_gym.utils.utils import JointState, LinkState, Pose, build_pose
from lr_gym.env_controllers.EnvironmentController import EnvironmentController
from lr_gym.env_controllers.JointEffortEnvController import JointEffortEnvController
from lr_gym.env_controllers.SimulatedEnvController import SimulatedEnvController
from typing import List, Tuple, Dict
import torch as th

class MujocoEffortEnvController(EnvironmentController, JointEffortEnvController, SimulatedEnvController):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def startController(self):

    @abstractmethod
    def step(self) -> float:

    @abstractmethod
    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[th.Tensor, float]]:

    @abstractmethod
    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],Tuple[JointState, float]]:

    @abstractmethod
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],Tuple[LinkState, float]]:

    @abstractmethod
    def resetWorld(self):

    @abstractmethod
    def getEnvTimeFromStartup(self) -> float:

    @abstractmethod
    def freerun(self, duration_sec : float):

    @abstractmethod
    def build_scenario(self, **kwargs):
    
    @abstractmethod
    def destroy_scenario(self, **kwargs):
