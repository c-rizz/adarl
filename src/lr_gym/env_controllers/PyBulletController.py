#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any

import pybullet as p

from lr_gym.utils.utils import JointState, LinkState, Pose
from lr_gym.env_controllers.EnvironmentController import EnvironmentController
from lr_gym.env_controllers.JointEffortEnvController import JointEffortEnvController
from lr_gym.env_controllers.SimulatedEnvController import SimulatedEnvController
import lr_gym.utils.PyBulletUtils as PyBulletUtils
import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog


class PyBulletController(EnvironmentController, JointEffortEnvController, SimulatedEnvController):
    """This class allows to control the execution of a PyBullet simulation.

    For what is possible it is meant to be interchangeable with GazeboController.
    """

    def __init__(self, stepLength_sec : float = 0.004166666666):
        """Initialize the Simulator controller.


        """
        super().__init__()
        self._stepLength_sec = stepLength_sec
        self._spawned_objects_ids = {}
        

    def startController(self):
        if not p.isConnected():
            raise ValueError("PyBullet is not connected")

        bodyIds = []
        for i in range(p.getNumBodies()):
            bodyIds.append(p.getBodyUniqueId(i))

        self._bodyAndJointIdToJointName = {}
        self._jointNamesToBodyAndJointId = {}
        self._bodyAndLinkIdToLinkName = {}
        self._linkNameToBodyAndLinkId = {}
        for bodyId in bodyIds:
            base_link_name, body_name = p.getBodyInfo(bodyId)
            base_link_name = (body_name.decode("utf-8"), base_link_name.decode("utf-8"))
            base_body_and_link_id = (bodyId, -1)
            self._linkNameToBodyAndLinkId[base_link_name] = base_body_and_link_id
            self._bodyAndLinkIdToLinkName[base_body_and_link_id] = base_link_name
            for jointId in range(p.getNumJoints(bodyId)):
                jointInfo = p.getJointInfo(bodyId,jointId)
                jointName = (body_name.decode("utf-8"), jointInfo[1].decode("utf-8"))
                linkName = (body_name.decode("utf-8"), jointInfo[12].decode("utf-8"))
                body_and_joint_ids = (bodyId,jointId)
                self._bodyAndJointIdToJointName[body_and_joint_ids] = jointName
                self._jointNamesToBodyAndJointId[jointName] = body_and_joint_ids
                self._bodyAndLinkIdToLinkName[body_and_joint_ids] = linkName
                self._linkNameToBodyAndLinkId[linkName] = body_and_joint_ids


        ggLog.info("self._bodyAndJointIdToJointName = "+str(self._bodyAndJointIdToJointName))
        ggLog.info("self._jointNamesToBodyAndJointId = "+str(self._jointNamesToBodyAndJointId))
        ggLog.info("self._bodyAndLinkIdToLinkName = "+str(self._bodyAndLinkIdToLinkName))
        ggLog.info("self._linkNameToBodyAndLinkId = "+str(self._linkNameToBodyAndLinkId))
        self._startStateId = p.saveState()
        self._simTime = 0

        if self._stepLength_sec % p.getPhysicsEngineParameters()["fixedTimeStep"] > 0.00001:
            ggLog.warn(f"PyBulletController: stepLength_sec is not a multiple of pybullet's fixedTimeStep (respecively {self._stepLength_sec} and {p.getPhysicsEngineParameters()['fixedTimeStep']})")

        if self._stepLength_sec<p.getPhysicsEngineParameters()["fixedTimeStep"]:
            p.setTimeStep(self._stepLength_sec)


    def _getJointName(self, bodyId, jointIndex):
        jointName = self._bodyAndJointIdToJointName[(bodyId,jointIndex)]
        return jointName

    def _getBodyAndJointId(self, jointName):
        return self._jointNamesToBodyAndJointId[jointName] # TODO: this ignores the model name, should use it

    def _getLinkName(self, bodyId, linkIndex):
        linkName = self._bodyAndLinkIdToLinkName[(bodyId,linkIndex)]
        return linkName

    def _getBodyAndLinkId(self, linkName):
        return self._linkNameToBodyAndLinkId[linkName]

    def resetWorld(self):
        p.restoreState(self._startStateId)
        self._simTime = 0

    def step(self) -> float:
        """Run the simulation for the specified time.

        Parameters
        ----------

        Returns
        -------
        None


        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        #p.setTimeStep(self._stepLength_sec) #This is here, but still, as stated in the pybulelt quickstart guide this should not be changed often
        stepLength = self.freerun(self._stepLength_sec)
        self._simTime += stepLength
        return stepLength

    def freerun(self, duration_sec: float):
        bullet_stepLength_sec = p.getPhysicsEngineParameters()["fixedTimeStep"]
        simsteps = int(duration_sec/bullet_stepLength_sec)
        # ggLog.info(f"PyBullet doing {duration_sec}/{bullet_stepLength_sec}={simsteps} steps")
        for i in range(simsteps):
            p.stepSimulation()
        return simsteps*bullet_stepLength_sec

    def getRenderings(self, requestedCameras : List[str]) -> List[Tuple[np.ndarray, float]]:
        raise NotImplementedError("Rendering is not supported for PyBullet")



    def setJointsEffortCommand(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        #For each bodyId I submit a request for joint motor control
        requests = {}
        for jt in jointTorques:
            bodyId, jointId = self._getBodyAndJointId(jt[0:2])
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont
            requests[bodyId][1].append(jt[2]) #requested torque

        for bodyId in requests.keys():
            p.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=requests[bodyId][0],
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=requests[bodyId][1])



    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        #For each bodyId I submit a request for joint state
        requests = {} #for each body id we will have a list of joints
        for jn in requestedJoints:
            bodyId, jointId = self._getBodyAndJointId(jn)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            requests[bodyId].append(jointId) #requested jont

        allStates = {}
        for bodyId in requests.keys():#for each bodyId make a request
            bodyStates = p.getJointStates(bodyId,requests[bodyId])
            for i in range(len(requests[bodyId])):#put the responses of this bodyId in allStates
                jointId = requests[bodyId][i]
                jointState = JointState([bodyStates[i][0]], [bodyStates[i][1]], [bodyStates[i][3]]) #NOTE: effort may not be reported when using torque control
                allStates[self._getJointName(bodyId,jointId)] = jointState


        return allStates


    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        requests_by_body = {} # one request per body
        for jn, js in jointStates.items():
            bodyId, jointId = self._getBodyAndJointId(jn)
            if bodyId not in requests_by_body: #If we haven't created a request for this body yet
                requests_by_body[bodyId] = []
            requests_by_body[bodyId].append((jointId, js)) #requested jont

        for bodyId, reqs in requests_by_body.items():#for each bodyId make a request
            ids = [r[0] for r in reqs]
            states = [r[1] for r in reqs]
            for s in states:
                if len(s.position) > 1:
                    raise NotImplementedError(f"Only 1-DOF joints are supported")
                if len(s.rate) > 1:
                    raise NotImplementedError(f"Only 1-DOF joints are supported")
                if len(s.effort) != 0:
                    raise NotImplementedError(f"Direct effort setting is not supported")
            p.resetJointStatesMultiDof(bodyId, ids, [s.position for s in states], [s.rate for s in states])



    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        #For each bodyId I submit a request for joint state
        requests = {} #for each body id we will have a list of joints
        for ln in requestedLinks:
            bodyId, linkId = self._getBodyAndLinkId(ln)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            requests[bodyId].append(linkId) #requested jont

        allStates = {}
        for bodyId in requests.keys():#for each bodyId make a request
            bodyStates = p.getLinkStates(bodyId,requests[bodyId],computeLinkVelocity=1)
            for i in range(len(requests[bodyId])):#put the responses of this bodyId in allStates
                #print("bodyStates["+str(i)+"] = "+str(bodyStates[i]))
                linkId = requests[bodyId][i]
                linkState = LinkState(  position_xyz = (bodyStates[i][0][0], bodyStates[i][0][1], bodyStates[i][0][2]),
                                        orientation_xyzw = (bodyStates[i][1][0], bodyStates[i][1][1], bodyStates[i][1][2], bodyStates[i][1][3]),
                                        pos_velocity_xyz = (bodyStates[i][6][0], bodyStates[i][6][1], bodyStates[i][6][2]),
                                        ang_velocity_xyz = (bodyStates[i][7][0], bodyStates[i][7][1], bodyStates[i][7][2]))
            
                allStates[self._getLinkName(bodyId,linkId)] = linkState

        #print("returning "+str(allStates))

        return allStates

    def setLinksStateDirect(self, linksStates: Dict[Tuple[str, str], LinkState]):
        requests = {}
        for ln, ls in linksStates.items():
            bodyId, linkId = self._getBodyAndLinkId(ln)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            if linkId != -1: # could also check if the base joint is a free-floating one, and use the joint to move the first link
                raise RuntimeError(f"Can only set pose for base links, but requested to move link {ln}")
            requests[bodyId].append(ls)

        for bodyId, state in requests.items():
            p.resetBasePositionAndOrientation(bodyId, state.pose.position, state.pose.getListXyzXyzw()[3:])

    def getEnvTimeFromStartup(self) -> float:
        return self._simTime

    def build_scenario(self, file_path, format = "urdf"):
        PyBulletUtils.buildSimpleEnv()
        self.spawn_model(model_file = file_path, model_format=format, model_name = "scenario")
        

    def destroy_scenario(self):
        self._spawned_objects_ids = {}
        PyBulletUtils.destroySimpleEnv()


    def spawn_model(self,   model_definition_string: str = None,
                            model_format: str = None,
                            model_file = None,
                            model_name: str = None,
                            pose: Pose = None,
                            model_kwargs: Dict[Any, Any] = {}) -> str:
        if model_definition_string is not None:
            raise AttributeError(f"Only file model descriptions are supported")
        if model_kwargs is not None and len(model_kwargs)>0:
            raise AttributeError(f"model_kwargs is not supported")
        if model_name in self._spawned_objects_ids:
            raise AttributeError(f"model name {model_name} is already present")
        self._spawned_objects_ids[model_name] = PyBulletUtils.loadModel(modelFilePath=model_file, fileFormat=model_format)
        ggLog.info(f"Spawned '{model_name}': {p.getBodyInfo(self._spawned_objects_ids[model_name])}")

    def delete_model(self, model_name: str):
        PyBulletUtils.unloadModel(self._spawned_objects_ids[model_name])
        self._spawned_objects_ids.pop(model_name)

    def setupLight(self):
        raise NotImplementedError(f"Lighting setup is not supported in PyBullet")
    
    