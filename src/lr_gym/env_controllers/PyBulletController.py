#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any

import pybullet as p

from lr_gym.utils.utils import JointState, LinkState, Pose
from lr_gym.env_controllers.EnvironmentController import EnvironmentController
from lr_gym.env_controllers.JointEffortEnvController import JointEffortEnvController
from lr_gym.env_controllers.SimulatedEnvController import SimulatedEnvController
from lr_gym.env_controllers.JointPositionEnvController import JointPositionEnvController
import lr_gym.utils.PyBulletUtils as PyBulletUtils
import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog
import quaternion
import xmltodict
import lr_gym.utils.utils
from pathlib import Path
import time





# def cvPose2BulletView(q, t):
#     """
#     cvPose2BulletView gets orientation and position as used 
#     in ROS-TF and opencv and coverts it to the view matrix used 
#     in openGL and pyBullet.
    
#     :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
#     :param t: ROS postion expressed as [tx, ty, tz]
#     :return:  4x4 view matrix as used in pybullet and openGL
    
#     """
#     R = quaternion.as_rotation_matrix(q)

#     T = np.vstack([np.hstack([R, np.array(t).reshape(3,1)]),
#                               np.array([0, 0, 0, 1])])
#     # Convert opencv convention to python convention
#     # By a 180 degrees rotation along X
#     Tc = np.array([[1,   0,    0,  0],
#                    [0,  -1,    0,  0],
#                    [0,   0,   -1,  0],
#                    [0,   0,    0,  1]]).reshape(4,4)
    
#     # pybullet pse is the inverse of the pose from the ROS-TF
#     T=Tc@np.linalg.inv(T)
#     # The transpose is needed for respecting the array structure of the OpenGL
#     viewMatrix = T.T.reshape(16)
#     return viewMatrix

class BulletCamera:
    def __init__(self, pose : Pose, hfov : float, width : int, height : int, near : float, far : float, link_name : Tuple[str,str], camera_name : str):
        self._width = width
        self._height = height
        self._pose   = pose if pose is not None else Pose(0,0,0,0,0,0,1)
        self._hfov   = hfov
        self._near   = near
        self._far    = far
        self.link_name = link_name
        self.camera_name = camera_name
        self._compute_matrixes()
        
    def _compute_matrixes(self):
        # ztop_to_ytop = quaternion.from_float_array([0.707, -0.707, 0.0, 0.0])
        # rot_matrix = quaternion.as_rotation_matrix(ztop_to_ytop*self._pose.orientation)
        # pose_vec = np.matmul(quaternion.as_rotation_matrix(ztop_to_ytop),np.array(self._pose.position))
        # extr_mat = np.zeros((4,4))
        # extr_mat[3,3] = 1
        # extr_mat[0:3,0:3] = rot_matrix
        # extr_mat[0:3,3] = -pose_vec
        # self._extrinsic_matrix = tuple(extr_mat.flatten(order = 'F'))
        n = "\n"
        # ggLog.info(f"em1 = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")
        # self._extrinsic_matrix = p.computeViewMatrixFromYawPitchRoll(   cameraTargetPosition=[0, 0, 0],
        #                                                                 distance=10,
        #                                                                 yaw=0,
        #                                                                 pitch=0,
        #                                                                 roll=0,
        #                                                                 upAxisIndex=2)
        # ggLog.info(f"em2 = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")
        
        # self._extrinsic_matrix = cvPose2BulletView(self._pose.orientation, self._pose.position)
        # ggLog.info(f"em3 = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")


        target_position = np.matmul(quaternion.as_rotation_matrix(self._pose.orientation),np.array([1.0, 0.0, 0.0])) + self._pose.position
        self._extrinsic_matrix = p.computeViewMatrix(cameraEyePosition = self._pose.position,
                                                     cameraTargetPosition = target_position,
                                                     cameraUpVector = [0,0,1])
        # ggLog.info(f"em = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")

        self._intrinsic_matrix = p.computeProjectionMatrixFOV(self._hfov*180/3.14159, self._width/self._height, self._near, self._far)
        # ggLog.info(f"em = {self._extrinsic_matrix}")
        # ggLog.info(f"im = \n{n.join([str(self._intrinsic_matrix[i::4]) for i in range(4)])}")

        # ggLog.info(f"im = {self._intrinsic_matrix}")

    def get_matrixes(self):
        return self._extrinsic_matrix, self._intrinsic_matrix

    def set_pose(self, pose : Pose):
        self._pose = pose
        self._compute_matrixes()

    def get_rendering(self):
        # ggLog.info(f"Getting camera image ({self._width}x{self._height}), {self._pose}")
        width, height, rgb, depth, segmentation = p.getCameraImage(width = self._width,
                                                                   height = self._height,
                                                                   viewMatrix = self._extrinsic_matrix,
                                                                   projectionMatrix = self._intrinsic_matrix,
                                                                   shadow=True,
                                                                   lightDirection=[1, 1, 1])
        img = np.array(rgb).reshape(height,width,4)[:,:,0:3]
        # ggLog.info(f"Got camera image")
        return img

class PyBulletController(EnvironmentController, JointEffortEnvController, SimulatedEnvController, JointPositionEnvController):
    """This class allows to control the execution of a PyBullet simulation.

    For what is possible it is meant to be interchangeable with GazeboController.
    """

    def __init__(self, stepLength_sec : float = 0.004166666666,
                        restore_on_reset = True):
        """Initialize the Simulator controller.


        """
        super().__init__()
        self._stepLength_sec = stepLength_sec
        self._spawned_objects_ids = {}
        self._cameras = {}
        self._commanded_torques = {}
        self._commanded_positions = {}
        
        self._max_joint_velocities_pos_control = {}
        self._max_joint_velocity_pos_control = 10
        self._max_torques_pos_control = {}
        self._max_torque_pos_control = 100
        self._restore_on_reset = restore_on_reset

    def _refresh_entities_ids(self):
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


        # ggLog.info("self._bodyAndJointIdToJointName = "+str(self._bodyAndJointIdToJointName))
        # ggLog.info("self._jointNamesToBodyAndJointId = "+str(self._jointNamesToBodyAndJointId))
        # ggLog.info("self._bodyAndLinkIdToLinkName = "+str(self._bodyAndLinkIdToLinkName))
        # ggLog.info("self._linkNameToBodyAndLinkId = "+str(self._linkNameToBodyAndLinkId))


    def startController(self):
        if not p.isConnected():
            raise ValueError("PyBullet is not connected")

        self._refresh_entities_ids()
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
        self._commanded_torques = {}
        self._commanded_positions = {}
        # ggLog.info(f"Resetting...")
        if self._restore_on_reset:
            p.restoreState(self._startStateId)
        # ggLog.info(f"Resetted")
        self._refresh_entities_ids()
        self._simTime = 0
        super().resetWorld()

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
        return stepLength

    def freerun(self, duration_sec: float):
        bullet_stepLength_sec = p.getPhysicsEngineParameters()["fixedTimeStep"]
        simsteps = int(duration_sec/bullet_stepLength_sec)
        # ggLog.info(f"PyBullet doing {duration_sec}/{bullet_stepLength_sec}={simsteps} steps")
        for i in range(simsteps):
            self._apply_commanded_torques()
            self._apply_commanded_positions()
            p.stepSimulation()
            self._simTime += bullet_stepLength_sec
        self._commanded_torques = {}
        return simsteps*bullet_stepLength_sec

    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[np.ndarray, float]]:
        ret = {}
        for cam_name in requestedCameras:
            camera = self._cameras[cam_name]
            linkstate = self.getLinksState([camera.link_name])[camera.link_name]
            camera.set_pose(linkstate.pose)
            ret[cam_name] = ((camera.get_rendering(), self.getEnvTimeFromReset()))
        return ret

    def _apply_commanded_torques(self):
        for bodyId in self._commanded_torques.keys():
            p.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=self._commanded_torques[bodyId][0],
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=self._commanded_torques[bodyId][1])

    def setJointsEffortCommand(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        #For each bodyId I submit a request for joint motor control
        requests = {}
        for jt in jointTorques:
            bodyId, jointId = self._getBodyAndJointId(jt[0:2])
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont
            requests[bodyId][1].append(jt[2]) #requested torque
        self._commanded_torques = requests


    def _apply_commanded_positions(self):
        for bodyId in self._commanded_positions.keys():
            p.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=self._commanded_positions[bodyId][0],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self._commanded_positions[bodyId][1],
                                        targetVelocities=self._commanded_positions[bodyId][2],
                                        forces=self._commanded_positions[bodyId][3])
            
    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float]) -> None:
        requests = {}
        for joint, position in jointPositions.items():
            bodyId, jointId = self._getBodyAndJointId(joint)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[],[],[])
            requests[bodyId][0].append(jointId) #requested joint
            requests[bodyId][1].append(position) #requested position
            requests[bodyId][2].append(self._max_joint_velocities_pos_control.get(joint,self._max_joint_velocity_pos_control)) #requested velocity
            requests[bodyId][3].append(self._max_torques_pos_control.get(joint,self._max_torque_pos_control)) #requested max torque
        self._commanded_positions = requests


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
                if len(s.effort) != 0 and any([e!=0 for e in s.effort]):
                    raise NotImplementedError(f"Direct effort setting is not supported, only zero effort setting is supported. Requested {s.effort}")
            p.resetJointStatesMultiDof(bodyId, ids, [s.position for s in states], [s.rate for s in states])



    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        # ggLog.info(f"Getting link states for {requestedLinks}")
        #For each bodyId I submit a request for joint state
        requests = {} #for each body id we will have a list of joints
        baserequests = []
        for ln in requestedLinks:
            bodyId, linkId = self._getBodyAndLinkId(ln)
            if linkId != -1:
                if bodyId not in requests: #If we haven't created a request for this body yet
                    requests[bodyId] = []
                requests[bodyId].append(linkId) #requested jont
            else:
                baserequests.append(bodyId) #requested jont

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
        for bodyId in baserequests: #for each bodyId make a request
            # ggLog.info(f"Getting pose of body {bodyId}")
            bodyPose = p.getBasePositionAndOrientation(bodyId)
            bodyVelocity = p.getBaseVelocity(bodyId)
            
            linkState = LinkState(  position_xyz = (bodyPose[0][0], bodyPose[0][1], bodyPose[0][2]),
                                    orientation_xyzw = (bodyPose[1][0], bodyPose[1][1], bodyPose[1][2], bodyPose[1][3]),
                                    pos_velocity_xyz = (bodyVelocity[0][0], bodyVelocity[0][1], bodyVelocity[0][2]),
                                    ang_velocity_xyz = (bodyVelocity[1][0], bodyVelocity[1][1], bodyVelocity[1][2]))
        
            allStates[self._getLinkName(bodyId,-1)] = linkState

        #print("returning "+str(allStates))
        # ggLog.info(f"Got link states for {allStates.keys()}")

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

        for bodyId, states in requests.items():
            for state in states:
                p.resetBasePositionAndOrientation(bodyId, state.pose.position, state.pose.getListXyzXyzw()[3:])

    def getEnvTimeFromStartup(self) -> float:
        return self._simTime

    def build_scenario(self, file_path, format = "urdf"):
        PyBulletUtils.startupPlaneWorld()
        if file_path is not None:
            self.spawn_model(model_file = file_path, model_format=format, model_name = "scenario")
        

    def destroy_scenario(self):
        self._spawned_objects_ids = {}
        PyBulletUtils.destroySimpleEnv()

    def _register_camera(self, camera : BulletCamera):
        self._cameras[camera.camera_name] = camera
        ggLog.info(f"Registered camera with name {camera.camera_name}")

    def _loadModel(self, modelFilePath : str, fileFormat : str = "urdf", model_kwargs = {}):
        if fileFormat.split(".")[-1] == "xacro":
            model_definition_string = lr_gym.utils.utils.compile_xacro_string(  model_definition_string=Path(modelFilePath).read_text(),
                                                                                model_kwargs=model_kwargs)
            compiled_file_path = f"/tmp/lr_gym_PyBulletUtils_xacrocompile_{int(time.time()*1000000)}_{hash(modelFilePath)}"
            Path(compiled_file_path).write_text(model_definition_string)
            modelFilePath = compiled_file_path
            fileFormat = ".".join(fileFormat.split(".")[:-1])
        
        if fileFormat == "urdf":
            bodyId = p.loadURDF(modelFilePath, flags=p.URDF_USE_SELF_COLLISION)
        elif fileFormat == "mjcf":
            bodyId = p.loadMJCF(modelFilePath, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0]
        elif fileFormat == "sdf":
            bodyId = p.loadSDF(modelFilePath)[0]
        else:
            raise AttributeError("Invalid format "+str(fileFormat))

        #p.changeDynamics(bodyId, -1, linearDamping=0, angularDamping=0)
        for j in range(p.getNumJoints(bodyId)):
            #p.changeDynamics(bodyId, j, linearDamping=0, angularDamping=0)
            p.setJointMotorControl2(bodyId,
                                    j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0,
                                    targetVelocity=0,
                                    positionGain=0.1,
                                    velocityGain=0.1,
                                    force=0) # Disable any position control motor by setting its max torque to zero
            p.setJointMotorControl2(bodyIndex=bodyId,
                                    jointIndex=j,
                                    controlMode=p.VELOCITY_CONTROL,
                                    force=0) # Disable velocity motor by setting its max torque to zero
            
            ggLog.info("Joint "+str(j)+" dynamics info: "+str(p.getDynamicsInfo(bodyId,j)))

        if fileFormat == "sdf":
            parsed_sdf = xmltodict.parse(Path(modelFilePath).read_text())
            ggLog.info(f"{parsed_sdf}")
            if "world" in parsed_sdf["sdf"]:
                world = parsed_sdf["sdf"]["world"]
            else:
                world = parsed_sdf["sdf"]
            models = world["model"]
            if type(models) != list:
                models = [models]
            for model in models:
                model_name = model["@name"]
                links = model.get("link", [])
                if type(links) != list:
                    links = [links]
                for link in links:
                    sensors = link.get("sensor", [])
                    link_name = link["@name"]
                    if type(sensors) != list:
                        sensors = [sensors]
                    for sensor in sensors:
                        if sensor["@type"] == "camera":
                            self._register_camera(BulletCamera(pose = None,
                                                               hfov =   float(sensor["camera"]["horizontal_fov"]),
                                                               width =  int(sensor["camera"]["image"]["width"]),
                                                               height = int(sensor["camera"]["image"]["height"]),
                                                               near =   float(sensor["camera"]["clip"]["near"]),
                                                               far =    float(sensor["camera"]["clip"]["far"]),
                                                               link_name=(model_name, link_name),
                                                               camera_name=sensor["@name"]))


        
        return bodyId

    def spawn_model(self,   model_definition_string: str = None,
                            model_format: str = None,
                            model_file = None,
                            model_name: str = None,
                            pose: Pose = None,
                            model_kwargs: Dict[Any, Any] = {}) -> str:
        if model_definition_string is not None:
            raise AttributeError(f"Only file model descriptions are supported")
        if model_name in self._spawned_objects_ids:
            raise AttributeError(f"model name {model_name} is already present")
        self._spawned_objects_ids[model_name] = self._loadModel(modelFilePath=model_file, fileFormat=model_format, model_kwargs=model_kwargs)
        self._refresh_entities_ids()
        ggLog.info(f"Spawned '{model_name}': {p.getBodyInfo(self._spawned_objects_ids[model_name])}")
        if pose is not None:
            p.resetBasePositionAndOrientation(self._spawned_objects_ids[model_name],
                                              pose.position, pose.getListXyzXyzw()[3:])
        return model_name

    def delete_model(self, model_name: str):
        PyBulletUtils.unloadModel(self._spawned_objects_ids[model_name])
        self._spawned_objects_ids.pop(model_name)
        self._refresh_entities_ids()

    def setupLight(self):
        raise NotImplementedError(f"Lighting setup is not supported in PyBullet")
    
    