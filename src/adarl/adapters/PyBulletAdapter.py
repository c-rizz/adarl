#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any, Optional

import pybullet

from adarl.utils.utils import JointState, LinkState, Pose, build_pose, buildQuaternion
from adarl.adapters.BaseAdapter import BaseAdapter
from adarl.adapters.BaseJointEffortAdapter import BaseJointEffortAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.BaseJointVelocityAdapter import BaseJointVelocityAdapter
import numpy as np
import adarl.utils.dbg.ggLog as ggLog
import quaternion
import xmltodict
import adarl.utils.utils
from pathlib import Path
import time
import threading
import os
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import pybullet_data



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
    def __init__(self, pose : Pose, hfov : float, width : int, height : int, near : float, far : float, link_name : Tuple[str,str], camera_name : str,
                 pybullet_controller):
        self._width = width
        self._height = height
        self._pose   = pose if pose is not None else build_pose(0,0,0,0,0,0,1)
        self._vfov   = hfov * height/width
        self._near   = near
        self._far    = far
        self.link_name = link_name
        self.camera_name = camera_name
        self._pybullet_controller = pybullet_controller
        self._compute_matrixes()
        self.setup_light(   lightDirection = [1,1,1],
                            lightColor = [0.9,0.9,0.9],
                            lightDistance = 100,
                            enable_shadows = True,
                            lightAmbientCoeff = 0.8,
                            lightDiffuseCoeff = 0.5,
                            lightSpecularCoeff = 0.1)
        
    def _compute_matrixes(self):
        # ztop_to_ytop = quaternion.from_float_array([0.707, -0.707, 0.0, 0.0])
        # rot_matrix = quaternion.as_rotation_matrix(ztop_to_ytop*self._pose.orientation)
        # pose_vec = np.matmul(quaternion.as_rotation_matrix(ztop_to_ytop),np.array(self._pose.position))
        # extr_mat = np.zeros((4,4))
        # extr_mat[3,3] = 1
        # extr_mat[0:3,0:3] = rot_matrix
        # extr_mat[0:3,3] = -pose_vec
        # self._extrinsic_matrix = tuple(extr_mat.flatten(order = 'F'))
        # n = "\n"
        # ggLog.info(f"em1 = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")
        # self._extrinsic_matrix = pybullet.computeViewMatrixFromYawPitchRoll(   cameraTargetPosition=[0, 0, 0],
        #                                                                 distance=10,
        #                                                                 yaw=0,
        #                                                                 pitch=0,
        #                                                                 roll=0,
        #                                                                 upAxisIndex=2)
        # ggLog.info(f"em2 = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")
        
        # self._extrinsic_matrix = cvPose2BulletView(self._pose.orientation, self._pose.position)
        # ggLog.info(f"em3 = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")

        orientation_quat = buildQuaternion(*self._pose.orientation_xyzw)
        cameraAxis =     np.matmul(quaternion.as_rotation_matrix(orientation_quat),np.array([1.0, 0.0, 0.0]))
        cameraUpVector = np.matmul(quaternion.as_rotation_matrix(orientation_quat),np.array([0.0, 0.0, 1.0]))
        # ggLog.info(f"cameraAxis = {cameraAxis}")
        # ggLog.info(f"cameraUpVector = {cameraUpVector}")
        target_position = cameraAxis + np.array(self._pose.position)
        # ggLog.info(f"Target = {target_position}")
        # ggLog.info(f"Eye    = {self._pose.position}")
        self._extrinsic_matrix = pybullet.computeViewMatrix(cameraEyePosition = np.array(self._pose.position),
                                                     cameraTargetPosition = target_position,
                                                     cameraUpVector = cameraUpVector)
        # ggLog.info(f"em = \n{n.join([str(self._extrinsic_matrix[i::4]) for i in range(4)])}")

        self._intrinsic_matrix = pybullet.computeProjectionMatrixFOV(self._vfov*180/3.14159, self._width/self._height, self._near, self._far)
        # ggLog.info(f"em = {self._extrinsic_matrix}")
        # ggLog.info(f"im = \n{n.join([str(self._intrinsic_matrix[i::4]) for i in range(4)])}")

        # ggLog.info(f"im = {self._intrinsic_matrix}")

    def get_matrixes(self):
        return self._extrinsic_matrix, self._intrinsic_matrix

    def set_pose(self, pose : Pose):
        self._pose = pose
        self._compute_matrixes()

    def setup_light(self,   lightDirection,
                            lightColor,
                            lightDistance,
                            enable_shadows,
                            lightAmbientCoeff,
                            lightDiffuseCoeff,
                            lightSpecularCoeff):
        self._lightDirection = lightDirection
        self._lightColor = lightColor
        self._lightDistance = lightDistance
        self._enable_shadows = enable_shadows
        self._lightAmbientCoeff = lightAmbientCoeff
        self._lightDiffuseCoeff = lightDiffuseCoeff
        self._lightSpecularCoeff = lightSpecularCoeff

    def get_rendering(self):
        if threading.current_thread() != self._pybullet_controller._pybullet_thread:
             # Couldn't find info about this in the docs, but I feel like it could be an issue. And I actually did get some segfaults
            ggLog.warn(f"Rendering on thread different from startup PyBullet thread. This may be a problem.")
        width, height, rgb, depth, segmentation = pybullet.getCameraImage(  width = self._width,
                                                                            height = self._height,
                                                                            viewMatrix = self._extrinsic_matrix,
                                                                            projectionMatrix = self._intrinsic_matrix,
                                                                            shadow=self._enable_shadows,
                                                                            lightDirection=self._lightDirection,
                                                                            lightColor = self._lightColor,
                                                                            lightDistance = self._lightDistance,
                                                                            lightAmbientCoeff = self._lightAmbientCoeff,
                                                                            lightDiffuseCoeff = self._lightDiffuseCoeff,
                                                                            lightSpecularCoeff = self._lightSpecularCoeff)
        img = np.array(rgb).reshape(height,width,4)[:,:,0:3]
        return img












class PyBulletAdapter(BaseAdapter, BaseJointEffortAdapter, BaseSimulationAdapter, BaseJointPositionAdapter, BaseJointVelocityAdapter):
    """This class allows to control the execution of a PyBullet simulation.

    """

    def __init__(self, stepLength_sec : float = 1/240,
                        restore_on_reset = True,
                        debug_gui : bool = False,
                        real_time_factor : Optional[float] = None,
                        global_max_torque_position_control : float = 100,
                        joints_max_torque_position_control : Dict[Tuple[str,str],float] = {},
                        global_max_velocity_position_control : float = 1,
                        joints_max_velocity_position_control : Dict[Tuple[str,str],float] = {},
                        global_max_acceleration_position_control : float = 10,
                        joints_max_acceleration_position_control : Dict[Tuple[str,str],float] = {},
                        simulation_step = 1/960):
        """Initialize the Simulator controller.


        """
        super().__init__()
        self._stepLength_sec = stepLength_sec
        self._simulation_step = simulation_step
        if self._stepLength_sec % self._simulation_step != 0:
            ggLog.warn(f"{__class__}: stepLength_sec {self._stepLength_sec} is not a multiple of "
                       f"simulation_step {self._simulation_step}, will not be able to sep of exactly stepLength_sec.")

        # Model names are the ones we use to uniquely identify bodies and links as (model_id, link_id)
        # Body ids are the ones pybullet uses to uniquely identify bodies
        # Body names are the names used by pybullet, I think they just come from the URDF, they can be the same for different bodies
        self._modelName_to_bodyId : Dict[str,int] = {"0":0} 
        self._bodyId_to_modelName : Dict[int,str]= {0:"0"}
        self._cameras = {}
        self._commanded_torques_by_body = {}
        self._commanded_torques_by_name = {}
        self._commanded_velocities_by_body : Dict[int, Tuple[List[int], List[float]]] = {}
        self._last_step_commanded_torques_by_name = {}
        self._commanded_trajectories = {}
        self._debug_gui = debug_gui
        self._reset_detected_contacts()
        if real_time_factor is not None and real_time_factor>0:
            self._real_time_factor = real_time_factor
        else:
            self._real_time_factor = None
        
        self._max_joint_velocities_pos_control = joints_max_velocity_position_control
        self._max_joint_velocity_pos_control = global_max_velocity_position_control
        self._max_joint_accelerations_pos_control = joints_max_acceleration_position_control
        self._max_joint_acceleration_pos_control = global_max_acceleration_position_control
        self._max_torques_pos_control = joints_max_torque_position_control
        self._max_torque_pos_control = global_max_torque_position_control
        self._restore_on_reset = restore_on_reset
        self._stepping_wtime_since_build = 0
        self._stepping_stime_since_build = 0
        self._freerun_time_since_build = 0
        self._build_time = time.monotonic()
        self._verbose = False
        self._monitored_contacts = []

        self.setupLight(    lightDirection = [1,1,1],
                            lightColor = [0.9,0.9,0.9],
                            lightDistance = 100,
                            enable_shadows = True,
                            lightAmbientCoeff = 0.8,
                            lightDiffuseCoeff = 0.5,
                            lightSpecularCoeff = 0.1)

        self.clear_commands()

    def _refresh_entities_ids(self, print_info = False):
        bodyIds = []
        for i in range(pybullet.getNumBodies()):
            bodyIds.append(pybullet.getBodyUniqueId(i))

        self._bodyAndJointIdToJointName = {}
        self._jointNamesToBodyAndJointId = {}
        self._bodyLinkIds_to_linkName = {}
        self._linkName_to_bodyLinkIds = {}
        dynamics_infos = ["mass","lat_frict","loc_inertia_diag","loc_inertial_pos","loc_inertial_orn","restitution","roll_friction","spin_friction","contact_damping","contact_stiffness","body_type","collision_margin"]
        for bodyId in bodyIds:
            base_link_name, _ = pybullet.getBodyInfo(bodyId)
            model_name = self._bodyId_to_modelName[bodyId]
            base_link_name = (model_name, base_link_name.decode("utf-8"))
            base_body_and_link_id = (bodyId, -1)
            if print_info:
                ggLog.info(f"Found base link {base_link_name}, with bodyid,link_id {base_body_and_link_id}")
                ggLog.info(f"DynamicsInfo: {list(zip(dynamics_infos,pybullet.getDynamicsInfo(bodyId,-1)))}")
            self._linkName_to_bodyLinkIds[base_link_name] = base_body_and_link_id
            self._bodyLinkIds_to_linkName[base_body_and_link_id] = base_link_name
            for jointId in range(pybullet.getNumJoints(bodyId)):
                jointInfo = pybullet.getJointInfo(bodyId,jointId)
                jointName = (model_name, jointInfo[1].decode("utf-8"))
                linkName = (model_name, jointInfo[12].decode("utf-8"))
                body_and_joint_ids = (bodyId,jointId)
                self._bodyAndJointIdToJointName[body_and_joint_ids] = jointName
                self._jointNamesToBodyAndJointId[jointName] = body_and_joint_ids
                self._bodyLinkIds_to_linkName[body_and_joint_ids] = linkName
                self._linkName_to_bodyLinkIds[linkName] = body_and_joint_ids
                if print_info:
                    ggLog.info(f"  Found regular link {linkName}, with bodyid,link_id/joint_id {body_and_joint_ids}")
                    ggLog.info(f"    DynamicsInfo: {list(zip(dynamics_infos,pybullet.getDynamicsInfo(bodyId,jointId)))}")
                    ggLog.info(f"    JointInfo of {jointName} : "+str(pybullet.getJointInfo(bodyId,jointId)))


        # ggLog.info("self._bodyAndJointIdToJointName = "+str(self._bodyAndJointIdToJointName))
        # ggLog.info("self._jointNamesToBodyAndJointId = "+str(self._jointNamesToBodyAndJointId))
        # ggLog.info("self._bodyAndLinkIdToLinkName = "+str(self._bodyAndLinkIdToLinkName))
        # ggLog.info("self._linkNameToBodyAndLinkId = "+str(self._linkNameToBodyAndLinkId))


    def startup(self):
        if not pybullet.isConnected():
            raise ValueError("PyBullet is not connected")

        self._refresh_entities_ids()
        self._startStateId = pybullet.saveState()
        self._simTime = 0


    def _getJointName(self, bodyId, jointIndex):
        jointName = self._bodyAndJointIdToJointName[(bodyId,jointIndex)]
        return jointName

    def _getBodyAndJointId(self, jointName):
        return self._jointNamesToBodyAndJointId[jointName] # TODO: this ignores the model name, should use it

    def _getLinkName(self, bodyId, linkIndex):
        return self._bodyLinkIds_to_linkName[(bodyId,linkIndex)]

    def _getBodyAndLinkId(self, linkName):
        return self._linkName_to_bodyLinkIds[linkName]

    def resetWorld(self):
        self.clear_commands()
        # ggLog.info(f"Resetting...")
        if self._restore_on_reset:
            pybullet.restoreState(self._startStateId)
        # ggLog.info(f"Resetted")
        self._refresh_entities_ids()
        self._simTime = 0
        self._prev_step_end_wall_time = time.monotonic()
        super().resetWorld()
        if self._verbose:
            ggLog.info(f"tot_step_stime = {self._stepping_stime_since_build}s, tot_step_wtime = {self._stepping_wtime_since_build}s, tot_wtime = {time.monotonic()-self._build_time}s, tot_freerun_wtime = {self._freerun_time_since_build}s")

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

        stepLength = self.freerun(self._stepLength_sec)
        self.clear_commands()
        return stepLength

    def freerun(self, duration_sec: float):
        tf0 = time.monotonic()

        self._reset_detected_contacts()
        #pybullet.setTimeStep(self._stepLength_sec) #This is here, but still, as stated in the pybulelt quickstart guide this should not be changed often
        self._last_step_commanded_torques = []

        # ggLog.info(f"PyBullet doing {duration_sec}/{self._bullet_stepLength_sec}={simsteps} steps")
        stepping_wtime = 0
        t0 = self._simTime
        while self._simTime-t0 < duration_sec:
            self._apply_controls()
            wtps = time.monotonic()
            pybullet.stepSimulation()
            self._read_new_contacts()
            stepping_wtime += time.monotonic()-wtps
            self._simTime += self._bullet_stepLength_sec
            if self._real_time_factor is not None and self._real_time_factor>0:
                sleep_time = self._bullet_stepLength_sec - (time.monotonic()-self._prev_step_end_wall_time)
                # print(f" sleep_time = {sleep_time}")
                if sleep_time > 0:
                    time.sleep(sleep_time*(1/self._real_time_factor))
            self._prev_step_end_wall_time = time.monotonic()
        self._stepping_wtime_since_build += stepping_wtime
        self._stepping_stime_since_build += self._simTime - t0
        self._freerun_time_since_build += time.monotonic()-tf0

        self._last_step_commanded_torques_by_name = {}
        for jn, t in self._last_step_commanded_torques:
            # ggLog.info(f"self._last_step_commanded_torques: {jn}, {t}")
            if jn not in self._last_step_commanded_torques_by_name:
                self._last_step_commanded_torques_by_name[jn] = []
            self._last_step_commanded_torques_by_name[jn].append(t)
        # ggLog.info(f"self._last_step_commanded_torques_by_name: {self._last_step_commanded_torques_by_name}")
        self._last_step_commanded_torques_by_name = {jn:sum(torque_list)/len(torque_list) for jn, torque_list in self._last_step_commanded_torques_by_name.items()}    
        self._last_step_commanded_torques_by_name.update(self._commanded_torques_by_name) # direct torque commands override other applied torques
        return self._simTime-t0
    
    def clear_commands(self):
        self._commanded_torques_by_body = {}
        self._commanded_torques_by_name = {}
        self._commanded_velocities_by_body = {}
        self._commanded_trajectories = {}


    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[np.ndarray, float]]:
        # ggLog.info(f"Rendering")
        ret = {}
        for cam_name in requestedCameras:
            camera = self._cameras[cam_name]
            linkstate = self.getLinksState([camera.link_name], use_com_frame=True)[camera.link_name]
            camera.set_pose(linkstate.pose)
            camera.setup_light(
                                lightDirection = self._lightDirection,
                                lightColor = self._lightColor,
                                lightDistance = self._lightDistance,
                                enable_shadows = self._enable_shadows,
                                lightAmbientCoeff = self._lightAmbientCoeff,
                                lightDiffuseCoeff = self._lightDiffuseCoeff,
                                lightSpecularCoeff = self._lightSpecularCoeff)
            ret[cam_name] = ((camera.get_rendering(), self.getEnvTimeFromReset()))
        return ret

    def _build_trajectory(self, t0, p0, v0, pf, samples, max_vel, max_acc):

        # TODO: implement v0 use

        t_a = max_vel/max_acc # acceleration duration
        x_a = max_vel*max_vel/(2*max_acc) # space to accelerate to max_vel
        d = abs(pf-p0)
        if d < 0.0001:
            trajectory_tpva = [(t0, pf, 0, 0)]
            return np.array(trajectory_tpva, dtype = np.float64)
        if d <= 2*x_a:
            x_a = d/2
            t_a = np.sqrt(((2*x_a)/max_acc))
            max_vel = t_a*max_acc
            t_coast = 0
        else:
            t_coast = (d-2*x_a)/max_vel # time spent at max speed
        t_f = 2*t_a + t_coast # end time
        t_d = t_f-t_a # time when deceleration starts
        x_d = d-x_a # position where deceleration starts
        if t_d<t_a: # may happen for numerical reasons
            t_d = t_a
        if np.isnan(t_f):
            raise RuntimeError(f"t_f is nan: t_f={t_f}, t_a={t_a}, d={d}, x_a={x_a}, max_vel={max_vel}, max_acc={max_acc}, pf={pf}, p0={p0}")
        # ggLog.info(f"{t0}, {p0}, {v0}, {pf}, {samples}, {max_vel}, {max_acc} : {t_a}, {x_a}, {t_coast}, {t_f}, {t_d}, {x_d}, {d}")
        samples = int(t_f*100)+1 #100Hz plus one final sample
        trajectory_tpva = [None] * samples
        for i in range(samples-1):
            t = t_f*(i/samples)
            if t <= t_a:
                x = 0.5*max_acc*t*t
                v = max_acc*t
                a = max_acc
            elif t>=t_d:
                td = t - t_d # time since deceleration start
                x = x_d + (max_vel*td - 0.5*max_acc*td*td)
                v = max_vel-max_acc*td
                a = -max_acc
            elif t >= t_a and t<=t_d:
                x = x_a + (t-t_a)*max_vel
                v = max_vel
                a = 0
            else:
                raise RuntimeError(f"This should not happen: t = {t}, t_a={t_a}, t_d={t_d}, p0={p0}, pf={pf}")
            trajectory_tpva[i] = (t, x, v, a)
        trajectory_tpva[-1] = (0, 0, 0, 0)
        if pf-p0<0:
            trajectory_tpva = [[t,-p,-v,-a] for t,p,v,a in trajectory_tpva]
        trajectory_tpva = [[t+t0,p+p0,v,a] for t,p,v,a in trajectory_tpva]
        trajectory_tpva[-1] = (t_f+t0, pf, 0, 0)
        return np.array(trajectory_tpva, dtype = np.float64)
            


    def _apply_controls(self):
        self._apply_commanded_torques()
        self._apply_commanded_velocities()
        self._apply_commanded_positions()

    def _apply_commanded_torques(self):
        for bodyId in self._commanded_torques_by_body.keys():
            pybullet.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=self._commanded_torques_by_body[bodyId][0],
                                        controlMode=pybullet.TORQUE_CONTROL,
                                        forces=self._commanded_torques_by_body[bodyId][1])
            
    
    def _apply_commanded_velocities(self):
        for bodyId, (jointIds, velocities) in self._commanded_velocities_by_body.items():
            max_torques = [min(self._max_torque_pos_control,
                                self._max_torques_pos_control.get(self._getJointName(bodyId, jointId),float("+inf")))
                                for jointId in jointIds]
            pybullet.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=jointIds,
                                        controlMode=pybullet.VELOCITY_CONTROL,
                                        targetVelocities=velocities,
                                        forces=max_torques
                                        )

    def _apply_commanded_positions(self):
        t = self.getEnvTimeFromStartup()
        for bodyId, joint_trajectories in self._commanded_trajectories.items():
            jointIds, positions, velocities, forces, pos_gains, vel_gains = ([],[],[],[],[],[])
            sample_time = None
            for jointId, traj_tpva in joint_trajectories:
                jointName = self._getJointName(bodyId, jointId)
                max_torque = min(self._max_torque_pos_control,
                                self._max_torques_pos_control.get(jointName,float("+inf")))
                sample_idx = np.searchsorted(traj_tpva[:,0], t) # index of the next trajectory sample (the first with time higher of t)
                if sample_idx>=traj_tpva.shape[0]:
                    sample_idx = traj_tpva.shape[0]-1
                time, pos, vel, acc = traj_tpva[sample_idx]
                jointIds.append(jointId)
                positions.append(pos)
                velocities.append(vel)
                forces.append(max_torque)
                pos_gains.append(None)
                vel_gains.append(None)
                sample_time = time
            # ggLog.info(f"Setting joint control for {bodyId}, sample={sample_time}: {body_commad}")
            pybullet.setJointMotorControlArray(bodyIndex=bodyId,
                                        jointIndices=jointIds,
                                        controlMode=pybullet.POSITION_CONTROL,
                                        targetPositions=positions,
                                        targetVelocities=velocities,
                                        forces=forces,
                                        # positionGains=pos_gains,
                                        # velocityGains=vel_gains
                                        )

    def setJointsEffortCommand(self, jointTorques : List[Tuple[Tuple[str,str],float]]) -> None:
        #For each bodyId I submit a request for joint motor control
        requests = {}
        for joint_name, torque in jointTorques:
            bodyId, jointId = self._getBodyAndJointId(joint_name)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont
            requests[bodyId][1].append(torque) #requested torque
        self._commanded_torques_by_name = {jn:t for jn,t in jointTorques}
        self._commanded_torques_by_body = requests

    def setJointsVelocityCommand(self, jointVelocities : List[Tuple[Tuple[str,str],float]]) -> None:
        #For each bodyId I submit a request for joint motor control
        requests : Dict[int, Tuple[List[int], List[float]]] = {}
        for joint_name, velocity in jointVelocities:
            bodyId, jointId = self._getBodyAndJointId(joint_name)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = ([],[])
            requests[bodyId][0].append(jointId) #requested jont
            requests[bodyId][1].append(velocity) #requested velocity
        # self._commanded_velocities_by_name = {jn:t for jn,t in jointVelocities}
        self._commanded_velocities_by_body = requests

    def setJointsPositionCommand(self,  jointPositions : Dict[Tuple[str,str],float],
                                        velocity_scaling : float = 1.0,
                                        acceleration_scaling : float = 1.0) -> None:
        requests = {}
        jointStates = self.getJointsState(requestedJoints=jointPositions.keys())
        t0 = self.getEnvTimeFromStartup()
        for joint, req_position in jointPositions.items():
            max_acceleration = min( self._max_joint_acceleration_pos_control,
                                    self._max_joint_accelerations_pos_control.get(joint,float("+inf")))
            max_velocity = min(self._max_joint_velocity_pos_control,
                               self._max_joint_velocities_pos_control.get(joint,float("+inf")))
            bodyId, jointId = self._getBodyAndJointId(joint)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            p0 = jointStates[joint].position[0]
            traj_tpva = self._build_trajectory(t0 = t0,
                                               p0 = p0,
                                               v0 = jointStates[joint].rate[0],
                                               pf = req_position,
                                               samples = 100,
                                               max_vel = max_velocity*velocity_scaling,
                                               max_acc = max_acceleration*acceleration_scaling)
            requests[bodyId].append((jointId, traj_tpva))
            # requests[bodyId][0].append(jointId) #requested joint
            # requests[bodyId][1].append(req_position) #requested position
            # requests[bodyId][2].append(0) #requested velocity
            # requests[bodyId][3].append(max_torque*acceleration_scaling) # Scale with the acceleration, kinda the same... :)
            # requests[bodyId][4].append(positionGain)
            # requests[bodyId][5].append(velocityGain)
            # np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
            # ggLog.info(f"Built trajectory from {p0} to {req_position} for joint {joint} at t0={t0}:\n{traj_tpva}")

        self._commanded_trajectories = requests


    def moveToJointPoseSync(self,   jointPositions : Dict[Tuple[str,str],float],
                                    velocity_scaling : float = 1.0,
                                    acceleration_scaling : float = 1.0,
                                    max_error : float = 0.001,
                                    step_time : Optional[float] = 0.1,
                                    wall_timeout_sec : float = 30,
                                    sim_timeout_sec : float = 30) -> None:
        if step_time is None:
            step_time = self._stepLength_sec
        joints = list(jointPositions.keys())
        req_pos = np.array([jointPositions[k] for k in joints])
        jstates = self.getJointsState(joints)
        curr_pos = np.array([jstates[k].position[0] for k in joints])

        t0_wall = time.monotonic()
        t0_sim = self.getEnvTimeFromStartup()

        self.setJointsPositionCommand(jointPositions=jointPositions,
                                        velocity_scaling=velocity_scaling,
                                        acceleration_scaling=acceleration_scaling)
        while np.max(req_pos - curr_pos) > max_error:
            self.freerun(duration_sec=step_time)
            jstates = self.getJointsState(joints)
            curr_pos = np.array([jstates[k].position[0] for k in joints])
            wall_d = time.monotonic()-t0_wall
            if wall_d > wall_timeout_sec:
                raise TimeoutError(f"wall timeout: {wall_d} > {wall_timeout_sec}")
            sim_d = time.monotonic()-t0_sim
            if sim_d > sim_timeout_sec:
                raise TimeoutError(f"sim tomeout: {sim_d} > {sim_timeout_sec}")

    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        #For each bodyId I submit a request for joint state
        requests = {} #for each body id we will have a list of joints
        for jn in requestedJoints:
            bodyId, jointId = self._getBodyAndJointId(jn)
            if bodyId not in requests: #If we haven't created a request for this body yet
                requests[bodyId] = []
            requests[bodyId].append((jointId, jn)) #requested jont

        allStates = {}
        for bodyId in requests.keys():#for each bodyId make a request
            jids = [r[0] for r in requests[bodyId]]
            jointStates = pybullet.getJointStates(bodyId,jids)
            for i in range(len(requests[bodyId])):#put the responses of this bodyId in allStates
                jointId, joint_name = requests[bodyId][i]
                pos, vel, effort = jointStates[i][0], jointStates[i][1], jointStates[i][3]
                # if the joint is commanded in torque the jointStates effort will be zero. But the actual effort is by definition the comanded one
                if joint_name in self._last_step_commanded_torques_by_name:
                    effort = self._last_step_commanded_torques_by_name[joint_name]
                allStates[self._getJointName(bodyId,jointId)] = JointState([pos], [vel], [effort])


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
            pybullet.resetJointStatesMultiDof(bodyId, ids, [s.position for s in states], [s.rate for s in states])



    def getLinksState(self, requestedLinks : List[Tuple[str,str]], use_com_frame : bool = False) -> Dict[Tuple[str,str],LinkState]:
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
            bodyStates = pybullet.getLinkStates(bodyId,requests[bodyId],computeLinkVelocity=1, computeForwardKinematics=1)
            for i in range(len(requests[bodyId])):#put the responses of this bodyId in allStates
                #print("bodyStates["+str(i)+"] = "+str(bodyStates[i]))
                linkId = requests[bodyId][i]
                if use_com_frame:
                    linkState = LinkState(  position_xyz =     bodyStates[i][0][:3],
                                            orientation_xyzw = bodyStates[i][1][:4],
                                            pos_velocity_xyz = bodyStates[i][6][:3],
                                            ang_velocity_xyz = bodyStates[i][7][:3])
                else:
                    # raise NotImplementedError()
                    linkState = LinkState(  position_xyz =     bodyStates[i][4][:3],
                                            orientation_xyzw = bodyStates[i][5][:4],
                                            pos_velocity_xyz = bodyStates[i][6][:3], # this is the com velocity!
                                            ang_velocity_xyz = bodyStates[i][7][:3])
                allStates[self._getLinkName(bodyId,linkId)] = linkState
        for bodyId in baserequests: #for each bodyId make a request
            # ggLog.info(f"Getting pose of body {bodyId}")
            bodyPose = pybullet.getBasePositionAndOrientation(bodyId)
            bodyVelocity = pybullet.getBaseVelocity(bodyId)
            if use_com_frame:
                linkState = LinkState(  position_xyz = bodyPose[0][:3],
                                        orientation_xyzw = bodyPose[1][:4],
                                        pos_velocity_xyz = bodyVelocity[0][:3],
                                        ang_velocity_xyz = bodyVelocity[1][:3])
            else:
                # These are expressed in the center-of-mass frame, we need to convert them to use the urdf frame
                local_inertia_pos, local_inertia_orient = pybullet.getDynamicsInfo(bodyId,-1)[3:5]
                # pybullet.multiplyTransform
                raise NotImplementedError()
        
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
                raise RuntimeError(f"Can only set pose for base links, but requested to move link {ln} (base link is {self._getLinkName(bodyId,-1)})")
            requests[bodyId].append(ls)

        for bodyId, states in requests.items():
            for state in states:
                pybullet.resetBasePositionAndOrientation(bodyId, state.pose.position, state.pose.orientation_xyzw)

    def getEnvTimeFromStartup(self) -> float:
        return self._simTime

    def build_scenario(self, file_path, format = "urdf"):
        if self._debug_gui:
            self._client_id = pybullet.connect(pybullet.GUI)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
        else:
            self._client_id = pybullet.connect(pybullet.DIRECT)
            plugin = pybullet.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._pybullet_thread = threading.current_thread()
        # ggLog.info("Started pybullet")
        pybullet.setGravity(0, 0, -9.8)
        # p.setDefaultContactERP(0.9)
        #print("self.numSolverIterations=",self.numSolverIterations)
        pybullet.setPhysicsEngineParameter( fixedTimeStep=self._simulation_step, # Originally it was 1/240, with numSubSteps = 4
                                    numSolverIterations=5,
                                    numSubSteps=1, # using substeps breakks contacts detection (as the funciton only returns the last substep information)
                                    enableFileCaching=0)

        ggLog.info("Physics engine parameters:"+str(pybullet.getPhysicsEngineParameters()))

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeObjId = pybullet.loadURDF("plane.urdf")
        pybullet.changeDynamics(planeObjId, -1, lateralFriction=0.8, restitution=0.5)
        adarl.utils.sigint_handler.setupSigintHandler()
        if file_path is not None:
            self.spawn_model(model_file = file_path, model_format=format, model_name = "scenario")
        self._bullet_stepLength_sec = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]


        if self._stepLength_sec % pybullet.getPhysicsEngineParameters()["fixedTimeStep"] > 0.00001:
            ggLog.warn(f"PyBulletAdapter: stepLength_sec is not a multiple of pybullet's fixedTimeStep (respecively {self._stepLength_sec} and {pybullet.getPhysicsEngineParameters()['fixedTimeStep']})")

        if self._stepLength_sec<pybullet.getPhysicsEngineParameters()["fixedTimeStep"]:
            raise RuntimeError(f"Requested stepLength_sec {self._stepLength_sec} is less than requested simulation_step {self._simulation_step}")
        

    def destroy_scenario(self):
        self._modelName_to_bodyId = {}
        pybullet.resetSimulation()
        pybullet.disconnect(self._client_id)

    def _register_camera(self, camera : BulletCamera):
        self._cameras[camera.camera_name] = camera
        ggLog.info(f"Registered camera with name {camera.camera_name} at link {camera.link_name}")

    def _loadModel(self, modelFilePath : str, fileFormat : str = "urdf", model_kwargs = {}, model_name = None):
        compiled_file_path = None
        if fileFormat.split(".")[-1] == "xacro":
            model_definition_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(modelFilePath).read_text(),
                                                                                model_kwargs=model_kwargs)
            compiled_file_path = f"/tmp/adarl_PyBulletUtils_xacrocompile_{int(time.time()*1000000)}_{os.getpid()}_{hash(modelFilePath)}"
            Path(compiled_file_path).write_text(model_definition_string)
            modelFilePath = compiled_file_path
            fileFormat = ".".join(fileFormat.split(".")[:-1])
        
        if fileFormat == "urdf":
            bodyId = pybullet.loadURDF(modelFilePath, flags=pybullet.URDF_USE_SELF_COLLISION|pybullet.URDF_PRINT_URDF_INFO)
        elif fileFormat == "mjcf":
            bodyId = pybullet.loadMJCF(modelFilePath, flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0]
        elif fileFormat == "sdf":
            bodyId = pybullet.loadSDF(modelFilePath)[0]
        else:
            raise AttributeError("Invalid format "+str(fileFormat))

        #pybullet.changeDynamics(bodyId, -1, linearDamping=0, angularDamping=0)
        for j in range(pybullet.getNumJoints(bodyId)):
            #pybullet.changeDynamics(bodyId, j, linearDamping=0, angularDamping=0)
            pybullet.setJointMotorControl2(bodyId,
                                    j,
                                    controlMode=pybullet.POSITION_CONTROL,
                                    targetPosition=0,
                                    targetVelocity=0,
                                    positionGain=0.1,
                                    velocityGain=0.1,
                                    force=0) # Disable any position control motor by setting its max torque to zero
            pybullet.setJointMotorControl2(bodyIndex=bodyId,
                                    jointIndex=j,
                                    controlMode=pybullet.VELOCITY_CONTROL,
                                    force=0) # Disable velocity motor by setting its max torque to zero
            

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
            if len(models)>1:
                raise RuntimeError(f"sdf files with more than one model are not supported right now, maybe it's easy to fix, maybe not.")
            for model in models:
                # model_name = model["@name"]
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
                                                               camera_name=sensor["@name"],
                                                               pybullet_controller=self))

        if compiled_file_path is not None:
            Path(compiled_file_path).unlink() # delete the compiled file
        return bodyId

    def spawn_model(self,   model_definition_string : Optional[str] = None,
                            model_format : Optional[str] = None,
                            model_file : Optional[str]= None,
                            model_name : Optional[str] = None,
                            pose : Optional[Pose] = None,
                            model_kwargs: Dict[Any, Any] = {}) -> str:
        if model_definition_string is not None:
            raise AttributeError(f"Only file model descriptions are supported")
        if model_name in self._modelName_to_bodyId:
            raise AttributeError(f"model name {model_name} is already present")
        body_id = self._loadModel(modelFilePath=model_file, fileFormat=model_format, model_kwargs=model_kwargs, model_name = model_name)
        self._modelName_to_bodyId[model_name] = body_id
        self._bodyId_to_modelName[body_id] = model_name
        self._refresh_entities_ids(print_info=True)
        ggLog.info(f"Spawned model '{model_name}' with body_id {body_id} and info {pybullet.getBodyInfo(self._modelName_to_bodyId[model_name])}")
        if pose is not None:
            pybullet.resetBasePositionAndOrientation(self._modelName_to_bodyId[model_name],
                                                    pose.position, pose.orientation_xyzw)
        return model_name

    def delete_model(self, model_name: str):
        pybullet.removeBody(self._modelName_to_bodyId[model_name])
        self._modelName_to_bodyId.pop(model_name)
        self._refresh_entities_ids()

    def setupLight(self,    lightDirection,
                            lightColor,
                            lightDistance,
                            enable_shadows,
                            lightAmbientCoeff,
                            lightDiffuseCoeff,
                            lightSpecularCoeff):
        self._lightDirection = lightDirection
        self._lightColor = lightColor
        self._lightDistance = lightDistance
        self._enable_shadows = enable_shadows
        self._lightAmbientCoeff = lightAmbientCoeff
        self._lightDiffuseCoeff = lightDiffuseCoeff
        self._lightSpecularCoeff = lightSpecularCoeff
    
    def monitor_contacts(self, monitored_contacts : List[Tuple[Optional[str],
                                                            Optional[str],
                                                            Optional[Tuple[str,str]],
                                                            Optional[Tuple[str,str]]]]):
        self._monitored_contacts = monitored_contacts

    def get_contacts(self) -> List[List[    Tuple[  Tuple[str,str],
                                                    Tuple[str,str],
                                                    Tuple[float,float,float],
                                                    float,
                                                    float]]]:
        """Returns the list of the contact readings for all the simulation steps in the last env step.
        """
        return self._detected_contacts

    def _reset_detected_contacts(self):
        self._detected_contacts : List[List[Tuple[Tuple[str,str],
                                                  Tuple[str,str],
                                                  Tuple[float,float,float],
                                                  float,
                                                  float]]] = []

    def _read_new_contacts(self):
        new_contacts = []
        for allowed_contacts in self._monitored_contacts:
            contacts = self._get_contacts(*allowed_contacts)
            new_contacts += contacts
        self._detected_contacts.append(new_contacts)

    def _get_contacts(self, 
                     model_a : Optional[str],
                     model_b : Optional[str],
                     link_a : Optional[Tuple[str,str]],
                     link_b : Optional[Tuple[str,str]]) -> List[Tuple[Tuple[str,str],Tuple[str,str],Tuple[float,float,float],float]]:
        if model_b and not model_a:
            raise ValueError(f"model_b should only be set if model_a is set")
        if link_b and not (link_a or model_a):
            raise ValueError(f"link_b should only be set if link_a or model_a are set")
        if link_a and model_a:
            raise ValueError(f"can only set one of model_a or link_a")
        if link_b and model_b:
            raise ValueError(f"can only set one of model_b or link_b")
        link_a_id, link_b_id = None, None
        if link_a:
            model_a = link_a[0]
            link_a_id = self._linkName_to_bodyLinkIds[link_a]
        if link_b:
            model_b = link_b[0]
            link_b_id = self._linkName_to_bodyLinkIds[link_b]

        body_a_id = self._modelName_to_bodyId[model_a] if model_a is not None else None
        body_b_id = self._modelName_to_bodyId[model_b] if model_b is not None else None
        # ggLog.info(f"bodyA = {body_a_id}, bodyB = {body_b_id}, linkIndexA = {link_a_id}, linkIndexB = {link_b_id}")
        kwargs = {}
        if body_a_id is not None: kwargs["bodyA"] = body_a_id
        if body_b_id is not None: kwargs["bodyB"] = body_b_id
        if link_a_id is not None: kwargs["linkIndexA"] = link_a_id
        if link_b_id is not None: kwargs["linkIndexB"] = link_b_id
        cpoints = pybullet.getContactPoints(**kwargs)
        ret = []
        for cp in cpoints:
            link1 = self._bodyLinkIds_to_linkName[(cp[1], cp[3])]
            link2 = self._bodyLinkIds_to_linkName[(cp[2], cp[4])]
            normal_2to1_xyz = cp[7]
            force = cp[9]
            duration = self._bullet_stepLength_sec
            ret.append((link1,
                        link2,
                        normal_2to1_xyz,
                        force,
                        duration))
        return ret

