#!/usr/bin/env python3

import pinocchio
import numpy as np
import adarl.utils.utils
from pathlib import Path
from typing import Literal
import copy
from typing import Iterable
import itertools
import faulthandler
from pinocchio.visualize import GepettoVisualizer
faulthandler.enable()

class Robot():
    def __init__(self, model_urdf_string : str):
        self._urdf_string = model_urdf_string
        self._model = pinocchio.buildModelFromXML(model_urdf_string)
        self._collision_geom_model = pinocchio.buildGeomFromUrdfString(self._model,
                                                                       self._urdf_string,
                                                                       pinocchio.GeometryType.COLLISION)
        self._model_data = self._model.createData()
        self._joint_position = pinocchio.randomConfiguration(self._model)
        self._joint_position = np.zeros(shape=(self._model.njoints-1,))
        self._collision_object_count = 0

        self._joint_names = [str(n) for n in self._model.names]
        self._joints_num = len(self._joint_names)
        self._joint_name_to_idx = {n:self._joint_names.index(n) for n in self._joint_names}
        self._joint_idx_to_name = {idx:name for name,idx in self._joint_name_to_idx.items()}

        self._frame_names = [frame.name for frame in self._model.frames]
        self._frame_name_to_idx = {n:self._frame_names.index(n) for n in self._frame_names}
        self._frame_idx_to_name = {idx:name for name,idx in self._frame_name_to_idx.items()}
        self._joints_to_frame_names = {}
        for i in range(len(self._joint_names)):
            jname = self._model.names[i]
            self._joints_to_frame_names[jname] = []
            for link in self._model.frames:
                if link.parent == i:
                    self._joints_to_frame_names[jname].append(link.name)
        self._joint_to_geoms = {frame:[] for frame in self._joint_names}
        for geom_obj in self._collision_geom_model.geometryObjects:
            self._joint_to_geoms[self._joint_idx_to_name[geom_obj.parentJoint]].append(geom_obj.name)
        self._need_to_recompute_forward_kin = True
        self._need_to_place_geoms = True
        self._current_collision_geom_pairs = set()
        self.set_collision_pairs("all")

    def set_collision_pairs(self, geom_pairs : Iterable[tuple[str,str]] | Literal['all'] = []):
        self._collision_pairs = copy.deepcopy(geom_pairs)
        geoms_num = self._collision_geom_model.ngeoms
        geom_names = [g.name for g in self._collision_geom_model.geometryObjects]
        if geom_pairs == "all":
            geom_pairs = [(g1,g2) for g1 in geom_names for g2 in geom_names]
        self._current_collision_geom_pairs = set(geom_pairs)
        collision_matrix = np.zeros(shape=(geoms_num, geoms_num), dtype=bool)
        for pair in self._current_collision_geom_pairs:
            collision_matrix[geom_names.index(pair[0]), geom_names.index(pair[1])] = True
            collision_matrix[geom_names.index(pair[1]), geom_names.index(pair[0])] = True # just set both pairs, to be safe
        self._collision_geom_model.setCollisionPairs(collision_matrix)
        self._collision_geom_model_data = pinocchio.GeometryData(self._collision_geom_model)

    def add_collision_pairs(self, geom_pairs : Iterable[tuple[str,str]]):
        geom_pairs = self._current_collision_geom_pairs.union(geom_pairs)
        self.set_collision_pairs(geom_pairs)

    def remove_collision_pairs(self, geom_pairs : Iterable[tuple[str,str]]):
        geom_pairs = set(geom_pairs).union([(g2,g1) for g1,g2 in geom_pairs]) # always also the flipped pair
        geom_pairs = self._current_collision_geom_pairs.difference(geom_pairs)
        self.set_collision_pairs(geom_pairs)

    def set_collision_pairs_from_frames(self, frame_pairs : list[tuple[str,str]]):
        frame_to_geoms = {frame:[] for frame in self._model.frames}
        for geom_obj in self._collision_geom_model.geometryObjects:
            frame_to_geoms[self._frame_idx_to_name[geom_obj.parentFrame]].append(geom_obj.name)
        geom_pairs = set()
        for pair in frame_pairs:
            geoms1 = frame_to_geoms[pair[0]]
            geoms2 = frame_to_geoms[pair[1]]
            for g1 in geoms1:
                for g2 in geoms2:
                    geom_pairs.add((g1,g2))
            geom_pairs.update([(g1,g2) for g1 in geoms1 for g2 in geoms2])
        return self.set_collision_pairs(list(geom_pairs))
    
    def get_geoms_under_joints(self, joints : list[str]):
        return [self._joint_to_geoms[j] for j in joints]
    
    def get_tree_joint_names_under_joint(self, joint_name : str) -> list[str]:
        return [self._joint_idx_to_name[jid] for jid in self._model.subtrees[self._joint_name_to_idx[joint_name]]]

    def get_conjoined_pairs(self):
        geom_groups = self.get_geoms_under_joints(self._joint_names)
        all_pairs = []
        for group in geom_groups:
            all_pairs += [(g1,g2) for g1 in group for g2 in group]
        return all_pairs


    def add_collision_box(self,  pose_xyz_xyzw : np.ndarray,
                                    collision_box_size_xyz : tuple[float,float,float],
                                    reference_frame = None,
                                    colliding_geoms : Iterable[str] | Literal["all"] = "all"):
        return self.add_collision_object(pose_xyz_xyzw = pose_xyz_xyzw,
                                  collision_geometry=pinocchio.hppfcl.Box(*collision_box_size_xyz),
                                  reference_frame=reference_frame,
                                  colliding_geoms=colliding_geoms)
        
    def add_collision_object(self,  pose_xyz_xyzw : np.ndarray,
                                    collision_geometry : pinocchio.hppfcl.CollisionGeometry,
                                    reference_frame = None,
                                    colliding_geoms : Iterable[str] | Literal["all"] = "all"):
        if reference_frame is not None:
            raise NotImplementedError(f"Something is wrong with frames that are not the base frame, for now don't use reference_frame")
        collision_obj_id = f"adarl_robot_helper_collision_object_{self._collision_object_count}"
        # the quaternion is build with eigen::map<>, so the xyzw order depends on the internal representation,
        # and the docs specify it's xyzw (https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html#a3eba7a582f77a8f30525614821d7056f)
        pose = pinocchio.XYZQUATToSE3(pose_xyz_xyzw.copy())
        geom = pinocchio.GeometryObject(name = collision_obj_id,
                                        parent_joint = 0 if reference_frame is None else self._frame_name_to_idx[reference_frame],
                                        collision_geometry = collision_geometry,
                                        placement = pose)

        self._collision_geom_model.addGeometryObject(geom)
        self._collision_geom_model_data = pinocchio.GeometryData(self._collision_geom_model)
        self._need_to_place_geoms = True

        self._collision_object_count += 1
        if colliding_geoms == "all":
            colliding_geoms = [g.name for g in self._collision_geom_model.geometryObjects]
        new_coll_pairs = [(geom.name, g2) for g2 in colliding_geoms]
        self.add_collision_pairs(new_coll_pairs)
        return collision_obj_id
    
    def remove_collision_object(self, collision_obj_id : str):
        self._collision_geom_model.removeGeometryObject(collision_obj_id)

    def set_collision_pairs_from_joints(self, joint_pairs : list[tuple[str,str]]):
        geom_pairs = set()
        for pair in joint_pairs:
            geoms1 = self._joint_to_geoms[pair[0]]
            geoms2 = self._joint_to_geoms[pair[1]]
            geom_pairs.update([(g1,g2) for g1 in geoms1 for g2 in geoms2])
        return self.set_collision_pairs(list(geom_pairs))

    def get_all_collisions(self):
        # self._update_forward_kinematics()
        # if self._need_to_place_geoms:
        #     pinocchio.updateGeometryPlacements(model=self._model,
        #                                        data=self._model_data,
        #                                        geom_model=self._collision_geom_model,
        #                                        geom_data=self._collision_geom_model_data)
        #     self._need_to_place_geoms = False
        # pinocchio.computeCollisions(geometry_model = self._collision_geom_model,
        #                             geometry_data = self._collision_geom_model_data)
        # this computeCollisions recoputes forward kinematics and geometry object placements
        pinocchio.computeCollisions(model = self._model,
                                    data = self._model_data,
                                    geometry_model = self._collision_geom_model,
                                    geometry_data = self._collision_geom_model_data,
                                    q = self._joint_position,
                                    stop_at_first_collision = False)
        ret = []
        for k in range(len(self._collision_geom_model.collisionPairs)):
            cr = self._collision_geom_model_data.collisionResults[k]
            cp = self._collision_geom_model.collisionPairs[k]
            if cr.isCollision():
                ret.append((self._collision_geom_model.geometryObjects[cp.first].name,
                            self._collision_geom_model.geometryObjects[cp.second].name))
        return ret

    def _update_forward_kinematics(self):
        if self._need_to_recompute_forward_kin:
            pinocchio.forwardKinematics(self._model, self._model_data, self._joint_position)
            self._need_to_recompute_forward_kin = False
            self._need_to_place_geoms = True

        
    def get_frame_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self._update_forward_kinematics()
        ret = {}
        for frame in self._model.frames:
            joint_frame_pose = self._model_data.oMi[frame.parent]
            link_pose = joint_frame_pose*frame.placement
            ret[frame.name] = link_pose.translation.T, adarl.utils.utils.quaternion_xyzw_from_rotmat(link_pose.rotation)
        return ret

    def get_joint_names(self) -> list[str]:
        return self._joint_names
    
    def get_frame_names(self) -> list[str]:
        return self._frame_names
    
    def get_geom_names(self) -> list[str]:
        return [str(geom.name) for geom in self._collision_geom_model.geometryObjects]

    def set_joint_pose(self, joints : np.ndarray):
        if len(joints) != len(self._joint_position):
            raise RuntimeError(f"Received {len(joints)} joints, but robot has {len(self._joint_position)}")
        self._joint_position = joints
        self._need_to_recompute_forward_kin = True
        self._need_to_place_geoms = True


    def set_joint_pose_by_names(self, joints : dict[str,np.ndarray]):
        for name in self.get_joint_names():
            if name in joints:
                self._joint_position = joints[name]


if __name__ == "__main__":
    leg_file = adarl.utils.utils.pkgutil_get_path("jumping_leg","models/leg_simple.urdf.xacro")
    # leg_file = adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf")
    model_definition_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(leg_file).read_text(),
                                                                        model_kwargs={})
    robot = Robot(model_definition_string)
    n = '\n'
    print(f"Joints: {robot.get_joint_names()}")
    print(f"Links: {robot.get_frame_names()}")
    print(f"Geoms: {robot.get_geom_names()}")
    print(f"Poses: {n.join([str(f) for f in robot.get_frame_poses().items()])}")
    robot.set_joint_pose(np.array([0.6,1.0,2.0]))
    print(f"New poses: {n.join([str(f) for f in robot.get_frame_poses().items()])}")

    robot.set_collision_pairs("all")
    leg_joints = robot.get_tree_joint_names_under_joint("rail_joint")
    print(f"leg_joints = {leg_joints}")
    leg_geoms = list(itertools.chain.from_iterable(robot.get_geoms_under_joints(leg_joints)))
    print(f"leg_geoms = {leg_geoms}")
    self_collisions = [(g1,g2) for g1 in leg_geoms for g2 in leg_geoms]
    self_collisions.append(("rail_link_0","slider_link_0"))
    print(f"Self collision pairs = {self_collisions}")
    print(f"Original collision pairs = {robot._current_collision_geom_pairs}")
    robot.remove_collision_pairs(self_collisions)
    print(f"collision pairs without self-collisions = {robot._current_collision_geom_pairs}")
    # robot.set_collision_pairs_from_joints([("knee_joint_1","rail_joint")])
    # print(f"{n.join([str(i) for i in robot.get_frame_poses().items()])}")

    print(f"Current collisions = {robot.get_all_collisions()}")

    foot_pos = robot.get_frame_poses()["foot_center_link"]
    platform_pos = np.array([0.08,0.2,0.22, 0.0,0.0,0.0,1.0])
    print(f"foot position     = {foot_pos}")
    print(f"platform position = {platform_pos}")
    robot.add_collision_box(    pose_xyz_xyzw=platform_pos,
                                # reference_frame="universe",
                                collision_box_size_xyz=(0.2,0.4,0.1))
    print("")
    print(f"collision pairs = {robot._current_collision_geom_pairs}")
    print(f"collisions = {robot.get_all_collisions()}")
