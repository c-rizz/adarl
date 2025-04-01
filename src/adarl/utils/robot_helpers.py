#!/usr/bin/env python3

from __future__ import annotations
import pinocchio
import numpy as np
import adarl.utils.utils
from pathlib import Path
from typing import Literal, Sequence
import copy
from typing import Iterable, TypedDict
import itertools
import faulthandler
from pinocchio.visualize import GepettoVisualizer
faulthandler.enable()
from enum import Enum
from adarl.utils.utils import quat_mul_xyzw_np, th_quat_conj, quat_conj_xyzw_np
import pprint

class JointProperties(TypedDict):
    joint_type : str

class Robot():
    JOINT_TYPES = Enum("JOINT_TYPES",["PRISMATIC",
                                  "REVOLUTE",
                                  "FIXED",
                                  "FLOATING",
                                  "CONTINUOUS"])
    
    def __init__(self, model_urdf_string : str):
        self._urdf_string = model_urdf_string
        self._model = pinocchio.buildModelFromXML(model_urdf_string)
        self._collision_geom_model = pinocchio.buildGeomFromUrdfString(self._model,
                                                                       self._urdf_string,
                                                                       pinocchio.GeometryType.COLLISION)
        self._model_data = self._model.createData()
        # self._joint_position = pinocchio.randomConfiguration(self._model)
        q_size = sum([self._model.joints[jid].nq for jid in range(1,self._model.njoints)])
        self._joint_position = np.zeros(shape=(q_size,))
        self._collision_object_count = 0
        self._collision_objects = {}

        self._joint_names = [str(n) for n in self._model.names]
        self._joints_num = len(self._joint_names)
        self._joint_name_to_idx = {n:self._joint_names.index(n) for n in self._joint_names}
        self._joint_idx_to_name = {idx:name for name,idx in self._joint_name_to_idx.items()}

        self._frame_names = [frame.name for frame in self._model.frames]
        self._frame_name_to_idx = {n:self._frame_names.index(n) for n in self._frame_names}
        self._frame_idx_to_name = {idx:name for name,idx in self._frame_name_to_idx.items()}
        self._joints_to_frame_names : dict[str,list[str]] = {}
        for i in range(len(self._joint_names)):
            jname = self._model.names[i]
            self._joints_to_frame_names[jname] = []
            for link in self._model.frames:
                if link.parent == i:
                    self._joints_to_frame_names[jname].append(link.name)
        self._frame_names_to_parent_joint_names : dict[str,str] = {}
        for jn,fns in self._joints_to_frame_names.items():
            for fn in fns:
                self._frame_names_to_parent_joint_names[fn] = jn
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

    def get_enabled_collision_pairs(self):
        return copy.deepcopy(self._current_collision_geom_pairs)

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
                                    colliding_geoms : Iterable[str] | Literal["all"] = "all",
                                    collision_obj_id = None):
        return self.add_collision_object(pose_xyz_xyzw = pose_xyz_xyzw,
                                  collision_geometry=pinocchio.hppfcl.Box(*collision_box_size_xyz),
                                  reference_frame=reference_frame,
                                  colliding_geoms=colliding_geoms,
                                  collision_obj_id=collision_obj_id)
        
    def add_collision_object(self,  pose_xyz_xyzw : np.ndarray,
                                    collision_geometry : pinocchio.hppfcl.CollisionGeometry,
                                    reference_frame = None,
                                    colliding_geoms : Iterable[str] | Literal["all"] = "all",
                                    collision_obj_id : str | None = None):
        if reference_frame is not None:
            raise NotImplementedError(f"Something is wrong with frames that are not the base frame, for now don't use reference_frame")
        if collision_obj_id is None:
            collision_obj_id = f"adarl_robot_helper_collision_object_{self._collision_object_count}"
        elif collision_obj_id in self._collision_objects:
            raise RuntimeError(f"A collision object with name '{collision_obj_id}' already exists")
        # the quaternion is build with eigen::map<>, so the xyzw order depends on the internal representation,
        # and the docs specify it's xyzw (https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html#a3eba7a582f77a8f30525614821d7056f)
        pose = pinocchio.XYZQUATToSE3(pose_xyz_xyzw.copy())
        geom = pinocchio.GeometryObject(name = collision_obj_id,
                                        parent_joint = 0 if reference_frame is None else self._frame_name_to_idx[reference_frame],
                                        collision_geometry = collision_geometry,
                                        placement = pose)

        idx = self._collision_geom_model.addGeometryObject(geom)
        self._collision_geom_model_data = pinocchio.GeometryData(self._collision_geom_model)
        self._need_to_place_geoms = True

        self._collision_object_count += 1
        self._collision_objects[collision_obj_id] = idx
        if colliding_geoms == "all":
            colliding_geoms = [g.name for g in self._collision_geom_model.geometryObjects]
        new_coll_pairs = [(geom.name, g2) for g2 in colliding_geoms]
        self.add_collision_pairs(new_coll_pairs)
        return collision_obj_id
    
    def move_collision_object(self, collision_obj_id : str, pose_xyz_xyzw : np.ndarray):
        geom = self._collision_geom_model.geometryObjects[self._collision_objects[collision_obj_id]]
        geom.placement = pinocchio.XYZQUATToSE3(pose_xyz_xyzw.copy())
        self._collision_geom_model_data = pinocchio.GeometryData(self._collision_geom_model)
        self._need_to_place_geoms = True

    
    def remove_collision_object(self, collision_obj_id : str):
        geom_id = self._collision_objects[collision_obj_id]
        geom_name = self._collision_geom_model.geometryObjects[geom_id].name
        self._collision_geom_model.removeGeometryObject(geom_id)
        self._collision_objects.pop(collision_obj_id)
        geom_pairs_to_remove = set()
        for pair in self._collision_pairs:
            if pair[0] == geom_name or pair[1] == geom_name:
                geom_pairs_to_remove.add(pair)
        self.remove_collision_pairs(geom_pairs_to_remove)



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
        ret : list[tuple[str,str]] = []
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

        
    def get_frame_poses_xyzxyzw(self, reference_frame : str | None = None,
                        frames : list[str] | None = None) -> dict[str, np.ndarray]:
        self._update_forward_kinematics()
        ret = {}
        ref_pose = None
        for frame in self._model.frames:
            joint_frame_pose = self._model_data.oMi[frame.parent]
            link_pose = joint_frame_pose*frame.placement
            if frames is None or frame.name in frames:
                ret[frame.name] = link_pose.translation.T, adarl.utils.utils.quaternion_xyzw_from_rotmat(link_pose.rotation)
            if reference_frame is not None and reference_frame == frame.name:
                ref_pose = link_pose.translation.T, adarl.utils.utils.quaternion_xyzw_from_rotmat(link_pose.rotation)
        if reference_frame is not None:
            if ref_pose is None:
                raise RuntimeError(f"Reference frame {reference_frame} not found")
            ref_pos, ref_orient = ref_pose
            ret = {fname: (pos-ref_pos, quat_mul_xyzw_np(orient,quat_conj_xyzw_np(ref_orient))) for fname, (pos, orient) in ret.items()}
        return {fname:np.concatenate([p_xyz,q_xyzw]) for fname,(p_xyz,q_xyzw) in ret.items()}
    

    def get_joint_names(self) -> list[str]:
        return self._joint_names
    
    def get_joint_properties(self, joint_names : list[str] | None = None) -> dict[str,dict[str,JointProperties]]:
        r = {}
        if joint_names is None:
            joint_names = self._joint_names
        for jn in joint_names:
            jid = self._joint_name_to_idx[jn]
            j = self._model.joints[jid]
            p = {}
            if j.idx_q < 0 or j.idx_v<0:
                p["type"] = Robot.JOINT_TYPES.FIXED # Not sure about this, but the universe joint that is added automatically apepars like this
            elif j.shortname() in ["JointModelRX","JointModelRY","JointModelRZ","JointModelRevoluteUnaligned"]:
                p["type"] = Robot.JOINT_TYPES.REVOLUTE
            elif j.shortname() in ["JointModelPX","JointModelPY","JointModelPZ","JointModelPrismaticUnaligned"]:
                p["type"] = Robot.JOINT_TYPES.PRISMATIC
            elif j.shortname() in ["JointModelFreeFlyer"]:
                p["type"] = Robot.JOINT_TYPES.FLOATING
            elif j.shortname() in ["JointModelRUBX","JointModelRUBY","JointModelRUBZ","JointModelRevoluteUnboundedUnaligned"]:
                p["type"] = Robot.JOINT_TYPES.CONTINUOUS
            else:
                raise RuntimeError(f"Unknown joint type {j.shortname()}")
            p["nq"] = j.nq
            p["nv"] = j.nv
            p["parent"] = self._joint_idx_to_name[self._model.parents[jid]]
            p["pinname"] = j.shortname()
            r[jn] = p
        return r
    
    def get_parent_joint(self, frame_name : str):
        return self._frame_names_to_parent_joint_names[frame_name]
    
    def get_frame_names(self) -> list[str]:
        return self._frame_names
    
    def get_tree_frame_names_under_frame(self, frame_name : str):
        joints = self.get_tree_joint_names_under_joint(self._frame_names_to_parent_joint_names[frame_name])
        frames : list[str] = []
        for j in joints:
            frames.extend(self._joints_to_frame_names[j])
        return frames
    
    def get_tree_frame_names_under_joint(self, joint_name : str):
        joints = self.get_tree_joint_names_under_joint(joint_name)
        frames : list[str] = []
        for j in joints:
            frames.extend(self._joints_to_frame_names[j])
        return frames
    
    def get_geom_names(self) -> list[str]:
        return [str(geom.name) for geom in self._collision_geom_model.geometryObjects]

    def set_joint_pose(self, joints : np.ndarray):
        if len(joints) != len(self._joint_position):
            raise RuntimeError(f"Received {len(joints)} joints, but robot has {len(self._joint_position)}")
        self._joint_position = joints
        self._need_to_recompute_forward_kin = True
        self._need_to_place_geoms = True

    def get_joint_pose(self):
        return copy.deepcopy(self._joint_position)


    def set_joint_pose_by_names(self, joints : dict[str,np.ndarray]):
        for name in self.get_joint_names():
            if name in joints:
                q_idx = self._model.joints[self._joint_name_to_idx[name]].idx_q
                nq = self._model.joints[self._joint_name_to_idx[name]].nq
                self._joint_position[q_idx:q_idx+nq] = joints[name]
        self._need_to_recompute_forward_kin = True
        self._need_to_place_geoms = True

    def disable_tree_self_collisions(self, root_joint : str | None = None, root_frame : str | None = None):
        if root_joint is None:
            if root_frame is None:
                raise RuntimeError(f"You must specify either root_joint or root_link")
            root_joint = self._frame_names_to_parent_joint_names[root_frame]
        tree_joints = self.get_tree_joint_names_under_joint(root_joint)
        leg_geoms = list(itertools.chain.from_iterable(self.get_geoms_under_joints(tree_joints)))
        self_collision_pairs = [(g1,g2) for g1 in leg_geoms for g2 in leg_geoms]
        self.remove_collision_pairs(self_collision_pairs)
        return self_collision_pairs
    
    def get_dbg_image(self):
        from panda3d_viewer import Viewer, ViewerConfig

        config = ViewerConfig()
        config.set_window_size(320, 240)
        config.enable_antialiasing(True, multisamples=4)
        config.enable_shadow(True)
        config.show_axes(False)
        config.show_grid(False)
        config.show_floor(True)

        with Viewer(window_type='offscreen', config=config) as viewer:
            from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
            visualizer = Panda3dVisualizer(self._model, self._collision_geom_model, self._collision_geom_model)
            visualizer.initViewer(viewer=viewer)
            visualizer.loadViewerModel(group_name=self._model.name)
            # visualizer.displayCollisions(True)
            visualizer.display(self._joint_position)
            viewer.reset_camera(pos=(0, 2, 1), look_at=(0, 0, 0.5))
            image_rgb = viewer.get_screenshot(requested_format='RGB')
        return image_rgb
    
    def get_joint_limits(self, joints : Sequence[str] | None = None) -> dict[str,np.ndarray]:
        if joints is None:
            joints = self.get_joint_names()
        limits_minmax_pve = {}
        p_minmax = np.stack([self._model.lowerPositionLimit,self._model.upperPositionLimit])
        v_minmax = np.stack([-self._model.velocityLimit,self._model.velocityLimit])
        e_minmax = np.stack([-self._model.effortLimit,self._model.effortLimit])
        for jn in joints:
            joint_idx = self._joint_name_to_idx[jn]
            q_idx = self._model.idx_qs[joint_idx]
            v_idx = self._model.idx_vs[joint_idx]
            limits_minmax_pve[jn] = np.stack([p_minmax[:,q_idx], v_minmax[:,v_idx], e_minmax[:,v_idx]]).transpose()
        return limits_minmax_pve

    def detect_always_present_collisions(self, moving_joints : Sequence[str], fixed_joints_pose : dict[str,np.ndarray], samples : int = 10000,
                                         threshold = 1.0):
        original_joint_pose = self.get_joint_pose()
        original_collision_pairs = self.get_enabled_collision_pairs()
        self.set_collision_pairs("all")
        # always_present_collisions = set()
        collision_counters = {}
        self.set_joint_pose_by_names(fixed_joints_pose)

        for i in range(samples):
            rand_pos = np.random.random(size=(len(moving_joints),))*2-1
            limits = self.get_joint_limits(moving_joints)
            limits_minmax = np.stack([limits[jn][:,0] for jn in moving_joints], axis = 1)
            pose = rand_pos*(limits_minmax[1]-limits_minmax[0])+limits_minmax[0]
            
            jpose_dict = {jn:pose[i] for i,jn in enumerate(moving_joints)}
            self.set_joint_pose_by_names(jpose_dict)
            collisions = self.get_all_collisions()
            # print(f"moving_joints = {moving_joints}")
            # print(f"jpose_dict = {jpose_dict}")
            # pprint.pprint(self.get_frame_poses_xyzxyzw())
            # pprint.pprint(collisions)
            # print(f"limits = {limits}")
            # print(f"jp = {self._joint_position}")
            # img = self.get_dbg_image()
            # import cv2
            # import time
            # print(img)
            # cv2.imwrite(f"robot_img{time.time()}.png", img)
            # time.sleep(1)
            # input("Press ENTER")
            # if i == 0:
            #     always_present_collisions = set(collisions)
            collision_counters.update({ln:collision_counters.get(ln,0)+1 for ln in collisions})
            # always_present_collisions = always_present_collisions.intersection(set(collisions))
        self.set_joint_pose(original_joint_pose)
        self.set_collision_pairs(original_collision_pairs)
        print(f"collision_counters = {collision_counters}")
        return {ln for ln, count in collision_counters.items() if count>=samples*threshold}



if __name__ == "__main__":
    import sys
    if len(sys.argv)==1:
        leg_file = adarl.utils.utils.pkgutil_get_path("adarl_envs","models/leg_rig_simple.urdf.xacro")
    else:
        leg_file = sys.argv[1]
    # leg_file = adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf")
    model_definition_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(leg_file).read_text(),
                                                                        model_kwargs={})
    robot = Robot(model_definition_string)
    n = '\n'
    print(f"Joints: {robot.get_joint_names()}")
    print(f"Joints: {robot.get_joint_properties()}")
    print(f"Links: {robot.get_frame_names()}")
    print(f"Geoms: {robot.get_geom_names()}")
    print(f"Poses: {n.join([str(f) for f in robot.get_frame_poses_xyzxyzw().items()])}")
    robot.set_joint_pose(np.array([0.6,1.0,2.0]))
    print(f"New poses: {n.join([str(f) for f in robot.get_frame_poses_xyzxyzw().items()])}")
    print(f"Joint limits = "+"\n - ".join([""]+[str(lims) for lims in robot.get_joint_limits().items()]))
    robot.set_collision_pairs("all")
    # leg_joints = robot.get_tree_joint_names_under_joint("rail_joint")
    # print(f"leg_joints = {leg_joints}")
    # leg_geoms = list(itertools.chain.from_iterable(robot.get_geoms_under_joints(leg_joints)))
    # print(f"leg_geoms = {leg_geoms}")
    # self_collisions = [(g1,g2) for g1 in leg_geoms for g2 in leg_geoms]
    # self_collisions.append(("rail_link_0","slider_link_0"))
    # print(f"Self collision pairs = {self_collisions}")
    # print(f"Original collision pairs = {robot._current_collision_geom_pairs}")
    # robot.remove_collision_pairs(self_collisions)
    robot.disable_tree_self_collisions("rail_joint")
    robot.remove_collision_pairs([("rail_link_0","slider_link_0")])
    print(f"collision pairs without self-collisions = {robot._current_collision_geom_pairs}")
    # robot.set_collision_pairs_from_joints([("knee_joint_1","rail_joint")])
    # print(f"{n.join([str(i) for i in robot.get_frame_poses_xyzxyzw().items()])}")

    print(f"Current collisions = {robot.get_all_collisions()}")

    foot_pos = robot.get_frame_poses_xyzxyzw()["foot_center_link"]
    platform_pos = np.array([0.08,0.2,0.22, 0.0,0.0,0.0,1.0])
    print(f"foot position     = {foot_pos}")
    print(f"platform position = {platform_pos}")
    co_id = robot.add_collision_box(pose_xyz_xyzw=platform_pos,
                                    # reference_frame="universe",
                                    collision_box_size_xyz=(0.2,0.4,0.1))
    print("")
    print(f"collision pairs = {robot._current_collision_geom_pairs}")
    print(f"collisions = {robot.get_all_collisions()}")

    robot.move_collision_object(collision_obj_id=co_id,
                                pose_xyz_xyzw=np.array([0.2,0.3,0.6, 0.0,0.0,0.0,1.0]))
    
    ground_co_id = robot.add_collision_box( pose_xyz_xyzw=np.array([0.,0.,0.,0.,0.,0.,1.]),
                                            collision_box_size_xyz=(1,1,0.05),
                                            collision_obj_id="ground_collision")
    print("")
    print(f"collision pairs = {robot._current_collision_geom_pairs}")
    print(f"collisions = {robot.get_all_collisions()}")

    img = robot.get_dbg_image()
    import cv2
    import time
    cv2.imwrite(f"robot_img{time.time()}.png", img)
