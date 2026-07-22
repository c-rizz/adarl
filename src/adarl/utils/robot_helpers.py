#!/usr/bin/env python3

from __future__ import annotations
import pprint
import pinocchio
import numpy as np
from pathlib import Path
from typing import Literal, Sequence
import copy
from typing import Iterable, TypedDict
import itertools
import faulthandler
import torch as th
import scipy
import time
# from pinocchio.visualize import GepettoVisualizer
faulthandler.enable()
from enum import Enum
from adarl.utils.utils import expand_default_dict, quat_mul_xyzw_np, quat_conj_xyzw_np, quaternion_xyzw_from_rotmat
import tempfile
import time
from dataclasses import dataclass

def buildModelFromMJCFString(model_string : str):
    with tempfile.NamedTemporaryFile(suffix=".mjcf", delete=True) as f:
        f.write(model_string.encode())
        f.flush()
        model = pinocchio.buildModelFromMJCF(f.name)
    return model

def buildGeomFromMJCFString(model, model_string : str, geom_type : pinocchio.GeometryType):
    with tempfile.NamedTemporaryFile(suffix=".mjcf", delete=True) as f:
        f.write(model_string.encode())
        f.flush()
        geom_model = pinocchio.buildGeomFromMJCF(model, f.name, geom_type)
    return geom_model


@dataclass
class ModelDescription:
    """One model to include in a Robot's combined pinocchio model.

    - description_string : the URDF/MJCF text
    - format             : "urdf" or "mjcf"
    - placement_xyz_xyzw : pose of this model's root in the frame it attaches to (default: identity)
    - attach_frame       : name of a frame (in the models merged so far) to attach to; None -> universe

    A fixed/jointless root adds no DOF and is rigidly placed by ``placement_xyz_xyzw``. A part with a
    free/floating joint contributes real DOFs, appended after the existing ones, so its pose lives in q.
    """
    description_string : str
    format : Literal["urdf", "sdf", "mjcf"] = "urdf"
    placement_xyz_xyzw : np.ndarray | None = None
    attach_frame : str | None = None


class JointProperties(TypedDict):
    joint_type : str

class Robot():
    JOINT_TYPES = Enum("JOINT_TYPES",[  "PRISMATIC",
                                        "REVOLUTE",
                                        "FIXED",
                                        "FLOATING",
                                        "CONTINUOUS"])
    
    def __init__(self, robot_description_string : str,
                       robot_description_format : Literal["urdf", "sdf", "mjcf"] = "urdf",
                       additional_models : list[ModelDescription] | None = None):
        # The robot is the base model; any additional_models (world, fixtures, free-floating objects)
        # are appended after it, so the robot keeps its q/v indices at the front. Static (fixed/
        # jointless) parts add no DOF; parts with free/floating joints add DOFs after the robot's.
        self._parts = [ModelDescription(robot_description_string, robot_description_format)]
        if additional_models is not None:
            self._parts.extend(additional_models)
        # kept for backward compatibility with code reading these attributes
        self._robot_string = robot_description_string
        self._robot_format = robot_description_format

        self._model, self._collision_geom_model = self._build_and_merge(self._parts)
        self._model_data = self._model.createData()
        # neutral() rather than zeros: gives valid unit quaternions for free-flyer/floating (and
        # continuous) joints, and reduces to zeros for plain revolute/prismatic robots.
        self._joint_position = pinocchio.neutral(self._model)
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

    @staticmethod
    def _build_model_and_geom(description_string : str, description_format : str):
        if description_format == "urdf":
            model = pinocchio.buildModelFromXML(description_string)
            geom = pinocchio.buildGeomFromUrdfString(model, description_string, pinocchio.GeometryType.COLLISION)
        elif description_format == "mjcf":
            model = buildModelFromMJCFString(description_string)
            geom = buildGeomFromMJCFString(model, description_string, pinocchio.GeometryType.COLLISION)
        else:
            # SDF would go through pinocchio.buildModelsFromSdf (note: plural, can return several models)
            raise NotImplementedError(f"Only urdf and mjcf formats are currently supported, but got {description_format}")
        return model, geom

    @staticmethod
    def _build_and_merge(parts : list[ModelDescription]):
        """Build each part and append them into a single (model, collision_geom_model)."""
        model, geom = Robot._build_model_and_geom(parts[0].description_string, parts[0].format)
        for part in parts[1:]:
            part_model, part_geom = Robot._build_model_and_geom(part.description_string, part.format)
            frame_idx = 0 if part.attach_frame is None else model.getFrameId(part.attach_frame)
            if part.placement_xyz_xyzw is None:
                aMb = pinocchio.SE3.Identity()
            else:
                aMb = pinocchio.XYZQUATToSE3(np.asarray(part.placement_xyz_xyzw, dtype=float).copy())
            model, geom = pinocchio.appendModel(model, part_model, geom, part_geom, frame_idx, aMb)
        return model, geom

    def __getstate__(self):
        d = dict(self.__dict__)
        # The pinocchio Model/GeometryModel and their Data are rebuilt from self._parts in
        # __setstate__ (GeometryModel is not picklable: stack-of-tasks/pinocchio#2089). Collision
        # objects/pairs added at runtime via add_collision_object are not persisted (as before).
        for k in ("_model", "_model_data", "_collision_geom_model", "_collision_geom_model_data"):
            d.pop(k, None)
        return d

    def __setstate__(self, d):
        model, geom = Robot._build_and_merge(d["_parts"])
        d["_model"] = model
        d["_model_data"] = model.createData()
        d["_collision_geom_model"] = geom
        d["_collision_geom_model_data"] = pinocchio.GeometryData(geom)
        self.__dict__.update(d)

    def set_collision_pairs(self, geom_pairs : Iterable[tuple[str,str]] | Literal["all"] = []):
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

    def get_adjacent_collision_pairs(self) -> list[tuple[str, str]]:
        """Return geom pairs whose parent joints are directly connected by a joint."""
        pairs = []
        for jid in range(1, self._model.njoints):  # skip universe joint at 0
            parent_jid = self._model.parents[jid]
            child_geoms = self._joint_to_geoms[self._joint_idx_to_name[jid]]
            parent_geoms = self._joint_to_geoms[self._joint_idx_to_name[parent_jid]]
            pairs += [(g1, g2) for g1 in child_geoms for g2 in parent_geoms]
        return pairs


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
            if cr.isCollision():
                cp = self._collision_geom_model.collisionPairs[k]
                ret.append((self._collision_geom_model.geometryObjects[cp.first].name,
                            self._collision_geom_model.geometryObjects[cp.second].name))
        return ret
    
    def has_collisions(self):

        # this computeCollisions recoputes forward kinematics and geometry object placements
        pinocchio.computeCollisions(model = self._model,
                                    data = self._model_data,
                                    geometry_model = self._collision_geom_model,
                                    geometry_data = self._collision_geom_model_data,
                                    q = self._joint_position,
                                    stop_at_first_collision = True)
        for k in range(len(self._collision_geom_model.collisionPairs)):
            cr = self._collision_geom_model_data.collisionResults[k]
            if cr.isCollision():
                cp = self._collision_geom_model.collisionPairs[k]
                collision_pair = (self._collision_geom_model.geometryObjects[cp.first].name,
                            self._collision_geom_model.geometryObjects[cp.second].name)
                return True, collision_pair
        return False, None

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
            is_reference_frame = reference_frame is not None and reference_frame == frame.name
            is_requested_frame = frames is None or frame.name in frames
            if is_requested_frame or is_reference_frame:
                joint_frame_pose : pinocchio.pinocchio_pywrap_default.SE3 = self._model_data.oMi[frame.parentJoint if hasattr(frame,"parentJoint") else frame.parent]
                link_pose = joint_frame_pose*frame.placement
                if is_requested_frame:
                    ret[frame.name] = link_pose.translation.T, quaternion_xyzw_from_rotmat(link_pose.rotation)
                if is_reference_frame:
                    ref_pose = link_pose.translation.T, quaternion_xyzw_from_rotmat(link_pose.rotation)
        if len(ret) != len(frames if frames is not None else self._frame_names):
            raise RuntimeError(f"Requested frames {frames} but only found poses for {list(ret.keys())}")
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
        for jn in joints:
            if jn not in self.get_joint_names():
                raise RuntimeError(f"Tried to move set position of joint {jn}, but it does not exist, existing joints = {self.get_joint_names()}")
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
        v_minmax = np.stack([-self._model.velocityLimit,    self._model.velocityLimit])
        e_minmax = np.stack([-self._model.effortLimit,      self._model.effortLimit])
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
        collision_rates = {ln:count/samples for ln,count in collision_counters.items()}
        # print(f"collision_rates (on {samples}) = {pprint.pformat(sorted(collision_rates.items(), key=lambda x:x[1], reverse=True))}")
        return {ln for ln, rate in collision_rates.items() if rate>=threshold}



if __name__ == "__main__":
    import sys
    from adarl.utils.utils import pkgutil_get_path, compile_xacro_string
    if len(sys.argv)==1:
        leg_file = pkgutil_get_path("adarl_envs","models/leg_rig_simple.urdf.xacro")
    else:
        leg_file = sys.argv[1]
    # leg_file = adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf")
    model_definition_string = compile_xacro_string(  model_definition_string=Path(leg_file).read_text(),
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


def find_pose_np(  root_joint : str,
                homing_body_pose_xyzxyzw : np.ndarray,
                controlled_joints : Sequence[tuple[str,str]],
                initial_pose_randomization_ranges : dict[tuple[str,str], float],
                initial_height_randomization_range : float,
                limits_minmax : np.ndarray,
                homing_pos : np.ndarray,
                noncontrolled_jointpos : dict[tuple[str,str], np.ndarray],
                robot_model : Robot | str,
                is_floating_base : bool,
                rng_seed,
                excluded_collision_pairs):
    t = time.monotonic()
    if isinstance(robot_model, str):
        robot_model = Robot(robot_model)
        robot_model.set_collision_pairs("all")
        robot_model.remove_collision_pairs(excluded_collision_pairs)
    t1 = time.monotonic()
    found = False
    coll_counter = {}
    samples = 1000
    jp_dict = noncontrolled_jointpos
    rng = np.random.default_rng(seed=rng_seed)
    truncnorm = scipy.stats.truncnorm(-1, 1, loc=0, scale=1/3)
    seen_collision_pairs = {}
    ranges = np.array([initial_pose_randomization_ranges[jn] for jn in controlled_joints], dtype=np.float32)
    for i in range(samples):
        norm_jpos = truncnorm.rvs(size=(len(controlled_joints),), random_state=rng).astype(np.float32)*ranges
        # norm_jpos = (rng.random(size=(len(controlled_joints),), dtype=np.float32)*2-1)*initial_pose_randomization_range
        # initial_joint_pose = unnormalize(((npos)),limits_minmax[0],limits_minmax[1])                
        initial_joint_pose = ((norm_jpos>=0)*((limits_minmax[1]-homing_pos)*norm_jpos + homing_pos) +
                                (norm_jpos< 0)*((homing_pos-limits_minmax[0])*norm_jpos + homing_pos))
        jp_dict.update({jn:initial_joint_pose[i] for i,jn in enumerate(controlled_joints)})
        robot_model.set_joint_pose_by_names({jn[1]:jp for jn,jp in jp_dict.items()})
        if is_floating_base:
            norm_height = (rng.random(size=(1,), dtype=np.float32)*2-1)*initial_height_randomization_range
            initial_body_pose_xyzxyzw = homing_body_pose_xyzxyzw.copy()
            initial_body_pose_xyzxyzw[2] += norm_height[0]
            robot_model.set_joint_pose_by_names({root_joint:initial_body_pose_xyzxyzw})
        has_collision, collision_pair = robot_model.has_collisions() # Returns True if there is any collision, and the first collision pair found (or None if no collision)
        if not has_collision:
            found = True
            initial_jpose = initial_joint_pose
            break
        seen_collision_pairs[collision_pair] = seen_collision_pairs.get(collision_pair, 0) + 1
        # collisions = robot_model.get_all_collisions()
        # # all_link_poses = self._robot_model.get_frame_poses_xyzxyzw() #frames=self._robot_model.get_tree_frame_names_under_joint(self._configuration.robot_root_joint))
        # # pprint.pprint(all_link_poses)
        # # all_links_z = np.stack([pose[2] for pose in all_link_poses.values()])
        # coll_counter.update({ln:coll_counter.get(ln,0)+1 for ln in collisions})                    
        # if len(collisions) == 0: # and np.all(all_links_z>0):
        #     # ggLog.info(f"joint_pose = {self._robot_model.get_joint_pose()}")
        #     # ggLog.info(f"selected all_link_poses = {all_link_poses}")
        #     found = True
        #     initial_jpose = initial_joint_pose
        #     break
    t2 = time.monotonic()
    if not found:
        initial_jpose = homing_pos
        seen_collision_pairs = {k:v/samples for k,v in seen_collision_pairs.items()}
        print(  f"Failed to find initial joint configuration."
                f" last collision seen = {collision_pair}\n"
                f" filtered collisions = {excluded_collision_pairs}\n"
                f" coll_ratio={seen_collision_pairs}")
    # ggLog.info(f"Model creation took {t1-t}s, pose search {t2-t1}s")
    return initial_jpose


def find_poses(root_joint : str,
                homing_body_pose_xyzxyzw : np.ndarray,
                controlled_joints : Sequence[tuple[str,str]],
                initial_pose_randomization_range : float | dict[tuple[str,str] | str, float],
                initial_height_randomization_range : float,
                limits_minmax : np.ndarray,
                homing_pos : np.ndarray,
                noncontrolled_jointpos : dict[tuple[str,str], np.ndarray],
                robot_model : Robot,
                is_floating_base : bool,
                seed : int,
                excluded_collision_pairs : set[tuple[str,str]],
                num_envs : int,
                ):
    seeds = np.random.default_rng(seed).integers(0, 1_000_000_000_000, size=(num_envs,))
    # seeds = [int(th.randint(low=0, high=1_000_000_000_000, size=(1,), generator=rng, device=homing_pos.device).item()) for _ in range(num_envs)]
    # with adarl.utils.mp_helper.get_context().Pool() as p:
    #     r = p.starmap(find_pose_np, [[ root_joint,
    #                             homing_body_pose_xyzxyzw,
    #                             controlled_joints,
    #                             initial_pose_randomization_range,
    #                             limits_np,
    #                             homing_np,
    #                             noncontrolled_jointpos_np,
    #                             robot_model._urdf_string,
    #                             is_floating_base,
    #                             seeds[i],
    #                             excluded_collision_pairs]
    #                          for i in range(num_envs)])
    #     return th.as_tensor(np.stack(r))

    original_collision_pairs = robot_model.get_enabled_collision_pairs()
    robot_model.set_collision_pairs("all")
    robot_model.remove_collision_pairs(excluded_collision_pairs)
    joint_ranges = expand_default_dict(initial_pose_randomization_range, controlled_joints)
    r = np.zeros(shape=(num_envs, len(controlled_joints)), dtype=np.float32)
    for v in range(num_envs): # TODO: this may be sloooooow, can I parallelize it?
        r[v] = find_pose_np(    root_joint,
                                homing_body_pose_xyzxyzw,
                                controlled_joints,
                                joint_ranges,
                                initial_height_randomization_range,
                                limits_minmax,
                                homing_pos,
                                noncontrolled_jointpos,
                                robot_model,
                                is_floating_base,
                                seeds[v],
                                excluded_collision_pairs)
    robot_model.set_collision_pairs(original_collision_pairs)

    return th.as_tensor(r)
