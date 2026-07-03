from __future__ import annotations
from abc import abstractmethod
from typing import Any, Sequence
from adarl.utils.utils import Pose, build_pose
from adarl.adapters.BaseVecAdapter import BaseVecAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef

import torch as th


class BaseVecSimulationAdapter(BaseVecAdapter):
    """ Interface for implementing vectorized simulation-backed adapters. """

    def setJointsAndLinksStateDirect(self,
                                     joint_names : Sequence[tuple[str,str]] | None = None,
                                     joint_states_pve : th.Tensor | None = None,
                                     link_names : Sequence[tuple[str,str]] | None = None,
                                     link_states_pose_vel : th.Tensor | None = None,
                                     vec_mask : th.Tensor | None = None):
        """Set joint and link state together, using a shared vec mask.

        The default implementation preserves existing behavior by delegating to
        `setLinksStateDirect()` first and `setJointsStateDirect()` second.
        Adapters can override this to fuse preprocessing or device-side updates.
        """
        if joint_names is None:
            if joint_states_pve is not None:
                raise ValueError("joint_states_pve was provided without joint_names")
        elif joint_states_pve is None:
            raise ValueError("joint_names was provided without joint_states_pve")

        if link_names is None:
            if link_states_pose_vel is not None:
                raise ValueError("link_states_pose_vel was provided without link_names")
        elif link_states_pose_vel is None:
            raise ValueError("link_names was provided without link_states_pose_vel")

        if link_names is not None:
            self.setLinksStateDirect(link_names=link_names,
                                     link_states_pose_vel=link_states_pose_vel,
                                     vec_mask=vec_mask)
        if joint_names is not None:
            self.setJointsStateDirect(joint_names=joint_names,
                                      joint_states_pve=joint_states_pve,
                                      vec_mask=vec_mask)

    @abstractmethod
    def setJointsStateDirect(self, joint_names : Sequence[tuple[str,str]], joint_states_pve : th.Tensor, vec_mask : th.Tensor | None = None):        
        """Set the state for a set of joints

        Parameters
        ----------
        joint_names : Sequence[tuple[str,str]]
            The names of the joints to set the state for
        joint_states_pve : th.Tensor
            A tensor of shape (vec_size, len(joint_names), 3) containins, position,velocity and effort for each joint
        vec_mask : th.Tensor
            Tensor of size (vec_size,), indicating which simulators to use or not use. If None, apply to all
        """
        raise NotImplementedError()
    
    @abstractmethod
    def setLinksStateDirect(self, link_names : Sequence[tuple[str,str]], link_states_pose_vel : th.Tensor, vec_mask : th.Tensor | None = None):
        """Set the state for a set of links

        Parameters
        ----------
        link_names : Sequence[tuple[str,str]]
            The names of the links to set the state for
        link_states_pose_vel : th.Tensor
            A tensor of shape (vec_size, len(link_names), 13), containing, for each link, position_xyz, orientation_xyzw, linear_velocity_xyz, angular_velocity_xyz
        vec_mask : th.Tensor
            Tensor of size (vec_size,), indicating which simulators to use or not use. If None, apply to all
        """
        raise NotImplementedError()

    @abstractmethod
    def setupLight(self):
        raise NotImplementedError()

    @abstractmethod
    def build_scenario(self, models : Sequence[ModelSpawnDef] = [], **kwargs):
        raise NotImplementedError()
        
    @abstractmethod
    def spawn_models(self, models : Sequence[ModelSpawnDef]) -> list[str]:
        """Spawn models in all of the simulations. Each model in the provided list will be spawned in all simulations.


        Parameters
        ----------
        models : Sequence[ModelSpawnDef]
            The models to spawn

        Returns
        -------
        list[str]
            The names of the spawned models
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_model(self, model_name : str):
        """Remove a model from all of the simulations
        Parameters
        ----------
        model_name : str
            Name of the model to be removed
        """
        raise NotImplementedError()
    
    def set_body_collisions(self, link_group_collisions : list[tuple[tuple[str,str], list[tuple[str,str]]]]):
        raise NotImplementedError()
    
    def set_link_impulses(self, link_ids : Sequence[Any],
                                force_torque_xyzxyz : th.Tensor,
                                durations : th.Tensor, delays : th.Tensor,
                                vec_mask : th.Tensor) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def sim_step_duration(self) -> th.Tensor:
        raise NotImplementedError()

    # @abstractmethod
    def alter_model(self,   link_masses : tuple[Any, th.Tensor] | None = None,
                            link_frictions : tuple[Any, th.Tensor] | None = None,
                            joint_armature_ratios : tuple[Any, th.Tensor] | None = None,
                            joint_damping_ratios : tuple[Any, th.Tensor] | None = None,
                            joint_frictionloss_ratios : tuple[Any, th.Tensor] | None = None,
                            com_position_diffs : tuple[Any, th.Tensor] | None = None,
                            com_quatxyzw_diffs : tuple[Any, th.Tensor] | None = None,
                            vec_mask : th.Tensor | None = None,
                            reset_first : bool = True):
        """_summary_

        Parameters
        ----------
        link_masses : tuple[Any, th.Tensor]
            tuple containing  alist of link ids (from get_link_id) and corresponding
            body masses, body masses should be in a tensor of size (vec_size, len(link_ids))
            body mass will be set to old_mass*(1+ratio)
        link_frictions : tuple[Any, th.Tensor]
            tuple containing a list of link ids (from get_link_id) and corresponding
            body friction ratios, where the new friction will be computed as old_friction*(1+ratio).
        joint_armature_ratios : tuple[Any, th.Tensor]
            tuple containing a list of joint ids (from get_joint_id) and corresponding
            joint armature ratios, where the new armature will be computed as old_armature*(1+ratio).
        joint_damping_ratios : tuple[Any, th.Tensor]
            tuple containing a list of joint ids (from get_joint_id) and corresponding
            joint damping ratios, where the new damping will be computed as old_damping*(1+ratio).
        joint_frictionloss_ratios : tuple[Any, th.Tensor]
            tuple containing a list of joint ids (from get_joint_id) and corresponding
            joint frictionloss ratios, where the new frictionloss will be computed as old_frictionloss*(1+ratio).
        com_position_diffs : tuple[Any, th.Tensor]
            tuple containing a list of link ids (from get_link_id) and corresponding
            COM position differences, where the new COM position will be computed as old_COM_position + diff
        com_quatxyzw_diffs : tuple[Any, th.Tensor]
            tuple containing a list of link ids (from get_link_id) and corresponding
            COM orientation differences in quatxyzw format, where the new COM orientation will be computed as old_COM_orientation + diff 
        vec_mask : th.Tensor
            Mask of shape (vec_size,) indicating which environments to update, if None all environments will be updated
        reset_first : bool
            Whether to reset the previous alterations before applying the new ones. If False, new alterations will
            be applied on top of the current model parameters, which might lead to compounding effects if
            the same parameters are altered multiple times.
        """
        raise NotImplementedError()
    
    def set_monitored_collision_pairs(self, collision_pairs: Sequence[tuple[tuple[str,str], tuple[str,str]]]):
        """Set collision pairs to monitor. Must be called before build_scenario.

        Pairs are buffered here and resolved to body ids / sensor adrs inside build_scenario,
        once the model is compiled and the lname2lid map exists. The warp backend additionally
        gets one mjSENS_CONTACT sensor per pair injected into the spec.

        Parameters
        ----------
        collision_pairs : Sequence[tuple[tuple[str,str], tuple[str,str]]]
            List of link name pairs to monitor for collisions.
            Each pair is ((model_a, link_a), (model_b, link_b)).
        """
        raise NotImplementedError()
    

    def check_colliding_links(self, requested_pairs: Sequence[tuple[tuple[str,str], tuple[str,str]]] | th.Tensor | None = None) -> th.Tensor:
        """Check if link pairs are colliding.
        
        Parameters
        ----------
        requested_pairs : Sequence[tuple[tuple[str,str], tuple[str,str]]] | th.Tensor | None
            If None, returns mask for all monitored pairs.
            If th.Tensor, indices into monitored pairs array.
            If Sequence, link name pairs to look up (must be monitored).
        
        Returns
        -------
        th.Tensor
            Boolean tensor of shape (vec_size, num_pairs) indicating collision status.
        """
        raise NotImplementedError()