from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import torch as th


class BaseVecCartesianPositionAdapter(ABC):
    """Vectorized counterpart of :class:`BaseCartesianPositionAdapter`.

    Adapters implementing this interface accept end-effector pose commands and take care of
    converting them to whatever the underlying simulator/robot needs (inverse kinematics, joint
    position commands, ...). As stated in :meth:`BaseAdapter.step`, a step of a position-controlled
    adapter may have a variable duration, i.e. it may last as long as the commanded movement takes.
    """

    @abstractmethod
    def setCartesianPoseCommand(self,   link_names : Sequence[tuple[str,str]],
                                        link_poses_xyz_xyzw : th.Tensor,
                                        vec_mask : th.Tensor | None = None) -> None:
        """Request a set of links to be moved to a specific cartesian pose.

        Parameters
        ----------
        link_names : Sequence[tuple[str,str]]
            Names of the links to be moved, in the form (model_name, link_name)
        link_poses_xyz_xyzw : th.Tensor
            Tensor of shape (vec_size, len(link_names), 7) containing, for each link in each
            environment, the requested pose as position_xyz + orientation_xyzw
        vec_mask : th.Tensor | None
            Tensor of shape (vec_size,) indicating which environments to apply the command to.
            If None, the command is applied to all of them.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cartesian_pose_command(self) -> tuple[Sequence[tuple[str,str]], th.Tensor]:
        """Get the currently commanded cartesian pose.

        Returns
        -------
        tuple[Sequence[tuple[str,str]], th.Tensor]
            The commanded link names and a (vec_size, len(link_names), 7) tensor with the commanded
            poses as position_xyz + orientation_xyzw.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cartesian_position_error(self) -> th.Tensor:
        """Get the distance between the current and the commanded position of each controlled link.

        Returns
        -------
        th.Tensor
            Tensor of shape (vec_size, number_of_controlled_links) with the position error in meters.
        """
        raise NotImplementedError()

    @abstractmethod
    def moveToEePoseSync(self,  link_names : Sequence[tuple[str,str]],
                                link_poses_xyz_xyzw : th.Tensor,
                                vec_mask : th.Tensor | None = None,
                                max_error : float = 0.001,
                                sim_timeout_sec : float = 30.0) -> None:
        """Move the specified links to the specified poses, running the simulation until they get there.

        This is meant to be used during reset/initialization procedures, not during an episode.

        Parameters
        ----------
        link_names : Sequence[tuple[str,str]]
            Names of the links to be moved
        link_poses_xyz_xyzw : th.Tensor
            Tensor of shape (vec_size, len(link_names), 7) with the requested poses
        vec_mask : th.Tensor | None
            Which environments to move. If None, all of them.
        max_error : float
            Position error under which the movement is considered completed, in meters
        sim_timeout_sec : float
            Give up after this amount of simulated time
        """
        raise NotImplementedError()
