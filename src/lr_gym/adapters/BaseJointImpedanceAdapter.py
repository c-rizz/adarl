from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Mapping
from lr_gym.adapters.BaseAdapter import BaseAdapter

class BaseJointImpedanceAdapter(BaseAdapter):
    @abstractmethod
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]]) -> None:
        """ Sets a joint impedance command. The command only is appied at the next call of step().
        After step() is called the command gets cleared out. Position, velocity and torque commands may override this.
        For each joint the command is a tuple of the format (position,velocity,effort,stiffness,damping)

        Parameters
        ----------
        joint_impedances_pvesd : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]
            Dictionary with:
             - key=(<model_name>,<joint_name>)
             - value=(<position_reference>,<velocity_reference>,<effort_reference>,<position_gain>,<velocity_gain>)
        """
        raise NotImplementedError()

    @abstractmethod    
    def apply_joint_impedances(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]]) -> None:
        """ Applies a joint impedance command immediately.
            Meant to be used outside of the normal control loop, just for resetting/initializing.
        """
        raise NotImplementedError()

