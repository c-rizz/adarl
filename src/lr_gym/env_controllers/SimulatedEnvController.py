from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from lr_gym.utils.utils import JointState, LinkState, Pose



class SimulatedEnvController(ABC):
    # @abstractmethod
    # def spawnModel(self):
    #     """Spawn a model in the environment, arguments depend on the type of SimulatedEnvController
    #     """
    #     raise NotImplementedError()

    # @abstractmethod
    # def deleteModel(self, model : str):
    #     """Delete a model from the environment"""
    #     raise NotImplementedError()

    @abstractmethod
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        raise NotImplementedError()
    
    @abstractmethod
    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        raise NotImplementedError()

    @abstractmethod
    def setupLight(self):
        raise NotImplementedError()

    @abstractmethod
    def spawn_model(self,   model_definition_string : str,
                            model_format : str,
                            model_file,
                            model_name : str,
                            pose : Pose,
                            model_kwargs : Dict[Any,Any]) -> str:
        """Spawn a model in the simulation

        Parameters
        ----------
        model_definition_string : str
            Model definition specified in as a string. e.g. an SDF definition
        model_format : str
            Format of the model definition. E.g. 'sdf' or 'urdf'
        model_file : _type_
            File to load the model definition from
        model_name : str
            Name to give to the spawned model
        pose : Pose
            Pose to spawn the model at
        model_kwargs : Dict[Any,Any]
            Arguments to use in interpreting the model definition

        Returns
        -------
        str
            The model name
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_model(self, model_name : str):
        """Remove a model from the simulation

        Parameters
        ----------
        model_name : str
            Name of the model to be removed
        """
        raise NotImplementedError()