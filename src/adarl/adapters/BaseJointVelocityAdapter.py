from abc import ABC, abstractmethod
from typing import Tuple, List

class BaseJointVelocityAdapter(ABC):
    @abstractmethod
    def setJointsVelocityCommand(self, jointVelocities : List[Tuple[Tuple[str,str],float]]) -> None:
        raise NotImplementedError()
