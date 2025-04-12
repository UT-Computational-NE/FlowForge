from abc import ABC, abstractmethod
from flowforge.input.System import System

class ModelWriter(ABC):
    """ An abstract class for model writers
    """

    @abstractmethod
    def write(self, system: System, model_name: str) -> None:
        """ Primary method for writing modeling inputs for FlowForge Systems

        Parameters
        ----------
        system : System
            The system whose modeling input is to be written
        model_name : str
            The name of the resulting input (used for file / folder creation)
        """
        pass
