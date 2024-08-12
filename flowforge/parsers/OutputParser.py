from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any

class OutputParser(ABC):
    """ An abstract class for parsing outputs from various solvers into a FlowForge Output object
    """

    @abstractmethod
    def parse(self) -> List[Any]:
        """ Primary method for parsing model outputs on to a FlowForge System

        The output being parsed must correspond to a model that is consistent
        with the system on which the output is being projected on.  In practically
        all cases, this means the model must have been generated using the
        FlowForge System.

        Returns
        -------
        List[Any]
            The parsed output
        """
        return
