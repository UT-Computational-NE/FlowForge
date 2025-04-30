from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any


class OutputParser(ABC):
    """An abstract base class for parsing outputs from various solvers into FlowForge data structures.

    This class defines the common interface that all solver output parsers must implement.
    Each concrete implementation of this class (for example, SythParser, OpenFoamParser)
    will handle parsing output from a specific solver format.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly. Concrete
    implementations must override the parse() method.
    """

    @abstractmethod
    def parse(self) -> List[Any]:
        """Primary method for parsing model outputs on to a FlowForge System

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
