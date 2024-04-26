from flowforge.input.System import System
from flowforge.writers.ModelWriter import ModelWriter

class OpenFoamWriter(ModelWriter):
    """ A class for OpenFOAM model writers

    Attributes
    ----------
    """

    def __init__(self):
        pass

    def write(self, system: System, model_name: str) -> None:
        self._write_0(system)
        self._write_constant(system)
        self._write_system(system)

    def _write_0(self, system: System) -> None:
        """ Private method for writing the initial conditions '0' files
        """
        pass

    def _write_constant(self, system: System) -> None:
        """ Private method for writing the constant files
        """
        pass

    def _write_system(self, system: System) -> None:
        """ Private method for writing the system files
        """
        self._write_blockMeshDict(system)
        self._write_controlDict(system)
        self._write_fvSchemes(system)
        self._write_fvSolution(system)


    def _write_blockMeshDict(self, system: System) -> None:
        """ Private method for writing the 'blockMeshDict' file
        """
        pass

    def _write_controlDict(self, system: System) -> None:
        """ Private method for writing the 'controlDict' file
        """
        pass

    def _write_fvSchemes(self, system: System) -> None:
        """ Private method for writing the 'fvSchemes' file
        """
        pass

    def _write_fvSolution(self, system: System) -> None:
        """ Private method for writing the 'fvSolution' file
        """
        pass

