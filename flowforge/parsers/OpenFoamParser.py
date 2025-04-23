from flowforge.parsers import OutputParser


class OpenFoamParser(OutputParser):
    """A class for OpenFOAM output parsing

    Attributes
    ----------
    """

    def __init__(self):
        """
        Initialize the OpenFoamParser.

        This method will set up any necessary configuration for parsing OpenFOAM output files,
        including paths to solution directories and configuration for specific versions of OpenFOAM.

        Notes
        -----
        Currently not implemented.
        """
        raise NotImplementedError("To Be Implemented")

    def parse(self) -> None:
        """
        Parse OpenFOAM output files into FlowForge data structures.

        This method will read OpenFOAM output files and extract relevant data like velocity,
        pressure, temperature fields, and convert them into a format that can be used by FlowForge.

        Returns
        -------
        None
            Currently not implemented.

        Notes
        -----
        Currently not implemented. When implemented, this method will return
        a list of data structures containing the parsed output from OpenFOAM simulations.
        """
        raise NotImplementedError("To Be Implemented")
