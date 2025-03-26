from flowforge.parsers import OutputParser


class SythParser(OutputParser):
    """A class for Syth output parsing

    Attributes
    ----------
    """

    def __init__(self):
        """
        Initialize the SythParser.

        This method will set up any necessary configuration for parsing Syth output files.

        Notes
        -----
        Currently not implemented.
        """
        raise NotImplementedError("To Be Implemented")

    def parse(self) -> None:
        """
        Parse Syth output files into FlowForge data structures.

        This method will read Syth output files, extract relevant data, and
        convert it into a format that can be used by FlowForge.

        Returns
        -------
        None
            Currently not implemented.

        Notes
        -----
        Currently not implemented. When implemented, this method will return
        a list of data structures containing the parsed output from Syth.
        """
        raise NotImplementedError("To Be Implemented")
