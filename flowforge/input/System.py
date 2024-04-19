from typing import Dict, List, Tuple
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.Components import Component
from flowforge.input.UnitConverter import UnitConverter


class System:
    """
    Controls the whole system by initializing all components and writing vtk solution file.
    """

    def __init__(self, components: List[Component], sysdict: Dict[str, str], unitdict: Dict[str, str]) -> None:
        """
        Initialize system of components

        Args:
            components: List of initialized components
            sysdict: Dictionary of system settings describing how to initialize
            unitdict: Dictionary of units used to convert input
        """
        self._components = []
        self._connectivity = []
        self._inBoundComp = None
        self._outBoundComp = None

        if "simple_loop" in sysdict:
            self._setupSimpleLoop(components, **sysdict["simple_loop"])
        elif "section" in sysdict:
            self._setupSection(components, **sysdict["section"])
        # TODO add additional types of systems that can be set up

        for comp in self._components:
            comp._convertUnits(UnitConverter(unitdict))

    def _setupSimpleLoop(self, components: List[Component], loop: Dict[str, str]) -> None:
        """
        Sets up a loop of components (last components outlet connects to first component input)

        Args:
            components: List of initialized components
            loop: Dictionary describing the loop
        """
        # Loop over each component in the loop, add those components to the list, define the connections between components
        for i, comp in enumerate(loop):
            self._components.append(components[comp])
            # connect the current component to the previous (exclude the first component because there isn't a previous)
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
            # If the last entry in the loop, connect the last component to the first
            if i == len(loop) - 1:
                self._connectivity.append((self._components[i], self._components[0]))

    def _setupSection(self, components: List[Component], section: Dict[str, str]) -> None:
        """
        Sets up a section (this is a model with defined inlet and outlet boundary conditions)

        Args:
            components: List of initialized components
            section: Dictionary describing the model
        """
        # Define the component the inlet boundary condition is applied to
        self._inBoundComp = components[section[0]]
        # Loop over each entry in section, add the components, and connect the compnents to each other
        for i, entry in enumerate(section):
            self._components.append(components[entry])
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
        # Define the component the outlet boundary condition is applied to
        self._outBoundComp = components[section[-1]]

    def getCellGenerator(self):
        """
        Generator interface that returns all of the nodes in a model sequentially
        Args:
        """
        for comp in self._components:
            yield from comp.getNodeGenerator()

    def getVTKMesh(self) -> VTKMesh:
        """
        The getVTKMesh function is a private function that starts with the
        first inlet at the origin and adds the mesh of every component to the
        system mesh. The system mesh is then stored as _sysMesh. The translation
        of each component is taken care of within the getVTKMesh function in the
        component classes.

        Args: None
        """
        inlet = (0, 0, 0)
        mesh = VTKMesh()
        for c in self._components:
            mesh += c.getVTKMesh(inlet)
            inlet = c.getOutlet(inlet)
        return mesh

    def writeVTKFile(self, filename: str, time: float = None) -> None:  # pylint:disable=unused-argument
        """
        The write system file will be used to export the whole system mesh into
        a VTK file to view in another program. This function calls the _getSystemMesh
        function to loop through the components and generate the meshes for each one.
        This function will then utilize the VTKFile class to create the exported file.

        Args:
            filename : str, name of the file to be exported
            time     : TODO - Implement
        """
        sysFile = VTKFile(filename, self.getVTKMesh())
        sysFile.writeFile()

    @property
    def nCell(self) -> int:
        """
        Returns the number of cells in the system.
        """
        ncell = 0
        for c in self._components:
            ncell += c.nCell
        return ncell

    @property
    def components(self) -> List[Component]:
        """
        Returns the system components
        """
        return self._components

    @property
    def connectivity(self) -> List[Tuple[Component, Component]]:
        """
        Returns the system component connectivities
        """
        return self._connectivity

    @property
    def inBoundComp(self) -> Component:
        """
        Returns the system inlet boundary component
        """
        return self._inBoundComp

    @property
    def outBoundComp(self) -> Component:
        """
        Returns the system outlet boundary component
        """
        return self._outBoundComp
