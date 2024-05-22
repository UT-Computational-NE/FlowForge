from typing import Dict, List, Tuple, Generator
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.Components import Component
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.BoundaryConditions import FBC, MassTempBC, PressureTempBC


class System:
    """ A class for representing a whole system of components

    Controls the whole system by initializing all components and writing vtk solution file

    Parameters
    ----------
    components : Dict[str, Component]
        Collection of initialized components with which to construct the system
    sysdict : Dict
        Dictionary of system settings describing how to initialize the system
    unitdict : Dict[str, str]
        Dictionary of units used for this system


    Attributes
    ----------
    components : List[Component]
        The collection of components which comprise the system
    connectivity : List[Tuple[Component, Component]]
        The connectivity map of the sections of the system.  The map is expressed as a list
        of component pairs, with the first element in the pair being the 'from' component and
        the second element the 'to' component.  The list is ordered from the 'starting segement'
        to the 'ending segment'
    inBoundComp : Component
        The system inlet boundary component
    outBoundComp : Component
        The system outlet boundary component
    nCells : int
        The number of cells in the system
    """

    def __init__(self, components: Dict[str, Component], sysdict: Dict, unitdict: Dict[str, str]) -> None:
        self._components = []
        self._connectivity = []
        self._inletBC = None
        self._outletBC = None

        if "simple_loop" in sysdict:
            self._setupSimpleLoop(components, **sysdict["simple_loop"])
        elif "segment" in sysdict:
            self._setupSegment(components, **sysdict["segment"])
        # TODO add additional types of systems that can be set up

        for comp in self._components:
            comp._convertUnits(UnitConverter(unitdict))
        #convert BC units
        if self._inletBC is not None:
            self._inletBC._convertUnits(UnitConverter(unitdict))
        if self._outletBC is not None:
            self._outletBC._convertUnits(UnitConverter(unitdict))

    def _setupSimpleLoop(self, components: Dict[str, Component], loop: List[str]) -> None:
        """ Private method for setting up a loop of components

        Here, a 'loop of components' means that the last component's outlet
        connects to first component input

        Parameters
        ----------
        components : Dict[str, Component]
            Collection of initialized components with which to construct the loop with
        loop : List[str]
            List specifying the construction of loop via component names.  Ordering is
            from the 'first' component of the loop to the 'last'.
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

    def _setupSegment(self, components: List[Component], order: List[str], boundary_conditions: Dict =  {}) -> None:
        """ Private method for setting up a segment

        Here, a segment refers to a model with defined inlet and outlet boundary conditions

        Parameters
        ----------
        components : Dict[str, Component]
            Collection of initialized components with which to construct the segment with
        order : List[str]
            List specifying the construction of segment via component names.  Ordering is
            from the 'first' component of the segment to the 'last'.
        boundary_conditions : Dict
            Dictionary of boundary conditions for the segment
        """
        # Loop over each entry in segment, add the components, and connect the compnents to each other
        for i, entry in enumerate(order):
            self._components.append(components[entry])
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
        # Define the inlet boundary condition
        if "inlet" in boundary_conditions:
            bctemporary = boundary_conditions["inlet"]
            #setup based on passed BCs
            if "mass_temp" in bctemporary:
                #check to make sure the first component is equal to the inlet BC component
                if order[0] != bctemporary["mass_temp"]["component"]:
                    raise TypeError("Segment inlet BC component does not equal first component!")
                #actually setup the inlet bc
                self._inletBC = MassTempBC(components[order[0]],**bctemporary["mass_temp"])
            elif "pressure_temp" in bctemporary:
                #check to make sure the first component is equal to the inlet BC component
                if order[0] != bctemporary["pressure_temp"]["component"]:
                    raise TypeError("Segment inlet BC component does not equal first component!")
                #actually setup the inlet bc
                self._inletBC = PressureTempBC(components[order[0]],**bctemporary["pressure_temp"])
            else:
                raise TypeError(f"Unknown inlet BC type: {list(bctemporary.keys())[0]}")
        else:
            #default inlet bc
            self._inletBC = MassTempBC(components[order[0]])
        # Define the outlet boundary condition
        if "outlet" in boundary_conditions:
            bctemporary = boundary_conditions["outlet"]
            #setup based on passed BCs
            if "mass_temp" in bctemporary:
                #check to make sure the first component is equal to the outlet BC component
                if order[-1] != bctemporary["mass_temp"]["component"]:
                    raise TypeError("Segment outlet BC component does not equal first component!")
                #actually setup the outlet bc
                self._outletBC = MassTempBC(components[order[-1]],**bctemporary["mass_temp"])
            elif "pressure_temp" in bctemporary:
                #check to make sure the first component is equal to the outlet BC component
                if order[-1] != bctemporary["pressure_temp"]["component"]:
                    raise TypeError("Segment outlet BC component does not equal first component!")
                #actually setup the outlet bc
                self._outletBC = PressureTempBC(components[order[-1]],**bctemporary["pressure_temp"])
            else:
                raise TypeError(f"Unknown outlet BC type: {list(bctemporary.keys())[0]}")
        else:
            #default outlet bc
            self._outletBC = PressureTempBC(components[order[-1]])

    def getCellGenerator(self) -> Generator[Component, None, None]:
        """ Generator for marching over the nodes (i.e. cells) of a system

        This method essentially allows one to march over the nodes of a system
        and be able to reference / use the component said node belongs to

        Yields
        ------
        Component
            The component associated with the node the generator is currently on

        """
        for comp in self._components:
            yield from comp.getNodeGenerator()

    def getVTKMesh(self) -> VTKMesh:
        """ Method for generating a VTK mesh of the system

        Returns
        -------
        VTKMesh
            The generated VTK mesh
        """
        inlet = (0, 0, 0)
        mesh = VTKMesh()
        for c in self._components:
            mesh += c.getVTKMesh(inlet)
            inlet = c.getOutlet(inlet)
        return mesh

    def writeVTKFile(self, filename: str) -> None:
        """ Method for generating and writing VTK meshes to file

        Parameters
        ----------
        filename : str
            The name of the file to write the VTK mesh to
        """
        sysFile = VTKFile(filename, self.getVTKMesh())
        sysFile.writeFile()

    @property
    def nCell(self) -> int:
        ncell = 0
        for c in self._components:
            ncell += c.nCell
        return ncell

    @property
    def components(self) -> List[Component]:
        return self._components

    @property
    def connectivity(self) -> List[Tuple[Component, Component]]:
        return self._connectivity

    @property
    def inletBC(self) -> FBC:
        return self._inletBC

    @property
    def outletBC(self) -> FBC:
        return self._outletBC
