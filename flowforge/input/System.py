from typing import Dict, List, Tuple, Generator
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.Components import Component
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.BoundaryConditions import MassMomentumBC, EnthalpyBC
import sys
from copy import deepcopy

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
    fluid : string
        The fluid to be used
    """

    def __init__(self, components: Dict[str, Component], sysdict: Dict, unitdict: Dict[str, str]) -> None:
        self._components = []
        self._connectivity = []
        self._bodyforces = []
        self._wallfunctions = []
        self._MMBC = None
        self._EBC = None
        self._fluid = None

        if "simple_loop" in sysdict:
            self._setupSimpleLoop(components, **sysdict["simple_loop"])
        elif "segment" in sysdict:
            self._setupSegment(components, **sysdict["segment"])
        # TODO add additional types of systems that can be set up

        for comp in self._components:
            comp._convertUnits(UnitConverter(unitdict))
        #convert BC units
        if self._MMBC is not None:
            self._MMBC._convertUnits(UnitConverter(unitdict))
        if self._EBC is not None:
            self._EBC._convertUnits(UnitConverter(unitdict))

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

    def _setupSegment(self, components: List[Component], order: List[str], boundary_conditions: Dict =  {}, fluid: str = "FLiBe") -> None:
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
        self._fluidname = fluid.lower()
        # Loop over each entry in segment, add the components, and connect the compnents to each other
        for i, entry in enumerate(order):
            self._components.append(deepcopy(components[entry["component"]]))
            bftemp = []
            wftemp = []
            if "BodyForces" in entry:
                bftemp = entry["BodyForces"]
            if "WallFunctions" in entry:
                wftemp = entry["WallFunctions"]
            #add a body force for the component if present
            self._bodyforces.append(bftemp)
            #add a wall function for the component if present
            self._wallfunctions.append(wftemp)
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
        #get the boundary conditions
        if "mass_momentum" in boundary_conditions:
            self._MMBC = MassMomentumBC(**boundary_conditions["mass_momentum"])
        else:
            self._MMBC = MassMomentumBC()
        if "enthalpy" in boundary_conditions:
            self._EBC = EnthalpyBC(**boundary_conditions["enthalpy"])
        else:
            self._EBC = EnthalpyBC()


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
    def bodyforces(self) -> List[str]:
        return self._bodyforces

    @property
    def wallfunctions(self) -> List[str]:
        return self._wallfunctions

    @property
    def EBC(self) -> EnthalpyBC:
        return self._EBC

    @property
    def MMBC(self) -> MassMomentumBC:
        return self._MMBC

    @property
    def fluidname(self) -> str:
        return self._fluidname
