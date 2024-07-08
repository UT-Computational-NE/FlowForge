from typing import Dict, List, Tuple, Generator
from copy import deepcopy
import numpy as np
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.Components import Component, Nozzle
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.BoundaryConditions import MassMomentumBC, EnthalpyBC

def make_continuous(components: List[Component], order: List[dict]):
    """Private method makes serial components continuous with respect to area change

    This method takes in a list of components and their order and inserts infitesimal nozzles between them
    that make the area change transitions continuous

    Parameters
    ----------
    cont_components : list
        list of components
    order : list
        order of those components

    Returns
    -------
    list, list
        The new component list and order with the inserted nozzles
    """
    num_connects=0
    discont_found = True
    while discont_found:
        discont_found=False
        #initialize the previous area as the first area
        prev_area = components[order[0]["component"]].inletArea
        for i, entry in enumerate(order):
            if abs(prev_area-components[entry["component"]].inletArea) \
              > 1.0E-12*(prev_area+components[entry["component"]].inletArea)/2:
                tempnozzle=Nozzle(L=1.0E-64,R_inlet=np.sqrt(prev_area/np.pi),R_outlet=
                                  np.sqrt(components[entry["component"]].inletArea/np.pi),
                                  theta=components[entry["component"]].theta*180/np.pi,\
                                    alpha = components[entry["component"]].alpha,
                                  Klossinlet=0,Klossoutlet=0,Klossavg=0,roughness=0)
                components[f'temp_nozzle_for_make_continuous_creation_in_system_{entry["component"]}_{num_connects}'] \
                  = deepcopy(tempnozzle)
                order = order[0:i] + [{'component' : 'temp_nozzle_for_make_continuous_creation_in_system_' \
                                       + f'{entry["component"]}_{num_connects}'}] + order[i:len(order)]
                num_connects += 1
                discont_found = True
                break
            prev_area = components[entry["component"]].outletArea
    return components, order

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

    def _setupSimpleLoop(self, components: Dict[str, Component], loop: List[dict], fluid: str = "FLiBe") -> None:
        """ Private method for setting up a loop of components

        Here, a 'loop of components' means that the last component's outlet
        connects to first component input

        Parameters
        ----------
        components : Dict[str, Component]
            Collection of initialized components with which to construct the loop with
        loop : List[dict]
            List specifying the construction of loop via component names and forces.  Ordering is
            from the 'first' component of the loop to the 'last'.
        """
        components, loop = make_continuous(components,loop)
        self._fluidname = fluid.lower()
        # Loop over each component in the loop, add those components to the list, define the connections between components
        for i, entry in enumerate(loop):
            self._components.append(deepcopy(components[entry["component"]]))
            bftemp = []
            wftemp = []
            if "BodyForces" in entry:
                bftemp = entry["BodyForces"]
            if "WallFunctions" in entry:
                wftemp = entry["WallFunctions"]
            #add a body force for the component if present
            self._bodyforces.append(deepcopy(bftemp))
            #add a wall function for the component if present
            self._wallfunctions.append(deepcopy(wftemp))
            # connect the current component to the previous (exclude the first component because there isn't a previous)
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
            # If the last entry in the loop, connect the last component to the first
            if i == len(loop) - 1:
                self._connectivity.append((self._components[i], self._components[0]))

    def _setupSegment(self, components: List[Component], order: List[dict],
                      boundary_conditions: Dict =  {}, fluid: str = "FLiBe") -> None:
        """ Private method for setting up a segment

        Here, a segment refers to a model with defined inlet and outlet boundary conditions

        Parameters
        ----------
        components : Dict[str, Component]
            Collection of initialized components with which to construct the segment with
        order : List[str]
            List specifying the construction of segment via component names and forces.  Ordering is
            from the 'first' component of the segment to the 'last'.
        boundary_conditions : Dict
            Dictionary of boundary conditions for the segment
        """
        components, order = make_continuous(components,order)
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
            self._bodyforces.append(deepcopy(bftemp))
            #add a wall function for the component if present
            self._wallfunctions.append(deepcopy(wftemp))
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
        #get the boundary conditions
        self._MMBC = MassMomentumBC()
        if "mass_momentum" in boundary_conditions:
            self._MMBC = MassMomentumBC(**boundary_conditions["mass_momentum"])
        self._EBC = EnthalpyBC()
        if "enthalpy" in boundary_conditions:
            self._EBC = EnthalpyBC(**boundary_conditions["enthalpy"])


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
