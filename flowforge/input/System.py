from typing import Dict, List, Tuple, Generator, Any
from copy import deepcopy
import numpy as np
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.Components import Component, Nozzle, Core
from flowforge.input.SolidComponents import SolidComponent
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.BodyForces import BodyForces
from flowforge.input.WallFunctions import WallFunctions
from flowforge.input.BoundaryConditions import BoundaryConditions
from flowforge.parsers.OutputParser import OutputParser


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
    num_connects = 0
    discont_found = True
    while discont_found:
        discont_found = False
        # initialize the previous area as the first area
        prev_area = components[order[0]["component"]].inletArea
        for i, entry in enumerate(order):
            if abs(prev_area - components[entry["component"]].inletArea) > 1.0e-12 * min(
                prev_area, components[entry["component"]].inletArea
            ):
                tempnozzle = Nozzle(
                    L=1.0e-64,
                    R_inlet=np.sqrt(prev_area / np.pi),
                    R_outlet=np.sqrt(components[entry["component"]].inletArea / np.pi),
                    theta=components[entry["component"]].theta * 180 / np.pi,
                    alpha=components[entry["component"]].alpha,
                    Klossinlet=0,
                    Klossoutlet=0,
                    Klossavg=0,
                    roughness=0,
                )
                components[f'temp_nozzle_for_make_continuous_creation_in_system_{entry["component"]}_{num_connects}'] = (
                    deepcopy(tempnozzle)
                )
                order = (
                    order[0:i]
                    + [
                        {
                            "component": "temp_nozzle_for_make_continuous_creation_in_system_"
                            + f'{entry["component"]}_{num_connects}'
                        }
                    ]
                    + order[i : len(order)]
                )
                num_connects += 1
                discont_found = True
                break
            prev_area = components[entry["component"]].outletArea
    return components, order


class System:
    """A class for representing a complete thermal-fluid system of components.

    The System class manages an entire thermal-fluid system composed of multiple components
    connected together. It handles connectivity between components, boundary conditions,
    fluid properties.

    A system can be configured as either:
    - A single segment (possibly of multiple components) with distinct inlet and outlet boundaries
    - A closed loop (possibly of multiple components) in circulation with no external boundaries

    The System class also handles unit conversions, boundary conditions, and provides
    interfaces for visualization and output parsing from various solvers.

    Parameters
    ----------
    components : Dict[str, Component]
        Collection of initialized components with which to construct the system.
        Each component is identified by a unique name.
    sysdict : Dict
        Dictionary of system settings describing how to initialize the system,
        including type (segment or loop), connectivity information, and optional
        parser configurations.
    unitdict : Dict[str, str]
        Dictionary of units used for this system. This allows specification of
        custom units for length, pressure, temperature, etc.

    Attributes
    ----------
    components : List[Component]
        The collection of components which comprise the system
    connectivity : List[Tuple[Component, Component]]
        The connectivity map of the sections of the system. The map is expressed as a list
        of component pairs, with the first element in the pair being the 'from' component and
        the second element the 'to' component. The list is ordered from the 'starting segment'
        to the 'ending segment'
    solidComponents : List[SolidComponent]]
        The collection of solid components which comprise the solid system
    solidConnectivity : List[Tuple[SolidComponent, SolidComponent]]
        Connectivity map for solid components, expressing pairs of components in a (previous-
        component, current-component) form. This list is used in meshing to build interfaces
        between components and the components that came before it
    inBoundComp : Component
        The system inlet boundary component
    outBoundComp : Component
        The system outlet boundary component
    nCells : int
        The number of cells in the system
    fluid : str
        The fluid to be used in the system
    gas : str, optional
        The gas to be used in the system (for two-phase simulations)
    output_parsers : Dict[str, OutputParser]
        The output parsers with which to parse output from various models and map to the System
    """

    def __init__(
        self,
        components: Dict[str, Component],
        sysdict: Dict,
        unitdict: Dict[str, str],
        solid_components: Dict[str, SolidComponent] = {},
        solid_controllers: Dict[str, Dict[str, dict]] = {}
    ) -> None:
        self._components = []
        self._solid_components = []
        self._output_parsers = {}
        self._connectivity = []
        self._solid_connectivity = []
        self._bodyforces = []
        self._wallfunctions = []
        self._component_htc = {} # For component-level HTC
        self._MMBC = None
        self._EBC = None
        self._VBC = None
        self._fluid = None
        self._gas = None
        self._htc = None # For system-level HTC

        # Initializes objects to be empty
        self._boundaryConditionContainer = BoundaryConditions(**{})
        self._bodyForceContainer = BodyForces(**{})
        self._wallFunctionContainer = WallFunctions(**{})
        self._isLoop = False  # Boolean defining if system is a loop or segment

        system_types = ["simple_loop", "segment", "solid_system"]
        assert (
            sum(k in sysdict for k in system_types) == 1
        ), f"Expected exactly one of {system_types}, found {[k for k in system_types if k in sysdict]}"

        if "simple_loop" in sysdict:
            self._setupSimpleLoop(components, **sysdict["simple_loop"])
        elif "segment" in sysdict:
            self._setupSegment(components, **sysdict["segment"])
        elif "solid_system" in sysdict:
            self._setupSolidSystem(solid_components, solid_controllers, **sysdict["solid_system"])
        # TODO add additional types of systems that can be set up

        if "parsers" in sysdict:
            self._setupParsers(sysdict["parsers"])

        for comp in self._components:
            comp._convertUnits(UnitConverter(unitdict))
        # convert BC units
        if self._boundaryConditionContainer is not None:
            self._boundaryConditionContainer._convertUnits(UnitConverter(unitdict))

    @property
    def core(self) -> List[Core]:
        """
        Returns the core component of the system.
        """
        cores = []
        for comp in self._components:
            if isinstance(comp, Core):
                cores.append(comp)

        if not cores:
            return Exception
        return cores

    def _setupSimpleLoop(
        self,
        components: Dict[str, Component],
        loop: List[dict],
        boundary_conditions: Dict = {},
        fluid: str = "FLiBe",
        gas=None,
        HTC= "DittusBoelter"
    ) -> None:
        """Private method for setting up a loop of components

        Here, a 'loop of components' means that the last component's outlet
        connects to first component input

        Parameters
        ----------
        components : Dict[str, Component]
            Collection of initialized components with which to construct the loop with
        loop : List[dict]
            List specifying the construction of loop via component names and forces.  Ordering is
            from the 'first' component of the loop to the 'last'.
        boundary_conditions : Dict
            Dictionary of boundary conditions for the segment
        fluid : str
            The working fluid used in the segment (e.g., "FLiBe"). Defaults to "FLiBe".
        gas  optional :
            An optional parameter to specify gas in the system (e.g. "Helium")

        """
        self._isLoop = True

        components, loop = make_continuous(components, loop)
        self._fluidname = fluid.lower()
        self._gasname = gas if gas is None else gas.lower()
        self._htc = HTC
        # Loop over each component in the loop, add those components to the list, define the connections between components
        for i, entry in enumerate(loop):
            current_component = deepcopy(components[entry["component"]])
            self._components.append(current_component)
            self._component_htc[current_component] = entry.get("HTC", HTC)
            bftemp = []
            wftemp = []
            if "BodyForces" in entry:
                bftemp = entry["BodyForces"]
            if "WallFunctions" in entry:
                wftemp = entry["WallFunctions"]
            # add a body force for the component if present
            self._bodyforces.append(deepcopy(bftemp))
            # add a wall function for the component if present
            self._wallfunctions.append(deepcopy(wftemp))
            # connect the current component to the previous (exclude the first component because there isn't a previous)
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
            # If the last entry in the loop, connect the last component to the first
            if i == len(loop) - 1:
                self._connectivity.append((self._components[i], self._components[0]))

        # get the boundary conditions
        self._boundaryConditionContainer = BoundaryConditions(**boundary_conditions)

    def _setupSegment(
        self,
        components: List[Component],
        order: List[dict],
        boundary_conditions: Dict = {},
        fluid: str = "FLiBe",
        gas=None,
        HTC = "DittusBoelter"
    ) -> None:
        """Private method for setting up a segment

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
        fluid : str
            The working fluid used in the segment (e.g., "FLiBe"). Defaults to "FLiBe".
        gas  optional :
            An optional parameter to specify gas in the system (e.g. "Helium")
        """
        self._isLoop = False

        components, order = make_continuous(components, order)
        self._fluidname = fluid.lower()
        self._gasname = gas if gas is None else gas.lower()
        self._htc = HTC
        # Loop over each entry in segment, add the components, and connect the compnents to each other
        for i, entry in enumerate(order):
            current_component = deepcopy(components[entry["component"]])
            self._components.append(current_component)
            self._component_htc[current_component] = entry.get("HTC", HTC)
            bftemp = []
            wftemp = []
            if "BodyForces" in entry:
                bftemp = entry["BodyForces"]
            if "WallFunctions" in entry:
                wftemp = entry["WallFunctions"]
            # add a body force for the component if present
            self._bodyforces.append(deepcopy(bftemp))
            # add a wall function for the component if present
            self._wallfunctions.append(deepcopy(wftemp))
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))

        # get the boundary conditions
        self._boundaryConditionContainer = BoundaryConditions(**boundary_conditions)

    def _setupSolidSystem(
        self,
        solid_components: Dict[str, SolidComponent],
        solid_controllers: Dict[str, Dict[str, dict]],
        order: List[str],
        boundary_conditions: Dict[str, Any]
    ):
        """
        Private method for setting up a solid system

        Given a set of components and their respective ordering, this method builds a list
        of the components in the correct order, as well as defines the connectivity of each
        component.

        Parameters
        ----------
        solid_components : Dict[str, SolidComponent]
            Set of initialized components, where the key is the components unique name
        solid_controllers : Dict
            Dictionary of controllers for the system
        order : List[Dict]
            Ordering of the components, where each element is a dict containing information
            on the specific component
        boundary_conditions : Dict
            Dictionary of boundary conditions for the system
        """

        for i, entry in enumerate(order):
            self._solid_components.append(deepcopy(solid_components[entry["component"]]))
            self._bodyforces.append(deepcopy(entry.get("BodyForces", [])))
            self._wallfunctions.append(deepcopy(entry.get("WallFunctions", [])))

            current_component = self._solid_components[i]
            previous_component = None if i == 0 else self._solid_components[i - 1]

            self._solid_connectivity.append((previous_component, current_component))

        body_forces = solid_controllers.get("bodyForce", {})
        wall_functions = solid_controllers.get("wallFunction", {})

        self._boundaryConditionContainer = BoundaryConditions(**boundary_conditions)
        self._bodyForceContainer = BodyForces(**body_forces)
        self._wallFunctionContainer = WallFunctions(**wall_functions)

    def _setupParsers(self, parser_dict: Dict) -> None:
        """Private method for setting up output parsers

        Parameters
        ----------
        parser_dict : Dict
            The input dictionary specifying the parsers to be setup
        """

        raise NotImplementedError("To Be Implemented")

    def getCellGenerator(self) -> Generator[Component, None, None]:
        """Generator for marching over the nodes (i.e. cells) of a system

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
        """Method for generating a VTK mesh of the system

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
        """Method for generating and writing VTK meshes to file

        Parameters
        ----------
        filename : str
            The name of the file to write the VTK mesh to
        """
        sysFile = VTKFile(filename, self.getVTKMesh())
        sysFile.writeFile()

    def getComponentHTC(self, component: Component) -> str:
        """Method for getting the heat transfer coefficient for a component

        Parameters
        ----------
        component : Component
            The component to get the HTC from

        Returns
        -------
        str
            The heat transfer coefficient for the component
        """
        return self._component_htc.get(component, self._htc)

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
    def solidComponents(self):
        return self._solid_components

    @property
    def solidConnectivity(self):
        return self._solid_connectivity

    @property
    def output_parsers(self) -> Dict[str, OutputParser]:
        return self._output_parsers

    @property
    def bodyforces(self) -> List[str]:
        return self._bodyforces

    @property
    def wallfunctions(self) -> List[str]:
        return self._wallfunctions

    @property
    def BoundaryConditions(self) -> BoundaryConditions:
        return self._boundaryConditionContainer

    @property
    def BodyForceContainer(self) -> BodyForces:
        return self._bodyForceContainer

    @property
    def WallFunctionContainer(self) -> WallFunctions:
        return self._wallFunctionContainer

    @property
    def isLoop(self) -> bool:
        return self._isLoop

    @property
    def fluidname(self) -> str:
        return self._fluidname

    @property
    def gasname(self) -> str:
        return self._gasname

    @property
    def system_htc(self) -> str:
        return self._htc

    @property
    def component_htc(self) -> Dict[Component, str]:
        return self._component_htc
