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
from flowforge.input.ComponentCoupler import couple as FluidSolidComponentCoupler

# GLOBAL VARIABLES
valid_solid_system_types = tuple(["solid_system"])
valid_fluid_system_types = tuple(["segment", "simple_loop"])

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
    solid_components : Dict[str, SolidComponent], optional
        Collection of initialized solid components with which to construct the solid system.
        Each component is identified by a unique name.
    solid_controllers : Dict[str, Dict[str, dict]], optional
        Dictionary of solid component controllers.
    coupled_components : List[Tuple[str, str]], optional
        List of tuples representing coupled components.
    options : Dict[str, bool], optional
        Dictionary of options for the system. For example, "make_continuous" can be set to
        False to disable the insertion of tiny nozzles between components with discontinuities.

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
    bodyforces : List[str]
        The body forces to be applied in the system
    wallfunctions : List[str]
        The wall functions to be applied in the system
    solid_body_forces : List[str]
        The body forces to be applied in the solid system
    solid_wall_functions : List[str]
        The wall functions to be applied in the solid system
    BoundaryConditionsContainer : BoundaryConditions
        The boundary conditions to be applied in the system, organized in a container class
    BodyForceContainer : BodyForces
        The body forces to be applied in the system, organized in a container class
    WallFunctionContainer : WallFunctions
        The wall functions to be applied in the system, organized in a container class
    isLoop : bool
        Boolean defining if system is a loop or segment
    make_continuous : bool
        Boolean defining whether to insert tiny nozzles between components with discontinuities
    """

    def __init__(
        self,
        components: Dict[str, Component],
        sysdict: Dict,
        unitdict: Dict[str, str],
        solid_components: Dict[str, SolidComponent] = {},
        solid_controllers: Dict[str, Dict[str, dict]] = {},
        coupled_components: List[Tuple[str, str]] = [],
        options: Dict[str, bool] = {}

    ) -> None:
        # Component Dicts
        self._fluid_components = []
        self._solid_components = []

        # Parsers
        self._output_parsers = {}

        # Connectivity
        self._fluid_connectivity = []
        self._solid_connectivity = []
        self._coupled_components_connectivity = []

        # Boundary conditions
        self._fluid_boundary_conditions = {}
        self._solid_boundary_conditions = {}

        # Controller Variables
        self._fluid_body_forces = []
        self._fluid_wall_functions = []
        self._solid_body_forces = []
        self._solid_wall_functions = []

        # Material variables
        self._fluid = None
        self._gas = None

        # Material names
        self._fluidname = None
        self._gasname = None

        # Physics object definitions
        self._fluid_boundary_conditions_definitions = {}
        self._solid_boundary_conditions_definitions = {}
        self._solid_body_forces_definitions = {}
        self._solid_wall_functions_definitions = {}

        # Initializes container objects to be empty
        self._boundary_condition_container = BoundaryConditions(**{})
        self._body_force_container = BodyForces(**{})
        self._wall_function_container = WallFunctions(**{})
        self._isLoop = False  # Boolean defining if system is a loop or segment
        self._make_continuous = True # defines whether to insert tiny nozzles between components with discontinuities
        if "make_continuous" in options:
            self._make_continuous = options["make_continuous"]

        self._setup_system(
            sys_dict=sysdict,
            fluid_components=components,
            solid_components=solid_components,
            solid_controllers=solid_controllers,
            coupled_components=coupled_components
        )

        if "parsers" in sysdict:
            self._setupParsers(sysdict["parsers"])

        self._unit_conversion(unitdict)

    @property
    def core(self) -> List[Core]:
        """
        Returns the core component of the system.
        """
        cores = []
        for comp in self._fluid_components:
            if isinstance(comp, Core):
                cores.append(comp)

        if not cores:
            return Exception
        return cores

    def _setup_system(self,
                      sys_dict: Dict,
                      fluid_components: Dict[str, Component] = None,
                      solid_components: Dict[str, SolidComponent] = None,
                      solid_controllers: Dict[str, Dict[str, dict]] = None,
                      coupled_components: List[Tuple[str, str]] = None
                      ) -> None:
        """
        """

        def is_coupled_system(sys_dict):
            """
            Checks the system dict to determine if the system is coupled
            """
            # Checks if system is coupled
            if any(sys in valid_solid_system_types for sys in sys_dict):
                if any(sys in valid_fluid_system_types for sys in sys_dict):
                    return True

            # System is non-coupled
            all_valid_systems = valid_fluid_system_types + valid_solid_system_types
            n_systems = len([sys for sys in sys_dict if sys in all_valid_systems])
            assert n_systems == 1, "Can only define (1) system"

            return False

        # Coupled system
        if is_coupled_system(sys_dict):
            self._setup_coupled_system(
                sys_dict=sys_dict,
                fluid_components=fluid_components,
                solid_components=solid_components,
                solid_controllers=solid_controllers,
                coupled_components=coupled_components
            )

            return

        ## Non-coupled system

        # Setup system
        if "simple_loop" in sys_dict:
            self._setupSimpleLoop(fluid_components, **sys_dict["simple_loop"])
        elif "segment" in sys_dict:
            self._setupSegment(fluid_components, **sys_dict["segment"])
        elif "solid_system" in sys_dict:
            self._setupSolidSystem(solid_components, solid_controllers, **sys_dict["solid_system"])

        # Set physics variables
        all_boundary_conditions = self._fluid_boundary_conditions_definitions | self._solid_boundary_conditions_definitions
        self._boundary_condition_container = BoundaryConditions(**all_boundary_conditions)
        # NOTE: Only for solid - should add for fluid later
        self._body_force_container = BodyForces(**self._solid_body_forces_definitions)
        self._wall_function_container = WallFunctions(**self._solid_wall_functions_definitions)


    def _setup_coupled_system(self,
                              sys_dict: Dict,
                              fluid_components: Dict[str, Component] = None,
                              solid_components: Dict[str, SolidComponent] = None,
                              solid_controllers: Dict[str, Dict[str, dict]] = None,
                              coupled_components: List[Tuple[str, str]] = None
                              ) -> None:
        """
        """

        n_fluid_systems = len([sys for sys in sys_dict if sys in valid_fluid_system_types])
        n_solid_systems = len([sys for sys in sys_dict if sys in valid_solid_system_types])

        assert n_fluid_systems == 1, "Cannot define multiple fluid systems"
        assert n_solid_systems == 1, "Cannot define multiple solid systems"

        # Couple components
        for fluid_name, solid_name in coupled_components.values():
            # If input backwards (not [fluid, solid]), this flips them
            if fluid_name not in fluid_components:
                fluid_name, solid_name = solid_name, fluid_name

            # Checks that components exist
            if (fluid_name not in fluid_components) or (solid_name not in solid_components):
                raise Exception(f"Could not find fluid component {fluid_name} and/or solid component {solid_name}")

            # Couples the two components
            coupled_fluid, coupled_solid = FluidSolidComponentCoupler(
                fluid_component=fluid_components[fluid_name],
                solid_component=solid_components[solid_name]
            )

            # Replaces original objects with coupled versions
            fluid_components[fluid_name] = coupled_fluid
            solid_components[solid_name] = coupled_solid


        # Setup system
        if "simple_loop" in sys_dict:
            self._setupSimpleLoop(fluid_components, **sys_dict["simple_loop"])
        elif "segment" in sys_dict:
            self._setupSegment(fluid_components, **sys_dict["segment"])

        if "solid_system" in sys_dict:
            self._setupSolidSystem(solid_components, solid_controllers, **sys_dict["solid_system"])

        # Set physics variables
        all_boundary_conditions = self._fluid_boundary_conditions_definitions | self._solid_boundary_conditions_definitions
        self._boundary_condition_container = BoundaryConditions(**all_boundary_conditions)
        # NOTE: Only for solid - should add for fluid later
        self._body_force_container = BodyForces(**self._solid_body_forces_definitions)
        self._wall_function_container = WallFunctions(**self._solid_wall_functions_definitions)

        # Creates list of coupled component object in the system
        for fluid_name, solid_name in coupled_components.values():
            # If input backwards (not [fluid, solid]), this flips them
            if fluid_name not in fluid_components:
                fluid_name, solid_name = solid_name, fluid_name

            built_fluid_components = [comp for comp in self._fluid_components if comp.name == fluid_name]
            built_solid_components = [comp for comp in self._solid_components if comp.name == solid_name]

            error_msg = "If a component is coupled, can only define (1) of them in the system"
            assert len(built_fluid_components) == 1, error_msg + f" (error in fluid system: {fluid_name})"
            assert len(built_solid_components) == 1, error_msg + f" (error in solid system: {solid_name})"

            self._coupled_components_connectivity.append((built_fluid_components[0], built_solid_components[0]))

    def _unit_conversion(self, unit_dict):
        """
        Performs unit conversion on the various objects
        """
        # Convert component units
        for comp in self._fluid_components:
            comp._convertUnits(UnitConverter(unit_dict))  # pylint: disable=protected-access
        for comp in self._solid_components:
            comp._convertUnits(UnitConverter(unit_dict))  # pylint: disable=protected-access
        # convert BC units
        if self._boundary_condition_container is not None:
            self._boundary_condition_container._convertUnits(UnitConverter(unit_dict))  # pylint: disable=protected-access

    def _setupSimpleLoop(
        self,
        components: Dict[str, Component],
        loop: List[dict],
        boundary_conditions: Dict = {},
        fluid: str = "FLiBe",
        gas=None,
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
        if self._make_continuous:
            components, loop = make_continuous(components, loop)
        self._fluidname = fluid.lower()
        self._gasname = gas if gas is None else gas.lower()
        # Loop over each component in the loop, add those components to the list, define the connections between components
        for i, entry in enumerate(loop):
            component_i = deepcopy(components[entry["component"]])
            component_i.name = entry["component"]
            self._fluid_components.append(component_i)
            bftemp = []
            wftemp = []
            if "BodyForces" in entry:
                bftemp = entry["BodyForces"]
            if "WallFunctions" in entry:
                wftemp = entry["WallFunctions"]
            # add a body force for the component if present
            self._fluid_body_forces.append(deepcopy(bftemp))
            # add a wall function for the component if present
            self._fluid_wall_functions.append(deepcopy(wftemp))
            # connect the current component to the previous (exclude the first component because there isn't a previous)
            if i > 0:
                self._fluid_connectivity.append((self._fluid_components[i - 1], self._fluid_components[i]))
            # If the last entry in the loop, connect the last component to the first
            if i == len(loop) - 1:
                self._fluid_connectivity.append((self._fluid_components[i], self._fluid_components[0]))

        # get the boundary conditions
        self._fluid_boundary_conditions_definitions = boundary_conditions

    def _setupSegment(
        self, components: List[Component], order: List[dict], boundary_conditions: Dict = {}, fluid: str = "FLiBe", gas=None
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

        if self._make_continuous:
            components, order = make_continuous(components, order)
        self._fluidname = fluid.lower()
        self._gasname = gas if gas is None else gas.lower()
        # Loop over each entry in segment, add the components, and connect the compnents to each other
        for i, entry in enumerate(order):
            component_i = deepcopy(components[entry["component"]])
            component_i.name = entry["component"]
            self._fluid_components.append(component_i)
            bftemp = []
            wftemp = []
            if "BodyForces" in entry:
                bftemp = entry["BodyForces"]
            if "WallFunctions" in entry:
                wftemp = entry["WallFunctions"]
            # add a body force for the component if present
            self._fluid_body_forces.append(deepcopy(bftemp))
            # add a wall function for the component if present
            self._fluid_wall_functions.append(deepcopy(wftemp))
            if i > 0:
                self._fluid_connectivity.append((self._fluid_components[i - 1], self._fluid_components[i]))

        # get the boundary conditions
        self._fluid_boundary_conditions_definitions = boundary_conditions

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
            component_i = deepcopy(solid_components[entry["component"]])
            component_i.name = entry["component"]
            self._solid_components.append(component_i)
            self._solid_body_forces.append(deepcopy(entry.get("BodyForces", [])))
            self._solid_wall_functions.append(deepcopy(entry.get("WallFunctions", [])))

            current_component = self._solid_components[i]
            previous_component = None if i == 0 else self._solid_components[i - 1]

            self._solid_connectivity.append((previous_component, current_component))

        self._solid_boundary_conditions_definitions = boundary_conditions
        self._solid_body_forces_definitions = solid_controllers.get("bodyForce", {})
        self._solid_wall_functions_definitions = solid_controllers.get("wallFunction", {})

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
        for comp in self._fluid_components:
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
        for c in self._fluid_components:
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

    @property
    def nCell(self) -> int:
        ncell = 0
        for c in self._fluid_components:
            ncell += c.nCell
        return ncell

    @property
    def components(self) -> List[Component]:
        return self._fluid_components

    @property
    def connectivity(self) -> List[Tuple[Component, Component]]:
        return self._fluid_connectivity

    @property
    def solidComponents(self):
        return self._solid_components

    @property
    def solidConnectivity(self):
        return self._solid_connectivity

    @property
    def coupled_components_connectivity(self):
        return self._coupled_components_connectivity

    @property
    def output_parsers(self) -> Dict[str, OutputParser]:
        return self._output_parsers

    @property
    def bodyforces(self) -> List[str]:
        return self._fluid_body_forces

    @property
    def wallfunctions(self) -> List[str]:
        return self._fluid_wall_functions

    @property
    def solid_body_forces(self) -> List[str]:
        return self._solid_body_forces

    @property
    def solid_wall_functions(self) -> List[str]:
        return self._solid_wall_functions

    @property
    def BoundaryConditions(self) -> BoundaryConditions:
        return self._boundary_condition_container

    @property
    def BodyForceContainer(self) -> BodyForces:
        return self._body_force_container

    @property
    def WallFunctionContainer(self) -> WallFunctions:
        return self._wall_function_container

    @property
    def isLoop(self) -> bool:
        return self._isLoop

    @property
    def fluidname(self) -> str:
        return self._fluidname

    @property
    def gasname(self) -> str:
        return self._gasname
