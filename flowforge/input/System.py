from typing import Dict, List, Tuple, Generator
from copy import deepcopy
import numpy as np
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.Components import Component, Nozzle, HexCore, Pump
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.BoundaryConditions import MassMomentumBC, EnthalpyBC, VoidBC, BoundaryConditions
from flowforge.input.BodyForces import BodyForces, GravitationalBF, PumpBF
from flowforge.input.WallFunctions import WallFunctions, FrictionWF
from flowforge.parsers.OutputParser import OutputParser


def make_continuous(components: List[Component], order: List[dict]):
    """Private method makes serial components continuous with respect to area change

    This method takes in a list of components and their order and inserts infinitesimal nozzles between them
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
    discontinuity_found = True
    while discontinuity_found:
        discontinuity_found = False
        # initialize the previous area as the first area
        prev_area = components[order[0]["component"]].inletArea
        for i, entry in enumerate(order):
            if abs(prev_area - components[entry["component"]].inletArea) > 1.0e-12 * min(
                prev_area, components[entry["component"]].inletArea
            ):
                temp_nozzle = Nozzle(
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
                components[
                    f'temp_nozzle_for_make_continuous_creation_in_system_{entry["component"]}_{num_connects}'
                ] = deepcopy(temp_nozzle)
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
                discontinuity_found = True
                break
            prev_area = components[entry["component"]].outletArea
    return components, order


class System:
    """A class for representing a whole system of components

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
    output_parsers : Dict[str, OutputParser]
        The output parsers with which to parse output from various models and map to the System
    """

    def __init__(self, components: Dict[str, Component], sys_dict: Dict,
                 unit_dict: Dict[str, str], controller_dict: Dict) -> None:
        self._components = []
        self._output_parsers = {}
        self._connectivity = []
        self._body_forces = []
        self._wall_functions = []
        self._VBC = None
        self._fluid = None
        self._gas = None

        self._BoundaryConditions = None  # ** Boundary Conditions **
        self._isLoop = False  # Boolean defining if system is a loop or segment

        if "simple_loop" in sys_dict:
            self._setupSimpleLoop(components, **sys_dict["simple_loop"])
        elif "segment" in sys_dict:
            self._setupSegment(components, controller_dict, **sys_dict["segment"])
        # TODO add additional types of systems that can be set up

        if "parsers" in sys_dict:
            self._setupParsers(sys_dict["parsers"])

        for comp in self._components:
            comp._convertUnits(UnitConverter(unit_dict))
        # convert BC units
        if self._VBC is not None:
            self._VBC._convertUnits(UnitConverter(unit_dict))
        if self._BoundaryConditions is not None:
            self._BoundaryConditions._convertUnits(UnitConverter(unit_dict))

    @property
    def core(self):
        """
        Returns the core component of the system.
        """
        for comp in self._components:
            if isinstance(comp, HexCore):
                return comp

        return Exception

    def _setupSimpleLoop(
        self,
        components: Dict[str, Component],
        controller_dict: Dict,
        unit_dict: Dict,
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

        components, loop = make_continuous(components, loop)
        self._fluidname = fluid.lower()
        self._gasname = gas if gas is None else gas.lower()
        # Loop over each component in the loop, add those components to the list, define the connections between components
        for i, entry in enumerate(loop):
            # Collect the body forces and wall functions associated with each component
            component_body_force_names = entry.get("BodyForces", [])
            component_wall_function_names = entry.get("BodyForces", [])
            component_body_forces = self._setupBodyForces(controller_dict["bodyForce"],
                                                          component_body_force_names,
                                                          unit_dict)
            component_wall_functions = self._setupWallFunctions(controller_dict["wallFunctions"],
                                                                component_wall_function_names,
                                                                unit_dict)

            # Add component and its body forces and wall functions
            self._components.append(deepcopy(components[entry["component"]]))
            self._body_forces.append([obj for _,obj in component_body_forces.items()])
            self._wall_functions.append([obj for _,obj in component_wall_functions.items()])

            # connect the current component to the previous (exclude the first component because there isn't a previous)
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))
            # If the last entry in the loop, connect the last component to the first
            if i == len(loop) - 1:
                self._connectivity.append((self._components[i], self._components[0]))

        # get the boundary conditions
        self._setupBoundaryConditions(boundary_conditions)

    def _setupSegment(
        self,
        components: List[Component],
        controller_dict: Dict,
        unit_dict: Dict,
        order: List[dict],
        boundary_conditions: Dict = {},
        fluid: str = "FLiBe",
        gas=None
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

        # Get defaults for friction and gravity
        default_gravity = self._getGravity(controller_dict["bodyForce"])
        default_friction = self._getDefaultFriction(controller_dict["wallFunctions"])

        # Loop over each entry in segment, add the components, and connect the components to each other
        for i, entry in enumerate(order):
            # Collect the body forces and wall functions associated with each component
            component_body_force_names = entry.get("BodyForces", [])
            component_wall_function_names = entry.get("BodyForces", [])
            component_body_forces = self._setupBodyForces(controller_dict["bodyForce"],
                                                          component_body_force_names,
                                                          unit_dict, default_gravity)
            component_wall_functions = self._setupWallFunctions(controller_dict["wallFunctions"],
                                                                component_wall_function_names,
                                                                unit_dict, default_friction)

            # Add pump body force
            if isinstance(components[entry["component"]], Pump):
                self._body_forces.append(PumpBF(components[entry["component"]].getPressureChange()))

            # Add component and its body forces and wall functions
            self._components.append(deepcopy(components[entry["component"]]))
            self._body_forces.append([obj for _,obj in component_body_forces.items()])
            self._wall_functions.append([obj for _,obj in component_wall_functions.items()])

            # Adds a connection between the current and previous component
            if i > 0:
                self._connectivity.append((self._components[i - 1], self._components[i]))

        # get the boundary conditions
        self._setupBoundaryConditions(boundary_conditions)

    def _setupParsers(self, parser_dict: Dict) -> None:
        """Private method for setting up output parsers

        Parameters
        ----------
        parser_dict : Dict
            The input dictionary specifying the parsers to be setup
        """

        raise NotImplementedError("To Be Implemented")

    def _setupBoundaryConditions(self, boundary_conditions : Dict):
        self._VBC = None
        if "void" in boundary_conditions:
            self._VBC = VoidBC(**boundary_conditions["void"])
        self._BoundaryConditions = None
        if ("void" not in boundary_conditions):
            self._BoundaryConditions = BoundaryConditions(**boundary_conditions)

    def _setupBodyForces(self, all_body_force_definitions : Dict, component_body_force_names, unit_dict,
                         gravity: GravitationalBF):
        # Extract body forces
        component_body_force_definitions = {name: definition for name, definition in
                                            all_body_force_definitions.items() if name in
                                            component_body_force_names}
        component_body_forces = BodyForces(**component_body_force_definitions)

        # Handle gravity
        if any([type(bf) == GravitationalBF for _, bf in component_body_forces.items()]):
            raise Exception("ERROR: Please do not define 'gravity' within components. Please only define a" \
            "single, global gravity under 'controllers'. (Note that the default gravity is set to <0, 0, -1> " \
            "with a magnitude of 9.81 m/s^2.)")
        component_body_forces["gravity"] = gravity

        # Convert units
        component_body_forces._convertUnits(UnitConverter(unit_dict))

        return component_body_forces

    def _setupWallFunctions(self, all_wall_function_definitions : Dict, component_wall_function_names, unit_dict,
                            default_friction: FrictionWF):
        # Extract wall functions
        component_wall_function_definitions = {name: definition for name, definition in
                                            all_wall_function_definitions.items() if name in
                                            component_wall_function_names}
        component_wall_functions = WallFunctions(**component_wall_function_definitions)

        # Handle friction
        friction_wfs = [wf for _, wf in component_wall_functions.items()]
        if len(friction_wfs) == 0:
            component_wall_functions["friction"] = default_friction
        elif len(friction_wfs) > 1:
            raise Exception("ERROR: Can only handle (1) friction formulation per component.")

        # Convert units
        component_wall_functions._convertUnits(UnitConverter(unit_dict))

        return component_wall_functions


    def _getDefaultFriction(self, wall_function_dict):
        wall_functions = WallFunctions(**wall_function_dict)
        friction_wfs = [wf for _, wf in wall_functions.items() if type(wf) == FrictionWF]
        if len(friction_wfs) == 0:
            default_friction = FrictionWF("default", True)
        elif not any([wf.isDefault for wf in friction_wfs]):
            default_friction = FrictionWF("default", True)
        else:
            defaults = [wf for wf in friction_wfs if wf.isDefault]
            if len(defaults) > 1:
                raise Exception("ERROR: Can only handle (1) default friction formulation.")
            default_friction = defaults[0]
        return default_friction

    def _getGravity(self, body_force_dict):
        body_forces = BodyForces(**body_force_dict)
        gravity_bfs = [bf for _, bf in body_forces.items() if type(bf) == GravitationalBF]
        if len(gravity_bfs) == 0:
            default_gravity = GravitationalBF(tuple([0,0,-1]), 9.81)
        elif len(gravity_bfs) > 1:
            raise Exception("ERROR: Cannot enter more than (1) gravity BF.")
        else:
            default_gravity = gravity_bfs[0]
        return default_gravity

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

    @property
    def nCell(self) -> int:
        n_cells = 0
        for c in self._components:
            n_cells += c.nCell
        return n_cells

    @property
    def components(self) -> List[Component]:
        return self._components

    @property
    def connectivity(self) -> List[Tuple[Component, Component]]:
        return self._connectivity

    @property
    def output_parsers(self) -> Dict[str, OutputParser]:
        return self._output_parsers

    @property
    def bodyforces(self) -> List[str]:
        return self._body_forces

    @property
    def wallfunctions(self) -> List[str]:
        return self._wall_functions

    @property
    def VBC(self) -> VoidBC:
        return self._VBC

    @property
    def BoundaryConditions(self) -> BoundaryConditions:
        return self._BoundaryConditions

    @property
    def isLoop(self) -> bool:
        return self._isLoop

    @property
    def fluidname(self) -> str:
        return self._fluidname

    @property
    def gasname(self) -> str:
        return self._gasname
