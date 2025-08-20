from typing import List
from copy import deepcopy
import numpy as np

import flowforge.input.Components as FluidComps
import flowforge.input.SolidComponents as SolidComps


class CoupledComponentInterface:
    """
    Interface to couple two (non-parallel) components

    Parameters
    ----------
    fluid_component : FluidComponents.(Component or SerialComponent)
        Fluid component used for coupling. This component may be either a
        singular component or a serial component
    solid_component : SolidComponents.(Component or SerialComponent)
        Solid component used for coupling. This component may be either a
        singular component or a serial component

    Attributes
    ----------
    fluidComponent : FluidComponents.(Component or SerialComponent)
        Fluid component used for coupling. This component may be either a
        singular component or a serial component
    solidComponent : SolidComponents.(Component or SerialComponent)
        Solid component used for coupling. This component may be either a
        singular component or a serial component
    nCells : int
        Number of total cells in both the solid and fluid component
    height : float
        Total height of both the solid and fluid component
    componentTypes : Dict[str, str]
        Component types (either 'component' or 'serial') for both input
        types ('fluid' or 'solid'). This is used in the
                    'buildCoupledComponents'
        method to determine which algorithm should be used to create the
        coupled components
    """

    def __init__(self, fluid_component, solid_component):

        self._fluid_component = fluid_component
        self._solid_component = solid_component
        self._component_types = {}
        self._n_cells = 0
        self._height = 0.0

        # perform checks
        self._checkComponentTypes(self.fluidComponent, self.solidComponent)
        if all(not isinstance(comp, FluidComps.Nozzle) for comp in fluid_component.baseComponents):
            self._checkNumberOfCells(self.fluidComponent.nCell, self.solidComponent.nCells)
        self._checkComponentHeight(self.fluidComponent.length, self.solidComponent.height)

    @property
    def fluidComponent(self):
        return self._fluid_component

    @property
    def solidComponent(self):
        return self._solid_component

    @property
    def nCells(self):
        return self._n_cells

    @property
    def height(self):
        return self._height

    @property
    def componentTypes(self):
        return self._component_types

    def _checkComponentTypes(self, fluid_comp, solid_comp):
        """
        Component-type check

        ensures that neither input component is a parallel-component type (which is
        unsupported for this coupling interface), and also sets the 'componentTypes'
        attribute for each input type ('fluid' or 'solid')
        """
        # Fluid Booleans -- Checking component types
        fluidIsSerial = isinstance(fluid_comp, FluidComps.SerialComponents)
        fluidIsParallel = isinstance(fluid_comp, FluidComps.ParallelComponents)

        # Solid Booleans -- Checking component types
        solidIsSerial = isinstance(solid_comp, SolidComps.SerialComponent)
        solidIsParallel = isinstance(solid_comp, SolidComps.ParallelComponent)

        assert (not fluidIsParallel) and (
            not solidIsParallel
        ), "Cannot create a coupling interface between parallel components using CCI"

        # Set fluid component type
        if fluidIsSerial:
            self._component_types["fluid"] = "serial"
        else:
            self._component_types["fluid"] = "component"

        # Set solid component type
        if solidIsSerial:
            self._component_types["solid"] = "serial"
        else:
            self._component_types["solid"] = "component"

    def _checkNumberOfCells(self, n_fluid_cells, n_solid_cells):
        """
        Ensures that the fluid and solid components have the same number of cells.

        NOTE that, if a Nozzle-type exists in the fluid, there are likely some added, infinitesimal transitional cells added to
        ensure that the pipes are continuous. In that case, this check should be ignored as the fluid and solid will almost
        certainly not have the same number of cells.
        """
        assert n_fluid_cells == n_solid_cells, "Coupled components must have the same number of axial cells"
        self._n_axial_cells = n_fluid_cells

    def _checkComponentHeight(self, fluid_height, solid_height):
        """
        Ensures that the fluid and solid components have the same total height
        """
        assert fluid_height == solid_height, "Coupled components must be the same height"
        self._height = fluid_height

    @staticmethod
    def _coupleSingularComponents(fluid_component: FluidComps.Component, solid_component: SolidComps.Component):
        """
        Given two singular (non-serial) components, this method couples them together.

        The actual coupling is performed by altering the 'CrossSection' of the solid component,
        adding the fluid component's 'CrossSection' as a hold in the solid's

        Parameters
        ----------
        fluid_component: FluidComps.Component
            Singular fluid component used in the coupling
        solid_component: SolidComps.Component
            Singular solid component used in the coupling

        Returns
        coupled_solid_component : SolidComps.Component
            Coupled solid component
        """
        # All components passed to this function should be singular (non-serial)
        assert not isinstance(
            fluid_component, FluidComps.SerialComponents
        ), "Components passed to this function should always be singular-components"
        assert not isinstance(
            solid_component, SolidComps.SerialComponent
        ), "Components passed to this function should always be singular-components"
        assert isinstance(
            fluid_component, FluidComps.Pipe
        ), "Currently only 'Pipe'-type fluid object are supported for fluid-solid coupling"

        # Copt components
        coupled_solid_component = deepcopy(solid_component)

        # Update the solid cross-section object
        fluid_channel_cross_section = fluid_component.crossSection
        coupled_solid_component.crossSection.channel = fluid_channel_cross_section

        return coupled_solid_component

    @staticmethod
    def _breakUpSingularSolidComponent(fluid_comp: FluidComps.SerialComponents, solid_comp: SolidComps.Component):
        """
        If the solid component is a singular component (non-serial), and the fluid component is a serial component,
        this method breaks the singular solid component into 'n'-serial component, where 'n' is the number of serial
        components in the fluid component.

        For each new solid serial component, it is coupled to the respective fluid component

        Parameters
        ----------
        fluid_comp: FluidComps.SerialComponents
            Fluid component used for coupling
        solid_comp: SolidComps.Component
            Solid component used for coupling

        Returns
        -------
        coupled_solid_component : SolidComps.SerialComponents
            Coupled serial component
        """
        # Extract serial-fluid information
        fluid_order = fluid_comp.order
        ordered_fluid_comps = fluid_comp.orderedComponentsList

        # Build each solid component
        coupled_solid_comps = {}
        solid_order = []
        for f_i in range(len(fluid_order)):
            # Needed fluid-information
            f_comp = ordered_fluid_comps[f_i]
            f_height = f_comp.length
            f_nCells = f_comp.nCell

            # Build component based on input component data
            coupled_solid_name = "c" + str(f_i + 1)
            coupled_solid_comp = SolidComps.Component(height=f_height, n_cells=f_nCells, material=solid_comp.material)
            coupled_solid_comp.crossSection = solid_comp.crossSection

            solid_order.append(coupled_solid_name)
            coupled_solid_comps[coupled_solid_name] = coupled_solid_comp

        # Build the serial component
        coupled_solid_component = SolidComps.SerialComponent(coupled_solid_comps, solid_order)

        return coupled_solid_component

    def _coupleSerialComponents(
        self, fluid_component: FluidComps.SerialComponents, solid_component: SolidComps.SerialComponent
    ):
        """
        If both the fluid and solid are serial components, this method maps and couples each solid
        component to its respective fluid component.

        NOTE that, for this version, the algorithm requires that the serial fluid component and the
             serial solid component are built in the same fashion. That is, each sub-component on
             the fluid side is the same length as a corresponding fluid component, making them able
             to map 1:1 to each other

        Parameters
        ----------
        fluid_component : FluidComps.SerialComponents
            Fluid serial-component which needs to be coupled
        solid_component : SolidComps.SerialComponent
            Solid serial-component which needs to be coupled

        Returns
        -------
        coupled_serial_solid_component : SolidComps.SerialComponent
            Final coupled version of the solid serial-component
        """

        def getFluidSerialHeights(fluid_components: List[FluidComps.Component]):
            """
            Given a serial fluid component, this method determines the domain (minimum and maximum z
            coordinates) for each sub-component.

            Parameters
            ----------
            fluid_components : List[FluidComps.Component]
                List of input fluid sub-components

            Returns
            -------
            fluid_serial_heights : List[Tuple[float, float]]
                List of the minimum and maximum z-coordinates for each sub-component
            """
            fluid_serial_heights = []
            f_inlet = 0.0
            for f_comp in fluid_components:
                inlet_outlet = (f_inlet, f_inlet + f_comp.length)
                fluid_serial_heights.append(inlet_outlet)
                f_inlet += f_comp.length
            return fluid_serial_heights

        def addInfinitesimalSolidComponents(
            fluid_component: FluidComps.SerialComponents, solid_component: SolidComps.SerialComponent
        ):
            """
            When building serial fluid components, if the pipe experiences a change in flow area, an infinitesimal
            nozzle component is added for continuity. This method (1) locates those added fluid components and (2) if
            their counterparts do not exist on the solid side, they are added.

            Parameters
            ----------
            fluid_component : FluidComps.SerialComponents
                Fluid serial-component which needs to be coupled
            solid_component : SolidComps.SerialComponent
                Solid serial-component which needs to be coupled

            Returns
            -------
            updated_solid_component : SolidComps.SerialComponent
                Solid serial-component which needs to be coupled
            """

            if len(fluid_component.orderedComponentsList) == len(solid_component.orderedComponents):
                return solid_component

            common_name = "temp_nozzle_for_make_continuous_creation_in_serialcomp"
            base_solid_name = "infinitesimal_component_for_continuous_fluid_component"
            updated_solid_order = []
            j = 0
            for comp_name in fluid_component.order:
                if common_name not in comp_name:
                    updated_solid_order.append(solid_component.order[j])
                    j += 1
                    continue
                updated_solid_order.append(base_solid_name + "__" + solid_component.order[j])

            updated_solid_components = {}
            for comp_name in updated_solid_order:
                if comp_name in solid_component.components:
                    updated_solid_components[comp_name] = solid_component.components[comp_name]
                    continue
                previous_comp_name = comp_name.split("__")[1]
                new_component = deepcopy(solid_component.components[previous_comp_name])
                new_component.height = 1e-64
                new_component.nCells = 1
                updated_solid_components[comp_name] = new_component

            updated_solid_component = SolidComps.SerialComponent(
                components=updated_solid_components, order=updated_solid_order
            )

            return updated_solid_component

        def coupleInfinitesimalSolidComponent(component_number, ordered_fluid_components, solid_component):
            """
            For these infinitesimal components made for fluid-component continuity, this method couples them to
            their corresponding infinitesimal solid components. To do so, since the nozzle has a non-uniform cross
            section, the previous fluid component's cross-section will be used.

            These infinitesimal components have a defined height of 1e-64 m, so the approximation of using the previous
            component's cross-section will have negligible impact on the system.

            Parameters
            ----------
            component_number : int
                Current component in examination. This parameter is used to determine the previous fluid component to
                extract its cross-section
            ordered_fluid_components : List[FluidComponents.Component]
                List of all fluid components in order
            solid_component : SolidComponents.Component
                Singular solid component used in the coupling

            Returns
            -------
            coupled_solid_component : SolidComps.Component
                Coupled solid component

            """

            assert isinstance(ordered_fluid_components[component_number], FluidComps.Nozzle)
            assert np.isclose(ordered_fluid_components[component_number].length, 0.0)

            previous_fluid_component = ordered_fluid_components[component_number - 1]
            fluid_channel_cross_section = previous_fluid_component.crossSection

            coupled_solid_component = deepcopy(solid_component)
            coupled_solid_component.crossSection.channel = fluid_channel_cross_section

            return coupled_solid_component

        # Adds any necessary infinitesimal components for proper coupling
        solid_component = addInfinitesimalSolidComponents(fluid_component, solid_component)

        # Get the inlets/outlets of the serial components
        ordered_fluid_comps = fluid_component.orderedComponentsList
        fluid_serial_heights = getFluidSerialHeights(ordered_fluid_comps)

        ordered_solid_comps = solid_component.orderedComponents
        solid_serial_heights = solid_component.getComponentInletAndOutlets()

        # Ensures that the fluid & solid domains are the same for each sub-component
        for i, (fluid_domain, solid_domain) in enumerate(zip(fluid_serial_heights, solid_serial_heights)):
            assert np.isclose(
                min(fluid_domain), min(solid_domain)
            ), f"min fluid: {min(fluid_domain)} | min solid: {min(solid_domain)}"
            assert np.isclose(
                max(fluid_domain), max(solid_domain)
            ), f"max fluid: {max(fluid_domain)} | max solid: {max(solid_domain)}"

        # Couples each sub-component
        coupled_solid_components = {}
        for i, (fluid_comp, solid_comp) in enumerate(zip(ordered_fluid_comps, ordered_solid_comps)):
            coupled_comp_name = solid_component.order[i]
            if isinstance(fluid_comp, FluidComps.Nozzle):
                coupled_solid_components[coupled_comp_name] = coupleInfinitesimalSolidComponent(
                    i, ordered_fluid_comps, solid_comp
                )
                continue
            coupled_solid_components[coupled_comp_name] = self._coupleSingularComponents(fluid_comp, solid_comp)

        # Creates the final, coupled serial component
        coupled_serial_solid_component = SolidComps.SerialComponent(
            components=coupled_solid_components, order=solid_component.order
        )
        return coupled_serial_solid_component

    def buildCoupledComponents(self):
        """
        Main function for the CCI.
        Handles the creation of coupled components, based on the input components of the class.

        Returns
        -------
        fluid_comp : FluidComps.(Component or SerialComponent)
            Coupled fluid component (Note that, for now, fluid components are unchanged in the
            coupling)
        coupled_solid_comp : SolidComps.(Component or SerialComponent)
            Coupled solid component
        """
        fluid_comp = deepcopy(self.fluidComponent)
        solid_comp = deepcopy(self.solidComponent)

        if all(self.componentTypes[i] == "component" for i in ["solid", "fluid"]):
            coupled_solid_comp = self._coupleSingularComponents(fluid_comp, solid_comp)

        elif self.componentTypes["solid"] == "serial" and self.componentTypes["fluid"] == "component":
            solid_order = solid_comp.order
            ordered_solid_comps = solid_comp.orderedComponents
            coupled_solid_comps = {}
            for name, comp in zip(solid_order, ordered_solid_comps):
                coupled_solid_comps[name] = self._coupleSingularComponents(fluid_comp, comp)
            coupled_solid_comp = SolidComps.SerialComponent(coupled_solid_comps, solid_order)

        elif self.componentTypes["solid"] == "component" and self.componentTypes["fluid"] == "serial":
            broken_solid_comp = self._breakUpSingularSolidComponent(fluid_comp, solid_comp)
            coupled_solid_comp = self._coupleSerialComponents(fluid_comp, broken_solid_comp)

        elif all(self.componentTypes[i] == "serial" for i in ["solid", "fluid"]):
            coupled_solid_comp = self._coupleSerialComponents(fluid_comp, solid_comp)

        else:
            solid_type = self.componentTypes["solid"]
            fluid_type = self.componentTypes["fluid"]
            raise Exception(f"Error: Solid of type {solid_type} and Fluid of type {fluid_type}")

        return fluid_comp, coupled_solid_comp


class CoupledCoreComponentInterface:
    """
    Interface to handle the mapping between fluid- and solid-core components

    Parameters
    ----------
    fluid_core_component : FluidComponents.Core
        Core fluid component object
    solid_core_component : SolidComponents.Core
        Core solid component object

    Attributes
    ----------
    coreFluidComponent : FluidComponents.Core
        Core fluid component object
    fluidComponents : Dict[str, FluidComponents.Component]
        Dict of all parallel components in 'coreFluidComponent' (not including
        non-parallel components, such as plenums or annulus)
    fluidComponentMap : List[List[str]]
        Spatial map of where these fluid components are placed, relative to one
        another

    coreSolidComponent : SolidComponents.Core
        core solid component object
    solidComponents : Dict[str, SolidComponents.Component]
         Dict of all parallel components in 'coreSolidComponent'
    solidComponentMap : List[List[str]]
        Spatial map of where these solid components are placed, relative to one
        another

    mapping : List[List[Tuple[str, str]]]
        Spatial map relating the solid and fluid component maps, where each element
        is a tuple, with the tuples first element being the fluid-component key, and
        the second being the solid-component key
    """

    def __init__(self, fluid_core_component: FluidComps.Core, solid_core_component: SolidComps.Core):
        # Fluid
        self._fluid_core_component = fluid_core_component
        self._fluid_components = fluid_core_component.myParallelComponents
        self._fluid_component_map = fluid_core_component.componentMap

        # Solid
        self._solid_core_component = solid_core_component
        self._solid_components = solid_core_component.components
        self._solid_component_map = solid_core_component.componentMap

        # Mapping between component maps
        self._mapping = self._buildMapping(self.fluidComponentMap, self.solidComponentMap)

    @property
    def coreFluidComponent(self):
        return self._fluid_core_component

    @property
    def fluidComponents(self):
        return self._fluid_components

    @property
    def fluidComponentMap(self):
        return self._fluid_component_map

    @property
    def coreSolidComponent(self):
        return self._solid_core_component

    @property
    def solidComponents(self):
        return self._solid_components

    @property
    def solidComponentMap(self):
        return self._solid_component_map

    @property
    def mapping(self):
        return self._mapping

    @staticmethod
    def _buildMapping(fluid_map, solid_map):
        """
        Builds a map of connections between the solid and fluid component maps

        Parameters
        ----------
        fluid_map : List[List[str]]
            map of the fluid components
        solid_map : List[List[str]]
            map of the solid components

        Returns
        -------
        mapping : List[List[Tuple[str, str]]]
            mapping between the solid_map and fluid_map, where each element is a
            tuple of (solid component, fluid component)
        """
        real_fluid_map = [[elem for elem in row if elem is not None] for row in fluid_map]

        assert len(solid_map) == len(real_fluid_map), "Maps are not of the same size"
        assert all(len(s_i) == len(f_i) for s_i, f_i in zip(solid_map, real_fluid_map)), "Maps are not of the same size"

        mapping = []
        for fluid_row, solid_row in zip(real_fluid_map, solid_map):
            row_mapping = []
            for fluid_element, solid_element in zip(fluid_row, solid_row):
                row_mapping.append(tuple([fluid_element, solid_element]))
            mapping.append(row_mapping)

        return mapping

    def buildCoupledComponents(self):
        """
        Using the mapping between the solid and fluid core components, this method
        couples each pair of component types, outputting a coupled version of the input.

        The key for the solid and fluid coupled components is in the form "i_j", where
        i and j correspond to values in the coupled map, where elements are in the form
        (i,j)

        To perform the coupling, a pair of components are sent to the
                                "ComponentCouplingInterface"
        where a new pair of coupled components are output from this interface.

        Returns
        -------
        coupled_fluid_components : Dict[str, fluidComponent]
            Set of coupled fluid components
        coupled_solid_components : Dict[str, SolidComponent]
            Set of coupled solid components
        """

        coupled_components_keys = []
        coupled_fluid_components = {}
        coupled_solid_components = {}
        for i, row in enumerate(self.mapping):
            for j, element in enumerate(row):
                fluid_key = element[0]
                full_fluid_key = f"{fluid_key}-{i+1}-{j+1}"
                solid_key = element[1]
                coupled_key = fluid_key + "_" + solid_key
                if coupled_key not in coupled_components_keys:
                    # Copy components
                    fluid_comp = deepcopy(self.fluidComponents[full_fluid_key])
                    solid_comp = deepcopy(self.solidComponents[solid_key])

                    CCI = CoupledComponentInterface(fluid_comp, solid_comp)

                    coupled_components = CCI.buildCoupledComponents()

                    # Add components to dict
                    coupled_components_keys.append(coupled_key)
                    coupled_fluid_components[fluid_key] = coupled_components[0]
                    coupled_solid_components[coupled_key] = coupled_components[1]

        return coupled_fluid_components, coupled_solid_components

    def buildCoupledCores(self):
        """
        Builds the coupled cores, and returns the new core objects as well as the coupled mapping
        between them.

        This method utilizes the 'buildCoupledComponents' to derive the set of coupled components,
        and then creates a new solid core using these coupled solid components.

        For the fluid core, since it remains unchanged due to the coupling, the original core object
        is copied and returned.

        Returns
        -------
        coupled_fluid_core : FluidComponents.Core
            Coupled version of the fluid core
        coupled_solid_core : SolidComponents.Core
            Coupled version of the solid core
        coupled_map : List[List[Tuple[str, str]]]
            Mapping between fluid and solid (parallel) components in space

        """
        _, coupled_solid_components = self.buildCoupledComponents()

        coupled_map = deepcopy(self.mapping)
        coupled_fluid_core = deepcopy(self.coreFluidComponent)
        coupled_solid_core = SolidComps.Core(components=coupled_solid_components, component_map=self.solidComponentMap)

        return coupled_fluid_core, coupled_solid_core, coupled_map
