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

    def __init__(self,
                 fluid_component,
                 solid_component):

        self._fluid_component = fluid_component
        self._solid_component = solid_component
        self._component_types = {}
        self._n_cells = 0
        self._height = 0.0

        # perform checks
        self._checkComponentTypes()
        self._checkNumberOfCells()
        self._checkComponentHeight()

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

    def _checkComponentTypes(self):
        """
        Component-type check

        ensures that neither input component is a parallel-component type (which is
        unsupported for this coupling interface), and also sets the 'componentTypes'
        attribute for each input type ('fluid' or 'solid')
        """
        # Fluid Booleans -- Checking component types
        fluidIsSerial = isinstance(self.fluidComponent, FluidComps.SerialComponents)
        fluidIsParallel = isinstance(self.fluidComponent, FluidComps.ParallelComponents)

        # Solid Booleans -- Checking component types
        solidIsSerial = isinstance(self.solidComponent, SolidComps.SerialComponent)
        solidIsParallel = isinstance(self.solidComponent, SolidComps.ParallelComponent)

        assert (not fluidIsParallel) and (not solidIsParallel), "Cannot create a coupling interface between parallel components using CCI"

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


    def _checkNumberOfCells(self):
        """
        Ensures that the fluid and solid components have the same number of cells
        """
        n_fluid_cells = self.fluidComponent.nCell
        n_solid_cells = self.solidComponent.nCells
        assert n_fluid_cells == n_solid_cells, "Coupled components must have the same number of axial cells"
        self._n_axial_cells = n_fluid_cells

    def _checkComponentHeight(self):
        """
        Ensures that the fluid and solid components have the same total height
        """
        fluid_height = self.fluidComponent.length
        solid_height = self.solidComponent.height
        assert fluid_height == solid_height, "Coupled components must be the same height"
        self._height = fluid_height

    @staticmethod
    def _coupleSingularComponents(fluid_component: FluidComps.Component,
                                  solid_component: SolidComps.Component):
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
        assert not isinstance(fluid_component, FluidComps.SerialComponents), "Components passed to this function should always be singular-components"
        assert not isinstance(solid_component, SolidComps.SerialComponent), "Components passed to this function should always be singular-components"
        assert isinstance(fluid_component, FluidComps.Pipe), "Currently only 'Pipe'-type fluid object are supported for fluid-solid coupling"

        # Copt components
        coupled_solid_component = deepcopy(solid_component)

        # Update the solid cross-section object
        fluid_channel_cross_section = fluid_component.crossSection
        coupled_solid_component.crossSection.channel = fluid_channel_cross_section

        return coupled_solid_component

    @staticmethod
    def _breakUpSingularSolidComponent(fluid_comp: FluidComps.SerialComponents,
                                       solid_comp: SolidComps.Component):
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
            coupled_solid_name = "c"+str(f_i+1)
            coupled_solid_comp = SolidComps.Component(
                height=f_height, n_cells=f_nCells, material=solid_comp.material,
                azimuthal_angle=solid_comp.azimuthalAngle, zenith_angle=solid_comp.zenithAngle)
            coupled_solid_comp.crossSection = solid_comp.crossSection

            solid_order.append(coupled_solid_name)
            coupled_solid_comps[coupled_solid_name] = coupled_solid_comp

        # Build the serial component
        coupled_solid_component = SolidComps.SerialComponent(coupled_solid_comps, solid_order)

        return coupled_solid_component


    def _coupleSerialComponents(self,
                                fluid_component: FluidComps.SerialComponents,
                                solid_component: SolidComps.SerialComponent):
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

        # Get the inlets/outlets of the serial components
        ordered_fluid_comps = fluid_component.orderedComponentsList
        fluid_serial_heights = getFluidSerialHeights(ordered_fluid_comps)

        ordered_solid_comps = solid_component.orderedComponents
        solid_serial_heights = solid_component.getComponentInletAndOutlets()

        # Ensures that the fluid & solid domains are the same for each sub-component
        for fluid_domain, solid_domain in zip(fluid_serial_heights, solid_serial_heights):
            assert np.isclose(min(fluid_domain), min(solid_domain))
            assert np.isclose(max(fluid_domain), max(solid_domain))

        # Couples each sub-component
        coupled_solid_components = {}
        for i, (fluid_comp, solid_comp) in enumerate(zip(ordered_fluid_comps, ordered_solid_comps)):
            coupled_comp_name = solid_component.order[i]
            coupled_solid_components[coupled_comp_name] = self._coupleSingularComponents(
                fluid_comp, solid_comp
            )

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

        if self.componentTypes["solid"] == "serial" and self.componentTypes["fluid"] == "component":
            solid_order = solid_comp.order
            ordered_solid_comps = solid_comp.orderedComponents
            coupled_solid_comps = {}
            for name, comp in zip(solid_order, ordered_solid_comps):
                coupled_solid_comps[name] = self._coupleSingularComponents(fluid_comp, comp)
            coupled_solid_comp = SolidComps.SerialComponent(coupled_solid_comps, solid_order)

        if self.componentTypes["solid"] == "component" and self.componentTypes["fluid"] == "serial":
            coupled_solid_comp = self._breakUpSingularSolidComponent(fluid_comp, solid_comp)

        if all(self.componentTypes[i] == "serial" for i in ["solid", "fluid"]):
            coupled_solid_comp = self._coupleSerialComponents(fluid_comp, solid_comp)

        return fluid_comp, coupled_solid_comp


class ParallelCoupledComponents:
    """
    Interface to handle the mapping between fluid- and solid-parallel components

    Parameters
    ----------
    fluid_parallel_component : FluidComponents.ParallelComponent
        Parallel fluid component object
    solid_parallel_component : SolidComponents.ParallelComponent
        Parallel solid component object

    Attributes
    ----------
    parallelFluidComponent : FluidComponents.ParallelComponent
        Parallel fluid component object
    fluidComponents : Dict[str, FluidComponents.Component]
        Dict of all parallel components in 'parallelFluidComponent' (not including
        non-parallel components, such as plenums or annulus)
    fluidComponentMap : List[List[str]]
        Spatial map of where these fluid components are placed, relative to one
        another

    parallelSolidComponent : SolidComponents.ParallelComponent
        Parallel solid component object
    solidComponents : Dict[str, SolidComponents.Component]
         Dict of all parallel components in 'solidComponents'
    solidComponentMap : List[List[str]]
        Spatial map of where these solid components are placed, relative to one
        another

    mapping : List[List[Tuple[str, str]]]
        Spatial map relating the solid and fluid component maps, where each element
        is a tuple, with the tuples first element being the fluid-component key, and
        the second being the solid-component key
    """

    def __init__(self,
                 fluid_parallel_component: FluidComps.ParallelComponents,
                 solid_parallel_component: SolidComps.ParallelComponent):
        # Fluid
        self._fluid_parallel_component = fluid_parallel_component
        self._fluid_components = fluid_parallel_component.myParallelComponents
        self._fluid_component_map = fluid_parallel_component.componentMap

        # Solid
        self._solid_parallel_component = solid_parallel_component
        self._solid_components = solid_parallel_component.components
        self._solid_component_map = solid_parallel_component.componentMap

        # Mapping between component maps
        self._mapping = self._buildMapping(self.fluidComponentMap,
                                           self.solidComponentMap)


    @property
    def parallelFluidComponent(self):
        return self._fluid_parallel_component

    @property
    def fluidComponents(self):
        return self._fluid_components

    @property
    def fluidComponentMap(self):
        return self._fluid_component_map

    @property
    def parallelSolidComponent(self):
        return self._solid_parallel_component

    @property
    def solidComponents(self):
        return self._solid_components

    @property
    def solidComponentMap(self):
        return self._solid_component_map

    @property
    def mapping(self):
        return self._mapping

    def _buildMapping(self, fluid_map, solid_map):
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
        assert len(solid_map) == len(fluid_map), "Maps are not of the same size"
        assert all(len(s_i) == len(f_i) for s_i, f_i in zip(solid_map, fluid_map)), "Maps are not of the same size"

        mapping = []
        for fluid_row, solid_row in zip(fluid_map, solid_map):
            row_mapping = []
            for fluid_element, solid_element in zip(fluid_row, solid_row):
                row_mapping.append(tuple([fluid_element, solid_element]))
            mapping.append(row_mapping)

        return mapping

    def buildCoupledComponents(self):
        """
        Using the mapping between the solid and fluid parallel components, this method
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
        coupled_solid_components : Dict[str, SolidComponent]
        """

        coupled_components = []
        coupled_fluid_components = {}
        coupled_solid_components = {}
        for row in self.mapping:
            for element in row:
                fluid_key = element[0]
                solid_key = element[1]
                coupled_key =  fluid_key + "_" + solid_key
                if coupled_key not in coupled_components:
                    # Copy components
                    fluid_comp = deepcopy(self.fluidComponents[fluid_key])
                    solid_comp = deepcopy(self.solidComponents[solid_key])

                    CCI = CoupledComponentInterface(fluid_comp, solid_comp)

                    coupled_components = CCI.buildCoupledComponents()

                    # Add components to dict
                    coupled_components.append(coupled_key)
                    coupled_fluid_components[coupled_key] = coupled_components[0]
                    coupled_solid_components[coupled_key] = coupled_components[1]

        return coupled_fluid_components, coupled_solid_components
