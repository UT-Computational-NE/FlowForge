from typing import List, Tuple
from copy import deepcopy
import numpy as np

import flowforge.input.Components as FluidComponents
import flowforge.input.SolidComponents as SolidComponents

_coupling_registry = {}

def register_coupler(component_cls):
    def decorator(coupler_cls):
        _coupling_registry[component_cls] = coupler_cls
        return coupler_cls
    return decorator

def couple(fluid_component, solid_component):
    """
    Function for coupling together a fluid and solid component

    Parameters
    ----------
    fluid_component : FluidComponents.Pipe
    solid_component : SolidComponents.Component

    Returns
    -------
    coupled_fluid_component : FluidComponents.SerialComponents
    coupled_solid_component : SolidComponents.Component
    """
    fluid_cls = type(fluid_component)
    solid_cls = type(solid_component)
    coupling_cls = _coupling_registry.get(tuple([fluid_cls, solid_cls]))

    return coupling_cls().couple(fluid_component, solid_component)

@register_coupler(Tuple[FluidComponents.Pipe, SolidComponents.Component])
class ComponentCoupler:
    """
    Coupler class used to create a coupled version of a 'fluid pipe component' and a 'solid
    serial component'
    """
    def __init__(self) -> None:
        pass

    def couple(self,
               fluid_component: FluidComponents.Pipe,
               solid_component: SolidComponents.Component
               ) -> Tuple[FluidComponents.Pipe, SolidComponents.Component]:
        """
        Couple method to create a coupling between a fluid pipe component and solid component

        Parameters
        ----------
        fluid_component : FluidComponents.Pipe
        solid_component : SolidComponents.Component

        Returns
        -------
        coupled_fluid_component : FluidComponents.Pipe
        coupled_solid_component : SolidComponents.Component
        """
        coupled_fluid_component = deepcopy(fluid_component)
        coupled_solid_component = deepcopy(solid_component)

        # Update the solid cross-section object
        coupled_solid_component.crossSection.channel = fluid_component.crossSection

        return coupled_fluid_component, coupled_solid_component

@register_coupler(Tuple[FluidComponents.SerialComponents, SolidComponents.Component])
class UniformlyEncasedSerialPipeCoupler:
    """
    Coupler class used to create a coupled version of a 'serial fluid component' and a 'solid
    component'
    """
    def __init__(self):
        pass

    def couple(self,
               fluid_component: FluidComponents.SerialComponents,
               solid_component: SolidComponents.Component
               ) -> Tuple[FluidComponents.SerialComponents, SolidComponents.SerialComponent]:
        """
        Couple method to create a coupling between a serial fluid component and solid component

        Parameters
        ----------
        fluid_component : FluidComponents.SerialComponents
        solid_component : SolidComponents.Component

        Returns
        -------
        coupled_fluid_component : FluidComponents.SerialComponents
        coupled_solid_component : SolidComponents.Component
        """
        # Extract serial-fluid information
        coupled_fluid_component = deepcopy(fluid_component)
        fluid_order = fluid_component.order
        ordered_fluid_comps = fluid_component.orderedComponentsList

        # Couple each solid component
        coupled_solid_components = {}
        solid_order = []
        for fluid_i in range(len(fluid_order)):
            # Needed fluid-information
            fluid_comp = ordered_fluid_comps[fluid_i]
            fluid_height = fluid_comp.length
            fluid_nCells = fluid_comp.nCell

            # Couple component based on input component data
            coupled_solid_name = "c" + str(fluid_i + 1)
            coupled_solid_comp = SolidComponents.Component(height=fluid_height, n_cells=fluid_nCells, material=solid_component.material)
            coupled_solid_comp.crossSection = solid_component.crossSection

            if type(fluid_comp) == FluidComponents.Pipe:
                fluid_channel_cross_section = fluid_comp.crossSection
            elif type(fluid_comp) == FluidComponents.Nozzle:
                previous_fluid_comp = ordered_fluid_comps[fluid_i - 1]
                fluid_channel_cross_section = previous_fluid_comp.crossSection
            else:
                raise Exception(f"Do not have the functionality to couple {type(fluid_comp)} and {type(coupled_solid_comp)}")

            coupled_solid_comp.crossSection.channel = fluid_channel_cross_section
            solid_order.append(coupled_solid_name)
            coupled_solid_components[coupled_solid_name] = coupled_solid_comp

        # couple the serial component
        coupled_solid_component = SolidComponents.SerialComponent(coupled_solid_components, solid_order)

        return coupled_fluid_component, coupled_solid_component


@register_coupler(Tuple[FluidComponents.Pipe, SolidComponents.SerialComponent])
class NonUniformlyEncasedPipe:
    """
    Coupler class used to create a coupled version of a 'fluid pipe component' and a 'solid
    serial component'
    """
    def __init__(self):
        pass

    def couple(self,
               fluid_component: FluidComponents.Pipe,
               solid_component: SolidComponents.SerialComponent
               ) -> Tuple[FluidComponents.Pipe, SolidComponents.SerialComponent]:
        """
        Couple method to create a coupling between a fluid pipe component and serial solid component

        Parameters
        ----------
        fluid_component : FluidComponents.Pipe
        solid_component : SolidComponents.SerialComponent

        Returns
        -------
        coupled_fluid_component : FluidComponents.Pipe
        coupled_solid_component : SolidComponents.SerialComponent
        """
        coupled_fluid_component = deepcopy(fluid_component)
        fluid_channel_cross_section = coupled_fluid_component.crossSection

        coupled_solid_components = {}
        for solid_name, solid_comp in solid_component.components.items():
            coupled_solid_comp = deepcopy(solid_comp)
            coupled_solid_comp.crossSection.channel = fluid_channel_cross_section
            coupled_solid_components[solid_name] = coupled_solid_comp

        coupled_solid_component = SolidComponents.SerialComponent(
            components=coupled_solid_components, order=solid_component.order
        )

        return coupled_fluid_component, coupled_solid_component


@register_coupler(Tuple[FluidComponents.SerialComponents, SolidComponents.SerialComponent])
class NonUniformlyEncasedSerialPipe:
    """
    Coupler class used to create a coupled version of a 'serial fluid component' and a 'solid
    serial component'
    """
    def __init__(self):
        pass

    def _getFluidSerialHeights(self, fluid_components: List[FluidComponents.Component]):
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

    def _addInfinitesimalSolidComponents(
        self, fluid_component: FluidComponents.SerialComponents, solid_component: SolidComponents.SerialComponent
    ):
        """
        When coupling serial fluid components, if the pipe experiences a change in flow area, an infinitesimal
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

        updated_solid_component = SolidComponents.SerialComponent(
            components=updated_solid_components, order=updated_solid_order
        )

        return updated_solid_component

    def _coupleInfinitesimalSolidComponent(self, component_number, ordered_fluid_components, solid_component):
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

        assert isinstance(ordered_fluid_components[component_number], FluidComponents.Nozzle)
        assert np.isclose(ordered_fluid_components[component_number].length, 0.0)

        previous_fluid_component = ordered_fluid_components[component_number - 1]
        fluid_channel_cross_section = previous_fluid_component.crossSection

        coupled_solid_component = deepcopy(solid_component)
        coupled_solid_component.crossSection.channel = fluid_channel_cross_section

        return coupled_solid_component

    def couple(self,
               fluid_component: FluidComponents.SerialComponents,
               solid_component: SolidComponents.SerialComponent
               ) -> Tuple[FluidComponents.SerialComponents, SolidComponents.SerialComponent]:
        """
        Couple method to create a coupling between a serial fluid component and serial solid component

        Parameters
        ----------
        fluid_component : FluidComponents.SerialComponents
        solid_component : SolidComponents.SerialComponent

        Returns
        -------
        coupled_fluid_component : FluidComponents.SerialComponents
        coupled_solid_component : SolidComponents.SerialComponent
        """

        coupled_fluid_component = deepcopy(fluid_component)

        # Adds any necessary infinitesimal components for proper coupling
        solid_component = self._addInfinitesimalSolidComponents(coupled_fluid_component, solid_component)

        # Get the inlets/outlets of the serial components
        ordered_fluid_comps = coupled_fluid_component.orderedComponentsList
        fluid_serial_heights = self._getFluidSerialHeights(ordered_fluid_comps)

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
            if isinstance(fluid_comp, FluidComponents.Nozzle):
                coupled_solid_components[coupled_comp_name] = self._coupleInfinitesimalSolidComponent(
                    i, ordered_fluid_comps, solid_comp
                )
            else:
                _, coupled_solid_components[coupled_comp_name] = ComponentCoupler().couple(fluid_comp, solid_comp)

        # Creates the final, coupled serial component
        coupled_serial_solid_component = SolidComponents.SerialComponent(
            components=coupled_solid_components, order=solid_component.order
        )

        return coupled_fluid_component, coupled_serial_solid_component

@register_coupler(Tuple[FluidComponents.CartCore, SolidComponents.Core])
class CartesianCore:
    def __init__(self):
        pass

    def couple(self, fluid_core, solid_core):
        return