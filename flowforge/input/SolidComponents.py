from typing import List, Dict, Tuple, Optional
from copy import deepcopy

# from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.Components import FluidCrossSection
import flowforge.input.Shapes as Shapes

# pragma pylint: disable=protected-access

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""
solid_component_list = {}


class SolidCrossSection(Shapes.CrossSection):
    """
    Solid cross section class

    Builds a cross section for solid components based off an input shape, computing geometric
    quantities needed for the solid solve

    Parameters
    ----------
    shape : Shape
        Shape type desired for the cross section

    Attributes
    ----------
    area : float
        Area of the solid component, defined as the base area minus the channel area
    baseArea : float
        Area of the shape input for this component
    channel : FluidCrossSection
        cross-section of the sub-channel running through this cross-section
    """

    def __init__(self, shape, **kwargs):
        super().__init__(shape, **kwargs)
        self._channel = None

    @property
    def area(self) -> float:
        if self.channel is None:
            return self.baseArea
        return self.baseArea - self.channel.flow_area

    @property
    def baseArea(self) -> float:
        return self.shape.area

    @property
    def channel(self) -> FluidCrossSection:
        return self._channel

    @channel.setter
    def channel(self, channel: FluidCrossSection):
        assert self.channel is None, "Should use 'changeChannel' method instead'"
        assert channel.flow_area < self.baseArea
        self._channel = channel

    def changeChannel(self, channel: FluidCrossSection) -> None:
        """
        Method used to explicitly change the channel running through the cross-section

        Parameters
        ----------
        channel : FluidCrossSection
            New channel running through the cross-section
        """
        assert channel.flow_area < self.baseArea
        self._channel = channel

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Private method for converting units of the component's internal attribute

        This method is especially useful for converting components to the expected units
        of the application in which they will be used.

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        self.shape._convertUnits(uc)
        if self.channel is not None:
            self.channel._convertUnits(uc)


class Component:
    """
    Abstract base class for solid components in the system

    Parameters
    ----------
    height : float
        Height of the component
    n_cells : int
        Number of discretizations made for the component
    material : str
        Name of the material used for the component
    azimuthal_angle : float
        Orientation angle of the component in the azimuthal direction
    zenith_angle : float
        Orientation angle of the component in the zenith (polar) direction
    cross_section : str, optional
        Type of cross-section used for the component

    Attributes
    ----------
    volume : float
        Volume of the component
    height : float
        Height of the component
    nCells : int
        Number of discretizations made for the component
    material : str
        Name of the material used for the component
    azimuthalAngle : float
        Orientation angle of the component in the azimuthal direction
    zenithAngle : float
        Orientation angle of the component in the zenith (polar) direction
    crossSection : SolidCrossSections.CrossSection
        Cross section used for the component
    """

    def __init__(
        self,
        height: float,
        n_cells: int,
        material: str = "graphite",
        azimuthal_angle: float = 0.0,
        zenith_angle: float = 0.0,
        cross_section: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._height = height
        self._nCells = n_cells
        self._material = material

        if cross_section is not None:
            self._crossSection = SolidCrossSection(shape=cross_section, **kwargs)
        else:
            self._crossSection = None

        assert azimuthal_angle == 0.0, "Cannot currently handle non-zero azimuthal angles"
        assert zenith_angle == 0.0, "Cannot currently handle non-zero zenith angles"
        self._azimuthal_angle = azimuthal_angle
        self._zenith_angle = zenith_angle

    @property
    def volume(self) -> float:
        return self.crossSection.area * self.height

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, height: float):
        self._height = height

    @property
    def nCells(self) -> int:
        return self._nCells

    @nCells.setter
    def nCells(self, n_cells: int):
        self._nCells = n_cells

    @property
    def material(self) -> str:
        return self._material

    @material.setter
    def material(self, material: str):
        self._material = material

    @property
    def azimuthalAngle(self) -> float:
        return self._azimuthal_angle

    @azimuthalAngle.setter
    def azimuthalAngle(self, azimuthal_angle: float):
        assert azimuthal_angle == 0.0, "Cannot currently handle non-zero azimuthal angles"
        self._azimuthal_angle = azimuthal_angle

    @property
    def zenithAngle(self) -> float:
        return self._zenith_angle

    @zenithAngle.setter
    def zenithAngle(self, zenith_angle: float):
        assert zenith_angle == 0.0, "Cannot currently handle non-zero zenith angles"
        self._zenith_angle = zenith_angle

    @property
    def crossSection(self) -> Shapes.CrossSection:
        return self._crossSection

    @crossSection.setter
    def crossSection(self, cross_section: SolidCrossSection):
        assert self._crossSection is None, "Can only add a cross-section object, not alter one."
        self._crossSection = deepcopy(cross_section)

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Private method for converting units of the component's internal attribute

        This method is especially useful for converting components to the expected units
        of the application in which they will be used.

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        self.height *= uc.lengthConversion
        if self.crossSection is not None:
            self.crossSection._convertUnits(uc)

    def baseComponents(self) -> List:
        """
        Returns the base component (itself)
        """
        return [self]

    @staticmethod
    def factory(input_dict: Dict):
        """
        Taking in a dictionary of component inputs, this factory method created each component.

        Parameters
        ----------
        input_dict : Dict
            Dictionary of component definitions
        """

        def buildComponent(parameters):
            return Component(**parameters)

        def buildSerialComponent(sub_components_definitions, order):
            sub_components = {name: buildComponent(sub_params)
                              for name, sub_params in sub_components_definitions.items()}
            return SerialComponent(sub_components, order)

        assert type(input_dict) == dict, f"Unknown input type: {type(input_dict)}"

        components = {}
        for comp_type, comp_inputs in input_dict.items():
            assert comp_type in solid_component_list, f"Error: unknown component type {comp_type}"

            for unique_name, parameters in comp_inputs.items():
                if comp_type == "component":
                    components[unique_name] = buildComponent(parameters)
                elif comp_type == "serial_component":
                    components[unique_name] = buildSerialComponent(parameters["component"], parameters["order"])
                else:
                    raise Exception(f"No 'build' method for component type '{comp_type}'")


        return components

solid_component_list["component"] = Component


class SerialComponent:
    """
    A collection of components which are formed in a serial manner, following the
    path of fluid flow. This can allow for the construction of a much longer solid
    component, with varying size, materials, etc.

    Parameters
    ----------
    components : Dict[str, SolidComponent]
        list of solid components used in the serial component
    order : List[str]
        order of the components in the full, serial component

    Attributes
    ----------
    componentOrder : List[str]
        order of components
    height : float
        total height of the serial component
    channel : Channel
        channel object built into the component
    """

    def __init__(self, components: Dict[str, Component], order: List[str]) -> None:
        self._components = deepcopy(components)
        self._order = order

    @property
    def components(self) -> Dict[str, Component]:
        return self._components

    @property
    def order(self) -> List[str]:
        return self._order

    @property
    def orderedComponents(self) -> List[Component]:
        return [self.components[key] for key in self.order]

    @property
    def volume(self) -> float:
        return sum(component.volume for component in self.orderedComponents)

    @property
    def height(self) -> float:
        return sum(component.height for component in self.orderedComponents)

    @property
    def nCells(self) -> int:
        return sum(component.nCells for component in self.orderedComponents)

    def baseComponents(self) -> List[Component]:
        """
        Method for retrieving the base components (components that are not Component collections)
        of a component collection
        """
        base_components = []
        for component in self.components.values():
            base_components.extend(component.baseComponents())
        return base_components

    def getComponentInletAndOutlets(self) -> List[Tuple[float, float]]:
        """
        Method for retrieving the inlet and outlet heights for each ordered component

        Returns
        -------
        inlets_and_outlets : List[Tuple[float, float]]
        """
        inlets_and_outlets = []
        inlet = 0.0
        for comp in self.orderedComponents:
            inlet_and_outlet = (inlet, inlet + comp.height)
            inlets_and_outlets.append(inlet_and_outlet)
            inlet += comp.height
        return inlets_and_outlets

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Private method for converting units of the component's internal attribute

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        for component in self.components.values():
            component._convertUnits(uc)


solid_component_list["serial_component"] = SerialComponent


class ParallelComponent:
    """
    A collection of components formed in parallel, with components being added
    together perpendicular to the path of fluid flow.

    Parameters
    ----------
    components : Dict[str, SolidComponent]
        dict containing the solid component
    component_map: List[List[str]]
        a map showing where the component lie relative to each other in an XY-cross section

    Attributes
    ----------
    components : Dict[str, SolidComponent]
        dict of all the components used in the collection
    componentMap : List[List[str]]
        a map showing where the component lie relative to each other in an XY-cross section
    volume : float
        total volume of the components in the collection
    nCells : int
        number of total cells in the collection
    """

    def __init__(
        self,
        components: Dict[str, Component],
        component_map: List[List[str]],
    ) -> None:

        self._components = deepcopy(components)
        self._componentMap = component_map

    @property
    def components(self) -> Dict[str, Component]:
        return self._components

    @property
    def componentMap(self) -> List[List[str]]:
        return self._componentMap

    @property
    def volume(self) -> float:
        return sum(sum(self.components[name].volume for name in row) for row in self.componentMap)

    @property
    def nCells(self) -> int:
        return sum(sum(self.components[name].nCells for name in row) for row in self.componentMap)

    def baseComponents(self) -> List[Component]:
        """
        Method for retrieving the base components (components that are not Component collections)
        of a component collection
        """
        base_components = []
        for component in self.components.values():
            base_components.extend(component.baseComponents())
        return base_components

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Private method for converting units of the component's internal attribute

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        for component in self.components.values():
            component._convertUnits(uc)


solid_component_list["parallel_component"] = ParallelComponent


class Core(ParallelComponent):
    """
    A collection of parallel, solid components which form a single-block core.

    In comparison with the more general "ParallelComponent" parent class, the
    "Core" child class should be stricter in what it allows as inputs. What this
    class should test for are:
        - All components are of proper types (either children of Component or a
            serial component)
        - Geometric consistency (all components are the same height, number of
            axial cells are the same)
        -

    Parameters
    ----------
    components : Dict[str, SolidComponent]
        dict containing the solid component
    component_map: List[List[str]]
        a map showing where the component lie relative to each other in an XY-cross section

    Attributes
    ----------
    components : Dict[str, SolidComponent]
        dict of all the components used in the collection
    componentMap : List[List[str]]
        a map showing where the component lie relative to each other in an XY-cross section
    volume : float
        total volume of the components in the collection
    nCells : int
        number of total cells in the collection
    coreHeight : double
        height of the core (z-direction)
    nAxialCells : int
        number of axial cells in the core
    """

    def __init__(
        self,
        components: Dict[str, Component],
        component_map: List[List[str]],
    ) -> None:

        # Parameters
        self._coreHeight = 0.0
        self._nAxialCells = 0

        # Checks
        self._componentTypeCheck(components)
        self._geometryCheck(components, component_map)

        super().__init__(components, component_map)

    @property
    def coreHeight(self) -> float:
        return self._coreHeight

    @coreHeight.setter
    def coreHeight(self, height):
        self._coreHeight = height

    @property
    def nAxialCells(self) -> int:
        print("Warning: 'nAxialCells' not fully implemented, will return '0'")
        return self._nAxialCells

    @nAxialCells.setter
    def nAxialCells(self, n_axial_cells):
        self._nAxialCells = n_axial_cells

    def _componentTypeCheck(self, components) -> None:
        """
        This method should check that all component types are acceptable.

        Core only accepts components which are either a "Component" or "SerialComponent" type.
        Also ensures that all base components are "Component"-types.

        Parameters
        ----------
        components : Dict[str, SolidComponent]
            dict containing the solid component
        """
        for comp_name, comp in components.items():
            assert isinstance(comp, (Component, SerialComponent)), "Incorrect component type (" + comp_name + ")"
            assert all(isinstance(comp_i, Component) for comp_i in comp.baseComponents()), (
                "Incorrect base component type (" + comp_name + ")"
            )

    def _geometryCheck(self, components, component_map) -> None:
        """
        This method should check for internal consistency of the input components
        geometry

        Checks:
            1. heights of components are all the same
            2. All base components are of 'rectangular' shape
            3. Lengths and widths of component cross sections are all
                the same

        Parameters
        ----------
        components : Dict[str, SolidComponent]
            dict containing the solid component
        component_map: List[List[str]]
            a map showing where the component lie relative to each other in an XY-cross section
        """

        # Reference values
        first_comp_name = component_map[0][0]
        first_component = deepcopy(components[first_comp_name])
        assert isinstance(first_component.baseComponents()[0].crossSection.shape, Shapes.Rectangle)
        reference_height = first_component.height
        reference_length = first_component.baseComponents()[0].crossSection.shape.height
        reference_width = first_component.baseComponents()[0].crossSection.shape.width

        # Make checks
        err = lambda comp_name, comparison_type : (
            f"Incorrect component {comparison_type} when compared to " +
            f"the reference component, {first_comp_name}. Error occurred " +
            f"in component '{comp_name}'"
        )
        for comp_name, comp in components.items():
            assert comp.height == reference_height, err(comp_name, "height (z)")
            for base_comp in comp.baseComponents():
                assert isinstance(base_comp.crossSection.shape, Shapes.Rectangle), (
                    "can only use rectangular or square cross sections in 'Core'"
                )
                assert base_comp.crossSection.shape.height == reference_length, err(comp_name, "length (x)")
                assert base_comp.crossSection.shape.width == reference_width, err(comp_name, "width (y)")


        self.coreHeight = reference_height  # Set core height to this reference height

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Private method for converting units of the component's internal attribute

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        super()._convertUnits(uc)
        self.coreHeight *= uc.lengthConversion


solid_component_list["core"] = Core
