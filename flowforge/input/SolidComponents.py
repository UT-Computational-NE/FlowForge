from typing import List, Dict, Tuple
import copy
import numpy as np

# from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.SolidCrossSections import CrossSection, cross_section_types

# pragma pylint: disable=protected-access

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""
solid_component_list = {}

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
    cross_section : str
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

    def __init__(self,
                 height: float,
                 n_cells: int,
                 material: str,
                 azimuthal_angle: float = 0.0,
                 zenith_angle: float = 0.0,
                 cross_section: str = None,
                 **kwargs
                 ) -> None:
        self._height = height
        self._n_cells = n_cells
        self._material = material

        if cross_section is not None:
            self._cross_section = self._buildCrossSection(cross_section, kwargs)
        else:
            self._cross_section = None

        # TODO: add azimuthal and zenith angle functionality
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
        return self._n_cells

    @nCells.setter
    def nCells(self, n_cells: int):
        self._n_cells = n_cells

    @property
    def material(self) -> str:
        return self._material

    @material.setter
    def material(self, material: str):
        self._material = material

    @property
    def azimuthalAngle(self) -> float:
        assert self.azimuthalAngle == 0.0, "Cannot currently handle non-zero azimuthal angles"
        return self._azimuthal_angle

    @azimuthalAngle.setter
    def azimuthalAngle(self, azimuthal_angle: float):
        self._azimuthal_angle = azimuthal_angle

    @property
    def zenithAngle(self) -> float:
        assert self.zenithAngle == 0.0, "Cannot currently handle non-zero zenith angles"
        return self._zenith_angle

    @zenithAngle.setter
    def zenithAngle(self, zenith_angle: float):
        self._zenith_angle = zenith_angle

    @property
    def crossSection(self) -> CrossSection:
        return self._cross_section

    @crossSection.setter
    def crossSection(self, cross_section: CrossSection):
        self._cross_section = cross_section


    def _buildCrossSection(cross_section_type: str, **kwargs) -> CrossSection:
        """
        Given a cross section type and the correct corresponding inputs (via
        kwargs), this method builds the cross section object used to compute
        the cross sectional area, as well as the volume

        Parameters
        ----------
        cross_section_type : str
            name of the desired cross section

        Returns
        -------
        cross_section : SolidCrossSections.CrossSection
            Cross section built using the inputs from kwargs
        """
        cross_section_obj = cross_section_types[cross_section_type]
        assert all(inp in kwargs.keys() for inp in cross_section_obj.inputs)
        cross_section = cross_section_obj(
            **{k: v for k, v in kwargs.items() if k in cross_section_obj.inputs}
        )
        return cross_section

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
        self.crossSection._convertUnits(uc)

    def baseComponents(self):
        """
        Method for retrieving the base components of a component.
        For components that are not collections, this will be itself
        """
        return [self]

    @staticmethod
    def factory(input_dict: Dict):
        """
        Factory for building a collection of components

        Parameters
        ----------
        indict : Dict
            The input dictionary specifying the components to be instantiated.  This dictionary can be comprised
            of two forms of inputs:

            1.) A Name-Component pair
                (Dict[str, Component])
            2.) A dictionary of component types, each type holding a dictionary of name-parameter_set pairs, with the
                name being the unique component's name, and the parameter_set another dictionary with key's corresponding
                to the __init__ signature of the associated component type
                (Dict[str, Dict[str, Dict[str, float]]])

        Returns
        -------
        components : Dict[str, Component]
            The collection of components built
            (key: Component name, value: Component object)
        """
        components = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                comp_type = key
                comps = value
                if comp_type in solid_component_list:
                    for name, parameters in comps.items():
                        components[name] = solid_component_list[comp_type](**parameters)
                else:
                    raise TypeError("Unknown component type: " + comp_type)
            elif isinstance(value, Component):
                name = key
                comp = value
                components[name] = comp
            else:
                raise TypeError(f"Unknown input dictionary: {key:s} type: {str(type(value)):s}")

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
        components = Component.factory(components)
        self._components = components
        self._order = order
        self._channel = None
        self._pipe = None
        self._pipe_information = None

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
        return sum(component.volume for component in self.baseComponents)

    @property
    def height(self) -> float:
        return sum(component.height for component in self.components.values())

    @property
    def nCells(self) -> int:
        return sum(component.nCells for component in self.baseComponents)

    def baseComponents(self) -> List[Component]:
        """
        Method for retrieving the base components (components that are not Component collections)
        of a component collection
        """
        base_components = []
        for component in self.components.values():
            base_components.extend(component.baseComponents)
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

        components = Component.factory(components)
        self._components = components
        self._component_map = component_map

    @property
    def components(self) -> Dict[str, Component]:
        return self._components

    @property
    def componentMap(self) -> List[List[str]]:
        return self._component_map

    @property
    def volume(self) -> float:
        return sum(component.volume for component in self.baseComponents)

    @property
    def nCells(self) -> int:
        return sum(component.nCells for component in self.baseComponents)

    def baseComponents(self) -> List[Component]:
        """
        Method for retrieving the base components (components that are not Component collections)
        of a component collection
        """
        base_components = []
        for component in self.components.values():
            base_components.extend(component.baseComponents)
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
    def __init__(self,
                 components: Dict[str, Component],
                 component_map: List[List[str]],
                 ) -> None:

        # Parameters
        self._core_height = 0.0
        self._n_axial_cells = 0

        # Checks
        self._componentTypeCheck(components)
        self.__geometryCheck(components, component_map)

        super().__init__(components, component_map)

    @property
    def coreHeight(self) -> float:
        return self._core_height

    @coreHeight.setter
    def coreHeight(self, height):
        self._core_height = height

    @property
    def nAxialCells(self) -> int:
        return self._n_axial_cells

    @nAxialCells.setter
    def nAxialCells(self, n_axial_cells):
        self._n_axial_cells = n_axial_cells

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
            assert isinstance(comp, Component) or isinstance(comp, SerialComponent), "Incorrect component type (" + comp_name + ")"
            assert all(isinstance(comp_i, Component) for comp_i in comp.baseComponents()), "Incorrect base component type (" + comp_name + ")"

    def _geometryCheck(self, components, component_map) -> None:
        """
        This method should check for internal consistency of the input components
        geometry

        Checks:
            1. heights of components are all the same
            2. number of axial cells is the same

        Parameters
        ----------
        components : Dict[str, SolidComponent]
            dict containing the solid component
        component_map: List[List[str]]
            a map showing where the component lie relative to each other in an XY-cross section
        """

        # Reference values
        first_comp_name = component_map[0][0]
        reference_height = copy.deepcopy(components[first_comp_name].height)
        reference_axial_cells = copy.deepcopy(components[first_comp_name].nCells)

        # Make checks
        for comp_name, comp in component_map.items():
            assert comp.height == reference_height, "Incorrect component height (" + comp_name + ", " + str(comp.height) + ")"
            assert comp.nCells == reference_axial_cells, "Incorrect component cell number (" + comp_name + ", " + str(comp.nCells) + ")"

        self.coreHeight = reference_height # Set core height to this reference height
        self.nAxialCells = reference_axial_cells # Set the number of axial cells to this reference value

solid_component_list["core"] = Core
