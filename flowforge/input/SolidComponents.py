from typing import List, Dict, Tuple, Generator, Optional, Literal, TypeAlias
import abc
import numpy as np
from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter

from flowforge.input.Components import cross_section_classes, cross_section_param_lists
from flowforge.input.Components import Core

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""
solid_component_list = {}

class SolidComponent:
    """
    Base class for all solid components in the system

    Parameters
    ----------
    """
    def __init__(self) -> None:
        self.uc = None

    @property
    def volume(self) -> float:
        raise NotImplementedError

    @property
    def height(self) -> float:
        raise NotImplementedError

    @property
    def nCells(self) -> int:
        raise NotImplementedError

    @property
    def solidMaterial(self) -> str:
        raise NotImplementedError

    @property
    def baseComponents(self):
        """Method for retrieving the base components of a component.
        For components that are not collections, this will be itself"""
        return [self]

    def getNodeGenerator(self):
        """Generator for marching over the nodes (i.e. cells) of a component

        This method essentially allows one to march over the nodes of a component
        and be able to reference / use the component said node belongs to

        Yields
        ------
        Component
            The component associated with the node the generator is currently on (i.e. self)
        """
        for _ in range(self.nCells):
            yield self

    @abc.abstractmethod
    def _convertUnits(self, uc: UnitConverter) -> None:
        """Private method for converting units of the component's internal attribute

        This method is especially useful for converting components to the expected units
        of the application in which they will be used.

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        raise NotImplementedError

    @staticmethod
    def factory(input_dict: Dict):
        """Factory for building a collection of components

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
        Dict[str, Component]
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
            elif isinstance(value, SolidComponent):
                name = key
                comp = value
                components[name] = comp
            else:
                raise TypeError(f"Unknown input dictionary: {key:s} type: {str(type(value)):s}")

        return components

class HexagonalPrism(SolidComponent):
    """
    A hexagonal prism solid component

    Parameters
    ----------
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError

solid_component_list["hexagonal_prism"] = HexagonalPrism

class Cuboid(SolidComponent):
    """
    A solid cuboid component

    Parameters
    ----------
    length         : length of the cuboid (x-direction)
    width          : width of the cuboid (y-direction)
    height         : height of the cuboid (z-direction)
    n_cells        : number of cells in the cuboid
    solid_material : name of the type of solid used in the component
    """
    def __init__(self,
                 length         : float,
                 width          : float,
                 height         : float,
                 n_cells        : int = 1,
                 solid_material : str = "graphite"
                 ) -> None:
        # Basic Parameters
        self._length  = length
        self._width   = width
        self._height  = height
        self._n_cells = n_cells
        self._solid_material = solid_material
        super().__init__()
        # Derived Parameters
        self._volume = self._length * self._width * self._height

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = volume

    @property
    def nCells(self) -> int:
        return self._n_cells

    @property
    def solidMaterial(self):
        return self._solid_material

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._length *= uc.lengthConversion
        self._width  *= uc.lengthConversion
        self._height *= uc.lengthConversion
        self._volume *= uc.volumeConversion

solid_component_list["cuboid"] = Cuboid

class CuboidWithChannel(Cuboid):
    """
    A cuboid solid component with a channel running through it

    Parameters
    ----------
    length                  : length of the cuboid (x-direction)
    width                   :  width of the cuboid (y-direction)
    height                  : height of the cuboid (z-direction)
    n_cells                 : number of cells in the cuboid
    solid_material          : name of the type of solid used in the component
    pipe_cross_section_type : name of the type of cross-section to use for the pipe

    Optional, Pipe Cross-Section Parameters
    ----------
    (Circular)
        * R : pipe radius
    (Stadium)
        * R : Radius of the semi circular portion of the stadium channel
        * A : Length of the rectangular portion of the stadium channel
    (Square)
        * W : Width of the square pipe
    (Rectangular)
        * W : Width of the rectangular pipe
        * H : Height of the rectangular pipe
    """
    def __init__(self,
                 length                  : float,
                 width                   : float,
                 height                  : float,
                 n_cells                 : int = 1,
                 solid_material          : str = "graphite",
                 pipe_cross_section_type : str = "circular",
                 **kwargs
                 ) -> None:
        super().__init__(length, width, height, n_cells, solid_material)
        # Pipe cross-sectional data
        self._cross_section = self._getChannelCrossSectionData(pipe_cross_section_type, **kwargs)
        self._fluid_area = self._cross_section.flow_area
        self._wetted_perimeter = self._cross_section.wetted_perimeter
        self._hydraulic_diameter = self._cross_section.hydraulic_diameter
        # Adjusts block volume due to a carving out of the pipe from the solid
        self.volume = self.volume - (self.fluidCrossSectionalArea * self.height)

    def _getChannelCrossSectionData(self, pipe_cross_section_type, **kwargs):
        """
        Given a cross section type name and input keyword arguments, this method checks the
        validity of the inputs and, if valid, builds the corresponding cross-section object
        """
        # Check that the cross-section type is a valid one
        err = f"Must input a valid cross-section type {list(cross_section_classes.keys())}"
        assert pipe_cross_section_type in cross_section_classes, err

        # Gets the cross section object and valid parameters
        CX_object = cross_section_classes[pipe_cross_section_type]
        valid_parameters = cross_section_param_lists[pipe_cross_section_type]

        # Checks that the desired input parameters are in input
        err = f"Parameters must be input: {valid_parameters}"
        assert all(p in kwargs for p in valid_parameters), err

        # Builds and returns cross-section object
        CX = CX_object(**{k: v for k, v in kwargs.items() if k in valid_parameters})
        return CX

    @property
    def crossSection(self):
        return self._cross_section

    @property
    def fluidCrossSectionalArea(self):
        return self._fluid_area

    @property
    def wettedPerimeter(self):
        return self._wetted_perimeter

    @property
    def hydraulicDiameter(self):
        return self._hydraulic_diameter

    def _convertUnits(self, uc: UnitConverter) -> None:
        super()._convertUnits(uc)
        self._fluid_area *= uc.areaConversion
        self._wetted_perimeter *= uc.lengthConversion
        self._hydraulic_diameter *= uc.lengthConversion

solid_component_list["cuboid_with_channel"] = CuboidWithChannel

class SolidComponentCollection(SolidComponent):
    """
    An abstract class to manage multiple SolidComponents in a system

    Parameters
    ----------
    components : Dict[str, SolidComponent]
        Collection of already initialized components
    """
    def __init__(self,
                 components: Dict[str, SolidComponent]
                 ) -> None:
        super().__init__()
        self._components = components

    @property
    def components(self):
        return self._components

    @property
    def volume(self) -> float:
        return sum(component.volume for component in self.baseComponents)

    @property
    def nCells(self) -> int:
        return sum(component.nCells for component in self.baseComponents)

    @property
    def baseComponents(self) -> List[SolidComponent]:
        """Method for retrieving the base components (components that are not Component collections)
        of a component collection"""
        base_components = []
        for component in self.myComponents.values():
            base_components.extend(component.baseComponents)
        return base_components

    def getNodeGenerator(self) -> Generator[SolidComponent, None, None]:
        yield from [component.getNodeGenerator() for component in self._myComponents.values()]

    def _convertUnits(self, uc: UnitConverter) -> None:
        for component in self.components.values():
            component._convertUnits(uc)

class SerialSolidComponents(SolidComponentCollection):
    """
    A collection of components which are formed in a serial manner, following the
    path of fluid flow. This can allow for the construction of a much longer solid
    component, with varying size, materials, etc.

    Parameters
    ----------
    """
    def __init__(self,
                 components: Dict[str, SolidComponent],
                 order: List[str],
                 **kwargs
                 ) -> None:
        components = SolidComponent.factory(components)
        super().__init__(components)
        self._order = order

    @property
    def componentOrder(self):
        return self._order

    @property
    def height(self) -> float:
        return sum(component.height for component in self.components.values())

class ParallelSolidComponents(SolidComponentCollection):
    """
    A collection of components formed in parallel, with components being added
    together perpendicular to the path of fluid flow.

    Parameters
    ----------
    """
    def __init__(self,
                 components: Dict[str, SolidComponent],
                 component_map: List[List[str]],
                 ) -> None:
        components = SolidComponent.factory(components)
        super().__init__(components)
        self._component_map = component_map

    @property
    def componentMap(self):
        return self._component_map

class SolidCore(ParallelSolidComponents):
    """
    A collection of parallel, solid components, forming an MSR core

    Parameters
    ----------
    """
    def __init__(self,
                 # Input Fluid Component
                 fluid_core: Core,
                 # Optional Solid Specifications
                 solid_components: Optional[Dict[str, SolidComponent]] = None,
                 solid_component_map: Optional[List[List[str]]] = None,
                 fluid_pipe_map: Optional[List[List[str]]] = None,
                 ) -> None:
        super().__init__(components, component_map)
