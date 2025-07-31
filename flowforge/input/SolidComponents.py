from typing import List, Dict, Tuple, Generator, Optional, Literal, TypeAlias
import abc
import numpy as np
import inspect

from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter

from flowforge.input.Components import cross_section_classes, cross_section_param_lists
from flowforge.input.Components import Core, CartCore
from flowforge.input.Components import Pipe

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

    @property
    def hasChannel(self):
        return False

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
    def _summary(self):
        return NotImplementedError

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
    def _carveChannelIntoBlock(component, pipe):
        solid_to_channeled_solid = {Cuboid: CuboidWithChannel}
        # Extract input parameters from non-channeled object
        comp_parameters  = inspect.signature(component.__class__.__init__).parameters
        input_parameters = {name: getattr(component, name)
                            for name in comp_parameters
                            if name != 'self'}
        # Create new channeled objects
        channeled_component = solid_to_channeled_solid[type(component)](
            **input_parameters,
            pipe_flow_area=pipe.flowArea,
            pipe_hydraulic_diameter=pipe.hydraulicDiameter
        )
        return channeled_component

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

    def printSummary(self, verbose=True):
        summary_dict = self._summary()

        if verbose == False:
            return summary_dict

        print(f"Class: {summary_dict['BASE']['Class']}")
        for top_key, sub_dict in summary_dict.items():
            if top_key == "BASE": continue
            print(f"  ({top_key})")
            for key, value in sub_dict.items():
                print(f"    {key}: {value}")
        print("\n")

        return summary_dict


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
    nCells        : number of cells in the cuboid
    solid_material : name of the type of solid used in the component
    """
    def __init__(self,
                 length         : float,
                 width          : float,
                 height         : float,
                 nCells        : int = 1,
                 solidMaterial : str = "graphite"
                 ) -> None:
        # Basic Parameters
        self._length  = length
        self._width   = width
        self._height  = height
        self._n_cells = nCells
        self._solid_material = solidMaterial
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

    def _summary(self):
        L, W, H = self.length, self.width, self.height

        summary_dict = {
            "BASE": {"Class": "Cuboid"},
            "Solid":
                {"Geometry (L x W x H)": (L, W, H),
                 "Volume": L*W*H,
                 "Material": self.solidMaterial,
                 "N-Cells": self.nCells},
            "General":
                {"Total Volume": self.volume,
                 "Height": H}
        }

        return summary_dict

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
                 nCells                  : int = 1,
                 solid_material          : str = "graphite",
                 pipe_cross_section_type : str = "circular",
                 pipe_flow_area          : float = None,
                 pipe_hydraulic_diameter : float = None,
                 **kwargs
                 ) -> None:
        super().__init__(length, width, height, nCells, solid_material)
        # Pipe cross-sectional data
        if (pipe_flow_area is None) and (pipe_hydraulic_diameter is None):
            self._cross_section = self._getChannelCrossSectionData(pipe_cross_section_type, **kwargs)
            self._fluid_area         = self._cross_section.flow_area
            self._wetted_perimeter   = self._cross_section.wetted_perimeter
            self._hydraulic_diameter = self._cross_section.hydraulic_diameter
        else:
            assert (pipe_flow_area is not None) and (pipe_hydraulic_diameter is not None)
            self._cross_section = None
            self._fluid_area         = pipe_flow_area
            self._hydraulic_diameter = pipe_hydraulic_diameter
            self._wetted_perimeter   = 4.0 * pipe_flow_area / pipe_hydraulic_diameter
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

    @property
    def hasChannel(self):
        return True

    def _convertUnits(self, uc: UnitConverter) -> None:
        super()._convertUnits(uc)
        self._fluid_area *= uc.areaConversion
        self._wetted_perimeter *= uc.lengthConversion
        self._hydraulic_diameter *= uc.lengthConversion

    def _summary(self):
        summary_dict = super()._summary()
        summary_dict["BASE"] = {"Class": "Cuboid With Channel"}
        summary_dict["Channel"] = {
            "Flow Area": self.fluidCrossSectionalArea,
            "Volume": self.fluidCrossSectionalArea * self.height,
            "Wetted Perimeter": self.wettedPerimeter,
            "Hydraulic Diameter": self.hydraulicDiameter
        }
        summary_dict["General"]["Total Volume"] = self.volume

        return summary_dict

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
        for component in self.components.values():
            base_components.extend(component.baseComponents)
        return base_components

    def getNodeGenerator(self) -> Generator[SolidComponent, None, None]:
        yield from [component.getNodeGenerator() for component in self.components.values()]

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

class SolidEncasedPipe(SerialSolidComponents):
    """
    A single pipe encased within a solid structure.

    Parameters
    ----------
    """
    def __init__(self,
                 pipe : Pipe, # TODO: make this a Dict[str, Pipe] w/ an order
                 components: Dict[str, SolidComponent],
                 order: List[str],
                 **kwargs
                 ) -> None:
        input_components = self._addChannelToSolidComponent(components, pipe)
        super().__init__(input_components, order, **kwargs)

    def _addChannelToSolidComponent(self,
                                    components: Dict[str, SolidComponent],
                                    pipe: Pipe
                                    ) -> Dict[str, SolidComponent]:
        channeled_components = {}
        channeled_solid_types = [CuboidWithChannel]
        for name, comp in components.items():
            if any(isinstance(comp, channel_comp) for channel_comp in channeled_solid_types):
                channeled_components[name] = comp
            else:
                channeled_components[name] = self._carveChannelIntoBlock(comp, pipe)
        return channeled_components

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
    (Fluid-Side)
    fluid_core : "Core" type from the fluid components. This input is required and many of the
                core specifics will come from this object
    (Solid-Side)
    solid_components : a dict of SolidComponent object that could be used as blocks in the core
    solid_component_map : maps the components in "solid_components" to specific spots in space
    fluid_pipe_map : A 1:1 map with "solid_component_map" that specifies where in "solid_component_map"
                the pipes are, relative to the solid blocks.
    solid_material : if "solid_component_map" is not specified, this input is required to create a uniform
                core with each block having the same solid properties
    global_offset : A vector of the <x, y, z> offsets between the center of the fluid and solid cores,
                    using the center of the fluid core as the reference origin (0,0,0). Note that this
                    is different than an offset for a CuboidWithChannel object, as an offset defined
                    within the CuboidWithChannel object is the channels center relative to the cuboids
                    center. For the global-offset, this is the offset of the fluid and solid *cores*, and
                    each offset set in the CuboidWithChannel object should be again adjusted by this global
                    offset.
    """
    def __init__(self,
                 # Input Fluid Component
                 fluid_core           : CartCore,
                 # Optional Solid Specifications
                 solid_components     : Optional[Dict[str, SolidComponent]] = None,
                 solid_component_map  : Optional[List[List[str]]]           = None,
                 fluid_pipe_map       : Optional[List[List[str]]]           = None,
                 solid_material       : Optional[str]                       = None,
                 core_height          : Optional[float]                     = None,
                 solid_component_type : Optional[str]                       = "cuboid",
                 n_axial_cells        : Optional[int]                       = None,
                 global_offset        : Optional[Tuple[int, int, int]]      = (0,0,0)
                 ) -> None:
        # User input fluid core
        self._fluid_core = fluid_core

        # Inputs used to create a uniform solid core
        self._solid_material = solid_material
        self._core_height = core_height
        self._solid_component_type = solid_component_type
        self._n_axial_cells = n_axial_cells

        # Solid-Fluid core offset
        self._global_offset = global_offset
        if global_offset != (0,0,0):
            # TODO: implement offset functionality
            raise Exception("Need to implement fluid/solid core offset functionality")

        # Validating the solid and fluid maps
        self._fluid_pipe_map, self._solid_component_map = self._formatSolidAndFluidMaps(
            fluid_pipe_map, solid_component_map)

        # Reformating the solid component list and map
        solid_components = self._checkAndReformatSolidComponents(
            self._solid_component_map,
            self._fluid_pipe_map,
            solid_components)

        self._solid_components = solid_components

        super().__init__(solid_components, solid_component_map)

    @property
    def fluidCore(self):
        return self._fluid_core

    @property
    def coreSolidMaterial(self):
        return self._solid_material

    @property
    def coreHeight(self):
        return self._core_height

    @property
    def solidComponentType(self):
        return self._solid_component_type

    @property
    def nAxialCells(self):
        return self._n_axial_cells

    @property
    def offsets(self):
        return self._offset

    @property
    def fluidPipeMap(self):
        return self._fluid_pipe_map

    # def _getSolidAndFluidMaps(self,
    #                           fluid_map : Optional[List[List[str]]] = None,
    #                           solid_map : Optional[List[List[str]]] = None):
    #     fluid_core_map = self.fluidCore.channelMap
    #     # If neither a fluid or solid map are input, derive a solid map from the fluid-core map
    #     if (fluid_map is None) and (solid_map is None):
    #         solid_map = [["_" for _ in range(len(f))] for f in fluid_core_map]
    #         return fluid_core_map, solid_map

    #     # If there is no input fluid map, ensure that the original input map is sufficient
    #     if (fluid_map is None) and (solid_map is not None):
    #         assert len(fluid_core_map) == len(solid_map)
    #         assert all(len(f) == len(s) for f, s in zip(fluid_core_map, solid_map))
    #         return fluid_core_map, solid_map

    #     # If there is no input solid map, create it based off fluid_map
    #     if (fluid_map is not None) and (solid_map is None):
    #         solid_map = [["_" for _ in range(len(f))] for f in fluid_map]

    #     # Ensure that user-input maps for the fluid and solid components are the same size
    #     assert len(fluid_map) == len(solid_map)
    #     assert all(len(f) == len(s) for f, s in zip(fluid_map, solid_map))

    #     # Check if the input map is the same as the one defined by the FluidCore
    #     if fluid_core_map == fluid_map:
    #         return fluid_map, solid_map

    #     # TODO: Allow for the user to have the input map NOT match the one from FluidCore.
    #     #       `-> This will allow for more complex solid-mesh geometries and more refined
    #     #           meshing
    #     raise NotImplementedError

    def _formatSolidAndFluidMaps(self,
                                 fluidMap : Optional[List[List[str]]] = None,
                                 solidMap : Optional[List[List[str]]] = None):
        """
        Given a fluid and solid map, this function checks for potential input
            scenarios and formats the maps based on these cases:

        1. (No solidMap) & (No fluidMap)
            - Assume fluidMap == fluidCore.channelMap
            - Copy this map to solidMap, with NULL components ('_')
              at each map location

        2. (solidMap) & (No fluidMap)
            - Assume fluidMap == fluidCore.channelMap
            - Check that shape(solidMap) == shape(fluidMap)

        3. (No solidMap) & (fluidMap)
            3A. (fluidMap == fluidCore.channelMap)
                - create a map of equal size and fill it with NULL
                  components ('_'), telling later methods to build
                  uniform components at these blocks
            3B. (fluidMap != fluidCore.channelMap)
                - Will need to add functionality to handle this scenario
                - By being unequal, we are telling the code to refine the
                  solid mesh. This refinement is not yet implemented

        4. (solidMap) & (fluidMap)
            4A. (fluidMap == fluidCore.channelMap)
                - Checks that maps are of the same size
                - Return both maps
            4B. (fluidMap != fluidCore.channelMap)
                - Will need to add functionality to handle this scenario
                - By being unequal, we are telling the code to refine the
                  solid mesh. This refinement is not yet implemented
        """
        # Loads in the map from the fluidCore component
        fluidCoreMap = self.fluidCore.channelMap
        # Checks to see if the two fluid maps are the same shape or not
        if (fluidMap is not None):
            fluidMapsAreTheSameShape = all(
                len(fluidMap) == len(fluidCoreMap),
                all(len(f1) == len(f2) for f1, f2 in zip(fluidMap, fluidCoreMap))
            )
        else: fluidMapsAreTheSameShape = False
        # If they are the same shape, the should be the same
        if fluidMapsAreTheSameShape:
            assert fluidMap == fluidCoreMap

        # - - - - - - - - Cases - - - - - - - - #
        ## Case 1
        if (solidMap is None) and (fluidMap is None):
            # Creates 1:1 solid map and adds NULL components
            solidMap = [["_" for _ in range(len(f))] for f in fluidCoreMap]
            return fluidCoreMap, solidMap

        ## Case 2
        elif (solidMap is not None) and (fluidMap is None):
            # Check consistent sizes
            assert len(fluidCoreMap) == len(solidMap)
            assert all(len(f) == len(s) for f, s in zip(fluidCoreMap, solidMap))
            # Sets fluidMap to fluidCoreMap
            return fluidCoreMap, solidMap

        ## Case 3
        elif (solidMap is None) and (fluidMap is not None):
            # Case 3A
            if fluidMapsAreTheSameShape:
                solidMap = [["_" for _ in range(len(f))] for f in fluidMap]
                return fluidMap, solidMap
            # Case 3B
            else:
                raise NotImplementedError

        ## Case 4
        else: # (solidMap is not None) and (fluidMap is not None)
            # Case 4A
            if fluidMapsAreTheSameShape:
                assert len(fluidCoreMap) == len(solidMap)
                assert all(len(f) == len(s) for f, s in zip(fluidCoreMap, solidMap))
                return fluidMap, solidMap
            # Case 4B
            else:
                raise NotImplementedError


    def _checkAndReformatSolidComponents(self, solidMap, fluidMap, componentList):
        """
        Using the reformatted solid and fluid maps, this method ensures that the
        component list is completed. If not, this method fills in any gaps and
        carves any channels into blocks which need them
        """
        if componentList is None:
            uniform_component_list = self._makeUniformSolidComponentList()
            return uniform_component_list

        fluidComponents = self.fluidCore.components
        # Iterates over all inputs into each map, extracting the ID of each
        #   given component in the map
        for solidRowIds, fluidRowIds in zip(solidMap, fluidMap):
            for solidCompId, fluidCompId in zip(solidRowIds, fluidRowIds):
                # Extracts the fluid component
                fluidComp = fluidComponents[fluidCompId]
                # If the solid component has been defined, this extracts it. If not
                # defined, a uniform component is built.
                if solidCompId in componentList:
                    solidComp = componentList[solidCompId]
                else:
                    solidComp = self._makeUniformComponent()
                if solidComp.hasChannel:
                    assert solidComp.fluidCrossSectionalArea == fluidComp.flowArea
                    assert solidComp.hydraulicDiameter == fluidComp.hydraulicDiameter
                else:
                    solidComp = self._carveChannelIntoBlock(solidComp, fluidComp)
                componentList[solidCompId] = solidComp

        return componentList



    # def _reformatSolidMap(self, solid_map, component_list):
    #     if component_list is None:
    #         uniform_component_list = self._makeUniformSolidComponentList()
    #         return self.fluidCore.componentMap, uniform_component_list

    #     print("solid_map: ", solid_map)
    #     print("component_list: ", component_list)
    #     for row in solid_map:
    #         for comp in row:
    #             if comp not in component_list:
    #                 component_list[comp] = self._makeUniformComponent()

    #     return solid_map, component_list

    def _makeUniformComponent(self):
        """
        Creates a single component, build using the foundational geometric inputs
        and basic input material
        """
        x_pitch   = self.fluidCore.xPitch
        y_pitch   = self.fluidCore.yPitch
        height    = self.coreHeight
        material  = self.coreSolidMaterial
        comp_type = self.solidComponentType
        n_cells   = self.nAxialCells

        assert x_pitch   is not None, "Must input an x_pitch for a uniform core"
        assert y_pitch   is not None, "Must input a y_pitch for a uniform core"
        assert height    is not None, "Must input a height for a uniform core"
        assert material  is not None, "Must input a solid material for a uniform core"
        assert comp_type is not None, "Must input a component-type for a uniform core"
        assert n_cells   is not None, "Must input a number of axial cells for a uniform core"

        component = solid_component_list[comp_type](
            length=x_pitch, width=y_pitch, height=height,
            nCells=n_cells, solidMaterial=material)

        return component

    def _makeUniformSolidComponentList(self):
        """
        In the case of which no solid-component list is input, this method
        creates uniform components (see above) for each point in the map,
        and then carves appropriate channels into them based on the pipes
        at those locations.
        If the case that there is no corresponding fluid component for a
        point in the solid map, the uniform component is kept without carving
        a channel
        """
        uniform_component = self._makeUniformComponent()
        parallel_fluid_components = self.fluidCore.components

        # Carves correct channels into the defined uniform component, based on the pipe-type
        #   going through it
        solid_component_list = {}
        for name, fluid_comp in parallel_fluid_components.items():
            solid_comp = self._carveChannelIntoBlock(uniform_component, fluid_comp)
            solid_component_list[name] = solid_comp

        # For any element in the fluid-map not associated with a defined pipe-type, a filled
        #   solid block is used
        for row in self.fluidPipeMap:
            for comp_name in row:
                if comp_name not in parallel_fluid_components:
                    solid_component_list[comp_name] = uniform_component.copy()

        return solid_component_list

    # def _carveProperChannels(self, solid_map, fluid_map, solid_components):
    #     for s_row, f_row in zip(solid_map, fluid_map):
    #         for s_comp, f_comp in zip(s_row, f_row):


    def _checkValidInputs(self, **inputs):
        return