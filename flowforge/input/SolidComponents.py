from typing import List, Dict, Tuple, Generator, Optional
import abc
import inspect
import copy

# from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter

from flowforge.input.Components import cross_section_classes, cross_section_param_lists, CrossSection
from flowforge.input.Components import Pipe, CartCore, SerialComponents
from flowforge.input.Components import Pipe

# pragma pylint: disable=protected-access

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""
solid_component_list = {}


class SolidComponent:
    """
    Base class for all solid components in the system

    Attributes
    ----------
    volume : float
        Volume of the component
    height : float
        Height of the component
    nCells : int
        Number of discretizations made for the component
    solidMaterial : str
        Name of the material used for the component
    baseComponents : List[SolidComponent]
        Used to collect base components in collections. For this base object,
        this just returns a list consisting of only itself
    hasChannel : bool
        Boolean denoting whether or not the component has a built-in channel
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
        """
        Method for retrieving the base components of a component.
        For components that are not collections, this will be itself
        """
        return [self]

    @property
    def hasChannel(self) -> bool:
        return False

    def getNodeGenerator(self):
        """
        Generator for marching over the nodes (i.e. cells) of a component

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
        return {}

    @abc.abstractmethod
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
            elif isinstance(value, SolidComponent):
                name = key
                comp = value
                components[name] = comp
            else:
                raise TypeError(f"Unknown input dictionary: {key:s} type: {str(type(value)):s}")

        return components

    def printSummary(self, verbose: bool = True):
        """
        Prints a detailed summary of the component. Returns the detailed dict if requested

        Parameters
        ----------
        verbose : bool
            Boolean to control if this method prints information

        Returns
        -------
        summary_dict : dict
            Dict containing details of the component
        """
        summary_dict = self._summary()

        if verbose is False:
            return summary_dict

        print(f"Class: {summary_dict['BASE']['Class']}")
        for top_key, sub_dict in summary_dict.items():
            if top_key == "BASE":
                continue
            print(f"  ({top_key})")
            for key, value in sub_dict.items():
                print(f"    {key}: {value}")
        print("\n")

        return summary_dict

class Cuboid(SolidComponent):
    """
    A cuboid solid component

    Parameters
    ----------
    length : float
        length of the cuboid (x-direction)
    width : float
        width of the cuboid (y-direction)
    height : float
        height of the cuboid (z-direction)
    n_cells : int
        number of cells in the cuboid
    solid_material : str
        name of the type of solid used in the component
    pipe_cross_section_type : str
        name of the type of cross-section to use for the pipe

    Optional, Pipe Cross-Section Parameters
    ----------
    (Circular)
        * R : float
            pipe radius
    (Stadium)
        * R : float
            Radius of the semi circular portion of the stadium channel
        * A : float
            Length of the rectangular portion of the stadium channel
    (Square)
        * W : float
            Width of the square pipe
    (Rectangular)
        * W : float
            Width of the rectangular pipe
        * H : float
            Height of the rectangular pipe

    Attributes
    ----------
    length : float
        x-length of the cuboid
    width: float
        y-length of the cuboid
    height: float
        z-length of the cuboid
    volume : float
        total volume of the cuboid. If a channel is added, the channels volume
        is subtracted.
    nCells : int
        Number of discretizations made for the component
    solidMaterial : str
        Name of the material used for the component
    channel : Channel
        Channel object defined for the component. Set as "None" if not channel is
        input
    hasChannel : bool
        Boolean denoting whether or not the component has a built-in channel
    """

    class Channel():
        """
        Channel object.

        User may input either a type of cross section and the appropriate key-word args, a
        fluid-component object (specifically, a Pipe object), or a CrossSection object, of
        which all the necessary information can be extracted.

        Parameters
        ----------
        height : float
            length of the channel
        cross_section_type : str
            name of the type of cross-section used for the channel (i.e. circular, stadium, etc.)
        fluid_component : Pipe
            Pipe object that the channel can base itself off of
        cross_section_object : CrossSection
            CrossSection object that the channel can derive its own geometric specifications from

        Attributes
        ----------
        height : float
            height of the channel
        volume : float
            volume of the channel
        crossSection : CrossSection
            CrossSection object for the channel
        fluidCrossSectionalArea : float
            cross-sectional area of the channel
        wettedPerimeter : float
            wetted perimeter of the channel
        hydraulicDiameter : float
            Dh (hydraulic diameter) of the channel
        """
        def __init__(self,
                     height               : float,
                     cross_section_type   : str = None,
                     fluid_component      : Pipe = None,
                     cross_section_object : CrossSection = None,
                     **kwargs):

            self._height = height

            assert (cross_section_type is not None) or (fluid_component is not None) or (cross_section_object is not None)
            if (cross_section_type is not None):
                assert (fluid_component is None) and (cross_section_object is None), "Can only handle one input type"
                self._cross_section = self._getChannelCrossSectionData(cross_section_type, **kwargs)
            elif (cross_section_object is not None):
                assert (fluid_component is None) and (cross_section_type is None), "Can only handle one input type"
                self._cross_section = cross_section_object
            else:
                assert (cross_section_object is None) and (cross_section_type is None), "Can only handle one input type"
                assert isinstance(fluid_component, Pipe), "For not, Cuboid can only handle Pipe-inputs"
                self._cross_section = fluid_component.crossSection

            self._fluid_area = self._cross_section.flow_area
            self._wetted_perimeter = self._cross_section.wetted_perimeter
            self._hydraulic_diameter = self._cross_section.hydraulic_diameter
            self._volume = self._fluid_area * self._height

        def _getChannelCrossSectionData(self, pipe_cross_section_type, **kwargs):
            """
            Given a cross section type name and input keyword arguments, this method checks the
            validity of the inputs and, if valid, builds the corresponding cross-section object

            Parameters
            ----------
            pipe_cross_section_type : str
                name of the type of cross-section used for the channel (i.e. circular, stadium, etc.)
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
        def height(self) -> float:
            return self._height

        @property
        def volume(self) -> float:
            return self._volume

        @property
        def crossSection(self) -> CrossSection:
            return self._cross_section

        @property
        def fluidCrossSectionalArea(self) -> float:
            return self._fluid_area

        @property
        def wettedPerimeter(self) -> float:
            return self._wetted_perimeter

        @property
        def hydraulicDiameter(self) -> float:
            return self._hydraulic_diameter

        def _convertUnits(self, uc: UnitConverter) -> None:
            """
            Private method for converting units of the component's internal attribute

            Parameters
            ----------
            uc : UnitConverter
                A unit converter which holds the 'from' units and 'to' units for the conversion
                and will ultimately provide the appropriate multipliers for unit conversion.
            """
            self._fluid_area *= uc.areaConversion
            self._wetted_perimeter *= uc.lengthConversion
            self._hydraulic_diameter *= uc.lengthConversion


    def __init__(self,
                 length,
                 width,
                 height,
                 n_cells,
                 solid_material = "graphite",
                 pipe_cross_section_type = None,
                 **kwargs):
        super().__init__()

        self._length = length
        self._width  = width
        self._height = height
        self._volume = length * width * height
        self._n_cells = n_cells
        self._solid_material = solid_material

        # If input, creates a channel within the pipe
        if (pipe_cross_section_type is not None):
            self._channel = self.Channel(height=self.height,
                                         cross_section_type=pipe_cross_section_type,
                                         **kwargs)
            self.volume -= self.channel.volume
            self._has_channel = True
        else:
            self._channel = None
            self._has_channel = False

        # Check validity of inputs
        self._checkDimensionValidity()
        if self.hasChannel:
            self._checkChannelValidity()

    def addChannel(self,
                   pipe_cross_section_type : str = None,
                   cross_section           : CrossSection = None,
                   fluid_component         : Pipe = None,
                   **kwargs) -> None:
        """
        Adds a channel to the cuboid object, using 1 of 3 input types:
            1) Type of cross section and appropriate keyword arguments
            2) A cross section object, which will contain all the information needed
                   to build the channel
            3) A fluid component that will have all the information needed to build
                   the channel

        Parameters
        ----------
        pipe_cross_section_type : str
            name of the type of cross-section used for the channel (i.e. circular, stadium, etc.)
        cross_section : CrossSection
            CrossSection object that the channel can derive its own geometric specifications from
        fluid_component : Pipe
            Pipe object that the channel can base itself off of
        """
        assert self.channel == None, "Cannot add a channel if one already exists"

        # Check that only *one* type of channel input is set
        assert (pipe_cross_section_type is not None) or (cross_section is not
                None) or (fluid_component is not None), "Need at least one input type"

        err = "Can only handle (1) channel input type"
        if (pipe_cross_section_type is not None):
            assert (fluid_component is None) and (cross_section is None), err
            self.channel = self._addChannelViaCrossSectionTypes(pipe_cross_section_type, **kwargs)
        elif (cross_section is not None):
            assert (pipe_cross_section_type is None) and (fluid_component is None), err
            self.channel = self._addChannelViaCrossSectionObject(cross_section)
        else:
            assert (pipe_cross_section_type is None) and (cross_section is None), err
            self.channel = self._addChannelViaFluidComponent(fluid_component)

        # Adjusts attributes and checks channel validity
        self.volume -= self.channel.volume
        self._has_channel = True
        self._checkChannelValidity()

    def _addChannelViaCrossSectionTypes(self, pipe_cross_section_type, **kwargs) -> Channel:
        """
        Adds a channel to the cuboid, using input type (1) from "addChannel()" method

        Parameters
        ----------
        pipe_cross_section_type : str
            name of the type of cross-section used for the channel (i.e. circular, stadium, etc.)

        Returns
        -------
        channel : Channel
            channel object
        """
        channel = self.Channel(height=self.height,
                               cross_section_type=pipe_cross_section_type,
                               **kwargs)
        return channel

    def _addChannelViaCrossSectionObject(self, cross_section):
        """
        Adds a channel to the cuboid, using input type (2) from "addChannel()" method

        Parameters
        ----------
        cross_section : CrossSection
            CrossSection object that the channel can derive its own geometric specifications from

        Returns
        -------
        channel : Channel
            channel object
        """
        channel = self.Channel(height=self.height,
                               cross_section_object=cross_section)
        return channel

    def _addChannelViaFluidComponent(self, fluid_component):
        """
        Adds a channel to the cuboid, using input type (3) from "addChannel()" method

        Parameters
        ----------
        fluid_component : Pipe
            Pipe object that the channel can base itself off of

        Returns
        -------
        channel : Channel
            channel object
        """
        assert isinstance(fluid_component, Pipe), "For now, only 'Pipe' types are accepted"
        channel = self.Channel(height=self.height,
                               fluid_component=fluid_component)
        return channel

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def volume(self):
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

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, channel):
        self._channel = channel

    @property
    def hasChannel(self):
        return self._has_channel

    def _checkDimensionValidity(self) -> None:
        """
        Checks that the dimensions of the cuboid are valid
        """
        # Check positive, non-zero dimensions
        assert self.length > 0
        assert self.width > 0
        assert self.height > 0
        assert self.volume > 0
        assert self.nCells > 0

    def _checkChannelValidity(self) -> None:
        """
        Checks that the dimensions of the channel are valid
        """
        assert self.hasChannel == True
        assert self.channel is not None
        # Check positive, non-zero dimensions
        assert self.channel.fluidCrossSectionalArea > 0.0
        assert self.channel.hydraulicDiameter > 0.0
        assert self.channel.wettedPerimeter > 0.0
        assert self.channel.volume > 0.0
        # Assert that the flow area is less than the area of the cuboid
        assert self.channel.fluidCrossSectionalArea < (self.length * self.width)
        assert self.channel.volume < (self.length * self.width * self.height)


    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Private method for converting units of the component's internal attribute

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        self._length *= uc.lengthConversion
        self._width  *= uc.lengthConversion
        self._height *= uc.lengthConversion
        self._volume *= uc.volumeConversion
        if self.hasChannel:
            self.channel._convertUnits(uc)

    def _summary(self):
        """
        Summary of the cuboid, as well as the channel if appropriate

        Returns
        -------
        summary_dict : dict
            dict containing information on the object
        """
        L, W, H = self.length, self.width, self.height

        summary_dict = {
            "BASE": {"Class": "Cuboid"},
            "Solid": {
                "Geometry (L x W x H)": (L, W, H),
                "Volume": L * W * H,
                "Material": self.solidMaterial,
                "N-Cells": self.nCells}
        }

        if self.hasChannel:
            summary_dict["Channel"] = {
                "Flow Area": self.channel.fluidCrossSectionalArea,
                "Volume": self.channel.fluidCrossSectionalArea * self.height,
                "Wetted Perimeter": self.channel.wettedPerimeter,
                "Hydraulic Diameter": self.channel.hydraulicDiameter,
            }

        summary_dict["General"] = {"Total Volume": self.volume, "Height": H,
                                   "HasChannel": self.hasChannel}

        return summary_dict

solid_component_list["cuboid"] = Cuboid

class SolidComponentCollection(SolidComponent):
    """
    An abstract class to manage multiple SolidComponents in a system

    Parameters
    ----------
    components : Dict[str, SolidComponent]
        Collection of already initialized components

    Attributes
    ----------
    components : Dict[str, SolidComponent]
        dict of all the components used in the collection
    volume : float
        total volume of the components in the collection
    nCells : int
        number of total cells in the collection
    baseComponents : List[SolidComponents]
        list of all the base SolidComponent objects used in the collection
    """

    def __init__(self, components: Dict[str, SolidComponent]) -> None:
        super().__init__()
        self._components = components

    @property
    def components(self) -> Dict[str, SolidComponent]:
        return self._components

    @property
    def volume(self) -> float:
        return sum(component.volume for component in self.baseComponents)

    @property
    def nCells(self) -> int:
        return sum(component.nCells for component in self.baseComponents)

    @property
    def baseComponents(self) -> List[SolidComponent]:
        """
        Method for retrieving the base components (components that are not Component collections)
        of a component collection
        """
        base_components = []
        for component in self.components.values():
            base_components.extend(component.baseComponents)
        return base_components

    def getNodeGenerator(self) -> Generator[SolidComponent, None, None]:
        """
        Generator for marching over the nodes (i.e. cells) of a component
        """
        yield from [component.getNodeGenerator() for component in self.components.values()]

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

    def _summary(self):
        """
        Summary of the cuboid, as well as the channel if appropriate
        """
        raise NotImplementedError


class SerialSolidComponents(SolidComponentCollection):
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

    class Channel():
        """
        Channel object

        Parameters
        ----------
        pipe : Pipe
            pipe component used to create the channel
        serial_pipe : SerialComponents
            serial-pipes used to create the channel

        Attributes
        ----------
        pipeInformation : Dict
            dict containing cell-by-cell information of the pipe. This cell-by-cell data will
            allow for proper channels to be made within each component of the SerialSolidComponent
            object
        pipe : Union[Pipe, SerialComponents]
            the input pipe
        """
        def __init__(self,
                     pipe: Pipe = None,
                     serial_pipe: SerialComponents = None):
            assert (pipe is None) or (serial_pipe is None), "Must pick one input. cannot define both."
            if pipe is not None:
                self._pipe = pipe
                self._pipe_information = self._extractSinglePipeInformation(pipe)
            else:
                self._pipe = serial_pipe
                self._pipe_information = self._extractSerialPipeInformation(serial_pipe)

        @property
        def pipeInformation(self) -> Dict:
            return self._pipe_information

        @property
        def pipe(self):
            return self._pipe

        def _extractSinglePipeInformation(self, pipe: Pipe):
            """
            Extracts the information from a single pipe input, which will have constant axial
            geometry

            Parameters
            ----------
            pipe : Pipe
                A single pipe

            Returns
            -------
            pipe_information : dict
                a dict of geometric information of the pipe

            """
            pipe_information = {}
            for cell_i in range(pipe.nCell):
                pipe_information[cell_i] = {
                    "crossSection": pipe.crossSection,
                    "height": pipe.length,
                    "flowArea": pipe.flowArea,
                    "hydraulicDiameter": pipe.hydraulicDiameter
                }
            return pipe_information

        def _extractSerialPipeInformation(self, serial_pipe: SerialComponents):
            """
            Extracts the information from a serial pipe input, which will have constant axial
            geometry

            Parameters
            ----------
            serial_pipe : SerialComponents
                A serial pipe component

            Returns
            -------
            pipe_information : dict
                a dict of geometric information of the serial-pipe

            """
            print("Need to add functionality to handel a serial pipe.")
            raise NotImplementedError

    def __init__(self, components: Dict[str, SolidComponent], order: List[str]) -> None:
        components = SolidComponent.factory(components)
        super().__init__(components)
        self._order = order
        self._channel = None

    @property
    def componentOrder(self) -> List[str]:
        return self._order

    @property
    def height(self) -> float:
        return sum(component.height for component in self.components.values())

    @property
    def channel(self) -> Channel:
        return self._channel

    @channel.setter
    def channel(self, channel):
        self._channel = channel

    def addChannel(self, pipe: Pipe = None, serial_pipe: SerialComponents = None) -> None:
        """
        Adds a channel to the serial component, using either a single pipe or a serial-pipe

        Parameters
        ----------
        pipe : Pipe
            pipe component used to create the channel
        serial_pipe : SerialComponents
            serial-pipes used to create the channel
        """
        assert self.channel == None, "Cannot add a channel if one already exists"
        self.channel = self.Channel(pipe, serial_pipe)
        self._pipe = self.channel.pipe
        self._pipe_information = self.channel.pipeInformation

    def _addChannelToInputComponents(self):
        """
        For each component, this method adds proper channels to each of the serial components

        The basic loop goes over each component type input (in the correct order), and maps the
            cells of the component to cells in the channel-object. A current requirement is that,
            for a given solid component, ALL of the channel geometry must be constant throughout
            that component. That is, we currently cannot handle a channel that changes geometry
            halfway up a component.
        With the mapping made, the method checks that the above requirement is satisfied, and then
            creates a channel-object based on the channel-geometry throughout the component. A
            channel is then added using these geometries.
        Once this is done for a single channel, the loop moves on to the next channel and does the
            same
        Future work can be done when functionality for "SerialComponents" is added, and this method
            will most likely need to be altered to work for that functionality.
        """
        cell_count = 0
        for comp_name in self.componentOrder:
            comp = self.components[comp_name]
            n_cells = comp.nCells
            assert all(
                self.channel.pipeInformation[cell_count][dim_type] ==
                self.channel.pipeInformation[ci][dim_type] for ci in
                range(cell_count, cell_count + n_cells)
                for dim_type in ["height", "flowArea", "hydraulicDiameter"]
            ), "For now, all pipe dimensions in a single solid component must be the same"
            cross_section = self.channel.pipeInformation[cell_count]["crossSection"]
            comp.addChannelViaCrossSectionObject(cross_section)
            cell_count += n_cells

    def _summary(self):
        """
        Summary of the serial components, as well as the channel(s) if appropriate
        """
        raise NotImplementedError


class ParallelSolidComponents(SolidComponentCollection):
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
    componentMap : List[List[str]]
        a map showing where the component lie relative to each other in an XY-cross section
    """

    def __init__(
        self,
        components: Dict[str, SolidComponent],
        component_map: List[List[str]],
    ) -> None:
        components = SolidComponent.factory(components)
        super().__init__(components)
        self._component_map = component_map

    @property
    def componentMap(self) -> List[List[str]]:
        return self._component_map

    def _summary(self):
        """
        Summary of the parallel components, as well as the channels where appropriate
        """
        raise NotImplementedError


class SolidCore(ParallelSolidComponents):
    """
    A collection of parallel, solid components, forming an MSR core

    Parameters
    ----------
    (Fluid-Side)
    fluid_core : CartCore
        "Core" type from the fluid components. This input is required and many of the core specifics will
            come from this object
    (Solid-Side)
    solid_components : Dict[str, SolidComponent]
        a dict of SolidComponent object that could be used as blocks in the core
    solid_component_map : List[List[str]]
        maps the components in "solid_components" to specific spots in space
    fluid_pipe_map : List[List[str]]
        A 1:1 map with "solid_component_map" that specifies where in "solid_component_map" the pipes are,
            relative to the solid blocks.
    solid_material : str
        if "solid_component_map" is not specified, this input is required to create a uniform core with
            each block having the same solid properties
    core_height : float
        total height of the core
    solid_component_type : str
        type general component. Optional parameter used only to make "UniformComponent"
    n_axial_cells : int
        number of axial cells in the core
    global_offset : Tuple[int, int, int]
        A vector of the <x, y, z> offsets between the center of the fluid and solid cores,
            using the center of the fluid core as the reference origin (0,0,0). Note that this
            is different than an offset for a CuboidWithChannel object, as an offset defined
            within the CuboidWithChannel object is the channels center relative to the cuboids
            center. For the global-offset, this is the offset of the fluid and solid *cores*, and
            each offset set in the CuboidWithChannel object should be again adjusted by this global
            offset.

    Attributes
    ----------
    fluidCore : CartCore
        "Core" type from the fluid components. This input is required and many of the core specifics will
            come from this object
    coreSolidMaterial : str
        if "solid_component_map" is not specified, this input is required to create a uniform core with
            each block having the same solid properties
    coreHeight : float
        total height of the core
    solidComponentType : str
        type general component. Optional parameter used only to make "UniformComponent"
    nAxialCells : int
        number of axial cells in the core
    globalOffset : Tuple[int, int, int]
        global offset between the fluid and solid cores
    fluidPipeMap : List[List[str]]
        A 1:1 map with "solid_component_map" that specifies where in "solid_component_map" the pipes are,
            relative to the solid blocks.
    """

    def __init__(
        self,
        # Input Fluid Component
        fluid_core: CartCore,
        # Optional Solid Specifications
        solid_components: Optional[Dict[str, SolidComponent]] = None,
        solid_component_map: Optional[List[List[str]]] = None,
        fluid_pipe_map: Optional[List[List[str]]] = None,
        solid_material: Optional[str] = None,
        core_height: Optional[float] = None,
        solid_component_type: Optional[str] = "cuboid",
        n_axial_cells: Optional[int] = None,
        global_offset: Optional[Tuple[int, int, int]] = (0, 0, 0),
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
        if global_offset != (0, 0, 0):
            # TODO: implement offset functionality
            raise Exception("Need to implement fluid/solid core offset functionality")

        # Validating the solid and fluid maps
        self._fluid_pipe_map, self._solid_component_map = self._formatSolidAndFluidMaps(fluid_pipe_map, solid_component_map)

        # Reformating the solid component list and map
        solid_components = self._checkAndReformatSolidComponents(
            self._solid_component_map, self._fluid_pipe_map, solid_components
        )

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
    def globalOffset(self):
        return self._global_offset

    @property
    def fluidPipeMap(self):
        return self._fluid_pipe_map

    def _formatSolidAndFluidMaps(self,
                                 fluidMap: Optional[List[List[str]]] = None,
                                 solidMap: Optional[List[List[str]]] = None
                                 ) -> None:
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

        Parameters
        ----------
        fluidMap : List[List[str]]
            locations of the fluid-channels on an XY-cross section
        solidMap : List[List[str]]
            locations of the solid components on an XY-cross section
        """
        # Loads in the map from the fluidCore component
        fluidCoreMap = self.fluidCore.channelMap
        # Checks to see if the two fluid maps are the same shape or not
        if fluidMap is not None:
            fluidMapsAreTheSameShape = all(
                len(fluidMap) == len(fluidCoreMap), all(len(f1) == len(f2) for f1, f2 in zip(fluidMap, fluidCoreMap))
            )
        else:
            fluidMapsAreTheSameShape = False
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
        if (solidMap is not None) and (fluidMap is None):
            # Check consistent sizes
            assert len(fluidCoreMap) == len(solidMap)
            assert all(len(f) == len(s) for f, s in zip(fluidCoreMap, solidMap))
            # Sets fluidMap to fluidCoreMap
            return fluidCoreMap, solidMap

        ## Case 3
        if (solidMap is None) and (fluidMap is not None):
            # Case 3A
            if fluidMapsAreTheSameShape:
                solidMap = [["_" for _ in range(len(f))] for f in fluidMap]
                return fluidMap, solidMap
            # Case 3B
            raise NotImplementedError

        ## Case 4
        # (solidMap is not None) and (fluidMap is not None)
        # Case 4A
        if fluidMapsAreTheSameShape:
            assert len(fluidCoreMap) == len(solidMap)
            assert all(len(f) == len(s) for f, s in zip(fluidCoreMap, solidMap))
            return fluidMap, solidMap
        # Case 4B
        raise NotImplementedError

    def _checkAndReformatSolidComponents(self, solidMap, fluidMap, componentList):
        """
        Using the reformatted solid and fluid maps, this method ensures that the
        component list is completed. If not, this method fills in any gaps and
        carves any channels into blocks which need them

        Parameters
        ----------
        fluidMap : List[List[str]]
            locations of the fluid-channels on an XY-cross section
        solidMap : List[List[str]]
            locations of the solid components on an XY-cross section
        componentList : Dict[str, SolidComponent]
            dict of SolidComponent object that could be used as blocks in the core
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
                    assert solidComp.channel.fluidCrossSectionalArea == fluidComp.flowArea
                    assert solidComp.channel.hydraulicDiameter == fluidComp.hydraulicDiameter
                else:
                    solidComp.addChannel(fluid_component=fluidComp)
                componentList[solidCompId] = solidComp

        return componentList

    def _makeUniformComponent(self):
        """
        Creates a single component, build using the foundational geometric inputs
        and basic input material

        Returns
        -------
        component : SolidComponent
            a uniform solid component
        """
        x_pitch = self.fluidCore.xPitch
        y_pitch = self.fluidCore.yPitch
        height = self.coreHeight
        material = self.coreSolidMaterial
        comp_type = self.solidComponentType
        n_cells = self.nAxialCells

        assert x_pitch is not None, "Must input an x_pitch for a uniform core"
        assert y_pitch is not None, "Must input a y_pitch for a uniform core"
        assert height is not None, "Must input a height for a uniform core"
        assert material is not None, "Must input a solid material for a uniform core"
        assert comp_type is not None, "Must input a component-type for a uniform core"
        assert n_cells is not None, "Must input a number of axial cells for a uniform core"

        component = solid_component_list[comp_type](
            length=x_pitch, width=y_pitch, height=height, n_cells=n_cells, solid_material=material
        )

        return component

    def _makeUniformSolidComponentList(self):
        """
        Creates a list of uniform components, with appropriate channels built in.

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
            solid_comp = copy.deepcopy(uniform_component)
            solid_comp.addChannel(fluid_component=fluid_comp)
            solid_component_list[name] = solid_comp

        # For any element in the fluid-map not associated with a defined pipe-type, a filled
        #   solid block is used
        for row in self.fluidPipeMap:
            for comp_name in row:
                if comp_name not in parallel_fluid_components:
                    solid_component_list[comp_name] = copy.deepcopy(uniform_component)

        return solid_component_list

    def _summary(self):
        """
        Summary of the parallel components, as well as the channels where appropriate
        """
        raise NotImplementedError
