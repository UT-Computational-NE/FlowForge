from typing import List, Dict, Tuple, Generator, Optional, Literal, TypeAlias
from six import add_metaclass
import numpy as np
from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter

from flowforge.input.Components import cross_section_classes, cross_section_param_lists

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
    """
    def __init__(self) -> None:
        self.uc = None

    @property
    def volume(self) -> float:
        raise NotImplementedError

    @property
    def nCells(self) -> int:
        raise NotImplementedError

class HexagonalPrism(SolidComponent):
    """
    A hexagonal prism solid component

    Attributes
    ----------
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class Cuboid(SolidComponent):
    """
    A solid cuboid component

    Attributes
    ----------
    length  : length of the cuboid (x-direction)
    width   :  width of the cuboid (y-direction)
    height  : height of the cuboid (z-direction)
    n_cells : number of cells in the cuboid
    """
    def __init__(self,
                 length  : float,
                 width   : float,
                 height  : float,
                 n_cells : int = 1):
        # Basic Parameters
        self._length  = length
        self._width   = width
        self._height  = height
        self._n_cells = n_cells
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

class CuboidWithChannel(Cuboid):
    """
    A cuboid solid component with a channel running through it

    Attributes
    ----------
    length  : length of the cuboid (x-direction)
    width   :  width of the cuboid (y-direction)
    height  : height of the cuboid (z-direction)
    n_cells : number of cells in the cuboid

    Optional, Pipe Cross-Section Inputs
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
                 pipe_cross_section_type : str = "circular",
                 **kwargs):
        super().__init__(length, width, height, n_cells)
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