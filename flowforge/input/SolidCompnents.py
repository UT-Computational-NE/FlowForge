from typing import List, Dict, Tuple, Generator, Optional, Literal, TypeAlias
from six import add_metaclass
import numpy as np
from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter

from flowforge.input.Components import CrossSection, CircularCrossSection, RectangularCrossSection, SquareCrossSection, \
                                       StadiumCrossSection, cross_section_classes, cross_section_param_lists

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
                 length: float,
                 width: float,
                 height: float,
                 n_cells: int = 1):
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

    @property
    def nCells(self) -> int:
        return self._n_cells

class ChannelCuboid(Cuboid):
    """
    A cuboid solid component with a channel running through it

    Attributes
    ----------
    length  : length of the cuboid (x-direction)
    width   :  width of the cuboid (y-direction)
    height  : height of the cuboid (z-direction)
    n_cells : number of cells in the cuboid
    ???
    """
    def __init__(self,
                 length,
                 width,
                 height,
                 n_cells = 1,
                 pipe_cross_section_type: str = "circular",
                 **kwargs):
        super().__init__(length, width, height, n_cells)
        self._cross_section = cross_section_classes[pipe_cross_section_type](
            **{k: v for k, v in kwargs.items() if k in cross_section_param_lists[pipe_cross_section_type]}
        )
        self._Ac = self._cross_section.flow_area
        self._Pw = self._cross_section.wetted_perimeter
        self._Dh = self._cross_section.hydraulic_diameter