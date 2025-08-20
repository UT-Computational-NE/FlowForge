import abc
import numpy as np

# from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter
from flowforge.input.Components import CrossSection as FluidCrossSection

# pragma pylint: disable=protected-access

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""


class CrossSection:
    """
    Abstract base class for solid cross sections

    Attributes
    ----------
    channel : FluidCrossSection
        cross-section of the sub-channel running through through this shape
    baseArea : float
        area of the shape, ignoring any channels running through
    area : float
        total area of the shape, defined as the base area with the channel
        area subtracted
    """

    # Class inputs
    inputs = ()

    def __init__(self) -> None:
        self._channel = None

    @property
    def area(self) -> float:
        if self.channel is None:
            return self.baseArea
        return self.baseArea - self.channel.flow_area

    @property
    @abc.abstractmethod
    def baseArea(self) -> float:
        pass

    @property
    def channel(self) -> FluidCrossSection:
        return self._channel

    @channel.setter
    def channel(self, channel: FluidCrossSection):
        assert self.channel is None, "Should use 'changeChannel' method instead'"
        assert channel.flow_area < self.baseArea
        self._channel = channel

    def changeChannel(self, channel: FluidCrossSection) -> None:
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
        if self.channel is not None:
            self.channel._convertUnits(uc)


class Rectangle(CrossSection):
    """
    Rectangular cross section

    Parameters
    ----------
    length : float
        length of the rectangle
    width : float
        width of the rectangle

    Attributes
    ----------
    length : float
        length of the rectangle
    width : float
        width of the rectangle
    baseArea : float
        area of the shape, ignoring any channels running through (A = L x W)
    channel : FluidCrossSection
        cross-section of the sub-channel running through through this shape
    area : float
        total area of the shape, defined as the base area with the channel
        area subtracted
    """

    # Class inputs
    inputs = ("length", "width")

    def __init__(self, length: float, width: float) -> None:
        super().__init__()
        self._length = length
        self._width = width

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, width):
        self._width = width

    @property
    def baseArea(self) -> float:
        return self.length * self.width

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
        super()._convertUnits(uc)
        self.length *= uc.lengthConversion
        self.width *= uc.lengthConversion


class Square(Rectangle):
    """
    Square cross section

    Parameters
    ----------
    length : float
        length of the square

    Attributes
    ----------
    length : float
        length of the square
    baseArea : float
        area of the shape, ignoring any channels running through (A = L^2)
    channel : FluidCrossSection
        cross-section of the sub-channel running through through this shape
    area : float
        total area of the shape, defined as the base area with the channel
        area subtracted
    """

    # Class inputs
    inputs = tuple(["length"])

    def __init__(self, length: float) -> None:
        super().__init__(length, length)


class Hexagon(CrossSection):
    """
    Hexagon cross section

    Parameters
    ----------
    length : float
        length of all sides of the hexagon

    Attributes
    ----------
    length : float
        length of all sides of the hexagon
    baseArea : float
        area of the shape, ignoring any channels running through (A = (3*sqrt[3]/2) * L^2)
    channel : FluidCrossSection
        cross-section of the sub-channel running through through this shape
    area : float
        total area of the shape, defined as the base area with the channel
        area subtracted
    """

    # Class inputs
    inputs = tuple(["length"])

    def __init__(self, length: float) -> None:
        super().__init__()
        self._BASE_AREA_COEFF = 3.0 * np.sqrt(3.0) / 2.0
        self._length = length

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def baseArea(self) -> float:
        return self._BASE_AREA_COEFF * self.length * self.length

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
        super()._convertUnits(uc)
        self.length *= uc.lengthConversion


cross_section_types = {"rectangle": Rectangle, "square": Square, "hexagon": Hexagon}
