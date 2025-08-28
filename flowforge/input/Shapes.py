from abc import ABC, abstractmethod
import numpy as np

from flowforge.input.UnitConverter import UnitConverter

class Shape(ABC):
    """
    Abstract base class for defining shapes

    Attributes
    ----------
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """

    _area : float
    _perimeter : float

    @property
    def area(self) -> float:
        return self._area

    @property
    def perimeter(self) -> float:
        return self._perimeter

    @abstractmethod
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


class Circle(Shape):
    """
    Circle Class

    Parameters
    ----------
    R : float
        Radius of the circle

    Attributes
    ----------
    radius : float
        Radius of the circle
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """

    # Class inputs
    inputs = tuple(["R"])

    def __init__(self, R: float) -> None:
        self._radius = R
        self._area = np.pi * R * R
        self._perimeter = 2.0 * np.pi * R

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, radius) -> None:
        self._radius = radius
        self._area = np.pi * radius * radius
        self._perimeter = 2.0 * np.pi * radius

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
        self.radius *= uc.lengthConversion


class Stadium(Shape):
    """
    Stadium Class

    Parameters
    ----------
    length : float
        Length of the rectangular portion of the stadium
    radius : float
        Radius of the semi circular portion of the stadium

    Attributes
    ----------
    length : float
        Length of the rectangular portion of the stadium
    radius : float
        Radius of the semi circular portion of the stadium
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """

    # Class inputs
    inputs = tuple(["A", "R"])

    def __init__(self, A: float, R: float) -> None:
        self._length = A
        self._radius = R
        self._area = (np.pi * R ** 2.0) + (2.0 * R * A)
        self._perimeter = 2.0 * (np.pi * R + A)

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length) -> None:
        self._length = length
        self._area = (np.pi * self.radius ** 2.0) + (2.0 * self.radius * length)
        self._perimeter = 2.0 * (np.pi * self.radius + length)

    @property
    def radius(self) -> float:
        return self.radius

    @radius.setter
    def radius(self, radius) -> None:
        self._radius = radius

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
        self.radius *= uc.lengthConversion
        self.length *= uc.lengthConversion


class Rectangle(Shape):
    """
    Rectangle Class

    Parameters
    ----------
    H : float
        Height of the rectangle
    W : float
        Width of the rectangle

    Attributes
    ----------
    height : float
        Height of the rectangle
    width : float
        Width of the rectangle
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """

    # Class inputs
    inputs = tuple(["H", "W"])

    def __init__(self, H: float, W: float) -> None:
        self._height = H
        self._width = W
        self._area = H * W
        self._perimeter = 2.0 * (H + W)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, height) -> None:
        self._height = height
        self._area = height * self.width
        self._perimeter = 2.0 * (height * self.width)

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, width) -> None:
        self._width = width
        self._area = self.height * width
        self._perimeter = 2.0 * (self.height * width)

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
        self.width *= uc.lengthConversion


class Square(Rectangle):
    """
    Rectangle Class

    Parameters
    ----------
    length : float
        Length of the square

    Attributes
    ----------
    length : float
        Length of the square
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """

    # Class inputs
    inputs = tuple(["W"])

    def __init__(self, width) -> None:
        super().__init__(width, width)


class Hexagon(Shape):
    """
    Hexagon Class

    Parameters
    ----------
    L : float
        Side length of the hexagon

    Attributes
    ----------
    length : float
        Side length of the hexagon
    width : float
        Width of the rectangle
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """

    # Class inputs
    inputs = tuple(["L"])

    def __init__(self, L) -> None:
        self._length = L
        self._BASE_AREA_COEFF = 3.0 * np.sqrt(3.0) / 2.0
        self._area = self._BASE_AREA_COEFF * L * L
        self._perimeter = 6.0 * L

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length) -> None:
        self._length = length
        self._area = self._BASE_AREA_COEFF * length * length
        self._perimeter = 6.0 * length

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
        self.length *= uc.lengthConversion

shape_types = {
    "circular": Circle,
    "square": Square,
    "rectangular": Rectangle,
    "stadium": Stadium,
    "hexagon": Hexagon
}

shape_inputs = {shape_key: shape_obj.inputs for shape_key, shape_obj in shape_types.items()}

class CrossSection:
    """
    Base class used to build cross sections

    Parameters
    ----------
    shape : Shape
        Shape type desired for the cross section

    Attributes
    ----------
    shape : Shape
        Shape type desired for the cross section
    """
    def __init__(self, shape, **kwargs):
        shape = self._buildShape(shape, **kwargs)

    @property
    def shape(self) -> Shape:
        return self._shape

    def _buildShape(shape_name, **kwargs):
        """
        Using the input shape and kwargs, this method checks that all the necessary inputs for the
        given shape are in kwargs, and then builds the shape using them
        """
        shape_object = shape_types[shape_name]
        assert all(inp in kwargs for inp in shape_object.inputs)
        shape = shape_object(**{k: v for k, v in kwargs.items() if k in shape_object.inputs})
        return shape