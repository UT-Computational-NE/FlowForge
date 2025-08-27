import abc
import numpy as np

from flowforge.input.UnitConverter import UnitConverter

class Shape:
    """
    Abstract base class for defining shapes

    Attributes
    ----------
    area : float
        Area of the given shape
    perimeter : float
        Perimeter of the given shape
    """
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def area(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def perimeter(self):
        raise NotImplementedError

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

    def __init__(self, R):
        super().__init__()
        self._radius = R

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def area(self):
        return np.pi * self.radius * self.radius

    @property
    def perimeter(self):
        return 2.0 * np.pi * self.radius

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

    def __init__(self, A, R):
        super().__init__()
        self._length = A
        self._radius = R

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def radius(self):
        return self.radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def area(self):
        return (np.pi * self.radius ** 2.0) + (2.0 * self.radius * self.length)

    @property
    def perimeter(self):
        raise 2.0 * (np.pi * self.radius + self.length)

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

    def __init__(self, H, W):
        super().__init__()
        self._height = H
        self._width = W

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width

    @property
    def area(self):
        return self.length * self.width

    @property
    def perimeter(self):
        return 2.0 * (self.length + self.width)

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

    def __init__(self, width):
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

    def __init__(self, L):
        super().__init__()
        self._length = L
        self._BASE_AREA_COEFF = 3.0 * np.sqrt(3.0) / 2.0

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    @property
    def area(self):
        return self._BASE_AREA_COEFF * self.length * self.length

    @property
    def perimeter(self):
        return 6.0 * self._length

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