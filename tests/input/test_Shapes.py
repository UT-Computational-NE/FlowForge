import numpy as np

from flowforge.input.Shapes import *
from flowforge import UnitConverter

_cm2m = 0.01
unit_dict = {"length": "cm"}
uc = UnitConverter(unit_dict)

def test_Circle():

    radius = 1.125  # m
    circle = Circle(R=radius)

    assert circle.radius == radius
    assert circle.area == np.pi * radius * radius
    assert circle.perimeter == 2.0 * np.pi * radius

    circle._convertUnits(uc=uc)
    assert np.isclose(circle.radius, radius * _cm2m)
    assert np.isclose(circle.area, np.pi * radius * radius * _cm2m * _cm2m)
    assert np.isclose(circle.perimeter, 2.0 * np.pi * radius * _cm2m)


def test_Stadium():

    radius = 1.125  # m
    length = 1.88  # m
    stadium = Stadium(A=length, R=radius)

    assert stadium.radius == radius
    assert stadium.length == length
    assert stadium.area == (np.pi * radius ** 2.0) + (2.0 * radius * length)
    assert stadium.perimeter == 2.0 * (np.pi * radius + length)

    stadium._convertUnits(uc=uc)
    assert np.isclose(stadium.radius, radius * _cm2m)
    assert np.isclose(stadium.length, length * _cm2m)
    assert np.isclose(stadium.area, ((np.pi * radius ** 2.0) + (2.0 * radius * length)) * _cm2m * _cm2m)
    assert np.isclose(stadium.perimeter, (2.0 * (np.pi * radius + length)) * _cm2m)


def test_Rectangle():

    height = 1.125  # m
    width = 0.9877 # m
    rectangle = Rectangle(H=height, W=width)

    assert rectangle.height == height
    assert rectangle.width == width
    assert rectangle.area == height * width
    assert rectangle.perimeter == 2.0 * (height + width)

    rectangle._convertUnits(uc=uc)
    assert np.isclose(rectangle.height, height * _cm2m)
    assert np.isclose(rectangle.width, width * _cm2m)
    assert np.isclose(rectangle.area, height * width * _cm2m * _cm2m)
    assert np.isclose(rectangle.perimeter, 2.0 * (height + width) * _cm2m)


def test_Square():

    width = 0.9877 # m
    square = Square(W=width)

    assert square.height == width
    assert square.width == width
    assert square.area == width * width
    assert square.perimeter == 2.0 * (width + width)

    square._convertUnits(uc=uc)
    assert np.isclose(square.height, width * _cm2m)
    assert np.isclose(square.width, width * _cm2m)
    assert np.isclose(square.area, width * width * _cm2m * _cm2m)
    assert np.isclose(square.perimeter, 2.0 * (width + width) * _cm2m)


def test_Hexagon():

    length = 0.9877 # m
    coeff = 3.0 * np.sqrt(3.0) / 2.0
    hexagon = Hexagon(L=length)

    assert hexagon.length == length
    assert hexagon.area == coeff * length * length
    assert hexagon.perimeter == 6.0 * length

    hexagon._convertUnits(uc=uc)
    assert np.isclose(hexagon.length, length * _cm2m)
    assert np.isclose(hexagon.area, coeff * length * length * _cm2m * _cm2m)
    assert np.isclose(hexagon.perimeter, 6.0 * length * _cm2m)


def test_CrossSection():
    radius = 1.812  # m
    length = 1.921  # m
    height = 1.870  # m
    width = 1.7790  # m

    circle = Circle(R=radius)
    stadium = Stadium(A=length, R=radius)
    rectangle = Rectangle(H=height, W=width)
    square = Square(W=width)
    hexagon = Hexagon(L=length)

    circle_CX = CrossSection(shape="circular", R=radius)
    stadium_CX = CrossSection(shape="stadium", A=length, R=radius)
    rectangle_CX = CrossSection(shape="rectangular", H=height, W=width)
    square_CX = CrossSection(shape="square", W=width)
    hexagon_CX = CrossSection(shape="hexagon", L=length)

    assert circle.area == circle_CX.shape.area
    assert stadium.area == stadium_CX.shape.area
    assert rectangle.area == rectangle_CX.shape.area
    assert square.area == square_CX.shape.area
    assert hexagon.area == hexagon_CX.shape.area

    assert circle.perimeter == circle_CX.shape.perimeter
    assert stadium.perimeter == stadium_CX.shape.perimeter
    assert rectangle.perimeter == rectangle_CX.shape.perimeter
    assert square.perimeter == square_CX.shape.perimeter
    assert hexagon.perimeter == hexagon_CX.shape.perimeter


if __name__ == "__main__":
    test_Circle()
    test_Stadium()
    test_Rectangle()
    test_Square()
    test_Hexagon()
    test_CrossSection()