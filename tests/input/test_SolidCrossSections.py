import numpy as np

from flowforge.input.SolidCrossSections import *
from flowforge.input.Components import CircularCrossSection, StadiumCrossSection
from flowforge import UnitConverter

_cm2m = 0.01
unit_dict = {"length": "cm"}
uc = UnitConverter(unit_dict)

def test_Rectangle():
    assert Rectangle.inputs == ("length", "width")

    length = 1.124 # m
    width = 2.0 # m
    rectangle = Rectangle(length = length,
                          width = width)

    assert rectangle.length == length
    assert rectangle.width == width
    assert rectangle.baseArea == length * width
    assert rectangle.area == rectangle.baseArea
    assert rectangle.channel is None

    rectangle._convertUnits(uc=uc)
    assert rectangle.length == length * _cm2m
    assert rectangle.width == width * _cm2m
    assert rectangle.baseArea == length * width * _cm2m * _cm2m


def test_Square():
    assert Square.inputs == tuple(["length"])

    length = 1.124 # m
    square = Square(length = length)

    assert square.length == length
    assert square.width == length
    assert square.baseArea == length * length
    assert square.area == square.baseArea
    assert square.channel is None

    square._convertUnits(uc=uc)
    assert square.length == length * _cm2m
    assert square.width == length * _cm2m
    assert square.baseArea == length * length * _cm2m * _cm2m


def test_Hexagon():
    assert Hexagon.inputs == tuple(["length"])

    length = 1.124 # m
    BASE_AREA_COEFF = 3.0 * np.sqrt(3.0) / 2.0
    hexagon = Hexagon(length = length)

    assert hexagon.length == length
    assert hexagon.baseArea == BASE_AREA_COEFF * length * length
    assert hexagon.area == hexagon.baseArea
    assert hexagon.channel is None

    hexagon._convertUnits(uc=uc)
    assert hexagon.length == length * _cm2m
    assert hexagon.baseArea == BASE_AREA_COEFF * (length * _cm2m) ** 2


def test_addChannel():
    length = 1.124 # m
    width = 2.0 # m
    rectangle = Rectangle(length = length,
                          width = width)
    R1 = 0.1 # m
    R2 = 0.1 # m
    A = 0.15 # m
    circle = CircularCrossSection(R=R1)
    stadium = StadiumCrossSection(R=R2, A=A)

    assert rectangle.baseArea == length * width
    assert circle.flow_area == np.pi * R1**2
    assert stadium.flow_area == np.pi * (R2 ** 2) + 2 * R2 * A

    rectangle.channel = circle
    assert rectangle.area == (length * width) - (np.pi * R1**2)

    rectangle.changeChannel(stadium)
    assert rectangle.area == (length * width) - (np.pi * (R2 ** 2) + 2 * R2 * A)

    rectangle._convertUnits(uc=uc)
    assert rectangle.area == ((length * width) * _cm2m**2 - (np.pi * (R2 ** 2) + 2 * R2 * A) * _cm2m**2)

if __name__ == "__main__":
    test_Rectangle()
    test_Square()
    test_Hexagon()
    test_addChannel()