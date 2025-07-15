import numpy as np
from numpy import pi, isclose
from flowforge.input.SolidComponents import (
    Cuboid,
    CuboidWithChannel,
    HexagonalPrism,
    SolidEncasedPipe
)
from flowforge.input.Components import (
    Pipe
)
from flowforge import UnitConverter

unit_dict = {"length": "cm"}
uc = UnitConverter(unit_dict)

def test_Cuboid():
    # Inputs
    length  = 10.0     # m
    width   = 15.3     # m
    height  = 11.1243  # m
    n_cells = 12

    # Base object
    cuboid = Cuboid(length, width, height, n_cells)

    # Base units
    assert cuboid.length == length
    assert cuboid.width  == width
    assert cuboid.height == height
    assert cuboid.nCells == n_cells
    assert cuboid.volume > 0
    assert cuboid.volume == length * width * height
    cuboid.volume = 0.1 * (length * width * height)
    assert cuboid.volume == 0.1 * (length * width * height)
    cuboid.volume = length * width * height

    # Convert units (assumes cm and converts to m)
    cuboid._convertUnits(uc)
    assert cuboid.length == 0.01 * length
    assert cuboid.width == 0.01 * width
    assert cuboid.height == 0.01 * height
    assert cuboid.volume == (0.01**3) * length * width * height

def test_CuboidWithChannel():
    # Cuboid dimensions
    length  = 10.0     # m
    width   = 15.3     # m
    height  = 11.1243  # m
    n_cells = 12

    # Pipe dimensions
    R = 1.123 # m
    A = 3.81  # m
    W = 2.0   # m
    H = 3.8   # m

    def _computeDh(flow_area, wetted_perimeter):
        """ Dh := 4A/P """
        return 4 * flow_area / wetted_perimeter

    def _computeVolumeWithPipe(length, width, height, fluid_area):
        """ V := V_solid - V_channel """
        return (length * width * height) - (fluid_area * height)

    # Create objects
    circle = CuboidWithChannel(
        length, width, height, n_cells, pipe_cross_section_type="circular",
        **{"R": R})
    stadium = CuboidWithChannel(
        length, width, height, n_cells, pipe_cross_section_type="stadium",
        **{"R": R, "A": A})
    square = CuboidWithChannel(
        length, width, height, n_cells, pipe_cross_section_type="square",
        **{"W": W})
    rectangle = CuboidWithChannel(
        length, width, height, n_cells, pipe_cross_section_type="rectangular",
        **{"W": W, "H": H})

    # Base units
    # -- Cuboid with Circular Pipe
    assert circle.fluidCrossSectionalArea == pi * R * R
    assert circle.wettedPerimeter == 2 * pi * R
    assert circle.hydraulicDiameter == _computeDh(
        circle.fluidCrossSectionalArea, circle.wettedPerimeter)
    assert circle.volume == _computeVolumeWithPipe(
        length, width, height, circle.fluidCrossSectionalArea)
    # -- Cuboid with Stadium Pipe
    assert stadium.fluidCrossSectionalArea == (pi * R * R) + (2 * R * A)
    assert stadium.wettedPerimeter == 2 * pi * R + 2 * A
    assert stadium.hydraulicDiameter == _computeDh(
        stadium.fluidCrossSectionalArea, stadium.wettedPerimeter)
    assert stadium.volume == _computeVolumeWithPipe(
        length, width, height, stadium.fluidCrossSectionalArea)
    # -- Cuboid with Square Pipe
    assert square.fluidCrossSectionalArea == W * W
    assert square.wettedPerimeter == 4 * W
    assert square.hydraulicDiameter == _computeDh(
        square.fluidCrossSectionalArea, square.wettedPerimeter)
    assert square.volume == _computeVolumeWithPipe(
        length, width, height, square.fluidCrossSectionalArea)
    # -- Cuboid with Rectangular Pipe
    assert rectangle.fluidCrossSectionalArea == W * H
    assert rectangle.wettedPerimeter == 2 * W + 2 * H
    assert rectangle.hydraulicDiameter == _computeDh(
        rectangle.fluidCrossSectionalArea, rectangle.wettedPerimeter)
    assert rectangle.volume == _computeVolumeWithPipe(
        length, width, height, rectangle.fluidCrossSectionalArea)

    # Convert units (assumes cm and converts to m)
    circle._convertUnits(uc)
    stadium._convertUnits(uc)
    square._convertUnits(uc)
    rectangle._convertUnits(uc)
    # -- Cuboid with Circular Pipe
    assert isclose(circle.fluidCrossSectionalArea, (0.01**2) * pi * R * R)
    assert isclose(circle.wettedPerimeter, 0.01 * 2 * pi * R)
    # -- Cuboid with Stadium Pipe
    assert isclose(stadium.fluidCrossSectionalArea, (0.01**2) * ((pi * R * R) + (2 * R * A)))
    assert isclose(stadium.wettedPerimeter, 0.01 * (2 * pi * R + 2 * A))
    # -- Cuboid with Square Pipe
    assert isclose(square.fluidCrossSectionalArea, (0.01**2) * W * W)
    assert isclose(square.wettedPerimeter, 0.01 * 4 * W)
    # -- Cuboid with Rectangular Pipe
    assert isclose(rectangle.fluidCrossSectionalArea, (0.01**2) * W * H)
    assert isclose(rectangle.wettedPerimeter, 0.01 * (2 * W + 2 * H))

def test_SolidEncasedPipe():
    # Cuboid dimensions
    length  = 10.0     # m
    width   = 15.3     # m
    height  = 11.1243  # m
    n_cells = 12

    # Pipe dimensions
    CX_type = "circular"
    R = 1.123 # m

    # Base objects
    pipe             = Pipe(L=height, cross_section_name=CX_type, n=n_cells, R=R)
    cuboid           = Cuboid(length, width, height, n_cells)
    channeled_cuboid = CuboidWithChannel(length, width, height, n_cells,
                                         pipe_cross_section_type=CX_type, R=R)

    # Test objects
    encasedPipe_noChannel   = SolidEncasedPipe(pipe=pipe, components={"1": cuboid}, order=["1"])
    encasedPipe_withChannel = SolidEncasedPipe(pipe=pipe, components={"1": channeled_cuboid}, order=["1"])

    assert all(isinstance(comp, CuboidWithChannel) for comp in encasedPipe_noChannel.baseComponents)
    assert all(isinstance(comp, CuboidWithChannel) for comp in encasedPipe_withChannel.baseComponents)
    assert all(c1.volume == c2.volume for c1, c2 in zip(encasedPipe_noChannel.baseComponents,
                                                        encasedPipe_noChannel.baseComponents))

    # Test SolidEncasedPipe with multiple input components, both with and without channels pre-built
    multi_components = {}
    order = []
    encasedPipe_multiSegment = SolidEncasedPipe(pipe       = pipe,
                                                components = multi_components,
                                                order      = order)

def test_SolidCore():


    return

if __name__ == "__main__":
    test_Cuboid()
    test_CuboidWithChannel()
    test_SolidEncasedPipe()
    test_SolidCore