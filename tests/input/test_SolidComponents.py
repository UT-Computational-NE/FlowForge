import numpy as np
from numpy import pi, isclose
from flowforge.input.SolidComponents import (
    Cuboid,
    CuboidWithChannel,
    HexagonalPrism,
    SolidEncasedPipe,
    SolidCore
)
from flowforge.input.Components import (
    Pipe,
    Core,
    CartCore
)
from flowforge import UnitConverter

unit_dict = {"length": "cm"}
uc = UnitConverter(unit_dict)

geometry = {
    "Cuboid":
        {"length": 10.0, "width": 15.3,
         "height": 11.1243, "n_cells": 12},

    "CuboidWithChannel":
        {"length": 10.0, "width": 15.3,
        "height": 11.1243, "n_cells": 12,
        "R": 1.123, "A": 3.81, "W": 2.0,
        "H": 3.8},

    "SolidEncasedPipe":
        {"length": 10.0, "width": 15.3,
        "height": 11.1243, "n_cells": 12,
        "R": 1.123, "CX": "circular"},

    "FluidCore":
        {"lower_plenum":
            {"L": 0.8, "Rin": 0.1, "Rout": 0.5},
         "upper_plenum":
            {"L": 0.8, "R": 0.5, "CX": "circular"},
         "pipe1":
            {"L": 5.5, "n": 10, "R": 0.1, "CX": "circular"},
         "pipe2":
            {"L": 5.5, "n": 10, "R": 0.2, "CX": "circular"},
         "pipe3":
            {"L": 5.5, "n": 10, "R": 0.075, "A": 0.025, "CX": "stadium"},
        }
}

def test_Cuboid():
    # Inputs
    geom    = geometry["Cuboid"]
    length  = geom["length"]
    width   = geom["width"]
    height  = geom["height"]
    n_cells = geom["n_cells"]

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
    geom    = geometry["CuboidWithChannel"]
    length  = geom["length"]
    width   = geom["width"]
    height  = geom["height"]
    n_cells = geom["n_cells"]

    # Pipe dimensions
    R = geom["R"]
    A = geom["A"]
    W = geom["W"]
    H = geom["H"]

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
    geom    = geometry["SolidEncasedPipe"]
    length  = geom["length"]
    width   = geom["width"]
    height  = geom["height"]
    n_cells = geom["n_cells"]

    # Pipe dimensions
    CX_type = geom["CX"]
    R       = geom["R"]

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
    # TODO: Finish this

def _createFluidCoreComponent(x_pitch, y_pitch):
    # ------- INPUTS ------- #
    lp_geom    = geometry["FluidCore"]["lower_plenum"]
    up_geom    = geometry["FluidCore"]["upper_plenum"]
    pipe1_geom = geometry["FluidCore"]["pipe1"]
    pipe2_geom = geometry["FluidCore"]["pipe2"]
    pipe3_geom = geometry["FluidCore"]["pipe3"]
        # * Plenums * #
    # Lower plenum
    L_lp, Rin_lp, Rout_lp = lp_geom["L"], lp_geom["Rin"], lp_geom["Rout"]
    # Upper Plenum
    L_up, R_up, CX_up = up_geom["L"], up_geom["R"], up_geom["CX"]
        # * Pipes * #
    # Pipe #1
    L_p1, N_p1, R_p1, CX_p1 = (pipe1_geom["L"], pipe1_geom["n"],
                               pipe1_geom["R"], pipe1_geom["CX"])
    # Pipe #2
    L_p2, N_p2, R_p2, CX_p2 = (pipe2_geom["L"], pipe2_geom["n"],
                               pipe2_geom["R"], pipe2_geom["CX"])
    # Pipe #3
    L_p3, N_p3, R_p3, A_p3, CX_p3 = (pipe3_geom["L"], pipe3_geom["n"],
                                     pipe3_geom["R"], pipe3_geom["A"],
                                     pipe3_geom["CX"])
    # ---------------------- #

    # Create component definitions
    lower_plenum = {"nozzle": {"L": L_lp, "R_inlet": Rin_lp, "R_outlet": Rout_lp}}
    upper_plenum = {"pipe": {"L": L_up, "R": R_up, "cross_section_name": CX_up}}
    pipes = {"1": Pipe(L=L_p1, n=N_p1, cross_section_name=CX_p1, R=R_p1),
             "2": Pipe(L=L_p2, n=N_p2, cross_section_name=CX_p2, R=R_p2),
             "3": Pipe(L=L_p3, n=N_p3, cross_section_name=CX_p3, R=R_p3, A=A_p3)}

    # Channel Map
    map_alignment = "center"
    channel_map = [
            ["1","2","1"],
        ["1","2","3","2","1"],
        ["2","3","3","3","2"],
        ["1","2","3","2","1"],
            ["1","2","1"]]

    # Build "CartCore" fluid component
    core = CartCore(x_pitch=x_pitch,
                    y_pitch=y_pitch,
                    components=pipes,
                    lower_plenum=lower_plenum,
                    upper_plenum=upper_plenum,
                    channel_map=channel_map,
                    map_alignment=map_alignment)

    return core

def _createSolidCoreComponents(fluid_core):
    """
    Core Input Types:
        1) Uniform solid core
        2) Refined uniform solid core
        3)
    """
    # General Inputs

    def _makeUniformCore(fluid_core):
        # Uniform Core
        solid_material = "graphite"
        solid_component_type = "cuboid"
        core_height = 5.5 # m
        n_axial_cells = 10 # cells
        core = SolidCore(
            fluid_core           = fluid_core,
            solid_material       = solid_material,
            core_height          = core_height,
            solid_component_type = solid_component_type,
            n_axial_cells        = n_axial_cells
        )
        return core

    def _makeNonUniformSolidCores(fluid_core):
        # * Pipes * #
        pipe1_geom = geometry["FluidCore"]["pipe1"]
        pipe2_geom = geometry["FluidCore"]["pipe2"]
        pipe3_geom = geometry["FluidCore"]["pipe3"]
        # Pipe #1
        L_p1, N_p1, R_p1, CX_p1 = (pipe1_geom["L"], pipe1_geom["n"],
                                   pipe1_geom["R"], pipe1_geom["CX"])
        # Pipe #2
        L_p2, N_p2, R_p2, CX_p2 = (pipe2_geom["L"], pipe2_geom["n"],
                                   pipe2_geom["R"], pipe2_geom["CX"])
        # Pipe #3
        L_p3, N_p3, R_p3, A_p3, CX_p3 = (pipe3_geom["L"], pipe3_geom["n"],
                                         pipe3_geom["R"], pipe3_geom["A"],
                                         pipe3_geom["CX"])
        assert (L_p1 == L_p2) and (L_p2 == L_p3)
        assert (N_p1 == N_p2) and (N_p2 == N_p3)
        x_pitch, y_pitch = fluid_core.xPitch, fluid_core.yPitch
        # Non-Uniform Solid Core
        solid_components = {}

        # SOLID COMPONENTS
        solid_components["all_channeled"] = {
            "1": CuboidWithChannel(length=x_pitch, width=y_pitch,
                                   height=L_p1, nCells=N_p1, R=R_p1,
                                   pipe_cross_section_type=CX_p1),
            "2": CuboidWithChannel(length=x_pitch, width=y_pitch,
                                   height=L_p2, nCells=N_p2, R=R_p2,
                                   pipe_cross_section_type=CX_p2),
            "3": CuboidWithChannel(length=x_pitch, width=y_pitch,
                                   height=L_p3, nCells=N_p3, R=R_p3, A=A_p3,
                                   pipe_cross_section_type=CX_p3)}

        solid_components["some_channeled"] = {
            "1": CuboidWithChannel(length=x_pitch, width=y_pitch,
                                   height=L_p1, nCells=N_p1, R=R_p1,
                                   pipe_cross_section_type=CX_p1),
            "2": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p2, nCells=N_p2),
            "3": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p3, nCells=N_p3)}

        solid_components["no_channeled"] = {
            "1": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p1, nCells=N_p1),
            "2": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p2, nCells=N_p2),
            "3": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p3, nCells=N_p3)}

        # COMPONENT MAPS
        basic_solid_map = [
            ["1","2","1"],
        ["1","2","3","2","1"],
        ["2","3","3","3","2"],
        ["1","2","3","2","1"],
            ["1","2","1"]]

        ### BUILD CORES ###
        core_basic = SolidCore(
            fluid_core          = fluid_core,
            solid_components    = solid_components["all_channeled"],
            solid_component_map = basic_solid_map,
            solid_material      = "graphite"
        )
        core_some_channeled = SolidCore(
            fluid_core          = fluid_core,
            solid_components    = solid_components["some_channeled"],
            solid_component_map = basic_solid_map,
            solid_material      = "graphite",
            core_height         = L_p1,
            n_axial_cells       = N_p1
        )
        core_no_channeled = SolidCore(
            fluid_core          = fluid_core,
            solid_components    = solid_components["no_channeled"],
            solid_component_map = basic_solid_map,
            solid_material      = "graphite",
            core_height         = L_p1,
            n_axial_cells       = N_p1
        )

        return (core_basic, core_some_channeled, core_no_channeled)

    # BUILDING SOLID CORE COMPONENTS
    uniform_core         = _makeUniformCore(fluid_core)
    nonuniform_cores     = _makeNonUniformSolidCores(fluid_core)

    return uniform_core, nonuniform_cores

def _test_solidCoreComponent(component_name, solid_core):

    def _testUniformCore(component_name, solid_core):
        if component_name != "uniform_core":
            return

        assert isinstance(solid_core.fluidCore, Core)
        fluidCore       = solid_core.fluidCore
        components      = solid_core.components
        baseComponents  = solid_core.baseComponents
        volume          = solid_core.volume
        pipeMap         = solid_core.fluidPipeMap
        nAxialCells     = solid_core.nAxialCells
        coreHeight      = solid_core.coreHeight

        # print(f"fluidCore: \n      {fluidCore}")
        # print(f"components: \n      {components}")
        # print(f"baseComponents: \n      {baseComponents}")
        # print(f"volume: \n      {volume}")
        # print(f"pipeMap: \n      {pipeMap}")
        # print(f"nAxialCells: \n      {nAxialCells}")
        # print(f"coreHeight: \n      {coreHeight}")

        for key, comp in components.items():
            print(f"KEY: {key}")
            comp.printSummary()

        # for comp in baseComponents:
        #     comp.printSummary()

        return

    _testUniformCore(component_name, solid_core)

    return

def test_SolidCore():
    # Fluid Core Inputs
    x_pitch = 0.5 # m
    y_pitch = 0.75 # m

    # Fluid Core Component
    fluid_core = _createFluidCoreComponent(x_pitch, y_pitch)

    # Solid Core Components
    uniform_core, nonuniform_cores = _createSolidCoreComponents(fluid_core)
    nonuniform_basic_core          = nonuniform_cores[0]
    nonuniform_some_channeled_core = nonuniform_cores[1]
    nonuniform_no_channeled_core   = nonuniform_cores[2]

    # Test Solid Core Components
    _test_solidCoreComponent("uniform_core"                   , uniform_core)
    # _test_solidCoreComponent("nonuniform_basic_core"          , nonuniform_basic_core)
    # _test_solidCoreComponent("nonuniform_some_channeled_core" , nonuniform_some_channeled_core)
    # _test_solidCoreComponent("nonuniform_no_channeled_core"   , nonuniform_no_channeled_core)

    return

if __name__ == "__main__":
    test_Cuboid()
    test_CuboidWithChannel()
    test_SolidEncasedPipe()
    test_SolidCore()