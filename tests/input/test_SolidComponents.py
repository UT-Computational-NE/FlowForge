import numpy as np
from numpy import pi, isclose
from flowforge.input.SolidComponents import (
    Cuboid,
    SerialSolidComponents,
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
    circle = Cuboid(
        length, width, height, n_cells, pipe_cross_section_type="circular",
        **{"R": R})
    stadium = Cuboid(
        length, width, height, n_cells, pipe_cross_section_type="stadium",
        **{"R": R, "A": A})
    square = Cuboid(
        length, width, height, n_cells, pipe_cross_section_type="square",
        **{"W": W})
    rectangle = Cuboid(
        length, width, height, n_cells, pipe_cross_section_type="rectangular",
        **{"W": W, "H": H})

    # Base units
    # -- Cuboid with Circular Pipe
    assert circle.channel.fluidCrossSectionalArea == pi * R * R
    assert circle.channel.wettedPerimeter == 2 * pi * R
    assert circle.channel.hydraulicDiameter == _computeDh(
        circle.channel.fluidCrossSectionalArea, circle.channel.wettedPerimeter)
    assert circle.volume == _computeVolumeWithPipe(
        length, width, height, circle.channel.fluidCrossSectionalArea)
    # -- Cuboid with Stadium Pipe
    assert stadium.channel.fluidCrossSectionalArea == (pi * R * R) + (2 * R * A)
    assert stadium.channel.wettedPerimeter == 2 * pi * R + 2 * A
    assert stadium.channel.hydraulicDiameter == _computeDh(
        stadium.channel.fluidCrossSectionalArea, stadium.channel.wettedPerimeter)
    assert stadium.volume == _computeVolumeWithPipe(
        length, width, height, stadium.channel.fluidCrossSectionalArea)
    # -- Cuboid with Square Pipe
    assert square.channel.fluidCrossSectionalArea == W * W
    assert square.channel.wettedPerimeter == 4 * W
    assert square.channel.hydraulicDiameter == _computeDh(
        square.channel.fluidCrossSectionalArea, square.channel.wettedPerimeter)
    assert square.volume == _computeVolumeWithPipe(
        length, width, height, square.channel.fluidCrossSectionalArea)
    # -- Cuboid with Rectangular Pipe
    assert rectangle.channel.fluidCrossSectionalArea == W * H
    assert rectangle.channel.wettedPerimeter == 2 * W + 2 * H
    assert rectangle.channel.hydraulicDiameter == _computeDh(
        rectangle.channel.fluidCrossSectionalArea, rectangle.channel.wettedPerimeter)
    assert rectangle.volume == _computeVolumeWithPipe(
        length, width, height, rectangle.channel.fluidCrossSectionalArea)

    # Convert units (assumes cm and converts to m)
    circle._convertUnits(uc)
    stadium._convertUnits(uc)
    square._convertUnits(uc)
    rectangle._convertUnits(uc)
    # -- Cuboid with Circular Pipe
    assert isclose(circle.channel.fluidCrossSectionalArea, (0.01**2) * pi * R * R)
    assert isclose(circle.channel.wettedPerimeter, 0.01 * 2 * pi * R)
    # -- Cuboid with Stadium Pipe
    assert isclose(stadium.channel.fluidCrossSectionalArea, (0.01**2) * ((pi * R * R) + (2 * R * A)))
    assert isclose(stadium.channel.wettedPerimeter, 0.01 * (2 * pi * R + 2 * A))
    # -- Cuboid with Square Pipe
    assert isclose(square.channel.fluidCrossSectionalArea, (0.01**2) * W * W)
    assert isclose(square.channel.wettedPerimeter, 0.01 * 4 * W)
    # -- Cuboid with Rectangular Pipe
    assert isclose(rectangle.channel.fluidCrossSectionalArea, (0.01**2) * W * H)
    assert isclose(rectangle.channel.wettedPerimeter, 0.01 * (2 * W + 2 * H))

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
    channeled_cuboid = Cuboid(length, width, height, n_cells,
                              pipe_cross_section_type=CX_type, R=R)

    # NOTE: I am not sure if this is really something we will want in the future. I have added it here as
    #       a template, but when actually being used (maybe after the solid conduction and CHT is in C++),
    #       I plan to revisit this for more complex functionality. For now, I will leave this note for
    #       future reference, and the code in SolidComponents.py as a complex and mostly completed template.
    #                                                                                       - Charlie

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
            "1": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p1, n_cells=N_p1, R=R_p1,
                        pipe_cross_section_type=CX_p1),
            "2": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p2, n_cells=N_p2, R=R_p2,
                        pipe_cross_section_type=CX_p2),
            "3": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p3, n_cells=N_p3, R=R_p3, A=A_p3,
                        pipe_cross_section_type=CX_p3)}

        solid_components["some_channeled"] = {
            "1": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p1, n_cells=N_p1, R=R_p1,
                        pipe_cross_section_type=CX_p1),
            "2": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p2, n_cells=N_p2),
            "3": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p3, n_cells=N_p3)}

        solid_components["no_channeled"] = {
            "1": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p1, n_cells=N_p1),
            "2": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p2, n_cells=N_p2),
            "3": Cuboid(length=x_pitch, width=y_pitch,
                        height=L_p3, n_cells=N_p3)}

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

    assert isinstance(solid_core, SolidCore)
    assert isinstance(solid_core.fluidCore, Core)
    fluidCore       = solid_core.fluidCore
    components      = solid_core.components
    baseComponents  = solid_core.baseComponents
    pipeMap         = solid_core.fluidPipeMap

    assert pipeMap == fluidCore.channelMap

    # Tests components
    for key, comp, baseComp in zip(components.keys(),
                                    components.values(),
                                    baseComponents):
        # Ensures the components and base components are the same
        # (Only true as none of the input components are serial-components)
        assert comp.printSummary(verbose=False) == baseComp.printSummary(verbose=False)
        assert isinstance(comp, Cuboid)

        # Ensures that components built around fluid components are of
        #   correct dimensions
        fluidComp         = fluidCore.components[key]
        fluid_summary     = {"Flow Area": fluidComp.flowArea,
                                "Wetted Perimeter": fluidComp.heatedPerimeter, # pctHeated = 1.0
                                "Hydraulic Diameter": fluidComp.hydraulicDiameter,
                                "Length": fluidComp.length,
                                "Volume": fluidComp.volume}
        component_summary = comp.printSummary(verbose=False)

        # Ensures proper solid geometry
        x_pitch, y_pitch = fluidCore.xPitch, fluidCore.yPitch
        assert component_summary["Solid"]["Geometry (L x W x H)"] == (x_pitch, y_pitch, fluid_summary["Length"])
        assert component_summary["Solid"]["Material"] == "graphite"

        # Ensures proper channel geometry
        assert component_summary["Channel"]["Flow Area"] == fluid_summary["Flow Area"]
        assert component_summary["Channel"]["Wetted Perimeter"] == fluid_summary["Wetted Perimeter"]
        assert component_summary["Channel"]["Hydraulic Diameter"] == fluid_summary["Hydraulic Diameter"]
        assert component_summary["Channel"]["Volume"] == fluid_summary["Volume"]

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
    _test_solidCoreComponent("nonuniform_basic_core"          , nonuniform_basic_core)
    _test_solidCoreComponent("nonuniform_some_channeled_core" , nonuniform_some_channeled_core)
    _test_solidCoreComponent("nonuniform_no_channeled_core"   , nonuniform_no_channeled_core)

    return

if __name__ == "__main__":
    test_Cuboid()
    test_CuboidWithChannel()
    test_SolidEncasedPipe()
    test_SolidCore()