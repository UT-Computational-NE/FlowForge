import numpy as np

from flowforge.input.SolidComponents import (
    SolidCrossSection,
    Component,
    SerialComponent,
    ParallelComponent,
    Core
)
from flowforge.input.Components import (
    FluidCrossSection
)
from flowforge.input.UnitConverter import UnitConverter

_cm2m = 0.01
unit_dict = {"length": "cm"}
uc = UnitConverter(unit_dict)


def test_Component():
    # Inputs
    height = 12.5 # m
    n_cells = 5
    material = "graphite"
    cross_section = "rectangular"
    cx_height = 5.11 # m
    cx_width = 4.03 # m

    # Basic functionalities
    comp = Component(height=height, n_cells=n_cells, material=material,
                     cross_section=cross_section, H=cx_height, W=cx_width)

    assert comp.height == height
    assert comp.crossSection.area == cx_height * cx_width
    assert comp.volume == height * cx_height * cx_width
    assert comp.nCells == n_cells
    assert comp.material == material
    assert comp.baseComponents() == [comp]

    # Complex cross-section functionalities
    del comp
    comp = Component(height=height, n_cells=n_cells, material=material)
    cross_section_obj = SolidCrossSection("rectangular", H=cx_height, W=cx_width)
    radius = 1.8 # m
    pipe_cross_section = FluidCrossSection(shape="circular", R=radius)

    assert comp.crossSection is None

    comp.crossSection = cross_section_obj
    assert isinstance(comp.crossSection, SolidCrossSection)
    assert comp.crossSection.area == cx_height * cx_width

    comp.crossSection.channel = pipe_cross_section
    assert comp.crossSection.area == (cx_height * cx_width) - (np.pi * radius * radius)


def test_SerialComponent():

    # Inputs
    cx_height = 1.81 # m
    cx_width = 2.1561 # m
    height_1 = 1.45 # m
    height_2 = 1.0911 # m
    height_3 = 3.746 # m
    height_4 = 2.9099 # m
    n_cells_1 = 3
    n_cells_2 = 2
    n_cells_3 = 12
    n_cells_4 = 10
    cross_section = "rectangular"
    material = "graphite"

    order = ["1", "2", "3", "4", "3", "2", "1"]
    total_height = height_1 + height_2 + height_3 + height_4 + height_3 + height_2 + height_1
    total_cells = n_cells_1 + n_cells_2 + n_cells_3 + n_cells_4 + n_cells_3 + n_cells_2 + n_cells_1

    # Building components
    comp_1 = Component(height=height_1, n_cells=n_cells_1, material=material,
                       cross_section=cross_section, H=cx_height, W=cx_width)
    comp_2 = Component(height=height_2, n_cells=n_cells_2, material=material,
                       cross_section=cross_section, H=cx_height, W=cx_width)
    comp_3 = Component(height=height_3, n_cells=n_cells_3, material=material,
                       cross_section=cross_section, H=cx_height, W=cx_width)
    comp_4 = Component(height=height_4, n_cells=n_cells_4, material=material,
                       cross_section=cross_section, H=cx_height, W=cx_width)
    components = {"1": comp_1, "2": comp_2, "3": comp_3, "4": comp_4}
    ordered_components = [comp_1, comp_2, comp_3, comp_4, comp_3, comp_2, comp_1]

    # Build serial component
    comp = SerialComponent(components=components, order=order)

    assert comp.components.keys() == components.keys()
    assert comp.order == order
    assert all(comp1.volume == comp2.volume for comp1, comp2 in zip(comp.orderedComponents, ordered_components))
    assert all(comp1.nCells == comp2.nCells for comp1, comp2 in zip(comp.orderedComponents, ordered_components))
    assert np.isclose(comp.height, total_height)
    assert np.isclose(comp.volume, total_height * cx_height * cx_width)
    assert comp.nCells == total_cells

    inlets_and_outlets = []
    inlet = 0.0
    for h in [height_1, height_2, height_3, height_4, height_3, height_2, height_1]:
        inlets_and_outlets.append((inlet, inlet + h))
        inlet += h
    assert comp.getComponentInletAndOutlets() == inlets_and_outlets

    # Cross-section functionality
    radius = 0.67 # m
    pipe_cross_section = FluidCrossSection("circular", R=radius)
    for comp_i in comp.components.values():
        comp_i.crossSection.channel = pipe_cross_section
    assert np.isclose(comp.volume, (total_height * cx_height * cx_width) - (total_height * np.pi * radius * radius))

    # Unit conversion
    del comp
    comp = SerialComponent(components=components, order=order)
    comp._convertUnits(uc=uc)
    assert np.isclose(comp.height, total_height * _cm2m)
    assert np.isclose(comp.volume, total_height * cx_height * cx_width * (_cm2m ** 3))


def test_ParallelComponents():

    ### Inputs
    ## Heights
    # Basic
    height_0 = 12.25 # m
    # Serial #1
    height_1_1 = 10.0 # m
    height_1_2 = 2.0 # m
    height_1_3 = 0.25 # m
    # Serial #2
    height_2_1 = 6.20 # m
    height_2_2 = 6.05 # m
    # Serial #3
    height_3_1 = 5.25 # m
    height_3_2 = 5.0 # m
    height_3_3 = 1.0 # m
    height_3_4 = 1.0 # m

    ## N-cells
    # Basic
    n_cells_0_1 = 10
    n_cells_0_2 = 8
    n_cells_0_3 = 15
    n_cells_0_4 = 2
    # Serial #1
    n_cells_1_1 = 10
    n_cells_1_2 = 2
    n_cells_1_3 = 4
    # Serial #2
    n_cells_2_1 = 6
    n_cells_2_2 = 6
    # Serial #3
    n_cells_3_1 = 15
    n_cells_3_2 = 10
    n_cells_3_3 = 5
    n_cells_3_4 = 1

    ## Cross-section
    cx_height = 1.78 # m
    cx_width = 2.01 # m
    cross_section = "rectangular"

    cx_length_hex = 1.1 # m
    cross_section_hex = "hexagon"

    ## Other
    material = "graphite"

    ### Components
    ## Basic components
    comp_0_1 = Component(height=height_0, n_cells=n_cells_0_1, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_0_2 = Component(height=height_0, n_cells=n_cells_0_2, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_0_3 = Component(height=height_0, n_cells=n_cells_0_3, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_0_4 = Component(height=height_0, n_cells=n_cells_0_4, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    ## Serial components
    # Serial component #1
    comp_1_1 = Component(height=height_1_1, n_cells=n_cells_1_1, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_1_2 = Component(height=height_1_2, n_cells=n_cells_1_2, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_1_3 = Component(height=height_1_3, n_cells=n_cells_1_3, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_1 = SerialComponent(components={"1_1": comp_1_1, "1_2": comp_1_2, "1_3": comp_1_3}, order=["1_1", "1_2", "1_3"])
    # Serial component #2
    comp_2_1 = Component(height=height_2_1, n_cells=n_cells_2_1, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_2_2 = Component(height=height_2_2, n_cells=n_cells_2_2, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)
    comp_2 = SerialComponent(components={"2_1": comp_2_1, "2_2": comp_2_2}, order=["2_1", "2_2"])
    # Serial component #3
    comp_3_1 = Component(height=height_3_1, n_cells=n_cells_3_1, material=material,
                         cross_section=cross_section_hex, L=cx_length_hex)
    comp_3_2 = Component(height=height_3_2, n_cells=n_cells_3_2, material=material,
                         cross_section=cross_section_hex, L=cx_length_hex)
    comp_3_3 = Component(height=height_3_3, n_cells=n_cells_3_3, material=material,
                         cross_section=cross_section_hex, L=cx_length_hex)
    comp_3_4 = Component(height=height_3_4, n_cells=n_cells_3_4, material=material,
                         cross_section=cross_section_hex, L=cx_length_hex)
    comp_3 = SerialComponent(components={"3_1": comp_3_1, "3_2": comp_3_2, "3_3": comp_3_3, "3_4": comp_3_4},
                             order=["3_1", "3_2", "3_3", "3_3"])

    ### Building parallel component
    components = {"01": comp_0_1, "02": comp_0_2, "03": comp_0_3, "04": comp_0_4,
                  "10": comp_1, "20": comp_2, "30": comp_3}
    mapping_1 = [
        ["01", "02", "01"],
        ["02", "03", "02"],
        ["01", "02", "01"]
    ]
    mapping_2 = [
              ["01"],
        ["10", "20", "10"],
              ["02"]
    ]
    mapping_3 = [
              ["02", "03", "02"],
        ["01", "20", "30", "20", "01"],
        ["01", "20", "30", "20", "01"],
        ["01", "20", "30", "20", "01"],
              ["02", "03", "02"]
    ]

    parallel_1 = ParallelComponent(components=components, component_map=mapping_1)
    parallel_2 = ParallelComponent(components=components, component_map=mapping_2)
    parallel_3 = ParallelComponent(components=components, component_map=mapping_3)

    ### Testing
    ## Parallel #1
    error_1 = "General issues with ParallelComponents"
    assert parallel_1.componentMap == mapping_1, error_1
    assert parallel_1.volume == sum(sum(components[comp].volume for comp in row) for row in mapping_1), error_1
    assert parallel_1.nCells == sum(sum(components[comp].nCells for comp in row) for row in mapping_1), error_1

    ## Parallel #2
    error_2 = "Issues with mixing regular 'Component's and 'SerialComponent's"
    assert parallel_2.componentMap == mapping_2, error_2
    assert parallel_2.volume == sum(sum(components[comp].volume for comp in row) for row in mapping_2), error_2
    assert parallel_2.nCells == sum(sum(components[comp].nCells for comp in row) for row in mapping_2), error_2

    ## Parallel #3
    error_3 = "Issues with mixing 'Rectangular' and 'Hexagonal' component-shape types"
    assert parallel_3.componentMap == mapping_3, error_3
    assert parallel_3.volume == sum(sum(components[comp].volume for comp in row) for row in mapping_3), error_3
    assert parallel_3.nCells == sum(sum(components[comp].nCells for comp in row) for row in mapping_3), error_3

    ## Unit conversion
    parallel_1._convertUnits(uc=uc)
    parallel_2._convertUnits(uc=uc)
    parallel_3._convertUnits(uc=uc)

    assert np.isclose(parallel_1.volume, sum(sum(components[comp].volume for comp in row) for row in mapping_1) * (_cm2m ** 3))
    assert np.isclose(parallel_2.volume, sum(sum(components[comp].volume for comp in row) for row in mapping_2) * (_cm2m ** 3))
    assert np.isclose(parallel_3.volume, sum(sum(components[comp].volume for comp in row) for row in mapping_3) * (_cm2m ** 3))


def test_Core():

    # Inputs
    height = 10.11 # m
    n_cells = 10
    cx_height = 1.12 # m
    cx_width = 2.01 # m
    cross_section = "rectangular"
    material = "graphite"

    comp = Component(height=height, n_cells=n_cells, material=material,
                     cross_section=cross_section, H=cx_height, W=cx_width)
    components = {"1": comp}
    mapping = [
             ["1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
             ["1", "1", "1"]]

    core = Core(components=components, component_map=mapping)

    assert core.coreHeight == height
    assert core.nAxialCells == n_cells
    assert core.volume == sum(sum(components[comp].volume for comp in row) for row in mapping)

    core._convertUnits(uc=uc)
    assert np.isclose(core.coreHeight, height * _cm2m)
    assert np.isclose(core.volume, sum(sum(components[comp].volume for comp in row) for row in mapping) * (_cm2m ** 3))

    bad_comp = Component(height=height+1, n_cells=n_cells, material=material,
                         cross_section=cross_section, H=cx_height, W=cx_width)

    try:
        core._geometryCheck({"1": comp, "2": bad_comp}, [["1"], ["1"]])
        raise Exception("Issue in '_geometryCheck' method")
    except AssertionError:
        pass

    try:
        core._componentTypeCheck({"1": comp, "2": bad_comp, "core": core})
        raise Exception("Issue in '_componentTypeCheck' method")
    except AssertionError:
        pass


    return


if __name__ == "__main__":
    test_Component()
    test_SerialComponent()
    test_ParallelComponents()
    test_Core()
