import numpy as np

from flowforge.input.CoupledComponentInterface import *
import flowforge.input.Components as FluidComps
import flowforge.input.SolidComponents as SolidComps

def _buildBasicComponents():
    ## Inputs
    # General
    height = 20.0 # m
    n_cells = 20
    # Solid
    length = 1.01 # m
    solid_1 = SolidComps.Component(height=height, n_cells=n_cells, cross_section="square", length=length)
    h_2_1, h_2_2, h_2_3 = 10.0, 5.0,  5.0 # m
    n_2_1, n_2_2, n_2_3 = 10,   5,    5
    assert sum([h_2_1, h_2_2, h_2_3]) == height
    assert sum([n_2_1, n_2_2, n_2_3]) == n_cells
    solid_2_1 = SolidComps.Component(height=h_2_1, n_cells=n_2_1, cross_section="square", length=length)
    solid_2_2 = SolidComps.Component(height=h_2_2, n_cells=n_2_2, cross_section="square", length=length)
    solid_2_3 = SolidComps.Component(height=h_2_3, n_cells=n_2_3, cross_section="square", length=length)
    solid_2 = SolidComps.SerialComponent(components={"1": solid_2_1, "2": solid_2_2, "3": solid_2_3},
                                         order=["1", "2", "3"])
    # Fluid
    radius = 0.25 # m
    fluid_1 = FluidComps.Pipe(L=height, cross_section_name="circular", n=n_cells, R=radius)
    h_2_1, h_2_2, h_2_3 = 10.0, 5.0,  5.0 # m
    n_2_1, n_2_2, n_2_3 = 10,   5,    5
    r_2_1, r_2_2, r_2_3 = 0.25, 0.01, 0.35 # m
    assert sum([h_2_1, h_2_2, h_2_3]) == height
    assert sum([n_2_1, n_2_2, n_2_3]) == n_cells
    fluid_2_1 = FluidComps.Pipe(L=h_2_1, n=n_2_1, cross_section_name="circular", R=r_2_1)
    fluid_2_2 = FluidComps.Pipe(L=h_2_2, n=n_2_2, cross_section_name="circular", R=r_2_2)
    fluid_2_3 = FluidComps.Pipe(L=h_2_3, n=n_2_3, cross_section_name="circular", R=r_2_3)
    fluid_2 = FluidComps.SerialComponents(components={"1": fluid_2_1, "2": fluid_2_2, "3": fluid_2_3},
                                          order=["1", "2", "3"])

    inputs = {
        "solid" : {
            "1" : {"h": height, "n": n_cells, "l": length, "w": length},
            "2" : {
                "1": {"h": h_2_1, "n": n_2_1, "l": length, "w": length},
                "2": {"h": h_2_2, "n": n_2_2, "l": length, "w": length},
                "3": {"h": h_2_3, "n": n_2_3, "l": length, "w": length},
            }
        },
        "fluid" : {
            "1" : {"h": height, "n": n_cells, "r": radius},
            "2" : {
                "1": {"h": h_2_1, "n": n_2_1, "r": r_2_1},
                "2": {"h": h_2_2, "n": n_2_2, "r": r_2_2},
                "3": {"h": h_2_3, "n": n_2_3, "r": r_2_3},
            }
        },
    }

    return inputs, fluid_1, fluid_2, solid_1, solid_2


def test_CoupledComponentInterfaceMethods():

    inputs, fluid_1, fluid_2, solid_1, solid_2 = _buildBasicComponents()
    solid_base_area = inputs["solid"]["1"]["l"] * inputs["solid"]["1"]["w"]
    fluid_area_1 = np.pi * ((inputs["fluid"]["1"]["r"]) ** 2)
    fluid_area_2_1 = np.pi * ((inputs["fluid"]["2"]["1"]["r"]) ** 2)
    fluid_area_2_2 = np.pi * ((inputs["fluid"]["2"]["2"]["r"]) ** 2)
    fluid_area_2_3 = np.pi * ((inputs["fluid"]["2"]["3"]["r"]) ** 2)

    ## Build CCI
    cci_simple = CoupledComponentInterface(fluid_component=fluid_1, solid_component=solid_1)

    ## coupleSingularComponents Method
    coupled_solid_1 = cci_simple._coupleSingularComponents(fluid_1, solid_1)
    coupled_solid_2_1 = cci_simple._coupleSingularComponents(fluid_2.myComponents["1"], solid_2.components["1"])
    coupled_solid_2_2 = cci_simple._coupleSingularComponents(fluid_2.myComponents["2"], solid_2.components["2"])
    coupled_solid_2_3 = cci_simple._coupleSingularComponents(fluid_2.myComponents["3"], solid_2.components["3"])

    assert coupled_solid_1.crossSection.area == solid_base_area - fluid_area_1
    assert coupled_solid_2_1.crossSection.area == solid_base_area - fluid_area_2_1
    assert coupled_solid_2_2.crossSection.area == solid_base_area - fluid_area_2_2
    assert coupled_solid_2_3.crossSection.area == solid_base_area - fluid_area_2_3

    del coupled_solid_1, coupled_solid_2_1, coupled_solid_2_2, coupled_solid_2_3

    ## breakUpSingularSolidComponent Method
    broken_solid = cci_simple._breakUpSingularSolidComponent(fluid_2, solid_1)
    assert type(broken_solid) == SolidComps.SerialComponent
    for solid_name, fluid_name in zip(broken_solid.order, fluid_2.order):
        solid_comp = broken_solid.components[solid_name]
        fluid_comp = fluid_2.myComponents[fluid_name]
        assert solid_comp.height == fluid_comp.length
        assert solid_comp.nCells == fluid_comp.nCell
        assert np.isclose(solid_comp.volume, solid_1.volume * (solid_comp.height / solid_1.height))

    del broken_solid

    ## coupleSerialComponents Method
    coupled_serial_solid_type1 = cci_simple._coupleSerialComponents(fluid_2, cci_simple._breakUpSingularSolidComponent(fluid_2, solid_1))
    coupled_serial_solid_type2 = cci_simple._coupleSerialComponents(fluid_2, solid_2)

    # Type 1
    check_name = "temp_nozzle_for_make_continuous_creation_in_serialcomp"
    assert len(coupled_serial_solid_type1.components) == len(fluid_2.myComponents)
    for i, (solid_name, fluid_name) in enumerate(zip(coupled_serial_solid_type1.order, fluid_2.order)):
        solid_comp = coupled_serial_solid_type1.components[solid_name]
        fluid_comp = fluid_2.myComponents[fluid_name]
        assert solid_comp.height == fluid_comp.length
        assert solid_comp.nCells == fluid_comp.nCell
        if check_name not in fluid_name:
            assert solid_comp.crossSection.area == solid_comp.crossSection.baseArea - fluid_comp.flowArea
        else:
            assert solid_comp.crossSection.area == solid_comp.crossSection.baseArea - fluid_2.myComponents[fluid_2.order[i-1]].flowArea

    # Type 2
    assert len(coupled_serial_solid_type2.components) == len(fluid_2.myComponents)
    for i, (solid_name, fluid_name) in enumerate(zip(coupled_serial_solid_type2.order, fluid_2.order)):
        solid_comp = coupled_serial_solid_type2.components[solid_name]
        fluid_comp = fluid_2.myComponents[fluid_name]
        assert solid_comp.height == fluid_comp.length
        assert solid_comp.nCells == fluid_comp.nCell
        if check_name not in fluid_name:
            assert solid_comp.crossSection.area == solid_comp.crossSection.baseArea - fluid_comp.flowArea
        else:
            assert solid_comp.crossSection.area == solid_comp.crossSection.baseArea - fluid_2.myComponents[fluid_2.order[i-1]].flowArea


def test_CoupledComponentInterface():

    inputs, fluid_1, fluid_2, solid_1, solid_2 = _buildBasicComponents()
    check_name = "temp_nozzle_for_make_continuous_creation_in_serialcomp"

    ## Build all CCIs
    cci_f1_s1 = CoupledComponentInterface(fluid_component=fluid_1, solid_component=solid_1)
    cci_f2_s1 = CoupledComponentInterface(fluid_component=fluid_2, solid_component=solid_1)
    cci_f1_s2 = CoupledComponentInterface(fluid_component=fluid_1, solid_component=solid_2)
    cci_f2_s2 = CoupledComponentInterface(fluid_component=fluid_2, solid_component=solid_2)

    # Fluid type 1, Solid type 1
    fluid_component, solid_component = cci_f1_s1.buildCoupledComponents()
    assert type(fluid_component) == FluidComps.Pipe
    assert type(solid_component) == SolidComps.Component
    assert not isinstance(fluid_component, FluidComps.SerialComponents)
    assert not isinstance(solid_component, SolidComps.SerialComponent)
    desired_area = (inputs["solid"]["1"]["l"] * inputs["solid"]["1"]["w"]) - (np.pi * inputs["fluid"]["1"]["r"] * inputs["fluid"]["1"]["r"])
    assert solid_component.crossSection.area == desired_area

    del fluid_component, solid_component, desired_area

    ## Fluid type 2, Solid type 1
    fluid_component, solid_component = cci_f2_s1.buildCoupledComponents()
    assert type(fluid_component) == FluidComps.SerialComponents
    assert type(solid_component) == SolidComps.SerialComponent
    for i, (fluid_name, solid_name) in enumerate(zip(fluid_component.order, solid_component.order)):
        fluid_comp = fluid_component.myComponents[fluid_name]
        solid_comp = solid_component.components[solid_name]
        assert fluid_comp.length == solid_comp.height
        assert fluid_comp.nCell == solid_comp.nCells
        if check_name in fluid_name:
            desired_area = (inputs["solid"]["1"]["l"] * inputs["solid"]["1"]["w"]) - (np.pi * (inputs["fluid"]["2"][fluid_component.order[i-1]]["r"] ** 2))
        else:
            desired_area = (inputs["solid"]["1"]["l"] * inputs["solid"]["1"]["w"]) - (np.pi * (inputs["fluid"]["2"][fluid_name]["r"] ** 2))
        assert solid_comp.crossSection.area == desired_area

    del fluid_component, solid_component, desired_area

    ## Fluid type 1, Solid type 2
    fluid_component, solid_component = cci_f1_s2.buildCoupledComponents()
    assert type(fluid_component) == FluidComps.Pipe
    assert type(solid_component) == SolidComps.SerialComponent
    pipe_area = np.pi * inputs["fluid"]["1"]["r"] * inputs["fluid"]["1"]["r"]
    for solid_name, solid_comp in solid_component.components.items():
        desired_area = (inputs["solid"]["2"][solid_name]["l"] * inputs["solid"]["2"][solid_name]["w"]) - pipe_area
        assert solid_comp.crossSection.area == desired_area
        assert solid_comp.height == inputs["solid"]["2"][solid_name]["h"]
        assert solid_comp.nCells == inputs["solid"]["2"][solid_name]["n"]

    del fluid_component, solid_component, pipe_area, desired_area

    ## Fluid type 2, Solid type 2
    fluid_component, solid_component = cci_f2_s2.buildCoupledComponents()
    assert type(fluid_component) == FluidComps.SerialComponents
    assert type(solid_component) == SolidComps.SerialComponent
    for i, (fluid_name, solid_name) in enumerate(zip(fluid_component.order, solid_component.order)):
        fluid_comp = fluid_component.myComponents[fluid_name]
        solid_comp = solid_component.components[solid_name]
        assert fluid_comp.length == solid_comp.height
        assert fluid_comp.nCell == solid_comp.nCells
        if check_name in fluid_name:
            pipe_area = np.pi * (inputs["fluid"]["2"][fluid_component.order[i-1]]["r"] ** 2)
            desired_area = (inputs["solid"]["2"][solid_component.order[i-1]]["l"] * inputs["solid"]["2"][solid_component.order[i-1]]["w"]) - pipe_area
        else:
            pipe_area = np.pi * (inputs["fluid"]["2"][fluid_name]["r"] ** 2)
            desired_area = (inputs["solid"]["2"][solid_name]["l"] * inputs["solid"]["2"][solid_name]["w"]) - pipe_area
        assert solid_comp.crossSection.area == desired_area

def _buildCoreComponents(inputs, fluid_components: dict, solid_components: dict):

    # Fluid Core
    fluid_map = [
             ["1", "2"],
        ["1", "2", "1", "2"]
    ]
    x_pitch = inputs["solid"]["1"]["l"]
    y_pitch = inputs["solid"]["1"]["w"]
    plenum_radius = 2.5 * max(x_pitch, y_pitch)
    lower_plenum = {"nozzle": {"L": 1.0, "R_inlet": 0.1*plenum_radius, "R_outlet": plenum_radius}}
    upper_plenum = {"pipe": {"L": 1.0, "cross_section_name": "circular", "R": plenum_radius}}

    fluid_core = FluidComps.CartCore(x_pitch=x_pitch, y_pitch=y_pitch,
                                     components=fluid_components, channel_map=fluid_map,
                                     lower_plenum=lower_plenum, upper_plenum=upper_plenum)

    # Solid Core
    solid_map = [
             ["1", "1"],
        ["2", "2", "2", "2"]
    ]

    solid_core = SolidComps.Core(components=solid_components, component_map=solid_map)

    return fluid_core, solid_core

def test_CoupledCoreComponentInterface():
    inputs, fluid_1, fluid_2, solid_1, solid_2 = _buildBasicComponents()
    fluid_components = {"1": fluid_1, "2": fluid_2}
    solid_components = {"1": solid_1, "2": solid_2}

    fluid_core, solid_core = _buildCoreComponents(inputs, fluid_components, solid_components)
    coreCCI = CoupledCoreComponentInterface(fluid_core_component=fluid_core, solid_core_component=solid_core)

    # Test '_buildMapping'
    mapping = coreCCI._buildMapping(fluid_map=fluid_core.componentMap, solid_map=solid_core.componentMap)
    for f_map, s_map, c_map in zip(fluid_core.componentMap, solid_core.componentMap, mapping):
        for f_i, s_i, c_i in zip(f_map, s_map, c_map):
            assert c_i == (f_i, s_i)

    # Test 'buildCoupledComponents'
    coupled_fluid_components, coupled_solid_components = coreCCI.buildCoupledComponents()

    for row in mapping:
        for element in row:
            fluid_key = element[0]
            solid_key = element[1]
            coupled_key =  fluid_key + "_" + solid_key

            fluid_comp = coupled_fluid_components[fluid_key]
            solid_comp = coupled_solid_components[coupled_key]

            if (fluid_key == "1") and (solid_key == "1"):
                assert type(fluid_comp) == FluidComps.Pipe
                assert type(solid_comp) == SolidComps.Component
                assert fluid_comp.length == solid_comp.height
                assert fluid_comp.nCell == solid_comp.nCells
                assert solid_comp.crossSection.area == solid_comp.crossSection.baseArea - fluid_comp.flowArea
            if (fluid_key == "1") and (solid_key == "2"):
                assert type(fluid_comp) == FluidComps.Pipe
                assert type(solid_comp) == SolidComps.SerialComponent
                assert all(comp.crossSection.area == comp.crossSection.baseArea - fluid_comp.flowArea
                           for comp in solid_comp.orderedComponents)
            if (fluid_key == "2") and (solid_key == "1"):
                assert type(fluid_comp) == FluidComps.SerialComponents
                assert type(solid_comp) == SolidComps.SerialComponent
                for i, (fluid_name, solid_name) in enumerate(zip(fluid_comp.order, solid_comp.order)):
                    fluid_comp_i = fluid_comp.myComponents[fluid_name]
                    solid_comp_i = solid_comp .components[solid_name]
                    assert fluid_comp_i.length == solid_comp_i.height
                    assert fluid_comp_i.nCell == solid_comp_i.nCells
                    if type(fluid_comp_i) == FluidComps.Nozzle:
                        desired_area = solid_comp_i.crossSection.baseArea - fluid_comp.myComponents[fluid_comp.order[i-1]].flowArea
                    else:
                        desired_area = solid_comp_i.crossSection.baseArea - fluid_comp_i.flowArea
                    assert solid_comp_i.crossSection.area == desired_area
            if (fluid_key == "2") and (solid_key == "2"):
                assert type(fluid_comp) == FluidComps.SerialComponents
                assert type(solid_comp) == SolidComps.SerialComponent
                for i, (fluid_name, solid_name) in enumerate(zip(fluid_comp.order, solid_comp.order)):
                    fluid_comp_i = fluid_comp.myComponents[fluid_name]
                    solid_comp_i = solid_comp .components[solid_name]
                    assert fluid_comp_i.length == solid_comp_i.height
                    assert fluid_comp_i.nCell == solid_comp_i.nCells
                    if type(fluid_comp_i) == FluidComps.Nozzle:
                        desired_area = solid_comp_i.crossSection.baseArea - fluid_comp.myComponents[fluid_comp.order[i-1]].flowArea
                    else:
                        desired_area = solid_comp_i.crossSection.baseArea - fluid_comp_i.flowArea
                    assert solid_comp_i.crossSection.area == desired_area


if __name__ == "__main__":
    test_CoupledComponentInterfaceMethods()
    test_CoupledComponentInterface()
    test_CoupledCoreComponentInterface()
