import numpy as np

from flowforge.input.ComponentCoupler import (
    couple,
    ComponentCoupler,
    UniformlyEncasedSerialPipeCoupler,
    NonUniformlyEncasedPipe,
    NonUniformlyEncasedSerialPipe,
    CartesianCore
)
import flowforge.input.Components as FluidComps
import flowforge.input.SolidComponents as SolidComps


def _buildBasicComponents():
    ## Inputs
    # General
    height = 20.0 # m
    n_cells = 20
    # Solid
    width = 1.01 # m
    solid_1 = SolidComps.Component(height=height, n_cells=n_cells, cross_section="square", W=width)
    h_2_1, h_2_2, h_2_3 = 10.0, 5.0,  5.0 # m
    n_2_1, n_2_2, n_2_3 = 10,   5,    5
    assert sum([h_2_1, h_2_2, h_2_3]) == height
    assert sum([n_2_1, n_2_2, n_2_3]) == n_cells
    solid_2_1 = SolidComps.Component(height=h_2_1, n_cells=n_2_1, cross_section="square", W=width)
    solid_2_2 = SolidComps.Component(height=h_2_2, n_cells=n_2_2, cross_section="square", W=width)
    solid_2_3 = SolidComps.Component(height=h_2_3, n_cells=n_2_3, cross_section="square", W=width)
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
            "1" : {"h": height, "n": n_cells, "l": width, "w": width},
            "2" : {
                "1": {"h": h_2_1, "n": n_2_1, "l": width, "w": width},
                "2": {"h": h_2_2, "n": n_2_2, "l": width, "w": width},
                "3": {"h": h_2_3, "n": n_2_3, "l": width, "w": width},
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


def test_ComponentCoupler():

    inputs, fluid, _, solid, _ = _buildBasicComponents()
    coupled_fluid, coupled_solid = ComponentCoupler().couple(fluid, solid)

    assert isinstance(coupled_fluid, FluidComps.Pipe)
    assert isinstance(coupled_solid, SolidComps.Component)

    base_area = coupled_solid.crossSection.baseArea
    pipe_area = coupled_fluid.flowArea
    theoretical_area = (inputs["solid"]["1"]["l"] * inputs["solid"]["1"]["w"]) - (np.pi * inputs["fluid"]["1"]["r"] ** 2)
    assert coupled_solid.crossSection.area == base_area - pipe_area
    assert coupled_solid.crossSection.area == theoretical_area


def test_UniformlyEncasedSerialPipeCoupler():

    inputs, _, fluid, solid, _ = _buildBasicComponents()
    coupled_fluid, coupled_solid = UniformlyEncasedSerialPipeCoupler().couple(fluid, solid)

    assert isinstance(coupled_fluid, FluidComps.SerialComponents)
    assert isinstance(coupled_solid, SolidComps.SerialComponent)

    count = 0
    for fluid_comp, solid_comp in zip(coupled_fluid.orderedComponentsList, coupled_solid.orderedComponents):
        if not isinstance(fluid_comp, FluidComps.Nozzle):  # If it is a nozzle, use the previous pipe area and do not
                                                           # override this variables value
            pipe_area = fluid_comp.flowArea
            count += 1
        base_area = solid_comp.crossSection.baseArea
        theoretical_area = (inputs["solid"]["1"]["l"] * inputs["solid"]["1"]["w"]) - (np.pi * inputs["fluid"]["2"][str(count)]["r"] ** 2)
        assert solid_comp.crossSection.area == base_area - pipe_area
        assert np.isclose(solid_comp.crossSection.area, theoretical_area)


def test_NonUniformlyEncasedPipe():

    inputs, fluid, _, _, solid = _buildBasicComponents()
    coupled_fluid, coupled_solid = NonUniformlyEncasedPipe().couple(fluid, solid)

    assert isinstance(coupled_fluid, FluidComps.Pipe)
    assert isinstance(coupled_solid, SolidComps.SerialComponent)

    pipe_area = coupled_fluid.flowArea
    for i, solid_comp in enumerate(coupled_solid.orderedComponents):
        base_area = solid_comp.crossSection.baseArea
        theoretical_area = (inputs["solid"]["2"][str(i+1)]["l"] * inputs["solid"]["2"][str(i+1)]["w"]) - (np.pi * inputs["fluid"]["1"]["r"] ** 2)
        assert solid_comp.crossSection.area == base_area - pipe_area
        assert np.isclose(solid_comp.crossSection.area, theoretical_area)


def test_NonUniformlyEncasedSerialPipe():

    inputs, _, fluid, _, solid = _buildBasicComponents()
    coupled_fluid, coupled_solid = NonUniformlyEncasedSerialPipe().couple(fluid, solid)

    assert isinstance(coupled_fluid, FluidComps.SerialComponents)
    assert isinstance(coupled_solid, SolidComps.SerialComponent)

    count = 0
    for fluid_comp, solid_comp in zip(coupled_fluid.orderedComponentsList, coupled_solid.orderedComponents):
        if not isinstance(fluid_comp, FluidComps.Nozzle):  # If it is a nozzle, use the previous pipe area and do not
                                                           # override this variables value
            pipe_area = fluid_comp.flowArea
            count += 1
        base_area = solid_comp.crossSection.baseArea
        i = str(count)
        theoretical_area = (inputs["solid"]["2"][i]["l"] * inputs["solid"]["2"][i]["w"]) - (np.pi * inputs["fluid"]["2"][i]["r"] ** 2)
        assert solid_comp.crossSection.area == base_area - pipe_area
        assert np.isclose(solid_comp.crossSection.area, theoretical_area)


if __name__ == "__main__":
    test_ComponentCoupler()
    test_UniformlyEncasedSerialPipeCoupler()
    test_NonUniformlyEncasedPipe()
    test_NonUniformlyEncasedSerialPipe()
