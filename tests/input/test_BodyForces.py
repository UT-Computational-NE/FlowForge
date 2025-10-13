from flowforge.input.BodyForces import *


def test_GeneralBF():

    bf_solid = GeneralBF("solid_temperature", EquationParser("700"))

    assert bf_solid.body_force_type is None
    assert bf_solid.simulation_type == "Solid"
    assert bf_solid.body_force_value.evaluate() == 700
    assert bf_solid.variable_name == "solid_temperature"

    bf_solid.add_cell(1)
    bf_solid.add_cell(20)
    bf_solid.add_cell(11)

    assert bf_solid.associated_cells == [1, 20, 11]

    new_cells = [1, 20, 10, 15]
    bf_solid.associated_cells = new_cells
    assert bf_solid.associated_cells == [1, 20, 10, 15]


def test_InternalHeatGenerationBF():

    bf_solid = InternalHeatGenerationBF("solid_temperature", EquationParser("700"))

    assert bf_solid.body_force_type == "InternalHeatGenerationBF"


def test_BodyForces():

    body_forces = {
        "temperature_bf" : {"type": "InternalHeatGenerationBF", "variable": "temperature", "value": 700},
        "enthalpy_bf" : {"type": "InternalHeatGenerationBF", "variable": "enthalpy", "value": 1e6}
    }

    BF = BodyForces(**body_forces)

    for bf_name, bf in BF.body_forces.items():
        assert bf.body_force_type == body_forces[bf_name]["type"]
        assert bf.variable_name == body_forces[bf_name]["variable"]
        assert bf.body_force_value.evaluate() == body_forces[bf_name]["value"]
        if "solid" in bf_name:
            assert bf.simulation_type == "Solid"
        else:
            assert bf.simulation_type == "Fluid"

    return


if __name__ == "__main__":
    test_GeneralBF()
    test_InternalHeatGenerationBF()

    test_BodyForces()