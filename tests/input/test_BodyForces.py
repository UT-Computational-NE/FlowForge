from flowforge.input.BodyForces import *


def test_GeneralBF():

    bf_solid = GeneralBF("solid_temperature", EquationParser("700"))

    assert bf_solid.body_force_type is None
    assert bf_solid.body_force_value.evaluate() == 700
    assert bf_solid.variable_name == "solid_temperature"


def test_InternalHeatGenerationBF():

    bf_solid = HeatGenerationBF("solid_temperature", EquationParser("700"))

    assert bf_solid.body_force_type == "heat_generation"


def test_BodyForces():

    body_forces = {
        "temperature_bf" : {"type": "heat_generation", "variable": "temperature", "value": 700},
        "enthalpy_bf" : {"type": "heat_generation", "variable": "enthalpy", "value": 1e6}
    }

    BF = BodyForces(**body_forces)

    for bf_name, bf in BF.body_forces.items():
        assert bf.body_force_type == body_forces[bf_name]["type"]
        assert bf.variable_name == body_forces[bf_name]["variable"]
        assert bf.body_force_value.evaluate() == body_forces[bf_name]["value"]

    return


if __name__ == "__main__":
    test_GeneralBF()
    test_InternalHeatGenerationBF()

    test_BodyForces()