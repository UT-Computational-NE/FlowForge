import numpy as np
from flowforge.input.BodyForces import *
from flowforge.input.System import SimulationType
from flowforge.input.Components import Pipe

def test_GeneralBF():

    bf = GeneralBodyForce(EquationParser("700"))

    assert bf.body_force_type is None
    assert bf.body_force_value.evaluate() == 700
    assert bf.variable_name == None
    assert bf.component == None

def test_InternalHeatGenerationBF():

    bf = InternalHeatGenerationBF(EquationParser("1e6"))

    assert bf.body_force_type == "InternalHeatGeneration"
    assert bf.variable_name == "power_density"

def test_DifferentialPressureBF():

    bf = DifferentialPressureBF(EquationParser("1e5"))

    assert bf.body_force_type == "DifferentialPressure"
    assert bf.variable_name == "pressure"

def test_attachComponent():

    bf = GeneralBodyForce(EquationParser("700"))
    comp = Pipe(L=1.0, cross_section_name="circular", R=1.0, n=10)
    bf.attach_component(comp)
    assert bf.component.length == 1.0

def test_convertUnits():

    bf = InternalHeatGenerationBF(EquationParser("1.0"))
    units = units = {"length": "cm", "power": "w"}
    uc = UnitConverter(units)
    bf.convertUnits(uc)
    assert np.isclose(bf.body_force_value.evaluate(), 1e6), bf.body_force_value.evaluate()

def test_BodyForces():

    body_forces = {
        "power_bf" : {"type": "InternalHeatGeneration", "value": 1e6},
        "pressure_bf" : {"type": "DifferentialPressure", "value": 1e5}
    }
    comp = Pipe(L=1.0, cross_section_name="circular", R=1.0, n=10)

    BF = BodyForces(SimulationType.FLUID)
    for name, bf_def in body_forces.items():
        BF.addBodyForce(name, bf_def, comp)

    assert BF.body_forces["power_bf_0"].body_force_type == "InternalHeatGeneration"
    assert BF.body_forces["power_bf_0"].variable_name == "power_density"
    assert BF.body_forces["power_bf_0"].body_force_value.evaluate() == 1e6
    assert BF.body_forces["power_bf_0"].component == comp

    assert BF.body_forces["pressure_bf_0"].body_force_type == "DifferentialPressure"
    assert BF.body_forces["pressure_bf_0"].variable_name == "pressure"
    assert BF.body_forces["pressure_bf_0"].body_force_value.evaluate() == 1e5
    assert BF.body_forces["pressure_bf_0"].component == comp


if __name__ == "__main__":
    test_GeneralBF()
    test_InternalHeatGenerationBF()
    test_DifferentialPressureBF()

    test_attachComponent()
    test_convertUnits()

    test_BodyForces()