from flowforge.input.WallFunctions import *


def test_GeneralWF():

    wf_solid = GeneralWF("outer", "solid_temperature", EquationParser("700"))

    assert wf_solid.wall_function_type is None
    assert wf_solid.surface_name == "outer"
    assert wf_solid.wall_function_value.evaluate() == 700
    assert wf_solid.variable_name == "solid_temperature"


def test_HeatFluxWF():

    wf_solid = HeatFluxWF("outer", "solid_temperature", EquationParser("700"))

    assert wf_solid.wall_function_type == "heat_flux"


def test_WallFunctions():

    wall_functions = {
        "temperature_wf" : {"type": "heat_flux", "surface": "outer", "variable": "temperature", "value": 700},
        "enthalpy_wf" : {"type": "heat_flux", "surface": "top", "variable": "enthalpy", "value": 1e6}
    }

    WF = WallFunctions(**wall_functions)

    for wf_name, wf in WF.wall_functions.items():
        assert wf.wall_function_type == wall_functions[wf_name]["type"]
        assert wf.surface_name == wall_functions[wf_name]["surface"]
        assert wf.variable_name == wall_functions[wf_name]["variable"]
        assert wf.wall_function_value.evaluate() == wall_functions[wf_name]["value"]


if __name__ == "__main__":
    test_GeneralWF()
    test_HeatFluxWF()

    test_WallFunctions()