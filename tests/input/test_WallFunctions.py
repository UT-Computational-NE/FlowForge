from flowforge.input.WallFunctions import *


def test_GeneralWF():

    wf_solid = GeneralWF("outer", "solid_temperature", EquationParser("700"))

    assert wf_solid.wall_function_type is None
    assert wf_solid.simulation_type == "Solid"
    assert wf_solid.surface_name == "outer"
    assert wf_solid.wall_function_value.evaluate() == 700
    assert wf_solid.variable_name == "solid_temperature"

    wf_solid.add_cell(1)
    wf_solid.add_cell(20)
    wf_solid.add_cell(11)

    assert wf_solid.associated_cells == [1, 20, 11]

    new_cells = [1, 20, 10, 15]
    wf_solid.associated_cells = new_cells
    assert wf_solid.associated_cells == [1, 20, 10, 15]


def test_HeatFluxWF():

    wf_solid = HeatFluxWF("outer", "solid_temperature", EquationParser("700"))

    assert wf_solid.wall_function_type == "HeatFluxWF"


def test_WallFunctions():

    wall_functions = {
        "temperature_wf" : {"type": "HeatFluxWF", "surface": "outer", "variable": "temperature", "value": 700},
        "enthalpy_wf" : {"type": "HeatFluxWF", "surface": "top", "variable": "enthalpy", "value": 1e6}
    }

    WF = WallFunctions(**wall_functions)

    for wf_name, wf in WF.wall_functions.items():
        assert wf.wall_function_type == wall_functions[wf_name]["type"]
        assert wf.surface_name == wall_functions[wf_name]["surface"]
        assert wf.variable_name == wall_functions[wf_name]["variable"]
        assert wf.wall_function_value.evaluate() == wall_functions[wf_name]["value"]
        if "solid" in wf_name:
            assert wf.simulation_type == "Solid"
        else:
            assert wf.simulation_type == "Fluid"


if __name__ == "__main__":
    test_GeneralWF()
    test_HeatFluxWF()

    test_WallFunctions()