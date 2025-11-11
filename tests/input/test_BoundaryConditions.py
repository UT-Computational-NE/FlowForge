from flowforge.input.BoundaryConditions import *


def test_GeneralBC():

    bc_fluid = GeneralBC("inlet", "temperature", "700")
    bc_solid = GeneralBC("bottom", "solid_temperature", "700")

    assert bc_fluid.boundary_type is None
    assert bc_fluid.simulation_type == "Fluid"
    assert bc_fluid.boundary_value.evaluate() == 700
    assert bc_fluid.variable_name == "temperature"
    assert bc_fluid.surface_name == "inlet"

    assert bc_solid.boundary_type is None
    assert bc_solid.simulation_type == "Solid"
    assert bc_solid.boundary_value.evaluate() == 700
    assert bc_solid.variable_name == "solid_temperature"
    assert bc_solid.surface_name == "bottom"


def test_DirichletBC():

    bc_fluid = DirichletBC("inlet", "temperature", "700")
    assert bc_fluid.boundary_type == "DirichletBC"

def test_BoundaryConditions():

    boundary_conditions = {
        "inlet_mdot" : {"boundary_type": "DirichletBC", "surface": "inlet", "variable": "mass_flow_rate", "value": 25.0},
        "outlet_pressure" : {"boundary_type": "DirichletBC", "surface": "outlet", "variable": "pressure", "value": 1e5},
        "inlet_temp" : {"boundary_type": "DirichletBC", "surface": "inlet", "variable": "temperature", "value": 700},
    }

    BC = BoundaryConditions(**boundary_conditions)
    for bc_name, bc in BC.boundary_conditions.items():
        assert bc.boundary_type == boundary_conditions[bc_name]["boundary_type"]
        assert bc.surface_name == boundary_conditions[bc_name]["surface"]
        assert bc.variable_name == boundary_conditions[bc_name]["variable"]
        assert bc.boundary_value.evaluate()  == boundary_conditions[bc_name]["value"]
        if "solid" in bc_name:
            assert bc.simulation_type == "Solid"
        else:
            assert bc.simulation_type == "Fluid"


if __name__ == "__main__":
    test_GeneralBC()
    test_DirichletBC()

    test_BoundaryConditions()