import os
import numpy as np
import json

from flowforge.input.System import *
from flowforge.input.Components import Component as FluidComponent
from flowforge.input.SolidComponents import Component as SolidComponent

def test_component_coupling():

    relative_path = "/testCoupledComponents/coupled_input_file.json"
    folder_path = os.path.dirname(os.path.abspath(__file__))
    input_file_path = folder_path + relative_path

    with open(input_file_path, "r") as rf:
        input_dict = json.load(rf)

    fluid_components = FluidComponent.factory(input_dict["components"])
    solid_components = SolidComponent.factory(input_dict.get("solid_components", {}))

    sys_var = System(components=fluid_components,
                     sysdict=input_dict.get("system", {}),
                     unitdict=input_dict.get("units", {}),
                     solid_components=solid_components,
                     solid_controllers=input_dict.get("solid_controllers", {}),
                     coupled_components=input_dict.get("coupled_components", {}))

    fluid_comps = sys_var.components
    solid_comps = sys_var.solidComponents
    coupled_connectivity = sys_var.coupled_components_connectivity

    for fluid_comp in fluid_comps:
        assert np.isclose(fluid_comp.flowArea, np.pi * (1.2e-2) ** 2)

    for solid_comp in solid_comps:
        assert np.isclose(solid_comp.crossSection.area, (3.4e-2 ** 2) - (np.pi * (1.2e-2) ** 2))

    assert len(coupled_connectivity) == 2
    assert coupled_connectivity[0] == (fluid_comps[1], solid_comps[0])
    assert coupled_connectivity[1] == (fluid_comps[2], solid_comps[1])

if __name__ == "__main__":
    test_component_coupling()