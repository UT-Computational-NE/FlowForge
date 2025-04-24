import json
from math import isclose

from flowforge.input.System import System
from flowforge.input.Components import Component


def test_single_segment():
    return  # This should be deleted once the test has been fully implemented

    inputfile = "testOpenFoamParser/single_segment/system.json"
    with open(inputfile, "r") as rf:
        input_dict = json.load(rf)

    components = Component.factory(input_dict["components"])

    system = System(components, input_dict.get("system", {}), input_dict.get("units", {}))

    openfoam_outputs = system.output_parsers["rho_foam"].parse()
    syth_outputs = system.output_parsers["rho_syth"].parse()

    assert len(openfoam_outputs) == len(syth_outputs)
    assert all(isclose(rho_foam, rho_syth) for rho_foam, rho_syth in zip(openfoam_outputs, syth_outputs))
