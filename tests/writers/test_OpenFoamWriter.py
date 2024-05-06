import pytest
import os
import json
from test_utils import outputs_match
from flowforge.input.System import System
from flowforge.input.Components import component_factory
from flowforge.writers.OpenfoamWriter import OpenFoamWriter

def test_single_segment():
    inputfile = "testOpenFoamWriter/single_segment/system.json"
    with open(inputfile, 'r') as rf:
        input_dict = json.load(rf)

    components = component_factory(input_dict['components'])
    system = System(components, input_dict.get('system', {}), input_dict.get('units', {}))

    writer = OpenFoamWriter()
    writer.write(system, "test")

    test_dir = "test"
    gold_dir = "testOpenFoamWriter/single_segment/gold"
    assert(outputs_match(test_dir, gold_dir))
    
    os.remove(test_dir)  