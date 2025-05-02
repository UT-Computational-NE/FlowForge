from flowforge.input.Components import (
    Pipe,
    Pump,
    Nozzle,
    Annulus,
    Tank,
    ParallelComponents,
    SerialComponents,
    ComponentCollection,
    HexCore,
)
from flowforge import UnitConverter

_cm2m = 0.01
unitdict = {"length": "cm"}
uc = UnitConverter(unitdict)


def test_pipe():
    p = Pipe(R=2.0, L=10.0)
    p._convertUnits(uc)
    assert p.flowArea > 0.0
    assert p.volume > 0.0
    assert p.length == 0.1
    assert p.hydraulicDiameter > 0.0
    assert p.heightChange == 0.1
    assert p.nCell == 1


def test_rectangular_pipe():
    p = Pipe(cross_section_name="rectangular", L=10, W=2.0, H=2.0)
    p._convertUnits(uc)
    assert p.flowArea == 0.0004
    assert p._Pw == 0.08
    assert p._Dh == 0.0016 / 0.08
    assert p.volume > 0.0
    assert p.length == 0.10
    assert p.hydraulicDiameter > 0.0
    assert p.nCell == 1


def test_stadium_pipe():
    p = Pipe(cross_section_name="stadium", L=12, R=3, A=2)
    p._convertUnits(uc)
    assert p.flowArea > 0.0001 * (3 * 9 + 2 * 3 * 2)
    assert p.flowArea < 0.0001 * (4 * 9 + 2 * 3 * 2)
    assert p._Pw > 0.01 * (2 * (3 * 3 + 2))
    assert p._Pw < 0.01 * (2 * (4 * 3 + 2))
    assert p.volume > 0.0
    assert p.length == 0.12
    assert p.hydraulicDiameter > 0.0
    assert p.nCell == 1


def test_pump():
    p = Pump(Ac=16.0, Dh=4.0, V=64.0, height=4.0, dP=50000.0)
    p._convertUnits(uc)
    assert p.flowArea > 0.0
    assert p.volume > 0.0
    assert p.hydraulicDiameter > 0.0
    assert p.heightChange > 0
    assert p.nCell == 1


def test_nozzle():
    n = Nozzle(L=10, R_inlet=2, R_outlet=4)
    n._convertUnits(uc)
    assert n.flowArea > 0.0
    assert n.volume > 0.0
    assert n.length == 0.1
    assert n.hydraulicDiameter > 0.0
    assert n.heightChange == 0.1
    assert n.nCell == 1
    assert n.getOutlet((0, 0, 0)) == (0, 0, 0.1)


def test_annulus():
    a = Annulus(L=10, R_inner=9, R_outer=10, n=10)
    a._convertUnits(uc)
    assert a.flowArea > 0.0
    assert a.length == 0.1
    assert a.heightChange == 0.1
    assert a.nCell == 10
    assert a.getOutlet((0, 0, 0)) == (0, 0, 0.1)


def test_tank():
    t = Tank(L=10, R=3, n=5)
    t._convertUnits(uc)
    assert t.flowArea > 0.0
    assert t.volume > 0.0
    assert t.length == 0.1
    assert t.hydraulicDiameter > 0.0
    assert t.heightChange == 0.1
    assert t.nCell == 5
    assert t.getOutlet((0, 0, 0)) == (0.03, 0, 0)


def test_parallel():
    components = {"pipe": {"p1": {"L": 10, "R": 1, "n": 10}}}
    centroids = {"p1": [0, 0]}
    lplen = {"nozzle": {"L": 1, "R_inlet": 0.5, "R_outlet": 1.2}}
    uplen = {"nozzle": {"L": 1, "R_inlet": 1.2, "R_outlet": 0.5}}
    annulus = {"annulus": {"L": 10, "R_inner": 1.1, "R_outer": 1.2, "n": 10}}
    p = ParallelComponents(components, centroids, lplen, uplen, annulus)
    p._convertUnits(uc)
    assert p._myComponents["p1"].length == 0.1
    assert p._myComponents["p1"].nCell == 10
    assert p._myComponents["p1"].hydraulicDiameter == 0.02
    for component in centroids:
        assert component in components["pipe"] or components["annulus"]
    assert p.nCell == 26
    assert p._myComponents["p1"].getOutlet((0, 0, 0)) == p._annulus.getOutlet((0, 0, 0))
    assert p._upperPlenum.getOutlet((0, 0, 0)) == (0, 0, 0.01)
    assert p.getOutlet((0, 0, 0)) == (0, 0, 0.12)


def test_serial():
    serial_dict = {"pipe": {"p1": {"L": 10, "R": 1, "n": 10}, "p2": {"L": 1, "R": 2, "n": 1, "Kloss": 1, "resolution": 6}}}
    order = ["p1", "p2"]
    s = SerialComponents(serial_dict, order)
    s._convertUnits(uc)
    assert s._myComponents["p1"].length == 0.1
    assert s._myComponents["p1"].nCell == 10
    assert s._myComponents["p1"].hydraulicDiameter == 0.02
    assert s._myComponents["p2"].length == 0.01
    assert s._myComponents["p2"].nCell == 1
    assert s._myComponents["p2"].hydraulicDiameter == 0.04
    for pipe in order:
        assert pipe in serial_dict["pipe"]
    assert s.nCell == 12
    assert s.getOutlet((0, 0, 0)) == (0, 0, 0.11)


def generate_components():
    """Generate a few components for testing methods"""
    components = {}

    components["pipe"] = Pipe(R=2.0, L=10.0)
    components["squarepipe"] = Pipe(L=12.0, cross_section_name="square", W=3)
    components["stadiumpipe"] = Pipe(L=13.0, cross_section_name="stadium", A=2, R=3.0)
    components["rectangularpipe"] = Pipe(L=8.0, cross_section_name="rectangular", H=3.0, W=5.0)
    components["pump"] = Pump(Ac=16.0, Dh=4.0, V=64.0, height=4.0, dP=50000.0)
    components["nozzle"] = Nozzle(L=10, R_inlet=2, R_outlet=4)
    components["annulus"] = Annulus(L=10, R_inner=9, R_outer=10, n=10)
    components["tank"] = Tank(L=10, R=3, n=5)

    parallel_components = {"pipe": {"p1": {"L": 10, "R": 1, "n": 10}}}
    centroids = {"p1": [0, 0]}
    lplen = {"nozzle": {"L": 1, "R_inlet": 0.5, "R_outlet": 1.2}}
    uplen = {"nozzle": {"L": 1, "R_inlet": 1.2, "R_outlet": 0.5}}
    annulus = {"annulus": {"L": 10, "R_inner": 1.1, "R_outer": 1.2, "n": 10}}
    p = ParallelComponents(parallel_components, centroids, lplen, uplen, annulus)
    components["parallel"] = p

    hexcore_components = {"pipe": {"1": {"L": 10, "R": 0.1, "n": 5}, "2": {"L": 10, "R": 0.2, "n": 1}}}
    # Modified to conform to hexagonal pattern requirements - n=2 ring: [2,3,2]
    hexmap = [[1, 1], [2, 1, 2], [1, 1]]
    lplen = {"nozzle": {"L": 1, "R_inlet": 0.5, "R_outlet": 1.2}}
    uplen = {"nozzle": {"L": 1, "R_inlet": 1.2, "R_outlet": 0.5}}
    annulus = {"annulus": {"L": 10, "R_inner": 1.1, "R_outer": 1.2, "n": 10}}
    orificing = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]]
    hc = HexCore(
        pitch=3,
        components=hexcore_components,
        channel_map=hexmap,
        lower_plenum=lplen,
        upper_plenum=uplen,
        annulus=annulus,
        orificing=orificing,
        non_channels=["0"],
    )
    components["hexcore"] = hc

    hexcore_components = {
        "serial_components": {
            "1": {
                "components": {
                    "pipe": {
                        "c1": {"L": 140.84, "R": 1.508, "n": 50},
                        "plate": {"L": 10, "R": 5.0, "n": 1, "resolution": 6, "Kloss": 0.8},
                    }
                },
                "order": ["plate", "c1"],
            },
            "2": {
                "components": {
                    "pipe": {"plate": {"L": 10, "R": 5.0, "n": 1, "resolution": 6, "Kloss": 0.8}},
                    "annulus": {"c2": {"L": 140.84, "R_inner": 1.9, "R_outer": 2.7, "n": 50}},
                },
                "order": ["plate", "c2"],
            },
        }
    }

    serial_dict = {"pipe": {"p1": {"L": 10, "R": 1, "n": 10}, "p2": {"L": 1, "R": 2, "n": 1, "Kloss": 1, "resolution": 6}}}
    order = ["p1", "p2"]
    s = SerialComponents(serial_dict, order)
    components["serial"] = s

    serial_dict = {
        "pipe": {
            "p1": {"L": 10, "R": 1, "n": 10},
            "p2": {"L": 1, "R": 2, "n": 1, "Kloss": 1, "resolution": 6},
            "p3": {"L": 10, "R": 1, "n": 10},
        }
    }
    order = ["p2", "p1", "p3"]
    s = SerialComponents(serial_dict, order)
    components["serial2"] = s

    return components


def test_baseComponents():
    """Test the component.baseComponents property"""
    components = generate_components()

    for component in components.values():
        base_components = component.baseComponents
        if not isinstance(component, ComponentCollection):
            # All leaf components should only have themselves as base components
            assert base_components == [component]

        if isinstance(component, ComponentCollection):
            # All composite components should have base components that are not component collections
            assert all(not isinstance(base_component, ComponentCollection) for base_component in base_components)

        if isinstance(component, ParallelComponents):
            assert component.lowerPlenum in base_components
            assert component.upperPlenum in base_components


def test_firstLastComponent():
    """Test the component.firstComponent and component.lastComponent properties"""
    components = generate_components()

    for component in components.values():
        if not isinstance(component, ComponentCollection):
            continue
        elif isinstance(component, ParallelComponents):
            assert component.firstComponent == component.lowerPlenum
            assert component.lastComponent == component.upperPlenum
        elif isinstance(component, SerialComponents):
            assert component.firstComponent == component._myComponents[component.order[0]]
            assert component.lastComponent == component._myComponents[component.order[-1]]


def test_orderedComponentsList():
    """Test SerialComponents.orderedComponentsList()"""
    components = generate_components()

    for component in components.values():
        if not isinstance(component, SerialComponents):
            continue
        ordered_components = component.orderedComponentsList

        component_order = component.order
        for i, comp in enumerate(ordered_components):
            assert comp == component._myComponents[component_order[i]]


if __name__ == "__main__":
    test_pipe()
    test_rectangular_pipe()
    test_stadium_pipe()
    test_pump()
    test_nozzle()
    test_annulus()
    test_tank()
    test_parallel()
    test_serial()
    test_baseComponents()
    test_firstLastComponent()
    test_orderedComponentsList()
