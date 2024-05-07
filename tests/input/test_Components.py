import flowforge.input.Components as comp
from flowforge import UnitConverter

_cm2m = 0.01
unitdict = {"length": "cm"}
uc = UnitConverter(unitdict)


def test_pipe():
    p = comp.Pipe(R=2.0, L=10.0)
    p._convertUnits(uc)
    assert p.flowArea > 0.0
    assert p.volume > 0.0
    assert p.length == 0.1
    assert p.hydraulicDiameter > 0.0
    assert p.heightChange == 0.1
    assert p.nCell == 1


def test_pump():
    p = comp.Pump(Ac=16.0, Dh=4.0, V=64.0, height=4.0, dP=50000.0)
    p._convertUnits(uc)
    assert p.flowArea > 0.0
    assert p.volume > 0.0
    assert p.hydraulicDiameter > 0.0
    assert p.heightChange > 0
    assert p.nCell == 1


def test_nozzle():
    n = comp.Nozzle(L=10, R_inlet=2, R_outlet=4)
    n._convertUnits(uc)
    assert n.flowArea > 0.0
    assert n.volume > 0.0
    assert n.length == 0.1
    assert n.hydraulicDiameter > 0.0
    assert n.heightChange == 0.1
    assert n.nCell == 1
    assert n.getOutlet((0, 0, 0)) == (0, 0, 0.1)


def test_annulus():
    a = comp.Annulus(L=10, R_inner=9, R_outer=10, n=10)
    a._convertUnits(uc)
    assert a.flowArea > 0.0
    assert a.length == 0.01
    assert a.heightChange == 0.01
    assert a.nCell == 10
    assert a.getOutlet((0, 0, 0)) == (0, 0, 0.1)


def test_tank():
    t = comp.Tank(L=10, R=3, n=5)
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
    p = comp.ParallelComponents(components, centroids, lplen, uplen, annulus)
    p._convertUnits(uc)
    assert p._myComponents["p1"].length == 0.01
    assert p._myComponents["p1"].nCell == 10
    assert p._myComponents["p1"].hydraulicDiameter == 0.02
    for component in centroids:
        assert component in components["pipe"] or components["annulus"]
    assert p.nCell == 22
    assert p._myComponents["p1"].getOutlet((0, 0, 0)) == p._annulus.getOutlet((0, 0, 0))
    assert p._upperPlenum.getOutlet((0, 0, 0)) == (0, 0, 0.01)
    assert p.getOutlet((0, 0, 0)) == (0, 0, 0.12)


def test_hexcore():
    components = {"pipe": {"1": {"L": 10, "R": 0.1, "n": 5}, "2": {"L": 10, "R": 0.2, "n": 1}}}
    hexmap = [[1, 1, 1, 1], [2, 1, 1, 1, 2], [1, 1, 1, 1]]
    lplen = {"nozzle": {"L": 1, "R_inlet": 0.5, "R_outlet": 1.2}}
    uplen = {"nozzle": {"L": 1, "R_inlet": 1.2, "R_outlet": 0.5}}
    annulus = {"annulus": {"L": 10, "R_inner": 1.1, "R_outer": 1.2, "n": 10}}
    hc = comp.HexCore(pitch=3, components=components, hexmap=hexmap, lower_plenum=lplen, upper_plenum=uplen, annulus=annulus)
    hc._convertUnits(uc)
    assert hc.nCell == 69
    assert hc._pitch == 0.03
    assert hc._map == hexmap
    assert hc.getOutlet((0, 0, 0)) == (0, 0, 12 * _cm2m)


def test_serial():
    serial_dict = {"pipe": {"p1": {"L": 10, "R": 1, "n": 10}, "p2": {"L": 1, "R": 2, "n": 1, "Kloss": 1, "resolution": 6}}}
    order = ["p1", "p2"]
    s = comp.SerialComponents(serial_dict, order)
    s._convertUnits(uc)
    assert s._myComponents["p1"].length == 0.01
    assert s._myComponents["p1"].nCell == 10
    assert s._myComponents["p1"].hydraulicDiameter == 0.02
    assert s._myComponents["p2"].length == 0.01
    assert s._myComponents["p2"].nCell == 1
    assert s._myComponents["p2"].hydraulicDiameter == 0.04
    for pipe in order:
        assert pipe in serial_dict["pipe"]
    assert s.nCell == 11
    assert s.getOutlet((0, 0, 0)) == (0, 0, 0.11)