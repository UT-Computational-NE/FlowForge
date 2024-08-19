from flowforge import UnitConverter


def test_defaults():
    units = {}
    uc = UnitConverter(units)
    assert uc.areaConversion == 1
    assert uc.densityConversion == 1
    assert uc.lengthConversion == 1
    assert uc.massFlowRateConversion == 1
    assert uc.powerConversion == 1
    assert uc.pressureConversion == 1
    assert uc.timeConversion == 1
    assert uc.volumeConversion == 1


def test_base_SI():
    units = {
        "length": "m",
        "volume": "m3",
        "time": "s",
        "pressure": "pa",
        "massFlowRate": "kg/s",
        "density": "kg/m3",
        "power": "w",
    }
    uc = UnitConverter(units)
    assert uc.areaConversion == 1
    assert uc.densityConversion == 1
    assert uc.lengthConversion == 1
    assert uc.massFlowRateConversion == 1
    assert uc.powerConversion == 1
    assert uc.pressureConversion == 1
    assert uc.timeConversion == 1
    assert uc.volumeConversion == 1


def test_conversions_to_SI():
    units = {
        "length": "ft",
        "volume": "ft3",
        "time": "hr",
        "pressure": "atm",
        "massFlowRate": "g/s",
        "density": "lb/in3",
        "power": "mw",
    }
    uc = UnitConverter(units)
    assert uc.areaConversion == 0.09290304
    assert uc.densityConversion == 27679
    assert uc.lengthConversion == 0.3048
    assert uc.massFlowRateConversion == 0.001
    assert uc.powerConversion == 1000000
    assert uc.pressureConversion == 101325
    assert uc.timeConversion == 3600
    assert uc.volumeConversion == 0.0283168


def test_temp_K():
    units = {"temperature": "K"}
    uc = UnitConverter(units)
    assert uc.temperatureConversion(500) == 500


def test_temp_C():
    units = {"temperature": "C"}
    uc = UnitConverter(units)
    assert uc.temperatureConversion(500) == 773.15


def test_temp_F():
    units = {"temperature": "F"}
    uc = UnitConverter(units)
    assert uc.temperatureConversion(500) == 533.15


def test_temp_R():
    units = {"temperature": "R"}
    uc = UnitConverter(units)
    assert round(uc.temperatureConversion(500), 3) == 277.778
