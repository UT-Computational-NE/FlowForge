from typing import Dict

# enthalpy to j/kg
enthalpydict = {"j/kg": 1.0}

# length to m
lendict = {"in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mm": 0.001, "cm": 0.01, "m": 1.0}

# volume in m3
voldict = {
    "ml": 1.0e-6,
    "l": 1.0e-3,
    "gal": 0.00378541,
    "qt": 0.000946353,
    "pt": 0.000437167,
    "c": 0.00024,
    "ft3": 0.0283168,
    "in3": 0.000016387,
    "m3": 1.0,
    "cm3": 1.0e-6,
}

# time to s
timedict = {"week": 604800, "hr": 3600, "min": 60, "ms": 0.001, "s": 1.0}

# pressure to Pa
presdict = {"kpa": 0.001, "pa": 1.0, "psi": 6894.76, "bar": 1e5, "atm": 101325, "torr": 133.322}

# mass flow rate to kg/s
mfrdict = {
    "kg/s": 1.0,
    "g/s": 0.001,
    "g/min": 0.00001667,
    "kg/min": 0.01667,
    "kg/hr": 0.0002778,
    "lb/s": 0.453592,
    "lb/min": 0.00755987,
    "lb/hr": 0.000125998,
}

# density to kg/m3
densdict = {"kg/m3": 1.0, "kg/cm3": 1.0e6, "g/cm3": 1.0e3, "g/m3": 1.0e-3, "lb/in3": 27679, "lb/ft3": 16.0185}

# power to W
powerdict = {"w": 1.0, "kw": 1.0e3, "mw": 1.0e6, "gw": 1.0e9, "tw": 1.0e12}


class UnitConverter:
    """Class for handling unit conversions for FlowForge Systems and Components

    On initialization, the unit conversion object starts by setting the
    default of all unit conversions to be equal to 1.0.  Using specifications
    dictionary, the unit converter checks the dictionary for each unit key.
    If the key is found, the unit conversion variable is then set by matching
    the unit given to the conversion from the appropriate :mod:`UnitConverter`
    conversion dictionaries.  These numbers are the unit conversions to be multiplied
    to return the values in SI units.

    If the unit given is not in our defined dictionaries, it will throw an exception
    for an unknown input unit type.

    Parameters
    ----------
    unitdict : Dict[str, str]
        Dictionary containing the unit names as the keys and the units that are used
        for the system / component inputs

    Attributes
    ----------
    lengthConversion : float
        The multiplier to convert length to units of :math:`m`
    areaConversion : float
        The multiplier to convert area to units of :math:`m^2`
    volumeConversion : float
        The multiplier to convert volume to units of :math:`m^3`
    timeConversion : float
        The multiplier to convert time to units of :math:`s`
    pressureConversion : float
        The multiplier to convert pressure to units of :math:`Pa`
    massFlowRateConversion : float
        The multiplier to convert mass flow rate to units of :math:`kg/s`
    densityConversion : float
        The multiplier to convert density to units of :math:`kg/(m^3)`
    powerConversion :float
        The multiplier to convert power to units of :math:`W`
    enthalpyConversion :float
        The multiplier to convert enthalpy to units of :math:`J/kg`
    """

    def __init__(self, unitdict: Dict[str, str]) -> None:
        self._lenconv = 1.0
        if "length" in unitdict:
            # converting to m
            if unitdict["length"].lower() in lendict:
                self._lenconv = lendict[unitdict["length"].lower()]
            else:
                raise Exception("Unknown length input type: " + unitdict["length"])

        self._volconv = self._lenconv * self._lenconv * self._lenconv
        if "volume" in unitdict:
            # converting to m3
            if unitdict["volume"].lower() in voldict:
                self._volconv = voldict[unitdict["volume"].lower()]
            else:
                raise Exception("Unknown volume input type: " + unitdict["volume"])

        self._timeconv = 1.0
        if "time" in unitdict:
            # converting to s
            if unitdict["time"].lower() in timedict:
                self._timeconv = timedict[unitdict["time"].lower()]
            else:
                raise Exception("Unknown time input type: " + unitdict["time"])

        self._presconv = 1.0
        if "pressure" in unitdict:
            # converting to Pa
            if unitdict["pressure"].lower() in presdict:
                self._presconv = presdict[unitdict["pressure"].lower()]
            else:
                raise Exception("Unknown pressure input type: " + unitdict["pressure"])

        self._mfrconv = 1.0
        if "massFlowRate" in unitdict:
            # converting to kg/s
            if unitdict["massFlowRate"].lower() in mfrdict:
                self._mfrconv = mfrdict[unitdict["massFlowRate"].lower()]
            else:
                raise Exception("Unknown pressure input type: " + unitdict["massFlowRate"])

        self._densconv = 1.0
        if "density" in unitdict:
            # converting to kg/m3
            if unitdict["density"].lower() in densdict:
                self._densconv = densdict[unitdict["density"].lower()]
            else:
                raise Exception("Unknown density input type: " + unitdict["density"])

        self._powerconv = 1.0
        if "power" in unitdict:
            # converting to W
            if unitdict["power"].lower() in powerdict:
                self._powerconv = powerdict[unitdict["power"].lower()]
            else:
                raise Exception("Unknown power input type: " + unitdict["power"])

        self._enthalpyconv = 1.0
        if "enthalpy" in unitdict:
            # converting to W
            if unitdict["enthalpy"].lower() in enthalpydict:
                self._enthalpyconv = enthalpydict[unitdict["enthalpy"].lower()]
            else:
                raise Exception("Unknown enthalpy input type: " + unitdict["enthalpy"])

        self._tempconv = lambda T: T
        scale_temp_by, shift_temp_by = 1, 0
        if "temperature" in unitdict:
            # converting to K
            if unitdict["temperature"] == "K":
                self._tempconv = lambda T: T
                scale_temp_by, shift_temp_by = 1, 0
            elif unitdict["temperature"] == "C":
                self._tempconv = lambda T: T + 273.15
                scale_temp_by, shift_temp_by = 1, 273.15
            elif unitdict["temperature"] == "F":
                self._tempconv = lambda T: (T - 32) * 5 / 9 + 273.15
                scale_temp_by, shift_temp_by = 5.0 / 9.0, 273.15 - (32.0 * 5.0 / 9.0)
            elif unitdict["temperature"] == "R":
                self._tempconv = lambda T: T * 5 / 9
                scale_temp_by, shift_temp_by = 5.0 / 9.0, 0
            else:
                raise Exception("Unknown temperature input type: " + unitdict["temperature"])
        self._tempconvfactors = [scale_temp_by, shift_temp_by]

    @property
    def lengthConversion(self) -> float:
        return self._lenconv

    @property
    def areaConversion(self) -> float:
        return self._lenconv * self._lenconv

    @property
    def volumeConversion(self) -> float:
        return self._volconv

    @property
    def timeConversion(self) -> float:
        return self._timeconv

    @property
    def pressureConversion(self) -> float:
        return self._presconv

    @property
    def massFlowRateConversion(self) -> float:
        return self._mfrconv

    @property
    def densityConversion(self) -> float:
        return self._densconv

    @property
    def powerConversion(self) -> float:
        return self._powerconv

    @property
    def enthalpyConversion(self) -> float:
        return self._enthalpyconv

    def temperatureConversion(self, T: float) -> float:
        """Method for performing a temperature conversion to Kelvin

        Parameters
        ----------
        T : float
            Temperature to be converted

        Returns
        -------
        float
            The equivalent temperature in :math:`K`
        """
        return self._tempconv(T)

    @property
    def temperatureConversionFactors(self) -> list:
        return self._tempconvfactors
