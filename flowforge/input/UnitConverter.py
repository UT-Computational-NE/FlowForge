# length to m
lendict = {
    'in': 0.0254,
    'ft': 0.3048,
    'yd': 0.9144,
    'mm': 0.001,
    'cm': 0.01,
    'm' : 1.0
}

# volume in m3
voldict = {
    'ml' : 1.0e-6,
    'l'  : 1.0e-3,
    'gal': 0.00378541,
    'qt' : 0.000946353,
    'pt' : 0.000437167,
    'c'  : 0.00024,
    'ft3': 0.0283168,
    'in3': 0.000016387,
    'm3' : 1.0,
    'cm3': 1.0e-6
}

# time to s
timedict = {
    'week': 604800,
    'hr'  : 3600,
    'min' : 60,
    'ms'  : 0.001,
    's'   : 1.0
}

# pressure to Pa
presdict = {
    'kpa' : 0.001,
    'pa'  : 1.0,
    'psi' : 6894.76,
    'bar' : 1e5,
    'atm' : 101325,
    'torr': 133.322
}

# mass flow rate to kg/s
mfrdict = {
    'kg/s'  : 1.0,
    'g/s'   : 0.001,
    'g/min' : 0.00001667,
    'kg/min': 0.01667,
    'kg/hr' : 0.0002778,
    'lb/s'  : 0.453592,
    'lb/min': 0.00755987,
    'lb/hr' : 0.000125998
}

# density to kg/m3
densdict = {
    'kg/m3' : 1.0,
    'kg/cm3': 1.0e6,
    'g/cm3' : 1.0e3,
    'g/m3'  : 1.0e-3,
    'lb/in3': 27679,
    'lb/ft3': 16.0185
}

# power to W
powerdict = {
    'w': 1.0,
    'kw': 1.0e3,
    'mw': 1.0e6,
    'gw': 1.0e9,
    'tw': 1.0e12
}

class UnitConverter:
    """
    Handles all unit conversions for system inputs.
    """
    def __init__(self, unitdict):
        """
        The __init__ function initializes the unit conversion class by setting the
        default of all unit conversions to be equal to 1.0. To initialize this class
        instance, a unit dictionary must be passed as an argument.

        This function then automatically checks the unit dict to see if it contains
        each unit key. If the key is found in the dict, the unit conversion variable
        is then set by matching the unit given to the conversion from the dictionaries
        above. These numbers are the unit conversions to be multiplied to return the
        values in SI units.

        If the unit given is not in our defined dictionaries, it will throw an exception
        for an unknown input unit type.

        The temperature conversion is the only unit that varies from the others. This
        conversion is done using lambda functions. Depending on the temperature unit
        input, a different function is used to return the temperature in K.

        Args:
            - unitdict : dict, dictionary containing the unit names as the keys and the
                         units that are used for the system inputs
        """
        self._lenconv = 1.0
        if "length" in unitdict:
            # converting to m
            if unitdict['length'].lower() in lendict:
                self._lenconv = lendict[unitdict['length'].lower()]
            else:
                raise Exception('Unknown length input type: ' + unitdict['length'])

        self._volconv = self._lenconv*self._lenconv*self._lenconv
        if "volume" in unitdict:
            # converting to m3
            if unitdict['volume'].lower() in voldict:
                self._volconv = voldict[unitdict['volume'].lower()]
            else:
                raise Exception('Unknown volume input type: ' + unitdict['volume'])

        self._timeconv = 1.0
        if "time" in unitdict:
            # converting to s
            if unitdict['time'].lower() in timedict:
                self._timeconv = timedict[unitdict['time'].lower()]
            else:
                raise Exception('Unknown time input type: ' + unitdict['time'])

        self._presconv = 1.0
        if "pressure" in unitdict:
            # converting to Pa
            if unitdict['pressure'].lower() in presdict:
                self._presconv = presdict[unitdict['pressure'].lower()]
            else:
                raise Exception('Unknown pressure input type: ' + unitdict['pressure'])

        self._mfrconv = 1.0
        if "massFlowRate" in unitdict:
            # converting to kg/s
            if unitdict['massFlowRate'].lower() in mfrdict:
                self._mfrconv = mfrdict[unitdict['massFlowRate'].lower()]
            else:
                raise Exception('Unknown pressure input type: ' + unitdict['massFlowRate'])

        self._densconv = 1.0
        if "density" in unitdict:
            # converting to kg/m3
            if unitdict['density'].lower() in densdict:
                self._densconv = densdict[unitdict['density'].lower()]
            else:
                raise Exception('Unknown density input type: ' + unitdict['density'])

        self._powerconv = 1.0
        if "power" in unitdict:
            # converting to W
            if unitdict['power'].lower() in powerdict:
                self._powerconv = powerdict[unitdict['power'].lower()]
            else:
                raise Exception('Unknown power input type: ' + unitdict['power'])

        self._tempconv = lambda T: T
        if "temperature" in unitdict:
            # converting to K
            if unitdict['temperature'] == "K":
                self._tempconv = lambda T: T
            elif unitdict['temperature'] == "C":
                self._tempconv = lambda T: T + 273.15
            elif unitdict['temperature'] == "F":
                self._tempconv = lambda T: (T - 32) * 5 / 9 + 273.15
            elif unitdict['temperature'] == "R":
                self._tempconv = lambda T: T * 5 / 9
            else:
                raise Exception('Unknown temperature input type: ' + unitdict['temperature'])

    @property
    def lengthConversion(self):
        """
        The lengthConversion function uses the stored _lenconv variable
        to return the system input length in m.
        """
        return self._lenconv

    @property
    def areaConversion(self):
        """
        The areaConversion function squares the stored _lenconv variable
        to return the system input area in m2.
        """
        return self._lenconv*self._lenconv

    @property
    def volumeConversion(self):
        """
        The volumeConversion function uses the stored _volconv variable
        to return the system input volume in m3.
        """
        return self._volconv

    @property
    def timeConversion(self):
        """
        The timeConversion function uses the stored _timeconv variable
        to return the system input time in s.
        """
        return self._timeconv

    @property
    def pressureConversion(self):
        """
        The pressureConversion function uses the stored _presconv variable
        to return the system input pressure in pa.
        """
        return self._presconv

    @property
    def massFlowRateConversion(self):
        """
        The massFlowRateConversion function uses the stored _mfrconv variable
        to return the system input mass flow rate in kg/s.
        """
        return self._mfrconv

    @property
    def densityConversion(self):
        """
        The densityConversion function uses the stored _densconv variable
        to return the system input density in kg/m3.
        """
        return self._densconv

    @property
    def powerConversion(self):
        """
        The powerConversion function uses the stored _powerconv variable
        to return the system input power in w.
        """
        return self._powerconv

    def temperatureConversion(self, T):
        """
        The temperatureConversion function uses the stored _tempconv variable
        to return the system input temperature in K.

        Args:
            - T : float, temperature
        """
        return self._tempconv(T)

if __name__ == "__main__":
    import json
    with open('sample.json', 'r') as f:
        input_dict = json.load(f)
        unit_dict = input_dict['units']
        converter = UnitConverter(unit_dict)
        print(converter.temperatureConversion(600))
