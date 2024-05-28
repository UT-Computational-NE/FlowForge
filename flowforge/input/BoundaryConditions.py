from flowforge.input.UnitConverter import UnitConverter

class MassMomentumBC():
    """ Class for mass and momentum BC

    Attributes
    ----------
    mdot : float
        Mass float rate [kg/s]
    surfaceP : float
        Surface pressure [Pa]
    Pside : str
        Side that pressure is on. "inlet" or "outlet"
    """
    def __init__(self, inlet: dict = {"mdot" : 1.0}, outlet: dict = {"pressure" : 101325.0}):
        if 'mdot' in inlet:
            self._mdot = inlet['mdot']
            self._Pside = 'outlet'
            self._surfaceP = outlet['pressure']
        else:
            self._mdot = outlet['mdot']
            self._Pside = 'inlet'
            self._surfaceP = inlet['pressure']
        assert self.surfaceP > 0
        assert self.Pside in('inlet', 'outlet')
        assert self.mdot != 0

    @property
    def mdot(self):
        return self._mdot

    @property
    def surfaceP(self):
        return self._surfaceP

    @property
    def Pside(self):
        return self._Pside

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._surfaceP *= uc.pressureConversion
        self._mdot *= uc.massFlowRateConversion

class EnthalpyBC():
    """ Class for enthalpy BC.

    Attributes
    ----------
    val_inlet : dict
        The inlet BC value
    type_inlet : dict
        The inlet BC type
    val_outlet : dict
        The outlet BC value
    type_outlet : dict
        The outlet BC type
    """
    def __init__(self, inlet: dict = {"temperature" : 873.15}, outlet: dict = {"temperature" : 873.15}):
        self._type_inlet = 'enthalpy'
        self._val_inlet = inlet
        if isinstance(inlet, dict):
            for key in inlet.keys():
                self._type_inlet = key
            self._val_inlet = inlet[self.type_inlet]
        self._type_outlet = 'enthalpy'
        self._val_outlet = outlet
        if isinstance(outlet, dict):
            for key in outlet.keys():
                self._type_outlet = key
            self._val_outlet = outlet[self.type_outlet]
        assert self.val_inlet > 0
        assert self.val_outlet > 0
        assert self.type_inlet in('temperature', 'enthalpy')
        assert self.type_outlet in('temperature', 'enthalpy')

    @property
    def val_inlet(self):
        return self._val_inlet

    @property
    def val_outlet(self):
        return self._val_outlet

    @property
    def type_inlet(self):
        return self._type_inlet

    @property
    def type_outlet(self):
        return self._type_outlet

    def _convertUnits(self, uc: UnitConverter) -> None:
        if self.type_inlet == 'temperature':
            self._val_inlet = uc.temperatureConversion(self._val_inlet)
        elif self.type_inlet == 'enthalpy':
            self._val_inlet *= uc.enthalpyConversion
        else:
            raise Exception("Unknown enthalpy BC type: " + self.type_inlet)
        if self.type_outlet == 'temperature':
            self._val_outlet = uc.temperatureConversion(self._val_outlet)
        elif self.type_outlet == 'enthalpy':
            self._val_outlet *= uc.enthalpyConversion
        else:
            raise Exception("Unknown enthalpy BC type: " + self.type_outlet)
