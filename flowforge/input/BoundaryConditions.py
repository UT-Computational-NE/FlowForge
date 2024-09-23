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
    def __init__(self, inlet: dict = None, outlet: dict = None):
        assert outlet

        if inlet:
            if 'mdot' in inlet:
                self._mdot = inlet['mdot']
                self._Pside = 'outlet'
                self._surfaceP = outlet['pressure']
            else:
                self._mdot = outlet['mdot']
                self._Pside = 'inlet'
                self._surfaceP = inlet['pressure']
            assert self.mdot != 0
        else:
            assert outlet
            assert 'mdot' not in outlet
            self._mdot = None
            self._Pside = 'outlet'
            self._surfaceP = outlet['pressure']

        assert self.surfaceP > 0
        assert self.Pside in('inlet', 'outlet')

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
        if self._mdot:
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
    def __init__(self, inlet: dict = None, outlet: dict = None):
        assert inlet or outlet

        self._type_inlet = None
        self._val_inlet = None
        if inlet:
            self._type_inlet = 'enthalpy'
            self._val_inlet = inlet
            if isinstance(inlet, dict):
                for key in inlet.keys():
                    self._type_inlet = key
                self._val_inlet = inlet[self.type_inlet]

            assert self.val_inlet > 0
            assert self.type_inlet in('temperature', 'enthalpy')

        self._type_outlet = None
        self._val_outlet = None
        if outlet:
            self._type_outlet = 'enthalpy'
            self._val_outlet = outlet
            if isinstance(outlet, dict):
                for key in outlet.keys():
                    self._type_outlet = key
                self._val_outlet = outlet[self.type_outlet]
            assert self.val_outlet > 0
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
        if self._val_inlet:
            if self.type_inlet == 'temperature':
                self._val_inlet = uc.temperatureConversion(self._val_inlet)
            elif self.type_inlet == 'enthalpy':
                self._val_inlet *= uc.enthalpyConversion
            else:
                raise Exception("Unknown enthalpy BC type: " + self.type_inlet)

        if self._val_outlet:
            if self.type_outlet == 'temperature':
                self._val_outlet = uc.temperatureConversion(self._val_outlet)
            elif self.type_outlet == 'enthalpy':
                self._val_outlet *= uc.enthalpyConversion
            else:
                raise Exception("Unknown enthalpy BC type: " + self.type_outlet)

class DirichletBC():
    """
    General class for Dirichlet boundary conditions

    Types of BC inputs:
      - mass flow rate
      - pressure
      - temperature
      - enthalpy
    
    bounday_conditions should be defined as a dict in the form:

    bounday_conditions = {
        "mass_flow_rate" : {"inlet/outlet", float},
        "pressure"       : {"inlet/outlet", float},
        "temperature"    : {"inlet/outlet", float},
        "enthalpy"       : {"inlet/outlet", float}
    }
    
    attributes:
      * _mdot        -> Tuple[string, float]
      * _pressure    -> Tuple[string, float]
      * _enthalpy    -> Tuple[string, float]
      * _temperature -> Tuple[string, float]

    """

    """
    TODO For the class:
      - For right now, this class can only handle 1 BC per property (mdot, pressure, ...)
      - Will need to add a method to hold mutiple BCs
        - Perhaps : _temperature -> dict : {"node1" : float, "node2" : float, ...}
        - Need to examine how this would function downstream before adding too much

      - Also, as a more long-term note, this class is *not* in the final form. As a goal, this
        class should not have to specify/hard-code the properties (temperature, mdot, ...), but
        should instead strive to be more general. It should take in, as inputs, a reference to 
        the variable of which the BC will be applied, the value at the boundary, and some mapping
        to the actual fluid mesh, as well as the mesh-index at which to apply the BC. In this 
        manner, one could theortically set a BC of any variable at any point in the mesh. As this
        is not a vitale feature of SyTH, it should be done later as an added feature.
                                                                                        - Charlie
    
    """


    def __init__(self, **bounday_conditions: dict):
        bc_types = list(bounday_conditions.keys())

        self._mdot = [None, None]
        self._pressure = [None, None]
        self._temperature = [None, None]
        self._enthalpy = [None, None]

        valid_bc_types = {"mass_flow_rate" : None,
                          "pressure" : None,
                          "temperature" : None,
                          "enthalpy" : None}

        for bc_type in [*valid_bc_types]:
            if bc_type in bc_types:
                valid_bc_types[bc_type] = [*bounday_conditions[bc_type],  # variable name
                                           *bounday_conditions[bc_type].values()]  # variable value

        self._mdot = valid_bc_types["mass_flow_rate"]
        self._pressure = valid_bc_types["pressure"]
        self._temperature = valid_bc_types["temperature"]
        self._enthalpy = valid_bc_types["enthalpy"]

    @property
    def mdot(self):
        return self._mdot

    @property
    def pressure(self):
        return self._pressure

    @property
    def temperature(self):
        return self._temperature

    @property
    def enthalpy(self):
        return self._enthalpy

    def _convertUnits(self, uc: UnitConverter) -> None:
        if self._mdot[0] is not None:
            self._mdot[1] *= uc.massFlowRateConversion
        if self._pressure[0] is not None:
            self._pressure[1] *= uc.pressureConversion
        if self._temperature[0] is not None:
            self._temperature[1] = uc.temperatureConversion(self._temperature[1])
        if self._enthalpy[0] is not None:
            self._enthalpy[1] *= uc.enthalpyConversion
        