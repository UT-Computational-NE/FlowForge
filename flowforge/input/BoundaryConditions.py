from flowforge.input.UnitConverter import UnitConverter
import abc

class MassMomentumBC:
    """Class for mass and momentum BC

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
            if "mdot" in inlet:
                self._mdot = inlet["mdot"]
                self._Pside = "outlet"
                self._surfaceP = outlet["pressure"]
            else:
                self._mdot = outlet["mdot"]
                self._Pside = "inlet"
                self._surfaceP = inlet["pressure"]
            assert self.mdot != 0
        else:
            assert outlet
            assert "mdot" not in outlet
            self._mdot = None
            self._Pside = "outlet"
            self._surfaceP = outlet["pressure"]

        assert self.surfaceP > 0
        assert self.Pside in ("inlet", "outlet")

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


class EnthalpyBC:
    """Class for enthalpy BC.

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
            self._type_inlet = "enthalpy"
            self._val_inlet = inlet
            if isinstance(inlet, dict):
                for key in inlet.keys():
                    self._type_inlet = key
                self._val_inlet = inlet[self.type_inlet]

            assert self.val_inlet > 0
            assert self.type_inlet in ("temperature", "enthalpy")

        self._type_outlet = None
        self._val_outlet = None
        if outlet:
            self._type_outlet = "enthalpy"
            self._val_outlet = outlet
            if isinstance(outlet, dict):
                for key in outlet.keys():
                    self._type_outlet = key
                self._val_outlet = outlet[self.type_outlet]
            assert self.val_outlet > 0
            assert self.type_outlet in ("temperature", "enthalpy")

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
            if self.type_inlet == "temperature":
                self._val_inlet = uc.temperatureConversion(self._val_inlet)
            elif self.type_inlet == "enthalpy":
                self._val_inlet *= uc.enthalpyConversion
            else:
                raise Exception("Unknown enthalpy BC type: " + self.type_inlet)

        if self._val_outlet:
            if self.type_outlet == "temperature":
                self._val_outlet = uc.temperatureConversion(self._val_outlet)
            elif self.type_outlet == "enthalpy":
                self._val_outlet *= uc.enthalpyConversion
            else:
                raise Exception("Unknown enthalpy BC type: " + self.type_outlet)


class BoundaryConditions:
    """
    Continer class for all input boundary conditions

    "bounday_conditions" should be defined as a dict in the form:

    bounday_conditions = {
        "unique_boundary_name" : {"boundary_type": "DirichletBC", "surface": "surface_name", "varaible": "varaible_name",  "value", float},
        "inlet_mdot"           : {"boundary_type": "DirichletBC", "surface": "inlet",        "varaible": "mass_flow_rate", "value", 25.0},
        "outlet_pressure"      : {"boundary_type": "DirichletBC", "surface": "outlet",       "varaible": "pressure",       "value", 1e5},
        "inlet_temperature"    : {"boundary_type": "DirichletBC", "surface": "inlet",        "varaible": "temperature",    "value", 700}
    }
    """
    def __init__(self, **bounday_conditions: dict):
        bc_names = list(bounday_conditions.keys())

        bc_objects = {"DirichletBC": DirichletBC}

        bc_list = []
        for bc_name in bc_names:
            bc_type = bounday_conditions[bc_name]["boundary_type"]
            bc_obj = bc_objects[bc_type]
            bc = bc_obj(bounday_conditions["surface"],
                        bounday_conditions["variable"],
                        bounday_conditions["value"])
            bc_list.append(bc)

        N = len(bc_names)
        self.bcs = dict()
        for i in range(N):
            self.bcs[bc_names[i]] = bc_list[i]

    @property
    def boundary_conditions(self):
        return self.bcs

    @boundary_conditions.setter
    def boundary_conditions(self, bc_dict):
        self.bcs = bc_dict

    def _convertUnits(self, uc: UnitConverter):
        converted_bcs = dict()
        for bc_name in [*self.bcs]:
            bc_obj = self.bcs[bc_name]
            bc_obj._convertUnits(uc)
            converted_bcs[bc_name] = bc_obj
        self.boundary_conditions = converted_bcs


class GeneralBC(abc.ABC):
    """
    General abstract class for boundary conditions

    Methods:
        - _convertUnits
        - _get_variable_conversion

    Attributes:
        - _surface_name: str
        - _variable_name : str
        - _value: float
    """
    def __init__(self, surface: str, variable: str, value: float):
        self._surface_name = surface
        self._variable_name = variable
        self._value = value

    @property
    def boundary_type(self):
        raise NotImplementedError

    @property
    def boundary_value(self):
        return self._value

    @boundary_value.setter
    def boundary_value(self, value):
        self._value = value

    @property
    def varaible_name(self):
        return self._variable_name

    @property
    def surface_name(self):
        return self._surface_name

    def _convertUnits(self, uc: UnitConverter) -> None:
        conversion_factor = self._get_variable_conversion(uc)
        self.boundary_value = self.boundary_value * conversion_factor

    def _get_variable_conversion(self, uc: UnitConverter):
        if self.varaible_name == "mass_flow_rate":
            conversion_factor = uc.massFlowRateConversion
        elif self.varaible_name == "pressure":
            conversion_factor = uc.pressureConversion
        elif self.varaible_name == "temperature":
            conversion_factor = uc.temperatureConversion(self.boundary_value) / self.boundary_value
        elif self.varaible_name == "enthalpy":
            conversion_factor = uc.enthalpyConversion
        else:
            raise Exception('ERROR: non-valid variable name: '+self.varaible_name+'.')
        return conversion_factor

class DirichletBC(GeneralBC):
    """
    Sub-class for Dirichlet boundary conditions
    """
    def __init__(self, surface: str, variable: str, value: float):
        super().__init__(surface, variable, value)

    @property
    def boundary_type(self):
        return "DirichletBC"
