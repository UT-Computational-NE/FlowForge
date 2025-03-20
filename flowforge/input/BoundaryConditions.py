import abc
from typing import Dict, List, Union, Optional, Any
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser

class MassMomentumBC:
    """Class for mass and momentum boundary conditions.

    This class handles mass flow rate and momentum-related boundary conditions
    for a fluid system. It enforces that a mass flow rate must be specified at one
    end of the system, and pressure must be specified at the other end.

    Parameters
    ----------
    inlet : dict, optional
        Dictionary containing inlet boundary conditions. Can include "mdot" (mass flow rate)
        or "pressure" (inlet pressure).
    outlet : dict, optional
        Dictionary containing outlet boundary conditions. Can include "mdot" (mass flow rate)
        or "pressure" (outlet pressure).

    Attributes
    ----------
    mdot : float
        Mass flow rate [kg/s]
    surfaceP : float
        Surface pressure [Pa]
    Pside : str
        Side that pressure is on. "inlet" or "outlet"

    Notes
    -----
    At least one of inlet or outlet must be specified. If both are specified,
    one must contain mass flow rate and the other must contain pressure.
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
    def mdot(self) -> Optional[float]:
        return self._mdot

    @property
    def surfaceP(self) -> float:
        return self._surfaceP

    @property
    def Pside(self) -> str:
        return self._Pside

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Convert units using the provided UnitConverter.

        This method converts pressures and mass flow rates from user-specified
        units to SI units using the conversion factors in the UnitConverter.

        Parameters
        ----------
        uc : UnitConverter
            The unit converter object containing conversion factors
        """
        self._surfaceP *= uc.pressureConversion
        if self._mdot:
            self._mdot *= uc.massFlowRateConversion


class EnthalpyBC:
    """Class for enthalpy boundary conditions.

    This class manages enthalpy or temperature boundary conditions at system inlets and outlets.
    It allows specifying either temperature or enthalpy at each boundary and handles the
    appropriate conversions.

    Parameters
    ----------
    inlet : dict, optional
        Dictionary containing inlet boundary conditions. Can specify either
        temperature or enthalpy values.
    outlet : dict, optional
        Dictionary containing outlet boundary conditions. Can specify either
        temperature or enthalpy values.

    Attributes
    ----------
    val_inlet : float
        The inlet boundary condition value
    type_inlet : str
        The inlet boundary condition type, either "temperature" or "enthalpy"
    val_outlet : float
        The outlet boundary condition value
    type_outlet : str
        The outlet boundary condition type, either "temperature" or "enthalpy"

    Notes
    -----
    At least one of inlet or outlet must be specified. Values must be positive.
    """

    def __init__(self, inlet: Optional[Dict] = None, outlet: Optional[Dict] = None):
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


    def val_inlet(self) -> float:
        return self._val_inlet

    @property


    def val_outlet(self) -> float:
        return self._val_outlet

    @property


    def type_inlet(self) -> str:
        return self._type_inlet

    @property


    def type_outlet(self) -> str:
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

class VoidBC:
    """Class for void fraction BC

    Attributes
    ----------
    mdot : float
        Mass flow rate [kg/s]
    void_fraction : float
        Void fraction [-]
    """

    def __init__(self, inlet: dict = None, outlet: dict = None):
        assert inlet or outlet

        self._type_inlet = None
        self._val_inlet = None
        if inlet:
            assert outlet is None
            if "mdot" in inlet:
                assert "void_fraction" not in inlet
                self._type_inlet = "mdot"
                self._val_inlet = inlet["mdot"]
                assert self.val_inlet != 0.0 # for opposite directional flow of voids
            else:
                assert "mdot" not in inlet
                self._type_inlet = "void_fraction"
                self._val_inlet = inlet["void_fraction"]
                assert self.val_inlet >= 0.0 and self.val_inlet <= 1.0
        self._type_outlet = None
        self._val_outlet = None
        if outlet:
            assert inlet is None
            if "mdot" in outlet:
                assert "void_fraction" not in outlet
                self._type_outlet = "mdot"
                self._val_outlet = outlet["mdot"]
                assert self.val_inlet != 0.0 # for opposite directional flow of voids
            else:
                assert "mdot" not in outlet
                self._type_outlet = "void_fraction"
                self._val_outlet = outlet["void_fraction"]
                assert self.val_outlet >= 0.0 and self.val_outlet <= 1.0

    @property


    def val_inlet(self) -> float:
        return self._val_inlet

    @property


    def val_outlet(self) -> float:
        return self._val_outlet

    @property


    def type_inlet(self) -> str:
        return self._type_inlet

    @property


    def type_outlet(self) -> str:
        return self._type_outlet

    def _convertUnits(self, uc: UnitConverter) -> None:
        if self.type_inlet == "mdot":
            self._val_inlet *= uc.massFlowRateConversion
        elif self.type_inlet == "void_fraction":
            pass # void fraction is non-dimensional
        else:
            raise Exception("Unknown void fraction BC type: " + self.type_inlet)


class BoundaryConditions:
    """
    Container class for all input boundary conditions

    "boundary_conditions" should be defined as a dict in the form:

    boundary_conditions = {
        "unique_boundary_name" :
                {"boundary_type": "DirichletBC", "surface": "surface_name", "variable": "variable_name",  "value", float},
        "inlet_mdot"           :
                {"boundary_type": "DirichletBC", "surface": "inlet",        "variable": "mass_flow_rate", "value", 25.0},
        "outlet_pressure"      :
                {"boundary_type": "DirichletBC", "surface": "outlet",       "variable": "pressure",       "value", 1e5},
        "inlet_temperature"    :
                {"boundary_type": "DirichletBC", "surface": "inlet",        "variable": "temperature",    "value", 700}
    }
    """
    def __init__(self, **boundary_conditions: dict):

        bc_objects = {"DirichletBC": DirichletBC}

        self.bcs = {}
        for bc_name, bc in boundary_conditions.items():
            bc_type = bc["boundary_type"]
            bc_obj  = bc_objects[bc_type]
            self.bcs[bc_name] = bc_obj(bc["surface"], bc["variable"], bc["value"])

    @property


    def boundary_conditions(self) -> Any:
        return self.bcs

    @boundary_conditions.setter
    def boundary_conditions(self, bc_dict):
        self.bcs = bc_dict

    def _convertUnits(self, uc: UnitConverter) -> None:
        converted_bcs = {}
        for bc_name in [*self.bcs]:
            bc_obj = self.bcs[bc_name]
            bc_obj.convertUnits(uc)
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
    def __init__(self, surface: str, variable: str, value):
        self._surface_name = surface
        self._variable_name = variable
        self._value = EquationParser(str(value))

        self.bc_type = "None"

    @property


    def boundary_type(self) -> Any:
        return self.bc_type

    @boundary_type.setter
    def boundary_type(self, bc_type):
        self.bc_type = bc_type

    @property


    def boundary_value(self) -> Any:
        return self._value

    @boundary_value.setter
    def boundary_value(self, value):
        self._value = value

    @property


    def variable_name(self) -> Any:
        return self._variable_name

    @property


    def surface_name(self) -> Any:
        return self._surface_name

    def convertUnits(self, uc: UnitConverter) -> None:
        scale_factor, shift_factor = self._get_variable_conversion(uc)
        self.boundary_value.performUnitConversion(scale_factor, shift_factor)

    def _get_variable_conversion(self, uc: UnitConverter):
        scale_factor, shift_factor = 1, 0
        if self.variable_name in ["mass_flow_rate","gas_mass_flow_rate"]:
            scale_factor = uc.massFlowRateConversion
        elif self.variable_name == "pressure":
            scale_factor = uc.pressureConversion
        elif self.variable_name == "temperature":
            scale_factor, shift_factor = uc.temperatureConversionFactors
        elif self.variable_name == "enthalpy":
            scale_factor = uc.enthalpyConversion
        elif self.variable_name == "void_fraction":
            pass # void fraction is non-dimensional
        elif self.variable_name.startswith("neutron_precursor_mass_concentration"):
            pass
        elif self.variable_name.startswith("decay_heat_precursor_mass_concentration"):
            pass
        else:
            raise Exception("ERROR: non-valid variable name: " + self.variable_name + ".")
        return scale_factor, shift_factor

class DirichletBC(GeneralBC):
    """
    Sub-class for Dirichlet boundary conditions
    """
    def __init__(self, surface: str, variable: str, value: float):
        super().__init__(surface, variable, value)
        self.boundary_type = "DirichletBC"
