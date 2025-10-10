import abc
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser


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
            bc_obj = bc_objects[bc_type]
            self.bcs[bc_name] = bc_obj(bc["surface"], bc["variable"], bc["value"])

    @property
    def boundary_conditions(self):
        return self.bcs

    @boundary_conditions.setter
    def boundary_conditions(self, bc_dict):
        self.bcs = bc_dict

    def _convertUnits(self, uc: UnitConverter):
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
    def boundary_type(self):
        return self.bc_type

    @boundary_type.setter
    def boundary_type(self, bc_type):
        self.bc_type = bc_type

    @property
    def boundary_value(self):
        return self._value

    @boundary_value.setter
    def boundary_value(self, value):
        self._value = value

    @property
    def variable_name(self):
        return self._variable_name

    @property
    def surface_name(self):
        return self._surface_name

    def convertUnits(self, uc: UnitConverter) -> None:
        scale_factor, shift_factor = self._get_variable_conversion(uc)
        self.boundary_value.performUnitConversion(scale_factor, shift_factor)

    def _get_variable_conversion(self, uc: UnitConverter):
        scale_factor, shift_factor = 1, 0
        if self.variable_name in ["mass_flow_rate", "gas_mass_flow_rate"]:
            scale_factor = uc.massFlowRateConversion
        elif self.variable_name == "pressure":
            scale_factor = uc.pressureConversion
        elif self.variable_name == "temperature":
            scale_factor, shift_factor = uc.temperatureConversionFactors
        elif self.variable_name == "enthalpy":
            scale_factor = uc.enthalpyConversion
        elif self.variable_name == "void_fraction":
            pass  # void fraction is non-dimensional
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
