import abc
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser


class BoundaryConditions:
    """
    Container class for all input boundary conditions

    "boundary_conditions" should be defined as a dict in the form:

    boundary_conditions = {
        "unique_boundary_name" :
                {"boundary_type": "BC_type",     "surface": "surface_name", "variable": "variable_name",  "value": float},
        "inlet_mdot"           :
                {"boundary_type": "DirichletBC", "surface": "inlet",        "variable": "mass_flow_rate", "value": 25.0},
        "outlet_pressure"      :
                {"boundary_type": "DirichletBC", "surface": "outlet",       "variable": "pressure",       "value": 1e5},
        "inlet_temperature"    :
                {"boundary_type": "DirichletBC", "surface": "inlet",        "variable": "temperature",    "value": 700},
        "outer_solid_temperature"    :
                {"boundary_type": "NeumannBC",   "surface": "outer",        "variable": "solid_enthalpy", "value": 1e6},
    }

    Parameters
    ----------
    boundary_conditions : dict[str, dict]
        Dict of boundary condition definitions

    Attributes
    ----------
    boundary_conditions : List[GeneralWF]
        List of built boundary condition objects

    """

    def __init__(self, **boundary_conditions: dict):

        bc_objects = {"DirichletBC": DirichletBC,
                      "NeumannBC": NeumannBC,
                      "AdiabaticBC": AdiabaticBC}
        self.bcs = {}
        for bc_name, bc in boundary_conditions.items():
            bc_type = bc["boundary_type"]
            bc_obj = bc_objects[bc_type]
            if bc_type in ("DirichletBC", "NeumannBC"):
                self.bcs[bc_name] = bc_obj(bc["surface"], bc["variable"], str(bc["value"]))
            elif bc_type == "AdiabaticBC":
                self.bcs[bc_name] = bc_obj(bc["surface"], bc["variable"])
            else:
                raise Exception(f"no such boundary type: {bc_type}")

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

    Parameters
    ----------
    surface : str
        Name of the surface of which this boundary condition is applied
    variable : str
        Name of the variable used for this boundary
    value : str
        Value input for the boundary condition

    Attributes
    ----------
    boundary_type : str
        Type of boundary condition
    boundary_value : EquationParser
        Function that, when evaluated, gives the boundary value
    variable_name : str
        Variable associated with this boundary
    surface_name : str
        Surface this boundary is applied to
    """

    def __init__(self, surface: str, variable: str, value: str):
        self._surface_name = surface
        self._variable_name = variable
        self._value = EquationParser(value)

        self._boundary_type = None

    @property
    def boundary_type(self):
        return self._boundary_type

    @boundary_type.setter
    def boundary_type(self, boundary_type):
        self._boundary_type = boundary_type

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
        """
        Converts units

        Parameters
        ----------
        uc : UnitConverter
            Unit converter object used to get the scale factors needed
        """
        scale_factor, shift_factor = uc.get_variable_conversion(self.variable_name)
        self.boundary_value.performUnitConversion(scale_factor, shift_factor)


class DirichletBC(GeneralBC):
    """
    Sub-class for Dirichlet boundary conditions

    Parameters
    ----------
    surface : str
        Name of the surface of which this boundary condition is applied
    variable : str
        Name of the variable used for this boundary
    value : str
        Value input for the boundary condition

    Attributes
    ----------
    boundary_type : str
        Type of boundary condition
    boundary_value : EquationParser
        Function that, when evaluated, gives the boundary value
    variable_name : str
        Variable associated with this boundary
    surface_name : str
        Surface this boundary is applied to
    """

    def __init__(self, surface: str, variable: str, value: str):
        super().__init__(surface, variable, value)
        self._boundary_type = "DirichletBC"

class NeumannBC(GeneralBC):
    """
    Sub-class for Neumann boundary conditions

    Parameters
    ----------
    surface : str
        Name of the surface of which this boundary condition is applied
    variable : str
        Name of the variable used for this boundary
    value : str
        Value input for the boundary condition

    Attributes
    ----------
    boundary_type : str
        Type of boundary condition
    boundary_value : EquationParser
        Function that, when evaluated, gives the boundary value
    variable_name : str
        Variable associated with this boundary
    surface_name : str
        Surface this boundary is applied to
    """

    def __init__(self, surface: str, variable: str, value: str):
        super().__init__(surface, variable, value)
        self._boundary_type = "NeumannBC"

class AdiabaticBC(GeneralBC):
    """
    Sub-class for Adiabatic boundary conditions

    Parameters
    ----------
    surface : str
        Name of the surface of which this boundary condition is applied
    variable : str
        Name of the variable used for this boundary

    Attributes
    ----------
    boundary_type : str
        Type of boundary condition
    boundary_value : EquationParser
        Function that, when evaluated, gives the boundary value
        (For adiabatic, set to "1.0")
    variable_name : str
        Variable associated with this boundary
    surface_name : str
        Surface this boundary is applied to
    """
    def __init__(self, surface, variable):
        super().__init__(surface, variable, "1.0")
        self._boundary_type == "AdiabaticBC"