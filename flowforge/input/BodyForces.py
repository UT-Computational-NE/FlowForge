import abc
from copy import deepcopy
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser


class BodyForces:
    """
    Container class for all input body forces

    Parameters
    ----------
    body_forces : dict[str, dict]
        Dict of body force definitions

    Attributes
    ----------
    body_forces : List[GeneralBF]
        List of built body force objects
    """

    def __init__(self, **body_forces):

        bf_objects = {"heat_generation": HeatGenerationBF}

        self._bfs = {}
        for bf_name, bf in body_forces.items():
            bf_obj = bf_objects[bf["type"]]
            input_value = EquationParser(str(bf["value"]))
            self._bfs[bf_name] = bf_obj(bf["variable"], input_value)

    @property
    def body_forces(self):
        return self._bfs

    @body_forces.setter
    def body_forces(self, body_forces: dict):
        self._bfs = body_forces

    def _convertUnits(self, uc: UnitConverter):
        converted_bfs = {}
        for bf_name, bf in self.body_forces.items():
            bf.convertUnits(uc)
            converted_bfs[bf_name] = deepcopy(bf)
        self.body_forces = converted_bfs


class GeneralBF(abc.ABC):
    """
    General abstract class for body forces

    Parameters
    ----------
    variable : str
        Name of the variable this body force applies to
    value : EquationParser
        Function that, when evaluated, provides the body-force value

    Attributes
    ----------
    body_force_type : str
        Type of body force
    simulation : str
        Simulation type this body force is associated with
    variable_name : str
        Variable this body force is associated with
    body_force_value : EquationParser
        Function that, when evaluated, gives the source value of the body force
    associated_cells : List[int]
        Indices of cells associated with this body force
    """

    def __init__(self,
                 variable: str,
                 value: EquationParser) -> None:

        self._variable_name = variable
        self._value = value

        self._body_force_type = None

    @property
    def body_force_type(self) -> str:
        return self._body_force_type

    @body_force_type.setter
    def body_force_type(self, body_force_type) -> None:
        self._body_force_type = body_force_type

    @property
    def variable_name(self) -> str:
        return self._variable_name

    @property
    def body_force_value(self) -> EquationParser:
        return self._value

    @body_force_value.setter
    def body_force_value(self, value: EquationParser) -> None:
        self._value = value

    def convertUnits(self, uc: UnitConverter) -> None:
        """
        Converts units

        Parameters
        ----------
        uc : UnitConverter
            Unit converter object used to get the scale factors needed
        """
        scale_factor, shift_factor = uc.get_variable_conversion(self.variable_name)
        self.body_force_value.performUnitConversion(scale_factor, shift_factor)


class HeatGenerationBF(GeneralBF):
    """
    Class for a Heat Generation Body Force

    Parameters
    ----------
    variable : str
        Name of the variable this body force applies to
    value : EquationParser
        Function that, when evaluated, provides the body-force value

    Attributes
    ----------
    body_force_type : str
        Type of body force
    simulation : str
        Simulation type this body force is associated with
    variable_name : str
        Variable this body force is associated with
    body_force_value : EquationParser
        Function that, when evaluated, gives the source value of the body force
    associated_cells : List[int]
        Indices of cells associated with this body force
    """

    def __init__(self, variable, power_value):
        assert variable in self.valid_variables
        super().__init__(variable, power_value)
        self.body_force_type = "heat_generation"

    @property
    def valid_variables(self):
        return (
            "power",
            "temperature", "solid_temperature",
            "enthalpy", "solid_enthalpy"
        )
