import abc
from typing import List, Tuple
from copy import deepcopy
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser


class WallFunctions:
    """
    Container class for all input wall functions

    Parameters
    ----------
    wall_functions : dict[str, dict]
        Dict of wall function definitions

    Attributes
    ----------
    wall_functions : List[GeneralWF]
        List of built wall function objects
    """
    def __init__(self, **wall_functions):

        wf_objects = {"heat_flux": HeatFluxWF}

        self._wfs = {}
        for wf_name, wf in wall_functions.items():
            wf_obj = wf_objects[wf["type"]]
            input_value = EquationParser(str(wf["value"]))
            self._wfs[wf_name] = wf_obj(wf["surface"], wf["variable"], input_value)

    @property
    def wall_functions(self):
        return self._wfs

    @wall_functions.setter
    def wall_functions(self, wall_functions: dict):
        self._wfs = wall_functions

    def _convertUnits(self, uc: UnitConverter):
        converted_wfs = {}
        for wf_name, wf in self.wall_functions.items():
            wf.convertUnits(uc)
            converted_wfs[wf_name] = deepcopy(wf)
        self.wall_functions = converted_wfs


class GeneralWF(abc.ABC):
    """
    General abstract class for wall functions

    Parameters
    ----------
    surface : str
        Surface that this wall function is applied to
    variable : str
        Variable associated with this wall function
    value : EquationParser
        Function that, when evaluated, gives the source value of the wall function

    Attributes
    ----------
    wall_function_type : str
        Type of wall function
    surface_name : str
        Surface that this wall function is applied to
    simulation : str
        Simulation type this wall function is associated with
    wall_function_value : EquationParser
        Function that, when evaluated, gives the source value of the wall function
    variable_name : str
        Variable associated with this wall function
    associated_cells : List[int]
        Indices of cells associated with this wall function
    """

    def __init__(self, surface: str, variable: str, value: EquationParser):
        self._surface_name = surface
        self._variable_name = variable
        self._value = value

        self._wall_function_type = None
        if "solid" in variable:
            self._simulation_type = "Solid"
        else:
            self._simulation_type = "Fluid"

    @property
    def wall_function_type(self) -> str:
        return self._wall_function_type

    @wall_function_type.setter
    def wall_function_type(self, wall_function_type) -> None:
        self._wall_function_type = wall_function_type

    @property
    def surface_name(self) -> str:
        return self._surface_name

    @property
    def simulation(self) -> str:
        return self._simulation_type

    @property
    def wall_function_value(self) -> EquationParser:
        return self._value

    @wall_function_value.setter
    def wall_function_value(self, value) -> None:
        self._value = value

    @property
    def variable_name(self) -> str:
        return self._variable_name

    def convertUnits(self, uc: UnitConverter) -> None:
        """
        Converts units

        Parameters
        ----------
        uc : UnitConverter
            Unit converter object used to get the scale factors needed
        """
        scale_factor, shift_factor = self._get_variable_conversion(uc)
        self.wall_function_value.performUnitConversion(scale_factor, shift_factor)

    def _get_variable_conversion(self, uc: UnitConverter) -> Tuple[float, float]:
        """
        For the variable associated with this wall function, this method extracts the proper
        scaling and shifting factors for unit conversion

        Parameters
        ----------
        uc : UnitConverter
            Unit converter object used to get the scale factors needed
        """
        scale_factor, shift_factor = 1, 0
        if self.variable_name in ["mass_flow_rate", "gas_mass_flow_rate"]:
            scale_factor = uc.massFlowRateConversion
        elif self.variable_name == "pressure":
            scale_factor = uc.pressureConversion
        elif self.variable_name in ("temperature", "solid_temperature"):
            scale_factor, shift_factor = uc.temperatureConversionFactors
        elif self.variable_name in ("enthalpy", "solid_enthalpy"):
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


class HeatFluxWF(GeneralWF):
    """
    Heat Flux Wall Function

    Parameters
    ----------
    surface : str
        Surface that this wall function is applied to
    variable : str
        Variable associated with this wall function
    value : EquationParser
        Function that, when evaluated, gives the source value of the wall function

    Attributes
    ----------
    wall_function_type : str
        Type of wall function
    surface_name : str
        Surface that this wall function is applied to
    simulation : str
        Simulation type this wall function is associated with
    wall_function_value : EquationParser
        Function that, when evaluated, gives the source value of the wall function
    variable_name : str
        Variable associated with this wall function
    associated_cells : List[int]
        Indices of cells associated with this wall function
    """

    def __init__(self, surface, variable, heat_flux):
        super().__init__(surface, variable, heat_flux)
        self.wall_function_type = "heat_flux"
