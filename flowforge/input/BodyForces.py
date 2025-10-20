import abc
from typing import List
from copy import deepcopy
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser


class BodyForces:
    """
    Container class for all input body forces
    """

    def __init__(self, **body_forces):

        bf_objects = {"InternalHeatGenerationBF": InternalHeatGenerationBF}

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

    Attributes
    ----------

    """

    def __init__(self,
                 variable: str,
                 value: EquationParser) -> None:

        self._variable_name = variable
        self._value = value
        self._associated_cells = []

        self._body_force_type = None
        if "solid" in variable:
            self._simulation_type = "Solid"
        else:
            self._simulation_type = "Fluid"

    @property
    def body_force_type(self):
        return self._body_force_type

    @body_force_type.setter
    def body_force_type(self, body_force_type):
        self._body_force_type = body_force_type

    @property
    def simulation(self):
        return self._simulation_type

    @property
    def variable_name(self):
        return self._variable_name

    @property
    def body_force_value(self):
        return self._value

    @body_force_value.setter
    def body_force_value(self, value):
        self._value = value

    @property
    def associated_cells(self):
        return self._associated_cells

    @associated_cells.setter
    def associated_cells(self, cells_indices: List[int]):
        self._associated_cells = cells_indices

    def add_cell(self, cell_index: int):
        self._associated_cells.append(cell_index)

    def convertUnits(self, uc: UnitConverter) -> None:
        scale_factor, shift_factor = self._get_variable_conversion(uc)
        self.body_force_value.performUnitConversion(scale_factor, shift_factor)

    def _get_variable_conversion(self, uc: UnitConverter):
        scale_factor, shift_factor = 1, 0
        if self.variable_name in ["mass_flow_rate", "gas_mass_flow_rate"]:
            scale_factor = uc.massFlowRateConversion
        elif self.variable_name == "pressure":
            scale_factor = uc.pressureConversion
        elif self.variable_name == "temperature" or self.variable_name == "solid_temperature":
            scale_factor, shift_factor = uc.temperatureConversionFactors
        elif self.variable_name == "enthalpy" or self.variable_name == "solid_enthalpy":
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


class InternalHeatGenerationBF(GeneralBF):
    """

    """

    def __init__(self, variable, power_value):
        assert variable in self.valid_variables
        super().__init__(variable, power_value)
        self.body_force_type = "InternalHeatGenerationBF"

    @property
    def valid_variables(self):
        return (
            "power",
            "temperature", "solid_temperature",
            "enthalpy", "solid_enthalpy"
        )