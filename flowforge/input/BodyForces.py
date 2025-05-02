import abc
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser

class BodyForces:
    """
    Container class for all input body forces
    """
    def __init__(self, **body_forces : dict):
        bf_objects = {"HeatSource": HeatSourceBF,
                      "Gravity": GravitationalBF}

        self.bfs = {}
        for bf_name, bf in body_forces.items():
            bf_type = bf["type"]
            bf_obj = bf_objects[bf_type]
            if bf_name in self.bfs.keys():
                raise Exception("ERROR: Body force with name '"+bf_name+"' already defined")
            self.setBodyForceObject(bf_obj, bf, bf_name)

    def setBodyForceObject(self, bf_base_object, bf_inputs, bf_name):
        if bf_base_object == GravitationalBF:
            self.bfs[bf_name] = bf_base_object(bf_inputs["vector"], bf_inputs["magnitude"])
        elif bf_base_object == HeatSourceBF:
            self.bfs[bf_name] = bf_base_object(bf_inputs["variable"], bf_inputs["source"])

    @property
    def body_forces(self):
        return self.bfs

    @body_forces.setter
    def body_forces(self, bfs):
        self.bfs = bfs

    def _convertUnits(self, uc: UnitConverter):
        converted_bfs = {}
        for bf_name in [*self.bfs]:
            bf_obj = self.bfs[bf_name]
            bf_obj.convertUnits(uc)
            converted_bfs[bf_name] = bf_obj
        self.body_forces = converted_bfs

class GeneralBF(abc.ABC):
    """
    General abstract class for body forces

    Methods:
        -

    Attributes
        -

    """
    def __init__(self, variable: str, source):
        self._variable_name = variable
        self._source_expression = EquationParser(str(source))
        self.body_force_type = None

    @property
    def controllerType(self):
        return self.body_force_type

    @property
    def sourceExpression(self):
        return self._source_expression

    @sourceExpression.setter
    def sourceExpression(self, expression):
        self._source_expression = EquationParser(str(expression))

    @property
    def variableName(self):
        return self._variable_name

    def convertUnits(self, uc: UnitConverter) -> None:
        scale_factor, shift_factor = self._get_variable_conversion(uc)
        self.sourceExpression.performUnitConversion(scale_factor, shift_factor)

    def _get_variable_conversion(self, uc: UnitConverter):
        scale_factor, shift_factor = 1, 0
        if self.variableName in ["mass_flow_rate", "gas_mass_flow_rate"]:
            scale_factor = uc.massFlowRateConversion
        elif self.variableName == "pressure":
            scale_factor = uc.pressureConversion
        elif self.variableName == "temperature":
            scale_factor, shift_factor = uc.temperatureConversionFactors
        elif self.variableName == "enthalpy":
            scale_factor = uc.enthalpyConversion
        elif self.variableName == "void_fraction":
            pass  # void fraction is non-dimensional
        elif self.variableName.startswith("neutron_precursor_mass_concentration"):
            pass
        elif self.variableName.startswith("decay_heat_precursor_mass_concentration"):
            pass
        elif self.variableName == "gravity":
            pass
        else:
            raise Exception("ERROR: non-valid variable name: " + self.variableName + ".")
        return scale_factor, shift_factor

class HeatSourceBF(GeneralBF):
    """
    Sub-class for a heat-source body force
    """

    def __init__(self, variable, source_expression):
        super().__init__(variable, source_expression)
        self.body_force_type = "HeatSourceBF"

class GravitationalBF(GeneralBF):
    """
    Sub-class for a gravitational body force
    """

    def __init__(self, vector: tuple, magnitude = 9.81):
        all_directions = ["x", "y", "z"]
        if len(vector) != 3:
            raise Exception("ERROR in Gravity vector -> Please input a 3D vector.")

        norm_vector = self.normalizeVector(vector)
        gravity_vector = tuple([magnitude * u for u in norm_vector])

        source_expression = ""
        for g_component, direction in zip(gravity_vector, all_directions):
            source_expression += str(g_component) + "*" + str(direction) + "+"
        if source_expression[-1] == "+":
            source_expression = source_expression[:-1]

        super().__init__("gravity", source_expression)
        self.body_force_type = "GravityBF"


    def normalizeVector(self, vector: tuple):
        vector_magnitude = sum([i*i for i in vector])
        return tuple([i/vector_magnitude for i in vector])

class PumpBF(GeneralBF):
    def __init__(self, pressure_change):
        super().__init__("pressure", pressure_change)