import abc
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser

class WallFunctions:
    """
    Container class for all input wall functions
    """
    def __init__(self, **wall_functions : dict):
        wf_objects = {"HeatFlux": HeatFluxWF,
                      "HeatConvection": HeatConvectionWF}

        self.wfs = {}
        for wf_name, wf in wall_functions.items():
            wf_type = wf["type"]
            wf_obj = wf_objects[wf_type]
            if wf_name in self.wfs.keys():
                raise Exception("ERROR: Body force with name '"+wf_name+"' already defined")
            self.setWallFunctionObject(wf_obj, wf, wf_name, wf_type)

    def setWallFunctionObject(self, wf_base_obj, wf_dict: dict, wf_name: str):
        if (wf_base_obj == HeatConvectionWF):
            self.wfs[wf_name] = wf_base_obj(wf_dict["variable"], wf_dict["ambient"],
                                            wf_dict["heat_transfer_coefficient"])
        elif (wf_base_obj == FrictionWF):
            self.wfs[wf_name] = wf_base_obj(wf_dict["formulation"], wf_dict["default"])
        else:
            self.wfs[wf_name] = wf_base_obj(wf_dict["variable"], wf_dict["source"])

    @property
    def wall_functions(self):
        return self.wfs

    @wall_functions.setter
    def wall_functions(self, wfs):
        self.wfs = wfs

    def _convertUnits(self, uc: UnitConverter):
        converted_wfs = {}
        for wf_name in [*self.wfs]:
            wf_obj = self.wfs[wf_name]
            wf_obj.convertUnits(uc)
            converted_wfs[wf_name] = wf_obj
        self.wall_functions = converted_wfs

class GeneralWF(abc.ABC):
    """
    General abstract class for wall functions

    Methods:
        -

    Attributes
        -

    """
    def __init__(self, variable: str, source):
        self._variable_name = variable
        self._source_expression = EquationParser(str(source))
        self.wall_function_type = None

    @property
    def wallFunctionType(self):
        return self.wall_function_type

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
        elif self.variableName == "friction":
            pass
        else:
            raise Exception("ERROR: non-valid variable name: " + self.variableName + ".")
        return scale_factor, shift_factor

class HeatFluxWF(GeneralWF):
    """
    Sub-class for heat-flux wall function
    """
    def __init__(self, variable, source_expression):
        super().__init__(variable, source_expression)
        self.body_force_type = "HeatFluxWF"

class HeatConvectionWF(GeneralWF):
    """
    Sub-class for heat-convection wall function
    """
    def __init__(self, variable, ambient, heat_transfer_coefficient):
        super().__init__(variable, ambient)
        self.body_force_type = "HeatConvectionWF"
        self._htc = heat_transfer_coefficient

    def getHTC(self):
        return self._htc

class FrictionWF(GeneralWF):
    """
    Sub-class for friction wall functions
    """
    def __init__(self, formulation, default=False):
        super().__init__("friction", 0.0)
        self._formulation = formulation
        self._is_default_friction = default

    @property
    def formulation(self):
        return self._formulation

    @property
    def isDefault(self):
        return self._is_default_friction