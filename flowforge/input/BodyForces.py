import abc
from copy import deepcopy
from typing import Union
from flowforge.input.UnitConverter import UnitConverter
from flowforge.parsers.EquationParser import EquationParser
import flowforge.input.Components as FluidComps
import flowforge.input.SolidComponents as SolidComps
from flowforge.input.System import SimulationType

SolidComponent = SolidComps.SolidComponent
FluidComponent = FluidComps.Component
GeneralComponent = Union[FluidComponent, SolidComponent]

class BodyForces:
    """
    """
    def __init__(self,
                 simulation_type: SimulationType):
        self._simulation_type = simulation_type
        self._body_forces = {}

    @property
    def valid_bf_objects(self) -> dict:
        return {
            "InternalHeatGeneration" : InternalHeatGenerationBF,
            "heat_generation"        : InternalHeatGenerationBF,
            "DifferentialPressure"   : DifferentialPressureBF,
            "dP"                     : DifferentialPressureBF
        }
    
    @property
    def body_forces(self) -> dict:
        return self._body_forces

    @property
    def simulation_type(self) -> SimulationType:
        return self._simulation_type

    def rename_body_force(self, name: str) -> str:
        """
        """
        n_duplicates = 0
        for bf in self.body_forces:
            if name in bf:
                n_duplicates += 1
        return name + f"_{n_duplicates}"

    def addBodyForce(self,
                     name: str,
                     body_force: dict,
                     component: GeneralComponent,
                     allow_duplicates: bool = True) -> None:
        """
        """
        if not allow_duplicates:
            assert name not in self._bfs, (
                f"Already have a body force '{name}' defined."
            )
        else:
            name = self.rename_body_force(name)

        # Get the body force base object
        bf_obj = self.valid_bf_objects[body_force["type"]]
        # Extract and convert the input value
        input_value = EquationParser(str(body_force["value"]))
        # Add built body force
        self._body_forces[name] = bf_obj(input_value)
        # Attach the input component
        self._body_forces[name].attach_component(component)

    def _convertUnits(self, uc: UnitConverter):
        converted_bfs = {}
        for bf_name, bf in self.body_forces.items():
            bf.convertUnits(uc)
            converted_bfs[bf_name] = deepcopy(bf)
        self.body_forces = converted_bfs
class GeneralBodyForce:
    """
    Body force parent object
    """
    def __init__(self, value: EquationParser):
        self._value = value
        self._variable_name = None
        self._body_force_type = None
        self._component = None
        
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

    @property
    def component(self) -> GeneralComponent:
        return self._component

    def attach_component(self, component: GeneralComponent):
        """
        Attaches the reference to a component to this body force

        Parameters
        ----------
        component : Union[FluidComponent, SolidComponent]
            Component that this body force is associated with
        """
        self._component = component

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

class InternalHeatGenerationBF(GeneralBodyForce):
    """
    Body force for an internal heat generation
    """
    def __init__(self, value):
        super().__init__(value)
        self.body_force_type = "InternalHeatGeneration"
        self._variable_name = "power_density"

class DifferentialPressureBF(GeneralBodyForce):
    """
    Body force for a differential pressure
    """
    def __init__(self, value):
        super().__init__(value)
        self.body_force_type = "DifferentialPressure"
        self._variable_name = "pressure"
