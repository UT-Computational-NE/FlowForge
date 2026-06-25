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

class WallFunctions:
    """
    """
    def __init__(self,
                 simulation_type: SimulationType):
        self._simulation_type = simulation_type
        self._wall_functions = {}

    @property
    def valid_wf_objects(self) -> dict:
        return {
            "HeatFlux"        : HeatFluxWF,
            "heat_flux"       : HeatFluxWF,
            "Frictionless"    : FrictionlessWF,
            "DefaultFriction" : DefaultFrictionWF
        }

    @property
    def wall_functions(self) -> dict:
        return self._wall_functions

    @property
    def simulation_type(self) -> SimulationType:
        return self._simulation_type

    def rename_wall_function(self, name: str) -> str:
        """
        """
        n_duplicates = 0
        for bf in self.wall_functions:
            if name in bf:
                n_duplicates += 1
        return name + f"_{n_duplicates}"

    def addWallFunction(self,
                        name: str,
                        wall_function: dict,
                        component: GeneralComponent,
                        allow_duplicates: bool = True) -> None:
        """
        """
        if not allow_duplicates:
            assert name not in self._bfs, (
                f"Already have a wall function '{name}' defined."
            )
        else:
            name = self.rename_body_force(name)

        # Get the wall function base object
        wf_obj = self.valid_wf_objects[wall_function["type"]]
        # Extract and convert the input value
        input_value = EquationParser(str(wall_function.get("value", 0.0)))
        # Add the built wall function
        self._wall_functions[name] = wf_obj(wall_function.get("surface", "wall"), input_value)
        # Attach the input component
        self._wall_functions[name].attach_component(component)

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
    wall_function_value : EquationParser
        Function that, when evaluated, gives the source value of the wall function
    variable_name : str
        Variable associated with this wall function
    """

    def __init__(self, surface: str, value: EquationParser):
        self._surface_name = surface
        self._value = value
        self._variable_name = None
        self._wall_function_type = None
        self._component = None

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
    def wall_function_value(self) -> EquationParser:
        return self._value

    @wall_function_value.setter
    def wall_function_value(self, value) -> None:
        self._value = value

    @property
    def variable_name(self) -> str:
        return self._variable_name

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
        self.wall_function_value.performUnitConversion(scale_factor, shift_factor)


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
    wall_function_value : EquationParser
        Function that, when evaluated, gives the source value of the wall function
    variable_name : str
        Variable associated with this wall function
    """

    def __init__(self, surface, heat_flux):
        super().__init__(surface, heat_flux)
        self.wall_function_type = "HeatFlux"
        self._variable_name = "power_flux"

class ConvectiveHeatTransferWF(GeneralWF):
    """
    """
    def __init__(self, surface, value):
        super().__init__(surface, value)
        self.wall_function_type = "ConvectiveHeatTransfer"
        self._variable_name = "temperature"

class FrictionBaseWF(GeneralWF, abc.ABC):
    """
    """
    def __init__(self, surface, value):
        super().__init__(surface, value)
        self.wall_function_type = "FrictionBase"

    def convertUnits(self, uc: UnitConverter) -> None:
        """
        """
        return

class DefaultFrictionWF(FrictionBaseWF):
    """
    """
    def __init__(self, surface, value):
        super().__init__(surface, value)
        self.wall_function_type = "DefaultFriction"

class FrictionlessWF(FrictionBaseWF):
    """
    """
    def __init__(self, surface, value):
        super().__init__(surface, value)
        self.wall_function_type = "Frictionless"