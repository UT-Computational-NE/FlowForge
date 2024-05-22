from flowforge.input.UnitConverter import UnitConverter
from six import add_metaclass
import abc

@add_metaclass(abc.ABCMeta)
class FBC:
    """ Super class for the fluid boundary conditions- requires a component.

    Attributes
    ----------
    comp : Component
        The component that the boundary condition is to be applied to
    comp_name : string
        The component name that the boundary condition is to be applied to
    """
    def __init__(self, comp, comp_name):
        self._comp = comp
        self._comp_name = comp_name

    @property
    def comp(self):
        return self._comp

    @property
    def comp_name(self):
        return self._comp_name

    @abc.abstractmethod
    def _convertUnits(self, uc: UnitConverter) -> None:
        """ Private method for converting units of the component's internal attribute

        This method is especially useful for converting components to the expected units
        of the application in which they will be used.

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        return NotImplementedError

class MassTempBC(FBC):
    """ Class for representing the boundary condition using mass and temperature.

    Attributes
    ----------
    mdot : float
        Mass float rate [kg/s]
    surfaceT : float
        Surface temperature [K]
    """
    def __init__(self, comp_object, component = 'p1', mdot: float = 1.0, T: float = 873.15):
        super().__init__(comp_object,component)
        self._mdot = mdot
        self._surfaceT = T

    @property
    def mdot(self):
        return self._mdot

    @property
    def surfaceT(self):
        return self._surfaceT

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._mdot *= uc.massFlowRateConversion
        self._surfaceT = uc.temperatureConversion(self._surfaceT)

class PressureTempBC(FBC):
    """ Class for representing the boundary condition using pressure and temperature.

    Attributes
    ----------
    _surfaceP : float
        The pressure at the surface [Pa]
    surfaceT : float
        Surface temperature [K]
    """
    def __init__(self, comp_object, component = '', P: float = 101325.0, T: float = 873.15):
        super().__init__(comp_object,component)
        self._surfaceP = P
        self._surfaceT = T

    @property
    def surfaceP(self):
        return self._surfaceP

    @property
    def surfaceT(self):
        return self._surfaceT

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._surfaceP *= uc.pressureConversion
        self._surfaceT = uc.temperatureConversion(self._surfaceT)
