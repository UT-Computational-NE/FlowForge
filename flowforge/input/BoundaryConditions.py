from flowforge.input.UnitConverter import UnitConverter

class FBC:
    """
    Super class for the fluid boundary conditions- requires a component.
    """
    def __init__(self, comp, comp_name):
        """
        Initializes the class by storing the component.

        Args:
            - component : Component, component of the BC.
        """
        self._comp = comp
        self._comp_name = comp_name

    @property
    def comp(self):
        """
        Returns the component.
        """
        return self._comp

    @property
    def comp_name(self):
        """
        Returns the component.
        """
        return self._comp_name

class MassTempBC(FBC):
    """
    Represents the boundary condition using mass and temperature.
    """
    def __init__(self, comp_object, component = 'p1', mdot: float = 1.0, T: float = 873.15):
        """
        The MassTempBC subclass initializes by sending the node_id to the base
        class for initialization and the surf_id to the SurfaceBC classes then stores all 2 of the property functions.

        note: the fluidMesh function addBoundarySurface() was created for
            the this boundary condition and is most likely needed
            in all cases and should be the surface referenced by surf_id

        Args:
            - component : Component, component of the BC.
            - surf_id : int, surface index (Boundary Surface)
            - mdot    : float, mass float rate [kg/s]
            - T       : float, surface temperature [K]
        """
        super().__init__(comp_object,component)
        self._mdot = mdot
        self._Tvalue = T

    @property
    def mdot(self):
        """
        Returns the mass float rate.
        """
        return self._mdot

    @property
    def surfaceT(self):
        """
        Returns the surface temperature.
        """
        return self._Tvalue

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Converts the BC units.
        """
        self._mdot *= uc.massFlowRateConversion
        self._Tvalue = uc.temperatureConversion(self._Tvalue)

class PressureTempBC(FBC):
    """
    Represents the boundary condition at the oulet
    """
    def __init__(self, comp_object, component = '', P: float = 101325.0, T: float = 873.15):
        """
        The OutletFlowBC subclass initializes by sending the node_id to the base
        class for initialization and the surf_id to the SurfaceBC classes then stores all 3 of the property functions.

        Args:
            - node_id : int, node index
            - surf_id : int, surface index
            - Pvalue  : float, pressure at surface [Pa]
            - Tvalue  : float, surface temperature [K]
        """
        super().__init__(comp_object,component)
        self._Pvalue = P
        self._Tvalue = T

    @property
    def surfaceP(self):
        """
        Returns the pressure at the surface.
        """
        return self._Pvalue

    @property
    def surfaceT(self):
        """
        Returns the surface temperature.
        """
        return self._Tvalue

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        Converts the BC units.
        """
        self._Pvalue *= uc.pressureConversion
        self._Tvalue = uc.temperatureConversion(self._Tvalue)
