from __future__ import annotations
import abc
from copy import deepcopy
from typing import List, Dict, Tuple, Generator
from six import add_metaclass
import numpy as np
from flowforge.visualization import VTKMesh, genAnnulus, genUniformCube, genCyl, genNozzle
from flowforge.input.UnitConverter import UnitConverter

_CYL_RESOLUTION = 50

# pragma pylint: disable=protected-access, abstract-method

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""
component_list = {}


@add_metaclass(abc.ABCMeta)
class Component:
    """Base class for all components of the system.

    Attributes
    ----------
    flowArea : float
        The component flow area (currently assumed that components have constant flow areas from inlet to outlet)
    length : float
        The component length
    hydraulicDiameter : float
        The component's hydraulic diameter
    heightChange : float
        The height change of the fluid flowing from the inlet to the outlet of the component
    nCell : int
        The number of cells the component consists of
    roughness : float
        The roughness of the component
    Kloss : float
        K-loss coefficient associated with pressure loss through the component
    volume : float
        The flow volume of the component
    inletArea : float
        The inlet area of the component
    outletArea : float
        The outlet area of the component
    """

    def __init__(self) -> None:
        self.uc = None
        self._roughness = 0.0
        self._kloss = 0.0

    @property
    @abc.abstractmethod
    def flowArea(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def length(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hydraulicDiameter(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def heatedPerimeter(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def heightChange(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nCell(self) -> int:
        raise NotImplementedError

    @property
    def roughness(self):
        return self._roughness

    @property
    def kloss(self) -> float:
        return self._kloss

    @property
    def volume(self) -> float:
        return self.flowArea * self.length

    @property
    def inletArea(self) -> float:
        return self.flowArea

    @property
    def outletArea(self) -> float:
        return self.flowArea

    @property
    def baseComponents(self) -> List[Component]:
        """Method for retrieving the base components of a component.
        For components that are not collections, this will be itself"""
        return [self]

    @abc.abstractmethod
    def getMomentumSource(self) -> float:
        """Method for getting the momentum source term of the component

        Returns
        -------
        float
            The magnitude of the component's momentum source
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:  # pylint:disable=unused-argument
        """Method for calculating the outlet coordinates of a components based on the coordinates of the component's inlet

        Parameters
        ----------
        inlet : Tuple[float, float, float]
            The component inlet :math:`(x,y,z)` coordinates from which to calculate the outlet coordinates from

        Returns
        -------
        Tuple[float, float, float]
            The calculated component outlet :math:`(x,y,z)` coordinates
        """
        return NotImplementedError

    @abc.abstractmethod
    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:  # pylint:disable=unused-argument
        """Method for generating a VTK mesh of the component

        Parameters
        ----------
        inlet : Tuple[float, float, float]
            The component inlet :math:`(x,y,z)` coordinates with which to provide a reference
            for the VTK mesh generation

        Returns
        -------
        VTKMesh
            The generated VTK mesh
        """
        return NotImplementedError

    def getNodeGenerator(self) -> Generator[Component, None, None]:
        """Generator for marching over the nodes (i.e. cells) of a component

        This method essentially allows one to march over the nodes of a component
        and be able to reference / use the component said node belongs to

        Yields
        ------
        Component
            The component associated with the node the generator is currently on (i.e. self)
        """
        for _ in range(self.nCell):
            yield self

    @abc.abstractmethod
    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        """Method for retrieving the bounding box information of a component

        Parameters
        ----------
        inlet : Tuple[float, float, float]
            The component inlet :math:`(x,y,z)` coordinates with which to provide a reference
            for the bounding box generation

        Returns
        -------
        Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]
            The parameters describing the bounding box of this component.
            (inlet_coordinate, outlet_coordinate, :math:`x-width/2`, :math:`y-width/2`, :math:`z-length`,
            :math:`\theta-angle`, :math:`\alpha-angle`)
        """
        return NotImplementedError

    @abc.abstractmethod
    def _convertUnits(self, uc: UnitConverter) -> None:
        """Private method for converting units of the component's internal attribute

        This method is especially useful for converting components to the expected units
        of the application in which they will be used.

        Parameters
        ----------
        uc : UnitConverter
            A unit converter which holds the 'from' units and 'to' units for the conversion
            and will ultimately provide the appropriate multipliers for unit conversion.
        """
        return NotImplementedError

    def rotate(self, x: float, y: float, z: float, theta: float = 0.0, alpha: float = 0.0) -> np.ndarray:
        """Method for rotating a point about the :math:`y-axis` and :math:`z-axis`

        Parameters
        ----------
        x : float
            The :math:`x-component` of the point to be rotated
        y : float
            The :math:`y-component` of the point to be rotated
        z : float
            The :math:`z-component` of the point to be rotated
        theta : float
            The rotation angle about the :math:`y-axis` (radians)
        alpha : float
            The rotation angle about the :math:`z-axis` (radians)

        Returns
        -------
        ndarray
            The new rotated :math:`(x,y,z)` coordinates
        """
        polar_rotate = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        azimuthal_rotate = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        new_vec = np.dot(azimuthal_rotate, np.dot(polar_rotate, np.array([x, y, z])))
        return new_vec


def component_factory(indict: Dict) -> Dict[str, Component]:
    """Factory for building a collection of components

    Parameters
    ----------
    indict : Dict
        The input dictionary specifying the components to be instantiated.  This dictionary can be comprised
        of two forms of inputs:

        1.) A Name-Component pair
            (Dict[str, Component])
        2.) A dictionary of component types, each type holding a dictionary of name-parameter_set pairs, with the
            name being the unique component's name, and the parameter_set another dictionary with key's corresponding
            to the __init__ signature of the associated component type
            (Dict[str, Dict[str, Dict[str, float]]])

    Returns
    -------
    Dict[str, Component]
        The collection of components built
        (key: Component name, value: Component object)
    """
    components = {}
    for key, value in indict.items():
        if isinstance(value, dict):
            comp_type = key
            comps = value
            if comp_type in component_list:
                for name, parameters in comps.items():
                    components[name] = component_list[comp_type](**parameters)
            else:
                raise TypeError("Unknown component type: " + comp_type)
        elif isinstance(value, Component):
            name = key
            comp = value
            components[name] = comp
        else:
            raise TypeError(f"Unknown input dictionary: {key:s} type: {str(type(value)):s}")

    return components


class Pipe(Component):
    """A pipe component

    Parameters
    ----------
    L : float
        Length of the pipe
    R : float
        Inner radius of the pipe
    Ac : float
        Flow area of the pipe
    Dh : float
        Hydraulic diameter of the pipe
    n : int
        Number of segments the pipe is divided into
    theta : float
        Orientation angle of the pipe in the polar direction
    alpha : float
        Orientation angle of the pipe in the azimuthal direction
    Kloss : float
        K-loss coefficient associated with pressure loss through the pipe
    roughness : float
        Pipe roughness
    pctHeated: float
        Fraction of the pipe that is heated. Used for calculating heated perimeter.
        Defaults to 1 (i.e. the entire pipe is heated)
    """

    def __init__(
        self,
        L: float,
        R: float = None,
        Ac: float = None,
        Dh: float = None,
        n: int = 1,
        theta: float = 0.0,
        alpha: float = 0.0,
        Kloss: float = 0,
        roughness: float = 0.0,
        pctHeated: float = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self._L = L
        self._n = n
        self._costh = np.cos(np.pi / 180 * theta)

        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._kloss = Kloss
        self._roughness = roughness
        self._kwargs = kwargs
        if R is None:
            assert Dh is not None and Ac is not None
            self._R = 0.5 * Dh
        else:
            self._R = R
        if Ac is None:
            self._Ac = np.pi * R * R
        else:
            self._Ac = Ac
        if Dh is None:
            self._Pw = 2 * np.pi * self._R
            self._Dh = 4.0 * self._Ac / self._Pw
        else:
            self._Dh = Dh
            self._Pw = 4.0 * self._Ac / self._Dh

        self._heatedPerimeter = pctHeated * self._Pw

        self._temps = np.zeros(self.nCell)

    @property
    def flowArea(self) -> float:
        return self._Ac

    @property
    def length(self) -> float:
        return self._L

    @property
    def hydraulicDiameter(self) -> float:
        return self._Dh

    @property
    def heatedPerimeter(self) -> float:
        return self._heatedPerimeter

    @property
    def heightChange(self) -> float:
        return self._costh * self._L

    @property
    def nCell(self) -> int:
        return self._n

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genCyl(self._L, self._R, nlayers=self._n, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._R, self._R, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._L *= uc.lengthConversion
        self._R *= uc.lengthConversion
        self._Ac *= uc.areaConversion
        self._Dh *= uc.lengthConversion
        self._Pw *= uc.lengthConversion
        self._roughness *= uc.lengthConversion
        self._heatedPerimeter *= uc.lengthConversion


component_list["pipe"] = Pipe


class SquarePipe(Pipe):
    """A square pipe component

    Parameters
    ----------
    L : float
        Length of the pipe
    W : float
        Width of the pipe (i.e. flat-to-flat)
    """

    def __init__(self, L: float, W: float, **kwargs) -> None:
        super().__init__(L=L, Dh=W, Ac=W**2, **kwargs)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genUniformCube(self._Dh, self._Dh, self._L, nz=self._n).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )


component_list["square_pipe"] = SquarePipe


class Tee(Pipe):
    """A Tee pipe component

    To be implemented

    Parameters
    ----------
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self._n == 1


component_list["tee"] = Tee


class SimpleTank(Pipe):
    """
    The SimpleTank class is a de facto tee component that contains two nodes

    The primary node connects east, west, and transverse surfaces.
    The transverse node is connected to the primary node and enforces a pressure and scalar
    boundary condition at the outer surface.

    NOTE : The number of cells is 1 for this component, but it actually contains a second node
    corresponding to the transverse element. In the future, this transverse node will be associated
    with the transverse component instance (i.e., the transverse pipe connected to the tee junction)
    instead.

    Parameters
    ----------
    transverse_area : float
        The cross sectional area of the transverse component/node/surface
    transverse_height : float
        The height of the transverse component/node
    """

    def __init__(self, transverse_area : float, transverse_height : float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transverseArea = transverse_area
        self.transverseHeight = transverse_height

    def _convertUnits(self, uc: UnitConverter) -> None:
        super()._convertUnits(uc)
        self.transverseArea *= uc.areaConversion
        self.transverseHeight *= uc.lengthConversion


component_list["simple_tank"] = SimpleTank

class Pump(Component):
    """A pump component

    Parameters
    ----------
    Ac : float
        Pump flow area
    Dh : float
        Pump Hydraulic Diameter
    V : float
        Pump flow volume
    height : float
        Change in height experienced by the fluid across the pump
    dP : float
        Change in pressure of the fluid caused by the pump
    roughness : float
        Pump roughness
    ptcHeated : float
        Fraction of the pump perimeter that is heated. Uses Dh to determine wetted perimeter
    Kloss : float
        K-loss coefficient associated with pressure loss through the pump
    """

    def __init__(
        self,
        Ac: float,
        Dh: float,
        V: float,
        height: float,
        dP: float,
        roughness: float = 0.0,
        ptcHeated: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self._Ac = Ac
        self._Dh = Dh
        self._V = V
        self._h = height
        self._dP = dP
        self._roughness = roughness
        self._kwargs = kwargs
        self._temps = np.zeros(self.nCell)

        wettedPerimeter = np.pi * Dh
        self._heatedPerimeter = ptcHeated * wettedPerimeter

    @property
    def flowArea(self) -> float:
        return self._Ac

    @property
    def length(self) -> float:
        return self._V / self._Ac

    @property
    def hydraulicDiameter(self) -> float:
        return self._Dh

    @property
    def heatedPerimeter(self) -> float:
        return self._heatedPerimeter

    @property
    def heightChange(self) -> float:
        return self._h

    @property
    def volume(self) -> float:
        return self._V

    @property
    def nCell(self) -> int:
        return 1

    def getMomentumSource(self) -> float:
        return self._dP * self._Ac

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        # for now making the assumption that it comes in bottom and out the side of the pump
        x = inlet[0] + self._Dh / 2
        y = inlet[1]
        z = inlet[2] + self.heightChange / 2
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genUniformCube(self._Dh, self._Dh, self._h, **self._kwargs).translate(inlet[0], inlet[1], inlet[2])

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        """Method for retrieving the bounding box information of a component

        Pumps are assumed to have orientations inline with the cartesian grid

        Parameters
        ----------
        inlet : Tuple[float, float, float]
            The component inlet :math:`(x,y,z)` coordinates with which to provide a reference
            for the bounding box generation

        Returns
        -------
        Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]
            The parameters describing the bounding box of this component.
            (inlet_coordinate, outlet_coordinate, :math:`x-width/2`, :math:`y-width/2`, :math:`z-length`,
            :math:`\theta-angle`, :math:`\alpha-angle`)
        """
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._Dh / 2, self._Dh / 2, self._h, 0.0, 0.0]

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._Ac *= uc.areaConversion
        self._Dh *= uc.lengthConversion
        self._V *= uc.volumeConversion
        self._h *= uc.lengthConversion
        self._dP *= uc.pressureConversion
        self._roughness *= uc.lengthConversion
        self._heatedPerimeter *= uc.lengthConversion


component_list["pump"] = Pump


class Nozzle(Component):
    """A nozzle component

    Parameters
    ----------
    L : float
        Length of the nozzle
    R_inlet : float
        Radius of the nozzle inlet
    R_outlet : float
        Radius of the nozzle outlet
    theta : float
        Orientation angle in the polar direction
    alpha : float
        Orientation angle in the azimuthal direction
    roughness : float
        Nozzle roughness
    Kloss : float
        K-loss coefficient associated with pressure loss through the nozzle
    ptcHeated : float
        Fraction of the nozzle that is heated. Used for calculating heated perimeter at the center of the nozzle.
    """

    def __init__(
        self,
        L: float,
        R_inlet: float,
        R_outlet: float,
        theta: float = 0.0,
        alpha: float = 0.0,
        kloss: float = 0.0,
        roughness: float = 0.0,
        ptcHeated: float = 1,
        resolution: int = _CYL_RESOLUTION,
        **kwargs,
    ) -> None:
        super().__init__()
        self._L = L
        self._Rin = R_inlet
        self._Rout = R_outlet
        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._Dh = self._Rin + self._Rout  # average radius needs div 2 but radius to diameter needs mult 2 so they cancel
        self._Ac = 0.25 * np.pi * self._Dh * self._Dh
        self._res = resolution
        self._kloss = kloss
        self._roughness = roughness
        self._kwargs = kwargs
        self._temps = np.zeros(self.nCell)

        wetted_perimeter = np.pi * (self._Rin + self._Rout) # 2 pi r and 1/2 from average also cancels
        self._heatedPerimeter = ptcHeated * wetted_perimeter

    @property
    def flowArea(self) -> float:
        return self._Ac

    @property
    def inletArea(self) -> float:
        return np.pi * self._Rin * self._Rin

    @property
    def outletArea(self) -> float:
        return np.pi * self._Rout * self._Rout

    @property
    def length(self) -> float:
        return self._L

    @property
    def hydraulicDiameter(self) -> float:
        return self._Dh

    @property
    def heatedPerimeter(self) -> float:
        return self._heatedPerimeter

    @property
    def heightChange(self) -> float:
        return self._L * np.cos(self._theta)

    @property
    def nCell(self) -> int:
        return 1

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genNozzle(self._L, self._Rin, self._Rout, resolution=self._res, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        outlet = self.getOutlet(inlet)
        if self._Rin >= self._Rout:
            big_r = self._Rin
        else:
            big_r = self._Rout
        return [inlet, outlet, big_r, big_r, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._L *= uc.lengthConversion
        self._Rin *= uc.lengthConversion
        self._Rout *= uc.lengthConversion
        self._Dh *= uc.lengthConversion
        self._Ac *= uc.areaConversion
        self._roughness *= uc.lengthConversion
        self._heatedPerimeter *= uc.lengthConversion


component_list["nozzle"] = Nozzle


class Annulus(Component):
    """A annulus component

    Parameters
    ----------
    L : float
        Length of the annulus
    R_inner : float
        Inner radius of the annulus
    R_outer : float
        Outer radius of the annulus
    n : int
        Number of layers in annulus
    theta : float
        Orientation angle in the polar direction
    alpha : float
        Orientation angle in the azimuthal direction
    roughness : float
        Annulus roughness
    Kloss : float
        K-loss coefficient associated with pressure loss through the annulus
    ptcHeated : float
        Fraction of the annulus wetted perimeter that is heated. Used for calculating heated perimeter.
    resolution : int
        Number of sides the annulus curvature is approximated with (specifically for VTK mesh generation)
    """

    def __init__(
        self,
        L: float,
        R_inner: float,
        R_outer: float,
        n: int = 1,
        theta: float = 0.0,
        alpha: float = 0.0,
        roughness: float = 0.0,
        ptcHeated: float = 1,
        resolution: int = _CYL_RESOLUTION,
        **kwargs,
    ) -> None:
        super().__init__()
        self._L = L
        self._Rin = R_inner
        self._Rout = R_outer
        self._n = n
        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._res = resolution
        self._roughness = roughness
        self._kwargs = kwargs
        self._temps = np.ones(self.nCell)

        wettedPerimeter = 2 * np.pi * (self._Rout + self._Rin)
        self._heatedPerimeter = ptcHeated * wettedPerimeter

    @property
    def flowArea(self) -> float:
        return np.pi * (self._Rout * self._Rout - self._Rin * self._Rin)

    @property
    def length(self) -> float:
        return self._L

    @property
    def hydraulicDiameter(self) -> float:
        return 2 * self.flowArea / (np.pi * (self._Rin + self._Rout))

    @property
    def heatedPerimeter(self) -> float:
        return self._heatedPerimeter

    @property
    def heightChange(self) -> float:
        return self._L * np.cos(self._theta)

    @property
    def nCell(self) -> int:
        return self._n

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genAnnulus(self._L, self._Rin, self._Rout, resolution=self._res, nlayers=self._n, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._Rout, self._Rout, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._L *= uc.lengthConversion
        self._Rin *= uc.lengthConversion
        self._Rout *= uc.lengthConversion
        self._roughness *= uc.lengthConversion
        self._heatedPerimeter *= uc.lengthConversion


component_list["annulus"] = Annulus


class Tank(Component):
    """A tank component

    Parameters
    ----------
    L : float
        Length of the tank
    R : float
        Radius of the tank
    n : int
        Number of segments tank is divided into
    theta : float
        Orientation angle of the tank in the polar direction
    alpha : float
        Orientation angle of the tank in the azimuthal direction
    roughness : float
        Tank roughness
    Kloss : float
        K-loss coefficient associated with pressure loss through the tank
    """

    def __init__(
        self,
        L: float,
        R: float,
        n: int = 1,
        theta: float = 0.0,
        alpha: float = 0.0,
        roughness: float = 0.0,
        ptcHeated: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self._Ac = np.pi * R * R
        self._Pw = 2.0 * np.pi * R
        self._Dh = 4.0 * self._Ac / self._Pw
        self._L = L
        self._n = n
        self._costh = np.cos(np.pi / 180 * theta)
        self._R = R
        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._roughness = roughness
        self._kwargs = kwargs
        self._temps = np.zeros(self.nCell)

        self._heatedPerimeter = ptcHeated * self._Pw

    @property
    def flowArea(self) -> float:
        return self._Ac

    @property
    def length(self) -> float:
        return self._L

    @property
    def hydraulicDiameter(self) -> float:
        return self._Dh

    @property
    def heatedPerimeter(self) -> float:
        return self._heatedPerimeter

    @property
    def heightChange(self) -> float:
        return self._costh * self._L

    @property
    def nCell(self) -> int:
        return self._n

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha) + self._R * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha) + self._R * np.sin(self._alpha)
        z = inlet[2]
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genCyl(self._L, self._R, resolution=_CYL_RESOLUTION, nlayers=self._n, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._R, self._R, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._Ac *= uc.areaConversion
        self._Pw *= uc.lengthConversion
        self._Dh *= uc.lengthConversion
        self._L *= uc.lengthConversion
        self._R *= uc.lengthConversion
        self._roughness *= uc.lengthConversion


component_list["tank"] = Tank


class ComponentCollection(Component):
    """An abstract class for a classes that manage multiple components

    Parameters
    ----------
    components : Dict[str, Component]
        Collection of already initialized components
    """

    def __init__(self, components: Dict[str, Component]) -> None:
        super().__init__()
        self._myComponents = components

    @property
    def flowArea(self) -> float:
        return NotImplementedError

    @property
    def volume(self) -> float:
        return NotImplementedError

    @property
    def inletArea(self) -> float:
        return NotImplementedError

    @property
    def outletArea(self) -> float:
        return NotImplementedError

    @property
    def length(self) -> float:
        return NotImplementedError

    @property
    def hydraulicDiameter(self) -> float:
        raise NotImplementedError

    @property
    def heatedPerimeter(self) -> float:
        raise NotImplementedError

    @property
    def heightChange(self) -> float:
        raise NotImplementedError

    @property
    def myComponents(self) -> List[Component]:
        return self._myComponents

    @property
    def baseComponents(self) -> List[Component]:
        """Method for retrieving the base components (components that are not Component collections)
        of a component collection"""
        base_components = []
        for component in self.myComponents.values():
            base_components.extend(component.baseComponents)
        return base_components

    @property
    @abc.abstractmethod
    def firstComponent(self) -> Component:
        """Return the first component in the collection.
        Implemented in the derived class.
        """
        return NotImplementedError

    @property
    @abc.abstractmethod
    def lastComponent(self) -> Component:
        """Return the last component in the collection.
        Implemented in the derived class.
        """
        return NotImplementedError

    def getNodeGenerator(self) -> Generator[Component, None, None]:
        yield from [component.getNodeGenerator() for component in self._myComponents.values()]

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def _convertUnits(self, uc: UnitConverter) -> None:
        for component in self._myComponents.values():
            component._convertUnits(uc)


class ParallelComponents(ComponentCollection):
    """A component for a collection of components through which flow passes in parallel

    Parameters
    ----------
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`component_factory`
    centroids : Dict[str, List[float]]
        The :math:`x-y` coordinates of the centroids of the parallel components.
        (key: component name, value: [x_coord, y_coord])
    lower_plenum : Dict[str, Dict[str,float]]
        The component specifications for the lower plenum
        (key: component type, value: component parameters dictionary)
    upper_plenum : Dict[str, Dict[str,float]]
        The component specifications for the upper plenum
        (key: component type, value: component parameters dictionary)
    annulus : Dict[str, Dict[str,float]]
        The component specifications for the annulus
        (key: component type, value: component parameters dictionary)

    Attributes
    ----------
    myComponents : List[Component]
        The collection of parallel components
    centroids : Dict[str, List[float]]
        The centroids of the parallel components
    lowerPlenum : Component
        The lower plenum of the parallel components
    upperPlenum : Component
        The upper plenum of the parallel components
    annulus : Component
        The annulus of the parallel components
    length : float
        The length of the parallel components starting from the
        lower plenum inlet, through to the upper plenum outlet
        (all parallel components are assumed to have the same length)
    nCell : int
        The total number of cells of all parallel components
    inletArea : float
        The inlet area of the lower plenum
    outletArea : float
        The outlet area of the upper plenum
    """

    def __init__(
        self,
        parallel_components: Dict,
        centroids: Dict[str, List[float]],
        lower_plenum: Dict[str, Dict[str, float]],
        upper_plenum: Dict[str, Dict[str, float]],
        annulus: Dict[str, Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        self._myParallelComponents = component_factory(parallel_components)
        self._centroids = centroids

        assert len(lower_plenum) == 1
        comp_type, parameters = list(lower_plenum.items())[0]
        self._lowerPlenum = component_list[comp_type](**parameters)

        assert len(upper_plenum) == 1
        comp_type, parameters = list(upper_plenum.items())[0]
        self._upperPlenum = component_list[comp_type](**parameters)

        myComponents = {
            **self._myParallelComponents,
            "lower_plenum": self._lowerPlenum,
            "upper_plenum": self._upperPlenum,
        }

        if annulus is None:
            self._annulus = None
        else:
            assert len(annulus) == 1
            comp_type, parameters = list(annulus.items())[0]
            self._annulus = component_list[comp_type](**parameters)
            myComponents["annulus"] = self._annulus

        self._kwargs = kwargs
        super().__init__(myComponents)

    @property
    def firstComponent(self) -> Component:
        return self._lowerPlenum

    @property
    def lastComponent(self) -> Component:
        return self._upperPlenum

    @property
    def flowArea(self) -> float:
        return NotImplementedError

    @property
    def volume(self) -> float:
        return NotImplementedError

    @property
    def inletArea(self) -> float:
        return self._lowerPlenum.inletArea

    @property
    def outletArea(self) -> float:
        return self._upperPlenum.outletArea

    @property
    def length(self) -> float:
        L = self._lowerPlenum.length + self._upperPlenum.length
        L += self._myParallelComponents[list(self._myParallelComponents.keys())[0]].length
        return L

    @property
    def hydraulicDiameter(self) -> float:
        raise NotImplementedError

    @property
    def heightChange(self) -> float:
        raise NotImplementedError

    @property
    def nCell(self) -> int:
        ncell = self._lowerPlenum.nCell + self._upperPlenum.nCell
        if self._annulus is not None:
            ncell += self._annulus.nCell
        for cname in self._centroids.keys():
            ncell += self._myParallelComponents[cname].nCell
        return ncell

    @property
    def myParallelComponents(self) -> Dict[str, Component]:
        """Returns only the parallel components (not including the lower plenum, upper plenum, or annulus)."""
        return self._myParallelComponents

    @property
    def centroids(self) -> Dict[str, List[float]]:
        return self._centroids

    @property
    def lowerPlenum(self) -> Component:
        return self._lowerPlenum

    @property
    def upperPlenum(self) -> Component:
        return self._upperPlenum

    @property
    def annulus(self) -> Component:
        return self._annulus

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        lower = self._lowerPlenum.getOutlet(inlet)
        firstcomp = list(self._myParallelComponents.items())[0][0]
        outlet = self._myParallelComponents[firstcomp].getOutlet(lower)
        outlet = (round(outlet[0], 5), round(outlet[1], 5), round(outlet[2], 5))
        for comp in self._myParallelComponents.values():
            compOutlet = comp.getOutlet(lower)
            compOutletRounded = round(compOutlet[0], 5), round(compOutlet[1], 5), round(compOutlet[2], 5)
            assert compOutletRounded == outlet
        outlet = self._upperPlenum.getOutlet(outlet)
        return outlet

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        mesh = VTKMesh()
        mesh += self._lowerPlenum.getVTKMesh(inlet)
        inlet2 = self._lowerPlenum.getOutlet(inlet)
        if self._annulus is not None:
            mesh += self._annulus.getVTKMesh(inlet2)
        for cname, centroid in self._centroids.items():
            i = (inlet2[0] + centroid[0], inlet2[1] + centroid[1], inlet2[2])
            mesh += self._myParallelComponents[cname].getVTKMesh(i)
        inlet2 = list(self._myParallelComponents.items())[0][1].getOutlet(inlet2)
        mesh += self._upperPlenum.getVTKMesh(inlet2)
        return mesh

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        raise NotImplementedError


component_list["parallel_components"] = ParallelComponents


class HexCore(ParallelComponents):
    """A hexagonal core component

    Parameters
    ----------
    pitch : float
        Distance between each of the fuel channels (serial components)
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`component_factory`
    hexmap : List[List[int]]
        list containing the serial components in the corresponding rows and
        columns of the hex map
    lower_plenum : Dict[str, Dict[str,float]]
        The component specifications for the lower plenum
        (key: component type, value: component parameters dictionary)
    upper_plenum : Dict[str, Dict[str,float]]
        The component specifications for the upper plenum
        (key: component type, value: component parameters dictionary)
    annulus : Dict[str, Dict[str,float]]
        The component specifications for the annulus
        (key: component type, value: component parameters dictionary)


    Attributes
    ----------
    myComponents : List[Component]
        The collection of parallel components
    centroids : Dict[str, List[float]]
        The centroids of the parallel components
    lowerPlenum : Component
        The lower plenum of the parallel components
    upperPlenum : Component
        The upper plenum of the parallel components
    annulus : Component
        The annulus of the parallel components
    length : float
        The length of the parallel components starting from the
        lower plenum inlet, through to the upper plenum outlet
        (all parallel components are assumed to have the same length)
    nCell : int
        The total number of cells of all parallel components
    inletArea : float
        The inlet area of the lower plenum
    outletArea : float
        The outlet area of the upper plenum
    """

    def __init__(
        self,
        pitch: float,
        components: Dict,
        hexmap: List[List[int]],
        lower_plenum: Dict[str, Dict[str, float]],
        upper_plenum: Dict[str, Dict[str, float]],
        annulus: Dict[str, Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        self._pitch = pitch
        self._map = hexmap

        self.tmpComponents = component_factory(components)
        extended_comps = {}
        centroids = {}

        for r, col in enumerate(self._map):
            for c, val in enumerate(col):
                cname = f"{str(val):s}-{r + 1:d}-{c + 1:d}"
                yc, xc = self._getChannelCoords(r, c)
                centroids[cname] = [xc, yc]
                extended_comps[cname] = deepcopy(self.tmpComponents[str(val)])

        super().__init__(extended_comps, centroids, lower_plenum, upper_plenum, annulus, **kwargs)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:

        core_inlet = self._lowerPlenum.getOutlet(inlet)
        channels = self.tmpComponents[list(self.tmpComponents.keys())[0]]._myComponents
        for c in channels.keys():
            channels[c]._convertUnits(self.uc)
            if c == "plate":
                core_inlet = channels[c].getOutlet(core_inlet)
        mesh = VTKMesh()
        mesh += super().getVTKMesh(inlet)
        return mesh

    def _getChannelCoords(self, r: int, c: int) -> Tuple[float, float]:
        """Private method which returns the calculated coordinates of map locations

        This method returns the :math:`x-y` coordinates of the particular location in the map,
        which is determined by r (row) and c (col).  This method is called repeatedly from the
        getVTKMesh function and is used to determine the inlet input for the mesh generation and
        translation.

        Parameters
        ----------
        r : int
            Row in the hexagonal map
        c : int
            Column in the hexagonal map

        Returns
        -------
        Tuple[float, float]
            The :math:`x-y` coordinates corresponding to the specified map location
        """

        dx = self._pitch
        dy = self._pitch * np.sqrt(3) / 2

        rc = len(self._map) / 2
        yoffset = 0.5 * (0.5 * self._pitch * np.tan(np.pi / 6) - 0.5 * self._pitch / np.cos(np.pi / 6))
        if len(self._map) % 2 == 0:
            yoffset += -0.5 * dy  # even rows
        yc = dy * (rc - r) + yoffset

        cc = np.floor(len(self._map[r]) / 2)
        xoffset = 0.0
        if len(self._map[r]) % 2 == 0:
            xoffset = 0.5 * dx  # even columns
        xc = dx * (c - cc) + xoffset

        return xc, yc

    def _convertUnits(self, uc: UnitConverter) -> None:
        self.uc = uc
        self._pitch *= uc.lengthConversion
        super()._convertUnits(uc)


component_list["hex_core"] = HexCore


class SerialComponents(ComponentCollection):
    """A component for a collection of components through which flow passes in serial (i.e. from one component into the next)

    Parameters
    ----------
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`component_factory`
    order : List[str]
        The order of the components listed in order from start to finish using the unique component names

    Attributes
    ----------
    myComponents : List[Component]
        The collection of serial components
    order : List[str]
        The ordering of the serial components from start to finish using the unique component names
    flowArea : float
        The flow area of the serial components
        (currently assumed that components have constant flow areas from inlet to outlet)
    length : float
        The total length of the serial components
    hydraulicDiameter : float
        The hydraulic diameter of the serial components
    nCell : int
        The total number of cells of the serial components
    volume : float
        The total volume of the serial components
    inletArea : float
        The inlet area of the serial components
    outletArea : float
        The outlet area of the serial component
    """

    def __init__(self, components: Dict[str, Dict[str, float]], order: List[str], **kwargs) -> None:
        super().__init__(component_factory(components))
        self._order = order
        self._kwargs = kwargs

    @property
    def firstComponent(self) -> Component:
        """Always returns the first component.
        If the first component is a collection, it will return the first component of that collection recursively"""
        first_component = self._myComponents[self._order[0]]
        if isinstance(first_component, ComponentCollection):
            return first_component.firstComponent
        return first_component

    @property
    def lastComponent(self) -> Component:
        """Always returns the last component.
        If the last component is a collection, it will return the last component of that collection recursively"""
        last_component = self._myComponents[self._order[-1]]
        if isinstance(last_component, ComponentCollection):
            return last_component.lastComponent
        return last_component

    @property
    def orderedComponentsList(self) -> List[Component]:
        """Returns a list of the components in the order they are listed in the order attribute."""
        return [self._myComponents[comp_key] for comp_key in self.order]

    @property
    def flowArea(self) -> float:
        return self._myComponents[self._order[0]].flowArea

    @property
    def inletArea(self) -> float:
        return self._myComponents[self._order[0]].inletArea

    @property
    def outletArea(self) -> float:
        return self._myComponents[self._order[-1]].outletArea

    @property
    def length(self) -> float:
        L = 0
        for c in self._myComponents.values():
            L += c.length
        return round(L, 5)

    @property
    def hydraulicDiameter(self) -> float:
        names = list(self._myComponents.keys())
        for c in names:
            if c[0] == "c":
                return round(self._myComponents[c]._R * 2, 6)
        raise Exception("Component with hydraulic diameter not found.")

    @property
    def heightChange(self) -> float:
        raise NotImplementedError

    @property
    def nCell(self) -> int:
        ncell = 0
        for cname in self._order:
            ncell += self._myComponents[cname].nCell
        return ncell

    @property
    def order(self) -> List[str]:
        return self._order

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for cname in self._order:
            inlet = self._myComponents[cname].getOutlet(inlet)
        return inlet

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        mesh = VTKMesh()
        for cname in self._order:
            mesh += self._myComponents[cname].getVTKMesh(inlet)
            inlet = self._myComponents[cname].getOutlet(inlet)
        return mesh

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        raise NotImplementedError


component_list["serial_components"] = SerialComponents
