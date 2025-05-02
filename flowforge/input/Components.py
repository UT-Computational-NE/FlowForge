from __future__ import annotations
import abc
from copy import deepcopy
from typing import List, Dict, Tuple, Generator, Optional, Literal, TypeAlias
from six import add_metaclass
import numpy as np
from flowforge.visualization import VTKMesh, genUniformAnnulus, genUniformCube, genUniformCylinder, genNozzle
from flowforge.input.UnitConverter import UnitConverter

_CYL_RESOLUTION = 50

# pragma pylint: disable=protected-access, abstract-method, too-many-public-methods

"""
The components dictionary provides a key, value pair of each type of component.
This can be used in a factory to build each component in a system.
"""
component_list = {}


# pylint: disable=too-many-public-methods
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
    klossInlet : float
        K-loss coefficient associated with pressure loss at the inlet of the component
    klossOutlet : float
        K-loss coefficient associated with pressure loss at the outlet of the component
    klossAvg : float
        K-loss coefficient associated with pressure loss across the component
    volume : float
        The flow volume of the component
    inletArea : float
        The inlet area of the component
    outletArea : float
        The outlet area of the component
    theta : float
        Orientation angle of the component in the polar direction
    alpha : float
        Orientation angle of the component in the azimuthal direction
    """

    def __init__(self) -> None:
        self.uc = None
        self._roughness = 0.0
        self._klossInlet = 0.0
        self._klossOutlet = 0.0
        self._klossAvg = 0.0
        self._theta = 0.0
        self._alpha = 0.0

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
    def roughness(self) -> float:
        return self._roughness

    @property
    def klossInlet(self) -> float:
        return self._klossInlet

    @property
    def klossOutlet(self) -> float:
        return self._klossOutlet

    @property
    def klossAvg(self) -> float:
        return self._klossAvg

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
    def theta(self) -> float:
        return self._theta

    @property
    def alpha(self) -> float:
        return self._alpha

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

    def addKlossInlet(self, kloss: float) -> None:
        """Method for adding to kloss inlet. Does not overwrite.

        Parameters
        ----------
        kloss : float
            The value added to klossInlet
        """
        self._klossInlet += kloss

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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

    @staticmethod
    def factory(indict: Dict) -> Dict[str, Component]:
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


class CrossSection(abc.ABC):
    """Abstract base class for cross-sectional areas"""

    @property
    @abc.abstractmethod
    def flow_area(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def wetted_perimeter(self) -> float:
        pass

    @property
    def hydraulic_diameter(self) -> float:
        return 4 * self.flow_area / self.wetted_perimeter


class CircularCrossSection(CrossSection):
    """Circular cross-sectional area

    Parameters
    ---------
    R: float
        Radius of the circular pipe
    """

    def __init__(self, R: float):
        self._R = R

    @property
    def R(self) -> float:
        return self._R

    @property
    def flow_area(self) -> float:
        return np.pi * self._R**2

    @property
    def wetted_perimeter(self) -> float:
        return 2 * np.pi * self._R


class RectangularCrossSection(CrossSection):
    """Rectangular cross-sectional area

    Parameters
    ----------
    W: float
        Width of the rectangular pipe
    H: float
        Height of the rectangular pipe
    """

    def __init__(self, W: float, H: float):
        self._W = W
        self._H = H

    @property
    def W(self) -> float:
        return self._W

    @property
    def H(self) -> float:
        return self._H

    @property
    def flow_area(self) -> float:
        return self._W * self._H

    @property
    def wetted_perimeter(self) -> float:
        return 2 * (self._W + self._H)


class SquareCrossSection(RectangularCrossSection):
    """Square cross-sectional area

    Parameters
    ----------
    W : float
        Width of the square pipe (i.e. flat-to-flat)
    """

    def __init__(self, W: float):
        super().__init__(W, W)


class StadiumCrossSection(CrossSection):
    """Stadium cross sectional area

    Parameters
    ----------
    A : float
        Length of the rectangular portion of the stadium channel
    R : float
        Radius of the semi cirulcar portion of the stadium channel
    """

    def __init__(self, A: float, R: float):
        self._A = A
        self._R = R

    @property
    def A(self) -> float:
        return self._A

    @property
    def R(self) -> float:
        return self._R

    @property
    def flow_area(self) -> float:
        return np.pi * (self._R) ** 2 + 2 * self._R * self._A

    @property
    def wetted_perimeter(self) -> float:
        return 2 * (np.pi * self._R + self._A)


cross_section_classes = {
    "circular": CircularCrossSection,
    "square": SquareCrossSection,
    "rectangular": RectangularCrossSection,
    "stadium": StadiumCrossSection,
}
cross_section_param_lists = {"circular": ["R"], "square": ["W"], "rectangular": ["H", "W"], "stadium": ["A", "R"]}


class Pipe(Component):
    """A pipe component

    Parameters
    ----------
    L : float
        Length of the pipe
    Cross_section_name : string
        Geometry of the cross section of the pipe. Allowed names are circular, square, rectangular, and stadium.
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
    Klossinlet : float
        K-loss coefficient associated with pressure loss at the inlet of the pipe
    Klossoutlet : float
        K-loss coefficient associated with pressure loss at the outlet of the pipe
    Klossavg : float
        K-loss coefficient associated with pressure loss across the pipe
    roughness : float
        Pipe roughness
    pctHeated: float
        Fraction of the pipe that is heated. Used for calculating heated perimeter.
        Defaults to 1 (i.e. the entire pipe is heated)
    """

    def __init__(
        self,
        L: float,
        cross_section_name="circular",
        n: int = 1,
        theta: float = 0.0,
        alpha: float = 0.0,
        Klossinlet: float = 0.0,
        Klossoutlet: float = 0.0,
        Klossavg: float = 0.0,
        roughness: float = 0.0,
        pctHeated: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self._L = L
        self._n = n
        self._costh = np.cos(np.pi / 180 * theta)

        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._klossInlet = Klossinlet
        self._klossOutlet = Klossoutlet
        self._klossAvg = Klossavg
        self._roughness = roughness
        self._kwargs = kwargs
        self._cross_section = cross_section_classes[cross_section_name](
            **{k: v for k, v in kwargs.items() if k in cross_section_param_lists[cross_section_name]}
        )
        self._Ac = self._cross_section.flow_area
        self._Pw = self._cross_section.wetted_perimeter
        self._Dh = self._cross_section.hydraulic_diameter
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
        # Fix to prevent R from being passed twice in case a circular cross section is used
        filtered_kwargs = {k: v for k, v in self._kwargs.items() if k != "R"}
        return genUniformCylinder(self._L, self._Dh / 2, naxial_layers=self._n, **filtered_kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._Dh / 2, self._Dh / 2, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        self._L *= uc.lengthConversion
        self._Ac *= uc.areaConversion
        self._Dh *= uc.lengthConversion
        self._Pw *= uc.lengthConversion
        self._roughness *= uc.lengthConversion
        self._heatedPerimeter *= uc.lengthConversion


component_list["pipe"] = Pipe


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

    def __init__(self, transverse_area: float, transverse_height: float, **kwargs) -> None:
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
    Klossinlet : float
        K-loss coefficient associated with pressure loss at the inlet of the pump
    Klossoutlet : float
        K-loss coefficient associated with pressure loss at the outlet of the pump
    Klossavg : float
        K-loss coefficient associated with pressure loss across the pump
    roughness : float
        Pump roughness
    ptcHeated : float
        Fraction of the pump perimeter that is heated. Uses Dh to determine wetted perimeter
    """

    def __init__(
        self,
        Ac: float,
        Dh: float,
        V: float,
        height: float,
        dP: float,
        Klossinlet: float = 0.0,
        Klossoutlet: float = 0.0,
        Klossavg: float = 0.0,
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
        self._klossInlet = Klossinlet
        self._klossOutlet = Klossoutlet
        self._klossAvg = Klossavg
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
    Klossinlet : float
        K-loss coefficient associated with pressure loss at the inlet of the nozzle
    Klossoutlet : float
        K-loss coefficient associated with pressure loss at the outlet of the nozzle
    Klossavg : float
        K-loss coefficient associated with pressure loss across the nozzle
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
        Klossinlet: float = 0.0,
        Klossoutlet: float = 0.0,
        Klossavg: float = 0.0,
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
        self._klossInlet = Klossinlet
        self._klossOutlet = Klossoutlet
        self._klossAvg = Klossavg
        self._roughness = roughness
        self._kwargs = kwargs
        self._temps = np.zeros(self.nCell)

        wetted_perimeter = np.pi * (self._Rin + self._Rout)  # 2 pi r and 1/2 from average also cancels
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
    Klossinlet : float
        K-loss coefficient associated with pressure loss at the inlet of the annulus
    Klossoutlet : float
        K-loss coefficient associated with pressure loss at the outlet of the annulus
    Klossavg : float
        K-loss coefficient associated with pressure loss across the annulus
    roughness : float
        Annulus roughness
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
        Klossinlet: float = 0.0,
        Klossoutlet: float = 0.0,
        Klossavg: float = 0.0,
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
        self._klossInlet = Klossinlet
        self._klossOutlet = Klossoutlet
        self._klossAvg = Klossavg
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

    @property
    def Rout(self) -> float:
        return self._Rout

    @property
    def Rin(self) -> float:
        return self._Rin

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        return genUniformAnnulus(
            self._L, self._Rin, self._Rout, resolution=self._res, naxial_layers=self._n, **self._kwargs
        ).translate(inlet[0], inlet[1], inlet[2], self._theta, self._alpha)

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
    Klossinlet : float
        K-loss coefficient associated with pressure loss at the inlet of the tank
    Klossoutlet : float
        K-loss coefficient associated with pressure loss at the outlet of the tank
    Klossavg : float
        K-loss coefficient associated with pressure loss across the tank
    roughness : float
        Tank roughness
    ptcHeated : float
        Fraction of the tank perimeter that is heated. Used for calculating heated perimeter.
    """

    def __init__(
        self,
        L: float,
        R: float,
        n: int = 1,
        theta: float = 0.0,
        alpha: float = 0.0,
        Klossinlet: float = 0.0,
        Klossoutlet: float = 0.0,
        Klossavg: float = 0.0,
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
        self._klossInlet = Klossinlet
        self._klossOutlet = Klossoutlet
        self._klossAvg = Klossavg
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
        return genUniformCylinder(
            self._L, self._R, resolution=_CYL_RESOLUTION, naxial_layers=self._n, **self._kwargs
        ).translate(inlet[0], inlet[1], inlet[2], self._theta, self._alpha)

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
        self._heatedPerimeter *= uc.lengthConversion


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
        raise NotImplementedError

    @property
    def volume(self) -> float:
        return sum(component.volume for component in self.baseComponents)

    @property
    def inletArea(self) -> float:
        raise NotImplementedError

    @property
    def outletArea(self) -> float:
        raise NotImplementedError

    @property
    def length(self) -> float:
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def lastComponent(self) -> Component:
        """Return the last component in the collection.
        Implemented in the derived class.
        """
        raise NotImplementedError

    def getNodeGenerator(self) -> Generator[Component, None, None]:
        yield from [component.getNodeGenerator() for component in self._myComponents.values()]

    def getMomentumSource(self) -> float:
        raise NotImplementedError

    def _convertUnits(self, uc: UnitConverter) -> None:
        for component in self._myComponents.values():
            component._convertUnits(uc)

    def addKlossInlet(self, kloss: float) -> None:
        self.firstComponent.addKlossInlet(kloss)


class ParallelComponents(ComponentCollection):
    """A component for a collection of components through which flow passes in parallel

    Parameters
    ----------
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`Component.factory`
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
    lowerManifold : Component
        The lower manifold of the parallel components
    upperManifold : Component
        The upper manifold of the parallel components
    lowerNozzle : Component
        The lower nozzle of the parallel components (connects lowerPlenum to lowerManifold)
    upperNozzle : Component
        The upper nozzle of the parallel components (connects upperPlenum to upperManifold)
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
        # parallel components first
        self._myParallelComponents = Component.factory(parallel_components)
        self._centroids = centroids
        # add parallel components to my components
        myComponents = {**self._myParallelComponents}

        parallel_in_area = 0.0
        parallel_out_area = 0.0
        parallel_theta = myComponents[next(iter(myComponents))].theta
        parallel_alpha = myComponents[next(iter(myComponents))].alpha
        # add an anulus if present
        if annulus is None:
            self._annulus = None
        else:
            assert len(annulus) == 1
            comp_type, parameters = list(annulus.items())[0]
            self._annulus = component_list[comp_type](**parameters)
            myComponents["annulus"] = self._annulus
            parallel_in_area += self._annulus.inletArea
            parallel_out_area += self._annulus.outletArea
            parallel_theta = self._annulus.theta
            parallel_alpha = self._annulus.alpha

        # compute the inlet and outlet area
        for tname, titem in self._myParallelComponents.items():
            parallel_in_area += titem.inletArea
            parallel_out_area += titem.outletArea
            assert parallel_theta == titem.theta
            assert parallel_alpha == titem.alpha
            parallel_theta = titem.theta
            parallel_alpha = titem.alpha

        # create the manifolds
        self._lowerManifold = component_list["pipe"](
            R=np.sqrt(parallel_in_area / np.pi), L=1.0e-64, theta=parallel_theta * 180 / np.pi, alpha=parallel_alpha
        )
        myComponents["lower_manifold"] = self._lowerManifold
        self._upperManifold = component_list["pipe"](
            R=np.sqrt(parallel_out_area / np.pi), L=1.0e-64, theta=parallel_theta * 180 / np.pi, alpha=parallel_alpha
        )
        myComponents["upper_manifold"] = self._upperManifold

        # add the upper and lower plenums
        assert len(lower_plenum) == 1
        comp_type, parameters = list(lower_plenum.items())[0]
        self._lowerPlenum = component_list[comp_type](**parameters)
        myComponents["lower_plenum"] = self._lowerPlenum
        assert parallel_theta == self._lowerPlenum.theta
        assert parallel_alpha == self._lowerPlenum.alpha
        assert len(upper_plenum) == 1
        comp_type, parameters = list(upper_plenum.items())[0]
        self._upperPlenum = component_list[comp_type](**parameters)
        myComponents["upper_plenum"] = self._upperPlenum
        assert parallel_theta == self._upperPlenum.theta
        assert parallel_alpha == self._upperPlenum.alpha

        # create the nozzles connecting manifolds to plenums
        self._lowerNozzle = component_list["nozzle"](
            R_inlet=np.sqrt(self._lowerPlenum.outletArea / np.pi),
            R_outlet=np.sqrt(self._lowerManifold.inletArea / np.pi),
            L=1.0e-64,
            theta=parallel_theta * 180 / np.pi,
            alpha=parallel_alpha,
        )
        myComponents["lower_nozzle"] = self._lowerNozzle
        self._upperNozzle = component_list["nozzle"](
            R_inlet=np.sqrt(self._upperPlenum.inletArea / np.pi),
            R_outlet=np.sqrt(self._upperManifold.outletArea / np.pi),
            L=1.0e-64,
            theta=parallel_theta * 180 / np.pi,
            alpha=parallel_alpha,
        )
        myComponents["upper_nozzle"] = self._upperNozzle

        self._kwargs = kwargs

        super().__init__(myComponents)

        self._theta = self._lowerPlenum.theta
        self._alpha = self._lowerPlenum.alpha

    @property
    def firstComponent(self) -> Component:
        return self._lowerPlenum

    @property
    def lastComponent(self) -> Component:
        return self._upperPlenum

    @property
    def flowArea(self) -> float:
        raise NotImplementedError

    @property
    def inletArea(self) -> float:
        return self._lowerPlenum.inletArea

    @property
    def outletArea(self) -> float:
        return self._upperPlenum.outletArea

    @property
    def length(self) -> float:
        L = self._lowerPlenum.length + self._upperPlenum.length
        L += self._lowerManifold.length + self._upperManifold.length
        L += self._lowerNozzle.length + self._upperNozzle.length
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
        ncell += self._lowerManifold.nCell + self._upperManifold.nCell
        ncell += self._lowerNozzle.nCell + self._upperNozzle.nCell
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
    def lowerManifold(self) -> Component:
        return self._lowerManifold

    @property
    def upperManifold(self) -> Component:
        return self._upperManifold

    @property
    def lowerNozzle(self) -> Component:
        return self._lowerNozzle

    @property
    def upperNozzle(self) -> Component:
        return self._upperNozzle

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
            # TODO this doesn't do anything with alpha...
            comp = self._myParallelComponents[cname]
            i = (
                inlet2[0] + centroid[0] * np.cos(comp.theta),
                inlet2[1] + centroid[1],
                inlet2[2] - centroid[0] * np.sin(comp.theta),
            )
            mesh += self._myParallelComponents[cname].getVTKMesh(i)
        inlet2 = list(self._myParallelComponents.items())[0][1].getOutlet(inlet2)
        mesh += self._upperPlenum.getVTKMesh(inlet2)
        return mesh

    def getBoundingBox(
        self, inlet: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float, float, float, float]:
        raise NotImplementedError

    def _convertUnits(self, uc: UnitConverter) -> None:
        self.uc = uc
        for cname, centroid in self._centroids.items():
            self._centroids[cname] = [cval * uc.lengthConversion for cval in centroid]
        super()._convertUnits(uc)


component_list["parallel_components"] = ParallelComponents


class Core(abc.ABC, ParallelComponents):
    """An abstract base class for reactor core components.

    Core is an abstract base class that extends ParallelComponents to provide the foundation
    for specific core geometry implementations like HexCore and CartCore. It manages a collection
    of parallel channels arranged in a two-dimensional pattern (map) with specified coordinates
    and optional orificing for flow control.

    Parameters
    ----------
    components : Dict
        The collection of components which comprise this core. The structure of
        this dictionary follows the same convention as :func:`Component.factory`
    channel_map : List[List[str]]
        A 2D grid representation of component placement in the core
    lower_plenum : Dict[str, Dict[str,float]]
        The component specifications for the lower plenum
        (key: component type, value: component parameters dictionary)
    upper_plenum : Dict[str, Dict[str,float]]
        The component specifications for the upper plenum
        (key: component type, value: component parameters dictionary)
    annulus : Dict[str, Dict[str,float]], optional
        The component specifications for the annulus surrounding the core
        (key: component type, value: component parameters dictionary)
    orificing : List[List[float]], optional
        K-loss values for each position in the channel_map, used to model flow resistance
        Must have the same dimensions as channel_map if provided
    """

    def __init__(
        self,
        components: Dict,
        channel_map: List[List[str]],
        lower_plenum: Dict[str, Dict[str, float]],
        upper_plenum: Dict[str, Dict[str, float]],
        annulus: Optional[Dict[str, Dict[str, float]]] = None,
        orificing: Optional[List[List[float]]] = None,
        **kwargs,
    ):
        self._map = channel_map
        self._orificing = orificing
        self._core_components = Component.factory(components)

        # Validate orificing dimensions if provided
        if self._orificing is not None:
            assert len(self._map) == len(self._orificing)
            assert np.all(len(map_row) == len(orificing_row) for map_row, orificing_row in zip(self._map, self._orificing))

        centroids = self._calculate_centroids()
        extended_comps = self._create_extended_components(centroids)
        super().__init__(extended_comps, centroids, lower_plenum, upper_plenum, annulus, **kwargs)

    def _calculate_centroids(self) -> Dict[str, List[float]]:
        """Calculate the centroid coordinates for each component in the core map.

        This method processes the channel map to determine the x-y coordinates for each
        component based on its row and column position, using the geometry-specific
        _getChannelCoords method to calculate the actual coordinate values.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary mapping component names to [x, y] centroid coordinates, where
            keys follow the format "value-row-column".
        """
        centroids = {}
        for row, col in enumerate(self._map):
            for column, value in enumerate(col):
                if value is not None:
                    cname = f"{str(value):s}-{row+ 1:d}-{column +1:d}"
                    x_centroid, y_centroid = self._getChannelCoords(row, column)
                    centroids[cname] = [x_centroid, y_centroid]
        return centroids

    def _create_extended_components(self, centroids: Dict[str, List[float]]) -> Dict[str, Component]:
        """Create extended components with proper orificing for each position in the core map.

        This method generates component instances for each position in the core map, applying
        appropriate orificing values if specified. Each component is a deep copy of the base
        component with a unique identifier derived from its position in the map.

        Parameters
        ----------
        centroids : Dict[str, List[float]]
            Dictionary mapping component names to [x, y] centroid coordinates

        Returns
        -------
        Dict[str, Component]
            Dictionary mapping component names to their instantiated components, with keys
            in the format "value-row-column" corresponding to entries in the centroids dictionary.
        """
        extended_comps = {}
        for cname, _ in centroids.items():
            # Parse the component identifier from the name (format: "value-row-column")
            parts = cname.split("-")
            component_id = parts[0]
            row = int(parts[1]) - 1
            column = int(parts[2]) - 1

            # Apply orificing if specified
            if self._orificing is not None and component_id in self._core_components:
                self._core_components[component_id].addKlossInlet(self._orificing[row][column])

            # Create a deep copy of the component
            extended_comps[cname] = deepcopy(self._core_components[component_id])

        return extended_comps

    def _getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """Method that returns the VTK mesh for a Core

        This method generates a visual representation of the core by calculating the
        core inlet position, adjusting for any plate components, and generating the
        complete mesh using the parent class implementation.

        Parameters
        ----------
        inlet : Tuple[float, float, float]
            The core inlet coordinates (x, y, z)

        Returns
        -------
        VTKMesh
            The generated VTK mesh for the entire core
        """
        # Calculate the core inlet position from the lower plenum outlet
        core_inlet = self._lowerPlenum.getOutlet(inlet)

        # Get the first component's subcomponents
        first_component_key = list(self._core_components.keys())[0]
        component_channels = self._core_components[first_component_key]._myComponents

        # Adjust inlet position if a plate component is present
        for channel_name, channel in component_channels.items():
            if channel_name == "plate":
                core_inlet = channel.getOutlet(core_inlet)
                break

        # Create and return the combined mesh
        mesh = VTKMesh()
        mesh += super().getVTKMesh(inlet)
        return mesh

    @abc.abstractmethod
    def _getChannelCoords(self, row: int, column: int) -> Tuple[float, float]:
        """Abstract base method for Core channel coordinates"""

    @abc.abstractmethod
    def _convertUnits(self, uc: UnitConverter) -> None:
        """Abstract base method for converting units"""


class HexCore(Core):
    """A hexagonal geometry reactor core component.

    HexCore implements a reactor core with components arranged in a hexagonal pattern,
    where each component is positioned based on a hexagonal grid. The arrangement
    follows a symmetric pattern with a specified pitch (distance between adjacent components).

    Parameters
    ----------
    pitch : float
        Distance between each of the fuel channels (serial components)
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`Component.factory`
    channel_map : List[List[str]]
        List containing the serial components in the corresponding rings of concentric hexagons
    lower_plenum : Dict[str, Dict[str,float]]
        The component specifications for the lower plenum
        (key: component type, value: component parameters dictionary)
    upper_plenum : Dict[str, Dict[str,float]]
        The component specifications for the upper plenum
        (key: component type, value: component parameters dictionary)
    annulus : Dict[str, Dict[str,float]], optional
        The component specifications for the annulus
        (key: component type, value: component parameters dictionary)
    orificing : List[List[float]], optional
        List containing the kloss values associated with serial components in the corresponding rows and
        columns or concentric rings of the core map - should have the same shape as core map
    non_channels : List[str], optional
        The list of non-channels to fill the core map with. Defaults to ["0"] if not provided.
        This is used to fill the core map with non-channels where needed.
    """

    def __init__(
        self,
        pitch: float,
        components: Dict,
        channel_map: List[List[str]],
        lower_plenum: Dict[str, Dict[str, float]],
        upper_plenum: Dict[str, Dict[str, float]],
        annulus: Optional[Dict[str, Dict[str, float]]] = None,
        orificing: Optional[List[List[float]]] = None,
        non_channels: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        if non_channels is None:
            non_channels = ["0"]
        assert pitch >= 0, f"pitch: {pitch} must be positive"
        self._validate_hex_map(channel_map)
        self._pitch = pitch
        filled_map = self._fill_map(channel_map, non_channels)
        super().__init__(components, filled_map, lower_plenum, upper_plenum, annulus, orificing, **kwargs)

    @staticmethod
    def _fill_map(channel_map: List[List[int]], non_channels: List[str]) -> List[List[Optional[str]]]:
        """Fill a hexagonal core map with components, handling void spaces appropriately.

        This method takes a partially populated hexagonal channel map and fills it
        with components based on the specified channels. It handles the proper alignment
        of components in the hexagonal grid, accounting for the specific geometry
        requirements of hexagonal patterns.

        Parameters
        ----------
        channel_map : List[List[int]]
            The core map to fill with component identifiers
        non_channels : List[str]
            The list of identifiers that should not be treated as components

        Returns
        -------
        List[List[Optional[str]]]
            The filled hexagonal core map with None values for positions without components
        """

        def default_map(num_rows: int) -> List[List[None]]:
            """Create a default hexagonal map with `num_rows` rows of None."""
            n = (num_rows - 1) // 2 + 1
            return [[None for _ in range(n + i)] for i in range(n)] + [
                [None for _ in range(num_rows - 1 - i)] for i in range(n - 1)
            ]

        def is_even(n: int) -> bool:
            """Check if a number is even."""
            if n % 2 == 0:
                return True
            return False

        num_rows = len(channel_map)
        filled_map = default_map(num_rows)
        len_map_rows = [(len(filled_map[row]), len(channel_map[row])) for row in range(num_rows)]
        for row, (len_filled_map_row, len_channel_map_row) in enumerate(len_map_rows):
            offset = 1 if is_even(len_filled_map_row) and not is_even(len_channel_map_row) else 0
            start = len_filled_map_row // 2 - len_channel_map_row // 2 - offset
            stop = start + len_channel_map_row
            for channel_map_elem, filled_map_elem in enumerate(range(start, stop)):
                if channel_map[row][channel_map_elem] not in non_channels:
                    filled_map[row][filled_map_elem] = channel_map[row][channel_map_elem]
        return filled_map

    @staticmethod
    def _validate_hex_map(channel_map: List[List[str]]) -> None:
        """Validate that a channel map conforms to hexagonal geometry requirements.

        This method verifies that the provided channel map is valid for a hexagonal core:
        - It must not be empty
        - It must have an odd number of rows to maintain symmetry
        - Row lengths must follow the hexagonal pattern constraint where each row's
          length must not exceed the expected pattern [n, n+1, …, 2*n-1, 2*n-2, …, n]

        Parameters
        ----------
        channel_map : List[List[str]]
            The hexagonal map configuration to validate

        Raises
        ------
        AssertionError
            If the channel map does not conform to hexagonal geometry requirements
        """
        num_rows = len(channel_map)
        assert num_rows > 0, "channel_map: must not be empty"
        assert num_rows % 2 == 1, f"channel_map: must have an odd number of rows, not {num_rows} rows"
        n = (num_rows + 1) // 2
        # expected lengths: [n, n+1, …, 2*n-1, 2*n-2, …, n]
        ascending = list(range(n, 2 * n))
        descending = list(range(2 * n - 2, n - 1, -1))
        expected = ascending + descending
        actual = [len(row) for row in channel_map]
        for row in range(num_rows):
            assert (
                actual[row] <= expected[row]
            ), f"channel_map: too many elements in row {row+1}: {actual[row]} > {expected[row]}"

    def _getChannelCoords(self, row: int, column: int) -> Tuple[float, float]:
        """Calculate the x-y coordinates for a position in the hexagonal core map.

        This method computes the precise spatial position for a component in the hexagonal
        grid based on its row and column indices. It accounts for hexagonal geometry spacing,
        including appropriate offsets for even/odd row and column counts to maintain
        the correct hexagonal packing arrangement.

        Parameters
        ----------
        row : int
            Row index in the hexagonal map (zero-based)
        column : int
            Column index in the hexagonal map (zero-based)

        Returns
        -------
        Tuple[float, float]
            The precise (x, y) coordinate pair for the specified map location
        """

        # Define horizontal and vertical spacing
        horizontal_spacing = self._pitch
        vertical_spacing = self._pitch * np.sqrt(3) / 2  # Hexagonal geometry height factor

        # Calculate row center position
        row_center_index = len(self._map) / 2
        y_offset = 0.0

        # Apply offset for even number of rows
        if len(self._map) % 2 == 0:
            y_offset = -0.5 * vertical_spacing  # Shift for even row count

        # Calculate y-coordinate (positive upward, so row increases downward)
        y_coordinate = vertical_spacing * (row_center_index - row) + y_offset

        # Calculate column center position
        column_center_index = np.floor(len(self._map[row]) / 2)
        x_offset = 0.0

        # Apply offset for even number of columns in this row
        if len(self._map[row]) % 2 == 0:
            x_offset = 0.5 * horizontal_spacing  # Shift for even column count

        # Calculate x-coordinate
        x_coordinate = horizontal_spacing * (column - column_center_index) + x_offset

        return x_coordinate, y_coordinate

    def _convertUnits(self, uc: UnitConverter) -> None:
        """Convert hexagonal core dimensions to the target unit system.

        This method applies unit conversion to the hexagonal core's pitch value
        and then cascades the conversion to all child components.
        """
        self.uc = uc
        self._pitch *= uc.lengthConversion
        super()._convertUnits(uc)


component_list["hex_core"] = HexCore


class CartCore(Core):
    """A Cartesian geometry reactor core component.

    CartCore implements a reactor core with components arranged in a rectangular grid pattern.
    Components are positioned based on x-pitch and y-pitch values, which can be different
    to create a non-square grid. The arrangement can be aligned to the left, right, or center.

    Parameters
    ----------
    x_pitch : float
        Distance between each of the fuel channels in the horizontal direction
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`Component.factory`
    channel_map : List[List[str]]
        List containing the serial components in the corresponding rows and
        columns of the map
    lower_plenum : Dict[str, Dict[str,float]]
        The component specifications for the lower plenum
        (key: component type, value: component parameters dictionary)
    upper_plenum : Dict[str, Dict[str,float]]
        The component specifications for the upper plenum
        (key: component type, value: component parameters dictionary)
    annulus : Dict[str, Dict[str,float]], optional
        The component specifications for the annulus
        (key: component type, value: component parameters dictionary)
    orificing : List[List[float]], optional
        List containing the kloss values associated with serial components in the corresponding rows and
        columns of the map - should have the same shape as channel_map
    non_channels : List[str], optional
        The list of non-channels to fill the core map with. Defaults to ["0"] if not provided.
        This is used to fill the core map with non-channels where needed.
    y_pitch : float, optional
        Distance between each of the fuel channels in the vertical direction.
        Defaults to the same value as x_pitch if not provided.
    map_alignment : {"left", "right", "center"}, optional
        The horizontal alignment strategy for the map. Defaults to "center".
    """

    Alignment: TypeAlias = Literal["right", "left", "center"]

    def __init__(
        self,
        x_pitch: float,
        components: Dict,
        channel_map: List[List[str]],
        lower_plenum: Dict[str, Dict[str, float]],
        upper_plenum: Dict[str, Dict[str, float]],
        annulus: Optional[Dict[str, Dict[str, float]]] = None,
        orificing: Optional[List[List[float]]] = None,
        non_channels: Optional[List[str]] = None,
        y_pitch: Optional[float] = None,
        map_alignment: Optional[Alignment] = "center",
        **kwargs,
    ) -> None:

        assert x_pitch >= 0, f"pitch: {x_pitch} must be positive"
        if y_pitch is None:
            y_pitch = x_pitch
        assert y_pitch >= 0, f"pitch: {y_pitch} must be positive"
        if non_channels is None:
            non_channels = ["0"]
        assert len(channel_map) > 0, f"map: {channel_map} must not be empty"
        filled_map = self._fill_map(channel_map, non_channels, map_alignment)
        num_rows = len(filled_map)
        num_cols = len(filled_map[0])
        self._center_column = (num_cols - 1) / 2
        self._center_row = (num_rows - 1) / 2
        self._x_pitch = x_pitch
        self._y_pitch = y_pitch
        super().__init__(components, filled_map, lower_plenum, upper_plenum, annulus, orificing, **kwargs)

    @staticmethod
    def _fill_map(
        channel_map: List[List[str]], non_channels: List[str], map_alignment: Alignment
    ) -> List[List[Optional[str]]]:
        """Fill a Cartesian core map with components based on specified alignment.

        This method takes a rectangular channel map and fills it with components,
        handling alignment options (left, right, or center). The alignment determines
        how components are positioned when rows have varying lengths.

        Parameters
        ----------
        channel_map : List[List[int]]
            The rectangular core map to fill with component identifiers
        non_channels : List[str]
            The list of identifiers that should not be treated as components
        map_alignment : Alignment
            The horizontal alignment strategy for the map ("left", "right", or "center")

        Returns
        -------
        List[List[Optional[str]]]
            The filled Cartesian core map with None values for positions without components

        Raises
        ------
        ValueError
            If the map alignment is not one of the supported options
        """

        def default_map(channel_map: List[List[str]]) -> List[List[None]]:
            num_rows = len(channel_map)
            max_cols = max(len(row) for row in channel_map)
            return [[None for _ in range(max_cols)] for _ in range(num_rows)]

        def center_fill(channel_map: List[List[int]], non_channels: List[str]) -> None:
            filled_map = default_map(channel_map)
            for row_index, row in enumerate(channel_map):
                row_length = len(row)
                offset = (len(filled_map[row_index]) - row_length) // 2
                for col_index, value in enumerate(row):
                    if value not in non_channels:
                        filled_map[row_index][col_index + offset] = value
            return filled_map

        def left_fill(channel_map: List[List[int]], non_channels: List[str]) -> None:
            filled_map = default_map(channel_map)
            for row_index, row in enumerate(channel_map):
                for col_index, value in enumerate(row):
                    if value not in non_channels:
                        filled_map[row_index][col_index] = value
            return filled_map

        def right_fill(channel_map: List[List[int]], non_channels: List[str]) -> None:
            filled_map = default_map(channel_map)
            for row_index, row in enumerate(channel_map):
                for col_index, value in enumerate(reversed(row)):
                    if value not in non_channels:
                        filled_map[row_index][-col_index - 1] = value
            return filled_map

        if map_alignment == "center":
            filled_map = center_fill(channel_map, non_channels)
        elif map_alignment == "right":
            filled_map = right_fill(channel_map, non_channels)
        elif map_alignment == "left":
            filled_map = left_fill(channel_map, non_channels)
        else:
            raise ValueError(f"Invalid map alignment: {map_alignment}. Must be 'left', 'right', or 'center'.")

        return filled_map

    def _getChannelCoords(self, row: int, column: int) -> Tuple[float, float]:
        """Calculate the x-y coordinates for a position in the Cartesian core map.

        This method computes the precise spatial position for a component in the
        rectangular grid based on its row and column indices. Positions are calculated
        relative to the center of the grid, with x-pitch and y-pitch determining
        the spacing between components.

        Parameters
        ----------
        row : int
            Row index in the Cartesian map (zero-based)
        column : int
            Column index in the Cartesian map (zero-based)

        Returns
        -------
        Tuple[float, float]
            The precise (x, y) coordinate pair for the specified map location
        """

        x_centroid = (column - self._center_column) * self._x_pitch
        y_centroid = -(row - self._center_row) * self._y_pitch
        return x_centroid, y_centroid

    def _convertUnits(self, uc: UnitConverter) -> None:
        """Convert Cartesian core dimensions to the target unit system.

        This method applies unit conversion to both the x-pitch and y-pitch values
        of the Cartesian core, then cascades the conversion to all child components.
        """
        self._x_pitch *= uc.lengthConversion
        self._y_pitch *= uc.lengthConversion
        super()._convertUnits(uc)


component_list["cart_core"] = CartCore


class SerialComponents(ComponentCollection):
    """A component for a collection of components through which flow passes in serial (i.e. from one component into the next)

    Parameters
    ----------
    components : Dict
        The collection parallel components which comprise this component.  The structure of
        this dictionary follows the same convention as :func:`Component.factory`
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
        cont_components = Component.factory(components)
        cont_components, order = cont_factory(cont_components, order)
        super().__init__(cont_components)
        self._order = order
        self._kwargs = kwargs
        if cont_components[order[0]]._theta != cont_components[order[-1]]._theta:
            raise Exception("Serial component theta for first and last components must match!")
        if cont_components[order[0]]._alpha != cont_components[order[-1]]._alpha:
            raise Exception("Serial component alpha for first and last components must match!")

        self._theta = cont_components[order[0]]._theta
        self._alpha = cont_components[order[0]]._alpha

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
                return round(self._myComponents[c].hydraulicDiameter, 6)
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


def cont_factory(cont_components, order):
    """Private method makes serial components continuous with respect to area change

    This method takes in a list of components and their order and inserts infitesimal nozzles between them
    that make the area change transitions continuous

    Parameters
    ----------
    cont_components : list
        list of components
    order : list
        order of those components

    Returns
    -------
    list, list
        The new component list and order with the inserted nozzles
    """
    num_connects = 0
    discont_found = True
    while discont_found:
        discont_found = False
        # initialize the previous area as the first area
        prev_area = cont_components[order[0]].inletArea
        for i, entry in enumerate(order):
            if abs(prev_area - cont_components[entry].inletArea) > 1.0e-12 * min(prev_area, cont_components[entry].inletArea):
                tempnozzle = component_list["nozzle"](
                    L=1.0e-64,
                    R_inlet=np.sqrt(prev_area / np.pi),
                    R_outlet=np.sqrt(cont_components[entry].inletArea / np.pi),
                    theta=cont_components[entry]._theta * 180 / np.pi,
                    alpha=cont_components[entry]._alpha,
                    Klossinlet=0,
                    Klossoutlet=0,
                    Klossavg=0,
                    roughness=0,
                )
                cont_components[f"temp_nozzle_for_make_continuous_creation_in_serialcomp_{entry}_{num_connects}"] = deepcopy(
                    tempnozzle
                )
                order = (
                    order[0:i]
                    + [f"temp_nozzle_for_make_continuous_creation_in_serialcomp_{entry}_{num_connects}"]
                    + order[i : len(order)]
                )
                num_connects += 1
                discont_found = True
                break
            prev_area = cont_components[entry].outletArea
    return cont_components, order


component_list["serial_components"] = SerialComponents
