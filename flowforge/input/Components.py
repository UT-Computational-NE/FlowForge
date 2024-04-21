import abc
from copy import deepcopy
from typing import List, Dict, Tuple
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
    """
    Base class for all components of the system.
    """

    def __init__(self) -> None:
        """
        Initializes the component.
        """
        self.uc = None

    @property
    @abc.abstractmethod
    def flowArea(self) -> float:
        """
        Abstract method which does any post-processing of the code output
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """
        Abstract method which does any post-processing of the code output
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hydraulicDiameter(self) -> float:
        """
        Abstract method which does any post-processing of the code output
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def heightChange(self) -> float:
        """
        Abstract method which does any post-processing of the code output
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nCell(self) -> int:
        """
        Abstract method which does any post-processing of the code output
        """
        raise NotImplementedError

    @property
    def volume(self) -> float:
        """
        Abstract method which does any post-processing of the code output
        """
        return self.flowArea * self.length

    @property
    def inletArea(self) -> float:
        """
        Default method to get inlet Area
        """
        return self.flowArea

    @property
    def outletArea(self) -> float:
        """
        Default method to get outlet Area
        """
        return self.flowArea

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:  # pylint:disable=unused-argument
        """
        Abstract method which does any post-processing of the code output
        """
        return NotImplementedError

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:  # pylint:disable=unused-argument
        """
        Abstract method which does any post-processing of the code output
        """
        return genUniformCube(self.length, self.length, self.heightChange)

    def getNodeGenerator(self):
        """
        Gets the fluid node generator.
        """
        for i in range(self.nCell):
            yield self

    def getBoundingBox(self, inlet: Tuple[float, float, float]) -> List[float]:
        """
        Abstract method which does any post-processing of the code output
        """
        outlet = self.getOutlet(inlet)
        return [(inlet[0] + outlet[0]) / 2, (inlet[1] + outlet[1]) / 2, (inlet[2] + outlet[1]) / 2]

    def rotate(self, d_x: float, d_y: float, d_z: float, theta: float = 0.0, alpha: float = 0.0) -> np.ndarray:
        """
        Method which rotates x, y, & z coordinates according to theta and alpha angles.
        As of now, it is only used for the node bounding box

        Args:
            d_x     : float, x coordinate
            d_y     : float, y coordinate
            d_z     : float, z coordinate
            theta   : (OPTIONAL) float, the degree of rotation desired about the y axis (polar)
            alpha   : (OPTIONAL) float, the degree of rotation desired about the z axis (azimuthal)
        """
        polar_rotate = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        azimuthal_rotate = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        new_vec = np.dot(azimuthal_rotate, np.dot(polar_rotate, np.array([d_x, d_y, d_z])))
        return new_vec


def component_factory(indict: Dict[str, Dict[str, float]]) -> List[Component]:
    """
    Factory to initialize all components in the system.

    Args:
        - indict : dict, input dictionary containing the components
    """
    components = {}
    for compname, comp in indict.items():
        if isinstance(comp, dict):
            if compname in component_list:
                for name, input_ in comp.items():
                    components[name] = component_list[compname](**input_)
            else:
                raise TypeError("Unknown component type: " + compname)
        elif isinstance(comp, Component):
            components[compname] = comp
        else:
            raise TypeError(f"Unknown input dictionary: {compname:s} type: {str(type(comp)):s}")

    return components


class Pipe(Component):
    """
    Pipe component.
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
        **kwargs,
    ) -> None:
        """
        The __init__ function initializes the pipe subclass of component by storing the
        values of the pipe.

        Args:
            L       : float, length of the pipe
            R       : (OPTIONAL) float, radius of the pipe
            Ac      : (OPTIONAL) float, flow area of the pipe
            Dh      : (OPTIONAL) float, hydraulic diameter of the pipe
            n       : (OPTIONAL) int, number of segments pipe is divided into
            theta   : (OPTIONAL) float, orientation angle of the pipe in the polar direction
            alpha   : (OPTIONAL) float, orientation angle of the pipe in the aziumathal direction
        """
        super().__init__()
        self._L = L
        self._n = n
        self._costh = np.cos(np.pi / 180 * theta)
        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._kloss = Kloss
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
        self._temps = np.zeros(self.nCell)

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the pipe.
        Args: None
        """
        return self._Ac

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the length of the pipe.
        Args: None
        """
        return self._L / self._n

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the pipe.
        Args: None
        """
        return self._Dh

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the value of the height change each segment of the pipe.
        Args: None
        """
        return self._costh * self._L / self._n

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the stored value of the number of segments the pipe is made up of.
        Args: None
        """
        return self._n

    def getMomentumSource(self) -> float:
        """
        Gets the momentum source.
        """
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function calculates where the outlet of the pipe will be
        and returns the point as a tuple. This function can be used by automating
        returning the outlet of one component and using this point as the inlet
        for the next component.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function calls the genCyl function from VTKShapes to create
        a mesh for a cylinder which will represent the pipe. This function will
        also automatically translate the shape to the inlet coordinates that are
        passed as an argument and will rotate the pipe based on the stored theta
        and alpha angles.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        return genCyl(self._L, self._R, nlayers=self._n, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(self, inlet: Tuple[float, float, float]) -> List[float]:
        """
        Gets a Bounding box for any pipe
        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._R, self._R, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self._L *= uc.lengthConversion
        self._R *= uc.lengthConversion
        self._Ac *= uc.areaConversion
        self._Dh *= uc.lengthConversion
        self._Pw *= uc.lengthConversion


component_list["pipe"] = Pipe


class SquarePipe(Pipe):
    """
    Square pipe component.
    """

    def __init__(self, L: float, W: float, **kwargs) -> None:
        """
        Initializes the square pipe instance.

        Args:
            - L (float)  : length of the pipe
            - W (float)  : width of the pipe (flat-to-flat)
        """
        super().__init__(L=L, Dh=W, Ac=W**2, **kwargs)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function calls the genCyl function from VTKShapes to create
        a mesh for a square pipe. This function will also automatically translate
        the shape to the inlet coordinates that are passed as an argument and will
        rotate the pipe based on the stored theta and alpha angles.

        Args:
            - inlet (tuple): contains the x, y, and z coordinates of the inlet point
        """
        return genUniformCube(self._Dh, self._Dh, self._L, nz=self._n).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )


component_list["square_pipe"] = SquarePipe


class Tee(Pipe):
    """
    Tee pipe component. To be implemented.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the tee pipe instance.
        """
        super().__init__(**kwargs)
        assert self._n == 1


component_list["tee"] = Tee


class Pump(Component):
    """
    Pump component.
    """

    def __init__(self, Ac: float, Dh: float, V: float, height: float, dP: float, **kwargs) -> None:
        """
        The __init__ function initializes the pump subclass of component by storing the
        values of the pump.

        Args:
            Ac      : float, flow area
            Dh      : float, hydraulic diameter
            V       : float, volume
            height  : float, height of the pump, change in the height of the fluid
            dP      : float, delta P, change in pressure that the pump creates
        """
        super().__init__()
        self._Ac = Ac
        self._Dh = Dh
        self._V = V
        self._h = height
        self._dP = dP
        self._kwargs = kwargs
        self._temps = np.zeros(self.nCell)

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the pump.
        Args: None
        """
        return self._Ac

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the Length of the pump.
        Args: None
        """
        return self._V / self._Ac

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the pump.
        Args: None
        """
        return self._Dh

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the stored value of the height change of the pump.
        Args: None
        """
        return self._h

    @property
    def volume(self) -> float:
        """
        The Volume property returns the stored value of the volume of the pump.
        Args: None
        """
        return self._V

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the number of cells the pump consists of, which is 1.
        Args: None
        """
        return 1

    def getMomentumSource(self) -> float:
        """
        Gets the momentum source.
        """
        return self._dP * self._Ac

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function calculates where the outlet of the pump will be
        and returns the point as a tuple. This function can be used by automating
        returning the outlet of one component and using this point as the inlet
        for the next component.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        # for now making the assumption that it comes in bottom and out the side of the pump
        x = inlet[0] + self._Dh / 2
        y = inlet[1]
        z = inlet[2] + self.heightChange / 2
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function calls the genCube function from VTKShapes to create
        a mesh for a cube which will represent the pump. This function will
        also automatically translate the shape to the inlet coordinates that are
        passed as an argument.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        return genUniformCube(self._Dh, self._Dh, self._h, **self._kwargs).translate(inlet[0], inlet[1], inlet[2])

    def getBoundingBox(self, inlet: Tuple[float, float, float]) -> List[float]:
        """
        Gets a Bounding box for any pump assuming orientation inline with cartesian grid
        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._Dh / 2, self._Dh / 2, self._h, 0.0, 0.0]

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self._Ac *= uc.areaConversion
        self._Dh *= uc.lengthConversion
        self._V *= uc.volumeConversion
        self._h *= uc.lengthConversion
        self._dP *= uc.pressureConversion


component_list["pump"] = Pump


class Nozzle(Component):
    """
    Nozzle component.
    """

    def __init__(
        self,
        L: float,
        R_inlet: float,
        R_outlet: float,
        theta: float = 0.0,
        alpha: float = 0.0,
        resolution: int = _CYL_RESOLUTION,
        **kwargs,
    ) -> None:
        """
        The __init__ function initializes the nozzle subclass of component by storing the
        values from the inputs.

        Args:
            L        : float, Length of the nozzle
            R_inlet  : float, Radius of the nozzle inlet
            R_outlet : float, Radius of the nozzle outlet
            theta    : (OPTIONAL) float, orientation angle in the polar direction
            alpha    : (OPTIONAL) float, orientation angle in the azimuthal direction
        """
        super().__init__()
        self._L = L
        self._Rin = R_inlet
        self._Rout = R_outlet
        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._Dh = self._Rin + self._Rout  # average radius needs div 2 but radius to diameter needs mult 2 so they cancel
        self._Ac = 0.25 * np.pi * self._Dh * self._Dh
        self._res = resolution
        self._kwargs = kwargs
        self._temps = np.ones(self.nCell)

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the nozzle.
        Args: None
        """
        return self._Ac

    @property
    def inletArea(self) -> float:
        """
        The InletArea property returns the inlet flow area of the nozzle.
        Args: None
        """
        return np.pi * self._Rin * self._Rin

    @property
    def outletArea(self) -> float:
        """
        The OutletArea property returns the outlet flow area of the nozzle.
        Args: None
        """
        return np.pi * self._Rout * self._Rout

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the length of each segment of the nozzle.
        Args: None
        """
        return self._L

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the nozzle.
        Args: None
        """
        return self._Dh

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the value of the height change each segment of the nozzle.
        Args: None
        """
        return self._L * np.cos(self._theta)

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the stored value of the number of segments the nozzle is made up of.
        Args: None
        """
        return 1

    def getMomentumSource(self) -> float:
        """
        Gets the momentum source.
        """
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function calculates where the outlet of the nozzle will be
        and returns the point as a tuple. This function can be used by automating
        returning the outlet of one component and using this point as the inlet
        for the next component.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function calls the genNozzle function from VTKShapes to create
        a mesh for a nozzle. This function will also automatically translate the shape
        to the inlet coordinates that are passed as an argument and will rotate it based
        on the stored theta and alpha angles.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        return genNozzle(self._L, self._Rin, self._Rout, resolution=self._res, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(self, inlet: Tuple[float, float, float]) -> List[float]:
        """
        Gets a Bounding box for any nozzle assuming inlet radius> outlet radius
        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        outlet = self.getOutlet(inlet)
        if self._Rin >= self._Rout:
            big_r = self._Rin
        else:
            big_r = self._Rout
        return [inlet, outlet, big_r, big_r, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self._L *= uc.lengthConversion
        self._Rin *= uc.lengthConversion
        self._Rout *= uc.lengthConversion
        self._Dh *= uc.lengthConversion
        self._Ac *= uc.areaConversion


component_list["nozzle"] = Nozzle


class Annulus(Component):
    """
    Annulus component.
    """

    def __init__(
        self,
        L: float,
        R_inner: float,
        R_outer: float,
        n: int = 1,
        theta: float = 0.0,
        alpha: float = 0.0,
        resolution: int = _CYL_RESOLUTION,
        **kwargs,
    ) -> None:
        """
        The __init__ function initializes the annulus subclass of component by storing the
        values from the inputs.

        Args:
            L        : float, Length of the annulus
            R_inner  : float, Inner radius of the annulus
            R_outer  : float, Outer radius of the annulus
            n        : (OPTIONAL) int, number of layers in annulus
            theta    : (OPTIONAL) float, orientation angle in the polar direction
            alpha    : (OPTIONAL) float, orientation angle in the azimuthal direction
        """
        super().__init__()
        self._L = L
        self._Rin = R_inner
        self._Rout = R_outer
        self._n = n
        self._theta = theta * np.pi / 180
        self._alpha = alpha * np.pi / 180
        self._res = resolution
        self._kwargs = kwargs
        self._temps = np.ones(self.nCell)

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the annulus.
        Args: None
        """
        return np.pi * (self._Rout * self._Rout - self._Rin * self._Rin)

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the length of the annulus.
        Args: None
        """
        return self._L / self._n

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the annulus.
        Args: None
        """
        return 2 * self.flowArea / (np.pi * (self._Rin + self._Rout))

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the value of the height change each segment of the annulus.
        Args: None
        """
        return self._L * np.cos(self._theta) / self._n

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the stored value of the number of segments the annulus is made up of.
        Args: None
        """
        return self._n

    def getMomentumSource(self) -> float:
        """
        Gets the momentum source.
        """
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function calculates where the outlet of the annulus will be
        and returns the point as a tuple. This function can be used by automating
        returning the outlet of one component and using this point as the inlet
        for the next component.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha)
        z = inlet[2] + self._L * np.cos(self._theta)
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function calls the genAnnulus function from VTKShapes to create
        a mesh for a annulus. This function will also automatically translate the shape
        to the inlet coordinates that are passed as an argument and will rotate it based
        on the stored theta and alpha angles.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        return genAnnulus(self._L, self._Rin, self._Rout, resolution=self._res, nlayers=self._n, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(self, inlet: Tuple[float, float, float]) -> List[float]:
        """
        Gets a Bounding box for annulus
        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._Rout, self._Rout, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self._L *= uc.lengthConversion
        self._Rin *= uc.lengthConversion
        self._Rout *= uc.lengthConversion


component_list["annulus"] = Annulus


class Tank(Component):
    """
    Tank component.
    """

    def __init__(self, L: float, R: float, n: int = 1, theta: float = 0.0, alpha: float = 0.0, **kwargs) -> None:
        """
        The __init__ function initializes the tank subclass of component by storing the
        values of the tank.

        Args:
            L       : float, length of the tank
            R       : float, radius of the tank
            n       : (OPTIONAL) int, number of segments tank is divided into
            theta   : (OPTIONAL) float, orientation angle of the tank in the polar direction
            alpha   : (OPTIONAL) float, orientation angle of the tank in the aziumathal direction
        """
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
        self._kwargs = kwargs
        self._temps = np.zeros(self.nCell)

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the tank.
        Args: None
        """
        return self._Ac

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the length of the tank.
        Args: None
        """
        return self._L

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the tank.
        Args: None
        """
        return self._Dh

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the value of the height change of the tank.
        Args: None
        """
        return self._costh * self._L

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the stored value of the number of segments the tank is made up of.
        Args: None
        """
        return self._n

    def getMomentumSource(self) -> float:
        """
        Gets the momentum source.
        """
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function calculates where the outlet of the tank will be
        and returns the point as a tuple. This function can be used by automating
        returning the outlet of one component and using this point as the inlet
        for the next component.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        x = inlet[0] + self._L * np.sin(self._theta) * np.cos(self._alpha) + self._R * np.cos(self._alpha)
        y = inlet[1] + self._L * np.sin(self._theta) * np.sin(self._alpha) + self._R * np.sin(self._alpha)
        z = inlet[2]
        return (x, y, z)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function calls the genCyl function from VTKShapes to create
        a mesh for a cylinder which will represent the tank. This function will
        also automatically translate the shape to the inlet coordinates that are
        passed as an argument and will rotate the tank based on the stored theta
        and alpha angles.

        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        return genCyl(self._L, self._R, resolution=_CYL_RESOLUTION, nlayers=self._n, **self._kwargs).translate(
            inlet[0], inlet[1], inlet[2], self._theta, self._alpha
        )

    def getBoundingBox(self, inlet: Tuple[float, float, float]) -> List[float]:
        """
        Gets a Bounding box for tank
        Args:
            inlet : tuple of floats, contains the x, y, and z coordinates of the
                    inlet point
        """
        outlet = self.getOutlet(inlet)
        return [inlet, outlet, self._R, self._R, self._L, self._theta, self._alpha]

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self._Ac *= uc.areaConversion
        self._Pw *= uc.lengthConversion
        self._Dh *= uc.lengthConversion
        self._L *= uc.lengthConversion
        self._R *= uc.lengthConversion


component_list["tank"] = Tank


class ParallelComponents(Component):
    """
    ParallelComponents handles the case with components in parallel.
    """

    def __init__(
        self,
        components: Dict[str, Dict[str, float]],
        centroids: Dict[str, float],
        lower_plenum: Dict[str, float],
        upper_plenum: Dict[str, float],
        annulus: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        """
        The __init__ function of the parallel_components class initializes the
        class instance by storing the components dictionary that is used to run the component_factory
        function. This recursive initialization will then automatically continue to initialize the
        components nested inside of the parallel components The __init__ also stores the dict of the
        centroid coordinates of the nested components.

        Args:
            components : dict, dictionary containing the any components to be rendered in parallel
            centroid   : dict, dictionary containing the x and y coordinates of the centroid of the
                         parallel components.
            **kwargs   : dict, dictionary containing any additional keyword arguments passed in
        """
        super().__init__()
        self._myComponents = component_factory(components)
        self._centroids = centroids

        assert len(lower_plenum) == 1
        compname, input_ = list(lower_plenum.items())[0]
        self._lowerPlenum = component_list[compname](**input_)

        assert len(upper_plenum) == 1
        compname, input_ = list(upper_plenum.items())[0]
        self._upperPlenum = component_list[compname](**input_)

        if annulus is None:
            self._annulus = None
        else:
            assert len(annulus) == 1
            compname, input_ = list(annulus.items())[0]
            self._annulus = component_list[compname](**input_)

        self._kwargs = kwargs

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the component.
        """
        return NotImplementedError

    @property
    def inletArea(self) -> float:
        """
        The InletArea property returns the inlet flow area of the parallel component.
        """
        return self._lowerPlenum.inletArea

    @property
    def outletArea(self) -> float:
        """
        The OutletArea property returns the outlet flow area of the parallel component.
        """
        return self._upperPlenum.outletArea

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the length of the component.
        """
        L = self._lowerPlenum.length + self._upperPlenum.length
        L += self._myComponents[list(self._myComponents.keys())[0]].length
        return L

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the component.
        """
        raise NotImplementedError

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the value of the height change of the component.
        """
        raise NotImplementedError

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the number of cells in this component.
        """
        ncell = self._lowerPlenum.nCell + self._upperPlenum.nCell
        if self._annulus is not None:
            ncell += self._annulus.nCell
        for cname in self._centroids.keys():
            ncell += self._myComponents[cname].nCell
        return ncell
    
    @property
    def myComponents(self) -> List[Component]:
        return self._myComponents


    @property
    def centroids(self) -> Dict[str, float]:
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
        """
        Gets the momentum source.
        """
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function will get the outlets of each of the components inside of the parallel_components
        and confirm that they are the same before proceeding.

        Args:
            inlet : tuple, contains the x, y, and z coordinates of the inlet, respectively
        """
        lower = self._lowerPlenum.getOutlet(inlet)
        firstcomp = list(self._myComponents.items())[0][0]
        outlet = self._myComponents[firstcomp].getOutlet(lower)
        outlet = (round(outlet[0], 5), round(outlet[1], 5), round(outlet[2], 5))
        for comp in self._myComponents.values():
            compOutlet = comp.getOutlet(lower)
            compOutletRounded = round(compOutlet[0], 5), round(compOutlet[1], 5), round(compOutlet[2], 5)
            assert compOutletRounded == outlet
        outlet = self._upperPlenum.getOutlet(outlet)
        return outlet

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function loops through each of the components contained inside parallel_components
        and calls the getVTKMesh for their specific component type. Each of these is added to the mesh
        and returned. When calling the nested getVTKMesh functions, the centroid is used to translate the
        x and y coordinates of the inlet to the corresponding location.

        Args:
            inlet : tuple, contains the x, y, and z coordinates of the inlet, respectively
        """
        mesh = VTKMesh()
        mesh += self._lowerPlenum.getVTKMesh(inlet)
        inlet2 = self._lowerPlenum.getOutlet(inlet)
        if self._annulus is not None:
            mesh += self._annulus.getVTKMesh(inlet2)
        for cname, centroid in self._centroids.items():
            i = (inlet2[0] + centroid[0], inlet2[1] + centroid[1], inlet2[2])
            mesh += self._myComponents[cname].getVTKMesh(i)
        inlet2 = list(self._myComponents.items())[0][1].getOutlet(inlet2)
        mesh += self._upperPlenum.getVTKMesh(inlet2)
        return mesh

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self._upperPlenum._convertUnits(uc)
        self._lowerPlenum._convertUnits(uc)
        if self._annulus is not None:
            self._annulus._convertUnits(uc)
        for c, comp in self._myComponents.items():
            comp._convertUnits(uc)
            self._centroids[c][0] *= uc.lengthConversion
            self._centroids[c][1] *= uc.lengthConversion


component_list["parallel_components"] = ParallelComponents


class HexCore(ParallelComponents):
    """
    Hexagonal core component.
    """

    def __init__(self, pitch: float, components: Dict[str, Dict[str, float]], hexmap: List[List[int]], **kwargs) -> None:
        """
        The __init__ function of the hex_core class initializes the class instance by storing the
        pitch between the channels, the components dictionary, which is again recursively passed into
        the component_factory to initialize any further nested components. The 2D list for the hexmap
        is also stored. This map allows for the components to be rendered in the desired hex formation.

        Args:
            pitch      : float, distance between each of the fuel channels (serial components)
            components : dict, dictionary containing the components to be rendered in parallel
            hexmap     : 2D list, list containing the serial components in the corresponding rows and
                         columns of the hex map
            **kwargs   : dict, dictionary containing any additional keyword arguments passed in
                            - ex. level, nrings, trimRadius for TriGrid initialization
        """
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

        super().__init__(extended_comps, centroids, **kwargs)

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function gets the mesh of the parallel portion of hex_core,
        then adds the core mesh

        Args:
            inlet : tuple, contains the x, y, and z coordinates of the inlet, respectively
        """
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
        """
        The _getChannelCoords function is a private function which returns the calculated x and y coordinates
        of the particular location in the map, which is determined by r (row) and c (col). This function
        is called repeatedly from the getVTKMesh function and is used to determine the inlet input for
        the mesh generation and translation.

        Args:
            r : int, row in the hexmap
            c : int, col in the hexmap
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
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        self.uc = uc
        self._pitch *= uc.lengthConversion
        super()._convertUnits(uc)


component_list["hex_core"] = HexCore


class SerialComponents(Component):
    """
    SerialComponents handles the case with components that are connected in series.
    """

    def __init__(self, components: Dict[str, Dict[str, float]], order: List[str], **kwargs) -> None:
        """
        The __init__ function of the serial_components class initializes the class instance by storing the
        dictionary of components that is produced from sending the components to the component_factory
        function. This will recursively initialize the components contained inside the serial_components.
        The order is the physical order of how to render the components nested under the specific serial
        component. This is stored as well.


        Args:
            components : dict, dictionary containing the components to be rendered in series
            order      : list, the order of the series of components that to be rendered
            **kwargs   : dict, dictionary containing any additional keyword arguments passed in
        """
        super().__init__()
        self._myComponents = component_factory(components)
        self._order = order
        self._kwargs = kwargs

    @property
    def flowArea(self) -> float:
        """
        The FlowArea property returns the stored value of the flow area of the component.
        Args: None
        """
        return self._myComponents[self._order[0]].flowArea

    @property
    def inletArea(self) -> float:
        """
        The InletArea property returns the inlet flow area of the parallel component.
        Args: None
        """
        return self._myComponents[self._order[0]].inletArea

    @property
    def outletArea(self) -> float:
        """
        The OutletArea property returns the outlet flow area of the parallel component.
        Args: None
        """
        return self._myComponents[self._order[-1]].outletArea

    @property
    def length(self) -> float:
        """
        The Length property returns the stored value of the length of the component.
        Args: None
        """
        L = 0
        for c in self._myComponents.values():
            L += c.length
        return round(L, 5)

    @property
    def hydraulicDiameter(self) -> float:
        """
        The HydraulicDiameter property returns the stored value of the hydraulic diameter of the component.
        Args: None
        """
        names = list(self._myComponents.keys())
        for c in names:
            if c[0] == "c":
                return round(self._myComponents[c]._R * 2, 6)
        raise Exception("Component with hydraulic diameter not found.")

    @property
    def heightChange(self) -> float:
        """
        The HeightChange property returns the value of the height change of the component.
        Args: None
        """
        raise NotImplementedError

    @property
    def nCell(self) -> int:
        """
        The nCell property returns the number of cells in this component.
        Args: None
        """
        ncell = 0
        for cname in self._order:
            ncell += self._myComponents[cname].nCell
        return ncell

    @property
    def myComponents(self) -> List[Component]:
        return self._myComponents

    @property
    def order(self) -> List[str]:
        return self._order

    def getMomentumSource(self) -> float:
        """
        Gets the momentum source.
        """
        raise NotImplementedError

    def getOutlet(self, inlet: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        The getOutlet function will get the outlet of the serial component. Because this component is in
        series, we looped through the components and got the outlet of each and used this outlet as the
        inlet for the next component. This allowed us to get the outlet of the whole series.

        Args:
            inlet : tuple, contains the x, y, and z coordinates of the inlet, respectively
        """
        for cname in self._order:
            inlet = self._myComponents[cname].getOutlet(inlet)
        return inlet

    def getVTKMesh(self, inlet: Tuple[float, float, float]) -> VTKMesh:
        """
        The getVTKMesh function loops through each of the components contained inside serial_components
        and calls the getVTKMesh for their specific component type. Each of these is added to the mesh.
        The inlet of the previous component is then used in the getOutlet function to determine the inlet
        of the following component in the series.

        Args:
            inlet : tuple, contains the x, y, and z coordinates of the inlet, respectively
        """
        mesh = VTKMesh()
        for cname in self._order:
            mesh += self._myComponents[cname].getVTKMesh(inlet)
            inlet = self._myComponents[cname].getOutlet(inlet)
        return mesh

    def _convertUnits(self, uc: UnitConverter) -> None:
        """
        This private function will pass in the unit converter and the stored dimensions
        in this component will be multiplied by the corresponding conversion. The units
        will be converted into the base SI units for accurate calculations throughout the
        solvers.

        Args:
            - uc : unit_converter, class that takes all the units from the input file and
                stores the conversions from those units to the base SI units
        """
        for cname in self._order:
            self._myComponents[cname]._convertUnits(uc)


component_list["serial_components"] = SerialComponents
