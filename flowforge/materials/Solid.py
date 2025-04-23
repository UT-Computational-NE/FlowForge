from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
import h5py
from scipy.interpolate import interp1d
from flowforge.materials.Material import Material


class Solid(Material, ABC):
    """Abstract base class for solid materials.

    This class inherits from the base Material class and defines the interface for
    solid material property calculations. It provides abstract methods for thermal
    conductivity, density, and specific heat, which must be implemented by concrete
    subclasses. It also provides a method to export these properties to HDF5 format.

    Attributes
    ----------
    name : str
        Name of the solid material, inherited from the Material class.

    Notes
    -----
    This is an abstract class and cannot be instantiated directly. Subclasses must
    implement the conductivity, density, and specific_heat methods.
    """

    @abstractmethod
    def conductivity(self, T: float) -> float:
        """Calculate thermal conductivity at a specified temperature.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            Thermal conductivity [W/m-K].
        """
        raise NotImplementedError

    @abstractmethod
    def density(self, T: float) -> float:
        """Calculate density at a specified temperature.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            Density [kg/m^3].
        """
        raise NotImplementedError

    @abstractmethod
    def specific_heat(self, T: float) -> float:
        """Calculate specific heat at a specified temperature.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        float
            Specific heat [kJ/kg-K].
        """
        raise NotImplementedError

    def exportHDF5(
        self, filename: str, path: str = "/", Tmin: float = 273.15, Tmax: float = 1273.15, thresh: float = 0.1
    ) -> None:
        """Export solid material properties to HDF5 file format.

        This method exports thermal conductivity, density, and specific heat data
        for the solid material to an HDF5 file. The data is exported over a specified
        temperature range with adaptive sampling to ensure accuracy.

        Parameters
        ----------
        filename : str
            Name of the HDF5 file to add data to.
        path : Optional[str]
            Path in the HDF5 file where solid properties will be written, by default "/".
        Tmin : Optional[float]
            Optional[Minimum temperature [K] to export property values, by default 273.15 (0°C).
        Tmax : Optional[float]
            Maximum temperature [K] to export property values, by default 1273.15 (1000°C).
        thresh : Optional[float]
            Tolerance for the linear interpolation error, by default 0.1.

        Notes
        -----
        If the specified group already exists in the HDF5 file, it will be deleted
        before the new data is written.
        """
        h5file = h5py.File(filename, "a")
        group_name = path + "solid_material"
        if group_name in h5file:
            del h5file[group_name]
        solid = h5file.create_group(path + "solid_material")
        k_temps, k_vals = self.exportMaterialProperty(self.conductivity, Tmin, Tmax, thresh)
        cp_temps, cp_vals = self.exportMaterialProperty(self.specific_heat, Tmin, Tmax, thresh)
        rho_temps, rho_vals = self.exportMaterialProperty(self.density, Tmin, Tmax, thresh)
        solid.create_dataset("conductivity/temps", data=k_temps, dtype=float)
        solid.create_dataset("conductivity/values", data=k_vals, dtype=float)
        solid.create_dataset("specific_heat/temps", data=cp_temps, dtype=float)
        solid.create_dataset("specific_heat/values", data=cp_vals, dtype=float)
        solid.create_dataset("density/temps", data=rho_temps, dtype=float)
        solid.create_dataset("density/values", data=rho_vals, dtype=float)
        h5file.close()


class Graphite(Solid):
    """Graphite solid material implementation.

    This class provides thermal property calculations for G-348 graphite, with
    property equations derived from published literature.

    Attributes
    ----------
    name : str
        Name of the material (inherited from Material class).

    Notes
    -----
    The property equations implemented in this class are based on:
    McEligot, Donald, Swank, W. David, Cottle, David L., and Valentin, Francisco I.
    "Thermal Properties of G-348 Graphite." United States: N. p., 2016.
    """

    def conductivity(self, T: float) -> float:
        return 134.0 - 0.1074 * (T - 273.15) + 3.719e-5 * (T - 273.15) ** 2

    def density(self, T: float) -> float:
        return (-6e-9 * (T - 273.15) ** 2 - 3e-5 * (T - 273.15) + 1.8891) * 1000

    def specific_heat(self, T: float) -> float:
        # convert T from C to K
        T = T + 273.15
        Cp = (
            0.538657 + 9.11129e-6 * T - 90.2725 / T - 43449.3 / (T * T) + 1.59309e7 / (T * T * T) - 1.43688e9 / (T * T * T * T)
        )
        # convert Cp from cal/g-K to kJ/kg-K
        Cp *= 4.184
        return Cp


class SS316H(Solid):
    """Stainless Steel 316H solid material implementation.

    This class provides thermal property calculations for SS316H stainless steel,
    with properties based on manufacturer specifications.

    Attributes
    ----------
    name : str
        Name of the material (inherited from Material class).

    Notes
    -----
    The property values implemented in this class are based on:
    Atlas Steels. "Grade Data Sheet 316/316L/316H." January 2011.
    https://www.atlassteels.com.au/documents/Atlas_Grade_datasheet_316_rev_Jan_2011.pdf

    Some properties, like density, are considered constant across the typical
    operational temperature range.
    """

    def conductivity(self, T: float) -> float:
        return 0.013 * T + 11.45305

    def density(self, T: Optional[float] = None) -> float:
        return 8000

    def specific_heat(self, T: Optional[float] = None) -> float:
        if T is not None and T < 273.15 or T > 373.15:
            print(
                f"WARNING: Temperature {T:.2f} K ({T-273.15:.2f} C) is outside of manufacturer specifications \
                  for specific heat of 0-100 C."
            )
        return 0.500


class User_Solid(Solid):
    """User-defined solid material implementation.

    This class allows users to define their own property functions for custom solid
    materials. The user provides functions for thermal conductivity, density, and
    specific heat as a function of temperature.

    Parameters
    ----------
    name : str
        Name of the solid material.
    k_funct : Callable
        Function that takes temperature as input and returns thermal conductivity.
    dens_funct : Callable
        Function that takes temperature as input and returns density.
    cp_funct : Callable
        Function that takes temperature as input and returns specific heat.

    Attributes
    ----------
    name : str
        Name of the material (inherited from Material class).
    k_funct : Callable
        Function for calculating thermal conductivity [W/m-K].
    dens_funct : Callable
        Function for calculating density [kg/m^3].
    cp_funct : Callable
        Function for calculating specific heat [kJ/kg-K].

    Notes
    -----
    The functions can be provided either as callable objects or as strings
    that can be evaluated into callable objects.
    """

    def __init__(self, name: str, k_funct: Callable, dens_funct: Callable, cp_funct: Callable):
        Solid.__init__(self, name)
        if isinstance(k_funct, str) and isinstance(dens_funct, str) and isinstance(cp_funct, str):
            self.k_funct = eval(str(k_funct))
            self.dens_funct = eval(str(dens_funct))
            self.cp_funct = eval(str(cp_funct))
        else:
            self.k_funct = k_funct
            self.dens_funct = dens_funct
            self.cp_funct = cp_funct

    def conductivity(self, T: float) -> float:
        return self.k_funct(T)

    def density(self, T: float) -> float:
        return self.dens_funct(T)

    def specific_heat(self, T: float) -> float:
        return self.cp_funct(T)


class Solid_table(User_Solid):
    """Table-based solid material implementation.

    This class allows users to define a solid material using tabular data for
    thermal properties. The tabular data is interpolated to create property
    functions that can be used for calculations.

    Parameters
    ----------
    name : str
        Name of the solid material.
    T_k : List[float]
        List of temperature values [°C] corresponding to conductivity values.
    k : List[float]
        List of thermal conductivity values [W/m-K] corresponding to T_k temperatures.
    T_dens : List[float]
        List of temperature values [°C] corresponding to density values.
    dens : List[float]
        List of density values [kg/m^3] corresponding to T_dens temperatures.
    T_cp : List[float]
        List of temperature values [°C] corresponding to specific heat values.
    cp : List[float]
        List of specific heat values [kJ/kg-K] corresponding to T_cp temperatures.

    Attributes
    ----------
    name : str
        Name of the material (inherited from Material class).
    T_k : List[float]
        List of temperatures for conductivity data.
    k : List[float]
        List of conductivity values.
    T_dens : List[float]
        List of temperatures for density data.
    dens : List[float]
        List of density values.
    T_cp : List[float]
        List of temperatures for specific heat data.
    cp : List[float]
        List of specific heat values.
    Tmin : Optional[float]
        Minimum temperature for export (set by export methods).
    Tmax : Optional[float]
        Maximum temperature for export (set by export methods).
    thresh : Optional[float]
        Error threshold for export (set by export methods).

    Notes
    -----
    The class uses linear interpolation between data points to estimate
    property values at any temperature within the provided ranges.
    Each property (conductivity, density, specific heat) can have its own
    temperature points.
    """

    def __init__(
        self,
        name: str,
        T_k: List[float],
        k: List[float],
        T_dens: List[float],
        dens: List[float],
        T_cp: List[float],
        cp: List[float],
    ):
        self.T_k = T_k
        self.k = k
        self.T_dens = T_dens
        self.dens = dens
        self.T_cp = T_cp
        self.cp = cp
        # Initialize parameters for export methods to avoid unused argument warnings
        self.Tmin = None
        self.Tmax = None
        self.thresh = None

        assert len(T_k) == len(k)
        assert len(T_dens) == len(dens)
        assert len(T_cp) == len(cp)

        User_Solid.__init__(self, name, interp1d(T_k, k), interp1d(T_dens, dens), interp1d(T_cp, cp))

    def exportConductivity(
        self, Tmin: Optional[float] = None, Tmax: Optional[float] = None, thresh: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """Export thermal conductivity data.

        This method returns the stored thermal conductivity data values
        and their corresponding temperature values. It also stores the
        provided parameters for potential future use.

        Parameters
        ----------
        Tmin : Optional[float]
            Minimum temperature [K], stored but not used for table export.
        Tmax : Optional[float]
            Maximum temperature [K], stored but not used for table export.
        thresh : Optional[float]
            Error threshold, stored but not used for table export.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing:
            - List of temperature values [°C]
            - List of corresponding conductivity values [W/m-K]
        """
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.thresh = thresh
        return self.T_k, self.k

    def exportDensity(
        self, Tmin: Optional[float] = None, Tmax: Optional[float] = None, thresh: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """Export density data.

        This method returns the stored density data values
        and their corresponding temperature values. It also stores the
        provided parameters for potential future use.

        Parameters
        ----------
        Tmin : Optional[float]
            Minimum temperature [K], stored but not used for table export.
        Tmax : Optional[float]
            Maximum temperature [K], stored but not used for table export.
        thresh : Optional[float]
            Error threshold, stored but not used for table export.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing:
            - List of temperature values [°C]
            - List of corresponding density values [kg/m^3]
        """
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.thresh = thresh
        return self.T_dens, self.dens

    def exportSpecificHeat(
        self, Tmin: Optional[float] = None, Tmax: Optional[float] = None, thresh: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """Export specific heat data.

        This method returns the stored specific heat data values
        and their corresponding temperature values. It also stores the
        provided parameters for potential future use.

        Parameters
        ----------
        Tmin : Optional[float]
            Minimum temperature [K], stored but not used for table export.
        Tmax : Optional[float]
            Maximum temperature [K], stored but not used for table export.
        thresh : Optional[float]
            Error threshold, stored but not used for table export.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing:
            - List of temperature values [°C]
            - List of corresponding specific heat values [kJ/kg-K]
        """
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.thresh = thresh
        return self.T_cp, self.cp
