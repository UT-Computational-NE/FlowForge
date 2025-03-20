from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
import h5py
from scipy.interpolate import interp1d
from flowforge.materials.Material import Material


class Solid(Material, ABC):
    """
    The Solid class inherits from the base Material class. The purpose of this class is to store material
    properties that are relevant to the solid domain calculations. The solid material properties that are stored
    are thermal conductivity, density, and specific heat of the material.
    """

    @abstractmethod
    def conductivity(self, T: float) -> float:
        """
        Thermal conductivity [W/m-K]

        Args:
            - T (float) : temperature [K]
        """
        raise NotImplementedError

    @abstractmethod
    def density(self, T: float) -> float:
        """
        Density [kg/m^3]

        Args:
            - T (float) : temperature [K]
        """
        raise NotImplementedError

    @abstractmethod
    def specific_heat(self, T: float) -> float:
        """
        Specific heat [kJ/kg-K]

        Args:
            - T (float) : temperature [K]
        """
        raise NotImplementedError

    def exportHDF5(
        self, filename: str, path: str = "/", Tmin: float = 273.15, Tmax: float = 1273.15, thresh: float = 0.1
    ) -> None:
        """
        The exportHDF5 function exports all of the property data for the Solid.

        Args:
            - filename : str, name of the HDF5 file to add data to
            - path     : (OPTIONAL) str, path in the h5 file solid properties will be written to
            - Tmin     : (OPTIONAL) float, [K] minimum temperature to export property values
            - Tmax     : (OPTIONAL) float, [K] maximum temperature to export property values
            - thresh   : (OPTIONAL) float, tolerance in the linear interpolation error
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
    """
    The graphite Solid subclass is specific to the properties of graphite from "McEligot, Donald,
    Swank, W. David, Cottle, David L., and Valentin, Francisco I. Thermal Properties of G-348 Graphite.
    United States: N. p., 2016."
    """

    def conductivity(self, T: float) -> float:
        """
        Returns the conductivity (k) of graphite at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : W/m-k
        """
        return 134.0 - 0.1074 * (T - 273.15) + 3.719e-5 * (T - 273.15) ** 2

    def density(self, T: float) -> float:
        """
        Returns the density (rho) of graphite at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : kg/m3
        """
        return (-6e-9 * (T - 273.15) ** 2 - 3e-5 * (T - 273.15) + 1.8891) * 1000

    def specific_heat(self, T: float) -> float:
        """
        Returns the specific heat (cp) of graphite at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : kJ/kg-K
        """
        # convert T from C to K
        T = T + 273.15
        Cp = (
            0.538657 + 9.11129e-6 * T - 90.2725 / T - 43449.3 / (T * T) + 1.59309e7 / (T * T * T) - 1.43688e9 / (T * T * T * T)
        )
        # convert Cp from cal/g-K to kJ/kg-K
        Cp *= 4.184
        return Cp


class SS316H(Solid):
    """
    The stainless steel 316H Solid subclass is specific to the properties from "Atlas Steels.
    'Grade Data Sheet 316/316L/316H.' January 2011.
    https://www.atlassteels.com.au/documents/Atlas_Grade_datasheet_316_rev_Jan_2011.pdf"
    """

    def conductivity(self, T: float) -> float:
        """
        Returns the conductivity (k) of SS316H at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : W/m-k
        """
        return 0.013 * T + 11.45305

    def density(self, T: Optional[float] = None) -> float:
        """
        Returns the density (rho) of SS316H at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : kg/m3
        """
        return 8000

    def specific_heat(self, T: Optional[float] = None) -> float:
        """
        Returns the specific heat (cp) of SS316H at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : kJ/kg-K
        """
        if T is not None and T < 273.15 or T > 373.15:
            print(
                f"WARNING: Temperature {T:.2f} K ({T-273.15:.2f} C) is outside of manufacturer specifications \
                  for specific heat of 0-100 C."
            )
        return 0.500


class User_Solid(Solid):
    """
    The User_Solid subclass of the Solid base class allows for the user to
    define their own property functions for the material being used.
    """

    def __init__(self, name: str, k_funct: Callable, dens_funct: Callable, cp_funct: Callable):
        """
        The User_Solid subclass initializes by sending the name to the base
        class for initialization and then stores all 3 of the property functions.

        Args:
            - name : string, name of the solid material
            - k_funct    : function of temperature which returns the conductivity of the material (k)
            - dens_funct : function of temperature which returns the density of the material (rho)
            - cp_funct   : function of temperature which returns the specific heat of the material (cp)
        """
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
        """
        Returns the conductivity (k) of the solid material at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : W/m-k
        """
        return self.k_funct(T)

    def density(self, T: float) -> float:
        """
        Returns the density (rho) of the solid material at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : kg/m3
        """
        return self.dens_funct(T)

    def specific_heat(self, T: float) -> float:
        """
        Returns the specific heat (cp) of the solid material at a specified temperature.

        Args:
            - T (float) : temperature [K]

        Units : kJ/kg-K
        """
        return self.cp_funct(T)


class Solid_table(User_Solid):
    """
    The Solid_table subclass of Solid_functions class allows the user to input
    tabular data and the data will be interpolated to generate property functions for the material
    being used. These functions are then sent to the Solid_functions class to be initialized
    there.
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
        """
        The __init__ function of the Solid_table class initializes the class instance by
        linearly interpolating between data points in a table. Solid material data of density,
        specific heat, and conductivity often come in the form of tables.

        This will allow the user to make a variable for each column that contains the values as a list.

        These lists will be used to interpolate an equation for each of the material properties and
        these equations will be initialized as a Solid_functions class using the super() method.

        The list inputs for the material properties must be the same length as the corresponding list of
        temperature values.

        Args:
            - name      : str, name of the material
            - T_k       : float list, list of temperature values (degrees C) that coincide with the k values
            - k         : float list, list of conductivity values (W/m-k) at each temperature in T_k
            - T_dens    : float list, list of temperature values (degrees C) that coincide with the dens values
            - dens      : float list, list of density values (kg/m3) at each temperature in T_dens
            - T_cp      : float list, list of temperature values (degrees C) that coincide with the cp values
            - cp        : float list, list of specific heat values (kJ/kg-K) at each temperature in T_cp
        """
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
        """
        This function exports the stored thermal conductivity data values
        and the corresponding temperature values.

        Args: None
        """
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.thresh = thresh
        return self.T_k, self.k

    def exportDensity(
        self, Tmin: Optional[float] = None, Tmax: Optional[float] = None, thresh: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """
        This function exports the stored density data values
        and the corresponding temperature values.

        Args: None
        """
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.thresh = thresh
        return self.T_dens, self.dens

    def exportSpecificHeat(
        self, Tmin: Optional[float] = None, Tmax: Optional[float] = None, thresh: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """
        This function exports the stored specific heat data values
        and the corresponding temperature values.

        Args: None
        """
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.thresh = thresh
        return self.T_cp, self.cp
