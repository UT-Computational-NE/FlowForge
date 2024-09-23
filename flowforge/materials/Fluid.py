import abc
import h5py
import numpy as np
from flowforge.materials.Material import Material

class Fluid(Material):
    """
    Fluid class stores fluid property data and returns values describing thermodynamic state of fluid given
    an enthalpy. Returns Thermal Conductivity, Density, Surface Tension, Specific Heat, Temperature,
    and Prandtl Number given enthalpy and returns Reynolds Number given enthalpy, velocity, and hydraulic diameter.
    The enthalpy function will return an enthalpy given a Temperature.
    """
    def __init__(self, name):
        """
        The __init__ function initializes the Fluid class instance by storing the name
        of the fluid.

        Args:
            - name : string, name of the fluid
        """
        super().__init__(name)
        self._fluid_property_array = [
            ["thermal_conductivity", self.conductivity],
            ["density", self.density],
            ["viscosity", self.viscosity],
            ["surface_tension", self.surface_tension],
            ["specific_heat", self.specific_heat],
            ["temperature", self.temperature]
        ]

    @abc.abstractmethod
    def conductivity(self, h):
        """
        Thermal conductivity [W/m-K]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def density(self, h):
        """
        Density [kg/m^3]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def viscosity(self, h):
        """
        Viscosity [kg/m-s]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def specific_heat(self, h):
        """
        Specific heat [J/kg-K]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def temperature(self, h):
        """
        Temperature [K]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def enthalpy(self, T):
        """
        Enthalpy [J/kg]
        """
        raise NotImplementedError

    def surface_tension(self, h):
        """
        Surface tension [N/m]

        This method is intended for liquid fluids surrounded by gases which have
        negligible influence on the liquid surface tension (ex: dry air, argon, helium).
        Concrete implementations which have a surface tension should override this method.
        """
        raise RuntimeError("Surface tension not implemented for this fluid")

    def Pr(self, h):
        """
        Prandtl number
        """
        return self.specific_heat(h)*self.viscosity(h) / self.conductivity(h)

    def Re(self, h, V, Dh) -> float:
        """
        Args:
            - h  : float, specific enthalpy
            - V  : float, velocity
            - Dh : float, hydraulic diameter

        Returns:
            - float, Reynold's number
        """
        return self.density(h)*V*Dh / self.viscosity(h)

    def fluidPropertyArray(self):
        """
        fluid property array stores property name and respective function which allows
        the export process to iterate over the array
        """
        return self._fluid_property_array

    def exportHDF5(self, filename, path="/", Tmin=273.15 ,Tmax=1300, thresh=1e-2):
        """
        The exportHDF5 function exports all of the property data for the Fluid.

        Args:
            - filename : str, name of the HDF5 file to add data to
            - path     : (OPTIONAL) str, path in the h5 file fluid_properties will be written to
            - Tmin     : (OPTIONAL) float, [K] minimum temperature to export property values
            - Tmax     : (OPTIONAL) float, [K] maximum temperature to export property values
            - thresh   : (OPTIONAL) float, tolerance in the linear interpolation error
        """
        Hmin = self.enthalpy(Tmin)
        Hmax = self.enthalpy(Tmax)
        h5file = h5py.File(filename, 'a')
        fluid = h5file.create_group(path+"fluid_properties")
        prop_array = self.fluidPropertyArray()
        #prop array holds name of property and property function
        #the name is used to name the dataset and the function is used to get the y data (property) and x data (enthalpy)
        for item in prop_array:
            prop = fluid.create_group(item[0])
            h_array, prop_array = self.exportMaterialProperty(item[1], Hmin, Hmax, thresh)
            prop.create_dataset(item[0], data=prop_array)
            prop.create_dataset("enthalpy", data=h_array)
        h5file.close()


class FLiBe_UF4(Fluid):
    """
    Sub class with specific FLiBe_UF4 fluid properties enumerated


    Notes
    -----

    Molar Percent: 67-33 mol% (2LiF-BeF_2)

    Caveats:
    In this section a list of temperature dependent caveats are presented
    with any listed uncertainties from the literature [1,2,3]

    References
    -----

    [1] C. Davis, "Implementation of Molten Salt Properties into RELAP5-3D/ATHENA," U.S. Department of Energy,
        Tech. Rep., Jan. 2005. doi: 10.2172/910991. Available: https://www.osti.gov/biblio/910991.

    [2] R. Romatoski and L.-W. Hu, "Fluoride salt coolant properties for nuclear reactor applications:
        A review," *Annals of Nuclear Energy*, vol. 109, pp. 635-647, Nov. 2017. doi: 10.1016/j.anucene.2017.05.036.

    [3] M. S. Sohal, M. A. Ebner, P. Sabharwall, and P. Sharpe, "Engineering Database of Liquid Salt Thermophysical
        and Thermochemical Properties," Idaho National Laboratory, Tech. Rep. INL/EXT-10-18297, Rev. 1, June 2013.
        Available: https://inldigitallibrary.inl.gov/sites/STI/STI/5698704.pdf.

    """
    def __init__(self, name):
        super().__init__(name)
        self._Tref = 273.15  # temperature where enthalpy is 0

    def conductivity(self, h):
        """
        Thermal conductivity [W/m-K]:
        Validated for temp range 459-610 K and at 873 K with ± 10-50% uncertainty,
        ref. [1], pg.  10, Table 7.
        ref. [2], pg. 638, Table 3.
        """
        return 1.1 + h*0

    def density(self, h):
        """
        Density [kg/m^3]:
        Validated for temp range 800-1080 K,
        ref. [3], pg. 6, eq. (2.13)
        """
        T = self.temperature(h)
        return 2413 - (0.488*T)

    def viscosity(self, h):
        """
        Dynamic viscosity [kg/m-s]:
        Validated for temp range 873-1073 K,
        ref. [3], pg. 7, eq. (2.18)
        """
        T = self.temperature(h)
        return (0.116e-3)*np.exp(3755.0/T)

    def surface_tension(self, h):
        """
        Surface tension [N/m]:
        Validated for temp range 773.15-1073.15 K with ± 3% uncertainty,
        under a dry argon, helium or nitrogen gas enviorment,
        ref. [3], pg. 8 and 33, eq. (2.19)
        """
        T = self.temperature(h)
        return 0.295778-((0.12e-3)*T)

    def specific_heat(self, h):
        """
        Specific heat capacity [J/kg-K] (Isobaric):
        Validated for temp range 788-1093 K with ± 3% uncertainty,
        ref. [1], pg.   3, Table 1.
        ref. [2], pg. 637, Table 2.
        """
        return 2386 + h*0

    def temperature(self, h):
        """
        Temperature [K]
        """
        temp = h/self.specific_heat(h) + self._Tref  # only true because specific heat is constant
        assert np.all(temp >= 0)
        return temp

    def enthalpy(self, T):
        """
        Specific enthalpy [J/kg]
        """
        return self.specific_heat(0)*(T - self._Tref) # only true because specific heat is constant


class Hitec(Fluid):
    """
    Sub class with specific NaNO3_NaNO2_KNO3 (Hitec) fluid properties enumerated

    Notes
    -----

    Molar Percent: 7-49-44 mol%
    Weight Percent: 7-40-53 wt.%

    Caveats:
    In this section a list of temperature dependent caveats are presented
    with uncertainties from the literature [1,2]

    References
    -----

    [1] R. Santini, L. Tadrist, J. Pantaloni, and P. Cerisier, "Measurement of thermal conductivity of molten salts
        in the range 100-500°C," *International Journal of Heat and Mass Transfer*, vol. 21, no. 4, pp. 623-626, 1984.
        doi: 10.1016/0017-9310(84)90034-6.

    [2] M. S. Sohal, M. A. Ebner, P. Sabharwall, and P. Sharpe, "Engineering Database of Liquid Salt Thermophysical
        and Thermochemical Properties," Idaho National Laboratory, Tech. Rep. INL/EXT-10-18297, Rev. 1, June 2013.
        Available: https://inldigitallibrary.inl.gov/sites/STI/STI/5698704.pdf.

    """
    def __init__(self, name):
        super().__init__(name)
        self._Tref = 273.15  # temperature where enthalpy is 0

    def conductivity(self, h):
        """
        Thermal conductivity [W/m-K]:
        Validated for temp range 373-773 K with ± 5% uncertainty,
        ref. [1], pg. 625, Table 1.
        """
        a =  0.78
        b = -1.25e-3
        c =  1.6e-6
        T = self.temperature(h)
        return a + (b*T) + (c*(T**2))

    def density(self, h):
        """
        Density [kg/m^3]:
        Validated for temp range 470-870 K with ± 2% uncertainty,
        ref. [2], pg. 13, eq. (2.31)
        """
        a =  2293.6
        b = -0.7497
        T = self.temperature(h)
        return a + (b*T)

    def viscosity(self, h):
        """
        Dynamic viscosity [kg/m-s]:
        Validated for temp range 420-710 K with ± 16% uncertainty,
        ref. [2], pg. 13, eq. (2.32)
        """
        a =  0.4737
        b = -2.297e-3
        c =  3.731e-6
        d = -2.019e-9
        T = self.temperature(h)
        return a + (b*T) + (c*(T**2)) + (d*(T**3))

    def surface_tension(self, h):
        """
        Surface tension [N/m]:
        Validated for temp range 570-670 K with ± 10% uncertainty,
        under a dry argon, helium or nitrogen gas enviorment,
        ref. [2], pg. 14 and 33, eq. (2.33)
        Note:
        """
        a = 0.14928
        b = -0.556e-4
        T = self.temperature(h)
        return a + (b*T)

    def specific_heat(self, h):
        """
        Specific heat capacity [J/kg-K] (Isobaric):
        Validated for temp range 426-776 K with ± 5% uncertainty,
        ref. [2], pg. 14, eq. (2.34)
        """
        a =  5806.0
        b = -10.833
        c =  7.2413e-3
        T = self.temperature(h)
        return a + (b*T) + (c*(T**2))

    def temperature(self, h):
        """
        Temperature [K]

        This method is based on takng the integral of:

        dh(T)/dT = C_p(T) from T_1 to T_2

        where,

        Delta_h(T) = h(T_2) - h(T_1) = h - 0

        and,

        C_p(T) = d*T^2 + c*T + b

        thus,

        h = d/3*T^3 + c/2*T^2 + b*T + a

        Note: h(T_1) is a reference term where enthalpy is 0 at T = T_1

        TODO: The temp method could be optmized and the math behind the method
        could be further explained.
        """
        d =  7.2413e-3
        c = -10.833
        b =  5806.0
        a = (0 - h) - (b*(self._Tref) + ((c/2)*(self._Tref**2)) + ((d/3)*(self._Tref**3)))
        if not np.isscalar(h):
            a_flattened = np.array(a).flatten()
            T_val = np.array([])
            for a_index in a_flattened:
                T_roots = np.roots(np.array([d/3,c/2,b,a_index]))
                assert np.sum(np.isclose(T_roots.imag, 0) & (T_roots.real >= 0)) == 1
                T_root = T_roots[np.isclose(T_roots.imag, 0) & (T_roots.real >= 0)].real[0]
                T_val = np.append(T_val,T_root)
            T_val = np.reshape(T_val,np.array(h).shape)
        else:
            T_roots = np.roots(np.array([d/3,c/2,b,a]))
            assert np.sum(np.isclose(T_roots.imag, 0) & (T_roots.real >= 0)) == 1
            T_val = T_roots[np.isclose(T_roots.imag, 0) & (T_roots.real >= 0)].real[0]
        return T_val

    def enthalpy(self, T):
        """
        Specific enthalpy [J/kg]
        """
        # a = 0
        b =  5806.0
        c = -10.833
        d =  7.2413e-3
        # Integrated specific heat from _Tref to T
        return b*(T-self._Tref)+((c/2)*((T**2)-(self._Tref**2)))+((d/3)*((T**3)-(self._Tref**3)))


class Helium(Fluid):
    """
    Sub class with constant specific properties of helium.

    References
    -----

    [1] National Bureau of Standards. Use of the Computer Language Pascal in Developing the Initial Graphics Exchange
    Specification (IGES) Subset Translator. NIST Technical Note 1334, U.S. Department of Commerce, 1990.
    https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1334.pdf.

    """
    def __init__(self, name):
        super().__init__(name)
        self._Tref = 273.15 # K, temperature where enthalpy is 0

    def conductivity(self, h):
        """
        Thermal conductivity at P=0.120 MPa, T=600 K (p.37) [W/m-K]
        """
        return 0.2524

    def density(self, h):
        """
        Density at P=0.120 MPa, T=600 K (p.36) [kg/m^3]
        """
        return 0.9626e-01

    def viscosity(self, h):
        """
        Dynamic viscosity at P=0.120 MPa, T=600 K (p.37) [kg/m-s]
        """
        return 32.22 * 1e-6

    def specific_heat(self, h):
        """
        Specific heat capacity at P=0.120 MPa, T=600 K (p.36) [J/kg-K] (Isobaric)
        """
        return 5193.0

    def temperature(self, h):
        """
        Temperature [K]
        """
        temp = h/self.specific_heat(h) + self._Tref
        assert np.all(temp >= 0)
        return temp

    def enthalpy(self, T):
        """
        Specific enthalpy [J/kg]
        """
        return self.specific_heat(0)*(T - self._Tref) # only true because specific heat is constant


class User_Fluid(Fluid):
    """
    The User_Fluid subclass of the Fluid base class allows for the user to
    define their own property functions for the material being used.
    """
    def __init__(self, name, therm_cond_funct, dens_funct, visco_funct,
                 spec_heat_funct, temp_funct, entha_funct,surf_tens_funct=None):
        """
        The User_Fluid subclass initializes by sending the name to the base
        class for initialization and then stores the 6 property functions
        with 1 optional property function.

        Args:
            - name : string, name of the solid material
            - therm_cond_funct   : function of enthalpy which returns the conductivity of the fluid [W/m-K]
            - dens_funct         : function of enthalpy which returns the density of the fluid [kg/m^3]
            - visco_funct        : function of enthalpy which returns the viscosity of the fluid [kg/m-s]
            - spec_heat_funct    : function of enthalpy which returns the specific heat of the fluid [J/kg-K]
            - temp_funct         : function of enthalpy which returns the temperature of the fluid [K]
            - entha_funct        : function of temperature which returns the enthalpy of the material [J/kg]
            - surf_tens_funct (optional) : function of enthalpy which returns the surface tension of the fluid [N/m]
        """
        super().__init__(name)
        self.thermal_conductivity_fun = therm_cond_funct
        self.density_fun = dens_funct
        self.viscosity_fun = visco_funct
        self.surface_tension_fun = surf_tens_funct
        self.specific_heat_fun = spec_heat_funct
        self.temperature_fun = temp_funct
        self.enthalpy_fun = entha_funct

    def conductivity(self, h):
        """
        Thermal conductivity [W/m-K]
        """
        return self.thermal_conductivity_fun(h)

    def density(self, h):
        """
        Density [kg/m^3]
        """
        return self.density_fun(h)

    def viscosity(self, h):
        """
        Viscosity [kg/m-s]
        """
        return self.viscosity_fun(h)

    def surface_tension(self, h):
        """
        Surface tension [N/m]
        """
        assert self.surface_tension_fun is not None
        return self.surface_tension_fun(h)

    def specific_heat(self, h):
        """
        Specific heat [J/kg-K]
        """
        return self.specific_heat_fun(h)

    def temperature(self, h):
        """
        Temperature [K]
        """
        return self.temperature_fun(h)

    def enthalpy(self, T):
        """
        Enthalpy [J/kg]
        """
        return self.enthalpy_fun(T)
