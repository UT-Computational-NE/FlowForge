import abc
import h5py
import numpy as np
from flowforge.materials.Material import Material

class Fluid(Material):
    """
    Fluid class stores fluid property data and returns values describing thermodynamic state of fluid given
    an enthalpy. Returns Thermal Conductivity, Density, Specific Heat, Temperature, and Prandtl Number given and enthalpy.
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
    """
    def __init__(self, name):
        super().__init__(name)
        self._Tref = 273.15  # temperature where enthalpy is 0

    def conductivity(self, h):
        """
        Thermal conductivity [W/m-K]
        """
        return 1.1*(h**0)

    def density(self, h):
        """
        Density [kg/m^3]
        """
        return 2413 - 0.488*self.temperature(h)

    def viscosity(self, h):
        """
        Dynamic viscosity [kg/m-s]
        """
        return 0.116*np.exp(3755.0/self.temperature(h))*1e-3

    def specific_heat(self, h):
        """
        Specific heat capacity [J/kg-K] (Isobaric)
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

    Caveates:
    In this section a list of temperature dependent caveates are presented
    with uncertainties from the liturature

    Thermal conductivity:       valid for temp range 373-773 K with plus minus 5% uncertainty  [1]
    Density:                    valid for temp range 470-870 K with plus minus 2% uncertainty  [2]
    Dynamic viscosity:          valid for temp range 420-710 K with plus minus 16% uncertainty [2]
    Specific heat:              valid for temp range 426-776 K with plus minus 5% uncertainty  [2]

    References
    -----

    [1] R. Santini, L. Tadrist, J. Pantaloni, and P. Cerisier, “Measurement of thermal
        conductivity of molten salts in the range 100-500 C,” Int. J. Heat Mass
        Transfer 27, 623-626 (1984).

    [2] M. S. Sohal, M. A. Ebner, P. Sabharwall, and P. Sharpe, “Engineering database
        of liquid salt thermophysical and thermochemical properties,” Technical
        Report No. INL/EXT-10-18297 (2010).

    """
    def __init__(self, name):
        super().__init__(name)
        self._Tref = 273.15  # temperature where enthalpy is 0

    def conductivity(self, h):
        """
        Thermal conductivity [W/m-K]
        """
        a =  0.78
        b = -1.25e-3
        c =  1.6e-6
        return a + (b*self.temperature(h)) + (c*(self.temperature(h)**2))

    def density(self, h):
        """
        Density [kg/m^3]
        """
        a =  2293.6
        b = -0.7497
        return a + (b*self.temperature(h))

    def viscosity(self, h):
        """
        Dynamic viscosity [kg/m-s]
        """
        a =  0.4737
        b = -2.297e-3
        c =  3.731e-6
        d = -2.019e-9
        return a + (b*self.temperature(h)) + (c*(self.temperature(h)**2)) + (d*(self.temperature(h)**3))

    def specific_heat(self, h):
        """
        Specific heat capacity [J/kg-K] (Isobaric)
        """
        a =  5806.0
        b = -10.833
        c =  7.2413e-3
        return a + (b*self.temperature(h)) + (c*(self.temperature(h)**2))

    def temperature(self, h):
        """
        Temperature [K]
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
        Thermal conductivity [W/m-K]
        """
        return 0.1735

    def density(self, h):
        """
        Density [kg/m^3]
        """
        return 0.0732

    def viscosity(self, h):
        """
        Dynamic viscosity [kg/m-s]
        """
        return 27.15e-6

    def specific_heat(self, h):
        """
        Specific heat capacity [J/kg-K] (Isobaric)
        """
        return 5193.0

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


class User_Fluid(Fluid):
    """
    The User_Fluid subclass of the Fluid base class allows for the user to
    define their own property functions for the material being used.
    """
    def __init__(self, name, therm_cond_funct, dens_funct, visco_funct, spec_heat_funct, temp_funct, entha_funct):
        """
        The User_Fluid subclass initializes by sending the name to the base
        class for initialization and then stores the 6 property functions.

        Args:
            - name : string, name of the solid material
            - therm_cond_funct : function of enthalpy which returns the conductivity of the fluid [W/m-K]
            - dens_funct       : function of enthalpy which returns the density of the fluid [kg/m^3]
            - visco_funct      : function of enthalpy which returns the viscosity of the fluid [kg/m-s]
            - spec_heat_funct  : function of enthalpy which returns the specific heat of the fluid [J/kg-K]
            - temp_funct       : function of enthalpy which returns the temperature of the fluid [K]
            - entha_funct      : function of temperature which returns the enthalpy of the material [J/kg]
        """
        super().__init__(name)
        self.thermal_conductivity_fun = therm_cond_funct
        self.density_fun = dens_funct
        self.viscosity_fun = visco_funct
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
