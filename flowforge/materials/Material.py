from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class Material:
    """
    The Material class contains the function to convert equations of material properties into tabular data
    that can be exported to hdf5. The Material class is inherited by Solid and Fluid materials.
    """

    def __init__(self, name: str):
        """
        The __init__ function initializes the Material class instance by storing the name
        of the material.

        Args:
            - name : string, name of the material
        """
        self.name = name

    def exportMaterialProperty(
        self, mat_property: Callable, Tmin: float, Tmax: float, thresh: float
    ) -> Tuple[NDArray, NDArray]:
        """
        This function takes a material property equation and a range of temperatures to interpolate
        property values at the corresponding temperatures. These values are added to 2 arrays which are returned.
        The purpose of this function is give the ability to export the property equations to an hdf5 file for use
        in the C++ code.

        Args:
            - mat_property : lambda function, the material property equation being exported
            - Tmin         : double, [K] minimum temperature in the range to be exported
            - Tmax         : double, [K] maximum temperature in the range to be exported
            - thresh       : double, error threshold for the interpolated values
        """

        n = 2
        Tnew = np.linspace(Tmin, Tmax, n)
        fnew = mat_property(Tnew)
        err = 2 * thresh
        while err >= thresh:
            Told = Tnew
            fold = fnew
            n *= 2
            Tnew = np.linspace(Tmin, Tmax, n)
            finter = interp1d(Told, fold)(Tnew)
            fnew = mat_property(Tnew)
            err = np.sqrt(np.dot(finter - fnew, finter - fnew) / float(n))

        return Told, fold
