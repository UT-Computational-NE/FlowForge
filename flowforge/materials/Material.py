from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class Material:
    """Base class for material property handling.

    The Material class contains functionality to convert equations of material
    properties into tabular data that can be exported to hdf5 format. This class
    serves as a base class that is inherited by both Solid and Fluid material classes.

    Parameters
    ----------
    name : str
        Name of the material.

    Attributes
    ----------
    name : str
        Name of the material.
    """

    def __init__(self, name: str):
        self.name = name

    def exportMaterialProperty(
        self, mat_property: Callable, Tmin: float, Tmax: float, thresh: float
    ) -> Tuple[NDArray, NDArray]:
        """Export material property equation as tabular data.

        This method takes a material property equation and a temperature range to
        interpolate property values at corresponding temperatures. It adaptively
        increases the sampling resolution until the interpolation error falls
        below the specified threshold. The resulting temperature and property
        value arrays can be exported to an hdf5 file for use in C++ code.

        Parameters
        ----------
        mat_property : Callable
            Material property equation/function that takes temperature as input
            and returns the corresponding property value.
        Tmin : float
            Minimum temperature [K] in the range to be exported.
        Tmax : float
            Maximum temperature [K] in the range to be exported.
        thresh : float
            Error threshold for determining when the interpolation is sufficiently accurate.
            The adaptive sampling stops when the RMS error falls below this value.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing:
            - Array of temperature values [K]
            - Array of corresponding material property values

        Notes
        -----
        The method uses an adaptive sampling approach that doubles the number of
        sampling points each iteration until the interpolation error falls below
        the specified threshold.
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
