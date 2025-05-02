from __future__ import annotations
from typing import Optional
import numpy as np
from pyevtk.hl import unstructuredGridToVTK
from flowforge.visualization import VTKMesh


class VTKFile:
    """Class which provides the ability to write VTK mesh data to a file

    If no data is set using the :meth:`__setitem__` method, the file will
    still write with a data list that is set by default to be empty. No errors
    will occur if a mesh is desired with no data values.

    Parameters
    ----------
    filepath : str
        Desired name of the export VTK file
    mesh : VTKMesh
        A fully defined :class:`VTKMesh` object to be used for file writing
    """

    def __init__(self, filepath: str, mesh: Optional[VTKMesh] = None):
        self.vtkmesh = mesh
        self.filepath = filepath
        self.data = {}

    def addMesh(self, mesh: VTKMesh) -> None:
        """Method for adding a :class:`VTKMesh`

        Parameters
        ----------
        mesh : VTKMesh
            A fully defined :class:`VTKMesh` object to be used for file writing
        """
        self.vtkmesh = mesh

    def __setitem__(self, key: str, data: np.ndarray) -> None:
        """Method used for assigning data to be written to the file

        The most useful way to use this function is the implicitly call
        it by doing something like: "data['pressure'] = pressure_data"

        If no data is set, the file will still write with a data list
        that is set by default to be empty. No errors will occur if a
        mesh is desired with no data values.

        Parameters
        ----------
        key : str
            Label for the data being entered.  This is the label that will
            be used in the exported file for the data
        data : np.ndarray
            Data values for the cells in the mesh, such as pressure or
            temperature values of that cell
        """
        self.data[key] = self._unroll_data(data)

    def _unroll_data(self, data: np.ndarray) -> np.ndarray:
        """Private method for setting values to the appropriate cells in the mesh

        This method loops through the inputed data array as well as the meshmap and
        sets the values to the appropriate cells in the mesh.

        Parameters
        ----------
        data : np.ndarray
            Data values to be 'unrolled'

        Returns
        -------
        np.ndarray
            The 'unrolled' data values
        """
        assert self.vtkmesh is not None
        values = np.zeros(self.vtkmesh.offsets.size)
        for i in range(len(self.vtkmesh.meshmap) - 1):
            values[int(self.vtkmesh.meshmap[i]) : int(self.vtkmesh.meshmap[i + 1])] = data[i]
        return values

    def writeFile(self) -> None:
        """Primary method for writing all data as a '.vtu' file.

        This file will allow you to visualize the meshes created. It will
        also allow the user to visualize any data stored using the :meth:`__setItem__` function.
        This function makes use of the :func:`unstructuredGridToVTK` function from the
        pyevtk library.
        """

        assert self.vtkmesh is not None

        unstructuredGridToVTK(
            self.filepath,
            *self.vtkmesh.points,
            connectivity=self.vtkmesh.connections,
            offsets=self.vtkmesh.offsets,
            cell_types=self.vtkmesh.ctypes,
            cellData=self.data,
        )
