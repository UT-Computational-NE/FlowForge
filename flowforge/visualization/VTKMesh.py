from __future__ import annotations
from typing import Tuple
import numpy as np


class VTKMesh:
    """
    The VTKMesh Class provides a way to store the mesh of the entire system.
    It provides the functions necessary to add meshes together as well as
    the function to translate the shapes in the coordinate system.
    """

    def __init__(
        self,
        points: np.ndarray = None,
        conn: np.ndarray = None,
        offsets: np.ndarray = None,
        ctypes: np.ndarray = None,
        meshmap: np.ndarray = None,
    ) -> None:
        """
        Initializes an empty mesh by default or stores the values inputted.
        The initialization of this class must either be empty or contain
        all the numpy arrays of data. An error will be raised if the mesh
        contains partial data.

        Args:
            points  : (OPTIONAL) np array, list of all points in the vtk mesh
            conn    : (OPTIONAL) np array, list of the point connections in the vtk mesh
            offsets : (OPTIONAL) np array, list of the offsets for each shape in the vtk mesh
            ctypes  : (OPTIONAL) np array, list of the object type of each cell. the values in this
                      list are imported from the pyevtk library
            meshmap : (OPTIONAL) np array, list of the cells in the mesh for inserting data values
                      such as pressure, temperature
        """
        inp_none = [points is None, conn is None, offsets is None, ctypes is None, meshmap is None]
        if any(inp_none):
            assert all(inp_none)
            self._x = np.array([])
            self._y = np.array([])
            self._z = np.array([])
            self._conn = np.array([], dtype=int)
            self._offsets = np.array([], dtype=int)
            self._ctypes = np.array([])
            self._meshmap = np.array([], dtype=int)
        else:
            self._x = points[0]
            self._y = points[1]
            self._z = points[2]
            self._conn = conn
            self._offsets = offsets
            self._ctypes = ctypes
            self._meshmap = meshmap

    @property
    def points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The 'points' property is a more convenient way to return the arrays of points
        in the mesh. Using '.points' will return a tuple containing the x, y, and z coordinates.

        Args: None
        """
        return (self._x, self._y, self._z)

    @property
    def connections(self) -> np.ndarray:
        """
        Returns the connections array.
        """
        return self._conn

    @property
    def offsets(self) -> np.ndarray:
        """
        Returns the offsets array.
        """
        return self._offsets

    @property
    def ctypes(self) -> np.ndarray:
        """
        Returns the ctypes array.
        """
        return self._ctypes

    @property
    def meshmap(self) -> np.ndarray:
        """
        Returns the meshmap array.
        """
        return self._meshmap

    def __add__(self, newmesh: VTKMesh) -> VTKMesh:
        """
        The __add__ function is a function that will overload the addition operator to
        add multiple meshes together. The most practical use of this operator would be
        to initialize a mesh and then use '+=' to add meshes to the main mesh.

        Args:
            newmesh : VTKMesh, the mesh that will be added to the first mesh
        """
        # concatenates the arrays inside the meshes to produce a single mesh to return
        combinedMesh = VTKMesh()
        combinedMesh._x = np.concatenate((self._x, newmesh._x))
        combinedMesh._y = np.concatenate((self._y, newmesh._y))
        combinedMesh._z = np.concatenate((self._z, newmesh._z))

        newmesh._conn += self._x.size
        combinedMesh._conn = np.concatenate((self._conn, newmesh._conn))

        if self._offsets.size > 0:
            newmesh._offsets += int(self._offsets[-1])
        combinedMesh._offsets = np.concatenate((self._offsets, newmesh._offsets))

        combinedMesh._ctypes = np.concatenate((self._ctypes, newmesh._ctypes))

        if self._meshmap.size > 0:
            newmesh._meshmap = np.delete(newmesh._meshmap, 0)
            newmesh._meshmap += int(self._meshmap[-1])

        # np.concatenate has dtype kwarg but pylint doesn't like it
        # (https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
        # pylint:disable=unexpected-keyword-arg
        combinedMesh._meshmap = np.concatenate((self._meshmap, newmesh._meshmap), dtype=int)

        return combinedMesh

    def translate(self, x: float, y: float, z: float, theta: float = 0, alpha: float = 0) -> VTKMesh:
        """
        The translate function serves the purpose of moving the shapes within the coordinate frame.
        This function has the ability to move the shapes in the x, y, and z axis and it also has
        the ability to rotate the shapes in the polar (theta) and azimuthal (alpha) directions.
        This will allow for the shape to be in any position and orientation desired.

        Args:
            x       : float, the desired x coordinate for the inlet
            y       : float, the desired y coordinate for the inlet
            z       : float, the desired z coordinate for the inlet
            theta   : (OPTIONAL) float, the degree of rotation desired about the y axis (polar)
            alpha   : (OPTIONAL) float, the degree of rotation desired about the z axis (azimuthal)
        """
        # rotation matrices
        polar_rotate = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        azimuthal_rotate = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        points = np.array([self._x, self._y, self._z])
        # multiplies the points by 2 rotation matrices
        points = np.dot(polar_rotate, points)
        points = np.dot(azimuthal_rotate, points)
        # linear translation of the points
        self._x = points[0] + x
        self._y = points[1] + y
        self._z = points[2] + z
        return self
