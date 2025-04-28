from __future__ import annotations
from typing import Tuple
import numpy as np
from pyevtk import vtk


class VTKMesh:
    """A class for storing a VTK mesh of a FlowForge system.

    It provides the functions necessary to add meshes together as well as
    the function to translate the shapes in the coordinate system.

    Attributes
    ----------
    points : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Collection of all points in the VTK mesh.  Points are returned
        as a Tuple of :math:`(x,y,z)` coordinate arrays, with the first element
        corresponding to all :math:`x` values, second element the :math:`y` values,
        and third element the :math:`z` values
    connections : np.ndarray
        Collection of point connections in the VTK mesh
    offsets : np.ndarray
        Collection of the offsets for each shape in the VTK mesh
    ctypes : np.ndarray
        Collection of object types of each cell.  The values are
        imported from the :mod:`pyevtk` library
    meshmap : np.ndarray
        Collection of cells in the mesh for inserting data values
        such as pressure and temperature
    """

    def __init__(
        self,
        points: np.ndarray = None,
        conn: np.ndarray = None,
        offsets: np.ndarray = None,
        ctypes: np.ndarray = None,
        meshmap: np.ndarray = None,
    ) -> None:
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
        return (self._x, self._y, self._z)

    @points.setter
    def points(self, points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._x, self._y, self._z = points

    @property
    def connections(self) -> np.ndarray:
        """
        Returns the VTK point connections.
        """
        return self._conn

    @connections.setter
    def connections(self, conn: np.ndarray) -> None:
        self._conn = conn

    @property
    def offsets(self) -> np.ndarray:
        """
        Returns the VTK offsets.
        """
        return self._offsets

    @offsets.setter
    def offsets(self, offsets: np.ndarray) -> None:
        self._offsets = offsets

    @property
    def ctypes(self) -> np.ndarray:
        """
        Returns the VTK ctypes.
        """
        return self._ctypes

    @ctypes.setter
    def ctypes(self, ctypes: np.ndarray) -> None:
        self._ctypes = ctypes

    @property
    def meshmap(self) -> np.ndarray:
        """
        Returns the VTK mesh map.
        """
        return self._meshmap

    @meshmap.setter
    def meshmap(self, meshmap: np.ndarray) -> None:
        self._meshmap = meshmap

    def __add__(self, newmesh: VTKMesh) -> VTKMesh:
        """A method for defining an addition operator to add multiple meshes together

        The most practical use of this operator would be to initialize a mesh and then
        use '+=' to add meshes to the main mesh.

        Parameters
        ----------
        newmesh : VTKMesh
            The mesh that will be added to the 'self' mesh

        Returns
        -------
        VTKMesh
            The new combined mesh of both newmesh and self
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

    def __repr__(self):
        """A method for defining the representation of the VTKMesh object.

        This method is used to print the VTKMesh object in a readable format.

        Returns
        -------
        str
            The string representation of the VTKMesh object
        """
        return (
            f"Points: ({self._x.size}, {self._y.size}, {self._z.size})\n"
            f"Connections: {self.connections}\n"
            f"Offsets: {self.offsets}\n"
            f"Ctypes: {self.ctypes}\n"
            f"Meshmap: {self.meshmap})"
        )

    def translate(self, x: float, y: float, z: float, theta: float = 0, alpha: float = 0) -> VTKMesh:
        """Method for translating the shapes within the coordinate frame.

        This function has the ability to move the shapes in the x, y, and z axis and it also has
        the ability to rotate the shapes in the polar (theta) and azimuthal (alpha) directions.
        This will allow for the shape to be in any position and orientation desired.

        Parameters
        ----------
        x : float
            The desired x coordinate for the inlet
        y : float
            The desired y coordinate for the inlet
        z : float
            The desired z coordinate for the inlet
        theta : float
            The degree of rotation desired about the y axis (polar)
        alpha : float
            The degree of rotation desired about the z axis (azimuthal)
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

    def combine_cells(self, cell_idx1: int, cell_idx2: int) -> VTKMesh:
        """Method for combining two cells into one cell

        This function will combine two cells into one cell and will update the connections,
        offsets, ctypes, and meshmap arrays to reflect the new cell.

        Parameters
        ----------
        cell_idx1 : int
            The index of the first cell to combine
        cell_idx2 : int
            The index of the second cell to combine

        Returns
        -------
        VTKMesh
            The new mesh with the combined cells
        """
        if cell_idx1 > cell_idx2:
            cell_idx1, cell_idx2 = cell_idx2, cell_idx1

        if self.ctypes[cell_idx1] == vtk.VtkHexahedron.tid and self.ctypes[cell_idx2] == vtk.VtkHexahedron.tid:
            return self._combine_hexahedrons(cell_idx1, cell_idx2)
        if self.ctypes[cell_idx1] == vtk.VtkHexahedron.tid and self.ctypes[cell_idx2] == vtk.VtkWedge.tid:
            raise TypeError("Support for combining a VtkHexahedron and a VtkWedge is not yet implemented.")
        if self.ctypes[cell_idx1] == vtk.VtkWedge.tid and self.ctypes[cell_idx2] == vtk.VtkHexahedron.tid:
            raise TypeError("Support for combining a VtkHexahedron and a VtkWedge is not yet implemented.")
        if self.ctypes[cell_idx1] == vtk.VtkWedge.tid and self.ctypes[cell_idx2] == vtk.VtkWedge.tid:
            raise TypeError("Support for combining two VtkWedges is not yet implemented.")

        raise TypeError(
            f"No function exists to merge the two cell types: {self.ctypes[cell_idx1]} and {self.ctypes[cell_idx2]}"
        )

    def _combine_hexahedrons(self, hexahedron_idx1: int, hexahedron_idx2: int) -> VTKMesh:
        """Method for combining two hexahedrons into one quadratic hexahedron

        This function will combine two hexahedrons into one quadratic hexahedron and will update
        the connections, offsets, ctypes, and meshmap arrays to reflect the new hexahedron.

        Parameters
        ----------
        hexahedron_idx1 : int
            The index of the first hexahedron to combine
        hexahedron_idx2 : int
            The index of the second hexahedron to combine

        Returns
        -------
        VTKMesh
            The new mesh with the combined hexahedron
        """
        x, y, z = self.points
        conn = self.connections.astype(int)
        offsets = np.insert(self.offsets.astype(int), 0, 0)  # inserts 0 at the beginning of the array
        ctypes = self.ctypes.astype(int)
        meshmap = self.meshmap.astype(int)

        cell1_conn = conn[offsets[hexahedron_idx1] : offsets[hexahedron_idx1 + 1]]
        cell2_conn = conn[offsets[hexahedron_idx2] : offsets[hexahedron_idx2 + 1]]

        shared_pts_idx = []
        shared_pts = []
        for c1 in cell1_conn:
            if c1 in cell2_conn:
                shared_pts_idx.append(np.where(c1 == cell2_conn)[0][0])
                shared_pts.append(cell2_conn[np.where(c1 == cell2_conn)[0][0]])

        shared_pts_idx.sort()

        last_pt = max(conn) + 1
        if shared_pts_idx == [0, 1, 4, 5]:  # merge along the x-axis
            new_conn = np.concatenate(
                (
                    [
                        cell1_conn[[0, 1, 5, 4]],
                        cell2_conn[[3, 2, 6, 7]],
                        np.arange(last_pt, last_pt + 8),
                        cell2_conn[[0, 1, 5, 4]],
                    ]
                )
            )
        elif shared_pts_idx == [0, 1, 2, 3]:  # merge along the y-axis
            new_conn = np.concatenate(
                (
                    [
                        cell1_conn[[0, 1, 2, 3]],
                        cell2_conn[[4, 5, 6, 7]],
                        np.arange(last_pt, last_pt + 8),
                        cell2_conn[[0, 1, 2, 3]],
                    ]
                )
            )
        elif shared_pts_idx == [0, 3, 4, 7]:  # merge along the z-axis
            new_conn = np.concatenate(
                (
                    [
                        cell1_conn[[0, 3, 7, 4]],
                        cell2_conn[[1, 2, 6, 5]],
                        np.arange(last_pt, last_pt + 8),
                        cell2_conn[[0, 3, 7, 4]],
                    ]
                )
            )
        else:
            raise NotImplementedError("Merging between the selected faces is not supported yet.")

        new_x = np.zeros((8), dtype=float)
        new_y = np.zeros((8), dtype=float)
        new_z = np.zeros((8), dtype=float)

        for i in range(8):
            if (i + 1) % 4 == 0:
                pt1, pt2 = new_conn[i], new_conn[i - 3]
            else:
                pt1, pt2 = new_conn[i], new_conn[i + 1]
            new_x[i] = np.average([x[pt1], x[pt2]])
            new_y[i] = np.average([y[pt1], y[pt2]])
            new_z[i] = np.average([z[pt1], z[pt2]])

        x = np.append(x, new_x)
        y = np.append(y, new_y)
        z = np.append(z, new_z)
        self.points = (x, y, z)

        self.connections = np.concatenate(
            (
                conn[: offsets[hexahedron_idx1]],
                new_conn,
                conn[offsets[hexahedron_idx1 + 1] : offsets[hexahedron_idx2]],
                conn[offsets[hexahedron_idx2 + 1] :],
            )
        )

        offsets = np.delete(offsets, 0)
        offsets[hexahedron_idx1:] += 12  # 12 additional pts to the quadratic hexahedron
        offsets[hexahedron_idx2:] -= 8  # 8 pts removed from the merged hexahedron
        self.offsets = np.delete(offsets, hexahedron_idx2)  # remove the second hexahedron offset

        ctypes[hexahedron_idx1] = vtk.VtkQuadraticHexahedron.tid
        self.ctypes = np.delete(ctypes, hexahedron_idx2)

        meshmap[hexahedron_idx2:] -= 1
        self.meshmap = np.delete(meshmap, hexahedron_idx2)

        return self
