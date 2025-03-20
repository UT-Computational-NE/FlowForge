import numpy as np
from pyevtk.vtk import VtkWedge, VtkHexahedron
from typing import Dict, Any
from flowforge.visualization import VTKMesh
from flowforge.visualization.VTKShapes import CYL_RESOLUTION


def genUniformCylinder(
    L: float, R: float, naxial_layers: int = 1, nradial_layers: int = 1, resolution: int = CYL_RESOLUTION, **kwargs
) -> VTKMesh:
    """Generates the vtk mesh for a cylinder with uniform cell divisions.

    Parameters
    ----------
    L : float
        Length of the cylinder.
    R : float
        Radius of the cylinder.
    naxial_layers : int, optional
        The number of axial layers the cylinder is comprised of. Default is None.
    nradial_layers : int, optional
        The number of radial layers the cylinder is comprised of. Default is None.
    resolution : int, optional
        The number of sides the cylinder is approximated with. Default is None.
    nazimuthal_data : ndarray of float, optional
        Number of azimuthal divisions for the solution data for each axial layer. Default is 1 (whole layer
        corresponds to 1 data value).

    Returns
    -------
    VTKMesh
        Object containing the vtk mesh data for a uniform cylinder.
    """

    mesh_r = np.linspace(0, R, nradial_layers + 1)
    mesh_z = np.linspace(0, L, naxial_layers + 1)
    mesh_theta = np.linspace(0, 2 * np.pi, resolution + 1)

    return _genCylinder(mesh_r, mesh_z, mesh_theta, **kwargs)


def genNonUniformCylinder(mesh_r: np.ndarray, mesh_z: np.ndarray, mesh_theta: np.ndarray, **kwargs) -> VTKMesh:
    """Generates the vtk mesh for a cylinder with non-uniform cell divisions.

    Parameters
    ----------
    mesh_r : ndarray of float
        Contains the radial meshing points for cell division. Note that the array must begin with 0.
    mesh_z : ndarray of float
        Contains the axial meshing points for cell division. Note that the array must begin with 0.
    mesh_theta : ndarray of float
        Contains the angular divisions of the cell. Note that the array must begin and end with 0 and 2*pi,
        and that the array must be at least 4 values in length.
    nazimuthal_data : ndarray of float, optional
        Number of azimuthal divisions for the solution data for each axial layer. Default is 1 (whole layer
        corresponds to 1 data value).

    Returns
    -------
    VTKMesh
        Object containing the vtk mesh data for a non-uniform cylinder.
    """

    assert mesh_r[0] == 0.0
    assert mesh_theta[0] == 0.0
    assert mesh_theta[-1] == 2 * np.pi
    assert mesh_theta.size >= 4
    return _genCylinder(mesh_r, mesh_z, mesh_theta, **kwargs)


def _genCylinder(mesh_r: np.ndarray, mesh_z: np.ndarray, mesh_theta: np.ndarray, **kwargs) -> VTKMesh:
    """Generates the vtk mesh for a cylinder.

    Parameters
    ----------
    mesh_r : ndarray of float
        Contains the radial meshing points for cell division. Note that the array must begin with 0.
    mesh_z : ndarray of float
        Contains the axial meshing points for cell division. Note that the array must begin with 0.
    mesh_theta : ndarray of float
        Contains the angular divisions of the cell. Note that the array must begin and end with 0 and 2*pi,
        and that the array must be at least 4 values in length.
    nazimuthal_data : ndarray of float, optional
        Number of azimuthal divisions for the solution data for each axial layer. Default is 1 (whole layer
        corresponds to 1 data value).

    Returns
    -------
    VTKMesh
        Object containing the vtk mesh data for a cylinder.
    """

    nazimuthal_data = kwargs.get("nazimuthal_data", 1)

    # pre-calculations
    naxial_layers = mesh_z.size - 1
    nradial_layers = mesh_r.size - 1
    nazimuthal_layers = mesh_theta.size - 1
    ncell = naxial_layers * nradial_layers * nazimuthal_layers

    # points
    npoints = (nradial_layers * nazimuthal_layers + 1) * (naxial_layers + 1)
    npoints_layer = int(npoints / mesh_z.size)
    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)

    point = 0
    for k in mesh_z:
        # defines the center point of each layer of points
        xx[point] = 0
        yy[point] = 0
        zz[point] = k
        point += 1
        for r in range(nradial_layers):
            for i in range(nazimuthal_layers):
                xx[point] = (mesh_r[r + 1]) * np.cos(mesh_theta[i])
                yy[point] = (mesh_r[r + 1]) * np.sin(mesh_theta[i])
                zz[point] = k
                point += 1

    # connections
    conn = np.zeros(naxial_layers * nazimuthal_layers * (6 + 8 * (nradial_layers - 1)), dtype=int)
    i = 0
    for k in range(naxial_layers):
        for r in range(nradial_layers):
            if r == 0:
                # inner circle
                for j in range(nazimuthal_layers):
                    j0 = j + 1 + k * npoints_layer
                    if j + 1 == nazimuthal_layers:
                        j1 = j0 - nazimuthal_layers + 1
                    else:
                        j1 = j0 + 1
                    conn[i + 0] = k * npoints_layer
                    conn[i + 1] = j0
                    conn[i + 2] = j1
                    conn[i + 3] = (k + 1) * npoints_layer
                    conn[i + 4] = j0 + npoints_layer
                    conn[i + 5] = j1 + npoints_layer
                    i += 6
            else:
                # outer rings
                for j in range(nazimuthal_layers):
                    j0 = (r - 1) * nazimuthal_layers + j + 1 + k * npoints_layer
                    if j + 1 == nazimuthal_layers:
                        j1 = j0 - nazimuthal_layers + 1
                    else:
                        j1 = j0 + 1
                    conn[i + 0] = j0
                    conn[i + 1] = j1
                    conn[i + 2] = j1 + nazimuthal_layers
                    conn[i + 3] = j0 + nazimuthal_layers
                    conn[i + 4] = j0 + npoints_layer
                    conn[i + 5] = j1 + npoints_layer
                    conn[i + 6] = j1 + nazimuthal_layers + npoints_layer
                    conn[i + 7] = j0 + nazimuthal_layers + npoints_layer
                    i += 8

    # offsets and cell types
    offsets = np.zeros(ncell, dtype=int)
    ctypes = np.ones(ncell, dtype=int)

    n = 0
    for k in range(naxial_layers):
        layer_start = k * nazimuthal_layers * nradial_layers
        layer_end = layer_start + nazimuthal_layers * nradial_layers
        for i in range(layer_start, layer_end):
            if i < layer_start + nazimuthal_layers:
                n += 6
                offsets[i] = n
                ctypes[i] *= VtkWedge.tid
            else:
                n += 8
                offsets[i] = n
                ctypes[i] *= VtkHexahedron.tid

    # meshmap
    meshmap = np.arange(0, ncell + 1, dtype=int)
    meshmap = np.zeros(naxial_layers * nazimuthal_data + 1, dtype=int)
    for i in range(meshmap.size):
        meshmap[i] = nazimuthal_layers * nradial_layers / nazimuthal_data * i

    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)
