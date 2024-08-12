import numpy as np
from pyevtk.vtk import VtkHexahedron
from flowforge.visualization import VTKMesh
from flowforge.visualization.VTKShapes import CYL_RESOLUTION


def genUniformAnnulus(
    L: float, Rin: float, Rout: float, naxial_layers: int = 1, nradial_layers: int = 1, resolution: int = CYL_RESOLUTION
) -> VTKMesh:
    """Generates the vtk mesh for an annulus

    Parameters
    ----------
    L : float
        length of the annulus
    Rin : float
        inner radius
    Rout : float
        outer radius
    naxial_layers : int, optional
        number of axial layers to split the mesh into
    nradial_layers : int, optional
        number of radial layers to split the mesh into
    resolution : int, optional
        number of azimuthal angles to split the mesh into

    Returns
    -------
    VTKMesh
        object containing the vtk mesh data for an annulus
    """
    mesh_r = np.linspace(Rin, Rout, nradial_layers + 1)
    mesh_z = np.linspace(0, L, naxial_layers + 1)
    mesh_theta = np.linspace(0, 2 * np.pi, resolution + 1)

    return _genAnnulus(mesh_r, mesh_z, mesh_theta)


def genNonUniformAnnulus(mesh_r: np.ndarray, mesh_z: np.ndarray, mesh_theta: np.ndarray) -> VTKMesh:
    """Generates the vtk mesh for an annulus

    Parameters
    ----------
    mesh_r : ndarray of float
        Contains the radial meshing points for cell division. Note that the array must begin with 0.
    mesh_z : ndarray of float
        Contains the axial meshing points for cell division. Note that the array must begin with 0.
    mesh_theta : ndarray of float
        Contains the angular divisions of the cell. Note that the array must begin and end with 0 and 2*pi,
        and that the array must be at least 4 values in length.

    Returns
    -------
    VTKMesh
        object containing the vtk mesh data for an annulus
    """
    if mesh_r[0] == 0:
        mesh_r = np.delete(mesh_r, 0)
    return _genAnnulus(mesh_r, mesh_z, mesh_theta)


def _genAnnulus(mesh_r: np.ndarray, mesh_z: np.ndarray, mesh_theta: np.ndarray, **kwargs) -> VTKMesh:
    """Generates the vtk mesh for an annulus

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
        object containing the vtk mesh data for an annulus
    """

    nazimuthal_data = kwargs.get("nazimuthal_data", 1)

    # pre-calculations
    naxial_layers = mesh_z.size - 1
    nradial_layers = mesh_r.size - 1
    nazimuthal_layers = mesh_theta.size - 1
    ncell = naxial_layers * nradial_layers * nazimuthal_layers

    # points
    npoints = (nradial_layers + 1) * nazimuthal_layers * (naxial_layers + 1)
    npoints_layer = int(npoints / mesh_z.size)
    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)

    point = 0
    for z in mesh_z:
        for r in mesh_r:
            for i in range(nazimuthal_layers):
                xx[point] = r * np.cos(mesh_theta[i])
                yy[point] = r * np.sin(mesh_theta[i])
                zz[point] = z
                point += 1

    # connections
    conn = np.zeros(ncell * 8, dtype=int)
    i = 0
    for k in range(naxial_layers):
        for r in range(nradial_layers):
            for j in range(nazimuthal_layers):
                j0 = r * nazimuthal_layers + j + k * npoints_layer
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

    # offsets
    offsets = np.zeros(ncell, dtype=int)
    for i in range(ncell):
        offsets[i] = (i + 1) * 8

    # ctypes
    ctypes = np.ones(ncell, dtype=int) * VtkHexahedron.tid

    # meshmap
    meshmap = np.arange(0, ncell + 1, dtype=int)
    meshmap = np.zeros(naxial_layers * nazimuthal_data + 1, dtype=int)
    for i in range(meshmap.size):
        meshmap[i] = nazimuthal_layers * nradial_layers / nazimuthal_data * i

    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)
