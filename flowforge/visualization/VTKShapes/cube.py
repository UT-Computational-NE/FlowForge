import numpy as np
from pyevtk.vtk import VtkHexahedron
from flowforge.visualization import VTKMesh


def genUniformCube(L: float, W: float, H: float, nx: int = 1, ny: int = 1, nz: int = 1) -> VTKMesh:
    """ Function for generating a mesh for a cube with uniform cell division

    Parameters
    ----------
    L : float
        Length
    W : float
        Width
    H : float
        Height
    nx : int
        Number of segments in :math:`x`
    ny : int
        Number of segments in :math:`y`
    nz : int
        Number of segments in :math:`z`

    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """
    dx = L / nx
    dy = W / ny
    dz = H / nz

    # points
    mesh_x = np.arange(-L / 2, L / 2 + dx, dx)
    mesh_y = np.arange(-W / 2, W / 2 + dy, dy)
    mesh_z = np.arange(0, H + dz, dz)

    return _genCube(mesh_x, mesh_y, mesh_z)


def genNonuniformCube(mesh_x: np.ndarray, mesh_y: np.ndarray, mesh_z: np.ndarray) -> VTKMesh:
    """ Function for generating a mesh for a cube with non-uniform cell division

    Parameters
    ----------
    mesh_x : np.ndarray
        The :math:`x-values` for the non-uniform mesh
    mesh_y : np.ndarray
        The :math:`y-values` for the non-uniform mesh
    mesh_z : np.ndarray
        The :math:`z-values` for the non-uniform mesh

    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """
    return _genCube(mesh_x, mesh_y, mesh_z)


def _genCube(mesh_x: np.ndarray, mesh_y: np.ndarray, mesh_z: np.ndarray) -> VTKMesh:
    """ Private method for generating the VTK mesh for a cube

    Parameters
    ----------
    mesh_x : np.ndarray
        The :math:`x-values` for the non-uniform mesh
    mesh_y : np.ndarray
        The :math:`y-values` for the non-uniform mesh
    mesh_z : np.ndarray
        The :math:`z-values` for the non-uniform mesh

    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """
    nx = len(mesh_x) - 1
    ny = len(mesh_y) - 1
    nz = len(mesh_z) - 1
    ncells = nx * ny * nz

    # points
    xx, yy, zz = np.meshgrid(mesh_x, mesh_y, mesh_z)
    xx = np.reshape(xx, xx.size)
    yy = np.reshape(yy, yy.size)
    zz = np.reshape(zz, zz.size)

    # connection data
    conn = np.zeros(ncells * 8)
    a = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn[a + 0] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1)
                conn[a + 1] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + 1
                conn[a + 2] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + (nz + 1) + 1
                conn[a + 3] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + (nz + 1)
                conn[a + 4] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + (nx + 1) * (nz + 1)
                conn[a + 5] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + (nx + 1) * (nz + 1) + 1
                conn[a + 6] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + (nx + 1) * (nz + 1) + (nz + 1) + 1
                conn[a + 7] = k + j * (nx + 1) * (nz + 1) + i * (nz + 1) + (nx + 1) * (nz + 1) + (nz + 1)
                a += 8

    # offset data
    offsets = np.zeros(ncells)
    for i in range(offsets.size):
        offsets[i] = (i + 1) * 8

    # cell types
    ctypes = np.ones(ncells) * VtkHexahedron.tid

    # mesh map
    meshmap = np.arange(0, ncells + 1, dtype=int)
    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)
