import numpy as np
from pyevtk.vtk import VtkHexahedron
from flowforge.visualization import VTKMesh

def genUniformSquare(L: float, W: float, nx: int, ny: int) -> VTKMesh:
    """Function for generating a mesh for a square with uniform cell division

    Parameters
    ----------
    L : float
        Length
    W : float
        Width
    nx : int
        Number of segments in :math:`x`
    ny : int
        Number of segments in :math:`y`

    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """

    dx = L/nx
    dy = W/ny

    mesh_x = np.arange(-L / 2, L / 2 + dx, dx)
    mesh_y = np.arange(-W / 2, W / 2 + dy, dy)

    return _genSquare(mesh_x, mesh_y)

def genNonuniformSquare(mesh_x: np.ndarray, mesh_y: np.ndarray) -> VTKMesh:
    """Function for generating a mesh for a cube with non-uniform cell division

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
    return _genSquare(mesh_x, mesh_y)

def _genSquare(mesh_x: np.ndarray, mesh_y: np.ndarray) -> VTKMesh:
    """Private method for generating the VTK mesh for a cube

    Parameters
    ----------
    mesh_x : np.ndarray
        The :math:`x-values` for the non-uniform mesh
    mesh_y : np.ndarray
        The :math:`y-values` for the non-uniform mesh

    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """
    nx = len(mesh_x) - 1
    ny = len(mesh_y) - 1
    ncells = nx * ny

    # points
    mesh_z = np.array([0])
    xx, yy, zz = np.meshgrid(mesh_x, mesh_y, mesh_z)
    xx = np.reshape(xx, xx.size)
    yy = np.reshape(yy, yy.size)
    zz = np.reshape(zz, zz.size)

    # connection data
    conn = np.zeros(ncells * 4)
    conn_i = 0
    for j in range(ny):
        for i in range(nx):
            conn[conn_i + 0] = j + i* (ny + 1)
            conn[conn_i + 1] = j + i* (ny + 1) + 1
            conn[conn_i + 2] = j + i* (ny + 1) + (ny + 1)
            conn[conn_i + 3] = j + i* (ny + 1) + (ny + 1) + 1
            conn_i += 4

    # offset data
    offsets = np.zeros(ncells)
    for i in range(offsets.size):
        offsets[i] = (i + 1) * 4

    # cell types
    ctypes = np.ones(ncells) * VtkHexahedron.tid

    # mesh map
    meshmap = np.arange(0, ncells + 1, dtype=int)
    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)