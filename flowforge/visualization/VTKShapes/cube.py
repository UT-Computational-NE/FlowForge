import numpy as np
from pyevtk.vtk import VtkHexahedron
from flowforge.visualization import VTKMesh

def genUniformCube(L, W, H, nx=1, ny=1, nz=1):
    """
    Generates the vtk mesh for a cube with uniform cell division
        Args:
            L  : float, Length
            W  : float, Width
            H  : float, Height
            nx : (OPTIONAL) int, Number of segments in x
            ny : (OPTIONAL) int, Number of segments in y
            nz : (OPTIONAL) int, Number of segments in z
    """
    dx = L / nx
    dy = W / ny
    dz = H / nz

    # points
    mesh_x = np.arange(-L/2, L/2 + dx, dx)
    mesh_y = np.arange(-W/2, W/2 + dy, dy)
    mesh_z = np.arange(   0,   H + dz, dz)

    return _genCube(mesh_x, mesh_y, mesh_z)


def genNonuniformCube(mesh_x, mesh_y, mesh_z):
    """
    Generates the vtk mesh for a cube with non-uniform cell division
        Args:
            mesh_x : float list, contains the x values for the non-uniform mesh
            mesh_y : float list, contains the y values for the non-uniform mesh
            mesh_z : float list, contains the z values for the non-uniform mesh
    """
    return _genCube(mesh_x, mesh_y, mesh_z)


def _genCube(mesh_x, mesh_y, mesh_z):
    """
    Generates the vtk mesh for a cube
        Args:
            mesh_x : float list, contains the x values for the non-uniform mesh
            mesh_y : float list, contains the y values for the non-uniform mesh
            mesh_z : float list, contains the z values for the non-uniform mesh
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
                conn[a+0] = k + j*(nx+1)*(nz+1) + i*(nz+1)
                conn[a+1] = k + j*(nx+1)*(nz+1) + i*(nz+1) + 1
                conn[a+2] = k + j*(nx+1)*(nz+1) + i*(nz+1) + (nz+1) + 1
                conn[a+3] = k + j*(nx+1)*(nz+1) + i*(nz+1) + (nz+1)
                conn[a+4] = k + j*(nx+1)*(nz+1) + i*(nz+1) + (nx+1)*(nz+1)
                conn[a+5] = k + j*(nx+1)*(nz+1) + i*(nz+1) + (nx+1)*(nz+1) + 1
                conn[a+6] = k + j*(nx+1)*(nz+1) + i*(nz+1) + (nx+1)*(nz+1) + (nz+1) + 1
                conn[a+7] = k + j*(nx+1)*(nz+1) + i*(nz+1) + (nx+1)*(nz+1) + (nz+1)
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
