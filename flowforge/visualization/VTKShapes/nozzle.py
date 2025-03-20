import numpy as np
from pyevtk.vtk import VtkWedge
from flowforge.visualization import VTKMesh
from flowforge.visualization.VTKShapes import CYL_RESOLUTION


def genNozzle(L: float, Rin: float, Rout: float, resolution: int = CYL_RESOLUTION) -> VTKMesh:
    """Function for generating a mesh for a nozzle

    Parameters
    ----------
    L : float
        Length
    Rin : float
        Inlet Radius
    Rout : float
        Outlet Radius
    resolution : int
        Number of sides the nozzle is approximated with

    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """
    nwedges = resolution
    nlayers = 1
    ncells = nwedges
    pi = np.pi
    sliceAngle = 2 * pi / nwedges  # radians
    angles = np.arange(0, 2 * pi, sliceAngle)
    xin = Rin * np.cos(angles)
    yin = Rin * np.sin(angles)
    xout = Rout * np.cos(angles)
    yout = Rout * np.sin(angles)
    z = np.array([0, L])

    # points
    npoints = (nwedges + 1) * 2
    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)
    i = 0
    for k in z:
        xx[i] = 0
        yy[i] = 0
        zz[i] = k
        i += 1
        for j in range(nwedges):
            if k == 0:
                xx[i] = xin[j]
                yy[i] = yin[j]
                zz[i] = k
            elif k == L:
                xx[i] = xout[j]
                yy[i] = yout[j]
                zz[i] = k
            i += 1

    # connections
    conn = np.zeros(nwedges * 6)
    i = 0
    for k in range(nlayers):
        for j in range(nwedges):
            j0 = j + (nwedges + 1) * k + 1
            if j + 1 == nwedges:
                j1 = (nwedges + 1) * k + 1
            else:
                j1 = j + ((nwedges + 1) * k) + 2
            conn[i + 0] = (nwedges + 1) * k
            conn[i + 1] = j0
            conn[i + 2] = j1
            conn[i + 3] = (nwedges + 1) * (k + 1)
            conn[i + 4] = j0 + nwedges + 1
            conn[i + 5] = j1 + nwedges + 1
            i += 6

    # offset data
    offsets = np.zeros(ncells)
    for i in range(ncells):
        offsets[i] = (i + 1) * 6

    # celltypes
    ctypes = np.ones(ncells) * VtkWedge.tid

    # meshmap
    meshmap = np.array([0, nwedges], dtype=int)

    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)
