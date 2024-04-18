import numpy as np
from pyevtk.vtk import VtkWedge
from flowforge.visualization import VTKMesh
from flowforge.visualization.VTKShapes import CYL_RESOLUTION


def genCyl(L: float, R: float, resolution: int = CYL_RESOLUTION, nlayers: int = 1) -> VTKMesh:
    """
    Generates mesh for a cylinder
        Args:
            L : float, Length
            R : float, Radius
            resolution : (OPTIONAL) int, number of sides the cylinder is approximated with
            nlayers    : (OPTIONAL) int, the number of vertical layers the cylinder is comprised of
    """
    nwedges = resolution
    ncells = nwedges * nlayers
    pi = np.pi
    sliceAngle = 2 * pi / nwedges  # radians
    angles = np.arange(0, 2 * pi, sliceAngle)
    # conversion from polar to cartesian coordinates
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    dz = L / nlayers
    z = np.linspace(0, L, nlayers + 1)

    # points
    npoints = (nwedges + 1) * (1 + nlayers)
    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)
    i = 0
    for k in z:
        # defines the center point of each layer of points
        xx[i] = 0
        yy[i] = 0
        zz[i] = k
        i += 1
        for j in range(nwedges):
            # assigns the values around the circle to the large points list at the layer height z = k
            xx[i] = x[j]
            yy[i] = y[j]
            zz[i] = k
            i += 1

    # connections
    conn = np.zeros(nwedges * nlayers * 6)
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

    # cell types
    ctypes = np.ones(ncells) * VtkWedge.tid

    # mesh map
    meshmap = np.zeros(nlayers + 1, dtype=int)
    for i in range(nlayers):
        meshmap[i + 1] = nwedges * (i + 1)
    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)
