import numpy as np
from pyevtk.vtk import VtkHexahedron
from flowforge.visualization import VTKMesh
from flowforge.visualization.VTKShapes import CYL_RESOLUTION


def genAnnulus(
    L: float, Rin: float, Rout: float, resolution: int = CYL_RESOLUTION, nazimuthal: int = 1, nlayers: int = 1
) -> VTKMesh:
    """
    Generates mesh for an annulus (a hollow cylinder)
        Args:
            L    : float, Length
            Rin  : float, Inner Radius
            Rout : float, Outer Radius
            resolution : int, (OPTIONAL) number of sides the annulus is approximated with
            nazimuthal : int, (OPTIONAL) number of sections to split in the azimuthal direction for stored data.
            nlayers    : int, (OPTIONAL) numbers of layers in the vertical direction
    """
    resolution = int(np.ceil(resolution / nazimuthal) * nazimuthal)
    ncells = resolution * nlayers
    npoints = 2 * resolution * (nlayers + 1)
    sliceAngle = 2 * np.pi / resolution  # radians
    angles = np.arange(0, 2 * np.pi, sliceAngle)
    xin = Rin * np.cos(angles)
    yin = Rin * np.sin(angles)
    xout = Rout * np.cos(angles)
    yout = Rout * np.sin(angles)
    dz = L / nlayers
    z = np.arange(0, L + dz, dz)

    # points
    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)
    i = 0
    for k in z:
        for j in range(resolution):
            xx[i] = xin[j]
            yy[i] = yin[j]
            zz[i] = k
            xx[i + 1] = xout[j]
            yy[i + 1] = yout[j]
            zz[i + 1] = k
            i += 2

    # connections
    conn = np.zeros(ncells * 8)
    i = 0
    for k in range(nlayers):
        for j in range(resolution):
            j0 = j * 2 + (k * resolution * 2)
            if j + 1 == resolution:
                j1 = k * resolution * 2
            else:
                j1 = j * 2 + (k * resolution * 2) + 2
            conn[i + 0] = j0
            conn[i + 1] = j0 + 1
            conn[i + 2] = j1 + 1
            conn[i + 3] = j1
            conn[i + 4] = j0 + resolution * 2
            conn[i + 5] = j0 + resolution * 2 + 1
            conn[i + 6] = j1 + resolution * 2 + 1
            conn[i + 7] = j1 + resolution * 2
            i += 8

    # offsets
    offsets = np.zeros(ncells)
    for i in range(ncells):
        offsets[i] = (i + 1) * 8

    # ctypes
    ctypes = np.ones(ncells) * VtkHexahedron.tid

    # meshmap
    meshmap = np.zeros(nlayers * nazimuthal + 1, dtype=int)
    for i in range(1, meshmap.size):
        meshmap[i] = resolution / nazimuthal * i

    points = (xx, yy, zz)

    return VTKMesh(points, conn, offsets, ctypes, meshmap)
