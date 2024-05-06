from typing import List
import numpy as np
from pyevtk.vtk import VtkWedge
from flowforge.visualization import VTKMesh

_sq32 = np.sqrt(3.0) / 2.0


def genTriGrid(sideLength: float, layerHeights: List[float], pointsPerRow: List[int]) -> VTKMesh:
    """ Function for generating a mesh mesh grid consisting of triangles.

    Parameters
    ----------
    sideLength : float
        Length of each side of the triangles
    layerHeights : List[float]
        Height of each layer in the mesh
    pointsPerRow : List[int]
        Number of points per row in the mesh
    
    Returns
    -------
    VTKMesh
        The generated VTK Mesh
    """
    # calculations
    ppr = pointsPerRow
    dx = sideLength
    dy = sideLength * _sq32
    npoints = sum(ppr) * (len(layerHeights) + 1)

    ntriangles = 0
    for row in range(ppr.size - 1):
        if ppr[row + 1] > ppr[row]:
            ntriangles += ppr[row] * 2 - 1
        elif ppr[row + 1] < ppr[row]:
            ntriangles += ppr[row + 1] * 2 - 1
    nwedges = ntriangles * len(layerHeights)

    # point data
    z = np.append(0, layerHeights)
    for i in range(1, z.size):
        z[i] += z[i - 1]
    xx, yy, zz = np.zeros(npoints), np.zeros(npoints), np.zeros(npoints)
    point = 0
    for zcoord in z:
        for j, row in enumerate(ppr):
            xoffset = 0.5 * dx * int(row % 2 == 0)
            yoffset = 0.5 * dy * int(ppr.size % 2 == 0)
            for i in range(row):
                xcoord = (-np.floor(0.5 * row) + i) * dx + xoffset
                ycoord = (np.floor(0.5 * ppr.size) - j) * dy + yoffset
                xx[point] = xcoord
                yy[point] = ycoord
                zz[point] = zcoord
                point += 1
    points = (xx, yy, zz)

    # connection data
    conn = np.zeros(nwedges * 6, dtype=int)
    points_per_level = np.sum(ppr)
    n = 0
    for k in range(z.size - 1):
        nprev = 0
        for row in range(ppr.size - 1):
            if ppr[row + 1] > ppr[row]:
                for i in range(ppr[row]):
                    conn[n + 0] = k * points_per_level + nprev + ppr[row] + i + np.floor(0.5 * (ppr[row + 1] - ppr[row]))
                    conn[n + 1] = k * points_per_level + nprev + ppr[row] + i + np.floor(0.5 * (ppr[row + 1] - ppr[row])) + 1
                    conn[n + 2] = k * points_per_level + nprev + ppr[row] + i - ppr[row]
                    conn[n + 3] = (k + 1) * points_per_level + nprev + ppr[row] + i + np.floor(0.5 * (ppr[row + 1] - ppr[row]))
                    conn[n + 4] = (
                        (k + 1) * points_per_level + nprev + ppr[row] + i + np.floor(0.5 * (ppr[row + 1] - ppr[row])) + 1
                    )
                    conn[n + 5] = (k + 1) * points_per_level + nprev + ppr[row] + i - ppr[row]
                    n += 6
                    if i != ppr[row] - 1:
                        conn[n + 0] = k * points_per_level + nprev + i
                        conn[n + 1] = k * points_per_level + nprev + i + 1
                        conn[n + 2] = k * points_per_level + nprev + i + np.ceil(0.5 * (ppr[row + 1] - ppr[row])) + ppr[row]
                        conn[n + 3] = (k + 1) * points_per_level + nprev + i
                        conn[n + 4] = (k + 1) * points_per_level + nprev + i + 1
                        conn[n + 5] = (
                            (k + 1) * points_per_level + nprev + i + np.ceil(0.5 * (ppr[row + 1] - ppr[row])) + ppr[row]
                        )
                        n += 6
            elif ppr[row + 1] < ppr[row]:
                for i in range(ppr[row + 1]):
                    conn[n + 0] = k * points_per_level + nprev + i + np.floor((ppr[row] - ppr[row + 1]) / 2)
                    conn[n + 1] = k * points_per_level + nprev + i + np.floor((ppr[row] - ppr[row + 1]) / 2) + 1
                    conn[n + 2] = k * points_per_level + nprev + i + ppr[row]
                    conn[n + 3] = (k + 1) * points_per_level + nprev + i + np.floor((ppr[row] - ppr[row + 1]) / 2)
                    conn[n + 4] = (k + 1) * points_per_level + nprev + i + np.floor((ppr[row] - ppr[row + 1]) / 2) + 1
                    conn[n + 5] = (k + 1) * points_per_level + nprev + i + ppr[row]
                    n += 6
                    if i != ppr[row + 1] - 1:
                        conn[n + 0] = k * points_per_level + nprev + ppr[row] + i
                        conn[n + 1] = k * points_per_level + nprev + ppr[row] + i + 1
                        conn[n + 2] = (
                            k * points_per_level + nprev + ppr[row] + i + np.ceil(0.5 * (ppr[row] - ppr[row + 1]) - ppr[row])
                        )
                        conn[n + 3] = (k + 1) * points_per_level + nprev + ppr[row] + i
                        conn[n + 4] = (k + 1) * points_per_level + nprev + ppr[row] + i + 1
                        conn[n + 5] = (
                            (k + 1) * points_per_level
                            + nprev
                            + ppr[row]
                            + i
                            + np.ceil(0.5 * (ppr[row] - ppr[row + 1]) - ppr[row])
                        )
                        n += 6
            nprev += ppr[row]

    # offset data
    offsets = 6 + 6 * np.arange(nwedges, dtype=int)

    # ctype data
    ctypes = np.ones(nwedges) * VtkWedge.tid

    # meshmap data
    meshmap = np.arange(nwedges + 1, dtype=int)

    return VTKMesh(points, conn[:n], offsets, ctypes, meshmap)
