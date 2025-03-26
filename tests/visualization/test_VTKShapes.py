import numpy as np
from flowforge.visualization import genUniformCube, genUniformCylinder, genNozzle, genUniformAnnulus


def test_Cube1():
    cube = genUniformCube(10, 6, 4)
    assert cube._x.size == 8
    assert cube._x[0] == -5
    assert cube._y.size == 8
    assert cube._y[0] == -3
    assert cube._z.size == 8
    assert cube._z[0] == 0
    assert cube._conn.size == 8
    assert cube._conn[0] == 0
    assert cube._offsets.size == 1
    assert cube._offsets[0] == 8
    assert cube._ctypes.size == 1
    assert cube._ctypes[0] == 12.0
    assert cube._meshmap.size == 2
    assert cube._meshmap[0] == 0
    assert cube._meshmap[1] == 1


def test_Cube2():
    cube = genUniformCube(2, 2, 2, 2, 2, 2)
    assert cube._x.size == 27
    assert cube._x[0] == -1
    assert cube._x[-1] == 1
    assert cube._y.size == 27
    assert cube._y[0] == -1
    assert cube._y[-1] == 1
    assert cube._z.size == 27
    assert cube._z[0] == 0
    assert cube._z[-1] == 2
    assert cube._conn.size == 64
    assert cube._conn[0] == 0
    assert cube._offsets.size == 8
    assert cube._offsets[0] == 8
    assert cube._ctypes.size == 8
    for ctype in cube._ctypes:
        assert ctype == 12.0
    assert cube._meshmap.size == 9
    for i in range(cube._meshmap.size):
        assert cube._meshmap[i] == i


def test_Cyl1():
    L, R = 10, 1
    cyl = genUniformCylinder(L, R)
    assert cyl._x.size == 18
    assert cyl._x[0] == 0
    assert cyl._y.size == 18
    assert cyl._y[0] == 0
    assert cyl._z.size == 18
    assert cyl._z[0] == 0
    assert cyl._conn.size == 48
    assert cyl._conn[0] == 0
    assert cyl._offsets.size == 8
    assert cyl._offsets[0] == 6
    assert cyl._ctypes.size == 8
    for ctype in cyl._ctypes:
        assert ctype == 13.0
    assert cyl._meshmap.size == 2
    assert cyl._meshmap[0] == 0
    assert cyl._meshmap[1] == 8
    for i in range(cyl._x.size):
        assert np.isclose(cyl._x[i] ** 2 + cyl._y[i] ** 2, R**2) or cyl._x[i] == cyl._y[i] == 0
        assert cyl._z[i] == 0 or cyl._z[i] == L


def test_Cyl2():
    (
        L,
        R,
    ) = (
        9,
        1,
    )
    cyl = genUniformCylinder(L, R, resolution=12, naxial_layers=3)
    assert cyl._x.size == 52
    assert cyl._x[0] == 0
    assert cyl._y.size == 52
    assert cyl._y[0] == 0
    assert cyl._z.size == 52
    assert cyl._z[0] == 0
    assert cyl._conn.size == 216
    assert cyl._conn[0] == 0
    assert cyl._offsets.size == 36
    assert cyl._offsets[0] == 6
    assert cyl._ctypes.size == 36
    for ctype in cyl._ctypes:
        assert ctype == 13.0
    assert cyl._meshmap.size == 4
    assert cyl._meshmap[0] == 0
    assert cyl._meshmap[1] == 12
    assert cyl._meshmap[2] == 24
    assert cyl._meshmap[3] == 36
    for i in range(cyl._x.size):
        assert round(cyl._x[i] ** 2 + cyl._y[i] ** 2) == R**2 or cyl._x[i] == cyl._y[i] == 0
        assert cyl._z[i] == 0 or cyl._z[i] == 3 or cyl._z[i] == 6 or cyl._z[i] == L


def test_Cyl3():
    (
        L,
        R,
    ) = (
        9,
        1,
    )
    cyl = genUniformCylinder(L, R, resolution=12, naxial_layers=3, nradial_layers=2)
    assert cyl._x.size == 100
    assert cyl._x[0] == 0
    assert cyl._y.size == 100
    assert cyl._y[0] == 0
    assert cyl._z.size == 100
    assert cyl._z[0] == 0
    assert cyl._conn.size == 504
    assert cyl._conn[0] == 0
    assert cyl._offsets.size == 72
    assert cyl._offsets[0] == 6
    assert cyl._ctypes.size == 72
    assert cyl._meshmap.size == 4
    assert cyl._meshmap[0] == 0
    assert cyl._meshmap[1] == 24
    assert cyl._meshmap[2] == 48
    assert cyl._meshmap[3] == 72


def test_Nozzle1():
    L, Rin, Rout = 10, 5, 2
    nozzle = genNozzle(L, Rin, Rout, resolution=12)
    assert nozzle._x.size == 26
    assert nozzle._x[0] == 0
    assert nozzle._y.size == 26
    assert nozzle._y[0] == 0
    assert nozzle._z.size == 26
    assert nozzle._z[0] == 0
    assert nozzle._conn.size == 72
    assert nozzle._conn[0] == 0
    assert nozzle._offsets.size == 12
    assert nozzle._offsets[0] == 6
    assert nozzle._ctypes.size == 12
    for ctype in nozzle._ctypes:
        assert ctype == 13.0
    assert nozzle._meshmap.size == 2
    assert nozzle._meshmap[0] == 0
    for i in range(nozzle._x.size):
        assert (
            round(nozzle._x[i] ** 2 + nozzle._y[i] ** 2) == Rin**2
            or round(nozzle._x[i] ** 2 + nozzle._y[i] ** 2) == Rout**2
            or nozzle._x[i] == nozzle._y[i] == 0
        )
        assert nozzle._z[i] == 0 or nozzle._z[i] == L


def test_Nozzle2():
    L, Rin, Rout = 10, 5, 2
    nozzle = genNozzle(L, Rin, Rout, resolution=20)
    assert nozzle._x.size == 42
    assert nozzle._x[0] == 0
    assert nozzle._y.size == 42
    assert nozzle._y[0] == 0
    assert nozzle._z.size == 42
    assert nozzle._z[0] == 0
    assert nozzle._conn.size == 120
    assert nozzle._conn[0] == 0
    assert nozzle._offsets.size == 20
    assert nozzle._offsets[0] == 6
    assert nozzle._ctypes.size == 20
    for ctype in nozzle._ctypes:
        assert ctype == 13.0
    assert nozzle._meshmap.size == 2
    assert nozzle._meshmap[0] == 0
    for i in range(nozzle._x.size):
        assert (
            round(nozzle._x[i] ** 2 + nozzle._y[i] ** 2) == Rin**2
            or round(nozzle._x[i] ** 2 + nozzle._y[i] ** 2) == Rout**2
            or nozzle._x[i] == nozzle._y[i] == 0
        )
        assert nozzle._z[i] == 0 or nozzle._z[i] == L


def test_Annulus1():
    L, Rin, Rout = 10, 9, 10
    annulus = genUniformAnnulus(L, Rin, Rout, resolution=12)
    assert annulus._x.size == 48
    assert annulus._x[0] == 9
    assert annulus._y.size == 48
    assert annulus._y[0] == 0
    assert annulus._z.size == 48
    assert annulus._z[0] == 0
    assert annulus._conn.size == 96
    assert annulus._conn[0] == 0
    assert annulus._offsets.size == 12
    assert annulus._offsets[0] == 8
    assert annulus._ctypes.size == 12
    assert annulus._ctypes[0] == 12.0
    assert annulus._meshmap.size == 2
    assert annulus._meshmap[0] == 0
    assert annulus._meshmap[1] == 12
    for i in range(annulus._x.size):
        assert (
            round(annulus._x[i] ** 2 + annulus._y[i] ** 2) == Rin**2
            or round(annulus._x[i] ** 2 + annulus._y[i] ** 2) == Rout**2
        )


def test_Annulus2():
    L, Rin, Rout = 10, 9, 10
    annulus = genUniformAnnulus(L, Rin, Rout, naxial_layers=2, resolution=12)
    assert annulus._x.size == 72
    assert annulus._x[0] == 9
    assert annulus._y.size == 72
    assert annulus._y[0] == 0
    assert annulus._z.size == 72
    assert annulus._z[0] == 0
    assert annulus._conn.size == 192
    assert annulus._conn[0] == 0
    assert annulus._offsets.size == 24
    assert annulus._offsets[0] == 8
    assert annulus._ctypes.size == 24
    assert annulus._ctypes[0] == 12.0
    assert annulus._meshmap.size == 3
    assert annulus._meshmap[0] == 0
    assert annulus._meshmap[1] == 12
    assert annulus._meshmap[2] == 24
    for i in range(annulus._x.size):
        assert (
            round(annulus._x[i] ** 2 + annulus._y[i] ** 2) == Rin**2
            or round(annulus._x[i] ** 2 + annulus._y[i] ** 2) == Rout**2
        )


def test_Annulus3():
    L, Rin, Rout = 10, 9, 10
    annulus = genUniformAnnulus(L, Rin, Rout, naxial_layers=2, nradial_layers=2, resolution=12)
    assert annulus._x.size == 108
    assert annulus._x[0] == 9
    assert annulus._y.size == 108
    assert annulus._y[0] == 0
    assert annulus._z.size == 108
    assert annulus._z[0] == 0
    assert annulus._conn.size == 384
    assert annulus._conn[0] == 0
    assert annulus._offsets.size == 48
    assert annulus._offsets[0] == 8
    assert annulus._ctypes.size == 48
    assert annulus._ctypes[0] == 12.0
    assert annulus._meshmap.size == 3
    assert annulus._meshmap[0] == 0
    assert annulus._meshmap[1] == 24
    assert annulus._meshmap[2] == 48
