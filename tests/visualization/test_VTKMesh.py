import pytest
import numpy as np
from flowforge.visualization import VTKMesh, genCyl, genUniformCube


def test_init():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    z = np.array([0, 1, 2])
    points = (x, y, z)
    with pytest.raises(Exception):
        VTKMesh(points, None, None, None, None)


def test_points():
    mesh = genCyl(2, 1)
    assert mesh.points == (mesh._x, mesh._y, mesh._z)


def test_add():
    mesh1 = genUniformCube(3, 3, 3)
    mesh2 = genUniformCube(5, 5, 5)
    assert mesh1._x.size == mesh2._x.size
    assert mesh1._conn.all() == mesh2._conn.all()
    assert mesh1._offsets.all() == mesh2._offsets.all()
    assert mesh1._ctypes.all() == mesh2._ctypes.all()
    assert mesh1._meshmap.all() == mesh2._meshmap.all()

    mesh3 = mesh1 + mesh2
    assert mesh3._x.size == mesh1._x.size * 2 == mesh2._x.size * 2
    assert mesh3._y.size == mesh1._y.size * 2 == mesh2._y.size * 2
    assert mesh3._z.size == mesh1._z.size * 2 == mesh2._z.size * 2
    assert mesh3._conn.size == mesh1._conn.size * 2
    assert mesh3._offsets.size == mesh1._offsets.size * 2
    assert mesh3._ctypes.size == mesh1._ctypes.size * 2
    assert mesh3._meshmap.size == mesh1._meshmap.size * 2 - 1


def test_rotate():
    mesh = genCyl(5, 1, resolution=4)

    assert round(mesh._x[0]) == round(mesh._x[5]) == 0
    assert round(mesh._x[1]) == round(mesh._x[6]) == 1
    assert round(mesh._x[2]) == round(mesh._x[7]) == 0
    assert round(mesh._x[3]) == round(mesh._x[8]) == -1
    assert round(mesh._x[4]) == round(mesh._x[9]) == 0

    assert round(mesh._y[0]) == round(mesh._y[5]) == 0
    assert round(mesh._y[1]) == round(mesh._y[6]) == 0
    assert round(mesh._y[2]) == round(mesh._y[7]) == 1
    assert round(mesh._y[3]) == round(mesh._y[8]) == 0
    assert round(mesh._y[4]) == round(mesh._y[9]) == -1

    for i in range(5):
        assert mesh._z[i] == 0
        assert mesh._z[i + 5] == 5

    mesh.translate(0, 0, 0, theta=np.pi / 2)

    for i in range(5):
        assert round(mesh._x[i]) == 0
        assert round(mesh._x[i + 5]) == 5

    assert round(mesh._y[0]) == round(mesh._y[5]) == 0
    assert round(mesh._y[1]) == round(mesh._y[6]) == 0
    assert round(mesh._y[2]) == round(mesh._y[7]) == 1
    assert round(mesh._y[3]) == round(mesh._y[8]) == 0
    assert round(mesh._y[4]) == round(mesh._y[9]) == -1

    assert round(mesh._z[0]) == round(mesh._z[5]) == 0
    assert round(mesh._z[1]) == round(mesh._z[6]) == -1
    assert round(mesh._z[2]) == round(mesh._z[7]) == 0
    assert round(mesh._z[3]) == round(mesh._z[8]) == 1
    assert round(mesh._z[4]) == round(mesh._z[9]) == 0

    mesh.translate(0, 0, 0, alpha=np.pi / 2)

    assert round(mesh._x[0]) == round(mesh._x[5]) == 0
    assert round(mesh._x[1]) == round(mesh._x[6]) == 0
    assert round(mesh._x[2]) == round(mesh._x[7]) == -1
    assert round(mesh._x[3]) == round(mesh._x[8]) == 0
    assert round(mesh._x[4]) == round(mesh._x[9]) == 1

    for i in range(5):
        assert round(mesh._y[i]) == 0
        assert round(mesh._y[i + 5]) == 5

    assert round(mesh._z[0]) == round(mesh._z[5]) == 0
    assert round(mesh._z[1]) == round(mesh._z[6]) == -1
    assert round(mesh._z[2]) == round(mesh._z[7]) == 0
    assert round(mesh._z[3]) == round(mesh._z[8]) == 1
    assert round(mesh._z[4]) == round(mesh._z[9]) == 0


def test_translate_and_rotate():
    mesh = genCyl(5, 1, resolution=4).translate(1, 1, 1, theta=np.pi / 2, alpha=np.pi / 2)

    assert round(mesh._x[0]) == round(mesh._x[5]) == 1
    assert round(mesh._x[1]) == round(mesh._x[6]) == 1
    assert round(mesh._x[2]) == round(mesh._x[7]) == 0
    assert round(mesh._x[3]) == round(mesh._x[8]) == 1
    assert round(mesh._x[4]) == round(mesh._x[9]) == 2

    for i in range(5):
        assert round(mesh._y[i]) == 1
        assert round(mesh._y[i + 5]) == 6

    assert round(mesh._z[0]) == round(mesh._z[5]) == 1
    assert round(mesh._z[1]) == round(mesh._z[6]) == 0
    assert round(mesh._z[2]) == round(mesh._z[7]) == 1
    assert round(mesh._z[3]) == round(mesh._z[8]) == 2
    assert round(mesh._z[4]) == round(mesh._z[9]) == 1
