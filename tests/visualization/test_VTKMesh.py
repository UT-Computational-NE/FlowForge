import pytest
import numpy as np
from flowforge.visualization import VTKMesh, genUniformCylinder, genUniformCube
import flowforge.visualization as vtk


def test_init():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    z = np.array([0, 1, 2])
    points = (x, y, z)
    with pytest.raises(Exception):
        VTKMesh(points, None, None, None, None)


def test_points():
    mesh = genUniformCylinder(2, 1)
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
    mesh = genUniformCylinder(5, 1, resolution=4)

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
    mesh = genUniformCylinder(5, 1, resolution=4).translate(1, 1, 1, theta=np.pi / 2, alpha=np.pi / 2)

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


def test_combine_hexahedron_hexahedron_2_cell_mesh():
    # generates a 2-cell mesh for testing
    cubes = genUniformCube(L=2.0, W=1.0, H=1.0, nx=2).translate(1.0, 0.5, 0.0)

    # tests combining two hexahedrons along the x-axis
    combined_mesh = cubes.combine_cells(cell_idx1=0, cell_idx2=1)
    assert combined_mesh.points[0].size == combined_mesh.points[1].size == combined_mesh.points[2].size == 20
    np.testing.assert_array_equal(
        combined_mesh.connections, np.array([0, 1, 7, 6, 4, 5, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 9, 8])
    )
    np.testing.assert_array_equal(combined_mesh.offsets, np.array([20]))
    np.testing.assert_array_equal(combined_mesh.ctypes, np.array([vtk.VtkQuadraticHexahedron.tid]))
    np.testing.assert_array_equal(combined_mesh.meshmap, np.array([0, 1]))

    # generates a 2-cell mesh for testing
    cubes = genUniformCube(L=1.0, W=2.0, H=1.0, ny=2).translate(0.5, 1.0, 0.0)

    # tests combining two hexahedrons along the y-axis
    combined_mesh = cubes.combine_cells(cell_idx1=0, cell_idx2=1)
    assert combined_mesh.points[0].size == combined_mesh.points[1].size == combined_mesh.points[2].size == 20
    np.testing.assert_array_equal(
        combined_mesh.connections, np.array([0, 1, 3, 2, 8, 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 4, 5, 7, 6])
    )
    np.testing.assert_array_equal(combined_mesh.offsets, np.array([20]))
    np.testing.assert_array_equal(combined_mesh.ctypes, np.array([vtk.VtkQuadraticHexahedron.tid]))
    np.testing.assert_array_equal(combined_mesh.meshmap, np.array([0, 1]))

    # generates a 2-cell mesh for testing
    cubes = genUniformCube(L=1.0, W=1.0, H=2.0, nz=2).translate(0.5, 0.5, 0.0)

    # tests combining two hexahedrons along the z-axis
    combined_mesh = cubes.combine_cells(cell_idx1=0, cell_idx2=1)
    assert combined_mesh.points[0].size == combined_mesh.points[1].size == combined_mesh.points[2].size == 20
    np.testing.assert_array_equal(
        combined_mesh.connections, np.array([0, 3, 9, 6, 2, 5, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 1, 4, 10, 7])
    )
    np.testing.assert_array_equal(combined_mesh.offsets, np.array([20]))
    np.testing.assert_array_equal(combined_mesh.ctypes, np.array([vtk.VtkQuadraticHexahedron.tid]))
    np.testing.assert_array_equal(combined_mesh.meshmap, np.array([0, 1]))


def test_combine_hexahedron_hexahedron_multi_cell():
    # generates a multi-cell mesh for testing
    cubes = genUniformCube(L=2.0, W=2.0, H=1.0, nx=2, ny=2, nz=1).translate(1.0, 1.0, 0.0)

    # tests combining two hexahedrons in a multi-cell mesh along the x-axis
    combined_mesh = cubes.combine_cells(cell_idx1=0, cell_idx2=1)
    assert combined_mesh.points[0].size == combined_mesh.points[1].size == combined_mesh.points[2].size == 26
    np.testing.assert_array_equal(
        combined_mesh.connections,
        np.array(
            [
                0,
                1,
                7,
                6,
                4,
                5,
                11,
                10,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                2,
                3,
                9,
                8,
                6,
                7,
                9,
                8,
                12,
                13,
                15,
                14,
                8,
                9,
                11,
                10,
                14,
                15,
                17,
                16,
            ]
        ),
    )
    np.testing.assert_array_equal(combined_mesh.offsets, np.array([20, 28, 36]))
    np.testing.assert_array_equal(
        combined_mesh.ctypes, np.array([vtk.VtkQuadraticHexahedron.tid, vtk.VtkHexahedron.tid, vtk.VtkHexahedron.tid])
    )
    np.testing.assert_array_equal(combined_mesh.meshmap, np.array([0, 1, 2, 3]))

    # generates a multi-cell mesh for testing
    cubes = genUniformCube(L=2.0, W=2.0, H=1.0, nx=2, ny=2, nz=1).translate(1.0, 1.0, 0.0)

    # tests combining two hexahedrons in a multi-cell mesh along the y-axis
    combined_mesh = cubes.combine_cells(cell_idx1=1, cell_idx2=3)
    assert combined_mesh.points[0].size == combined_mesh.points[1].size == combined_mesh.points[2].size == 26
    np.testing.assert_array_equal(
        combined_mesh.connections,
        np.array(
            [
                0,
                1,
                3,
                2,
                6,
                7,
                9,
                8,
                2,
                3,
                5,
                4,
                14,
                15,
                17,
                16,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                8,
                9,
                11,
                10,
                6,
                7,
                9,
                8,
                12,
                13,
                15,
                14,
            ]
        ),
    )
    np.testing.assert_array_equal(combined_mesh.offsets, np.array([8, 28, 36]))
    np.testing.assert_array_equal(
        combined_mesh.ctypes, np.array([vtk.VtkHexahedron.tid, vtk.VtkQuadraticHexahedron.tid, vtk.VtkHexahedron.tid])
    )
    np.testing.assert_array_equal(combined_mesh.meshmap, np.array([0, 1, 2, 3]))

    # generates a multi-cell mesh for testing
    cubes = genUniformCube(L=2.0, W=1.0, H=2.0, nx=2, ny=1, nz=2).translate(1.0, 0.5, 0.0)

    # tests combining two hexahedrons in a multi-cell mesh along the z-axis
    combined_mesh = cubes.combine_cells(cell_idx1=1, cell_idx2=3)
    assert combined_mesh.points[0].size == combined_mesh.points[1].size == combined_mesh.points[2].size == 26
    np.testing.assert_array_equal(
        combined_mesh.connections,
        np.array(
            [
                0,
                1,
                4,
                3,
                9,
                10,
                13,
                12,
                3,
                6,
                15,
                12,
                5,
                8,
                17,
                14,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                4,
                7,
                16,
                13,
                1,
                2,
                5,
                4,
                10,
                11,
                14,
                13,
            ]
        ),
    )
    np.testing.assert_array_equal(combined_mesh.offsets, np.array([8, 28, 36]))
    np.testing.assert_array_equal(
        combined_mesh.ctypes, np.array([vtk.VtkHexahedron.tid, vtk.VtkQuadraticHexahedron.tid, vtk.VtkHexahedron.tid])
    )
    np.testing.assert_array_equal(combined_mesh.meshmap, np.array([0, 1, 2, 3]))
