import os
import h5py
from flowforge.meshing.FluidMesh import FluidMesh, Node, Surface


def test_node_and_surface():
    node0 = Node(0.05, 0.03, 0.678, 0.898, 0.6)
    node1 = Node(0.04, 0.04, 0.778, 0.988)
    assert node0.volume == 0.05
    assert node0.hydraulicDiameter == 0.03
    assert node0.heatedArea == 0.678 * 0.898
    assert node0.length == 0.898
    assert node0.area == 0.05 / 0.898
    assert node0.costh == 0.6
    node1.index = 22
    node0.index = 7
    assert node1.index == 22
    surf1 = Surface(0.045)
    surf2 = Surface(0.035)
    assert surf1.area == 0.045
    node0.inSurface = surf1
    node0.outSurface = surf2
    assert node0.outSurface != node0.inSurface != []
    surf1.index = 17
    surf2.index = 32
    assert surf1.index == 17
    surf2.fromNode = node0
    surf2.toNode = node1
    assert surf2.fromNode != surf2.toNode != []
    try:
        assert node0.index < 0
        passed = False
    except AssertionError:
        passed = True
    assert passed


def compare_group(ref, comp):
    for k, v in ref.items():
        assert k in comp
        if isinstance(v, h5py.Dataset):
            assert v[()] == comp[k][()]
        else:
            compare_group(v, comp[k])


def test_fluidMesh():
    node2 = Node(0.03, 0.05, 0.007, 0.787)
    mesh = FluidMesh()
    mesh.addNode(node2)
    assert node2.index >= 0
    for i in range(5):
        mesh.addNode(Node(0.1 * (i + 1), 0.05 * (i + 1), 0.04 * (i + 1), 0.2 * (i + 1)))
    for i in range(6):
        mesh.addConnection(Surface(0.07 * (i + 1)), mesh.getNode(i), mesh.getNode((i + 1) % 6))
    assert mesh.nextNodeIndex == 6
    assert mesh.prevNodeIndex == 5

    test = str(os.path.dirname(__file__)) + "/testFluidMesh/test.h5"
    reference = str(os.path.dirname(__file__)) + "/testFluidMesh/fluid_mesh_reference.h5"

    if os.path.exists(test):
        os.remove(test)

    mesh.exportHDF5(test)

    with h5py.File(reference, "r") as h5r, h5py.File(test, "r") as h5c:
        compare_group(h5r, h5c)

    # cleanup
    if os.path.exists(test):
        os.remove(test)
