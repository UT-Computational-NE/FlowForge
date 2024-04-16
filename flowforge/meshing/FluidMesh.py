import pprint
import h5py


class Node:
    """
    Nodes are 3D volumes that make up the fluid mesh network.
    If the fluid mesh is a series of connected pipes, one of the pipes defined by a right cylinder would be a node.
    The insurface/outsurface objects indicate which surface fluid is flowing in/out. For a pipe, the insurface would
    be a circle with diameter=hydDiam
    """

    def __init__(self, V, Dh, Ph, L, costh=0.0, boundingbox=[]):
        """
        The initialization of the node class.

        Args:
            - V  : float, the volume of the node in m^3
            - Dh : float, the Hydraulic Diameter of the node in m
            - Ph : float, the heated perimeter of the node in m
            - L  : float, the length of the node in m^3
            - costh : (OPTIONAL) float, the cosine of the angle with respect to the z-axis, default=0.0 (orthogonal to z)
        """
        self._volume = V
        self._hydDiam = Dh
        self._heatedPerm = Ph
        self._length = L
        self._costh = costh
        self._idx = -1
        self._inSurface = []
        self._outSurface = []
        self._boundingBox = boundingbox

    @property
    def volume(self):
        """
        Returns volume of the node.
        """
        return self._volume

    @property
    def hydraulicDiameter(self):
        """
        Returns hydraulic diameter of the node.
        """
        return self._hydDiam

    @property
    def heatedArea(self):
        """
        Returns the heated area of the node.
        """
        return self._heatedPerm * self._length

    @property
    def length(self):
        """
        Returns the length of the node.
        """
        return self._length

    @property
    def area(self):
        """
        Returns the area of the node.
        """
        return self._volume / self._length

    @property
    def costh(self):
        """
        Returns cosine of the angle w.r.t. the z-axis of the node.
        """
        return self._costh

    @property
    def index(self):
        """
        Returns the index of the node.
        """
        assert self._idx >= 0
        return self._idx

    @index.setter
    def index(self, idx):
        """
        Setter function for the index of the node.
        """
        assert self._idx == -1
        self._idx = idx

    @property
    def inSurfaces(self):
        """
        Returns the inlet surface of the node.
        """
        return self._inSurface

    @inSurfaces.setter
    def inSurfaces(self, surf):
        """
        Setter for the inlet surface of the node.
        """
        assert self._idx >= 0
        self._inSurface.append(surf)

    @property
    def outSurfaces(self):
        """
        Returns the outlet surface of the node.
        """
        return self._outSurface

    @outSurfaces.setter
    def outSurfaces(self, surf):
        """
        Sets the outlet surface of the node.
        """
        assert self._idx >= 0
        self._outSurface.append(surf)

    def exportHDF5(self, group):
        """
        Exports a node's attributes (volume, hydraulic diameter, etc.) as well as a list of inlet surfaces,
        and a list of out surfaces as their own datasets under a group labeled by node and its index to
        an HDF5 group giving in arg.

        Args:
            group : h5py.Group, group in a HDF5 file

        Example:
            f=h5py.File('example.h5','w')
            a_group=f.create_group("/example_node")
            some_node.exportHDF5(a_group)
        """
        inID = []
        outID = []
        for m in self.outSurfaces:
            outID.append(m.index)
        for s in self.inSurfaces:
            inID.append(s.index)
        info = group.create_group(f"node_{self.index:04d}")
        info.create_dataset("volume", data=self.volume)
        info.create_dataset("hydraulic_diameter", data=self.hydraulicDiameter)
        info.create_dataset("heated_area", data=self.heatedArea)
        info.create_dataset("length", data=self.length)
        info.create_dataset("cosine_theta", data=self.costh)
        info.create_dataset("in_surface_indices", data=inID)
        info.create_dataset("out_surface_indices", data=outID)
        info.create_dataset("bounding_box", data=self._boundingBox)


class Surface:
    """
    Surfaces desribe the connection surfaces between nodes. They are characterized by their area and index.
    fromNode and toNode indicate which nodes the surface is connecting and in what order.
    """

    def __init__(self, A):
        """
        The initialization of the surface class.

        Args:
            - A : float, the area of the surface in m^2
        """
        self._area = A
        self._fromNode = []
        self._toNode = []
        self._idx = -1

    @property
    def area(self):
        """
        Returns the area of the surface.
        """
        return self._area

    @property
    def index(self):
        """
        Returns the surface index.
        """
        assert self._idx >= 0
        return self._idx

    @index.setter
    def index(self, idx):
        """
        Setter for the surface index.
        """
        assert self._idx == -1
        self._idx = idx

    @property
    def fromNode(self):
        """
        Returns the node index that the flow is moving from.
        """
        assert self._idx >= 0
        return self._fromNode

    @fromNode.setter
    def fromNode(self, below):
        """
        Sets the node index that the flow is moving from.

        Args:
            - below : int, node index
        """
        assert self._idx >= 0
        assert below.index >= 0
        self._fromNode.append(below)

    @property
    def toNode(self):
        """
        Returns the node index that the flow is moving to.
        """
        assert self._idx >= 0
        return self._toNode

    @toNode.setter
    def toNode(self, above):
        """
        Sets the node index that the flow is moving to.

        Args:
            - above : int, node index
        """
        assert self._idx >= 0
        assert above.index >= 0
        self._toNode.append(above)

    def exportHDF5(self, group):
        """
        Exports a surfaces's area, from Node, and to Node to their own dataset under a
        group labeled by surface and its index to an HDF5 group given by arg

        Args:
            group : h5py.Group, group in a HDF5 file

        Ex:
            f=h5py.File('example.h5','w')
            a_group=f.create_group("/example_surface")
            surf.exportHDF5(a_group)
        """
        from_node = []
        to_node = []
        for m in self.fromNode:
            from_node.append(m.index)
        for s in self.toNode:
            to_node.append(s.index)
        info = group.create_group(f"surface_{self.index:04d}")
        info.create_dataset("area", data=self.area)
        info.create_dataset("from_node_index", data=from_node, dtype=int)
        info.create_dataset("to_node_index", data=to_node, dtype=int)


class FluidMesh:
    """
    A mesh is "discretization of a geometric domain into small simple shapes" (nodes) and is made up by the
    nodes and the surfaces that connect the nodes to each other.
    """

    def __init__(self):
        """
        Initializes the fluid mesh with no nodes or surfaces.
        """
        self._nodes = []
        self._surfs = []
        self._nextNodeID = 0
        self._nextSurfID = 0

    @property
    def nodes(self):
        """
        Returns the list of nodes in the fluid mesh.
        """
        return self._nodes

    @property
    def surfaces(self):
        """
        Returns the list of surfaces in the fluid mesh.
        """
        return self._surfs

    def addNode(self, n):
        """
        Adds a node to the mesh. Nodes can be added to the mesh but are not apart of the loop
        until connection surfaces are inputted and ascribed to nodes using addConnection().

        Args:
            - n : Node class, the node which is being added to the mesh
        """
        n.index = self._nextNodeID
        self._nextNodeID += 1
        self._nodes.append(n)

    def getNode(self, idx):
        """
        Gets a node from the mesh

        Args:
            - idx : int, the node index

        Returns:
            - Node class, the node which was requested
        """
        assert 0 <= idx < self._nextNodeID
        return self._nodes[idx]

    @property
    def nextNodeIndex(self):
        """
        Returns the next node index.
        """
        return self._nextNodeID

    @property
    def prevNodeIndex(self):
        """
        Returns the previous node index.
        """
        assert self._nextNodeID > 0
        return self._nextNodeID - 1

    @property
    def nNodes(self):
        """
        Returns the number of nodes in the fluid mesh.
        """
        return len(self._nodes)

    @property
    def nConnections(self):
        """
        Returns the number of connections in the fluid mesh.
        """
        return len(self._surfs)

    def Nodes(self):
        """
        Generator interface for getting each node sequentially
        """
        yield from self._nodes

    def Surfaces(self):
        """
        Generator interface for getting each surface sequentially
        """
        yield from self._surfs

    def addConnection(self, surf, below, above):
        """
        Adds a surface to the mesh.

        Args:
            - surf  : Surface class, the surface which is being added to the mesh
            - below : Node class, Node the flow is coming from
            - above : Node class, Node flow is going to
        """
        surf.index = self._nextSurfID
        self._nextSurfID += 1
        self._surfs.append(surf)

        surf.fromNode = below
        surf.toNode = above

        below.outSurfaces = surf
        above.inSurfaces = surf

    def addBoundarySurface(self, n, insurf=None, outsurf=None):
        """
        Adds a free surface to the mesh.

        Args:
            - n : Node class, node the boundary will be attached to
            - insurf  : Surface class, inlet boundary surface
            - outsurf : Surface class, outlet boundary surface

        Notes:
            - Only insurf or outsurf can be provided, not both
        """
        assert (insurf is None) ^ (outsurf is None)  # XOR logic to ensure one but not both insurf and outsurf are passed in
        if insurf is not None:
            insurf.index = self._nextSurfID
            self._nextSurfID += 1
            self._surfs.append(insurf)

            insurf.toNode = n
            n.inSurfaces = insurf

        if outsurf is not None:
            outsurf.index = self._nextSurfID
            self._nextSurfID += 1
            self._surfs.append(outsurf)

            outsurf.fromNode = n
            n.outSurfaces = outsurf

    def printNodes(self):
        """
        Prints the nodes of the fluid mesh.
        """
        for n in self._nodes:
            print(f"Node {n.index:d}")
            pprint.pprint(vars(n))

    def exportHDF5(self, filename, path="/"):
        """
        Exports a fluid mesh to HDF5 in a group for nodes and a group for surfaces. The nodes and surfaces are
        each generated as a group by their index. Nodes have datasets for respective volume, hydralic diameters,
        a list of in surfaces, and a list of out surfaces. Surfaces have datasets for their respectuve area,
        fromNode, and toNode. The number of total surfaces and nodes are added to each respective group as a dataset.

        Args:
            filename : str, HDF5 file name
            path     : (OPTIONAL) str, path inside hdf5 file where fluid_mesh will be written,
                       if not set fluid_mesh will be added under root

        Example:
            f=h5py.File('example.h5','w')
            loop=f.create_group("/example_mesh")
            some_mesh.exportHDF5(loop)
        """
        f = h5py.File(filename, "a")
        loop = f.create_group(path + "fluid_mesh")
        surfs = loop.create_group("surfaces")
        nodes = loop.create_group("nodes")
        for node in self.Nodes():
            node.exportHDF5(nodes)
        for surf in self.Surfaces():
            surf.exportHDF5(surfs)
        nodes.create_dataset("number_of_nodes", data=self.nNodes)
        surfs.create_dataset("number_of_surfaces", data=self.nConnections)
        f.close()


if __name__ == "__main__":
    mesh = FluidMesh()
    for i in range(5):
        mesh.addNode(Node(0.1 * (i + 1), 0.05 * (i + 1), 0.04 * (i + 1), 0.2 * (i + 1)))

    for i in range(5):
        mesh.addConnection(Surface(0.07 * (i + 1)), mesh.getNode(i), mesh.getNode((i + 1) % 5))

    for i in range(5):
        print(i)
        print(vars(mesh.surfaces[i]))
