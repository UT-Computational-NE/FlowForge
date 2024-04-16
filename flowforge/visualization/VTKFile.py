from pyevtk.hl import unstructuredGridToVTK
import numpy as np


class VTKFile:
    """
    The VTKFile Class provides the ability to convert the mesh data into a file
    that is able to be rendering using additional software (i.e. VisIt). This class
    must be initialized with a file path. This will be the name of the file that is
    saved after calling the 'writeFile()' function.

    If no data is set using the __setitem__ function, the file will still write with a data list that is set by default to be
    empty. No errors will occur if a mesh is desired with no data values.
    """

    def __init__(self, filepath, mesh=None):
        """
        The VTKFile class is initialized with the desired file path of the export file.
        An example if you want the file to be created in the current directory would
        be to use filepath = "./VTKfilename"

        Args:
            filepath : str, desired name of the export VTK file
            mesh     : (OPTIONAL) VTKMesh, a fully defined VTKMesh class instance.
                       This can be added after initialization with the 'addMesh' function.
        """
        self.vtkmesh = mesh
        self.filepath = filepath
        self.data = {}

    def addMesh(self, mesh):
        """
        The addMesh function stores the mesh input to prepare to export to a VTK file for viewing.

        Args:
            mesh : VTKMesh, class instance that is fully populated with mesh data
        """
        self.vtkmesh = mesh

    def __setitem__(self, key, data):
        """
        The __setitem__ function sets the mesh data using the _unroll_data function.
        The most useful way to use this function is the implicitly call it by doing something like:
        "data['pressure'] = pressure_data"

        If no data is set, the file will still write with a data list that is set by default to be
        empty. No errors will occur if a mesh is desired with no data values.

        Args:
            key  : str, this is the name of the data being entered. This is the name that
                   will appear when visualizing the exported file in external software
            data : np array, this contains the data values for the cells in the mesh
                   such as the pressure or temperature values of that cell
        """
        self.data[key] = self._unroll_data(data)

    def _unroll_data(self, data):
        """
        The _unroll_data function is a helper function that loops through the inputed data array
        as well as the meshmap and sets the values to the appropriate cells in the mesh.

        Args:
            data : np array, this contains the data values for the cells in the mesh
                   such as the pressure or temperature values of that cell
        """
        assert self.vtkmesh is not None
        values = np.zeros(self.vtkmesh.offsets.size)
        for i in range(len(self.vtkmesh.meshmap) - 1):
            values[int(self.vtkmesh.meshmap[i]) : int(self.vtkmesh.meshmap[i + 1])] = data[i]
        return values

    def writeFile(self):
        """
        The writeFile function exports all the saved data as a '.vtu' file.
        This file will allow you to visualize the meshes created. It will
        also allow the user to visualize any data stored using the '__setItem__' function.
        This function makes use of the 'unstructuredGridToVTK' function from the
        pyevtk library.

        Args: None
        """
        assert self.vtkmesh is not None
        unstructuredGridToVTK(
            self.filepath,
            *self.vtkmesh.points,
            connectivity=self.vtkmesh.connections,
            offsets=self.vtkmesh.offsets,
            cell_types=self.vtkmesh.ctypes,
            cellData=self.data
        )
