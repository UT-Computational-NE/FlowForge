import pytest
import os
import numpy as np
from flowforge.visualization import VTKFile, VTKMesh, genUniformCube, genUniformCylinder
from vtk import vtkXMLUnstructuredGridReader


def generate_reference():
    """
    The generate_reference function generates a VTK file that will be compared to the test files.
    This is only to be used if certain that you would like to overwrite the previous reference file.
    """
    mymesh = genUniformCylinder(10, 1, resolution=20, naxial_layers=10)
    myfile = VTKFile("testVTK/referenceFile", mesh=mymesh)
    P = np.arange(10)
    myfile["pressure"] = P
    myfile.writeFile()


def compare_files(file_ref, file_test):
    ref_reader = vtkXMLUnstructuredGridReader()
    ref_reader.SetFileName(file_ref)
    ref_reader.Update()
    ref_data = ref_reader.GetOutput()

    test_reader = vtkXMLUnstructuredGridReader()
    test_reader.SetFileName(file_test)
    test_reader.Update()
    test_data = test_reader.GetOutput()

    assert ref_data.GetNumberOfCells() == test_data.GetNumberOfCells()
    assert ref_data.GetNumberOfPoints() == test_data.GetNumberOfPoints()
    assert ref_data.GetDataObjectType() == test_data.GetDataObjectType()
    assert ref_data.GetActualMemorySize() == test_data.GetActualMemorySize()
    assert ref_data.GetCellType(0) == test_data.GetCellType(0)


def test_no_filename_input():
    with pytest.raises(Exception):
        myfile = VTKFile()


def test_no_mesh():
    filename = "filename"
    myfile = VTKFile(filename)
    assert myfile.vtkmesh == None
    assert myfile.filepath == filename
    assert myfile.data == {}
    with pytest.raises(Exception):
        myfile.writeFile()


def test_empty_mesh():
    filename = "filename"
    mymesh = VTKMesh()
    myfile = VTKFile(filename, mymesh)
    assert myfile.vtkmesh == mymesh
    assert myfile.filepath == filename
    assert myfile.data == {}
    with pytest.raises(Exception):
        myfile.writeFile()


def test_mesh_no_data():
    filename = "filename"
    mymesh = genUniformCube(3, 3, 3)
    myfile = VTKFile(filename, mesh=mymesh)
    assert myfile.vtkmesh is not None
    myfile.writeFile()
    assert os.path.exists(f"{filename}.vtu")
    os.remove(f"{filename}.vtu")


def test_mesh_with_data():
    import os

    mymesh = genUniformCylinder(10, 1, resolution=20, naxial_layers=10)
    myfile = VTKFile("meshwithdata", mesh=mymesh)
    assert myfile.vtkmesh is not None
    P = np.arange(10)
    myfile["pressure"] = P
    assert myfile.data != {}
    assert myfile.data["pressure"].all() == P.all()
    myfile.writeFile()
    assert os.path.exists("meshwithdata.vtu")
    compare_files(os.path.dirname(__file__) + "/testVTK/referenceFile.vtu", "meshwithdata.vtu")
    os.remove("meshwithdata.vtu")
