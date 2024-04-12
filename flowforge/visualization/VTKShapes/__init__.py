CYL_RESOLUTION = 8

"""
Functions to generate shape meshes.

General Method to VTK Mesh Creation in the following functions:
1. Define every point in the mesh.
2. Define the connections of those points. In a way this step would be
    "connecting the dots" in order to draw the lines of the 3d shape you want
3. Define the offset of each shape in the whole mesh
    (i.e. define how many points are in the shape you are drawing with the connections)
4. Define the cell types (from import pyevtk library)
    VtkHexahedron - 6 sided 3d shape
    VtkWedge - triangular prism
5. Define the mesh map. This is a list we implemented to be able to identify where
    the data would need to be added (i.e. Pressure, Temperature). This data will be able to
    be visualized in the rendering software (such as VisIt).
"""
