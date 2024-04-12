import flowforge.meshing.FluidMesh as fm
from flowforge.visualization.VTKMesh import VTKMesh
from flowforge.visualization.VTKFile import VTKFile
from flowforge.input.UnitConverter import UnitConverter

class System:
    """
    Controls the whole system by initializing all components and writing vtk solution file.
    """
    def __init__(self, components, sysdict, unitdict):
        """
        Initialize system of components

        Args:
            components: List of initialized components
            sysdict: Dictionary of system settings describing how to initialize
            unitdict: Dictionary of units used to convert input
        """
        self._components = []
        self._connectivity = []
        self._inBoundComp = None
        self._outBoundComp = None
        self._fluidMesh = fm.FluidMesh()

        if "simple_loop" in sysdict:
            self._setupSimpleLoop(components, **sysdict['simple_loop'])
        elif "section" in sysdict:
            self._setupSection(components, **sysdict['section'])
        # TODO add additional types of systems that can be set up

        for comp in self._components:
            comp._convertUnits(UnitConverter(unitdict))

    @property
    def fluidMesh(self):
        """
        Returns the fluid mesh of the system.
        """
        return self._fluidMesh

    def _setupSimpleLoop(self, components, loop):
        """
        Sets up a loop of components (last components outlet connects to first component input)

        Args:
            components: List of initialized components
            loop: Dictionary describing the loop
        """
        # Loop over each component in the loop, add those components to the list, define the connections between components
        for i, comp in enumerate(loop):
            self._components.append(components[comp])
            # connect the current component to the previous (exclude the first component because there isn't a previous)
            if i > 0:
                self._connectivity.append((self._components[i-1], self._components[i]))
            # If the last entry in the loop, connect the last component to the first
            if i == len(loop) - 1:
                self._connectivity.append((self._components[i], self._components[0]))

    def _setupSection(self, components, section):
        """
        Sets up a section (this is a model with defined inlet and outlet boundary conditions)

        Args:
            components: List of initialized components
            section: Dictionary describing the model
        """
        # Define the component the inlet boundary condition is applied to
        self._inBoundComp = components[section[0]]
        # Loop over each entry in section, add the components, and connect the compnents to each other
        for i, entry in enumerate(section):
            self._components.append(components[entry])
            if i > 0:
                self._connectivity.append((self._components[i-1], self._components[i]))
        # Define the component the outlet boundary condition is applied to
        self._outBoundComp = components[section[-1]]

    def getCellGenerator(self):
        """
        Generator interface that returns all of the nodes in a model sequentially
        Args:
        """
        for comp in self._components:
            yield from comp.getNodeGenerator()

    def getMesh(self):
        """
        Interface sets up the fluid mesh for the full system
        """
        # Loop over all of the compnents in the model and add the component fluid mesh to the systems fluid mesh
        inlet_coords=(0,0,0)
        for i, comp in enumerate(self._components):
            # If the first component and a boundary component is defined, pass that boundary component in as a
            # surface only inlet, otherwise don't define an inlet or outlet
            if i == 0 and self._inBoundComp is not None:
                comp.setupFluidMesh(self._fluidMesh, inlet=(fm.Surface(self._inBoundComp.inletArea), None),
                                    inlet_coords=inlet_coords)

            else:
                comp.setupFluidMesh(self._fluidMesh, inlet_coords=inlet_coords)

            inlet_coords=comp.getOutlet(inlet_coords)
        # Glue components together with connections.  Test that the inlet and outlet areas are the same between
        # components in the system.
        # TODO: need to set up these connections as we setup FluidMesh...
        # comp.setupFluidmesh, add intermediate surface, etc.
        for c_down, c_up in self._connectivity:
            if abs(1. - c_down.outletArea / c_up.inletArea) > 1e-2:
                print(f'WARNING: Flow areas do not agree for adjacent components {type(c_down).__name__:s} \
                      {type(c_up).__name__:s} with areas {c_down.outletArea:f} and {c_up.inletArea:f} respectively')
            self._fluidMesh.addConnection(fm.Surface(min(c_down.outletArea, c_up.inletArea)),
                                          self._fluidMesh.getNode(c_down.lastNodeIndex),
                                          self._fluidMesh.getNode(c_up.firstNodeIndex))

        #if there is an outlet boundary condition, add a boundary surface to the last node in the
        if self._outBoundComp is not None:
            self._fluidMesh.addBoundarySurface(self._fluidMesh.getNode(self._outBoundComp.lastNodeIndex),
                                               outsurf=fm.Surface(self._outBoundComp.outletArea))

    def getVTKMesh(self):
        """
        The getVTKMesh function is a private function that starts with the
        first inlet at the origin and adds the mesh of every component to the
        system mesh. The system mesh is then stored as _sysMesh. The translation
        of each component is taken care of within the getVTKMesh function in the
        component classes.

        Args: None
        """
        inlet = (0, 0, 0)
        mesh = VTKMesh()
        for c in self._components:
            mesh += c.getVTKMesh(inlet)
            inlet = c.getOutlet(inlet)
        return mesh

    def writeVTKFile(self, filename, time=None): #pylint:disable=unused-argument
        """
        The write system file will be used to export the whole system mesh into
        a VTK file to view in another program. This function calls the _getSystemMesh
        function to loop through the components and generate the meshes for each one.
        This function will then utilize the VTKFile class to create the exported file.

        Args:
            filename : str, name of the file to be exported
            time     : TODO - Implement
        """
        sysFile = VTKFile(filename, self.getVTKMesh())
        sysFile.writeFile()

    @property
    def nCell(self):
        """
        Returns the number of cells in the system.
        """
        ncell = 0
        for c in self._components:
            ncell += c.nCell
        return ncell

if __name__ == "__main__":
    import json
    from syth.Components import component_factory

    with open('sample3.json', 'r') as f:
        input_dict = json.load(f)

    comp = component_factory(input_dict['components'])
    sys = System(comp, input_dict.get('system', {}), input_dict.get('units', {}))
    sys.writeVTKFile("sample3")
    sys.getMesh()
