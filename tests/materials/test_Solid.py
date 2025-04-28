import os
import h5py
import numpy as np
from pytest import approx
from flowforge.materials.Solid import Graphite, Solid_table, SS316H


# tests the functionality of the Solid_table class
def test_Solid_table():
    g = Graphite(name="Graphite 1")

    # McEligot, Donald, Swank, W. David, Cottle, David L., and Valentin, Francisco I. Thermal Properties of G-348 Graphite. United States: N. p., 2016.
    T = [22.6, 101.0, 199.3, 301.6, 401.6, 501.6, 601.6, 701.7, 801.3, 900.8, 1000.9]
    T = [ti + 273.15 for ti in T]
    k = [133.02, 128.54, 117.62, 106.03, 96.70, 88.61, 82.22, 76.52, 71.78, 67.88, 64.26]
    dens = [1888.5, 1886.3, 1883.5, 1880.4, 1877.2, 1873.9, 1870.5, 1867.0, 1863.4, 1859.6, 1855.7]
    cp = [0.72619, 0.93315, 1.15447, 1.34107, 1.48683, 1.60353, 1.69743, 1.77360, 1.83558, 1.88668, 1.92944]
    table_test = Solid_table(name="Graphite 2", T_k=T, k=k, T_dens=T, dens=dens, T_cp=T, cp=cp)

    np.testing.assert_allclose(g.conductivity(500), table_test.conductivity(500), rtol=1)
    np.testing.assert_allclose(g.density(500), table_test.density(500), rtol=1)
    np.testing.assert_allclose(g.specific_heat(500), table_test.specific_heat(500), rtol=1)


# tests that the exportHDF5 function properly exports the data
def test_exportHDF5():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    ref_filename = os.path.join(current_dir, "testSolid/ref_SolidTest.h5")
    filename = os.path.join(current_dir, "SolidTest.h5")
    path = "material_properties/"
    g = Graphite(name="Graphite")
    g.exportHDF5(filename, path, Tmin=273.15, Tmax=1273.15)

    # to create new reference file:
    # g.exportHDF5(ref_filename, path, Tmin=273.15, Tmax=1273.15)

    with h5py.File(filename, "r") as test_file, h5py.File(ref_filename, "r") as ref_file:
        path = "material_properties/solid_material/"

        # conductivity
        k_path = path + "conductivity/"
        assert test_file[k_path + "temps"][()].size == ref_file[k_path + "temps"][()].size
        for i in range(test_file[k_path + "temps"][()].size):
            np.testing.assert_almost_equal(test_file[k_path + "temps"][()][i], ref_file[k_path + "temps"][()][i], decimal=3)
            np.testing.assert_almost_equal(test_file[k_path + "values"][()][i], ref_file[k_path + "values"][()][i], decimal=3)

        # specific heat
        cp_path = path + "specific_heat/"
        assert test_file[cp_path + "temps"][()].size == ref_file[cp_path + "temps"][()].size
        for i in range(test_file[cp_path + "temps"][()].size):
            np.testing.assert_almost_equal(test_file[cp_path + "temps"][()][i], ref_file[cp_path + "temps"][()][i], decimal=3)
            np.testing.assert_almost_equal(
                test_file[cp_path + "values"][()][i], ref_file[cp_path + "values"][()][i], decimal=3
            )

        # density
        rho_path = path + "density/"
        assert test_file[rho_path + "temps"][()].size == ref_file[rho_path + "temps"][()].size
        for i in range(test_file[rho_path + "temps"][()].size):
            np.testing.assert_almost_equal(
                test_file[rho_path + "temps"][()][i], ref_file[rho_path + "temps"][()][i], decimal=3
            )
            np.testing.assert_almost_equal(
                test_file[rho_path + "values"][()][i], ref_file[rho_path + "values"][()][i], decimal=3
            )

    os.remove(filename)


def test_ss316h():
    ss316h = SS316H(name="SS316H")
    np.testing.assert_almost_equal(ss316h.conductivity(T=373.15), 16.3, decimal=2)
    np.testing.assert_almost_equal(ss316h.conductivity(T=773.15), 21.5, decimal=2)
    np.testing.assert_almost_equal(ss316h.density(T=373.15), 8000, decimal=2)
    np.testing.assert_almost_equal(ss316h.specific_heat(T=373.15), 0.500, decimal=2)
