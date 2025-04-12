import flowforge.materials.Fluid as fluma
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def compare_group(ref, comp):
    """
    Compare two HDF5 groups recursively.

    Args:
        ref: The first HDF5 group/dataset.
        comp: The second HDF5 group to compare with the first group.

    Returns:
        True if the two groups are the same.

    Raises:
        AssertionError: If the two groups are not the same.
    """
    for k, v in ref.items():
        assert k in comp
        if isinstance(v, h5py.Dataset):
            assert v[()].all() == comp[k][()].all()
        else:
            compare_group(v, comp[k])

    return True


def compare_ref(Tmin, Tmax, thresh, salt, ref_file, test_file):

    current_dir = os.path.dirname(__file__)
    test_folder_dir = f"{current_dir}/testFluid"

    test_filepath = f"{test_folder_dir}/{test_file}"  # Test file that will be made
    ref_filepath = f"{test_folder_dir}/{ref_file}"  # Reference file path

    if os.path.exists(test_filepath):
        os.remove(test_filepath)

    salt.exportHDF5(test_filepath, "", Tmin, Tmax, thresh)
    with h5py.File(test_filepath, "r") as test_file, h5py.File(ref_filepath, "r") as ref_file:
        compare_group(test_file, ref_file)

    # cleanup
    if os.path.exists(test_filepath):
        os.remove(test_filepath)


def test_flibe():  # test the hdf5 export function
    salt = fluma.FLiBe_UF4("salt")
    compare_ref(
        Tmin=400, Tmax=1600, thresh=1e-2, salt=salt, ref_file="ref_FluidTest_FLiBe_UF4.h5", test_file="FluidTest_FLiBe_UF4.h5"
    )


def test_hitec():  # test the hdf5 export function
    salt = fluma.Hitec("salt")
    np.testing.assert_approx_equal(salt.conductivity(500000), 0.54112348483547, significant=8)
    np.testing.assert_approx_equal(salt.density(500000), 1957.746305236, significant=8)
    np.testing.assert_approx_equal(salt.viscosity(500000), 0.01193414245605, significant=8)
    np.testing.assert_approx_equal(salt.surface_tension(500000), 0.124372083171, significant=8)
    np.testing.assert_approx_equal(salt.specific_heat(500000), 2406.2428723738, significant=8)
    np.testing.assert_approx_equal(salt.temperature(500000), 447.98411563, significant=8)
    np.testing.assert_approx_equal(salt.enthalpy(447.98411563), 500000.0, significant=8)
    compare_ref(Tmin=475, Tmax=700, thresh=1e-2, salt=salt, ref_file="ref_FluidTest_Hitec.h5", test_file="FluidTest_Hitec.h5")


"""
        try:
            compare_group(test_file, ref_file)
        except:
            print("ERROR IN FLUIDMATERIAL FILE COMPARISON")
        else:
            os.remove(test_filepath)
"""
# plotter is used to look at plots of the fluid properties for flibe to make sure they agree with expectations
"""
def plotter(tmin, tmax):
    salt = fluma.FLiBe_UF4('salt')
    temp=np.linspace(tmin,tmax,500)
    h=salt.enthalpy(temp)
    tc=salt.conductivity(h)
    d=salt.density(h)
    v=salt.viscosity(h)
    sh=salt.specific_heat(h)


    plt.plot(h,sh)
    plt.title('enthalpy vs. specific heat')
    plt.xlabel('specific enthalpy [J/kg]')
    plt.ylabel(' specific heat [J/kg-K]')
    plt.show()

    plt.plot(temp,h)
    plt.title('temperature vs. enthalpy')
    plt.xlabel('temperature [k]')
    plt.ylabel('specific enthalpy [J/kg]')
    plt.show()

    plt.plot(h,tc)
    plt.title('enthalpy vs. thermal conductivity')
    plt.xlabel('specific enthalpy [J/kg]')
    plt.ylabel(' conductivity [W/m-K]')
    plt.show()

    plt.plot(h,d)
    plt.title('enthalpy vs. density')
    plt.xlabel('specific enthalpy [J/kg]')
    plt.ylabel('density [kg/m^3]')
    plt.show()

    plt.plot(h,v)
    plt.title('enthalpy vs. viscosity')
    plt.xlabel('specific enthalpy [J/kg]')
    plt.ylabel('viscosity [kg/m-s]')
    plt.show()


plotter(500,1400)
"""
