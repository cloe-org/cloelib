# import jax.numpy as np
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.auxiliary import photoz_rsd_correction
from cloelib.observables.clusters.halo_clustering import TwoPoint3DHaloClustering
from cloelib.observables.clusters.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


def _test_clustering(CL, perturbations):
    z_test = np.array([0.0, 1.0])
    r_test = np.array([30.0, 60.0, 90.0])
    lob_test = np.array([50.0])
    k_test = np.geomspace(1e-4, 10, 500)

    print("    alcock_paczynski_correction_factor")
    ref_APcorr = np.array([1.0162, 1.016033])
    assert_allclose(
        CL.core.alcock_paczynski_correction_factor(z_test), ref_APcorr, rtol=1e-04
    )

    print("    radial_shell_window_and_volume")
    ref_radial_shell_window_and_volume0 = np.array(
        [
            [[9.99995884e-01, -1.35023865e-05], [9.99989679e-01, 8.02679298e-06]],
            [[9.99995885e-01, -1.37175506e-05], [9.99989682e-01, 8.32722698e-06]],
        ]
    )

    ref_radial_shell_window_and_volume1 = np.array(
        [[830784.512318, 2254986.533435], [830373.221984, 2253870.173956]]
    )

    WF, VF = CL.core.radial_shell_window_and_volume(z_test, k_test, r_test)
    assert_allclose(WF[:, :, [0, -1]], ref_radial_shell_window_and_volume0, rtol=1e-03)
    assert_allclose(VF, ref_radial_shell_window_and_volume1, rtol=1e-03)

    print("    Pk_IR_func")
    ref_Pk_IR = np.array([[4.228415e02, 1.061383e-01], [1.561681e02, 3.934860e-02]])

    Pk_test = perturbations.matter_power_spectrum(
        z_test, k_test, hubble_units=True, k_hunit=True
    )
    assert_allclose(
        CL.core.Pk_IR_func(k_test, Pk_test)[:, [0, -1]], ref_Pk_IR, rtol=1e-4
    )


def test_clustering():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _ns = 0.96
    _mnu = 0.06
    _As = 2.0e-9
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=_ns,
        mnu=_mnu,
        As=_As,
        gamma_MG=0.0,
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    matter_statistics = MatterStatistics(
        perturbations,
        # z=integ_ztrue_arr,
        k=np.geomspace(1e-4, 10, 500),
    )

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    CL = TwoPoint3DHaloClustering(matter_statistics, background_fid)
    _test_clustering(CL, perturbations)


def test_cosmo_photoz_rsd_correction():

    print("# Cosmology parameters")
    _cosmo_pars = dict(
        H0=67.7,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_b0=0.022 / 0.677**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    z_test = np.array([0.0, 1.0])
    lob_test = np.array([50.0])
    k_test = np.geomspace(1e-4, 10, 500)
    zobs_scatter = np.array([5.0, 5.1])

    background = CAMBBackground(**_cosmo_pars)

    corr0, corr1, corr2 = photoz_rsd_correction(
        background,
        z_test,
        k_test,
        zobs_scatter,
    )

    # test values

    ref_phz_rsd_0 = np.array(
        [[5.7111615e-01, 5.9122967e-06], [8.0073649e-01, 1.0336005e-05]]
    )
    ref_phz_rsd_1 = np.array(
        [[1.089998e-01, 1.384733e-16], [3.820448e-01, 1.228974e-15]]
    )
    ref_phz_rsd_2 = np.array(
        [[1.263118e-02, 2.432424e-27], [9.154732e-02, 1.095958e-25]]
    )

    assert_allclose(corr0[:, [0, -1]], ref_phz_rsd_0, rtol=1e-04)
    assert_allclose(corr1[:, [0, -1]], ref_phz_rsd_1, rtol=1e-04)
    assert_allclose(corr2[:, [0, -1]], ref_phz_rsd_2, rtol=1e-04)
