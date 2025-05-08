import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.clustering import HaloClustering

def _test_clustering(CL):
    z_test = np.linspace(0.0, 2.0, 20)
    r_test = np.geomspace(20, 150, 30)
    lob_test = np.logspace(0.2, 2.2, 20)

    print("    APcorr_func")
    assert_allclose(CL.APcorr_func(z_test), XXX)
    print("    WF_ra")
    assert_allclose(CL.WF_ra(z_test, r_test), XXX)

    print("    Pk_IR_func")
    Pk_test = perturbations.matter_power_spectrum(
        z_test, CL.k, hubble_units=True, k_hunit=True
    )
    assert_allclose(CL.Pk_IR_func(Pk_test), XXX)

    print("    photoz_rsd_correction")
    sigma_zob = SF.scatter_zobs_z(lob_test, z_test)
    assert_allclose(CL.photoz_rsd_correction(z_test, sigma_zob), XXX)


def test_clustering():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.0,
        As=2e-9,
        gamma_MG=0.0,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))


    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    perturbations_fid = CAMBLinearPerturbations(
        background_fid, np.linspace(0.0, 2.0, 100)
    )
    k_min = 1e-4
    k_max = 2e0
    k_div = 300
    nonu = True
    CL = HaloClustering(perturbations, perturbations_fid, nonu, k_div, k_min, k_max)
    _test_clustering(CL)
