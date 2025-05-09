import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.covariance import HaloCovariance


def test_count_covariance():
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

    # counts covariance
    print("# counts covariance")

    area = 15000
    nbins_z = 10
    L = 20

    k_min = 1e-4
    k_max = 2e0
    k_div = 300

    zbins = np.linspace(0, 2, nbins_z + 1)
    k_test = np.geomspace(k_min, k_max, k_div)

    CC = HaloCovariance(perturbations, area, nbins_z, k_test, L)

    print("    Covariance coefficients")
    KL = CC.Kl_coeff()
    # All validation values have to be updated with extarnal values
    assert_allclose(
        KL[:5], [0.282095, 0.310942, 0.1095, -0.074565, -0.091053], rtol=5e-6
    )

    print("    Covariance window")
    iz = 0
    zarr_iz = np.linspace(zbins[iz], zbins[iz + 1], 31)
    # All validation values have to be updated with extarnal values
    assert_allclose(CC.cov_window(iz, zarr_iz, KL)[0,:5], np.array([0.99959593, 0.99956826, 0.9995387,  0.99950711, 0.99947336]),rtol=1e-6)
