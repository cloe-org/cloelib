import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.observables.clusters.halo_statistics import (
    HaloStatistics,
    HaloStatisticsTinker,
    HaloStatisticsCastro,
)
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations


def _test_halostatistics(HS, HS_tinker, HS_castro):
    z_test = np.linspace(0.01, 1.0, 20)
    k_test = np.logspace(-2, 1, 100)
    R_test = np.logspace(-1, 1, 20)
    M_test = np.logspace(14, 15, 50)

    print("    window")
    assert_allclose(HS.window(k_test, R_test), XXX)
    print("    radius_M")
    assert_allclose(HS.radius_M(M_test), XXX)
    print("    delta_c")
    assert_allclose(HS.delta_c(z_test), XXX)
    print("    get_Delta_crit")
    assert_allclose(HS.get_Delta_crit(z_test), XXX)
    print("    sigma_z_R")
    assert_allclose(HS.sigma_z_R(z_test, R_test), XXX)
    print("    sigma_z_M")
    assert_allclose(HS.sigma_z_M(z_test, M_test), XXX)
    print("    nu_z_M")
    assert_allclose(HS.nu_z_M(z_test, M_test), XXX)
    print("    dlns_dlnR")
    assert_allclose(HS.dlns_dlnR(z_test, M_test), XXX)

    assert_allclose(HS_tinker.bias(z_test, M_test), XXX)

    assert_allclose(HS_castro.dn_dm(z_test, M_test), XXX)
    assert_allclose(HS_castro.bias(z_test, M_test), XXX)


def test_halostatistics():
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

    # HaloStatistics
    print("# HaloStatistics")
    HS = HaloStatistics(perturbations, "vir")
    HS_tinker = HaloStatisticsTinker(perturbations, "vir")
    HS_castro = HaloStatisticsCastro(perturbations, "vir")
    _test_halostatistics(HS, HS_tinker, HS_castro)
