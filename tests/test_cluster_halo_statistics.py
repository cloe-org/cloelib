import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.observables.clusters.halo_statistics import (
    HaloStatistics,
    HaloStatisticsTinker,
    HaloStatisticsCastro,
)
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations


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

    # tests
    z_test = np.linspace(0.01, 1.0, 5)
    k_test = np.logspace(-2, 1, 5)
    R_test = np.logspace(-1, 1, 5)
    M_test = np.logspace(14, 15, 5)

    print("    window")
    W, dWdx = HS.window(k_test, R_test)
    _ref = [9.999999e-01, 9.999968e-01, 9.999000e-01, 9.968413e-01, 9.035060e-01]
    assert_allclose(W[0], _ref)
    _ref = [-0.0002, -0.001125, -0.006324, -0.035485, -0.186105]
    assert_allclose(dWdx[0], _ref, atol=5e-07)
    print("    radius_M")
    _ref = [6.523691, 7.903632, 9.575468, 11.600945, 14.054865]
    assert_allclose(HS.radius_M(M_test), _ref)
    print("    delta_c")
    _ref = [1.6761, 1.679699, 1.681938, 1.683347, 1.684253]
    assert_allclose(HS.delta_c(z_test), _ref, rtol=5e-7)
    print("    get_Delta_crit")
    _ref = [103.350844, 123.433339, 139.138379, 150.283152, 157.909004]
    assert_allclose(HS.get_Delta_crit(z_test)[:5], _ref)
    print("    sigma_z_R")
    _ref = [5.025379, 3.622691, 2.402038, 1.406789, 0.683393]
    assert_allclose(HS.sigma_z_R(z_test, R_test)[0, :5], _ref, rtol=5e-7)
    print("    sigma_z_M")
    _ref = [0.917697, 0.807373, 0.705316, 0.611551, 0.526045]
    assert_allclose(HS.sigma_z_M(z_test, M_test)[0], _ref, rtol=1e-6)
    print("    nu_z_M")
    _ref = [1.826419, 2.075992, 2.376381, 2.740734, 3.186231]
    assert_allclose(HS.nu_z_M(z_test, M_test)[0], _ref, rtol=5e-7)
    print("    dlns_dlnR")
    _ref = [-0.649908, -0.685486, -0.723499, -0.763745, -0.806543]
    assert_allclose(HS.dlns_dlnR(z_test, M_test)[0], _ref, rtol=1e-6)

    print("    bias Tinker")
    _ref = [2.170256, 2.699147, 3.452148, 4.543852, 6.157447]
    assert_allclose(HS_tinker.bias(z_test, M_test)[0], _ref, rtol=5e-7)

    print("    dn_dm Castro")
    _ref = [3.963046e-19, 9.968662e-20, 2.154962e-20, 3.719759e-21, 4.585653e-22]
    assert_allclose(HS_castro.dn_dm(z_test, M_test)[0], _ref, rtol=5e-7)

    print("    bias Castro")
    _ref = [2.160949, 2.681549, 3.414679, 4.450279, 5.896728]
    assert_allclose(HS_castro.bias(z_test, M_test)[0], _ref, rtol=5e-7)
