# import jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.auxiliary.halo_helpers import convert_to_Delta_crit
from cloelib.observables.halos.halo_abundance import (
    CastroHaloAbundance,
    TinkerHaloAbundance,
)
from cloelib.observables.halos.halo_model_properties import HaloModelProperties


def test_HaloModelProperties():
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
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    # HaloModelProperties
    print("# HaloModelProperties")
    HS = HaloModelProperties(perturbations)
    HS_tinker = TinkerHaloAbundance(HS)
    HS_castro = CastroHaloAbundance(HS)

    # tests
    z_test = np.linspace(0.01, 1.0, 5)
    k_test = np.logspace(-2, 1, 5)
    R_test = np.logspace(-1, 1, 5)
    M_test = np.logspace(14, 15, 5)

    print("    window")
    W, dWdx = HS_tinker.window(k_test, R_test)
    _ref = [9.999999e-01, 9.999968e-01, 9.999000e-01, 9.968413e-01, 9.035060e-01]
    assert_allclose(W[0], _ref)
    _ref = [-0.0002, -0.001125, -0.006324, -0.035485, -0.186105]
    assert_allclose(dWdx[0], _ref, atol=5e-07)
    print("    radius_M")
    _ref = [6.523691, 7.903632, 9.575468, 11.600945, 14.054865]
    assert_allclose(HS_tinker.radius_M(M_test), _ref)
    print("    delta_c")
    _ref = [1.676099, 1.67969, 1.681921, 1.683324, 1.684226]
    assert_allclose(HS_tinker.delta_c(z_test), _ref, rtol=5e-7)
    print("    convert_to_Delta_crit")
    _ref = [103.349057, 123.372358, 139.008183, 150.092972, 157.673498]
    assert_allclose(
        convert_to_Delta_crit("vir", background=background, z=z_test)[:5], _ref
    )
    print("    sigma_z_R")
    _ref = [4.175587, 3.550908, 2.366934, 1.386543, 0.673829]
    assert_allclose(HS_tinker.sigma_z_R(z_test, R_test)[0, :5], _ref, rtol=5e-3)
    print("    sigma_z_M")
    _ref = [0.908559, 0.799362, 0.698348, 0.605542, 0.520909]
    assert_allclose(HS_tinker.sigma_z_M(z_test, M_test)[0], _ref, rtol=1e-3)
    print("    nu_z_M")
    _ref = [1.851075, 2.103741, 2.407783, 2.776472, 3.227135]
    assert_allclose(HS_tinker.nu_z_M(z_test, M_test)[0], _ref, rtol=5e-3)
    print("    dlns_dlnR")
    _ref = [-0.649273, -0.684757, -0.722659, -0.76278, -0.805423]
    assert_allclose(3 * HS_tinker.dlns_dlnM(z_test, M_test)[0], _ref, rtol=1e-3)

    print("    bias Tinker")
    _ref = [2.205833, 2.746553, 3.516339, 4.632278, 6.281516]
    assert_allclose(HS_tinker.bias(z_test, M_test)[0], _ref, rtol=5e-3)

    print("    dn_dm Castro")
    _ref = [3.887839e-19, 9.699365e-20, 2.072788e-20, 3.521802e-21, 4.244946e-22]
    assert_allclose(HS_castro.dn_dm(z_test, M_test)[0], _ref, rtol=5e-3)

    print("    bias Castro")
    _ref = [2.196687, 2.727811, 3.475523, 4.531292, 6.005312]
    assert_allclose(HS_castro.bias(z_test, M_test)[0], _ref, rtol=5e-3)
