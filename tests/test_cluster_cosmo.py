import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)


def test_cosmo():
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
        mnu=0.1,
        As=2e-9,
        gamma_MG=0.0,
    )

    # background
    background = CAMBBackground(**_cosmo_pars)
    assert background.Omega_m(0, nonu=True) < background.Omega_m(0, nonu=False)
    assert_allclose(
        background.Omega_m(0, nonu=True),
        _cosmo_pars["Omega_cdm0"] + _cosmo_pars["Omega_b0"],
        rtol=1e-07,
    )
    assert_allclose(background.rho_crit(0), 1.27203085e11, rtol=1e-07)
    assert_allclose(background.rdrag(), 147.50225, rtol=1e-05)

    # camb linear
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    assert_allclose(perturbations.matter_power_spectrum(0, 1), 80.527367)
    assert_allclose(perturbations.matter_power_spectrum(0, 1, nonu=True), 81.748209)

    # camb non-linear
    perturbations_nl = CAMBNonLinearPerturbations(
        background, np.linspace(0.0, 2.0, 100)
    )
    assert_allclose(perturbations_nl.matter_power_spectrum(0, 1), 735.860906)
    assert_allclose(perturbations_nl.matter_power_spectrum(0, 1, nonu=True), 747.017036)
