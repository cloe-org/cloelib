# impoer jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology import derived_cosmology
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
        N_mnu=1,
    )

    # background
    background = CAMBBackground(**_cosmo_pars)

    # camb linear
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    assert_allclose(perturbations.matter_power_spectrum(0, 1), 80.534861)
    assert_allclose(perturbations.matter_power_spectrum_cb(0, 1), 81.748209, rtol=1e-03)

    # camb non-linear
    perturbations_nl = CAMBNonLinearPerturbations(
        background, np.linspace(0.0, 2.0, 100)
    )
    assert_allclose(
        perturbations_nl.matter_power_spectrum(0, 1), 736.010737, rtol=1.0e-03
    )
    assert_allclose(
        perturbations_nl.matter_power_spectrum_cb(0, 1), 747.017036, rtol=1.0e-03
    )
