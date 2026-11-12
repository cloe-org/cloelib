"""Fixtures shared by `test_tatt.py` and `test_tatt_m.py`."""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)


@pytest.fixture(scope="module")
def cosmo_setup():
    H0 = 67.7
    h = H0 / 100.0
    background = CAMBBackground(
        H0=H0,
        Omega_b0=0.022 / h**2,
        Omega_cdm0=0.12 / h**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        alpha_s=0.0,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )
    z_auto = np.linspace(0.01, 1100.0, 100)
    z = np.linspace(0.2, 2.0, 15)
    perturbations = CAMBNonLinearPerturbations(background, None, z_auto)
    return perturbations, z


@pytest.fixture(scope="module")
def linear_perturbations(cosmo_setup):
    """Linear perturbations, as `PBJTATTLoopComputer` requires."""
    perturbations, z = cosmo_setup
    return CAMBLinearPerturbations(perturbations.background, perturbations.z)
