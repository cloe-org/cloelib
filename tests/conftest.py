"""Shared pytest fixtures used across more than one test module.

`cosmo_setup`/`linear_perturbations` live here (rather than being defined
in one test module and imported into another) specifically so
`test_tatt.py` and `test_tatt_m.py` can share them with no import at all -
pytest auto-discovers fixtures in `conftest.py` for every test module in
this directory. Importing a fixture function by name instead (the
previous approach) makes every test function that takes it as a
parameter look, to `ruff`, like it's shadowing an unused import (F811) -
a parameter name coincidentally matching a fixture isn't something a
linter can distinguish from an actual bug, so that approach needed a
file-level `noqa`. Defining the fixture here avoids the import
(and the lint problem) entirely.
"""

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
    """A *linear* Perturbations object, as `PBJTATTLoopComputer` requires
    (FAST-PT's one-loop integrals are only valid starting from the linear
    Pk - see its docstring) - `cosmo_setup`'s own `perturbations` is
    nonlinear.
    """
    perturbations, z = cosmo_setup
    return CAMBLinearPerturbations(perturbations.background, perturbations.z)
