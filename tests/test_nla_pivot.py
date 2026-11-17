"""Tests for the NLA intrinsic-alignment pivot redshift (`z0IA`, gh-709).

`ShearTracer.get_window_IA` scales the NLA amplitude as
``((1+z)/(1+z0IA))**EtaIA``, so ``AIA`` is the amplitude at the pivot
redshift. ``z0IA`` is optional and defaults to 0.0, which reproduces the
previous ``(1+z)**EtaIA`` behaviour exactly.
"""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo import ShearTracer

N_Z_BINS = 2
ETA_IA = -0.41


@pytest.fixture(scope="module")
def setup():
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
    z = np.linspace(0.2, 2.0, 20)
    perturbations = CAMBNonLinearPerturbations(
        background, None, np.linspace(0.01, 1100.0, 100)
    )
    dndz = np.ones((N_Z_BINS, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    return perturbations, dndz, z


def _nla_window(setup, **ia_params):
    perturbations, dndz, z = setup
    nuisance = {
        **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(N_Z_BINS)},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(N_Z_BINS)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(N_Z_BINS)},
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": ETA_IA,
        **ia_params,
    }
    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=nuisance,
        ia_model="NLA",
    )
    return np.asarray(tracer.get_window_IA(z))


def test_default_pivot_is_zero(setup):
    """Omitting `z0IA` is identical to `z0IA=0.0` (backward compatible)."""
    np.testing.assert_array_equal(_nla_window(setup), _nla_window(setup, z0IA=0.0))


@pytest.mark.parametrize("z0", [0.3, 0.62, 1.0])
def test_pivot_equivalent_to_amplitude_rescaling(setup, z0):
    """A pivot z0 is equivalent to rescaling AIA by (1+z0)**(-EtaIA)."""
    with_pivot = _nla_window(setup, AIA=1.0, z0IA=z0)
    rescaled = _nla_window(setup, AIA=(1.0 + z0) ** (-ETA_IA))
    np.testing.assert_allclose(with_pivot, rescaled, rtol=1e-12, atol=0.0)
    # And it actually changes the kernel when EtaIA != 0.
    assert not np.allclose(with_pivot, _nla_window(setup))


def test_pivot_irrelevant_without_redshift_evolution(setup):
    """With EtaIA = 0 the pivot has no effect."""
    np.testing.assert_array_equal(
        _nla_window(setup, EtaIA=0.0, z0IA=0.62),
        _nla_window(setup, EtaIA=0.0),
    )
