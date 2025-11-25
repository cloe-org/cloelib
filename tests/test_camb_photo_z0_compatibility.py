"""
Unit tests to verify CAMB automatic z=0 inclusion is compatible with photo tracers.

These tests ensure that:
1. CAMB perturbations automatically include z=0 for sigma8 computation
2. Photo tracers (ShearTracer, PositionsTracer) work with such perturbations
3. No numerical errors or collisions occur when computing observables
"""

import pytest
import numpy as np

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo import ShearTracer


@pytest.fixture
def camb_photo_setup():
    """Create CAMB perturbations and photo tracer setup without z=0 in user input."""
    # Create background (using same params as test_camb_cosmology.py)
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2

    background = CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )

    # Tracer setup (without z=0) - for Limber integration
    tracer_z = np.linspace(0.2, 2.0, 40)

    # User redshifts WITHOUT z=0 (z=0 should be added automatically)
    # Use same grid as tracer for compatibility
    user_z = tracer_z
    perturbations = CAMBNonLinearPerturbations(background, user_z)
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(tracer_z)))
    dndz /= np.trapz(dndz, tracer_z, axis=1)[:, None]

    return perturbations, tracer_z, dndz, n_z_bins


def test_no_collision_and_power_spectrum_at_all_z(camb_photo_setup):
    """
    Test that perturbations.z and tracer.z remain separate without collision.

    This test verifies:
    1. CAMB perturbations have z=0 (for sigma8 computation)
    2. Tracer arrays do NOT have z=0 (for Limber integration)
    3. Power spectrum can be computed at all perturbations.z including z=0
    4. No numerical errors occur despite different z arrays
    """
    perturbations, tracer_z, dndz, n_z_bins = camb_photo_setup

    # Create nuisance parameters
    nuisance_params = {
        **{f"multiplicative_bias_{i+1}": 0.0 for i in range(n_z_bins)},
        **{f"dz_shear_{i+1}": 0.0 for i in range(n_z_bins)},
        "AIA": 1.0,
        "CIA": 0.0164,
        "EtaIA": -0.41,
    }

    # Create shear tracer
    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=tracer_z,
        nuisance_params=nuisance_params,
    )

    # Verify z array separation
    assert 0.0 not in shear_tracer.z, "Tracer z should NOT have z=0"
    assert shear_tracer.z.min() > 0.0, "All tracer z values should be > 0"
    assert (
        0.0 in shear_tracer.perturbations.z
        or shear_tracer.perturbations.z.min() < 0.001
    ), "Perturbations z should have z=0"
    assert len(shear_tracer.perturbations.z) == len(shear_tracer.z) + 1, (
        "Perturbations should have exactly one more redshift point (z=0)"
    )

    # Verify arrays are actually different
    assert not np.array_equal(
        shear_tracer.z, shear_tracer.perturbations.z
    ), "Tracer and perturbations z arrays should be different"

    # Now verify that power spectrum can be computed at ALL perturbations.z (including z=0)
    ks = np.logspace(-3, 1, 20)  # Wavenumber grid

    # Compute power spectrum at all perturbations redshifts
    pk_all_z = perturbations.matter_power_spectrum(perturbations.z, ks)

    # Verify power spectrum output shape and values
    assert pk_all_z.shape[0] == len(perturbations.z), "P(k,z) should have row for each z"
    assert pk_all_z.shape[1] == len(ks), "P(k,z) should have column for each k"
    assert np.all(np.isfinite(pk_all_z)), "All P(k,z) values should be finite"
    assert np.all(pk_all_z > 0), "All P(k,z) values should be positive"

    # Specifically check z=0 power spectrum (first row)
    pk_z0 = pk_all_z[0, :]
    assert np.all(np.isfinite(pk_z0)), "P(k, z=0) should be finite"
    assert np.all(pk_z0 > 0), "P(k, z=0) should be positive"

    # Verify power spectrum decreases with increasing z (at fixed k)
    # P(k, z=0) should be larger than P(k, z>0) due to growth
    for i in range(1, min(5, len(perturbations.z))):
        assert np.all(pk_all_z[0, :] > pk_all_z[i, :]), (
            f"P(k, z=0) should be > P(k, z={perturbations.z[i]:.2f})"
        )
