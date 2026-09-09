"""
Tests for the Contribution decomposition (see `CONTRIBUTION_ARCHITECTURE.md`).

`test_photo_tracers_characterization.py` already proves `get_window()`'s
output is unchanged bit-for-bit. These tests instead pin the seam itself:
that `get_contributions()` returns objects whose `compute_kernel` matches
the original component methods exactly, and that a tracer's window really
is their sum - the property `ia_model="TATT"` and any future non-linear
bias contribution rely on when they swap one contribution for another.
"""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.positions import (
    GalaxyBiasContribution,
    MagnificationContribution,
)
from cloelib.observables.photo.shear import (
    IntrinsicAlignmentContribution,
    LensingContribution,
)


@pytest.fixture(scope="module")
def tracers():
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

    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params={
            **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
            **{f"dz_shear_{i + 1}": 0.0 for i in range(n_z_bins)},
            **{f"width_shear_{i + 1}": 1.0 for i in range(n_z_bins)},
            "AIA": 1.0,
            "CIA": 0.0134,
            "EtaIA": -0.41,
        },
    )
    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params={
            **{f"dz_pos_{i + 1}": 0.0 for i in range(n_z_bins)},
            **{f"width_pos_{i + 1}": 1.0 for i in range(n_z_bins)},
            **{f"magnification_bias_{i + 1}": 0.3 for i in range(n_z_bins)},
            "b1_photo_bin0": 1.1,
            "b1_photo_bin1": 1.4,
        },
    )
    return {"z": z, "shear_tracer": shear_tracer, "pos_tracer": pos_tracer}


def test_shear_tracer_contributions_match_component_methods(tracers):
    tracer = tracers["shear_tracer"]
    z = tracers["z"]

    contributions = tracer.get_contributions()
    assert len(contributions) == 2
    assert isinstance(contributions[0], LensingContribution)
    assert isinstance(contributions[1], IntrinsicAlignmentContribution)

    np.testing.assert_array_equal(
        np.asarray(contributions[0].compute_kernel(z)),
        np.asarray(tracer.get_window_lensing(z)),
    )
    np.testing.assert_array_equal(
        np.asarray(contributions[1].compute_kernel(z)),
        np.asarray(tracer.get_window_IA(z)),
    )


def test_shear_tracer_window_is_sum_of_contributions(tracers):
    tracer = tracers["shear_tracer"]
    z = tracers["z"]

    kernel_sum = sum(c.compute_kernel(z) for c in tracer.get_contributions())
    expected = kernel_sum * (1 + np.array(tracer.m_bias)[:, None])

    np.testing.assert_array_equal(
        np.asarray(tracer.get_window(z)), np.asarray(expected)
    )


def test_positions_tracer_contributions_match_component_methods(tracers):
    tracer = tracers["pos_tracer"]
    z = tracers["z"]

    contributions = tracer.get_contributions()
    assert len(contributions) == 2
    assert isinstance(contributions[0], GalaxyBiasContribution)
    assert isinstance(contributions[1], MagnificationContribution)

    np.testing.assert_array_equal(
        np.asarray(contributions[0].compute_kernel(z)),
        np.asarray(tracer.get_window_positions(z)),
    )
    np.testing.assert_array_equal(
        np.asarray(contributions[1].compute_kernel(z)),
        np.asarray(tracer.get_window_magnification(z)),
    )


def test_positions_tracer_window_is_sum_of_contributions(tracers):
    tracer = tracers["pos_tracer"]
    z = tracers["z"]

    expected = sum(c.compute_kernel(z) for c in tracer.get_contributions())

    np.testing.assert_array_equal(
        np.asarray(tracer.get_window(z)), np.asarray(expected)
    )
