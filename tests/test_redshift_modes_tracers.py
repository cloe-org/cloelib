"""Tests for redshift-mode systematics in photo tracers."""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo import PositionsTracer, ShearTracer


@pytest.fixture(scope="module")
def modes_tracers():
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

    modes = np.zeros((2, n_z_bins, len(z)))
    modes[0, 0] = z - z.mean()
    modes[1, 1] = z - z.mean()

    def build_tracers(shear_amplitudes, pos_amplitudes=None):
        if pos_amplitudes is None:
            pos_amplitudes = shear_amplitudes

        shear_params = {
            f"dndz_mode_shear_{k + 1}": amplitude
            for k, amplitude in enumerate(shear_amplitudes)
        }

        pos_params = {
            f"dndz_mode_pos_{k + 1}": amplitude
            for k, amplitude in enumerate(pos_amplitudes)
        }

        shear_tracer = ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            nuisance_params={
                **shear_params,
                **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
                "AIA": 1.0,
                "CIA": 0.0134,
                "EtaIA": -0.41,
            },
            ia_model="NLA",
            dndz_redshift_model="modes",
            dndz_modes=modes,
        )

        pos_tracer = PositionsTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params={
                **pos_params,
                **{f"magnification_bias_{i + 1}": 0.3 for i in range(n_z_bins)},
                "b1_photo_bin0": 1.1,
                "b1_photo_bin1": 1.4,
            },
            dndz_redshift_model="modes",
            dndz_modes=modes,
        )

        return shear_tracer, pos_tracer

    # Default tracers with zero mode amplitudes, for existing tests.
    shear_tracer, pos_tracer = build_tracers([0.0, 0.0])

    return {
        "z": z,
        "dndz": dndz,
        "modes": modes,
        "shear_tracer": shear_tracer,
        "pos_tracer": pos_tracer,
        "build_tracers": build_tracers,
    }


@pytest.mark.parametrize("tracer_name", ["shear_tracer", "pos_tracer"])
def test_zero_mode_amplitudes_reproduce_fiducial_dndz(modes_tracers, tracer_name):
    tracer = modes_tracers[tracer_name]
    dndz = modes_tracers["dndz"]

    np.testing.assert_allclose(
        np.asarray(tracer.dndz_with_systematics),
        dndz,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("tracer_name", ["shear_tracer", "pos_tracer"])
def test_mode_amplitudes_change_only_their_own_bin(modes_tracers, tracer_name):
    z = modes_tracers["z"]
    dndz = modes_tracers["dndz"]
    modes = modes_tracers["modes"]

    amplitudes = np.array([0.2, 0.0])
    expected = dndz + np.einsum("k,kiz->iz", amplitudes, modes)
    expected = np.maximum(expected, 0.0)
    expected /= np.trapezoid(expected, z, axis=1)[:, None]

    shear_tracer, pos_tracer = modes_tracers["build_tracers"](amplitudes, amplitudes)
    tracer = {
        "shear_tracer": shear_tracer,
        "pos_tracer": pos_tracer,
    }[tracer_name]

    np.testing.assert_allclose(
        np.asarray(tracer.dndz_with_systematics),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    # The second bin should remain unchanged.
    np.testing.assert_allclose(
        np.asarray(tracer.dndz_with_systematics)[1],
        dndz[1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_shear_and_positions_modes_are_independent(modes_tracers):
    shear_tracer, pos_tracer = modes_tracers["build_tracers"]([0.2, 0.0], [0.0, 0.0])

    np.testing.assert_allclose(
        np.asarray(pos_tracer.dndz_with_systematics),
        modes_tracers["dndz"],
    )

    assert not np.allclose(
        np.asarray(shear_tracer.dndz_with_systematics),
        modes_tracers["dndz"],
    )
