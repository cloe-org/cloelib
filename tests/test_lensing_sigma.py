"""Tests of the modified-gravity lensing parameter Sigma(z) in the lensing kernels.

Sigma multiplies every lensing kernel (cosmic shear, magnification bias and CMB
lensing) when the perturbations provide a ``Sigma`` method, and defaults to one
otherwise. These tests are backend-agnostic: any perturbations implementation
exposing ``Sigma`` (e.g. the MG boost module) can reuse ``_WithSigma`` here.
"""

import pytest
import numpy as np

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.photo import ShearTracer, PositionsTracer


@pytest.fixture
def lensing_setup():
    """CAMB perturbations plus the redshift grids and n(z) used by the lensing tracers."""
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
        alpha_s=0.0,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )

    z_auto = np.linspace(0.01, 1100.0, 100)
    z_cross = np.linspace(0.2, 2.0, 40)
    perturbations = CAMBNonLinearPerturbations(background, None, z_auto)
    n_z_bins = 1
    dndz = np.ones((n_z_bins, len(z_cross)))
    dndz /= np.trapezoid(dndz, z_cross, axis=1)[:, None]
    return perturbations, z_auto, z_cross, dndz


class _WithSigma:
    """Proxy adding a redshift-dependent MG lensing parameter Sigma(z) to a backend."""

    def __init__(self, perturbations, sigma):
        self._perturbations = perturbations
        self._sigma = sigma

    def __getattr__(self, name):
        return getattr(self._perturbations, name)

    def Sigma(self, z):
        return self._sigma(z)


def test_sigma_scales_all_lensing_kernels(lensing_setup):
    """Sigma(z) multiplies the magnification-bias and CMB-lensing kernels exactly as
    it does the shear kernel; without Sigma the kernels are unchanged (Sigma = 1)."""
    perturbations, z_auto, z_cross, dndz = lensing_setup
    sigma = lambda z: 1.0 + 0.05 * np.asarray(z)  # noqa: E731
    nuisance_pos = {
        "b1_photo_bin1": 1.0,
        "dz_pos_1": 0.0,
        "width_pos_1": 1.0,
        "magnification_bias_1": 0.5,
    }
    nuisance_shear = {
        "AIA": 0.0,
        "CIA": 0.0,
        "EtaIA": 0.0,
        "multiplicative_bias_1": 0.0,
        "dz_shear_1": 0.0,
        "width_shear_1": 1.0,
    }
    gr = perturbations
    mg = _WithSigma(perturbations, sigma)

    pos_gr = PositionsTracer(gr, dndz, z_cross, "per_bin", nuisance_pos)
    pos_mg = PositionsTracer(mg, dndz, z_cross, "per_bin", nuisance_pos)
    np.testing.assert_allclose(
        np.asarray(pos_mg.get_window_magnification(z_cross)),
        np.asarray(pos_gr.get_window_magnification(z_cross)) * sigma(z_cross)[None, :],
        rtol=1e-10,
    )

    cmb_gr = CMBLensingTracer(gr, z_auto)
    cmb_mg = CMBLensingTracer(mg, z_auto)
    np.testing.assert_allclose(
        np.asarray(cmb_mg.get_window(z_auto)),
        np.asarray(cmb_gr.get_window(z_auto)) * sigma(z_auto)[None, :],
        rtol=1e-10,
    )

    shear_gr = ShearTracer(gr, dndz, z_cross, nuisance_shear, ia_model="NLA")
    shear_mg = ShearTracer(mg, dndz, z_cross, nuisance_shear, ia_model="NLA")
    np.testing.assert_allclose(
        np.asarray(shear_mg.get_window_lensing(z_cross)),
        np.asarray(shear_gr.get_window_lensing(z_cross)) * sigma(z_cross)[None, :],
        rtol=1e-10,
    )
