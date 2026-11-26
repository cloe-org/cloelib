"""
Unit tests to verify cmb tracers run correctly.
"""

import pytest
import numpy as np

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from cosmolib.data import AngularPowerSpectrum


@pytest.fixture
def camb_cmb_setup():
    """Create CAMB perturbations and position/shear tracers."""
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


def test_cmb_lensing_window_cl(camb_cmb_setup):
    """
    Test that cmb lensing window function auto and cross power spectra can be computed without error.

    It further verifies that:
    1. CMB lensing window function has the expected shape.
    2. CMB lensing power spectrum and cross-correlations have the expected shapes.
    """
    perturbations, z_auto, z_cross, dndz = camb_cmb_setup

    # Create nuisance parameters
    nuisance_pos = {
        "b1_photo_bin1": 1.0,
        "dz_pos_1": 0.0,
        "width_pos_1": 1.0,
        "magnification_bias_1": 0.0,
    }
    nuisance_shear = {
        "AIA": 0.0,
        "CIA": 0.0,
        "EtaIA": 0.0,
        "multiplicative_bias_1": 0.0,
        "dz_shear_1": 0.0,
        "width_shear_1": 1.0,
    }

    # Create CMB lensing tracers
    cmblens_cross = CMBLensingTracer(perturbations=perturbations, z=z_cross)
    cmblens_auto = CMBLensingTracer(perturbations=perturbations, z=z_auto)

    window_cross = cmblens_cross.get_window(z_cross)
    window_auto = cmblens_auto.get_window(z_auto)

    # Check that the window functions have the expected shape
    assert window_cross.shape == (1, len(z_cross))
    assert window_auto.shape == (1, len(z_auto))

    # Create position tracer
    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z_cross,
        galaxy_bias_model="per_bin",
        nuisance_params=nuisance_pos,
    )

    # Create shear tracer
    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z_cross,
        nuisance_params=nuisance_shear,
    )

    nl = 10
    lmin = 10
    lmax = 100
    bins = np.linspace(lmin, lmax, nl)
    ells = 0.5 * (bins[0:-1] + bins[1:])
    ells_full = np.arange(lmax)

    twopoint_kpos = AngularTwoPoint(cmblens_cross, pos_tracer)
    twopoint_kshe = AngularTwoPoint(cmblens_cross, shear_tracer)
    twopoint_kk = AngularTwoPoint(cmblens_auto, cmblens_auto)

    cells = {
        **twopoint_kpos.get_Cl(ells_full, 0, perturbations.k),
        **twopoint_kshe.get_Cl(ells_full, 0, perturbations.k),
        **twopoint_kk.get_Cl(ells_full, 0, perturbations.k),
    }

    # Check that cells has the right number of elements and right keys
    assert len(cells.keys()) == 3
    assert ("CMBL", "POS", 1, 1) in cells.keys()
    assert ("CMBL", "SHE", 1, 1) in cells.keys()
    assert ("CMBL", "CMBL", 1, 1) in cells.keys()
    assert cells[("CMBL", "POS", 1, 1)].shape == (lmax,)
    assert cells[("CMBL", "SHE", 1, 1)].shape == (2, lmax)
    assert cells[("CMBL", "CMBL", 1, 1)].shape == (lmax,)

    # We also check everything works well if we reverse the order of the probes
    twopoint_posk = AngularTwoPoint(pos_tracer, cmblens_cross)
    twopoint_shek = AngularTwoPoint(shear_tracer, cmblens_cross)
    cells_reverse = {
        **twopoint_posk.get_Cl(ells_full, 0, perturbations.k),
        **twopoint_shek.get_Cl(ells_full, 0, perturbations.k),
    }

    assert len(cells_reverse.keys()) == 2
    assert ("CMBL", "POS", 1, 1) in cells_reverse.keys()
    assert ("CMBL", "SHE", 1, 1) in cells_reverse.keys()
    assert cells_reverse[("CMBL", "POS", 1, 1)].shape == (lmax,)
    assert cells_reverse[("CMBL", "SHE", 1, 1)].shape == (2, lmax)

    # Check pseudo-cl
    rng = np.random.default_rng(42)
    mixing_matrix = {
        key: AngularPowerSpectrum(
            array=rng.random((nl-1, lmax)),
            ell=ells,
            lower=bins[0:-1],
            upper=bins[1:],
        )
        for key in [
            ("CMBL", "POS", 1, 1),
            ("CMBL", "SHE", 1, 1),
            ("CMBL", "CMBL", 1, 1),
        ]
    }

    pseudo_cells = {
        **twopoint_kpos.get_pseudo_Cl(0, perturbations.k, mixing_matrix),
        **twopoint_kshe.get_pseudo_Cl(0, perturbations.k, mixing_matrix),
        **twopoint_kk.get_pseudo_Cl(0, perturbations.k, mixing_matrix),
    }

    # Check that pseudo_cells has the right number of elements and right keys
    assert len(pseudo_cells.keys()) == 3
    assert ("CMBL", "POS", 1, 1) in pseudo_cells.keys()
    assert ("CMBL", "SHE", 1, 1) in pseudo_cells.keys()
    assert ("CMBL", "CMBL", 1, 1) in pseudo_cells.keys()
    assert pseudo_cells[("CMBL", "POS", 1, 1)].shape == (nl-1,)
    assert pseudo_cells[("CMBL", "SHE", 1, 1)].shape == (2, nl-1)
    assert pseudo_cells[("CMBL", "CMBL", 1, 1)].shape == (nl-1,)

    # Check that pseudo_cells prediction agree with full cells convoluted with mixing matrix
    for key in [("CMBL", "POS", 1, 1), ("CMBL", "CMBL", 1, 1)]:
        np.testing.assert_allclose(
            pseudo_cells[key].array,
            mixing_matrix[key].array @ cells[key].array,
            rtol=1e-10,
        )
        
    key = ("CMBL", "SHE", 1, 1)
    for idx in [0, 1]:
        np.testing.assert_allclose(
            pseudo_cells[key].array[idx],
            mixing_matrix[key].array @ cells[key].array[idx],
            rtol=1e-10,
        )
    
    
