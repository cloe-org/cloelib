"""Tests for the kernel conventions of halo-model analyses (no GODMAX needed)."""

import pytest
from numpy.testing import assert_allclose

from cloelib.observables.kernel_conventions import (
    GODMAX_BEAM_FWHM_ARCMIN,
    GODMAX_CMB_SOURCE_REDSHIFT,
    PRESETS,
    kernel_convention,
)

NUISANCE = {
    "multiplicative_bias_1": 0.01,
    "dz_shear_1": 0.02,
    "width_shear_1": 1.1,
    "dz_pos_1": 0.03,
    "width_pos_1": 0.9,
    "magnification_bias_1": 0.8,
    "AIA": 0.5,
    "EtaIA": 1.0,
    "CIA": 0.0134,
    "b1_photo_bin0": 1.5,
}


def test_godmax_preset():
    convention = kernel_convention("godmax")
    assert convention.shear_options() == {"ia_model": "NLA", "spin2_prefactor": False}
    assert convention.positions_options() == {
        "galaxy_bias_model": "hod",
        "include_rsd": False,
    }
    assert convention.cmb_options() == {"z_source": GODMAX_CMB_SOURCE_REDSHIFT}
    assert convention.tsz_options() == {"beam_fwhm_arcmin": GODMAX_BEAM_FWHM_ARCMIN}

    params = convention.nuisance(NUISANCE)
    assert params is not NUISANCE
    assert params["magnification_bias_1"] == 0.0
    assert params["dz_pos_1"] == 0.0
    assert params["width_pos_1"] == 1.0
    assert params["width_shear_1"] == 1.0
    # Kept in both codes.
    assert params["dz_shear_1"] == NUISANCE["dz_shear_1"]
    assert params["multiplicative_bias_1"] == NUISANCE["multiplicative_bias_1"]
    # NLA with pivot: A (1+z)^eta (1+z0)^-eta, default z0 = 0.62.
    assert_allclose(params["AIA"], 0.5 * 1.62**-1.0, rtol=1e-14)
    assert_allclose(
        convention.nuisance(dict(NUISANCE, z0IA=0.3))["AIA"], 0.5 / 1.3, rtol=1e-14
    )


def test_baseline_preset_keeps_nuisances():
    convention = kernel_convention("baseline")
    assert convention.nuisance(NUISANCE) == NUISANCE
    assert convention.shear_options() == {"ia_model": "NLA", "spin2_prefactor": True}
    assert convention.positions_options() == {
        "galaxy_bias_model": "poly",
        "include_rsd": True,
    }
    assert convention.cmb_options() == {"z_source": None}
    assert convention.tsz_options() == {"beam_fwhm_arcmin": 0.0}


def test_overrides():
    convention = kernel_convention(
        "godmax", magnification=True, rsd=True, galaxy_bias_model="per_bin"
    )
    assert convention.positions_options() == {
        "galaxy_bias_model": "per_bin",
        "include_rsd": True,
    }
    assert convention.nuisance(NUISANCE)["magnification_bias_1"] == 0.8
    assert (
        kernel_convention("godmax", ia_model=None).shear_options()["ia_model"] is None
    )
    # Overrides do not change the presets.
    assert PRESETS["godmax"].magnification is False


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        ({"ia_model": "TATT"}, NotImplementedError, "TATT"),
        ({"galaxy_bias_model": "nonlinear"}, NotImplementedError, "nonlinear"),
        ({"galaxy_bias_model": "linear"}, ValueError, "Unknown galaxy_bias_model"),
        ({"ia_model": "LA"}, ValueError, "Unknown ia_model"),
        ({"y_beam_fwhm_arcmin": -1.0}, ValueError, "non-negative"),
        ({"magnifcation": True}, ValueError, "Unknown kernel-convention options"),
    ],
)
def test_invalid_conventions_raise(overrides, error, match):
    with pytest.raises(error, match=match):
        kernel_convention("godmax", **overrides)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        kernel_convention("cloe")
