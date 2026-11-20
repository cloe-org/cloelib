"""Tests for the validation of the GODMAX settings that run without GODMAX.

`_galaxy_config` and `build_godmax_config` only translate the settings, and the
baryon-boost check runs before GODMAX is imported.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from cloelib.cosmology.cosmology import with_baryon_boost
from cloelib.cosmology.godmax_cosmology import (
    _OWNED_KEYS,
    GODMAXNonLinearPerturbations,
    _galaxy_config,
    build_godmax_config,
    godmax_config_from_params,
)

Z_EDGES = [[0.2, 0.5], [0.5, 0.8], [0.8, 1.1]]


def test_scalar_hod_is_combined():
    sim, analysis = _galaxy_config(
        {"nbar": 5e-4, "z_edges": Z_EDGES, "hod": {"fcen": 0.9}}
    )
    assert analysis["hod_params_model"] == "combined"
    assert_array_equal(sim["fcen_array"], [0.9])


def test_per_bin_hod_passes_through():
    values = [1.0, 0.9, 0.8, 0.7]
    sim, analysis = _galaxy_config(
        {
            "nbar": 5e-4,
            "z_edges": Z_EDGES,
            "hod": {"fcen": values, "alphasat_Nsat": 1.1},
        }
    )
    assert analysis["hod_params_model"] == "perbin"
    assert_array_equal(sim["fcen_array"], values)
    # Scalars may be mixed with per-bin arrays.
    assert_array_equal(sim["alphasat_Nsat_array"], [1.1])
    assert_array_equal(analysis["nz_lens_info_dict"]["z_edges_bins_lens"], Z_EDGES)


@pytest.mark.parametrize("values", [[1.0, 0.9], [1.0, 0.9, 0.8], [1.0] * 5])
def test_wrong_per_bin_length_raises(values):
    # Regression: GODMAX reuses the last entry of a short array without warning.
    with pytest.raises(ValueError, match="'fcen' has .* needs 4"):
        _galaxy_config({"nbar": 5e-4, "z_edges": Z_EDGES, "hod": {"fcen": values}})


@pytest.mark.parametrize("values", [[], [[1.0, 0.9], [0.8, 0.7]]])
def test_malformed_hod_array_raises(values):
    with pytest.raises(ValueError, match="scalar or a non-empty 1D array"):
        _galaxy_config({"nbar": 5e-4, "z_edges": Z_EDGES, "hod": {"fcen": values}})


def test_per_bin_hod_without_z_edges_raises():
    with pytest.raises(ValueError, match="z_edges"):
        _galaxy_config({"nbar": 5e-4, "hod": {"fcen": [1.0, 0.8]}})


@pytest.mark.parametrize(
    "z_edges",
    [
        [0.2, 0.5, 0.5, 0.8],
        jnp.zeros((0, 2)),
        [[0.2, 0.5, 0.8]],
    ],
)
def test_wrong_z_edges_shape_raises(z_edges):
    with pytest.raises(ValueError, match=r"shape \(n_bins, 2\)"):
        _galaxy_config({"nbar": 5e-4, "z_edges": z_edges})


def test_reversed_z_edges_raise():
    with pytest.raises(ValueError, match="lower edge below the upper edge"):
        _galaxy_config({"nbar": 5e-4, "z_edges": [[0.2, 0.5], [0.8, 0.5]]})


def test_rejects_baryon_boost():
    """GODMAX already includes baryons, so with_baryon_boost cannot add more."""

    class UnitBoost:
        def baryonic_suppression(self, zs, ks, k_hunit=False):
            return 1.0

    Combined = with_baryon_boost(GODMAXNonLinearPerturbations, UnitBoost)
    with pytest.raises(TypeError, match="already includes baryons"):
        Combined(None, None, jnp.array([0.1, 0.5]))


def test_zheng_hod_threshold_from_log10Mmin_Ncen():
    sim, analysis = _galaxy_config(
        {"z_edges": Z_EDGES, "hod": {"log10Mmin_Ncen": [12.0, 12.2, 12.4, 12.6]}},
        zheng_hod=True,
    )
    assert_array_equal(sim["log10Mmin_Ncen_array"], [12.0, 12.2, 12.4, 12.6])
    assert analysis["hod_params_model"] == "perbin"
    assert "nbar_gal_comoving_val" not in analysis


@pytest.mark.parametrize("galaxies", [{"nbar": 5e-4}, {"nbar_z": [0.5]}])
def test_zheng_hod_rejects_nbar(galaxies):
    with pytest.raises(ValueError, match="not used with Zheng_HOD"):
        _galaxy_config(galaxies, zheng_hod=True)


@pytest.mark.parametrize(
    "section, key",
    [(section, key) for section, owned in _OWNED_KEYS.items() for key in owned],
)
def test_owned_keys_rejected_in_godmax_config(section, key):
    with pytest.raises(ValueError, match="is set by cloelib from"):
        build_godmax_config(
            jnp.array([0.1, 0.5]), True, None, {section: {key: None}}, {}
        )


GODMAX_PARAMS = {
    "sim_params": {"cosmo": {"H0": 67.2}, "theta_ej_0": 3.0},
    "halo_params": {"z_array": [0.1, 0.5], "nk": 64},
    "analysis": {
        "cosmology_backend": "jax_cosmo",
        "model_tSZ": True,
        "nbar_gal_comoving_val": 5e-4,
        "Zheng_HOD": False,
    },
    "foo": {"bar": 1},
}


def test_godmax_config_from_params_drops_owned_keys():
    config = godmax_config_from_params(GODMAX_PARAMS)
    assert config["sim_params"] == {"theta_ej_0": 3.0}
    assert config["halo_params"] == {"nk": 64}
    assert config["analysis"] == {"Zheng_HOD": False}
    assert "foo" not in config
    build_godmax_config(jnp.array([0.1, 0.5]), True, None, config, {})


def test_godmax_config_from_params_precedence():
    """The file replaces cloelib's defaults; explicit arguments override the file."""
    config = godmax_config_from_params(GODMAX_PARAMS)
    sim, halo, _, _ = build_godmax_config(
        jnp.array([0.1, 0.5]), True, None, config, {"theta_ej_0": 4.0}
    )
    assert halo["nk"] == 64
    assert sim["theta_ej_0"] == 4.0
