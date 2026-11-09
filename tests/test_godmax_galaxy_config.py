"""Tests for the validation of the GODMAX `galaxies` settings.

`_galaxy_config` only translates the settings, so these tests run without GODMAX.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from cloelib.cosmology.godmax_cosmology import _galaxy_config

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
