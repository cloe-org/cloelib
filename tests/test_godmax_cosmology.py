"""Tests for the GODMAX halo-model backend (`cloelib.cosmology.godmax_cosmology`).

GODMAX is an optional dependency; the whole module is skipped without it. The
grids are small: these tests check interfaces, normalisation and units, not
accuracy.
"""

import importlib.util

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)

if importlib.util.find_spec("godmax") is None:
    pytest.skip("godmax is not installed", allow_module_level=True)

from cloelib.cosmology.godmax_cosmology import GODMAXNonLinearPerturbations  # noqa: E402

REDSHIFTS = jnp.linspace(0.05, 1.5, 8)
ELLS = jnp.logspace(1.0, 3.5, 6)
# The ell grid, the unit beam and the projection grid are only used by the
# GODMAX C_ell reference of the unit test.
GODMAX_CONFIG = {
    "halo_params": {"nM": 12, "nk": 32, "ell_array": ELLS},
    "analysis": {
        "beam_fwhm_arcmin": 0.0,
        "zmin_for_Cls": float(REDSHIFTS[0]),
        "zmax_for_Cls": float(REDSHIFTS[-1]),
        "nz_for_Cls": len(REDSHIFTS),
    },
}


def make_background(**overrides):
    parameters = dict(
        H0=67.2,
        Omega_b0=0.049,
        Omega_cdm0=0.261,
        Omega_k0=0.0,
        As=2.1e-9,
        ns=0.96,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.55,
        N_mnu=0,
    )
    parameters.update(overrides)
    return JAXBackground(**parameters)


@pytest.fixture(scope="module")
def background():
    return make_background()


@pytest.fixture(scope="module")
def halomodel(background):
    """Matter + tSZ halo model."""
    return GODMAXNonLinearPerturbations(
        background,
        JAXLinearPerturbations(background),
        REDSHIFTS,
        godmax_config=GODMAX_CONFIG,
        theta_ej_0=4.0,
    )


@pytest.fixture(scope="module")
def halomodel_matter():
    """Matter-only halo model with twice the primordial amplitude."""
    return GODMAXNonLinearPerturbations(
        make_background(As=4.2e-9),
        None,
        REDSHIFTS,
        model_tSZ=False,
        godmax_config=GODMAX_CONFIG,
        theta_ej_0=4.0,
    )


@pytest.fixture(scope="module")
def halomodel_galaxies(background):
    """Halo model with HOD galaxies."""
    return GODMAXNonLinearPerturbations(
        background,
        None,
        REDSHIFTS,
        galaxies={"nbar": 5e-4},
        godmax_config=GODMAX_CONFIG,
    )


def test_perturbations_protocol(halomodel):
    assert isinstance(halomodel, Perturbations)
    assert isinstance(halomodel.linearperturbations, JAXLinearPerturbations)
    assert halomodel.z.shape == REDSHIFTS.shape
    assert halomodel.k.shape == (32,)
    assert halomodel.Pk.shape == (len(REDSHIFTS), 32)


def test_matter_power_spectrum_shape(halomodel):
    zs = jnp.array([0.2, 0.9])
    ks = jnp.logspace(-3, 1, 7)
    pk = halomodel.matter_power_spectrum(zs, ks)
    assert pk.shape == (2, 7)
    assert bool(jnp.all(jnp.isfinite(pk)) and jnp.all(pk > 0))
    assert halomodel.matter_power_spectrum(0.5, ks).shape == (1, 7)


def test_godmax_uses_cloelib_linear_power(halomodel, background):
    """GODMAX's linear P(k) is cloelib's, normalised to the sigma8 derived from As."""
    h = background.h
    godmax = halomodel.godmax
    expected = halomodel.linearperturbations.matter_power_spectrum(
        halomodel.z, halomodel.k
    )
    assert_allclose(godmax.plin_kz_mat.T / h**3, expected, rtol=1e-10)
    sigma8 = background.interface_args["JAXparams"]["sigma_8"]
    assert_allclose(halomodel.sigma8_0(), sigma8, rtol=0)
    assert_allclose(godmax.cosmo_params["sigma8"], sigma8, rtol=0)


def test_amplitude_scales_with_As(halomodel, halomodel_matter):
    """Doubling As doubles the linear P(k) GODMAX runs on (sigma8^2 doubles)."""
    ratio = halomodel_matter.godmax.plin_kz_mat / halomodel.godmax.plin_kz_mat
    assert_allclose(ratio, 2.0, rtol=1e-10)
    assert_allclose(
        halomodel_matter.sigma8_0() ** 2 / halomodel.sigma8_0() ** 2, 2.0, rtol=1e-10
    )


def test_matter_power_spectrum_on_nodes(halomodel):
    """On the nodes, halofit times the response is GODMAX's own matter spectrum."""
    pk = halomodel.matter_power_spectrum(halomodel.z, halomodel.k)
    assert_allclose(pk, halomodel.Pk, rtol=1e-10)
    h = halomodel.background.h
    assert_allclose(halomodel.Pk, halomodel.godmax.Pmm_tot_mat.T / h**3, rtol=1e-12)


def test_baryonic_response(halomodel):
    """No baryonic effect on large scales, suppression on small scales."""
    zs = halomodel.z
    large = halomodel.baryonic_response(zs, jnp.array([1e-3]))
    small = halomodel.baryonic_response(zs, jnp.array([5.0]))
    assert_allclose(large, 1.0, atol=0.01)
    assert bool(jnp.all(small < 0.95))
    halofit = JAXNonLinearPerturbations(
        halomodel.background, halomodel.linearperturbations
    )
    ks = jnp.array([1e-3, 5.0])
    assert_allclose(
        halomodel.matter_power_spectrum(zs, ks),
        halofit.matter_power_spectrum(zs, ks) * jnp.hstack([large, small]),
        rtol=1e-10,
    )


def test_growth_delegates_to_linear(halomodel):
    zs = jnp.array([0.0, 0.5, 1.0])
    linear = halomodel.linearperturbations
    assert_allclose(halomodel.growth_factor(zs), linear.growth_factor(zs))
    assert_allclose(halomodel.growth_rate(zs), linear.growth_rate(zs))
    assert halomodel.growth_rate().shape == REDSHIFTS.shape
    # No massive neutrinos: cb is the total matter.
    assert_allclose(
        halomodel.matter_power_spectrum_cb(zs, halomodel.k),
        halomodel.matter_power_spectrum(zs, halomodel.k),
        rtol=0,
    )


@pytest.mark.parametrize(
    "overrides", [{"Omega_k0": 0.01}, {"wa": 0.1}, {"mnu": 0.06, "N_mnu": 1}]
)
def test_unsupported_cosmology_raises(overrides):
    with pytest.raises(ValueError, match="flat w0CDM"):
        GODMAXNonLinearPerturbations(make_background(**overrides), None, REDSHIFTS)


def test_invalid_inputs_raise(background):
    with pytest.raises(ValueError, match="JAXBackground"):
        GODMAXNonLinearPerturbations(object(), None, REDSHIFTS)
    with pytest.raises(ValueError, match="different background"):
        GODMAXNonLinearPerturbations(
            background, JAXLinearPerturbations(make_background()), REDSHIFTS
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        GODMAXNonLinearPerturbations(background, None, REDSHIFTS[::-1])
    with pytest.raises(ValueError, match="Unknown halo-model parameters"):
        GODMAXNonLinearPerturbations(background, None, REDSHIFTS, theta_ej=4.0)
    with pytest.raises(ValueError, match="cosmology is taken from"):
        GODMAXNonLinearPerturbations(
            background, None, REDSHIFTS, godmax_config={"sim_params": {"cosmo": {}}}
        )
    with pytest.raises(ValueError, match="nbar"):
        GODMAXNonLinearPerturbations(background, None, REDSHIFTS, galaxies={})
    with pytest.raises(ValueError, match="z_edges"):
        GODMAXNonLinearPerturbations(
            background,
            None,
            REDSHIFTS,
            galaxies={"nbar": 5e-4, "hod": {"fcen": [1.0, 0.8]}},
        )


def test_tsz_spectra(halomodel):
    zs = jnp.array([0.2, 0.9])
    ks = jnp.logspace(-2, 0.5, 5)
    for pair in (("m", "y"), ("y", "m"), ("y", "y")):
        pk = halomodel.power_spectrum(*pair, zs, ks)
        assert pk.shape == (2, 5)
        assert bool(jnp.all(jnp.isfinite(pk)) and jnp.all(pk > 0))
    assert_allclose(
        halomodel.power_spectrum("m", "m", zs, ks),
        halomodel.matter_power_spectrum(zs, ks),
    )


def test_missing_sectors_raise(halomodel, halomodel_matter):
    zs, ks = halomodel.z, halomodel.k
    with pytest.raises(ValueError, match="model_tSZ=True"):
        halomodel_matter.power_spectrum("y", "y", zs, ks)
    with pytest.raises(ValueError, match="galaxies"):
        halomodel.power_spectrum("g", "g", zs, ks)
    with pytest.raises(ValueError, match="Unknown fields"):
        halomodel.power_spectrum("m", "x", zs, ks)


def test_tsz_units_against_godmax_cls(halomodel, background):
    """Limber integrals in cloelib units reproduce GODMAX's own C_ell.

    GODMAX integrates in h-units; the y spectra carry one extra power of h per
    y leg, so this checks the unit conversion of the yy (two legs) and matter-y
    (one leg) spectra. The projection grid is the halo-model nodes, so no
    interpolation in redshift enters.
    """
    from godmax.get_Cls import get_Cl

    godmax_cl = get_Cl({}, {}, {}, {}, Pkz_obj=halomodel.godmax)
    h = background.h
    zs = halomodel.z
    chi = background.comoving_distance(zs)
    dchi_dz = SPEED_OF_LIGHT / 1000 / background.hubble_parameter(zs)
    w_y = 1.0 / (1.0 + zs)
    # GODMAX lensing kernel of its (single, flat) source bin, per Mpc/h.
    w_kappa = godmax_cl.Wk_mat[0] * h

    cl_yy, cl_ky = [], []
    for ell in ELLS:
        ks = (ell + 0.5) / chi
        p_yy = jnp.diagonal(halomodel.power_spectrum("y", "y", zs, ks))
        p_my = jnp.diagonal(halomodel.power_spectrum("m", "y", zs, ks))
        cl_yy.append(jnp.trapezoid(w_y**2 / chi**2 * dchi_dz * p_yy, zs))
        cl_ky.append(jnp.trapezoid(w_kappa * w_y / chi**2 * dchi_dz * p_my, zs))

    assert_allclose(jnp.array(cl_yy), godmax_cl.Cl_y_y_signal_mat, rtol=1e-6)
    assert_allclose(jnp.array(cl_ky), godmax_cl.Cl_kappa_y_tot_mat[:, 0], rtol=1e-6)


def test_galaxy_spectra(halomodel_galaxies):
    hm = halomodel_galaxies
    assert set(hm.spectra) == {
        ("m", "m"),
        ("m", "y"),
        ("y", "y"),
        ("g", "g"),
        ("g", "m"),
        ("g", "y"),
    }
    zs = jnp.array([0.2, 0.9])
    ks = jnp.logspace(-2, 0.5, 5)
    for pair in (("g", "g"), ("g", "m"), ("m", "g"), ("g", "y")):
        pk = hm.power_spectrum(*pair, zs, ks)
        assert pk.shape == (2, 5)
        assert bool(jnp.all(jnp.isfinite(pk)) and jnp.all(pk > 0))


def test_galaxy_large_scale_bias(halomodel_galaxies):
    """On large scales the galaxy spectra follow one linear bias: P_gg P_mm = P_gm^2."""
    hm = halomodel_galaxies
    ks = jnp.array([2e-3])
    p_mm = hm.matter_power_spectrum(hm.z, ks)
    p_gm = hm.power_spectrum("g", "m", hm.z, ks)
    p_gg = hm.power_spectrum("g", "g", hm.z, ks)
    bias = p_gm / p_mm
    assert bool(jnp.all(bias > 0.5) and jnp.all(bias < 5.0))
    assert_allclose(p_gg * p_mm, p_gm**2, rtol=0.05)
