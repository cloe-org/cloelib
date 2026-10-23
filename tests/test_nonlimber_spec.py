"""Tests for the beyond-Limber metadata of Contributions (`NonLimberSpec`),
the T-tilde naming rule built on it, and `RSDContribution`.

The Limber computation must be untouched by any of this:
`get_contributions()` / `get_window()` are checked to be unchanged by
`include_rsd`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cloelib.cosmology.jax_cosmology import JAXBackground, JAXNonLinearPerturbations
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import (
    NonLimberSpec,
    nonlimber_contributions,
    nonlimber_spec_of,
    t_tilde_names,
)
from cloelib.observables.photo.positions import (
    GalaxyBiasContribution,
    MagnificationContribution,
    RSDContribution,
)
from cloelib.observables.photo.shear import (
    LensingContribution,
    NLAContribution,
    TATTContribution,
)

jax.config.update("jax_enable_x64", True)

SPECS = {
    "pos": GalaxyBiasContribution.nonlimber_spec,
    "mag": MagnificationContribution.nonlimber_spec,
    "rsd": RSDContribution.nonlimber_spec,
    "she": LensingContribution.nonlimber_spec,
    "ia": NLAContribution.nonlimber_spec,
}

# Reference table
# (`W_ELL_CONFIG`), as `(name_12, name_21)` per ordered pair of kernels.
REFERENCE_NAMES = {
    # galaxy-galaxy
    ("pos", "pos"): ("T_2_00", "T_2_00"),
    ("mag", "mag"): ("T_minus2_00", "T_minus2_00"),
    ("pos", "mag"): ("T_0_00", "T_0_00"),
    ("pos", "rsd"): ("T_2_02", "T_2_20"),
    ("rsd", "pos"): ("T_2_20", "T_2_02"),
    ("mag", "rsd"): ("T_0_02", "T_0_20"),
    ("rsd", "mag"): ("T_0_20", "T_0_02"),
    ("rsd", "rsd"): ("T_2_22", "T_2_22"),
    # shear-shear
    ("she", "she"): ("T_minus2_00", "T_minus2_00"),
    ("she", "ia"): ("T_minus2_00", "T_minus2_00"),
    ("ia", "ia"): ("T_minus2_00", "T_minus2_00"),
    # galaxy-shear
    ("pos", "she"): ("T_0_00", "T_0_00"),
    ("mag", "she"): ("T_minus2_00", "T_minus2_00"),
    ("pos", "ia"): ("T_0_00", "T_0_00"),
    ("mag", "ia"): ("T_minus2_00", "T_minus2_00"),
    ("rsd", "she"): ("T_0_20", "T_0_02"),
    ("rsd", "ia"): ("T_0_20", "T_0_02"),
}


@pytest.mark.parametrize("pair", list(REFERENCE_NAMES))
def test_t_tilde_names_match_reference_table(pair):
    """t_tilde_names picks the same T-tilde as the reference implementation.

    REFERENCE_NAMES is the w_ell assignment of notebooks/my_utils.py for every
    pair of kernels (GG, LL and GL), including the asymmetric RSD pairs where
    the two orderings need different matrices.
    """
    a, b = pair
    assert t_tilde_names(SPECS[a], SPECS[b]) == REFERENCE_NAMES[pair]


@pytest.mark.parametrize("pair", list(REFERENCE_NAMES))
def test_t_tilde_names_swapped_pair_swaps_the_names(pair):
    """Exchanging the two contributions exchanges (name_12, name_21).

    The two names are the matrices of the two orderings in pair_integral; this
    keeps A x B and B x A consistent.
    """
    a, b = pair
    name_12, name_21 = t_tilde_names(SPECS[a], SPECS[b])
    assert t_tilde_names(SPECS[b], SPECS[a]) == (name_21, name_12)


def test_t_tilde_names_follow_the_additive_rule():
    """The rule itself: T_{p_a + p_b}_{d_a d_b}, with the derivative digits swapped for name_21.

    Uses synthetic specs, so the rule is tested on its own; that the real
    contributions declare the right powers and derivatives is what the
    reference-table and Table A1 tests check.
    """
    spec_a = NonLimberSpec(-1, 0, lambda ells: ells, False)
    spec_b = NonLimberSpec(1, 2, lambda ells: ells, False)
    assert t_tilde_names(spec_a, spec_b) == ("T_0_02", "T_0_20")


# ---------------------------------------------------------------------------
# NonLimberSpec / prefactors
# ---------------------------------------------------------------------------
def test_specs_are_hashable_and_static():
    # lensing and NLA are the same kind of kernel beyond Limber
    """NonLimberSpec is a frozen, hashable value (safe to read while tracing).

    Lensing and NLA carry identical specs: beyond Limber they are the same
    kind of kernel, differing only in how the kernel itself is computed.
    """
    assert SPECS["she"] == SPECS["ia"]
    assert len({*SPECS.values()}) == len(SPECS) - 1
    with pytest.raises(Exception):
        SPECS["pos"].power = 3  # frozen


def test_ell_prefactors():
    """The multipole prefactors: 1, -1 (RSD), ell(ell + 1) and sqrt((ell + 2)! / (ell - 2)!).

    The RSD -1 is the sign of the Kaiser term (see the dedicated test below).
    """
    ells = jnp.array([2.0, 10.0, 100.0])
    np.testing.assert_allclose(SPECS["pos"].ell_prefactor(ells), 1.0)
    np.testing.assert_allclose(SPECS["rsd"].ell_prefactor(ells), -1.0)
    np.testing.assert_allclose(SPECS["mag"].ell_prefactor(ells), ells * (ells + 1))
    np.testing.assert_allclose(
        SPECS["she"].ell_prefactor(ells),
        np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1)),
    )
    assert SPECS["she"].ell_prefactor is SPECS["ia"].ell_prefactor


def test_chi2_division_flags():
    """Which kernels are divided by chi^2 when moved to the chi grid.

    The three kernels with power -1 (lensing, magnification, IA) are, as the
    kernels with power +1 (galaxy bias, RSD) are not: the flag follows the
    spec and is what kernel_on_chi_R is called with.
    """
    assert not SPECS["pos"].divide_by_chi2 and not SPECS["rsd"].divide_by_chi2
    assert all(SPECS[k].divide_by_chi2 for k in ("mag", "she", "ia"))


def test_missing_spec_gives_a_clear_error():
    """A contribution without a beyond-Limber spec is rejected explicitly.

    TATT must not inherit NLA's spec: its spectra are signed and cannot be
    combined at unequal times, so using it beyond Limber has to fail loudly,
    not silently reuse the NLA recipe.
    """
    with pytest.raises(NotImplementedError, match="nonlimber_spec"):
        nonlimber_spec_of(object())
    # TATT has no beyond-Limber support (its spectra are signed and cannot
    # be combined at unequal times): it must not inherit one from NLA.
    assert not hasattr(TATTContribution, "nonlimber_spec")
    assert nonlimber_spec_of(LensingContribution(None)) is SPECS["she"]


# ---------------------------------------------------------------------------
# Tracers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup():
    """A JAX cosmology, positions tracers (with or without RSD) and an NLA shear tracer.

    Pure JAX, so no external Boltzmann code is needed.
    """
    background = JAXBackground(
        H0=67.7,
        Omega_b0=0.05,
        Omega_cdm0=0.25,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )
    perturbations = JAXNonLinearPerturbations(background=background)
    z = jnp.linspace(0.2, 2.0, 15)
    n_bins = 2
    dndz = jnp.ones((n_bins, len(z)))
    dndz = dndz / jnp.trapezoid(dndz, z, axis=1)[:, None]

    def positions(include_rsd, dz=0.0, width=1.0):
        return PositionsTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params={
                **{f"dz_pos_{i + 1}": dz for i in range(n_bins)},
                **{f"width_pos_{i + 1}": width for i in range(n_bins)},
                **{f"magnification_bias_{i + 1}": 0.3 for i in range(n_bins)},
                "b1_photo_bin0": 1.1,
                "b1_photo_bin1": 1.4,
            },
            include_rsd=include_rsd,
        )

    shear = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params={
            **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(n_bins)},
            **{f"dz_shear_{i + 1}": 0.0 for i in range(n_bins)},
            **{f"width_shear_{i + 1}": 1.0 for i in range(n_bins)},
            "AIA": 1.0,
            "CIA": 0.0134,
            "EtaIA": -0.41,
        },
        ia_model="NLA",
    )
    return {
        "background": background,
        "perturbations": perturbations,
        "z": z,
        "positions": positions,
        "shear": shear,
    }


def test_nonlimber_contributions_without_rsd_are_the_limber_ones(setup):
    """Without RSD, beyond Limber uses exactly the contributions of the Limber window.

    So only tracers that opt in (RSD) differ between the two computations.
    """
    pos = setup["positions"](include_rsd=False)
    assert pos.rsd is None
    assert pos.get_nonlimber_contributions() == pos.get_contributions()
    assert nonlimber_contributions(pos) == pos.get_contributions()


def test_nonlimber_contributions_add_rsd_only_beyond_limber(setup):
    """include_rsd=True adds RSDContribution beyond Limber, and only there.

    get_contributions() (hence the Limber computation) must keep returning
    (bias, magnification).
    """
    pos = setup["positions"](include_rsd=True)
    limber = pos.get_contributions()
    beyond = nonlimber_contributions(pos)

    assert limber == (pos.bias, pos.magnification)
    assert beyond == (pos.bias, pos.magnification, pos.rsd)
    assert isinstance(pos.rsd, RSDContribution)


def test_include_rsd_leaves_the_limber_window_untouched(setup):
    """Regression guard: adding the RSD contribution does not change get_window().

    The Limber code treats RSD through its own ell-dependent window; summing
    RSDContribution into get_window() too would double count it.
    """
    z = setup["z"]
    without = setup["positions"](include_rsd=False).get_window(z)
    with_rsd = setup["positions"](include_rsd=True).get_window(z)
    np.testing.assert_array_equal(without, with_rsd)


def test_shear_tracer_uses_its_limber_contributions(setup):
    """ShearTracer needs no beyond-Limber hook: the helper falls back to get_contributions()."""
    shear = setup["shear"]
    assert not hasattr(shear, "get_nonlimber_contributions")
    assert nonlimber_contributions(shear) == shear.get_contributions()
    assert nonlimber_contributions(shear) == (shear.lensing, shear.ia)


def test_rsd_kernel_is_n_H_f_over_c(setup):
    """The RSD kernel is W(z) = n_i(z) H(z) f(z) / c.

    Rebuilt independently from the background (H in km/s/Mpc) and the growth
    rate, so a wrong unit or a missing factor shows up.
    """
    z = setup["z"]
    pos = setup["positions"](include_rsd=True)
    c_km_s = 299792.458

    H = setup["background"].hubble_parameter(z, units="km/s/Mpc")
    f = setup["perturbations"].growth_rate(z)
    expected = pos.dndz_shifted * (H * f / c_km_s)[None, :]

    kernel = pos.rsd.compute_kernel(z)
    assert kernel.shape == (2, len(z))
    np.testing.assert_allclose(kernel, expected, rtol=1e-12)
    assert np.all(np.asarray(kernel) >= 0.0)


def test_rsd_kernel_uses_the_shifted_dndz(setup):
    """The RSD kernel follows the redshift distribution with its nuisance shift applied.

    It is the same dn/dz the other kernels (bias, magnification) use, not the raw one.
    """
    z = setup["z"]
    plain = setup["positions"](include_rsd=True)
    shifted = setup["positions"](include_rsd=True, dz=0.05)

    assert not np.allclose(plain.dndz_shifted, shifted.dndz_shifted)
    H = setup["background"].hubble_parameter(z, units="km/s/Mpc")
    f = setup["perturbations"].growth_rate(z)
    np.testing.assert_allclose(
        shifted.rsd.compute_kernel(z),
        shifted.dndz_shifted * (H * f / 299792.458)[None, :],
        rtol=1e-12,
    )
    assert not np.allclose(plain.rsd.compute_kernel(z), shifted.rsd.compute_kernel(z))


# ---------------------------------------------------------------------------
# Table A1 of Chiarenza et al. (Blast.jl, arXiv:2609.01855)
# ---------------------------------------------------------------------------
def _F(ells):
    return (ells + 2) * (ells + 1) * ells * (ells - 1)  # (l+2)!/(l-2)!


def _ll(ells):
    return ells * (ells + 1)


# (A, B): (ell prefactor, power of k in the integrand, w_ell IDs in the order
# the paper lists them). The sign of the RSD prefactor is not compared (see
# `test_rsd_prefactor_sign_is_the_kaiser_one`).
PAPER_TABLE_A1 = {
    ("pos", "pos"): (lambda ells: 1.0 + 0 * ells, 2, ["T_2_00"]),
    ("pos", "mag"): (_ll, 0, ["T_0_00"]),
    ("pos", "rsd"): (lambda ells: 1.0 + 0 * ells, 2, ["T_2_02", "T_2_20"]),
    ("mag", "rsd"): (_ll, 0, ["T_0_02", "T_0_20"]),
    ("mag", "mag"): (lambda ells: _ll(ells) ** 2, -2, ["T_minus2_00"]),
    ("rsd", "rsd"): (lambda ells: 1.0 + 0 * ells, 2, ["T_2_22"]),
    ("pos", "she"): (lambda ells: jnp.sqrt(_F(ells)), 0, ["T_0_00"]),
    ("pos", "ia"): (lambda ells: jnp.sqrt(_F(ells)), 0, ["T_0_00"]),
    ("rsd", "she"): (lambda ells: jnp.sqrt(_F(ells)), 0, ["T_0_20", "T_0_02"]),
    ("rsd", "ia"): (lambda ells: jnp.sqrt(_F(ells)), 0, ["T_0_20", "T_0_02"]),
    ("mag", "she"): (lambda ells: _ll(ells) * jnp.sqrt(_F(ells)), -2, ["T_minus2_00"]),
    ("mag", "ia"): (lambda ells: _ll(ells) * jnp.sqrt(_F(ells)), -2, ["T_minus2_00"]),
    ("she", "she"): (_F, -2, ["T_minus2_00"]),
    ("she", "ia"): (_F, -2, ["T_minus2_00"]),
    ("ia", "ia"): (_F, -2, ["T_minus2_00"]),
}


@pytest.mark.parametrize("pair", list(PAPER_TABLE_A1))
def test_matches_paper_table_a1(pair):
    """Each pair of kernels agrees with its row of Table A1 (Chiarenza et al.).

    Checks the w_ell IDs in the paper's order, the product of the ell
    prefactors (up to the RSD sign) and the power of k in the integrand,
    which has to equal the summed powers p_a + p_b of the two specs.
    """
    a, b = pair
    prefactor, k_power, w_ids = PAPER_TABLE_A1[pair]
    ells = jnp.array([2.0, 3.0, 10.0, 100.0, 1000.0])

    # w_ell IDs, in the paper's order (without repetitions)
    name_12, name_21 = t_tilde_names(SPECS[a], SPECS[b])
    assert list(dict.fromkeys([name_12, name_21])) == w_ids

    # product of the two kernels' ell prefactors (up to the RSD sign)
    ours = SPECS[a].ell_prefactor(ells) * SPECS[b].ell_prefactor(ells)
    np.testing.assert_allclose(jnp.abs(ours), prefactor(ells), rtol=1e-12)

    # the power of k in the integrand is the summed T-tilde power
    assert SPECS[a].power + SPECS[b].power == k_power


def test_every_pair_of_blast_needs_only_the_papers_eight_w_ell():
    # (delta + RSD + mu)^2, (gamma + IA)^2 and (delta + RSD + mu)(gamma + IA),
    # Eqs. A4-A6 without the f_NL terms: exactly these w_ell, no others.
    """Eqs. A4-A6 (without f_NL) involve exactly the eight w_ell of the paper.

    All pairs of (delta, RSD, mu) x (delta, RSD, mu), (gamma, IA) x (gamma, IA)
    and the mixed ones, in both orders, use these matrices and no others, so
    the dataset holds everything that is needed.
    """
    gal = ("pos", "mag", "rsd")
    shear = ("she", "ia")
    pairs = (
        [(a, b) for a in gal for b in gal]
        + [(a, b) for a in shear for b in shear]
        + [(a, b) for a in gal for b in shear]
        + [(a, b) for a in shear for b in gal]
    )
    names = {n for a, b in pairs for n in t_tilde_names(SPECS[a], SPECS[b])}
    assert names == {
        "T_2_00",
        "T_0_00",
        "T_2_02",
        "T_2_20",
        "T_0_02",
        "T_0_20",
        "T_minus2_00",
        "T_2_22",
    }


def test_rsd_kernel_matches_paper_eq_20(setup):
    # W^RSD_i(z) = H(z)/c f(z) n_i(z), Eq. 20 (n_i = shifted dn/dz)
    """The RSD kernel is W^RSD = (H / c) f n_i, Eq. 20 of the paper."""
    z = setup["z"]
    pos = setup["positions"](include_rsd=True)
    H = setup["background"].hubble_parameter(z, units="km/s/Mpc")
    f = setup["perturbations"].growth_rate(z)
    np.testing.assert_allclose(
        pos.rsd.compute_kernel(z), pos.dndz_shifted * H * f / 299792.458, rtol=1e-12
    )


def test_rsd_prefactor_sign_is_the_kaiser_one():
    # Table A1 lists the RSD prefactor as 1 with j''_l: the sign is left
    # implicit there. A plane wave has d^2/dx^2 -> -mu^2, i.e. the Kaiser
    # term f mu^2 delta enters as -f j''_l, so each RSD kernel carries -1:
    # cross terms (delta-RSD, ..) are negative, RSD-RSD is positive.
    """The RSD kernels carry -1, the sign of the Kaiser term.

    Table A1 leaves the sign implicit (prefactor 1 with j''). For a plane
    wave d^2/dx^2 -> -mu^2, so f mu^2 delta enters as -f j'': cross terms
    (delta-RSD, ...) come out negative and RSD-RSD positive.
    """
    ells = jnp.array([2.0, 50.0])
    rsd = SPECS["rsd"].ell_prefactor(ells)
    pos = SPECS["pos"].ell_prefactor(ells)
    assert np.all(np.asarray(rsd * pos) < 0)
    assert np.all(np.asarray(rsd * rsd) > 0)
