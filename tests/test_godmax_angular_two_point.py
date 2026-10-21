"""Tests for angular power spectra on halo-model spectra (GODMAX).

cloelib projects the halo-model power spectra (GODMAX,
`cloelib.cosmology.godmax_cosmology`) with its own Limber integral. Each pair of
contribution fields (matter, HOD galaxies "g", Compton-y "y") is integrated
against its halo-model spectrum. The results are checked against an
independent Limber integral and, with the `"godmax"` kernel convention,
against GODMAX's own C_ell. GODMAX is an optional dependency; the whole module
is skipped without it.

The halo-model redshift nodes are the tracers' redshift grid, so that GODMAX does
not interpolate its spectra or distances in redshift, and the k grid is fine
enough for the two codes' different k interpolations to agree.
"""

import copy
import importlib.util

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloelib.auxiliary.math_utils import simpsons_weights_avg
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)
from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.kernel_conventions import kernel_convention
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import contribution_field
from cloelib.observables.photo.spectrum_engine import needs_generalized_engine
from cloelib.observables.tsz import tSZTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

if importlib.util.find_spec("godmax") is None:
    pytest.skip("godmax is not installed", allow_module_level=True)

from cloelib.cosmology.godmax_cosmology import GODMAXNonLinearPerturbations  # noqa: E402

Z = jnp.linspace(0.05, 2.0, 79)
ELLS = jnp.logspace(1.0, 3.3, 6)
SHEAR_FACTOR = jnp.sqrt((ELLS + 2) * (ELLS + 1) * ELLS * (ELLS - 1)) / (ELLS + 0.5) ** 2
# NLA in GODMAX's parametrisation; cloelib's "NLA_pivot" maps it exactly.
A_IA, ETA_IA, Z0_IA, C_IA = 0.5, 1.0, 0.62, 0.0134

NUISANCE = {
    "multiplicative_bias_1": 0.0,
    "dz_shear_1": 0.0,
    "width_shear_1": 1.0,
    "dz_pos_1": 0.0,
    "width_pos_1": 1.0,
    "magnification_bias_1": 0.8,
    "AIA": A_IA,
    "EtaIA": ETA_IA,
    "z0IA": Z0_IA,
    "CIA": C_IA,
    "b1_photo_bin0": 1.6,
}


def gaussian_nz(mean, sigma):
    nz = jnp.exp(-0.5 * ((Z - mean) / sigma) ** 2)
    return nz / jnp.trapezoid(nz, Z)


NZ_SHEAR = gaussian_nz(1.0, 0.2)
NZ_GALAXIES = gaussian_nz(0.5, 0.1)


@pytest.fixture(scope="module")
def halomodel():
    """Halo model with tSZ and galaxies; the extra settings are for GODMAX's C_ell."""
    background = JAXBackground(
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
    godmax_config = {
        "halo_params": {"nM": 12, "nk": 128, "ell_array": ELLS},
        "analysis": {
            "beam_fwhm_arcmin": 1.4,
            "zmin_for_Cls": float(Z[0]),
            "zmax_for_Cls": float(Z[-1]),
            "nz_for_Cls": len(Z),
            "nz_source_info_dict": {
                "nbins": 1,
                "z_array_source": Z,
                "nz0": np.asarray(NZ_SHEAR),
            },
            "nz_lens_info_dict": {
                "nbins_lens": 1,
                "z_array_lens": Z,
                "nz0": np.asarray(NZ_GALAXIES),
            },
        },
        "other_params": {
            "A_IA": A_IA,
            "eta_IA": ETA_IA,
            "z0_IA": Z0_IA,
            "C1_rhocrit": C_IA,
        },
    }
    return GODMAXNonLinearPerturbations(
        background,
        JAXLinearPerturbations(background),
        Z,
        galaxies={"nbar": 5e-4},
        godmax_config=godmax_config,
    )


def make_tracers(halomodel, convention, nuisance=NUISANCE):
    """Shear, galaxy, CMB lensing and tSZ tracers of a kernel convention."""
    params = convention.nuisance(nuisance)
    return {
        "SHE": ShearTracer(
            halomodel, NZ_SHEAR[None], Z, params, **convention.shear_options()
        ),
        "POS": PositionsTracer(
            halomodel,
            NZ_GALAXIES[None],
            Z,
            nuisance_params=params,
            **convention.positions_options(),
        ),
        "CMBL": CMBLensingTracer(halomodel, Z, **convention.cmb_options()),
        "Y": tSZTracer(halomodel, Z, **convention.tsz_options()),
    }


def all_cls(halomodel, tracers):
    """C_ell of every tracer pair, first bins."""
    spectra = {}
    names = list(tracers)
    for i, name1 in enumerate(names):
        for name2 in names[i:]:
            two_point = AngularTwoPoint(tracers[name1], tracers[name2])
            spectra[name1, name2] = two_point.get_Cl_tensor(ELLS, 0, halomodel.k)[
                :, 0, 0
            ]
    return spectra


@pytest.fixture(scope="module")
def godmax_tracers(halomodel):
    return make_tracers(halomodel, kernel_convention("godmax"))


@pytest.fixture(scope="module")
def godmax_cls(halomodel, godmax_tracers):
    return all_cls(halomodel, godmax_tracers)


@pytest.fixture(scope="module")
def godmax_reference(halomodel):
    """C_ell of GODMAX's own projection; its lensing slot is shear or CMB lensing."""
    from godmax.get_Cls import get_Cl

    shear_run = get_Cl({}, {}, {}, {}, Pkz_obj=halomodel.godmax)
    cmb_setup = copy.copy(halomodel.godmax)
    cmb_setup.is_cmb_lensing = True
    cmb_run = get_Cl({}, {}, {}, {}, Pkz_obj=cmb_setup)
    return {
        ("SHE", "SHE"): shear_run.Cl_kappa_kappa_tot_mat[:, 0, 0],
        ("SHE", "POS"): shear_run.Cl_gal_kappa_tot_mat[:, 0, 0],
        ("SHE", "Y"): shear_run.Cl_kappa_y_tot_mat[:, 0],
        ("POS", "POS"): shear_run.Cl_gal_gal_tot_mat[:, 0, 0],
        ("POS", "CMBL"): cmb_run.Cl_gal_kappa_tot_mat[:, 0, 0],
        ("POS", "Y"): shear_run.Cl_gal_y_tot_mat[:, 0],
        ("CMBL", "CMBL"): cmb_run.Cl_kappa_kappa_tot_mat[:, 0, 0],
        ("CMBL", "Y"): cmb_run.Cl_kappa_y_tot_mat[:, 0],
        ("Y", "Y"): shear_run.Cl_y_y_signal_mat,
    }


def leg_factor(tracer):
    """ell factor of one leg: spin-2 factor of shear, beam of y."""
    factor = SHEAR_FACTOR if tracer.prefact_toggle else jnp.ones_like(ELLS)
    if hasattr(tracer, "ell_factor"):
        factor = factor * tracer.ell_factor(ELLS)
    return factor


def limber(halomodel, tracer1, tracer2, contributions1=None, contributions2=None):
    """Independent Limber integral, contribution by contribution.

    Each pair of contributions is integrated against the halo-model spectrum of
    the two fields they trace.
    """
    contributions1 = contributions1 or tracer1.get_contributions()
    contributions2 = contributions2 or tracer2.get_contributions()
    background = halomodel.background
    chi = background.comoving_distance(Z)
    dchi_dz = SPEED_OF_LIGHT / 1000 / background.hubble_parameter(Z)
    weights = simpsons_weights_avg(len(Z)) * (Z[1] - Z[0])
    cl = jnp.zeros(len(ELLS))
    for c1 in contributions1:
        for c2 in contributions2:
            kernel = c1.compute_kernel(Z)[0] * c2.compute_kernel(Z)[0]
            f1, f2 = contribution_field(c1), contribution_field(c2)
            values = []
            for ell in ELLS:
                ks = (ell + 0.5) / chi
                pk = jnp.diagonal(halomodel.power_spectrum(f1, f2, Z, ks))
                values.append(jnp.sum(weights * dchi_dz * kernel / chi**2 * pk))
            cl = cl + jnp.array(values)
    return cl * leg_factor(tracer1) * leg_factor(tracer2)


PAIRS = [
    ("SHE", "SHE"),
    ("SHE", "POS"),
    ("SHE", "Y"),
    ("POS", "POS"),
    ("POS", "CMBL"),
    ("POS", "Y"),
    ("CMBL", "CMBL"),
    ("CMBL", "Y"),
    ("Y", "Y"),
]


# --- "godmax" convention: reproduces GODMAX's own C_ell ------------------------


@pytest.mark.parametrize(
    "pair, rtol",
    [(pair, 5e-3) for pair in PAIRS if pair != ("Y", "Y")]
    # yy is dominated by low redshift, where Simpson (cloelib) and trapezoid
    # (GODMAX) differ most.
    + [(("Y", "Y"), 2.5e-2)],
)
def test_godmax_convention_matches_godmax(godmax_cls, godmax_reference, pair, rtol):
    """With the "godmax" kernels (NLA with pivot, z_s = 1100, no spin-2 factor,
    1.4' beam, HOD, no magnification), cloelib reproduces GODMAX's C_ell."""
    assert_allclose(godmax_cls[pair], godmax_reference[pair], rtol=rtol)


@pytest.mark.parametrize("pair", PAIRS)
def test_pairs_match_independent_limber(halomodel, godmax_tracers, godmax_cls, pair):
    """Every pair runs on the standard Limber path with its halo-model spectra."""
    tracer1, tracer2 = godmax_tracers[pair[0]], godmax_tracers[pair[1]]
    assert not needs_generalized_engine(
        tracer1.get_contributions(), tracer2.get_contributions()
    )
    assert_allclose(godmax_cls[pair], limber(halomodel, tracer1, tracer2), rtol=5e-3)


# --- Overrides: magnification, RSD, linear bias --------------------------------


@pytest.fixture(scope="module")
def magnification_tracers(halomodel):
    return make_tracers(halomodel, kernel_convention("godmax", magnification=True))


@pytest.fixture(scope="module")
def magnification_cls(halomodel, magnification_tracers):
    return all_cls(halomodel, magnification_tracers)


def test_magnification_changes_only_galaxy_pairs(godmax_cls, magnification_cls):
    for pair in PAIRS:
        if "POS" in pair:
            assert not np.allclose(
                magnification_cls[pair], godmax_cls[pair], rtol=1e-4, atol=0
            )
        else:
            assert_allclose(magnification_cls[pair], godmax_cls[pair], rtol=1e-12)


@pytest.mark.parametrize("other", ["SHE", "POS", "CMBL", "Y"])
def test_magnification_term_uses_matter_spectra(
    halomodel, godmax_cls, magnification_tracers, magnification_cls, other
):
    """The magnification term traces matter: against HOD galaxies it uses P_gm,
    against tSZ P_my, against lensing P_mm."""
    galaxies = magnification_tracers["POS"]
    pair = ("POS", other) if ("POS", other) in godmax_cls else (other, "POS")
    difference = magnification_cls[pair] - godmax_cls[pair]
    if other == "POS":
        density, magnification = galaxies.bias, galaxies.magnification
        expected = 2 * limber(
            halomodel, galaxies, galaxies, (density,), (magnification,)
        ) + limber(halomodel, galaxies, galaxies, (magnification,), (magnification,))
    else:
        expected = limber(
            halomodel,
            galaxies,
            magnification_tracers[other],
            (galaxies.magnification,),
            None,
        )
    assert_allclose(difference, expected, rtol=5e-3)


def test_rsd_with_hod_and_tsz(halomodel, godmax_tracers):
    """RSD is available with HOD galaxies and against tSZ. The RSD term traces
    matter, so its pairing with tSZ is the same with HOD or with linear bias."""
    rsd = make_tracers(halomodel, kernel_convention("godmax", rsd=True))
    linear = make_tracers(
        halomodel, kernel_convention("godmax", galaxy_bias_model="per_bin")
    )
    linear_rsd = make_tracers(
        halomodel,
        kernel_convention("godmax", galaxy_bias_model="per_bin", rsd=True),
    )

    def cl(tracers, other):
        return AngularTwoPoint(tracers["POS"], tracers[other]).get_Cl_tensor(
            ELLS, 0, halomodel.k
        )[:, 0, 0]

    hod_rsd_term = cl(rsd, "Y") - cl(godmax_tracers, "Y")
    linear_rsd_term = cl(linear_rsd, "Y") - cl(linear, "Y")
    assert bool(jnp.all(jnp.isfinite(hod_rsd_term)))
    assert bool(jnp.all(hod_rsd_term != 0))
    assert_allclose(hod_rsd_term, linear_rsd_term, rtol=1e-6)
    for other in ("POS", "SHE", "CMBL"):
        assert not np.allclose(
            cl(rsd, other), cl(godmax_tracers, other), rtol=1e-6, atol=0
        )
    # RSD only touches the galaxy kernel.
    assert_allclose(
        AngularTwoPoint(rsd["SHE"], rsd["Y"]).get_Cl_tensor(ELLS, 0, halomodel.k),
        AngularTwoPoint(godmax_tracers["SHE"], godmax_tracers["Y"]).get_Cl_tensor(
            ELLS, 0, halomodel.k
        ),
        rtol=1e-12,
    )


@pytest.mark.parametrize("other", ["SHE", "POS", "CMBL", "Y"])
def test_linear_bias_uses_matter_spectra(halomodel, other):
    """With a linear bias the galaxy kernel carries b, and the pairings use the
    halo-model matter spectra (P_mm, P_my) instead of the HOD ones."""
    tracers = make_tracers(
        halomodel, kernel_convention("godmax", galaxy_bias_model="per_bin")
    )
    galaxies = tracers["POS"]
    assert contribution_field(galaxies.bias) == "m"
    cl = AngularTwoPoint(galaxies, tracers[other]).get_Cl_tensor(ELLS, 0, halomodel.k)
    expected = limber(halomodel, galaxies, tracers[other])
    assert_allclose(cl[:, 0, 0], expected, rtol=5e-3)


def test_hod_follows_a_linear_bias_on_large_scales(halomodel, godmax_cls):
    """At low ell the HOD spectra are close to linear bias times matter."""
    in_bin = (Z > 0.15) & (Z < 0.85)
    ks = jnp.array([2e-3])
    bias = (
        halomodel.power_spectrum("g", "m", Z[in_bin], ks)
        / halomodel.matter_power_spectrum(Z[in_bin], ks)
    )[:, 0]
    coefficients = np.polyfit(np.asarray(Z[in_bin]), np.asarray(bias), 3)[::-1]
    nuisance = dict(NUISANCE)
    for order, coefficient in enumerate(coefficients):
        nuisance[f"b1_photo_poly{order}"] = float(coefficient)
    linear = make_tracers(
        halomodel, kernel_convention("godmax", galaxy_bias_model="poly"), nuisance
    )
    for other in ("POS", "CMBL"):
        cl_linear = AngularTwoPoint(linear["POS"], linear[other]).get_Cl_tensor(
            ELLS, 0, halomodel.k
        )
        assert_allclose(godmax_cls["POS", other][:2], cl_linear[:2, 0, 0], rtol=0.08)


def test_baseline_convention_runs_every_pair(halomodel):
    """cloelib's baseline kernels (linear bias, magnification, RSD, NLA) on the
    halo-model spectra, for every pair."""
    nuisance = dict(NUISANCE, **{f"b1_photo_poly{i}": 1.0 for i in range(4)})
    tracers = make_tracers(halomodel, kernel_convention("baseline"), nuisance)
    assert tracers["SHE"].prefact_toggle == 1
    assert tracers["POS"].include_rsd
    for pair, cl in all_cls(halomodel, tracers).items():
        assert bool(jnp.all(jnp.isfinite(cl))), pair


# --- Switches ------------------------------------------------------------------


def test_switch_defaults_keep_cloelib_behaviour(halomodel):
    halofit = JAXNonLinearPerturbations(
        halomodel.background, halomodel.linearperturbations
    )
    default = CMBLensingTracer(halofit, Z)
    explicit = CMBLensingTracer(halofit, Z, z_source=halomodel.background.z_star)
    assert_allclose(default.get_window(Z), explicit.get_window(Z), rtol=0)
    assert ShearTracer(halofit, NZ_SHEAR[None], Z, NUISANCE).prefact_toggle == 1
    no_beam = tSZTracer(halomodel, Z)
    assert_allclose(no_beam.ell_factor(ELLS), 1.0, rtol=0)


def test_y_beam(halomodel):
    tracer = tSZTracer(halomodel, Z, beam_fwhm_arcmin=1.4)
    sigma = np.deg2rad(1.4 / 60) / np.sqrt(8 * np.log(2))
    assert_allclose(
        tracer.beam(ELLS), np.exp(-ELLS * (ELLS + 1) * sigma**2 / 2), rtol=1e-12
    )
    with pytest.raises(ValueError, match="non-negative"):
        tSZTracer(halomodel, Z, beam_fwhm_arcmin=-1.0)


# --- Output packaging ----------------------------------------------------------


def test_y_output_keys_and_shapes(halomodel, godmax_tracers):
    n_ell = len(ELLS)
    two_bins = jnp.stack([NZ_GALAXIES, gaussian_nz(0.9, 0.1)])
    nuisance = dict(NUISANCE, dz_pos_2=0.0, width_pos_2=1.0, magnification_bias_2=0.0)
    galaxies = PositionsTracer(halomodel, two_bins, Z, "hod", nuisance)
    expected = {
        ("Y", "Y"): {("Y", "Y", 1, 1): (n_ell,)},
        ("CMBL", "Y"): {("CMBL", "Y", 1, 1): (n_ell,)},
        ("SHE", "Y"): {("SHE", "Y", 1, 1): (2, n_ell)},
        ("POS", "Y"): {("POS", "Y", 1, 1): (n_ell,), ("POS", "Y", 2, 1): (n_ell,)},
    }
    setup = dict(godmax_tracers, POS=galaxies)
    for (name1, name2), shapes in expected.items():
        forward = AngularTwoPoint(setup[name1], setup[name2]).get_Cl(
            ELLS, 0, halomodel.k
        )
        backward = AngularTwoPoint(setup[name2], setup[name1]).get_Cl(
            ELLS, 0, halomodel.k
        )
        assert set(forward) == set(backward) == set(shapes)
        for key, shape in shapes.items():
            assert forward[key].array.shape == shape
            assert_allclose(forward[key].array, backward[key].array, rtol=1e-12)
    she_y = AngularTwoPoint(setup["SHE"], setup["Y"]).get_Cl(ELLS, 0, halomodel.k)
    assert bool(jnp.all(she_y["SHE", "Y", 1, 1].array[1] == 0))


# --- What is not available -----------------------------------------------------


def test_halo_model_tracers_need_halo_model_spectra(halomodel):
    halofit = JAXNonLinearPerturbations(
        halomodel.background, halomodel.linearperturbations
    )
    with pytest.raises(ValueError, match="model_tSZ=True"):
        tSZTracer(halofit, Z)
    with pytest.raises(ValueError, match="galaxies="):
        PositionsTracer(halofit, NZ_GALAXIES[None], Z, "hod", NUISANCE)
    with pytest.raises(NotImplementedError, match="use_Pcb"):
        PositionsTracer(halomodel, NZ_GALAXIES[None], Z, "hod", NUISANCE, use_Pcb=True)
    with pytest.raises(ValueError, match="equal to zero"):
        tSZTracer(halomodel, jnp.linspace(0.0, 1.0, 5))


def test_tatt_and_nonlinear_bias_raise(halomodel):
    class StubLoopComputer:
        def compute(self, name):
            return lambda matter_pk, ks, zs: jnp.zeros(len(ks))

    nuisance = dict(NUISANCE, A2IA=1.0, bTA=1.0)
    with pytest.raises(NotImplementedError, match="ia_model='TATT'"):
        ShearTracer(
            halomodel,
            NZ_SHEAR[None],
            Z,
            nuisance,
            ia_model="TATT",
            tatt_loop_computer=StubLoopComputer(),
        )
    with pytest.raises(NotImplementedError, match="galaxy_bias_model='nonlinear'"):
        PositionsTracer(
            halomodel,
            NZ_GALAXIES[None],
            Z,
            "nonlinear",
            NUISANCE,
            nl_bias_loop_computer=StubLoopComputer(),
        )


def test_redshift_grid_must_be_covered(halomodel):
    wider = jnp.linspace(0.02, 2.0, 80)
    nz = jnp.exp(-0.5 * ((wider - 1.0) / 0.2) ** 2)
    shear = ShearTracer(
        halomodel, (nz / jnp.trapezoid(nz, wider))[None], wider, NUISANCE
    )
    tsz = tSZTracer(halomodel, wider)
    with pytest.raises(ValueError, match="outside the halo-model nodes"):
        AngularTwoPoint(shear, tsz).get_Cl_tensor(ELLS, 0, halomodel.k)
