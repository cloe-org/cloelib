"""Tests for the linear Limber Cl of `AngularTwoPoint` (`_limber_linear_cl`).

The linear Limber Cl is the Limber counterpart of the beyond-Limber one: the
same windows, quadrature and prefactors as `get_Cl_tensor`, with the linear
spectra of the backend. It is checked against `get_Cl_tensor` of the same
tracers built directly on the linear backend (JAX, so no external code).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cloelib.auxiliary.t_tildes import NonLimberGrid
from cloelib.cosmology.jax_cosmology import JAXBackground, JAXNonLinearPerturbations
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import CB, MATTER
from cloelib.observables.photo.spectrum_engine import SpectraBank
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

jax.config.update("jax_enable_x64", True)

Z_PK = jnp.linspace(0.0, 3.5, 60)
KS = jnp.logspace(-4, 1, 120)
ELLS = jnp.array([2.0, 14.3, 97.1, 211.6])


@pytest.fixture(scope="module")
def nonlinear():
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
    nonlinear = JAXNonLinearPerturbations(background)
    # Evaluated on the fly: no tabulated grids, so set the ones
    # `AngularTwoPoint` reads.
    nonlinear.z = Z_PK
    nonlinear.k = KS
    nonlinear.linearperturbations.z = Z_PK
    nonlinear.linearperturbations.k = KS
    return nonlinear


@pytest.fixture(scope="module")
def tomography():
    z = jnp.linspace(0.01, 3.0, 300)
    dndz = jnp.array([jnp.exp(-0.5 * ((z - zc) / 0.15) ** 2) for zc in (0.6, 1.2)])
    dndz = dndz / jnp.trapezoid(dndz, z, axis=1)[:, None]
    return z, dndz


def _positions(perturbations, tomography, include_rsd=False, use_Pcb=False):
    z, dndz = tomography
    nuisance = {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(2)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(2)},
        **{f"magnification_bias_{i + 1}": 0.4 for i in range(2)},
        **{f"b1_photo_bin{i}": b for i, b in enumerate((1.3, 1.7))},
    }
    return PositionsTracer(
        perturbations,
        dndz,
        z,
        "per_bin",
        nuisance,
        include_rsd=include_rsd,
        use_Pcb=use_Pcb,
    )


def _shear(perturbations, tomography):
    z, dndz = tomography
    nuisance = {
        **{f"multiplicative_bias_{i + 1}": m for i, m in enumerate((0.01, -0.02))},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(2)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(2)},
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": -0.41,
    }
    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=nuisance,
        ia_model="NLA",
    )


def _tracers(kind, perturbations, tomography):
    """The pair of tracers `kind` built on `perturbations`."""
    if kind == "gg":
        g = _positions(perturbations, tomography, include_rsd=True)
        return g, g
    if kind == "gg_pcb":
        g = _positions(perturbations, tomography, include_rsd=True, use_Pcb=True)
        return g, g
    if kind == "gs":
        return (
            _positions(perturbations, tomography, include_rsd=True),
            _shear(perturbations, tomography),
        )
    if kind == "gs_pcb":
        return (
            _positions(perturbations, tomography, include_rsd=True, use_Pcb=True),
            _shear(perturbations, tomography),
        )
    s = _shear(perturbations, tomography)
    return s, s


def test_legacy_default_bank_is_the_nonlinear_one(nonlinear, tomography):
    """Passing no bank is the same as passing the bank of `tracer1.perturbations`."""
    two_point = AngularTwoPoint(*_tracers("gs", nonlinear, tomography))
    bank = SpectraBank.from_perturbations(nonlinear, KS, nonlinear.z)
    np.testing.assert_array_equal(
        two_point._compute_cl_legacy(ELLS, None, KS),
        two_point._compute_cl_legacy(ELLS, None, KS, bank=bank),
    )


@pytest.mark.parametrize("kind", ["gg", "gg_pcb", "gs", "gs_pcb", "ss"])
def test_linear_limber_matches_tracers_on_the_linear_backend(
    kind, nonlinear, tomography
):
    """Only the spectra change: same Cl as the tracers built on the linear backend.

    Covers RSD, magnification, NLA, the multiplicative shear bias and, with
    `use_Pcb`, the cb and cb x matter spectra.
    """
    two_point = AngularTwoPoint(*_tracers(kind, nonlinear, tomography))
    out = two_point._limber_linear_cl(ELLS, KS, two_point._linear_bank(KS))

    linear = nonlinear.linearperturbations
    expected = AngularTwoPoint(*_tracers(kind, linear, tomography)).get_Cl_tensor(
        ELLS, None, KS
    )
    assert out.shape == (len(ELLS), 2, 2)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    # and it is not the non-linear Limber Cl
    assert not np.allclose(
        out, two_point.get_Cl_tensor(ELLS, None, KS), rtol=1e-3, atol=0.0
    )


def test_linear_bank_cross_spectrum_is_the_transfer_function_one(nonlinear):
    """The cb x matter linear spectrum is sqrt(P_cb^lin P_mm^lin), as `T_cb T_m`."""
    two_point = AngularTwoPoint(SimpleNamespace(perturbations=nonlinear), None)
    bank = two_point._linear_bank(KS)
    np.testing.assert_allclose(
        bank.base(CB, MATTER),
        jnp.sqrt(bank.base(CB, CB) * bank.base(MATTER, MATTER)),
        rtol=1e-12,
    )


def test_linear_limber_on_the_t_tilde_nodes(nonlinear, tomography):
    """Runs on the (non-integer) multipole nodes of the T-tilde dataset."""
    ells = NonLimberGrid.blast_128_64_161().ells
    two_point = AngularTwoPoint(*_tracers("gs", nonlinear, tomography))
    out = two_point._limber_linear_cl(ells, KS, two_point._linear_bank(KS))
    assert out.shape == (len(ells), 2, 2)
    assert np.all(np.isfinite(out))


def test_linear_bank_without_linear_spectrum():
    """A backend with no linear counterpart gives a clear error."""
    NoLinearNonLinearPerturbations = type("NoLinearNonLinearPerturbations", (), {})
    tracer = SimpleNamespace(perturbations=NoLinearNonLinearPerturbations())
    with pytest.raises(ValueError, match="NoLinearNonLinearPerturbations"):
        AngularTwoPoint(tracer, tracer)._linear_bank(KS)
