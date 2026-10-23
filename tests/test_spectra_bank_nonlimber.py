"""Tests for the beyond-Limber part of `SpectraBank`.

A bank is built from a real (JAX) linear cosmology and a tiny fake T-tilde
dataset on a small grid, so that everything runs in a couple of seconds
while exercising the same code as the real 128 x 64 x 161 grids.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cloelib.auxiliary.chebyshev import chebyshev_points_interval
from cloelib.auxiliary.nonlimber_kernels import unequal_time_pk, w_ell
from cloelib.auxiliary.t_tildes import NonLimberGrid, TTildeStore, directory_name
from cloelib.cosmology.jax_cosmology import JAXBackground, JAXLinearPerturbations
from cloelib.observables.photo.spectrum_engine import SpectraBank, SpectrumRequest

jax.config.update("jax_enable_x64", True)

N_ELL, N_CHI, N_R, N_CHEB = 3, 6, 4, 9
FAKE_ELLS = np.array([2.0, 14.3, 97.1])


@pytest.fixture(scope="module")
def cosmology():
    """A real JAX linear cosmology: its background and linear perturbations."""
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
    return background, JAXLinearPerturbations(background)


@pytest.fixture
def grid():
    """A tiny grid: 6 chi, 4 R, 9 Chebyshev nodes in log10 k on [1e-3, 1] 1/Mpc."""
    return NonLimberGrid(
        ells=jnp.array(FAKE_ELLS),
        chi=jnp.linspace(100.0, 2000.0, N_CHI),
        R=jnp.linspace(0.3, 1.0, N_R),
        k_cheb=chebyshev_points_interval(N_CHEB - 1, -3.0, 0.0),
        w_chi=jnp.ones(N_CHI),
        w_R=jnp.ones(N_R),
    )


@pytest.fixture
def store(tmp_path, grid):
    """A fake dataset with three random T-tilde matrices, wrapped in a store."""
    rng = np.random.default_rng(0)
    for name in ("T_2_00", "T_minus2_00", "T_0_20"):
        folder = tmp_path / directory_name(name)
        folder.mkdir()
        for ell in FAKE_ELLS:
            block = rng.normal(size=(1, N_CHI, N_R, N_CHEB))
            np.save(folder / f"T_tilde_l_{ell:.1f}.npy", block)
    return TTildeStore(tmp_path, grid=grid)


class CountingPerturbations:
    """Wraps linear perturbations and counts the evaluations of the spectrum."""

    def __init__(self, perturbations):
        self._perturbations = perturbations
        self.background = perturbations.background
        self.calls = 0

    def matter_power_spectrum(self, zs, ks, **kwargs):
        self.calls += 1
        return self._perturbations.matter_power_spectrum(zs, ks, **kwargs)


@pytest.fixture
def bank(cosmology, store):
    """A bank configured for beyond-Limber (the Limber arguments are dummies)."""
    _, linear = cosmology
    ks = jnp.logspace(-3, 0, 5)
    zs = jnp.linspace(0.1, 2.0, 4)
    return SpectraBank(
        jnp.ones((4, 5)),
        ks,
        zs,
        linear_perturbations=CountingPerturbations(linear),
        t_tildes=store,
    )


# ---------------------------------------------------------------------------
# The existing (Limber) behaviour is untouched
# ---------------------------------------------------------------------------
def test_limber_bank_is_built_and_works_as_before():
    """A bank built the old way, with three arguments, still memoizes `get`."""
    calls = []

    def compute(matter_pk, ks, zs):
        calls.append(1)
        return matter_pk * 2.0

    bank = SpectraBank(jnp.ones((2, 3)), jnp.arange(3.0), jnp.arange(2.0))
    request = SpectrumRequest("twice", compute)

    assert bank.get(request) is bank.get(request)
    assert len(calls) == 1
    assert bank.linear_perturbations is None and bank.t_tildes is None


def test_nonlimber_needs_both_linear_perturbations_and_t_tildes(cosmology, store):
    """Beyond-Limber methods fail clearly when the bank is not configured for them."""
    _, linear = cosmology
    args = (jnp.ones((2, 3)), jnp.arange(3.0), jnp.arange(2.0))
    for bank in (
        SpectraBank(*args),
        SpectraBank(*args, linear_perturbations=linear),
        SpectraBank(*args, t_tildes=store),
    ):
        for call in (
            bank.z_of_chi,
            bank.nonlimber_cheb,
            lambda: bank.w_ell("T_2_00"),
        ):
            with pytest.raises(ValueError, match="linear_perturbations.*t_tildes"):
                call()


# ---------------------------------------------------------------------------
# from_perturbations
# ---------------------------------------------------------------------------
def test_from_perturbations_evaluates_the_matter_pk(cosmology):
    """The bank holds the perturbations' spectrum on the requested grids."""
    _, linear = cosmology
    ks, zs = jnp.logspace(-3, 0, 5), jnp.linspace(0.1, 2.0, 4)

    bank = SpectraBank.from_perturbations(linear, ks, zs)

    np.testing.assert_array_equal(bank.matter_pk, linear.matter_power_spectrum(zs, ks))
    assert bank.matter_pk.shape == (4, 5)
    np.testing.assert_array_equal(bank.ks, ks)
    np.testing.assert_array_equal(bank.zs, zs)


# ---------------------------------------------------------------------------
# z(chi), Chebyshev coefficients, w_ell
# ---------------------------------------------------------------------------
def test_z_of_chi_inverts_the_comoving_distance(bank, cosmology, grid):
    """z(chi) is the redshift at which the background's comoving distance is chi."""
    background, _ = cosmology
    z = bank.z_of_chi()
    assert z.shape == (N_CHI,)
    np.testing.assert_allclose(background.comoving_distance(z), grid.chi, rtol=1e-6)


def test_nonlimber_cheb_expands_the_unequal_time_linear_pk(bank, cosmology, grid):
    """The coefficients reproduce P_lin(k, chi, R chi) at the Chebyshev nodes.

    Rebuilds the unequal-time spectrum independently (the linear Pk at
    z(chi) and at the nodes, through `unequal_time_pk`) and sums the Chebyshev
    series with numpy: this checks the node order, the normalization and the
    mapping of log10 k onto [-1, 1] that the T-tilde matrices are built on.
    """
    _, linear = cosmology
    coeffs = bank.nonlimber_cheb()
    assert coeffs.shape == (N_CHEB, N_CHI, N_R)

    Pk_chi = linear.matter_power_spectrum(bank.z_of_chi(), 10.0**grid.k_cheb)
    expected = np.asarray(unequal_time_pk(Pk_chi, grid.chi, grid.R))  # (k, chi, R)

    u = np.asarray(grid.k_cheb)
    u = (2 * u - (u.max() + u.min())) / (u.max() - u.min())  # -> [-1, 1]
    for j in range(N_CHI):
        for r in range(N_R):
            values = np.polynomial.chebyshev.chebval(u, np.asarray(coeffs[:, j, r]))
            np.testing.assert_allclose(values, expected[:, j, r], rtol=1e-8)


def test_nonlimber_cheb_is_computed_once(bank):
    """The coefficients (and the linear Pk behind them) are shared and memoized."""
    first = bank.nonlimber_cheb()
    assert bank.nonlimber_cheb() is first
    assert bank.linear_perturbations.calls == 1


def test_w_ell_contracts_the_coefficients_with_the_t_tilde(bank, store):
    """w_ell(name) is the contraction of the shared coefficients with that matrix."""
    out = bank.w_ell("T_minus2_00")
    assert out.shape == (N_ELL, N_CHI, N_R)
    np.testing.assert_allclose(
        out, w_ell(bank.nonlimber_cheb(), store.get("T_minus2_00")), rtol=1e-12
    )


def test_w_ell_with_a_unit_t_tilde_picks_the_first_coefficient(bank, grid, tmp_path):
    """A T-tilde that is 1 for n = 0 and 0 otherwise gives w_ell = c_0(chi, R).

    Pins the axes of the contraction (the Chebyshev index of the coefficients
    is the last one of the matrix) with a case that is known by inspection.
    """
    folder = tmp_path / "unit" / directory_name("T_2_00")
    folder.mkdir(parents=True)
    unit = np.zeros((1, N_CHI, N_R, N_CHEB))
    unit[..., 0] = 1.0
    for ell in FAKE_ELLS:
        np.save(folder / f"T_tilde_l_{ell:.1f}.npy", unit)
    bank.t_tildes = TTildeStore(tmp_path / "unit", grid=grid)

    out = bank.w_ell("T_2_00")
    for ell in range(N_ELL):
        np.testing.assert_allclose(out[ell], bank.nonlimber_cheb()[0], rtol=1e-12)


def test_w_ell_is_memoized_per_name_and_loads_only_what_is_asked(
    bank, store, monkeypatch
):
    """Each w_ell is computed once, and untouched matrices are never read."""
    loads = []
    original = np.load
    monkeypatch.setattr(
        np,
        "load",
        lambda path, *a, **k: loads.append(str(path)) or original(path, *a, **k),
    )

    first = bank.w_ell("T_2_00")
    assert bank.w_ell("T_2_00") is first
    bank.w_ell("T_minus2_00")

    assert bank.linear_perturbations.calls == 1  # one shared set of coefficients
    assert len(loads) == 2 * N_ELL
    assert not any(directory_name("T_0_20") in f for f in loads)


def test_w_ell_unknown_matrix(bank):
    """Asking for a matrix the dataset does not hold is a FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Missing T-tilde directory"):
        bank.w_ell("T_2_22")
