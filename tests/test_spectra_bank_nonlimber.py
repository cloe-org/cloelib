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
from cloelib.observables.photo.contributions import CB, MATTER
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
    """Wraps linear perturbations and counts the evaluations of the spectra."""

    def __init__(self, perturbations):
        self._perturbations = perturbations
        self.background = perturbations.background
        self.calls = 0

    def matter_power_spectrum(self, zs, ks, **kwargs):
        self.calls += 1
        return self._perturbations.matter_power_spectrum(zs, ks, **kwargs)

    def matter_power_spectrum_cb(self, zs, ks, **kwargs):
        self.calls += 1
        return self._perturbations.matter_power_spectrum_cb(zs, ks, **kwargs)


class NonLinearStub:
    """Stands in for a non-linear backend: only exposes `linearperturbations`."""

    def __init__(self, linearperturbations):
        self.linearperturbations = linearperturbations


LIMBER_ARGS = (jnp.ones((4, 5)), jnp.logspace(-3, 0, 5), jnp.linspace(0.1, 2.0, 4))


@pytest.fixture
def bank(cosmology, store):
    """A bank configured for beyond-Limber (the Limber arguments are dummies).

    The linear spectrum is resolved from the backend's `linearperturbations`,
    wrapped to count the evaluations (`linear_calls`).
    """
    _, linear = cosmology
    backend = NonLinearStub(CountingPerturbations(linear))
    return SpectraBank(*LIMBER_ARGS, perturbations=backend, t_tildes=store)


def linear_calls(bank):
    """Evaluations of the linear spectra made by `bank` so far."""
    return bank.perturbations.linearperturbations.calls


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


def test_nonlimber_needs_t_tildes(cosmology):
    """Without T-tilde matrices the beyond-Limber methods fail clearly."""
    _, linear = cosmology
    bank = SpectraBank(*LIMBER_ARGS, perturbations=NonLinearStub(linear))
    for call in (
        bank.z_of_chi,
        lambda: bank.nonlimber_transfer(MATTER),
        bank.nonlimber_cheb,
        lambda: bank.w_ell("T_2_00"),
    ):
        with pytest.raises(ValueError, match="t_tildes"):
            call()


# ---------------------------------------------------------------------------
# Where the linear spectrum comes from
# ---------------------------------------------------------------------------
def test_linear_from_the_backend_linearperturbations(bank):
    """By default the linear spectrum is the backend's `linearperturbations`."""
    bank.nonlimber_transfer(MATTER)
    assert linear_calls(bank) == 1


def test_linear_backend_is_its_own_linear_source(cosmology, store, grid):
    """A linear backend passed as `perturbations` is used directly."""
    _, linear = cosmology
    bank = SpectraBank(*LIMBER_ARGS, perturbations=linear, t_tildes=store)
    np.testing.assert_allclose(
        bank.nonlimber_transfer(MATTER) ** 2,
        linear.matter_power_spectrum(bank.z_of_chi(), 10.0**grid.k_cheb),
        rtol=1e-12,
    )


def test_override_takes_precedence(cosmology, store):
    """`linear_perturbations=` wins over what the backend exposes."""
    _, linear = cosmology
    override = CountingPerturbations(linear)
    backend = NonLinearStub(CountingPerturbations(linear))
    bank = SpectraBank(
        *LIMBER_ARGS,
        perturbations=backend,
        linear_perturbations=override,
        t_tildes=store,
    )
    bank.nonlimber_transfer(MATTER)
    assert override.calls == 1 and backend.linearperturbations.calls == 0


def test_override_covers_backends_without_a_linear_source(cosmology, store):
    """A backend exposing no linear spectrum needs the override, and works with it.

    Mimics `CAMBNonLinearPerturbations`: no `linearperturbations`, and not
    a linear backend itself.
    """
    _, linear = cosmology
    opaque = object()
    bank = SpectraBank(*LIMBER_ARGS, perturbations=opaque, t_tildes=store)
    with pytest.raises(ValueError, match="exposes no `.linearperturbations`"):
        bank.nonlimber_transfer(MATTER)

    bank = SpectraBank(
        *LIMBER_ARGS,
        perturbations=opaque,
        linear_perturbations=linear,
        t_tildes=store,
    )
    assert bank.nonlimber_transfer(MATTER).shape == (N_CHI, N_CHEB)


def test_bank_without_perturbations_needs_the_override(cosmology, store):
    """A bank built from `matter_pk` alone has nothing to resolve from."""
    _, linear = cosmology
    bank = SpectraBank(*LIMBER_ARGS, t_tildes=store)
    with pytest.raises(ValueError, match="no `perturbations`"):
        bank.nonlimber_transfer(MATTER)

    bank = SpectraBank(*LIMBER_ARGS, linear_perturbations=linear, t_tildes=store)
    assert bank.nonlimber_transfer(MATTER).shape == (N_CHI, N_CHEB)


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

    T = jnp.sqrt(linear.matter_power_spectrum(bank.z_of_chi(), 10.0**grid.k_cheb))
    expected = np.asarray(unequal_time_pk(T, T, grid.chi, grid.R))  # (k, chi, R)

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
    assert linear_calls(bank) == 1


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

    assert linear_calls(bank) == 1  # one shared set of coefficients
    assert len(loads) == 2 * N_ELL
    assert not any(directory_name("T_0_20") in f for f in loads)


def test_w_ell_unknown_matrix(bank):
    """Asking for a matrix the dataset does not hold is a FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Missing T-tilde directory"):
        bank.w_ell("T_2_22")


# ---------------------------------------------------------------------------
# Transfer functions and pairs of fields
# ---------------------------------------------------------------------------
def test_nonlimber_transfer_is_the_sqrt_of_the_linear_pk(bank, cosmology, grid):
    """T_X(k, chi) = sqrt(P_XX^lin(k, z(chi))) at the Chebyshev nodes, per field."""
    _, linear = cosmology
    z, k = bank.z_of_chi(), 10.0**grid.k_cheb
    for field, method in (
        (MATTER, linear.matter_power_spectrum),
        (CB, linear.matter_power_spectrum_cb),
    ):
        T = bank.nonlimber_transfer(field)
        assert T.shape == (N_CHI, N_CHEB)
        np.testing.assert_allclose(T**2, method(z, k), rtol=1e-12)


def test_same_field_is_the_geometric_mean(bank, cosmology, grid):
    """For A = B the transfer-function form reproduces the geometric mean.

    sqrt(P(k, chi) P(k, R chi)) with P(k, R chi) from the Akima interpolation
    of log10 P: the formula used before the transfer functions were
    introduced, so this pins the matter-matter result to the previous one.
    """
    from cloelib.auxiliary.akima import akima_interpolation

    _, linear = cosmology
    Pk_chi = linear.matter_power_spectrum(bank.z_of_chi(), 10.0**grid.k_cheb)
    chi_R = jnp.outer(grid.chi, grid.R).flatten()
    Pk_chi_R = 10 ** akima_interpolation(jnp.log10(Pk_chi), grid.chi, chi_R, axis=0)
    Pk_chi_R = Pk_chi_R.reshape(N_CHI, N_R, N_CHEB)
    geometric = np.sqrt(np.einsum("ck,crk->kcr", Pk_chi, Pk_chi_R))

    T = bank.nonlimber_transfer(MATTER)
    np.testing.assert_allclose(
        unequal_time_pk(T, T, grid.chi, grid.R), geometric, rtol=1e-10
    )


def test_mixed_fields_need_both_orderings(bank):
    """cb at chi and matter at R chi differs from the reverse ordering.

    With massive neutrinos T_cb != T_m, and at R = 1 (equal times) the two
    orderings must coincide.
    """
    cb_m = bank.nonlimber_cheb(CB, MATTER)
    m_cb = bank.nonlimber_cheb(MATTER, CB)
    assert not np.allclose(cb_m, m_cb, rtol=1e-6)
    np.testing.assert_allclose(cb_m[:, :, -1], m_cb[:, :, -1], rtol=1e-10)


def test_fields_are_memoized_separately(bank):
    """Each ordered pair of fields has its own coefficients and w_ell.

    The transfer functions are shared: two spectrum evaluations (one per
    field) serve every pair.
    """
    w_mm = bank.w_ell("T_2_00")
    w_cbm = bank.w_ell("T_2_00", CB, MATTER)
    w_mcb = bank.w_ell("T_2_00", MATTER, CB)
    assert w_mm is bank.w_ell("T_2_00", MATTER, MATTER)
    assert w_cbm is bank.w_ell("T_2_00", CB, MATTER)
    assert w_cbm is not w_mcb and w_cbm is not w_mm
    bank.nonlimber_cheb(CB, CB)
    assert linear_calls(bank) == 2


def test_unknown_field(bank):
    """A field other than MATTER/CB is rejected before anything is computed."""
    for call in (
        lambda: bank.nonlimber_transfer("gas"),
        lambda: bank.nonlimber_cheb(MATTER, "gas"),
        lambda: bank.w_ell("T_2_00", "gas", MATTER),
    ):
        with pytest.raises(ValueError, match="Unknown traced field"):
            call()
    assert linear_calls(bank) == 0
