"""Tests for `beyond_limber.nonlimber_cl` and `beyond_limber.interpolate_correction`.

Built, like `test_spectra_bank_nonlimber.py`, from a real (JAX) linear
cosmology and a tiny fake T-tilde dataset with random matrices: these tests
check how the pairs of contributions are assembled (T-tilde names, fields,
orderings, prefactors), not the physics of the matrices. The
`interpolate_correction` tests at the end are pure array tests on the
multipole nodes of the standard dataset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cloelib.auxiliary.chebyshev import chebyshev_points_interval
from cloelib.auxiliary.nonlimber_kernels import kernel_on_chi_R, pair_integral
from cloelib.auxiliary.t_tildes import NonLimberGrid, TTildeStore, directory_name
from cloelib.cosmology.jax_cosmology import JAXBackground, JAXLinearPerturbations
from cloelib.observables.photo.contributions import (
    CB,
    MATTER,
    NonLimberSpec,
    nonlimber_contributions,
    ell_prefactor_magnification,
    ell_prefactor_one,
    ell_prefactor_rsd,
)
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.positions import RSDContribution
from cloelib.observables.photo.spectrum_engine import SpectraBank
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.summary_statistics.beyond_limber import (
    interpolate_correction,
    nonlimber_cl,
)

jax.config.update("jax_enable_x64", True)

N_ELL, N_CHI, N_R, N_CHEB = 3, 6, 4, 9
FAKE_ELLS = np.array([2.0, 14.3, 97.1])
# Every matrix a pair of the contributions below can ask for.
T_NAMES = (
    "T_2_00",
    "T_2_02",
    "T_2_20",
    "T_2_22",
    "T_0_00",
    "T_0_02",
    "T_0_20",
    "T_minus2_00",
)
Z = jnp.linspace(0.0, 1.0, 200)

DENSITY = NonLimberSpec(1, 0, ell_prefactor_one, False)
RSD = NonLimberSpec(1, 2, ell_prefactor_rsd, False)
LENSING = NonLimberSpec(-1, 0, ell_prefactor_magnification, True)


@pytest.fixture(scope="module")
def perturbations():
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
    return JAXLinearPerturbations(background)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    grid = NonLimberGrid(
        ells=jnp.array(FAKE_ELLS),
        chi=jnp.linspace(100.0, 2000.0, N_CHI),
        R=jnp.linspace(0.3, 1.0, N_R),
        k_cheb=chebyshev_points_interval(N_CHEB - 1, -3.0, 0.0),
        w_chi=jnp.linspace(1.0, 2.0, N_CHI),
        w_R=jnp.linspace(0.5, 1.0, N_R),
    )
    path = tmp_path_factory.mktemp("t_tildes")
    rng = np.random.default_rng(0)
    for name in T_NAMES:
        folder = path / directory_name(name)
        folder.mkdir()
        for ell in FAKE_ELLS:
            block = rng.normal(size=(1, N_CHI, N_R, N_CHEB))
            np.save(folder / f"T_tilde_l_{ell:.1f}.npy", block)
    return TTildeStore(path, grid=grid)


@pytest.fixture
def bank(perturbations, store):
    return SpectraBank(None, None, None, perturbations, t_tildes=store)


class FakeContribution:
    """Gaussian bins in z, with a given spec and traced field."""

    def __init__(self, spec, means, field=MATTER):
        self.nonlimber_spec = spec
        self.field = field
        self.means = jnp.asarray(means)
        self.calls = 0

    def compute_kernel(self, z):
        self.calls += 1
        return jnp.exp(-0.5 * ((z[None, :] - self.means[:, None]) / 0.1) ** 2)


def by_hand(c1, c2, name_12, name_21, bank):
    """`pair_integral` of one pair, with names and fields written out."""
    grid = bank.t_tildes.grid
    K1, K1_R = kernel_on_chi_R(
        c1.compute_kernel(Z),
        Z,
        bank.z_of_chi(),
        grid.chi,
        grid.R,
        c1.nonlimber_spec.divide_by_chi2,
    )
    K2, K2_R = kernel_on_chi_R(
        c2.compute_kernel(Z),
        Z,
        bank.z_of_chi(),
        grid.chi,
        grid.R,
        c2.nonlimber_spec.divide_by_chi2,
    )
    return pair_integral(
        K1,
        K2,
        K1_R,
        K2_R,
        bank.projected_density(name_12, c1.field, c2.field),
        bank.projected_density(name_21, c2.field, c1.field),
        c1.nonlimber_spec.ell_prefactor(grid.ells)
        * c2.nonlimber_spec.ell_prefactor(grid.ells),
        grid.w_chi,
        grid.w_R,
    )


def mixed_contributions():
    """Density (cb), RSD (cb), lensing (matter): every kind of pair."""
    return (
        FakeContribution(DENSITY, [0.3, 0.5], field=CB),
        FakeContribution(RSD, [0.3, 0.5], field=CB),
        FakeContribution(LENSING, [0.4, 0.6]),
    )


def test_one_pair_matches_pair_integral(bank):
    """Density (cb) x RSD (matter): the names and fields of both orderings."""
    density = FakeContribution(DENSITY, [0.3, 0.5], field=CB)
    rsd = FakeContribution(RSD, [0.4])
    expected = by_hand(density, rsd, "T_2_02", "T_2_20", bank)

    result = nonlimber_cl([density], [rsd], Z, bank)

    assert result.shape == (N_ELL, 2, 1)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_sum_over_pairs(bank):
    """The result is the sum of the single-pair results."""
    first = mixed_contributions()
    second = mixed_contributions()

    result = nonlimber_cl(first, second, Z, bank)

    expected = sum(nonlimber_cl([a], [b], Z, bank) for a in first for b in second)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_swapping_the_tracers_transposes(bank):
    """C^{XY}_{ij} = C^{YX}_{ji}, also for RSD pairs and mixed fields."""
    first = mixed_contributions()
    second = (
        FakeContribution(LENSING, [0.35, 0.45, 0.65]),
        FakeContribution(RSD, [0.45, 0.55, 0.65], field=MATTER),
    )

    forward = nonlimber_cl(first, second, Z, bank)
    backward = nonlimber_cl(second, first, Z, bank)

    np.testing.assert_allclose(forward, jnp.swapaxes(backward, 1, 2), rtol=1e-10)


def test_kernels_computed_once_per_contribution(bank):
    """The same contribution object in both sets is resampled once."""
    contributions = mixed_contributions()

    nonlimber_cl(contributions, contributions, Z, bank)

    assert [c.calls for c in contributions] == [1, 1, 1]


def test_contribution_without_spec_raises(bank):
    class NoSpec:
        def compute_kernel(self, z):
            return jnp.ones((1, len(z)))

    with pytest.raises(NotImplementedError, match="NoSpec"):
        nonlimber_cl([FakeContribution(DENSITY, [0.5])], [NoSpec()], Z, bank)


@pytest.mark.parametrize("use_Pcb, field", [(False, MATTER), (True, CB)])
def test_rsd_traces_the_galaxy_field(use_Pcb, field):
    """As in the Limber RSD window: cb with `use_Pcb=True`."""

    class Tracer:
        pass

    tracer = Tracer()
    tracer.use_Pcb = use_Pcb
    assert RSDContribution(tracer).field == field


# ---------------------------------------------------------------------------
# AngularTwoPoint: multiplicative shear bias
# ---------------------------------------------------------------------------
KS = jnp.logspace(-4, 1, 100)
M_BIAS = (0.01, -0.02)
# Tracer grid: `PositionsTracer`/`ShearTracer` reject z = 0.
Z_TRACER = jnp.linspace(0.01, 1.0, 200)


@pytest.fixture(scope="module")
def tracer_perturbations(perturbations):
    """Linear backend with the `.z`/`.k` grids `AngularTwoPoint` reads."""
    linear = JAXLinearPerturbations(perturbations.background)
    linear.z = jnp.linspace(0.0, 1.5, 40)
    linear.k = KS
    return linear


def _dndz():
    dndz = jnp.exp(
        -0.5 * ((Z_TRACER[None, :] - jnp.array([0.3, 0.6])[:, None]) / 0.1) ** 2
    )
    return dndz / jnp.trapezoid(dndz, Z_TRACER, axis=1)[:, None]


def _positions(perturbations, include_rsd=False):
    nuisance = {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(2)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(2)},
        **{f"magnification_bias_{i + 1}": 0.4 for i in range(2)},
        **{f"b1_photo_bin{i}": b for i, b in enumerate((1.3, 1.7))},
    }
    return PositionsTracer(
        perturbations,
        _dndz(),
        Z_TRACER,
        "per_bin",
        nuisance,
        include_rsd=include_rsd,
    )


def _shear(perturbations, m_bias):
    nuisance = {
        **{f"multiplicative_bias_{i + 1}": m for i, m in enumerate(m_bias)},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(2)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(2)},
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": -0.41,
    }
    return ShearTracer(
        perturbations=perturbations,
        dndz=_dndz(),
        z=Z_TRACER,
        nuisance_params=nuisance,
        ia_model="NLA",
    )


def _two_point(kind, perturbations, m_bias):
    if kind == "gg":
        g = _positions(perturbations, include_rsd=True)
        return AngularTwoPoint(g, g)
    if kind == "gs":
        return AngularTwoPoint(
            _positions(perturbations, include_rsd=True), _shear(perturbations, m_bias)
        )
    s = _shear(perturbations, m_bias)
    return AngularTwoPoint(s, s)


def _calibration(kind):
    """(1 + m1_i)(1 + m2_j) of the pair `kind`, shape `(1, 2, 2)`."""
    one_plus_m = 1.0 + np.array(M_BIAS)
    m1 = one_plus_m if kind == "ss" else np.ones(2)
    m2 = np.ones(2) if kind == "gg" else one_plus_m
    return (m1[:, None] * m2[None, :])[None]


@pytest.mark.parametrize("kind", ["gg", "gs", "ss"])
def test_nonlimber_cl_method_applies_the_multiplicative_bias(
    kind, tracer_perturbations, store
):
    """`_nonlimber_cl` is `nonlimber_cl` of the raw kernels times (1+m_i)(1+m_j).

    The galaxy tracers have RSD: it enters the beyond-Limber Cl through
    `nonlimber_contributions`, as it enters the Limber one through its window.
    """
    two_point = _two_point(kind, tracer_perturbations, M_BIAS)
    bank = two_point._linear_bank(KS, t_tildes=store)

    raw = nonlimber_cl(
        nonlimber_contributions(two_point.tracer1),
        nonlimber_contributions(two_point.tracer2),
        Z_TRACER,
        bank,
    )
    np.testing.assert_allclose(
        two_point._nonlimber_cl(bank), raw * _calibration(kind), rtol=1e-12
    )


@pytest.mark.parametrize("kind", ["gs", "ss"])
def test_limber_and_nonlimber_have_the_same_calibration(
    kind, tracer_perturbations, store
):
    """The multiplicative bias scales the two Cl, hence their difference, alike.

    The linear Limber Cl gets it from `ShearTracer.get_window`, the
    beyond-Limber one from `_apply_multiplicative_bias`.
    """
    ells = store.grid.ells
    out = {}
    for label, m_bias in [("m", M_BIAS), ("no_m", (0.0, 0.0))]:
        two_point = _two_point(kind, tracer_perturbations, m_bias)
        bank = two_point._linear_bank(KS, t_tildes=store)
        out[label] = (
            two_point._nonlimber_cl(bank),
            two_point._limber_linear_cl(ells, KS, bank),
        )

    for with_m, without_m in zip(out["m"], out["no_m"]):
        np.testing.assert_allclose(with_m, without_m * _calibration(kind), rtol=1e-12)


def test_nonlimber_cl_method_includes_rsd(tracer_perturbations, store):
    """With `include_rsd=True` the RSD term changes the beyond-Limber Cl."""
    g_rsd = _positions(tracer_perturbations, include_rsd=True)
    g = _positions(tracer_perturbations)
    bank = AngularTwoPoint(g, g)._linear_bank(KS, t_tildes=store)
    with_rsd = AngularTwoPoint(g_rsd, g_rsd)._nonlimber_cl(bank)
    without_rsd = AngularTwoPoint(g, g)._nonlimber_cl(bank)
    assert not np.allclose(with_rsd, without_rsd, rtol=1e-6, atol=0.0)


def test_nonlimber_correction_is_the_difference_on_the_nodes(
    tracer_perturbations, store
):
    """On the nodes up to ell_max: nonlimber - linear Limber; zero above and below."""
    two_point = _two_point("gs", tracer_perturbations, M_BIAS)
    bank = two_point._linear_bank(KS, t_tildes=store)
    nodes = store.grid.ells
    ell_max = float(nodes[1])

    delta = two_point._nonlimber_cl(bank) - two_point._limber_linear_cl(nodes, KS, bank)
    ells = jnp.concatenate([jnp.array([0.0, 1.0]), nodes])
    out = two_point._nonlimber_correction(ells, KS, bank, ell_max)

    assert out.shape == (len(ells), 2, 2)
    np.testing.assert_array_equal(out[:2], 0.0)
    np.testing.assert_allclose(out[2:4], delta[:2], rtol=1e-10)
    np.testing.assert_array_equal(out[4:], 0.0)


# ---------------------------------------------------------------------------
# interpolate_correction
# ---------------------------------------------------------------------------
# Multipole nodes of the standard dataset (no matrices are read).
BLAST_NODES = NonLimberGrid.blast_128_64_161().ells
ELL_MAX = 200.0


def smooth(ells):
    """Smooth in log ell, decaying and changing sign (as a cross spectrum can)."""
    x = jnp.log(ells)
    return jnp.cos(x) * jnp.exp(-0.3 * x)


def test_interpolation_reproduces_the_nodes_up_to_ell_max():
    delta = smooth(BLAST_NODES)
    nodes_in = BLAST_NODES[BLAST_NODES <= ELL_MAX]
    np.testing.assert_allclose(
        interpolate_correction(delta, BLAST_NODES, nodes_in, ELL_MAX),
        delta[BLAST_NODES <= ELL_MAX],
        rtol=1e-12,
        atol=1e-15,
    )


def test_interpolation_zero_outside_the_range():
    """Zero above ell_max and below the first node (ell = 0, 1)."""
    ells = jnp.array([0.0, 1.0, 1.5, 200.5, 211.0, 500.0])
    out = interpolate_correction(smooth(BLAST_NODES), BLAST_NODES, ells, ELL_MAX)
    np.testing.assert_array_equal(out, 0.0)


def test_interpolation_accurate_between_the_nodes():
    """Values, not logs: fine where the function crosses zero."""
    ells = jnp.arange(2.0, ELL_MAX + 1.0)
    out = interpolate_correction(smooth(BLAST_NODES), BLAST_NODES, ells, ELL_MAX)
    np.testing.assert_allclose(out, smooth(ells), atol=5e-3)
    assert np.any(np.diff(np.sign(np.asarray(smooth(ells)))) != 0)


def test_interpolation_of_bins_is_independent():
    """Extra axes are carried along: each bin pair is its own function."""
    delta = jnp.stack(
        [smooth(BLAST_NODES), 2.0 * smooth(BLAST_NODES), -smooth(BLAST_NODES)], axis=-1
    )
    delta = delta[:, None, :] * jnp.array([1.0, 3.0])[None, :, None]
    ells = jnp.array([0.0, 2.0, 7.3, 150.0, 300.0])
    out = interpolate_correction(delta, BLAST_NODES, ells, ELL_MAX)
    assert out.shape == (len(ells), 2, 3)
    single = interpolate_correction(smooth(BLAST_NODES), BLAST_NODES, ells, ELL_MAX)
    np.testing.assert_allclose(out[:, 1, 2], -3.0 * single, rtol=1e-12)


def test_interpolation_gradient_is_finite_with_masked_queries():
    """jax.grad through the mask stays finite at ell = 0 (log ell = -inf)."""
    ells = jnp.arange(0.0, 300.0)

    def loss(delta):
        return jnp.sum(interpolate_correction(delta, BLAST_NODES, ells, ELL_MAX) ** 2)

    grad = jax.grad(loss)(smooth(BLAST_NODES))
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)


def test_interpolation_jittable():
    ells = jnp.arange(0.0, 300.0)
    f = jax.jit(lambda d, e: interpolate_correction(d, BLAST_NODES, e, ELL_MAX))
    np.testing.assert_allclose(
        f(smooth(BLAST_NODES), ells),
        interpolate_correction(smooth(BLAST_NODES), BLAST_NODES, ells, ELL_MAX),
        rtol=1e-12,
    )


@pytest.mark.parametrize("ell_max", [1.9, 211.7, 1000.0])
def test_interpolation_ell_max_must_lie_within_the_nodes(ell_max):
    with pytest.raises(ValueError, match="ell_max"):
        interpolate_correction(
            smooth(BLAST_NODES), BLAST_NODES, jnp.array([10.0]), ell_max
        )
