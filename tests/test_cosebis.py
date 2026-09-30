import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import j0, jv
from cosmolib.data import AngularPowerSpectrum, TwoPointCorrelationFunction

from cloelib.auxiliary.math_utils import quadrature_weights, simpsons_weights_jit
from cloelib.summary_statistics.angular_two_point import (
    get_cosebis_from_cl,
    get_cosebis_from_2pcf,
)

pytest.importorskip("pylevin")
pytest.importorskip("mpmath")
from cloelib.auxiliary.cosebi_helpers import (  # noqa: E402
    get_T_plus_minus,
    get_W_ell,
    get_roots_and_norms,
    tm,
    tp,
)

jax.config.update("jax_enable_x64", True)

N_MODES = 3
KEY = ("SHE", "SHE", 1, 1)
THETA = np.radians(np.geomspace(1.0, 400.0, 400) / 60)
ELLS_LIN = np.arange(1, 5001).astype(float)
ELLS_LOG = np.geomspace(1, 5000, 2000)


def cl_ee(ells):
    """Toy E-mode spectrum, damped so the Hankel transforms converge."""
    return 1e-9 * (ells / 100.0) ** -1.2 * np.exp(-((ells / 1000.0) ** 2))


def cosebis_from_cl(ells):
    w_ell = get_W_ell(THETA, N_MODES, ells, 4)
    arr = jnp.zeros((2, 2, ells.size)).at[0, 0].set(cl_ee(ells))
    cells = {KEY: AngularPowerSpectrum(array=arr, ell=jnp.asarray(ells))}
    ns = jnp.arange(1, N_MODES + 1)
    return get_cosebis_from_cl(cells, jnp.asarray(ells), w_ell, ns)[KEY]


@pytest.fixture(scope="module")
def cosebis_lin():
    return cosebis_from_cl(ELLS_LIN)


@pytest.fixture(scope="module")
def xi_and_kernels():
    """xi_+/- of the toy spectrum and the T_+/- kernels on THETA."""
    ells = ELLS_LIN
    lt = np.outer(THETA, ells)
    weight = ells * cl_ee(ells) / (2 * np.pi)
    xi_plus = j0(lt) @ weight
    xi_minus = jv(4, lt) @ weight

    T_plus, T_minus = get_T_plus_minus(THETA, N_MODES)
    return xi_plus, xi_minus, T_plus, T_minus


def test_T_plus_minus_match_mpmath():
    # Float64 Chebyshev kernels against the full-precision mpmath expressions
    theta = np.radians(np.geomspace(0.5, 300.0, 50) / 60)
    nmax = 15
    T_plus, T_minus = get_T_plus_minus(theta, nmax)
    assert T_plus.shape == T_minus.shape == (nmax + 1, theta.size)
    np.testing.assert_array_equal(T_plus[0], 0)
    rn, nn, coeff_j = get_roots_and_norms(theta[-1], theta[0], nmax)
    for n in (1, 7, nmax):
        ref_p = np.array([float(v) for v in tp(n, theta, theta[0], nn, rn)])
        ref_m = np.array([float(v) for v in tm(n, theta, theta[0], nn, coeff_j)])
        np.testing.assert_allclose(
            T_plus[n], ref_p, rtol=0, atol=1e-12 * np.abs(ref_p).max()
        )
        np.testing.assert_allclose(
            T_minus[n], ref_m, rtol=0, atol=1e-12 * np.abs(ref_m).max()
        )


def test_quadrature_weights():
    # Unit spacing reproduces the plain Simpson weights
    x = np.arange(1, 101).astype(float)
    np.testing.assert_array_equal(quadrature_weights(x), simpsons_weights_jit(100))

    f = lambda x: x**-1.5  # noqa: E731
    exact = lambda a, b: 2 * (a**-0.5 - b**-0.5)  # noqa: E731
    for x, rel in [
        (np.linspace(1, 100, 1001), 1e-5),  # linear: Simpson
        (np.geomspace(1, 100, 1001), 1e-5),  # logarithmic: Simpson in ln x
        (np.sort(np.random.default_rng(0).uniform(1, 100, 5000)), 1e-3),  # trapezoid
    ]:
        integral = np.dot(quadrature_weights(x), f(x))
        assert integral == pytest.approx(exact(x[0], x[-1]), rel=rel)


def test_cosebis_from_cl_independent_of_ell_grid(cosebis_lin):
    cosebis_log = cosebis_from_cl(ELLS_LOG)
    np.testing.assert_allclose(
        cosebis_log.array[0, 0], cosebis_lin.array[0, 0], rtol=1e-4
    )


def test_cosebis_from_cl_thetas_in_arcmin(cosebis_lin):
    # Same unit as the Euclid LE3 products read by euclidlib
    assert cosebis_lin.thmin == pytest.approx(1.0)
    assert cosebis_lin.thmax == pytest.approx(400.0)


def test_cosebis_from_2pcf_matches_cl(cosebis_lin, xi_and_kernels):
    xi_plus, xi_minus, T_plus, T_minus = xi_and_kernels
    arr = jnp.zeros((2, 2, THETA.size)).at[0, 0].set(xi_plus).at[1, 1].set(xi_minus)
    twopcf = {
        KEY: TwoPointCorrelationFunction(array=arr, theta=jnp.asarray(THETA)),
        ("POS", "POS", 1, 1): TwoPointCorrelationFunction(
            array=jnp.asarray(xi_plus), theta=jnp.asarray(THETA)
        ),
    }
    out = get_cosebis_from_2pcf(
        twopcf, jnp.asarray(THETA), T_plus, T_minus, jnp.arange(1, N_MODES + 1)
    )

    # Only SHE-SHE pairs are processed
    assert list(out) == [KEY]
    cosebi = out[KEY]
    assert cosebi.array.shape == (2, 2, N_MODES)
    np.testing.assert_array_equal(cosebi.mode, np.arange(1, N_MODES + 1))
    assert cosebi.thmin == pytest.approx(1.0)  # arcmin
    assert cosebi.thmax == pytest.approx(400.0)

    # Real- and harmonic-space routes agree; no B-modes from a pure E spectrum
    ee = np.asarray(cosebi.array[0, 0])
    np.testing.assert_allclose(ee, cosebis_lin.array[0, 0], rtol=1e-4)
    np.testing.assert_allclose(np.asarray(cosebi.array[1, 1]) / ee, 0, atol=1e-4)
