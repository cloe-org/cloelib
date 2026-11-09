import numpy as np
import pytest
from scipy.special import gamma, hyp1f1

from cloelib.summary_statistics.angular_correlation_function_fftlog import (
    CorrelationTransform,
    ell_grid,
    hankel_fftlog,
)


def _log_bins(theta_min, theta_max, n):
    edges = np.radians(np.geomspace(theta_min, theta_max, n + 1) / 60)
    return np.sqrt(edges[:-1] * edges[1:]), edges


def _gaussian_closed_form(theta, nu, amplitude, sigma):
    # Hankel transform of C_l = A exp(-l^2 / 2 sigma^2) (Gradshteyn & Ryzhik 6.631.1).
    p = 0.5 / sigma**2
    return (
        amplitude
        / (2 * np.pi)
        * gamma(nu / 2 + 1)
        * theta**nu
        / (2 ** (nu + 1) * p ** (nu / 2 + 1) * gamma(nu + 1))
        * hyp1f1(nu / 2 + 1, nu + 1, -(theta**2) / (4 * p))
    )


def test_hankel_gaussian_pair():
    # exp(-x^2/2) is its own Hankel transform of order 0.
    x = np.logspace(-3, 3, 4096)
    y, g = hankel_fftlog(x, np.exp(-0.5 * x**2), nu=0)
    inside = (y > 0.1) & (y < 3.0)
    np.testing.assert_allclose(g[inside], np.exp(-0.5 * y[inside] ** 2), atol=1e-4)
    assert y.size == x.size


@pytest.mark.parametrize("nu", [0, 2, 4])
def test_closed_form(nu):
    theta, _ = _log_bins(0.5, 20.0, 12)
    transform = CorrelationTransform(ell_grid(40000, 2048), theta)
    amplitude, sigma = 1e-8, 600.0
    c_ell = amplitude * np.exp(-0.5 * (transform.ells / sigma) ** 2)
    exact = _gaussian_closed_form(theta, nu, amplitude, sigma)
    np.testing.assert_allclose(transform(c_ell, nu), exact, rtol=1e-4)


def test_ell_grid():
    ells = ell_grid(40000, 2048)
    assert ells.size == 2048 and ells[0] == 2.0 and ells[-1] == 40000.0
    np.testing.assert_allclose(np.diff(np.log(ells)), np.diff(np.log(ells))[0])


def test_bin_average():
    theta, edges = _log_bins(0.5, 20.0, 12)
    amplitude, sigma = 1e-8, 600.0
    averaged = CorrelationTransform(ell_grid(), theta, theta_edges=edges, n_sub=16)
    c_ell = amplitude * np.exp(-0.5 * (averaged.ells / sigma) ** 2)
    expected = []
    for low, high in zip(edges[:-1], edges[1:]):
        fine = np.linspace(low, high, 2001)
        values = _gaussian_closed_form(fine, 2, amplitude, sigma)
        expected.append(np.trapezoid(values * fine, fine) / np.trapezoid(fine, fine))
    np.testing.assert_allclose(averaged(c_ell, 2), expected, rtol=1e-3)
    with pytest.raises(ValueError, match="needs their edges"):
        CorrelationTransform(ell_grid(), theta, n_sub=4)
    with pytest.raises(ValueError, match="do not bound"):
        CorrelationTransform(ell_grid(), theta, theta_edges=edges[:-1], n_sub=4)


def test_widen_tail_and_sign():
    theta, _ = _log_bins(2.5, 250.0, 20)
    transform = CorrelationTransform(ell_grid(40000, 512), theta)
    # A spectrum crossing zero at the top of the grid must not blow up.
    c_ell = 1e-9 / (1 + (transform.ells / 500.0) ** 2) - 2e-14
    assert np.isfinite(transform._widen(c_ell)).all()
    assert np.isfinite(transform(c_ell, 0)).all()
    # The transform is linear.
    np.testing.assert_allclose(transform(2 * c_ell, 4), 2 * transform(c_ell, 4))
