from cloelib.auxiliary.math_utils import (
    cached_stacked_simpson,
    simpsons_weights_avg,
    simpsons_weights_jit,
    simpsons_weights_odd,
    simpsons_weights_even,
    stacked_simpson,
    ensure_z_zero_included,
)
from scipy import integrate
from numpy.testing import assert_array_equal
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_simpson():
    assert_array_equal(simpsons_weights_jit(11), simpsons_weights_odd(11))
    assert_array_equal(simpsons_weights_jit(10), simpsons_weights_even(10))

    x = np.arange(0, 101)
    y = np.power(x, 3)
    w = simpsons_weights_jit(len(x))
    assert np.allclose(integrate.simpson(y, x=x), np.dot(y, w), rtol=1e-5)

    x = np.arange(0, 100)
    y = np.power(x, 3)
    w = simpsons_weights_jit(len(x))
    assert np.allclose(integrate.simpson(y, x=x), np.dot(y, w), rtol=1e-5)


# --- simpsons_weights_avg: parity-robust weights used by the Limber redshift integral ---


@pytest.mark.parametrize("n", range(1, 16))
def test_simpsons_weights_avg_properties(n):
    w = np.asarray(simpsons_weights_avg(n))
    assert w.shape == (n,)
    # constants and linear functions are integrated exactly (unit spacing, n - 1 intervals)
    x = np.arange(n, dtype=float)
    assert np.isclose(w.sum(), n - 1)
    assert np.isclose(np.dot(w, 2 * x + 1), (n - 1) ** 2 + (n - 1))
    np.testing.assert_allclose(w, w[::-1])
    if n >= 8:
        # no alternating 4/3, 2/3 Simpson pattern away from the ends
        np.testing.assert_allclose(w[3:-3], 1.0)


@pytest.mark.parametrize("n", [4, 6, 10, 100])
def test_simpsons_weights_avg_even_is_unchanged(n):
    np.testing.assert_allclose(
        simpsons_weights_avg(n), simpsons_weights_even(n), rtol=1e-14
    )


def test_simpsons_weights_avg_converges():
    errors = []
    for n in (51, 101, 201):
        x = np.linspace(0.0, np.pi, n)
        w = np.asarray(simpsons_weights_avg(n)) * (x[1] - x[0])
        errors.append(abs(np.dot(w, np.sin(x)) - 2.0))
    assert errors[0] < 1e-3
    assert errors[2] < errors[1] < errors[0]


# --- stacked_simpson / cached_stacked_simpson: cumulative weights of the window functions ---


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 11])
def test_stacked_simpson_rows(n):
    w = np.asarray(stacked_simpson(n))
    assert w.shape == (n, n)
    # row i integrates from node i to the last node: n - 1 - i intervals
    np.testing.assert_allclose(w.sum(axis=1), np.arange(n - 1, -1, -1, dtype=float))
    for i in range(n):
        np.testing.assert_array_equal(w[i, :i], 0.0)
        np.testing.assert_allclose(w[i, i:], simpsons_weights_avg(n - i))
    np.testing.assert_array_equal(w[-1], 0.0)  # a single node spans no interval
    if n >= 2:
        np.testing.assert_allclose(w[-2, -2:], [0.5, 0.5])  # two nodes: trapezoid


def test_cached_stacked_simpson_survives_retrace():
    """The cache must hold a concrete array, not a tracer from a previous trace."""

    @jax.jit
    def f(x):
        return cached_stacked_simpson(x.shape[0]) @ x

    x = jnp.arange(7.0)
    expected = np.asarray(stacked_simpson(7)) @ np.arange(7.0)
    np.testing.assert_allclose(f(x), expected)
    jax.clear_caches()  # forces a re-trace; used to raise UnexpectedTracerError
    np.testing.assert_allclose(f(x), expected)
    np.testing.assert_allclose(cached_stacked_simpson(7) @ x, expected)


def test_ensure_z_zero_included_adds_missing_zero():
    z, order = ensure_z_zero_included(np.array([1.0, 1.2, 1.4, 1.65]))
    assert_array_equal(z, [0.0, 1.0, 1.2, 1.4, 1.65])
    # z is already ascending, so the descending-to-z order is a plain reversal
    assert_array_equal(order, [4, 3, 2, 1, 0])


def test_ensure_z_zero_included_order_reorders_descending_camb_output():
    """`order` must map a quantity camb always returns sorted by descending
    z (regardless of input order) onto `z`'s own order, even when `z` is
    left unsorted (the `z_min <= 0.001` branch doesn't sort its input)."""
    redshifts = np.array([1.4, 1.0, 1.65, 1.2, 0.0])
    z, order = ensure_z_zero_included(redshifts)
    assert_array_equal(z, redshifts)

    camb_descending_z = np.sort(z)[::-1]
    camb_quantity = np.arange(len(z))[::-1]  # highest value at z=0, decreasing with z
    reordered = camb_quantity[order]

    for zi, value in zip(z, reordered):
        expected_value = camb_quantity[np.where(camb_descending_z == zi)[0][0]]
        assert value == expected_value
