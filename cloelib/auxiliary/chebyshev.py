"""Module for Chebyshev polynomial."""

import scipy
import jax
import numpy as np
import jax.numpy as jnp
from jax import Array


def chebyshev_points(n: int) -> Array:
    """Compute the Chebyshev points of the first kind.

    Args:
        n: Number of Chebyshev points to compute.
        dtype: The desired array type (jnp.ndarray or np.ndarray).

    Returns:
        An array of Chebyshev points of the specified type.
    """
    k = jnp.arange(n)
    points = jnp.cos(jnp.pi * (2 * k + 1) / (2 * n))

    return points

def chebyshev_points_interval(n: int, a: float, b: float) -> Array:
    """Compute Chebyshev points on the interval [a, b].

    Args:
        n: Number of Chebyshev points to compute.
        a: Start of the interval.
        b: End of the interval.

    Returns:
        An array of Chebyshev points on the interval [a, b].
    """
    cheb_pts = chebyshev_points(n)
    mapped_pts = 0.5 * (b - a) * (cheb_pts + 1) + a
    return mapped_pts

def chebyshev_coefficients(f_values: Array) -> Array:
    """Compute Chebyshev coefficients from function values at Chebyshev points.

    Args:
        f_values: Function values at Chebyshev points.
    Returns:
        Chebyshev coefficients.
    """
    N = len(f_values)
    c = jax.scipy.fft.dct(f_values, type=2, norm=None) / N
    c = c.at[0].multiply(0.5)
    #c = c.at[N].multiply(0.5)
    return c

def chebyshev_interpolation(x: Array, c: Array) -> Array:
    """Evaluate Chebyshev interpolation at given points.

    Args:
        c: Chebyshev coefficients.
        x: Points where to evaluate the interpolation.

    Returns:
        Interpolated values at points x.
    """
    x_scaled = (2 * x - (x.min() + x.max())) / (x.max() - x.min())
    return np.polynomial.chebyshev.chebval(x_scaled, c)

def clenshaws_curtis_quadrature(n: int, a: float, b: float) -> tuple[Array, Array]:
    """Compute the Clenshaw-Curtis quadrature of a function f on [a, b].

    Args:
        n: Number of Chebyshev points to use.
        a: Start of the interval.
        b: End of the interval.
    Returns:
        cheb_pts : Chebyshev points on [a, b].
        weights : Corresponding weights for Clenshaw-Curtis quadrature.
    """
    
    # Modified Chebyshev moments of the first kind
    mu = jnp.zeros(n, dtype=float)
    for i in range(0, n, 2):
        if i != 1:
            mu[i] = 2.0 / (1.0 - i**2)

    # Compute weights using the modified Chebyshev moments
    w = jax.scipy.fft.dct(mu, type=2, norm=None) / n
    w = w.at[0].multiply(0.5)
    #c = c.at[N].multiply(0.5)
    w = (b - a) / 2 * w

    return chebyshev_points_interval(n, a, b), w