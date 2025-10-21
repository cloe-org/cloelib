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
    c = c.at[N].multiply(0.5)
    return c

def chebyshev_interpolation(x: Array, c: Array) -> Array:
    """Evaluate Chebyshev interpolation at given points.

    Args:
        c: Chebyshev coefficients.
        x: Points where to evaluate the interpolation.

    Returns:
        Interpolated values at points x.
    """
    return np.polynomial.chebyshev.chebval(x, c)















