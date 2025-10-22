import pytest
import numpy as np
import jax.numpy as jnp

from cloelib.auxiliary.chebyshev import (
    chebyshev_points,
    chebyshev_points_interval,
    chebyshev_coefficients,
    chebyshev_interpolation,
)

def test_chebyshev_points():
    n = 10
    expected_points = jnp.array([
        -jnp.cos(jnp.pi/20),
        -jnp.cos(3*jnp.pi/20),
        -1/jnp.sqrt(2),
        -jnp.sin(3*jnp.pi/20),
        -jnp.sin(jnp.pi/20),
        jnp.sin(jnp.pi/20),
        jnp.sin(3*jnp.pi/20),
        1/jnp.sqrt(2),
        jnp.cos(3*jnp.pi/20),
        jnp.cos(jnp.pi/20),
    ])[::-1]
    computed_points = chebyshev_points(n)
    np.testing.assert_allclose(computed_points, expected_points, rtol=1e-6)

def test_chebyshev_points_interval():
    a, b = -5, 7
    n = 10
    expected_points = jnp.array([
        -jnp.cos(jnp.pi/20),
        -jnp.cos(3*jnp.pi/20),
        -1/jnp.sqrt(2),
        -jnp.sin(3*jnp.pi/20),
        -jnp.sin(jnp.pi/20),
        jnp.sin(jnp.pi/20),
        jnp.sin(3*jnp.pi/20),
        1/jnp.sqrt(2),
        jnp.cos(3*jnp.pi/20),
        jnp.cos(jnp.pi/20),
    ])[::-1]
    expected_points = 0.5 * (b - a) * (expected_points + 1) + a
    computed_points = chebyshev_points_interval(n, a, b)
    np.testing.assert_allclose(computed_points, expected_points, rtol=1e-5)