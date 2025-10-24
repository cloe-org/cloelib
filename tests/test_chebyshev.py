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
        1.0,
        0.9510565162951536,
        0.8090169943749475,
        0.5877852522924731,
        0.30901699437494745,
        0.0,
        -0.30901699437494734,
        -0.587785252292473,
        -0.8090169943749473,
        -0.9510565162951535,
        -1.0
    ])
    computed_points = chebyshev_points(n)
    np.testing.assert_allclose(computed_points, expected_points, atol=1e-6, rtol=1e-6)


def test_chebyshev_points_interval():
    a, b = -5, 10
    n = 10
    expected_points = jnp.array([
        10.0,
        9.632923872213652,
        8.567627457812106,
        6.908389392193548,
        4.817627457812106,
        2.5,
        0.18237254218789456,
        -1.9083893921935475,
        -3.567627457812105,
        -4.632923872213651,
        -5.0
    ])
    computed_points = chebyshev_points_interval(n, a, b)
    np.testing.assert_allclose(computed_points, expected_points, atol=1e-5, rtol=1e-5)

"""
def test_chebyshev_interpolation():
    x_start, x_stop = -5, 10
    x_eval = jnp.linspace(x_start, x_stop, 99)

    x = chebyshev_points_interval(40, x_start, x_stop)
    y = jnp.sin(x)

    cheby_coeff = chebyshev_coefficients(y)
    y_approx = chebyshev_interpolation(x_eval, cheby_coeff)

    np.testing.assert_allclose(y_approx, jnp.sin(x_eval), rtol=1e-5)
"""