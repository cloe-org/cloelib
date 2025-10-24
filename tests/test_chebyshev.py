import pytest
import numpy as np
import jax.numpy as jnp

from cloelib.auxiliary.chebyshev import (
    chebyshev_points,
    chebyshev_points_interval,
    chebyshev_coefficients,
    chebyshev_interpolation,
    clenshaws_curtis_quadrature,
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


def test_chebyshev_coefficients():

    x = chebyshev_points_interval(10, -5, 7)
    y = np.sin(x)
    cheb_coeff = chebyshev_coefficients(y)

    cheb_coeff_blast = jnp.array([ 
        0.12676361, 
        -0.29898586,  
        0.40874146, 
        -0.12401901,  
        0.60189207,
        0.39126627, 
        -0.41377925, 
        -0.13988857,
        0.09605752, 
        0.02065833,
        -0.01171998])

    np.testing.assert_allclose(cheb_coeff, cheb_coeff_blast, rtol=1e-5)

def test_clencurtis_quadrature():
    n = 10000
    a, b = 0.0, np.pi/2

    clencur_grid, clencur_weights = clenshaws_curtis_quadrature(n, a, b)
    y = np.sin(clencur_grid)
    integral_approx = jnp.einsum('i,i->', y, clencur_weights)

    integral_exact = 1.0 

    np.testing.assert_allclose(integral_approx, integral_exact, rtol=1e-5)