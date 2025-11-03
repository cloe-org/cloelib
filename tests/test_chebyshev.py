import pytest
import numpy as np
import jax.numpy as jnp

from cloelib.cosmology.jax_cosmology import JAXBackground

from cloelib.auxiliary.chebyshev import (
    comoving_distance_to_redshift,
    chebyshev_points,
    chebyshev_points_interval,
    chebyshev_coefficients,
    clenshaws_curtis_quadrature,
)


def test_comoving_distance_to_redshift():
    background = JAXBackground(
        H0=70.0,
        Omega_cdm0=0.25,
        Omega_b0=0.05,
        w0=-1,
        wa=0,
        Omega_k0=0.0,
        ns=0.96,
        As=2e-9,
        mnu=0.06,
        gamma_MG=0.545,
        N_mnu=1,
    )
    chi_true = np.linspace(26.0, 7000.0, 100)
    z_from_chi = comoving_distance_to_redshift(chi_true, background)
    chi_reconstructed = background.comoving_distance(z_from_chi)

    np.testing.assert_allclose(chi_true, chi_reconstructed, rtol=1e-6)


def test_chebyshev_points():
    n = 10
    expected_points = jnp.array(
        [
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
            -1.0,
        ]
    )
    computed_points = chebyshev_points(n)
    np.testing.assert_allclose(computed_points, expected_points, atol=1e-6, rtol=1e-6)


def test_chebyshev_points_interval():
    a, b = -5, 10
    n = 10
    expected_points = jnp.array(
        [
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
            -5.0,
        ]
    )
    computed_points = chebyshev_points_interval(n, a, b)
    np.testing.assert_allclose(computed_points, expected_points, atol=1e-5, rtol=1e-5)


def test_chebyshev_coefficients():
    x = chebyshev_points_interval(10, -5, 7)
    y = np.sin(x)
    cheb_coeff = chebyshev_coefficients(y)

    cheb_coeff_blast = jnp.array(
        [
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
            -0.01171998,
        ]
    )

    np.testing.assert_allclose(cheb_coeff, cheb_coeff_blast, rtol=1e-5)


def test_clenshaw_curtis_weights():
    n = 10
    a, b = -42, 55

    _, weights = clenshaws_curtis_quadrature(n, a, b)
    expected_weights = jnp.array(
        [
            0.5987654320987654,
            5.653521643743801,
            10.926289681898062,
            14.644091710758374,
            16.67733153150099,
            16.677331531500993,
            14.644091710758374,
            10.926289681898064,
            5.653521643743801,
            0.5987654320987654,
        ]
    )

    np.testing.assert_allclose(weights, expected_weights)


def test_clenshaw_curtis_quadrature():
    n = 10000
    a, b = 0.0, np.pi / 2

    clenshaw_curtis_grid, clenshaw_curtis_weights = clenshaws_curtis_quadrature(n, a, b)
    y = np.sin(clenshaw_curtis_grid)
    integral_approx = jnp.einsum("i,i->", y, clenshaw_curtis_weights)

    integral_exact = 1.0

    np.testing.assert_allclose(integral_approx, integral_exact, rtol=1e-4)
