import numpy as np
import jax.numpy as jnp

from cloelib.auxiliary.chebyshev import (
    chebyshev_points,
    chebyshev_points_interval,
    chebyshev_coefficients,
    clenshaw_curtis_quadrature,
)


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

    x = jnp.logspace(jnp.log10(1), jnp.log10(1000), 20)
    f = x**2
    coeffs = chebyshev_coefficients(f)

    blast_coeffs = jnp.array(
        [
            75543.76021970608,
            -143978.59106578442,
            126310.33988252498,
            -105188.83128233605,
            85680.27355126342,
            -69678.47497565328,
            57225.543407614256,
            -47720.80718239051,
            40488.50842236123,
            -34962.75175989318,
            30713.04064698487,
            -27425.10211314193,
            24871.833786479478,
            -22890.323098583416,
            21363.083885207016,
            -20206.107396030897,
            19359.340355971337,
            -18781.363851434588,
            18444.77584188734,
            -9167.147274751725,
        ]
    )

    np.testing.assert_allclose(coeffs, blast_coeffs, rtol=1e-5)


def test_chebyshev_coefficients_along_the_first_axis():
    """On an n-d array the expansion runs along axis 0, column by column.

    Matches the 1D expansion of every column, and the series evaluated with
    numpy reproduces the values at the nodes.
    """
    n = 12
    x = chebyshev_points_interval(n, -2.0, 3.0)
    f = jnp.stack(
        [jnp.sin(x)[:, None] * jnp.arange(1.0, 4.0), jnp.exp(x)[:, None] * jnp.ones(3)],
        axis=-1,
    )  # shape (n + 1, 3, 2)

    coeffs = chebyshev_coefficients(f)

    assert coeffs.shape == f.shape
    u = np.asarray(chebyshev_points(n))
    for i in range(3):
        for j in range(2):
            np.testing.assert_allclose(
                coeffs[:, i, j], chebyshev_coefficients(f[:, i, j]), rtol=1e-12
            )
            np.testing.assert_allclose(
                np.polynomial.chebyshev.chebval(u, np.asarray(coeffs[:, i, j])),
                f[:, i, j],
                rtol=1e-10,
            )


def test_clenshaw_curtis_weights():
    n = 10
    a, b = -42, 55

    _, weights = clenshaw_curtis_quadrature(n, a, b)
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

    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)


def test_clenshaw_curtis_quadrature():
    n = 10000
    a, b = 0.0, np.pi / 2

    clenshaw_curtis_grid, clenshaw_curtis_weights = clenshaw_curtis_quadrature(n, a, b)
    y = np.sin(clenshaw_curtis_grid)
    integral_approx = jnp.einsum("i,i->", y, clenshaw_curtis_weights)

    integral_exact = 1.0

    np.testing.assert_allclose(integral_approx, integral_exact, rtol=1e-4)
