import numpy as np
import jax
import jax.numpy as jnp
from cloelib.auxiliary.akima import _akima_slopes, _akima_coefficients

jax.config.update("jax_enable_x64", True)


def test_akima_slopes_basic():
    """Test _akima_slopes with simple linear data."""
    # Linear function
    t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    m = _akima_slopes(t, u)

    # For linear data, all slopes should be 1.0
    # m has length n+3 = 8
    # Interior slopes m[2:n+1] should all be 1.0
    assert m.shape == (8,)
    # Check interior slopes
    np.testing.assert_allclose(m[2:6], 1.0)


def test_akima_slopes_quadratic():
    """Test _akima_slopes with quadratic data."""
    t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = t**2

    m = _akima_slopes(t, u)

    # Slopes should be approximately diff(u)/diff(t) = [1, 3, 5, 7]
    # These go in m[2:6]
    expected_interior = jnp.array([1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(m[2:6], expected_interior)


def test_akima_coefficients_linear():
    """Test _akima_coefficients with linear data."""
    t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    m = _akima_slopes(t, u)
    b, c, d = _akima_coefficients(t, m)

    # For linear data:
    # - b should be all 1.0 (slope)
    # - c and d should be all 0.0 (no curvature)
    np.testing.assert_allclose(b, 1.0, rtol=1e-10)
    np.testing.assert_allclose(c, 0.0, atol=1e-10)
    np.testing.assert_allclose(d, 0.0, atol=1e-10)
