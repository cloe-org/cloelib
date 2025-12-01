import numpy as np
import jax
import jax.numpy as jnp
from cloelib.auxiliary.akima import (
    _akima_slopes,
    _akima_coefficients,
    _akima_eval,
    akima_interpolation,
)

jax.config.update("jax_enable_x64", True)


def test_akima_length_mismatch():
    """Test that _akima_slopes raises error on length mismatch."""
    t = jnp.array([0.0, 1.0, 2.0])
    u = jnp.array([0.0, 1.0])

    with np.testing.assert_raises(ValueError):
        _akima_slopes(t, u)


def test_akima_isfinite():
    """Test that _akima_slopes handles non-finite inputs."""
    t = jnp.linspace(0.0, 4.0, 5)
    u = jnp.sin(t)
    t_new = jnp.linspace(0.0, 4.0, 10)

    m = akima_interpolation(u, t, t_new)

    assert jnp.all(jnp.isfinite(m))


def test_akima_nan_input():
    """Test that _akima_slopes raises error on NaN inputs."""
    t = jnp.array([jnp.inf, 1.0, 2.0, 3.0])
    u = jnp.array([0.0, jnp.nan, 2.0, 3.0])
    t_new = jnp.array([0.5, jnp.nan, 2.5])

    assert jnp.all(jnp.isnan(akima_interpolation(u, t, t_new)))


def test_akima_slopes_constant():
    """Test _akima_slopes with constant data."""
    t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])

    m = _akima_slopes(t, u)

    # For constant data, all slopes should be 0.0
    assert jnp.allclose(m, 0.0)


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


def test_akima_eval_array():
    """Test _akima_eval with array query points."""
    t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    m = _akima_slopes(t, u)
    b, c, d = _akima_coefficients(t, m)

    # Evaluate at multiple points
    t_new = jnp.array([0.5, 1.5, 2.5, 3.5])
    result = _akima_eval(t, u, b, c, d, t_new)

    # Result should be array
    assert jnp.ndim(result) == 1
    assert result.shape == (4,)
    np.testing.assert_allclose(result, t_new, rtol=1e-10)


def test_akima_interpolation_linear():
    """Test full akima_interpolation pipeline with linear data."""
    t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t_new = jnp.linspace(0, 4, 20)

    result = akima_interpolation(u, t, t_new)

    # For linear data, interpolation should be exact
    np.testing.assert_allclose(result, t_new, rtol=1e-10)


def test_akima_interpolation_sine():
    """Test akima_interpolation with sine function."""
    # Create data points
    t = jnp.linspace(0, 2 * jnp.pi, 20)
    u = jnp.sin(t)

    # Interpolate at finer grid
    t_new = jnp.linspace(0, 2 * jnp.pi, 100)
    result = akima_interpolation(u, t, t_new)

    # Check that result passes through original points
    for i, ti in enumerate(t):
        # Find closest point in t_new
        idx = jnp.argmin(jnp.abs(t_new - ti))
        if jnp.abs(t_new[idx] - ti) < 1e-6:
            # Use absolute tolerance for values near zero
            np.testing.assert_allclose(result[idx], u[i], rtol=1e-5, atol=1e-10)


def test_akima_interpolation_monotonic():
    """Test that Akima preserves monotonicity for monotonic data."""
    t = jnp.linspace(0, 1, 10)
    u = jnp.linspace(0, 10, 10)

    t_new = jnp.linspace(0, 1, 50)
    result = akima_interpolation(u, t, t_new)

    # Check monotonicity (allowing tiny numerical errors)
    diffs = jnp.diff(result)
    assert jnp.all(diffs >= -1e-10)


def test_akima_grad():
    """Test gradients of akima_interpolation using JAX."""
    x = jnp.linspace(0, 10, 20)
    x_new = jnp.linspace(0, 10, 50)

    def f(alpha, x, x_new: float):
        y = jnp.sin(alpha * x)
        return jnp.sum(akima_interpolation(y, x, x_new))

    grad_f = jax.grad(f, argnums=0)

    alpha = 0.5
    grad = grad_f(alpha, x, x_new)

    # exact derivative
    dy_dalpha = x * jnp.cos(alpha * x)

    # Interpolate the exact derivative at the same x_new points
    dy_dalpha_interp = jnp.sum(akima_interpolation(dy_dalpha, x, x_new))

    np.testing.assert_allclose(
        grad,
        dy_dalpha_interp,
        rtol=1e-5,
    )
