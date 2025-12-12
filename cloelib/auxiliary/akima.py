"""Module to implement Akima interpolation using JAX."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _akima_slopes(t: jnp.ndarray, u: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """
    Computes Akima 1D slopes along the specified axis.
    If axis != 0, moves it to the first position internally.

    Parameters:
    t : jax.numpy.ndarray, shape (n,)
        x-coordinates of data points.
    u : jax.numpy.ndarray, shape (..., n, ...)
        Function values; interpolation is done along `axis`.
    axis : int
        Axis along which to compute slopes.

    Returns:
    m: jax.numpy.ndarray
        Array of same shape as `u` but with length n+3 along the interpolated axis.
    """

    if axis != 0:
        u = jnp.moveaxis(u, axis, 0)
        moved = True
    else:
        moved = False

    n = u.shape[0]
    if n != len(t):
        raise ValueError("Length of t must match size of u along interpolation axis.")

    dt = jnp.diff(t)
    du = jnp.diff(u, axis=0)
    shape = (dt.shape[0],) + (1,) * (u.ndim - 1)
    m_partial = du / dt.reshape(shape)

    pad_shape = (n + 3, *u.shape[1:])
    m = jnp.zeros(pad_shape, dtype=u.dtype)
    m = m.at[2:-2, ...].set(m_partial)

    m = m.at[1, ...].set(2 * m[2, ...] - m[3, ...])
    m = m.at[0, ...].set(2 * m[1, ...] - m[2, ...])
    m = m.at[-2, ...].set(2 * m[-3, ...] - m[-4, ...])
    m = m.at[-1, ...].set(2 * m[-2, ...] - m[-3, ...])

    if moved:
        m = jnp.moveaxis(m, 0, axis)

    return m


_akima_slopes = jax.jit(_akima_slopes, static_argnames=["axis"])


def _akima_coefficients(t: jnp.ndarray, m: jnp.ndarray, axis: int = 0):
    """
    Computes the Akima 1D interpolation coefficients along the specified axis.

    Parameters
    ----------
    t : jnp.ndarray, shape (n,)
        x-coordinates of the data points.
    m : jnp.ndarray
        Array of slopes of shape (..., n+3, ...) where `axis` identifies the interpolation axis.
    axis : int
        Axis along which to compute the coefficients.

    Returns
    -------
    tuple of jnp.ndarray (b, c, d)
        b : shape matching `m` with length n along `axis`
        c, d : shape matching `m` with length n-1 along `axis`
    """

    if axis != 0:
        m = jnp.moveaxis(m, axis, 0)
        moved = True
    else:
        moved = False

    n = len(t)
    dt = jnp.diff(t)
    shape = (dt.shape[0],) + (1,) * (m.ndim - 1)
    eps_akima = jnp.finfo(m.dtype).eps * 100

    # average slope
    b = (m[3 : n + 3, ...] + m[:n, ...]) / 2.0

    # weighted adjustment where slope differences are large
    dm = jnp.abs(jnp.diff(m, axis=0))
    f1 = dm[2 : n + 2, ...]
    f2 = dm[:n, ...]
    f12 = f1 + f2
    mask = f12 > eps_akima

    weighted_b = (f1 * m[1 : n + 1, ...] + f2 * m[2 : n + 2, ...]) / f12
    b = jnp.where(mask, weighted_b, b)

    m3 = m[2 : n + 1, ...]
    b1 = b[:-1, ...]
    b2 = b[1:, ...]

    c = (3.0 * m3 - 2.0 * b1 - b2) / dt.reshape(shape)
    d = (b1 + b2 - 2.0 * m3) / (dt.reshape(shape) ** 2)

    if moved:
        b = jnp.moveaxis(b, 0, axis)
        c = jnp.moveaxis(c, 0, axis)
        d = jnp.moveaxis(d, 0, axis)

    return b, c, d


_akima_coefficients = jax.jit(_akima_coefficients, static_argnames=["axis"])


def _akima_eval(
    t: jnp.ndarray,
    u: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    tq: jnp.ndarray,
    axis: int = 0,
):
    """
    Evaluates the Akima 1D interpolation along the specified axis.

    Parameters
    ----------
    t : jnp.ndarray, shape (n,)
        x-coordinates of the data points.
    u : jnp.ndarray
        Function values of shape (..., n, ...) along `axis`.
    b, c, d : jnp.ndarray
        Akima coefficients with matching shapes:
        - b: (..., n, ...)
        - c, d: (..., n-1, ...)
    tq : jnp.ndarray, shape (k,)
        Points where interpolation is evaluated.
    axis : int
        Axis along which to evaluate the interpolation.

    Returns
    -------
    jnp.ndarray
        Interpolated values of shape (..., k, ...) along `axis`.
    """

    if axis != 0:
        u = jnp.moveaxis(u, axis, 0)
        b = jnp.moveaxis(b, axis, 0)
        c = jnp.moveaxis(c, axis, 0)
        d = jnp.moveaxis(d, axis, 0)
        moved = True
    else:
        moved = False

    n = u.shape[0]
    if n != len(t):
        raise ValueError("Length of t must match size of u along interpolation axis.")

    indices = jnp.searchsorted(t, tq, side="right") - 1
    indices = jnp.clip(indices, 0, n - 2)

    tq = jnp.atleast_1d(tq)
    dt = tq - t[indices]
    shape = (dt.shape[0],) + (1,) * (u.ndim - 1)
    dt_reshaped = dt.reshape(shape)

    u_interp = (
        u[indices, ...]
        + b[indices, ...] * dt_reshaped
        + c[indices, ...] * dt_reshaped**2
        + d[indices, ...] * dt_reshaped**3
    )

    if moved:
        u_interp = jnp.moveaxis(u_interp, 0, axis)

    return u_interp


_akima_eval = jax.jit(_akima_eval, static_argnames=["axis"])


def akima_interpolation(u: jnp.ndarray, t: jnp.ndarray, tq: jnp.ndarray, axis: int = 0):
    """
    Performs Akima 1D interpolation along the specified axis.

    Parameters
    ----------
    u : jnp.ndarray
        Function values, shape (..., n, ...), interpolation along `axis`.
    t : jnp.ndarray, shape (n,)
        x-coordinates of the data points.
    tq : jnp.ndarray, shape (k,)
        Points where interpolation is to be evaluated.
    axis : int
        Axis along which to interpolate.

    Returns
    -------
    jnp.ndarray
        Interpolated values of shape (..., k, ...) along `axis`.
    """

    sort_idx = jnp.argsort(t)
    t_sorted = t[sort_idx]
    u_sorted = jnp.take(u, sort_idx, axis=axis)

    m = _akima_slopes(t_sorted, u_sorted, axis)
    b, c, d = _akima_coefficients(t_sorted, m, axis)
    return _akima_eval(t_sorted, u_sorted, b, c, d, tq, axis)


akima_interpolation = jax.jit(akima_interpolation, static_argnames=["axis"])
