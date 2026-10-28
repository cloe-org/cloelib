r"""Chebyshev nodes, Chebyshev expansions and Clenshaw-Curtis quadrature.

The nodes are the Chebyshev extrema $x_k = \cos(\pi k / n)$, $k = 0, \dots, n$
(Chebyshev-Lobatto nodes, including the endpoints), ordered from $+1$ to $-1$.
On these nodes the Chebyshev coefficients of a function are given by a
discrete cosine transform of type I, computed here with a real FFT.
"""

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def dct_type1(f_values: Array) -> Array:
    """Discrete cosine transform of type I along the first axis.

    Computed as the real FFT of the even extension
    `[f_0, ..., f_n, f_{n-1}, ..., f_1]` (no normalization).

    Args:
        f_values: Input values, shape `(n + 1, ...)` with `n >= 1`.

    Returns:
        DCT-I of the input along the first axis, same shape as `f_values`.
    """
    internal_reversed = jnp.flip(f_values[1:-1], axis=0)
    x_ext = jnp.concatenate([f_values, internal_reversed], axis=0)
    X = jnp.fft.rfft(x_ext, axis=0)

    return jnp.real(X[: f_values.shape[0]])


def chebyshev_points(n: int) -> Array:
    r"""Chebyshev extrema (Chebyshev-Lobatto nodes) on [-1, 1].

    Args:
        n: Polynomial degree; `n + 1` nodes are returned.

    Returns:
        The nodes $\cos(\pi k / n)$, $k = 0, \dots, n$, from 1 to -1.
    """
    k = jnp.arange(n + 1)
    return jnp.cos(jnp.pi * k / n)


def chebyshev_points_interval(n: int, x_start: float, x_stop: float) -> Array:
    """Chebyshev extrema (Chebyshev-Lobatto nodes) mapped to [x_start, x_stop].

    Args:
        n: Polynomial degree; `n + 1` nodes are returned.
        x_start: Start of the interval.
        x_stop: End of the interval.

    Returns:
        The nodes of `chebyshev_points(n)` mapped linearly to the interval,
        from `x_stop` to `x_start`.
    """
    cheb_pts = chebyshev_points(n)
    return 0.5 * (x_stop - x_start) * (cheb_pts + 1) + x_start


@jax.jit
def chebyshev_coefficients(f_values: Array) -> Array:
    r"""Chebyshev coefficients from the values of a function at the Chebyshev extrema.

    The interpolant is $f(x) = \sum_{j=0}^{n} c_j T_j(x)$ on the interval the
    nodes were mapped to (rescaled to [-1, 1]), and the expansion is done
    independently along every trailing axis.

    Args:
        f_values: Values at the nodes of `chebyshev_points(n)` (or
            `chebyshev_points_interval`), in the same order, along the first
            axis; shape `(n + 1, ...)`.

    Returns:
        The coefficients $c_j$ along the first axis, same shape as `f_values`.
    """
    N = f_values.shape[0]
    c = dct_type1(f_values) / (N - 1)

    scale = jnp.ones(N).at[0].set(0.5).at[-1].set(0.5)
    return c * scale.reshape((N,) + (1,) * (f_values.ndim - 1))


def clenshaw_curtis_quadrature(n: int, a: float, b: float) -> tuple[Array, Array]:
    """Clenshaw-Curtis quadrature rule on [a, b].

    Args:
        n: Number of quadrature nodes, i.e. the nodes of
            `chebyshev_points_interval(n - 1, a, b)`.
        a: Start of the interval.
        b: End of the interval.

    Returns:
        The nodes on [a, b] (from `b` to `a`) and their weights, both of
        shape `(n,)`.
    """
    N = n - 1

    # mu moments, mu_k = 2 / (1 - k^2) if k is even, 0 if k is odd
    k = jnp.arange(n)
    mu = jnp.where(k % 2 == 0, 2.0 / (1.0 - k**2), 0.0)
    mu = mu.at[0].set(2.0)
    mu = mu.at[-1].set(mu[-1] / 2.0)

    # IDCT-I using rFFT
    internal_reversed = jnp.flip(mu[1:-1])
    mu_ext = jnp.concatenate([mu, internal_reversed])

    w = jnp.fft.rfft(mu_ext)
    w = jnp.real(w[:n]) / N
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[-1].set(w[-1] / 2.0)

    # rescaling weights to the interval [a, b]
    w = w * (b - a) / 2.0

    return chebyshev_points_interval(N, a, b), w
