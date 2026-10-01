"""Module for mathematical functions."""

from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp


def ensure_z_zero_included(redshifts: np.ndarray) -> np.ndarray:
    """
    Ensure z=0 is in redshift array for sigma8(z=0) computation.

    This utility function checks if z=0 is present in the redshift array,
    and adds it if missing. This is necessary for computing sigma8(z=0)
    which is required for cosmological analyses.

    Parameters
    ----------
    redshifts : np.ndarray
        Array of redshift values

    Returns
    -------
    np.ndarray
        Redshift array with z=0 included (sorted)
    """
    z_min = redshifts.min()
    if z_min > 0.001:
        return np.sort(np.append(redshifts, 0.0))
    else:
        return redshifts


def simpsons_weights_odd(num_el: int) -> jnp.ndarray:
    """Simpson's rule weights when num_el is odd."""
    w = jnp.zeros(num_el)
    w = w.at[0].set(1 / 3)
    w = w.at[1::2].set(4 / 3)
    w = w.at[2::2].set(2 / 3)
    w = w.at[-1].set(1 / 3)
    return w


def simpsons_weights_even(num_el: int) -> jnp.ndarray:
    """Simpson's rule weights when num_el is even."""
    num_el = int(num_el)
    w_odd_end: jnp.ndarray = simpsons_weights_odd(num_el - 1)
    w_odd_end = w_odd_end.at[-1].add(1 / 2)
    w_odd_end = jnp.append(w_odd_end, 1 / 2)
    w_odd_start: jnp.ndarray = simpsons_weights_odd(num_el - 1)
    w_odd_start = w_odd_start.at[0].add(1 / 2)
    w_odd_start = jnp.append(1 / 2, w_odd_start)
    return (w_odd_start + w_odd_end) / 2.0


def simpsons_weights_jax(num_el: int) -> jnp.ndarray:
    """JAX-compatible Simpson's weights computation."""
    return jax.lax.cond(
        num_el % 2 == 1,
        lambda: simpsons_weights_odd(num_el),
        lambda: simpsons_weights_even(num_el),
    )


# JIT-compiled version with static argument
simpsons_weights_jit = jax.jit(simpsons_weights_jax, static_argnums=(0,))


def _simpson_odd_np(num_el: int) -> np.ndarray:
    """Composite Simpson weights (unit spacing) for an odd number of nodes >= 3."""
    w = np.empty(num_el)
    w[1::2] = 4 / 3
    w[2::2] = 2 / 3
    w[0] = w[-1] = 1 / 3
    return w


def _simpson_even_np(num_el: int) -> np.ndarray:
    """Even-N Simpson weights (unit spacing): the average of Simpson on the first
    and on the last N-1 nodes, each closed with a trapezoid on the remaining
    interval. Same rule as `simpsons_weights_even`; N = 2 is the trapezoid rule."""
    if num_el == 2:
        return np.array([0.5, 0.5])
    w_end = np.append(_simpson_odd_np(num_el - 1), 0.0)
    w_end[-2:] += 0.5
    w_start = np.append(0.0, _simpson_odd_np(num_el - 1))
    w_start[:2] += 0.5
    return (w_start + w_end) / 2.0


def _simpsons_weights_avg_np(num_el: int) -> np.ndarray:
    """Parity-robust Simpson weights (unit spacing), see `simpsons_weights_avg`."""
    num_el = int(num_el)
    if num_el < 1:
        raise ValueError("num_el must be >= 1")
    if num_el == 1:
        return np.zeros(1)  # a single node spans no interval
    if num_el % 2 == 0:
        return _simpson_even_np(num_el)
    # odd N: the same averaging, one level up - even-N Simpson on the first and on
    # the last N-1 nodes, each closed with a trapezoid on the remaining interval
    w_end = np.append(_simpson_even_np(num_el - 1), 0.0)
    w_end[-2:] += 0.5
    w_start = np.append(0.0, _simpson_even_np(num_el - 1))
    w_start[:2] += 0.5
    return (w_start + w_end) / 2.0


@lru_cache(maxsize=None)
def _simpsons_weights_avg_cached(num_el: int) -> jnp.ndarray:
    # Built eagerly (concrete array, never a tracer), so caching it is safe even
    # when the first call happens while JAX is tracing.
    with jax.ensure_compile_time_eval():
        return jnp.asarray(_simpsons_weights_avg_np(num_el))


def simpsons_weights_avg(num_el: int) -> jnp.ndarray:
    """Simpson-type quadrature weights (unit spacing) without the alternating
    4/3, 2/3 pattern, for any number of nodes.

    The composite Simpson rule (`simpsons_weights_odd`) alternates 4/3, 2/3 in the
    interior. For an integrand that is under-resolved by the grid - e.g. the
    Limber integrand of a narrow tomographic bin, which goes as n(z)^2 - that
    makes the result depend on where a peak falls between nodes, and the error
    oscillates with the number of nodes. `simpsons_weights_even` already avoids
    this by averaging two shifted Simpson rules: its weights are 1 except at the
    first and last two nodes (5/12, 13/12). This function uses that rule for even
    `num_el` (unchanged) and, for odd `num_el`, the analogous average of two
    shifted even-N rules, whose weights are 1 except at the first and last three
    nodes (11/24, 1, 25/24). One node gives weight 0, two nodes the trapezoid rule.

    Args:
        num_el (int): number of nodes (a static Python int).

    Returns:
        jnp.ndarray: weights of length `num_el`; multiply by the grid spacing.
    """
    return _simpsons_weights_avg_cached(int(num_el))


def stack_zeros_and_simpson(num_weights: int, num_zeros: int) -> jnp.ndarray:
    """Row of `stacked_simpson`: `num_zeros` zeros followed by the
    `simpsons_weights_avg` weights for `num_weights` nodes."""
    return jnp.asarray(
        np.concatenate([np.zeros(num_zeros), _simpsons_weights_avg_np(num_weights)])
    )


def _stacked_simpson_np(n: int) -> np.ndarray:
    w = np.zeros((n, n))
    for i in range(n):
        w[i, i:] = _simpsons_weights_avg_np(n - i)
    return w


def stacked_simpson(n: int) -> jnp.ndarray:
    """Matrix of quadrature weights (unit spacing) for cumulative integrals to the
    end of the grid: row `i` integrates over nodes `i, ..., n-1` with
    `simpsons_weights_avg` (zero weights before node `i`).

    Row `i` therefore sums to the number of remaining intervals, `n - 1 - i`: the
    last row (a single node) is zero and the one before it is the trapezoid rule.
    """
    return jnp.asarray(_stacked_simpson_np(int(n)))


@lru_cache(maxsize=None)
def _cached_stacked_simpson_py(n: int) -> jnp.ndarray:
    # Built eagerly with numpy and materialised outside any trace, so the cache
    # holds a concrete array. (Caching the output of a jitted function here used to
    # store a tracer, which failed with `UnexpectedTracerError` on any re-trace
    # for the same `n`, e.g. after `jax.clear_caches()`.)
    with jax.ensure_compile_time_eval():
        return jnp.asarray(_stacked_simpson_np(int(n)))


def cached_stacked_simpson(n: int) -> jnp.ndarray:
    """Cached `stacked_simpson(n)`. `n` must be a static Python int (e.g. `len(z)`);
    safe to call inside jitted code."""
    return _cached_stacked_simpson_py(int(n))


def legendre(n, x):
    """Write documentation (TODO)."""
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return x
    else:
        P0 = jnp.ones_like(x)
        P1 = x
        for k in range(2, n + 1):
            Pn = ((2 * k - 1) * x * P1 - (k - 1) * P0) / k
            P0, P1 = P1, Pn
        return Pn


def simps(f, a, b, N=128):
    """Write documentation (TODO)."""
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S
