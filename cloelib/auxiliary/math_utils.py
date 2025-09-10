"""Module for mathematical functions."""

from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from typing import TypeVar, Union

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


def simpsons_weights_odd(num_el: int) -> T:
    """Simpson's rule weights when num_el is odd."""
    w = jnp.zeros(num_el)
    w = w.at[0].set(1 / 3)
    w = w.at[1::2].set(4 / 3)
    w = w.at[2::2].set(2 / 3)
    w = w.at[-1].set(1 / 3)
    return w


def simpsons_weights_even(num_el: int) -> T:
    """Simpson's rule weights when num_el is even."""
    num_el = int(num_el)
    w_odd_end = simpsons_weights_odd(num_el - 1)
    w_odd_end = w_odd_end.at[-1].add(1 / 2)
    w_odd_end = jnp.append(w_odd_end, 1 / 2)
    w_odd_start = simpsons_weights_odd(num_el - 1)
    w_odd_start = w_odd_start.at[0].add(1 / 2)
    w_odd_start = jnp.append(1 / 2, w_odd_start)
    return (w_odd_start + w_odd_end) / 2.0


def simpsons_weights_jax(num_el: int) -> T:
    """JAX-compatible Simpson's weights computation."""
    return jax.lax.cond(
        num_el % 2 == 1,
        lambda: simpsons_weights_odd(num_el),
        lambda: simpsons_weights_even(num_el),
    )


# JIT-compiled version with static argument
simpsons_weights_jit = jax.jit(simpsons_weights_jax, static_argnums=(0,))


def stack_zeros_and_simpson(num_weights: int, num_zeros: int) -> jnp.ndarray:
    """Simpson's rule weights."""
    if num_weights % 2 == 1:
        weights = simpsons_weights_odd(num_weights)
    else:
        weights = simpsons_weights_even(num_weights)
    zeros_array = jnp.zeros(num_zeros)
    return jnp.concatenate([zeros_array, weights], axis=0)


def stacked_simpson(n: int) -> jnp.ndarray:
    """Simpson's rule weights, with a stacking."""
    rows = []
    for i in range(n):
        row = stack_zeros_and_simpson(n - i, i)
        rows.append(row)
    return jnp.vstack(rows)


@lru_cache(maxsize=None)
def _cached_stacked_simpson_py(n: int) -> jnp.ndarray:
    return stacked_simpson(n)


def cached_stacked_simpson(n: int) -> jnp.ndarray:
    """Cache the simpson weights calculation."""
    return _cached_stacked_simpson_py(n)


cached_stacked_simpson = jax.jit(cached_stacked_simpson, static_argnums=0)


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
