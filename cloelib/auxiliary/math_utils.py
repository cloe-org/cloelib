"""Module for mathematical functions."""
import jax
import jax.numpy as jnp

def simpsons_weights_odd(num_el):
    """Write documentation (TODO)."""
    w = jnp.zeros(num_el)
    w = w.at[0].set(1/3)
    w = w.at[1::2].set(4/3)
    w = w.at[2::2].set(2/3)
    w = w.at[-1].set(1/3)
    return w

def simpsons_weights_even(num_el):
    """Write documentation (TODO)."""
    num_el = int(num_el)
    w_odd_end = simpsons_weights_odd(num_el - 1)
    w_odd_end = w_odd_end.at[-1].add(1/6)
    w_odd_end = jnp.append(w_odd_end, 1/6)

    w_odd_start = simpsons_weights_odd(num_el - 1)
    w_odd_start = w_odd_start.at[0].add(1/6)
    w_odd_start = jnp.append(1/6, w_odd_start)

    w_odd = (w_odd_start + w_odd_end) / 2.0
    return w_odd

def stack_zeros_and_simpson(num_weights, num_zeros):
    """Write documentation (TODO)."""
    # Compute Simpson weights for num_weights elements.
    if num_weights % 2 == 1:
        weights = simpsons_weights_odd(num_weights)
    else:
        weights = simpsons_weights_even(num_weights)

    # Create an array of zeros of length num_zeros.
    zeros_array = jnp.zeros(num_zeros)

    # Concatenate zeros on the left and weights on the right.
    return jnp.concatenate([zeros_array, weights], axis=0)

def stacked_simpson(n):
    """Write documentation (TODO)."""
    rows = []
    # For row i (0-indexed), we want i zeros on the left and (n - i) Simpson weights.
    for i in range(n):
        num_zeros = i
        num_weights = n - i
        row = stack_zeros_and_simpson(num_weights, num_zeros)
        rows.append(row)
    return jnp.vstack(rows)

_cached_stacked_simpson = {}

def cached_stacked_simpson(n: int):
    """Write documentation (TODO)."""
    key = str(n)
    # TODO for the moment we are using a plain "if", later we are gonna need a lax conditional
    # or a better cache mechanism
    if key not in _cached_stacked_simpson:
        _cached_stacked_simpson[key] = stacked_simpson(n)
    return _cached_stacked_simpson[key]

def legendre(n, x):
    """Write documentation (TODO)."""
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return x
    else:
        P0 = jnp.ones_like(x)
        P1 = x
        for k in range(2, n+1):
            Pn = ((2*k - 1) * x * P1 - (k - 1) * P0) / k
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

