import jax
import jax.numpy as jnp

# This will go to a math-utils module, for the moment stays here
max_size = 300

def simpsons_weights_odd(num_el):
    w = jnp.zeros(num_el)
    w = w.at[0].set(1/3)
    w = w.at[1::2].set(4/3)
    w = w.at[2::2].set(2/3)
    w = w.at[-1].set(1/3)
    return w

def simpsons_weights_even(num_el):
    num_el = int(num_el)
    w_odd_end = simpsons_weights_odd(num_el - 1)
    w_odd_end = w_odd_end.at[-1].add(1/6)
    w_odd_end = jnp.append(w_odd_end, 1/6)

    w_odd_start = simpsons_weights_odd(num_el - 1)
    w_odd_start = w_odd_start.at[0].add(1/6)
    w_odd_start = jnp.append(1/6, w_odd_start)

    w_odd = (w_odd_start + w_odd_end) / 2.0
    return w_odd

def precompute_simpsons_weights(max_size):
    weights = []
    for n in range(1, max_size + 1):
        if n % 2 == 1:
            w = simpsons_weights_odd(n)
        else:
            w = simpsons_weights_even(n)
        padded_w = jnp.pad(w, (0, max_size - n), constant_values=0)
        weights.append(padded_w)
    return jnp.array(weights)

 # Set this to the maximum number of elements you expect
precomputed_weights = precompute_simpsons_weights(max_size)

def get_simpsons_weights(n):
    return precomputed_weights[n - 1]

get_simpsons_weights_jit = jax.jit(get_simpsons_weights)

def simpsons_weights_odd(num_el):
    w = jnp.zeros(num_el)
    w = w.at[0].set(1/3)
    w = w.at[1::2].set(4/3)
    w = w.at[2::2].set(2/3)
    w = w.at[-1].set(1/3)
    return w

def simpsons_weights_even(num_el):
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
    rows = []
    # For row i (0-indexed), we want i zeros on the left and (n - i) Simpson weights.
    for i in range(n):
        num_zeros = i
        num_weights = n - i
        row = stack_zeros_and_simpson(num_weights, num_zeros)
        rows.append(row)
    return jnp.vstack(rows)

#this 100 is clearly hardcoded. Think of a better mechanism!
precomputed_simpson_matrix = stacked_simpson(100)

def legendre(n, x):
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

def simps_jax(y, x=None, dx=1.0):
    """
    Simpson's rule integration in JAX.
    Parameters:
    - y: Array of function values to integrate.
    - x: Array of sample points corresponding to y (optional).
    - dx: Spacing between sample points if x is None (default: 1.0).
    """
    N = len(y)
    if N % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of samples.")

    if x is None:
        x = jnp.arange(N) * dx

    h = jnp.diff(x)
    S = y[0] + y[-1] + 4 * jnp.sum(y[1:-1:2]) + 2 * jnp.sum(y[2:-2:2])
    return (h[0] / 3) * S
