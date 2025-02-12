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
