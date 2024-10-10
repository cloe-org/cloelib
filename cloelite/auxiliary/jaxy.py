import jax.numpy as jnp
import numpy as onp


def jax_atleast_1d(x):
    #return x
    return jnp.atleast_1d(x)