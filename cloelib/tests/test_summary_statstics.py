import pytest
import cloelib.summary_statistics.angular_correlation_function_wigner

import jax.numpy as jnp
from jax import random
import pytest

def test_wigner_cache():
    beta = 1.
    ell = jnp.linspace(2,200)
    d_2_0_ell()
    assert d_0_0_ell(beta, ell) == _d_0_0_ell_compute(beta, ell)
    assert d_2_0_ell(beta, ell) == _d_2_0_ell_compute(beta, ell)
    assert d_2_2_ell(beta, ell) == _d_2_2_ell_compute(beta, ell)
    assert d_2_m2_ell(beta, ell) == _d_2_m2_ell_compute(beta, ell)
