import pytest
from  cloelib.summary_statistics import angular_correlation_function_wigner

import jax.numpy as jnp
from jax import random
import pytest

def test_wigner_cache():
    beta = 1.
    ell = jnp.linspace(2,200)
    d_2_0_ell()
    assert angular_correlation_function_wigner.d_0_0_ell(beta, ell) == angular_correlation_function_wigner._d_0_0_ell_compute(beta, ell)
    assert angular_correlation_function_wigner.d_2_0_ell(beta, ell) == angular_correlation_function_wigner._d_2_0_ell_compute(beta, ell)
    assert angular_correlation_function_wigner.d_2_2_ell(beta, ell) == angular_correlation_function_wigner._d_2_2_ell_compute(beta, ell)
    assert angular_correlation_function_wigner.d_2_m2_ell(beta, ell) == angular_correlation_function_wigner._d_2_m2_ell_compute(beta, ell)
