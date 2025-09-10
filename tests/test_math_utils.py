from cloelib.auxiliary.math_utils import (
    simpsons_weights_jit,
    simpsons_weights_odd,
    simpsons_weights_even,
)
from scipy import integrate
from numpy.testing import assert_array_equal
import numpy as np


def test_simpson():
    assert_array_equal(simpsons_weights_jit(11), simpsons_weights_odd(11))
    assert_array_equal(simpsons_weights_jit(10), simpsons_weights_even(10))

    x = np.arange(0, 101)
    y = np.power(x, 3)
    w = simpsons_weights_jit(len(x))
    assert np.allclose(integrate.simpson(y, x=x), np.dot(y, w), rtol=1e-5)

    x = np.arange(0, 100)
    y = np.power(x, 3)
    w = simpsons_weights_jit(len(x))
    assert np.allclose(integrate.simpson(y, x=x), np.dot(y, w), rtol=1e-5)
