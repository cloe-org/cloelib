import pytest
from cloelib.auxiliary.math_utils import simpsons_weights_jit, simpsons_weights_odd, simpsons_weights_even
from scipy import integrate
from numpy.testing import assert_array_equal

def test_simpson_even():
    assert_array_equal(simpsons_weights_jit(11) == simpsons_weights_odd(11))
    assert_array_equal(simpsons_weights_jit(10) == simpsons_weights_even(10))
