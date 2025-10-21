import pytest
import numpy as np
import jax.numpy as jnp

from cloelib.auxiliary.chebyshev import (
    chebyshev_points,
    chebyshev_points_interval,
    chebyshev_coefficients,
    chebyshev_interpolation,
)