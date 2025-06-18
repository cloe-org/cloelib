"""
Angular correlation function implementation using Wigner small-d matrices.

This module provides `AngularCorrelationFunctionWigner`, a class that computes
two-point angular correlation functions xi(theta) from angular power spectra Cl,
using spherical harmonic transforms involving Wigner d-matrices.

The Wigner d-matrices follow the recurrence relations from:
https://arxiv.org/pdf/1702.05301
"""

import jax.numpy as np
import jax
from jax import jit
from functools import lru_cache, wraps

from cloelib.observables.photo import ShearTracer #, PositionsTracer
from .angular_correlation_function import AngularCorrelationFunction

def memoize_jax(func):
    """Memoize functions with JAX array arguments."""
    @lru_cache(maxsize=None)
    def cached_func(*hashable_args, **hashable_kwargs):
        args = [
            jnp.frombuffer(arg[0], dtype=arg[2]).reshape(arg[1])
            if isinstance(arg, tuple) and len(arg) == 3
            else arg
            for arg in hashable_args
        ]
        kwargs = {
            k: jnp.frombuffer(v[0], dtype=v[2]).reshape(v[1])
            if isinstance(v, tuple) and len(v) == 3
            else v
            for k, v in hashable_kwargs.items()
        }
        return func(*args, **kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        hashable_args = tuple(
            (arg.tobytes(), arg.shape, arg.dtype)
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype')
            else arg
            for arg in args
        )
        hashable_kwargs_dict = {
            k: (v.tobytes(), v.shape, v.dtype)
            if hasattr(v, 'shape') and hasattr(v, 'dtype')
            else v
            for k, v in kwargs.items()
        }
        return cached_func(*hashable_args, **hashable_kwargs_dict)

    return wrapper

@jit
def _d_0_0_ell_compute(beta, ell):
    """
    Compute d_00^ell(beta).

    The computation uses the Wigner d-matrix
    recurrence relations in a JIT-compatible way.
    """
    base_case_0 = np.ones_like(beta)
    base_case_1 = np.cos(beta)

    def recurrence_fn(l, vals):
        prev, prev2 = vals
        new_val = ((2 * l - 1) / (l) * base_case_1 * prev - ((l - 1) / (l)) * prev2)

        return new_val, prev

    return np.where(ell == 0, base_case_0,
                     np.where(ell == 1, base_case_1,
                               jax.lax.fori_loop(2, ell + 1, recurrence_fn, (base_case_1, base_case_0))[0]))

@jit
def _d_2_2_ell_compute(beta, ell):
    """
    Compute d_22^ell(beta).

    The computation uses recurrence for small ell
    and an approximation for large ell, in a JIT-compatible way.
    """
    # Base cases
    base_case_2 = (1/4) * (1 + np.cos(beta))**2
    base_case_3 = np.cos(beta / 2) ** 4 * (3 * np.cos(beta) - 2)

    # Recurrence relation for small ell
    def recurrence_fn(l, vals):
        prev, prev2 = vals
        new_val = (l * (2 * l - 1) / (l**2 - 4)) * (
            (d_0_0_ell(beta, 1) - (4 / (l * (l - 1)))) * prev
            - (((l - 1)**2 - 4) / ((l - 1) * (2 * l - 1))) * prev2
        )
        return new_val, prev

    # Approximation for large ell (fixed to explicitly pass `beta`)
    def approximation_fn(l, vals):
        prev, prev2 = vals
        new_val = 2 * d_0_0_ell(beta, 1) * prev - prev2
        return new_val, prev

    # Compute using a JIT-compatible conditional switch
    def compute_d_2_2(ell):
        return jax.lax.cond(
            ell < 30000,
            lambda: jax.lax.fori_loop(4, ell + 1, recurrence_fn, (base_case_3, base_case_2))[0],
            lambda: jax.lax.fori_loop(50, ell + 1, approximation_fn, (base_case_3, base_case_2))[0]
        )

    return jax.lax.cond(
        ell == 2, lambda: base_case_2,
        lambda: jax.lax.cond(
            ell == 3, lambda: base_case_3,
            lambda: compute_d_2_2(ell)
        )
    )



@jit
def _d_2_m2_ell_compute(beta, ell):
    """
    Compute d_2-2^ell(beta).

    The computation uses recurrence for small ell
    and an approximation for large ell, in a JIT-compatible way.
    """
    # Base cases
    base_case_2 = (1/4) * (1 - np.cos(beta))**2
    base_case_3 = np.sin(beta / 2) ** 4 * (3 * np.cos(beta) + 2)

    # Recurrence relation for small ell
    def recurrence_fn(l, vals):
        prev, prev2 = vals
        new_val = (l * (2 * l - 1) / (l**2 - 4)) * (
            (d_0_0_ell(beta, 1) + (4 / (l * (l - 1)))) * prev
            - (((l - 1)**2 - 4) / ((l - 1) * (2 * l - 1))) * prev2
        )
        return new_val, prev

    # Approximation for large ell (fixed to explicitly pass `beta`)
    def approximation_fn(l, vals):
        prev, prev2 = vals
        new_val = 2 * d_0_0_ell(beta, 1) * prev - prev2
        return new_val, prev

    # Compute using a JIT-compatible conditional switch
    def compute_d_2_2(ell):
        return jax.lax.cond(
            ell < 30000,
            lambda: jax.lax.fori_loop(4, ell + 1, recurrence_fn, (base_case_3, base_case_2))[0],
            lambda: jax.lax.fori_loop(50, ell + 1, approximation_fn, (base_case_3, base_case_2))[0]
        )

    return jax.lax.cond(
        ell == 2, lambda: base_case_2,
        lambda: jax.lax.cond(
            ell == 3, lambda: base_case_3,
            lambda: compute_d_2_2(ell)
        )
    )





@jit
def _d_2_0_ell_compute(beta, ell):
    """
    Compute d_20^ell(beta).

    The computation uses recurrence for small ell
    and an approximation for large ell, in a JIT-compatible way.
    """
    # Base cases
    base_case_2 = np.sqrt(3/8) * np.sin(beta)**2
    base_case_3 = (np.sqrt(30)/4) * np.sin(beta)**2 * np.cos(beta)

    # Recurrence relation for small ell
    def recurrence_fn(l, vals):
        prev, prev2 = vals
        sqrt_l2_4 = np.sqrt(l**2 - 4)
        sqrt_lm1_2_4 = np.sqrt((l - 1)**2 - 4)
        new_val = ((2 * l - 1) / sqrt_l2_4) * (
            d_0_0_ell(beta, 1) * prev - (sqrt_lm1_2_4 / (2 * l - 1)) * prev2
        )
        return new_val, prev

    # Approximation for large ell (fixed to explicitly pass `beta`)
    def approximation_fn(l, vals):
        prev, prev2 = vals
        new_val = 2 * d_0_0_ell(beta, 1) * prev - prev2
        return new_val, prev

    # Compute using a JIT-compatible conditional switch
    def compute_d_2_0(ell):
        return jax.lax.cond(
            ell < 30000,
            lambda: jax.lax.fori_loop(4, ell + 1, recurrence_fn, (base_case_3, base_case_2))[0],
            lambda: jax.lax.fori_loop(50, ell + 1, approximation_fn, (base_case_3, base_case_2))[0]
        )

    return jax.lax.cond(
        ell == 2, lambda: base_case_2,
        lambda: jax.lax.cond(
            ell == 3, lambda: base_case_3,
            lambda: compute_d_2_0(ell)
        )
    )

# -----------------------------------------------------------------------------------
# Vectorized versions of Wigner d-matrix functions
# -----------------------------------------------------------------------------------

d_0_0_ell = memoize_jax(_d_0_0_ell_compute)
d_2_2_ell = memoize_jax(_d_2_2_ell_compute)
d_2_m2_ell = memoize_jax(_d_2_m2_ell_compute)
d_2_0_ell = memoize_jax(_d_2_0_ell_compute)

_d_0_0_vmap_compute = jax.vmap(jax.vmap(_d_0_0_ell_compute, (None, 0)), (0, None))
_d_2_2_vmap_compute = jax.vmap(jax.vmap(_d_2_2_ell_compute, (None, 0)), (0, None))
_d_2_m2_vmap_compute = jax.vmap(jax.vmap(_d_2_m2_ell_compute, (None, 0)), (0, None))
_d_2_0_vmap_compute = jax.vmap(jax.vmap(_d_2_0_ell_compute, (None, 0)), (0, None))

d_0_0_vmap = memoize_jax(_d_0_0_vmap_compute)
d_2_2_vmap = memoize_jax(_d_2_2_vmap_compute)
d_2_m2_vmap = memoize_jax(_d_2_m2_vmap_compute)
d_2_0_vmap = memoize_jax(_d_2_0_vmap_compute)


# -----------------------------------------------------------------------------------
# Angular correlation function using Wigner d-matrices
# -----------------------------------------------------------------------------------
class AngularCorrelationFunctionWigner(AngularCorrelationFunction):
    """Correlation function implementation using Wigner small-d matrices."""

    def __init__(self, angular_two_point, ells, ks):
        """
        Initialize CorrelationFunction with an AngularTwoPoint instance.

        Thie real-space angular correlation functions are computed from
        angular power spectra Cl via spherical harmonic projection. The spin
        configuration is inferred from the types of tracers in the provided
        AngularTwoPoint object.

        Parameters
        ----------
        angular_two_point : AngularTwoPoint object
            Object providing Cl evaluation and tracers.
        ells: jnp.ndarray
            Multipole moments at which the Cl spectrum is evaluated.
        ks: jnp.ndarray
            Wavenumber grid (only needed for computing Cl via angular_two_point).
        """
        self.angular_two_point = angular_two_point
        self.ells=ells
        self.ks=ks

        # Determine spin values based on tracer type
        self.s1 = 2 if isinstance(angular_two_point.tracer1, ShearTracer) else 0
        self.s2 = 2 if isinstance(angular_two_point.tracer2, ShearTracer) else 0

    def get_xi(self, theta):
        """
        Compute the angular correlation function xi(theta) using the Wigner d-matrices.

        TODO: Also implement FFTLog which is likely faster
        WARNING: Currently assumes B-modes are zero, as they are not passed on from AngularTwoPoint

        Args:
            theta (jax.numpy.ndarray): Angles in radians.

        Returns:
            if at least one tracer is spin 0 (clustering or GGL):
                jax.numpy.ndarray: Computed xi(theta)
            if both tracers are spin 2 (cosmic shear):
                (jax.numpy.ndarray, jax.numpy.ndarray): Computed xi_+(theta) and xi_-(theta).
        """
        # Compute Cl using the AngularTwoPoint instance
        Cl_EE = self.angular_two_point.get_Cl(self.ells, nl=0, ks=self.ks)

        Cl_BB = np.zeros_like(Cl_EE)  # No B-modes included, set to zero for now
        Cl_EB = np.zeros_like(Cl_EE)
        Cl_BE = np.zeros_like(Cl_EE)

        Cl_plus = (Cl_EE+Cl_BB) + (Cl_EB+Cl_BE)
        Cl_minus = (Cl_EE+Cl_BB) - (Cl_EB+Cl_BE)


        Ntomo1 = Cl_EE.shape[1]  # Number of tomographic bins
        Ntomo2 = Cl_EE.shape[2]  # Number of tomographic bins

        Ntheta = len(theta)

        # Compute Wigner-d matrix elements
        if self.s1 == 0 and self.s2 == 0:
            d_l_theta_plus = d_0_0_vmap(theta, self.ells)
            d_l_theta_minus = d_l_theta_plus
        elif self.s1 == 2 and self.s2 == 0:
            d_l_theta_plus = d_2_0_vmap(theta, self.ells)
            d_l_theta_minus = d_l_theta_plus
        elif self.s1 == 2 and self.s2 == 2:
            d_l_theta_plus = d_2_2_vmap(theta, self.ells)
            d_l_theta_minus = d_2_m2_vmap(theta, self.ells)
        else:
            raise ValueError("Spin values not as expected")

        # Compute the prefactor (2\ell + 1) / (4\pi)
        prefactor = (2 * self.ells + 1) / (4 * np.pi)

        # Initialize xi arrays
        xi_plus = np.zeros((Ntheta, Ntomo1, Ntomo2))
        if self.s2 == 2:
            xi_minus = np.zeros((Ntheta, Ntomo1, Ntomo2))

        # Vectorized computation over (theta, tomo1, tomo2)
        # Compute xi_plus
        xi_plus = np.einsum('l,lij,θl->θij', prefactor, Cl_plus, d_l_theta_plus)

        # Compute xi_minus if we have a spin2 tracer
        if self.s2 == 2:
            xi_minus = xi_minus = (-1) ** self.s2 * np.einsum('l,lij,θl->θij', prefactor, Cl_minus, d_l_theta_minus)
            return xi_plus, xi_minus
        else:
            return xi_plus
