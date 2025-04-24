import jax.numpy as np
import jax
from jax import jit

#from jax.scipy.special import gammaln

from cloelib.observables.photo import ShearTracer #, PositionsTracer
from .angular_correlation_function import AngularCorrelationFunction

"""
Angular correlation function implementation using Wigner small-d matrices.

This module provides `AngularCorrelationFunctionWigner`, a class that computes
two-point angular correlation functions xi(theta) from angular power spectra Cl,
using spherical harmonic transforms involving Wigner d-matrices.

The Wigner d-matrices follow the recurrence relations from:
https://arxiv.org/pdf/1702.05301
"""


# -----------------------------------------------------------------------------------
# Wigner d-matrix recurrence relations
# -----------------------------------------------------------------------------------
@jit
def d_0_0_ell(beta, ell):
    base_case_0 = np.ones_like(beta)
    base_case_1 = np.cos(beta)
    
    def body_fn(l, vals):
        prev, prev2 = vals
        new_val = ((2 * l - 1) / (l) * base_case_1 * prev - ((l - 1) / (l)) * prev2)

        return new_val, prev
    
    return np.where(ell == 0, base_case_0, 
                     np.where(ell == 1, base_case_1, 
                               jax.lax.fori_loop(2, ell + 1, body_fn, (base_case_1, base_case_0))[0]))

@jit
def d_2_2_ell(beta, ell):
    """
    Computes d_22^ell(beta) using recurrence for small ell
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
def d_2_m2_ell(beta, ell):
    """
    Computes d_2-2^ell(beta) using recurrence for small ell
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
def d_2_0_ell(beta, ell):
    """
    Computes d_20^ell(beta) using recurrence for small ell
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

d_0_0_vmap = jax.vmap(jax.vmap(d_0_0_ell, (None, 0)), (0, None))
d_2_2_vmap = jax.vmap(jax.vmap(d_2_2_ell, (None, 0)), (0, None))
d_2_m2_vmap = jax.vmap(jax.vmap(d_2_m2_ell, (None, 0)), (0, None))

d_2_0_vmap = jax.vmap(jax.vmap(d_2_0_ell, (None, 0)), (0, None))


# -----------------------------------------------------------------------------------
# Angular correlation function using Wigner d-matrices
# -----------------------------------------------------------------------------------
class AngularCorrelationFunctionWigner(AngularCorrelationFunction):
    """
    Correlation function implementation using Wigner small-d matrices.

    This class computes real-space angular correlation functions from
    angular power spectra Cl via spherical harmonic projection. The spin
    configuration is inferred from the types of tracers in the provided
    AngularTwoPoint object.

    Parameters
    ----------
    angular_two_point : AngularTwoPoint object
        Object providing Cl evaluation and tracers.
    ells: jnp.ndarray
        Multipole moments at which the Cl spectrum is evaluated.
    ks : jnp.ndarray
        Wavenumber grid (only needed for computing Cl via angular_two_point).
    """
    def __init__(self, angular_two_point, ells, ks):
        """
        Initializes CorrelationFunction with an AngularTwoPoint instance.
        Also automatically sets the spins s1 and s2 dependent on the type of tracer
        in AngularTwoPoint

        Args:
            angular_two_point (AngularTwoPoint): Instance providing Cl values.
            ells (jnp.ndarray): Multipole moments at which the Cl spectrum is evaluated.
            ks (jnp.ndarray): Wavenumber grid (only needed for computing Cl via angular_two_point).
        """
        self.angular_two_point = angular_two_point
        self.ells=ells
        self.ks=ks

        # Determine spin values based on tracer type
        self.s1 = 2 if isinstance(angular_two_point.tracer1, ShearTracer) else 0
        self.s2 = 2 if isinstance(angular_two_point.tracer2, ShearTracer) else 0


    def get_xi(self, theta):
        """
        Compute the angular correlation function xi(theta) using the Wigner d-matrices

        TODO: Also implement FFTLog which is likely faster
        WARNING: Currently assumes B-modes are zero, as they are not passed on from AngularTwoPoint

        Args:
            ells (jax.numpy.ndarray): Multipole moments.
            ks (jax.numpy.ndarray): Wavenumber grid.
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