import jax.numpy as np
import jax
from jax.scipy.special import gammaln

from cloelib.observables.photo import ShearTracer, PositionsTracer
from scipy.special import eval_jacobi
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

import jax
import jax.numpy as np
from jax import jit, grad, lax

def legendre_polynomial(n, x):
    """Computes P_n(x) using JAX-friendly recurrence."""
    def body_fun(i, carry):
        P_nm2, P_nm1 = carry
        P_n = ((2 * i - 1) * x * P_nm1 - (i - 1) * P_nm2) / i
        return (P_nm1, P_n)

    P_nm2 = np.ones_like(x)  # P_0(x)
    P_nm1 = x  # P_1(x)

    # If n == 0, return P_0(x), else compute recursively
    return lax.cond(
        n == 0,
        lambda: P_nm2,
        lambda: lax.fori_loop(2, n + 1, body_fun, (P_nm2, P_nm1))[1]
    )

def jacobi_polynomial(n, alpha, beta, x):
    """Computes P_n^(alpha, beta)(x) using JAX-friendly recurrence."""
    def body_fun(i, carry):
        P_nm2, P_nm1 = carry
        A_k = (2 * i + alpha + beta - 1) * (2 * i + alpha + beta - 2)
        B_k = (2 * i + alpha + beta - 2) * (2 * i + alpha + beta - 1) * (2 * i + alpha + beta)
        C_k = (2 * i + alpha + beta) * (alpha + i - 1) * (beta + i - 1)

        P_n = ((B_k * (x - (alpha**2 - beta**2) / A_k) * P_nm1) - (C_k * P_nm2)) / (A_k * i)
        return (P_nm1, P_n)

    P_nm2 = np.ones_like(x)  # P_0^(alpha, beta)(x)
    P_nm1 = 0.5 * ((alpha - beta) + (alpha + beta + 2) * x)  # P_1^(alpha, beta)(x)

    return lax.cond(
        n == 0,
        lambda: P_nm2,
        lambda: lax.fori_loop(2, n + 1, body_fun, (P_nm2, P_nm1))[1]
    )




@jit
def d_0_0(ell, beta):
    """Computes d_{0,0}^ell(beta) using Legendre polynomials."""
    return legendre_polynomial(ell, np.cos(beta))

@jit
def d_2_2(ell, beta):
    """Computes d_{2,2}^ell(beta) using precomputed values when possible."""
    ell = np.asarray(ell)  # Ensure JAX-traced value
    exact_coeff = np.sqrt((ell - 1) * ell * (ell + 1) * (ell + 2) / 4)
    approx_coeff = ell**2 / 2  # Large-ell approximation
    coeff = np.where(ell > 200, approx_coeff, exact_coeff)

    # Use precomputed values for large ell, otherwise use JAX recurrence
    P_jacobi = jacobi_polynomial(ell - 2, 2, 2, np.cos(beta)),
   

    return (-1) ** ell * coeff * (np.sin(beta / 2) ** 2) * P_jacobi


@jit
def d_2_0(ell, beta):
    """Computes d_{2,0}^ell(beta) using precomputed values when possible."""
    ell = np.asarray(ell)  # Ensure JAX-traced value
    exact_coeff = np.sqrt((ell - 1) * ell * (ell + 1) * (ell + 2) / 4)
    approx_coeff = (ell ** 2) / 2  # Large-ell approximation
    coeff = np.where(ell > 200, approx_coeff, exact_coeff)

    # Use precomputed values for large ell, otherwise use JAX recurrence
    P_jacobi = jacobi_polynomial(ell - 2, 1, 1, np.cos(beta)),


    return coeff * np.sin(beta) ** 2 * P_jacobi
# Vectorize over `ell` and beta
d_0_0_vmap = jax.vmap(jax.vmap(d_0_0, in_axes=(0, None)) , in_axes=(None, 0))
d_2_2_vmap = jax.vmap(jax.vmap(d_2_2, in_axes=(0, None)) , in_axes=(None, 0))
d_2_0_vmap = jax.vmap(jax.vmap(d_2_0, in_axes=(0, None)) , in_axes=(None, 0))

class CorrelationFunction:
    def __init__(self, angular_two_point):
        """
        Initializes CorrelationFunction with an AngularTwoPoint instance.

        Args:
            angular_two_point (AngularTwoPoint): Instance providing Cl values.
        """
        self.angular_two_point = angular_two_point

        # Determine spin values based on tracer type
        self.s1 = 2 if isinstance(angular_two_point.tracer1, ShearTracer) else 0
        self.s2 = 2 if isinstance(angular_two_point.tracer2, ShearTracer) else 0


    def get_xi(self, ells, ks, theta):
        """
        Compute the angular correlation function xi(theta) using a numerically stable summation.

        Args:
            ells (jax.numpy.ndarray): Multipole moments.
            ks (jax.numpy.ndarray): Wavenumber grid.
            theta (jax.numpy.ndarray): Angles in radians.

        Returns:
            (jax.numpy.ndarray, jax.numpy.ndarray): Computed xi_+(theta) and xi_-(theta).
        """
        # Compute Cl using the AngularTwoPoint instance
        Cls = self.angular_two_point.get_Cl(ells, nl=0, ks=ks)

        Cl_EE = Cls
        Cl_BB = np.zeros_like(Cl_EE)  # No B-modes included

        Ntomo = Cls.shape[1]  # Number of tomographic bins
        Ntheta = len(theta)

        # Compute Wigner-d matrix elements
        if self.s1 == 0 and self.s2 == 0:
            d_l_theta_plus = d_0_0_vmap(ells, theta)
            d_l_theta_minus = d_l_theta_plus
        elif self.s1 == 2 and self.s2 == 0:
            d_l_theta_plus = d_2_0_vmap(ells, theta)
            d_l_theta_minus = d_l_theta_plus
        elif self.s1 == 2 and self.s2 == 2:
            d_l_theta_plus = d_2_2_vmap(ells, theta)
            sign = np.where(ells % 2 == 0, 1.0, -1.0)
            d_l_theta_minus = d_l_theta_plus * sign
        else:
            raise ValueError("Spin values not as expected")

        # Compute the prefactor (2ℓ + 1) / (4π)
        prefactor = (2 * ells + 1) / (4 * np.pi)

        # Initialize xi arrays
        xi_plus = np.zeros((Ntheta, Ntomo, Ntomo))
        xi_minus = np.zeros((Ntheta, Ntomo, Ntomo))

        # Iterate over tomographic bins
        for i in range(Ntomo):
            for j in range(Ntomo):
                Cl_ab_plus = Cl_EE[:, i, j]  # Shape: (Nells,)
                Cl_ab_minus = Cl_ab_plus  # Ignore B-modes for now

                for k in range(Ntheta):

                    # Compute log-sum-exp terms for numerical stability
                    log_terms_plus = np.log(prefactor) + np.log(np.abs(Cl_ab_plus)) + np.log(np.abs(d_l_theta_plus[k]))
                    log_terms_minus = np.log(prefactor) + np.log(np.abs(Cl_ab_minus)) + np.log(np.abs(d_l_theta_minus[k]))

                    # Compute max values for log-sum-exp trick
                    max_log_plus = np.max(log_terms_plus, axis=0)
                    max_log_minus = np.max(log_terms_minus, axis=0)

                    # Numerically stable sum over ells
                    xi_plus = xi_plus.at[k, i, j].set(np.exp(max_log_plus) * np.sum(np.exp(log_terms_plus - max_log_plus), axis=0))
                    xi_minus = xi_minus.at[k, i, j].set(np.exp(max_log_minus) * np.sum(np.exp(log_terms_minus - max_log_minus), axis=0))

        return xi_plus, xi_minus
