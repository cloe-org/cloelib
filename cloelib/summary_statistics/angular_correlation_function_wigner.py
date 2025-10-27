import jax.numpy as np
import jax
from jax import jit
from euclidlib.photo._le3_pk_wl import Result
from cloelib.observables.photo import ShearTracer, PositionsTracer
from .angular_correlation_function import AngularCorrelationFunction
from cloelib.auxiliary.cache import memoize_jax

@jit
def _d_0_0_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{0\,0}(\beta)\).

    For the two lowest multipoles (ℓ = 0, 1) the element is returned from
    closed-form expressions.  Higher orders are computed with a stable
    three-term recurrence that is compatible with ``jax.jit`` tracing.

    Parameters
    ----------
    beta : float or jax.numpy.ndarray
        Polar angle in **radians**.  May be a scalar or an array
        broadcastable to the shape of *ell*.
    ell : int or jax.numpy.ndarray
        Total angular-momentum quantum number ``ℓ ≥ 0``.  Accepts a Python
        ``int`` (known at compile time) or a 0-D JAX array.

    Returns
    -------
    d00_ell : same type as *beta*
        The value of \(d^{\ell}_{0\,0}(\beta)\), with dtype promotions
        governed by JAX’s standard casting rules.
    """
    base_case_0 = np.ones_like(beta)
    base_case_1 = np.cos(beta)

    def recurrence_fn(ell, vals):
        prev, prev2 = vals
        new_val = (2 * ell - 1) / (ell) * base_case_1 * prev - (
            (ell - 1) / (ell)
        ) * prev2

        return new_val, prev

    return np.where(
        ell == 0,
        base_case_0,
        np.where(
            ell == 1,
            base_case_1,
            jax.lax.fori_loop(2, ell + 1, recurrence_fn, (base_case_1, base_case_0))[0],
        ),
    )


@jit
def _d_2_2_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{2\,2}(\beta)\).

    Closed-form expressions are returned for the first two multipoles
    (ℓ = 2, 3).  For higher orders a numerically stable three-term
    recurrence is used, and once ℓ exceeds a configurable threshold
    (ℓ ≈ 30 000) the algorithm switches to an asymptotic two-term
    approximation that remains compatible with ``jax.jit``.

    Parameters
    ----------
    beta : float
        Polar angle in **radians** at which the element is evaluated.
        May be a scalar or an array broadcastable to the shape of
        *ell*.
    ell : int
        Total angular-momentum quantum number ``ℓ ≥ 2``.  Can be a Python
        ``int`` (traced at compile time) or a 0-D JAX array.

    Returns
    -------
    d22_ell : same type as *beta*
        The value of \(d^{\ell}_{2\,2}(\beta)\), with dtype promoted by
        JAX according to its standard casting rules.
    """
    # Base cases
    base_case_2 = (1 / 4) * (1 + np.cos(beta)) ** 2
    base_case_3 = np.cos(beta / 2) ** 4 * (3 * np.cos(beta) - 2)

    # Recurrence relation for small ell
    def recurrence_fn(ell, vals):
        prev, prev2 = vals
        new_val = (ell * (2 * ell - 1) / (ell**2 - 4)) * (
            (_d_0_0_ell_compute(beta, 1) - (4 / (ell * (ell - 1)))) * prev
            - (((ell - 1) ** 2 - 4) / ((ell - 1) * (2 * ell - 1))) * prev2
        )
        return new_val, prev

    # Approximation for large ell (fixed to explicitly pass `beta`)
    def approximation_fn(ell, vals):
        prev, prev2 = vals
        new_val = 2 * _d_0_0_ell_compute(beta, 1) * prev - prev2
        return new_val, prev

    # Compute using a JIT-compatible conditional switch
    def compute_d_2_2(ell):
        return jax.lax.cond(
            ell < 30000,
            lambda: jax.lax.fori_loop(
                4, ell + 1, recurrence_fn, (base_case_3, base_case_2)
            )[0],
            lambda: jax.lax.fori_loop(
                50, ell + 1, approximation_fn, (base_case_3, base_case_2)
            )[0],
        )

    return jax.lax.cond(
        ell == 2,
        lambda: base_case_2,
        lambda: jax.lax.cond(ell == 3, lambda: base_case_3, lambda: compute_d_2_2(ell)),
    )


@jit
def _d_2_m2_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{2,\,-2}(\beta)\).

    For the first two multipoles (ℓ = 2, 3) the value is returned from
    closed-form expressions.  Higher orders are obtained with a stable
    three-term recurrence.  When ℓ exceeds a configurable threshold
    (ℓ ≈ 30 000) the algorithm switches to an asymptotic two-term
    approximation that remains compatible with ``jax.jit``.

    Parameters
    ----------
    beta : float
        Polar angle in **radians** at which the element is evaluated.
        May be a scalar or an array broadcastable to the shape of
        *ell*.
    ell : int
        Total angular-momentum quantum number ``ℓ ≥ 2``.  Can be a
        Python ``int`` (traced at compile time) or a 0-D JAX array.

    Returns
    -------
    d2m2_ell : same type as *beta*
        The value of \(d^{\ell}_{2,\,-2}(\beta)\), with dtype promoted
        by JAX according to its standard casting rules.
    """
    # Base cases
    base_case_2 = (1 / 4) * (1 - np.cos(beta)) ** 2
    base_case_3 = np.sin(beta / 2) ** 4 * (3 * np.cos(beta) + 2)

    # Recurrence relation for small ell
    def recurrence_fn(ell, vals):
        prev, prev2 = vals
        new_val = (ell * (2 * ell - 1) / (ell**2 - 4)) * (
            (_d_0_0_ell_compute(beta, 1) + (4 / (ell * (ell - 1)))) * prev
            - (((ell - 1) ** 2 - 4) / ((ell - 1) * (2 * ell - 1))) * prev2
        )
        return new_val, prev

    # Approximation for large ell (fixed to explicitly pass `beta`)
    def approximation_fn(ell, vals):
        prev, prev2 = vals
        new_val = 2 * _d_0_0_ell_compute(beta, 1) * prev - prev2
        return new_val, prev

    # Compute using a JIT-compatible conditional switch
    def compute_d_2_2(ell):
        return jax.lax.cond(
            ell < 30000,
            lambda: jax.lax.fori_loop(
                4, ell + 1, recurrence_fn, (base_case_3, base_case_2)
            )[0],
            lambda: jax.lax.fori_loop(
                50, ell + 1, approximation_fn, (base_case_3, base_case_2)
            )[0],
        )

    return jax.lax.cond(
        ell == 2,
        lambda: base_case_2,
        lambda: jax.lax.cond(ell == 3, lambda: base_case_3, lambda: compute_d_2_2(ell)),
    )


@jit
def _d_2_0_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{20}(\beta)\).

    For the lowest multipoles (ℓ = 2, 3) the value is returned from a closed–
    form expression; higher orders are obtained recursively from
    \(d^{\ell-1}_{20}\) and \(d^{\ell-2}_{20}\).
    Beyond a configurable threshold (ℓ ≈ 30 000) the stable three-term
    *approximation*
    \(d^{\ell}_{20} ≈ 2\,d^{1}_{00}\,d^{\ell-1}_{20}-d^{\ell-2}_{20}\)
    is used to avoid numerical overflow while remaining compatible with
    `jax.jit`.

    Parameters
    ----------
    beta : float
        Polar angle in radians at which the element is evaluated.  May be a
        scalar or an array broadcastable to the shape of *ell*.
    ell : int
        Total angular-momentum quantum number ℓ ≥ 2.  Can be a Python `int`
        (traced at compile time) or a 0-D JAX array.

    Returns
    -------
    d20_ell : same type as *beta*
        The value of \(d^{\ell}_{20}(\beta)\), with dtype promoted by JAX
        according to standard casting rules.
    """
    # Base cases
    base_case_2 = np.sqrt(3 / 8) * np.sin(beta) ** 2
    base_case_3 = (np.sqrt(30) / 4) * np.sin(beta) ** 2 * np.cos(beta)

    # Recurrence relation for small ell
    def recurrence_fn(ell, vals):
        prev, prev2 = vals
        sqrt_l2_4 = np.sqrt(ell**2 - 4)
        sqrt_lm1_2_4 = np.sqrt((ell - 1) ** 2 - 4)
        new_val = ((2 * ell - 1) / sqrt_l2_4) * (
            _d_0_0_ell_compute(beta, 1) * prev - (sqrt_lm1_2_4 / (2 * ell - 1)) * prev2
        )
        return new_val, prev

    # Approximation for large ell (fixed to explicitly pass `beta`)
    def approximation_fn(ell, vals):
        prev, prev2 = vals
        new_val = 2 * _d_0_0_ell_compute(beta, 1) * prev - prev2
        return new_val, prev

    # Compute using a JIT-compatible conditional switch
    def compute_d_2_0(ell):
        return jax.lax.cond(
            ell < 30000,
            lambda: jax.lax.fori_loop(
                4, ell + 1, recurrence_fn, (base_case_3, base_case_2)
            )[0],
            lambda: jax.lax.fori_loop(
                50, ell + 1, approximation_fn, (base_case_3, base_case_2)
            )[0],
        )

    return jax.lax.cond(
        ell == 2,
        lambda: base_case_2,
        lambda: jax.lax.cond(ell == 3, lambda: base_case_3, lambda: compute_d_2_0(ell)),
    )


# define memoized versions of the Wigner d-matrix functions
d_0_0_ell = memoize_jax(_d_0_0_ell_compute)
d_2_2_ell = memoize_jax(_d_2_2_ell_compute)
d_2_m2_ell = memoize_jax(_d_2_m2_ell_compute)
d_2_0_ell = memoize_jax(_d_2_0_ell_compute)

# define vectorized version of the Wigner d-matrix functions
_d_0_0_vmap_compute = jax.vmap(jax.vmap(_d_0_0_ell_compute, (None, 0)), (0, None))
_d_2_2_vmap_compute = jax.vmap(jax.vmap(_d_2_2_ell_compute, (None, 0)), (0, None))
_d_2_m2_vmap_compute = jax.vmap(jax.vmap(_d_2_m2_ell_compute, (None, 0)), (0, None))
_d_2_0_vmap_compute = jax.vmap(jax.vmap(_d_2_0_ell_compute, (None, 0)), (0, None))

# define memoized versions of the vectorized Wigner d-matrix functions
d_0_0_vmap = memoize_jax(_d_0_0_vmap_compute)
d_2_2_vmap = memoize_jax(_d_2_2_vmap_compute)
d_2_m2_vmap = memoize_jax(_d_2_m2_vmap_compute)
d_2_0_vmap = memoize_jax(_d_2_0_vmap_compute)


class AngularCorrelationFunctionWigner(AngularCorrelationFunction):
    """Correlation function implementation using Wigner small-d matrices."""

    def __init__(self, angular_two_point, ells, ks):
        """
        Initialize CorrelationFunction with an AngularTwoPoint instance.

        The real-space angular correlation functions are computed from
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
        self.ells = ells
        self.ks = ks

        # Determine spin values based on tracer type
        self.s1 = 2 if isinstance(angular_two_point.tracer1, ShearTracer) else 0
        self.s2 = 2 if isinstance(angular_two_point.tracer2, ShearTracer) else 0
        if isinstance(angular_two_point.tracer1, ShearTracer) and isinstance(
            angular_two_point.tracer2, ShearTracer
        ):
            self.keys = ("SHE", "SHE")
        elif isinstance(angular_two_point.tracer1, PositionsTracer) and isinstance(
            angular_two_point.tracer2, PositionsTracer
        ):
            self.keys = ("POS", "POS")
        elif isinstance(angular_two_point.tracer1, PositionsTracer) and isinstance(
            angular_two_point.tracer2, ShearTracer
        ):
            self.keys = ("POS", "SHE")
        elif isinstance(angular_two_point.tracer1, ShearTracer) and isinstance(
            angular_two_point.tracer2, PositionsTracer
        ):
            # the order of this tuple does not matter
            # cosmolib format only supports ("POS", "SHE")
            self.keys = ("POS", "SHE")
        else:
            raise ValueError("Unsupported tracer combination")
    
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
        self.angular_two_point.get_Cl(self.ells, nl=0, ks=self.ks)

        Cl_EE = self.angular_two_point.C_ell_calc
        Cl_BB = np.zeros_like(Cl_EE)  # No B-modes included, set to zero for now
        Cl_EB = np.zeros_like(Cl_EE)
        Cl_BE = np.zeros_like(Cl_EE)

        Cl_plus = (Cl_EE + Cl_BB) + (Cl_EB + Cl_BE)
        Cl_minus = (Cl_EE + Cl_BB) - (Cl_EB + Cl_BE)

        Ntomo1 = Cl_EE.shape[1]  # Number of tomographic bins
        Ntomo2 = Cl_EE.shape[2]  # Number of tomographic bins

        Ntheta = len(theta)

        # Compute Wigner-d matrix elements
        if self.s1 == 0 and self.s2 == 0:
            d_ell_theta_plus = d_0_0_vmap(theta, self.ells)
            d_ell_theta_minus = d_ell_theta_plus
        elif self.s1 == 2 and self.s2 == 0:
            d_ell_theta_plus = d_2_0_vmap(theta, self.ells)
            d_ell_theta_minus = d_ell_theta_plus
        elif self.s1 == 2 and self.s2 == 2:
            d_ell_theta_plus = d_2_2_vmap(theta, self.ells)
            d_ell_theta_minus = d_2_m2_vmap(theta, self.ells)
        else:
            raise ValueError("Spin values not as expected")

        # Compute the prefactor (2*ell + 1) / (4*pi)
        prefactor = (2 * self.ells + 1) / (4 * np.pi)

        # Initialize xi arrays
        xi_plus = np.zeros((Ntheta, Ntomo1, Ntomo2))
        if self.s2 == 2:
            xi_minus = np.zeros((Ntheta, Ntomo1, Ntomo2))

        # Vectorized computation over (theta, tomo1, tomo2)
        # Compute xi_plus

        xi_plus = np.einsum("L,LIJ,TL->TIJ", prefactor, Cl_plus, d_ell_theta_plus)

        # Compute xi_minus if we have a spin2 tracer
        if self.s2 == 2:
            xi_minus = (-1) ** self.s2 * np.einsum(
                "L,LIJ,TL->TIJ", prefactor, Cl_minus, d_ell_theta_minus
            )
        _DictKey = tuple[str, str, int, int]
        xi_dict: dict[_DictKey, Result] = {}
        if self.s1 == 0 and self.s2 == 0:
            axis = (0,)
            for i in range(Ntomo1):
                for j in range(i,Ntomo2):
                    key = ("POS","POS",int(i+1),int(j+1))
                    xi_dict[key] = Result(array=xi_plus[:,i,j],ell=theta, axis=axis)
        elif self.s1 == 2 and self.s2 == 0:
            axis = (1,)
            for i in range(Ntomo1):
                for j in range(i,Ntomo2):
                    key = ("POS","SHEAR",int(i+1),int(j+1))
                    xi_dict[key] = Result(array=np.array([xi_plus[:,j,i],np.zeros(len(theta))]),ell=theta, axis=axis)
        elif self.s1 == 2 and self.s2 == 2: 
            axis = (2,)
            for i in range(Ntomo1):
                for j in range(i,Ntomo2):
                    key = ("SHEAR","SHEAR",int(i+1),int(j+1))
                    xi_dict[key] = Result(array=np.array([[xi_plus[:,i,j],np.zeros(len(theta))],[np.zeros(len(theta)),xi_minus[:,i,j]]]),ell=theta, axis=axis) 
            
        else:
            
            raise ValueError("Spin values not as expected")
        return xi_dict
