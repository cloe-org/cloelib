from functools import partial

import jax.numpy as np
import jax
from jax import jit
from cosmolib.data import TwoPointCorrelationFunction
from cloelib.observables.photo import ShearTracer, PositionsTracer
from .angular_correlation_function import AngularCorrelationFunction
from cloelib.auxiliary.cache import memoize_jax


# Each Wigner small-d element d^l_{m m'}(beta) is computed with an upward
# three-term recurrence in l, seeded by closed-form expressions at the two
# lowest multipoles l0 and l0 + 1. Each element is defined by
#   - l0: the lowest multipole (max(|m|, |m'|)),
#   - an ``init`` function returning (d^{l0}, d^{l0+1}),
#   - a ``step`` function returning d^l from (cos(beta), l, d^{l-1}, d^{l-2}).
# The same definitions are used for single-multipole evaluation
# (``_d_*_ell_compute``) and for full tables over all multipoles
# (``_d_*_vmap_compute``).


def _d_0_0_init(beta):
    return np.ones_like(beta), np.cos(beta)


def _d_0_0_step(cos_beta, ell, prev, prev2):
    return (2 * ell - 1) / ell * cos_beta * prev - (ell - 1) / ell * prev2


def _d_2_2_init(beta):
    return (1 / 4) * (1 + np.cos(beta)) ** 2, np.cos(beta / 2) ** 4 * (
        3 * np.cos(beta) - 2
    )


def _d_2_2_step(cos_beta, ell, prev, prev2):
    return (ell * (2 * ell - 1) / (ell**2 - 4)) * (
        (cos_beta - (4 / (ell * (ell - 1)))) * prev
        - (((ell - 1) ** 2 - 4) / ((ell - 1) * (2 * ell - 1))) * prev2
    )


def _d_2_m2_init(beta):
    return (1 / 4) * (1 - np.cos(beta)) ** 2, np.sin(beta / 2) ** 4 * (
        3 * np.cos(beta) + 2
    )


def _d_2_m2_step(cos_beta, ell, prev, prev2):
    return (ell * (2 * ell - 1) / (ell**2 - 4)) * (
        (cos_beta + (4 / (ell * (ell - 1)))) * prev
        - (((ell - 1) ** 2 - 4) / ((ell - 1) * (2 * ell - 1))) * prev2
    )


def _d_2_0_init(beta):
    return np.sqrt(3 / 8) * np.sin(beta) ** 2, (np.sqrt(30) / 4) * np.sin(
        beta
    ) ** 2 * np.cos(beta)


def _d_2_0_step(cos_beta, ell, prev, prev2):
    sqrt_l2_4 = np.sqrt(ell**2 - 4)
    sqrt_lm1_2_4 = np.sqrt((ell - 1) ** 2 - 4)
    return ((2 * ell - 1) / sqrt_l2_4) * (
        cos_beta * prev - (sqrt_lm1_2_4 / (2 * ell - 1)) * prev2
    )


def _recurrence_value(beta, ell, l0, init, step):
    """Evaluate d^ell(beta) for a single (possibly traced) multipole ``ell >= l0``."""
    cos_beta = np.cos(beta)
    d_l0, d_l1 = init(beta)

    def body(ell_i, vals):
        prev, prev2 = vals
        return step(cos_beta, ell_i, prev, prev2), prev

    d_ell = jax.lax.fori_loop(l0 + 2, ell + 1, body, (d_l1, d_l0))[0]
    return np.where(ell == l0, d_l0, np.where(ell == l0 + 1, d_l1, d_ell))


def _recurrence_table(beta, lmax, l0, init, step):
    """
    Evaluate d^l(beta) for all multipoles 0 <= l <= lmax in a single pass.

    Returns an array of shape ``(lmax + 1,) + beta.shape``. Entries with
    ``l < l0`` (where the element is not defined) are set to zero.
    """
    beta = np.asarray(beta, dtype=float)
    cos_beta = np.cos(beta)
    d_l0, d_l1 = init(beta)

    def body(vals, ell):
        prev, prev2 = vals
        new_val = step(cos_beta, ell, prev, prev2)
        return (new_val, prev), new_val

    ells = np.arange(l0 + 2, lmax + 1, dtype=beta.dtype)
    _, d_rest = jax.lax.scan(body, (d_l1, d_l0), ells)
    table = np.concatenate(
        [np.zeros((l0,) + beta.shape, beta.dtype), d_l0[None], d_l1[None], d_rest]
    )
    return table[: lmax + 1]


@jit
def _d_0_0_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{0\,0}(\beta)\).

    Parameters
    ----------
    beta : float or jax.numpy.ndarray
        Polar angle in **radians**.
    ell : int or jax.numpy.ndarray
        Multipole ``ℓ ≥ 0``. Can be a Python ``int`` or a 0-D JAX array.

    Returns
    -------
    d00_ell : same type as *beta*
        The value of \(d^{\ell}_{0\,0}(\beta)\).
    """
    return _recurrence_value(beta, ell, 0, _d_0_0_init, _d_0_0_step)


@jit
def _d_2_2_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{2\,2}(\beta)\).

    Parameters
    ----------
    beta : float or jax.numpy.ndarray
        Polar angle in **radians**.
    ell : int or jax.numpy.ndarray
        Multipole ``ℓ ≥ 2``. Can be a Python ``int`` or a 0-D JAX array.

    Returns
    -------
    d22_ell : same type as *beta*
        The value of \(d^{\ell}_{2\,2}(\beta)\).
    """
    return _recurrence_value(beta, ell, 2, _d_2_2_init, _d_2_2_step)


@jit
def _d_2_m2_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{2,\,-2}(\beta)\).

    Parameters
    ----------
    beta : float or jax.numpy.ndarray
        Polar angle in **radians**.
    ell : int or jax.numpy.ndarray
        Multipole ``ℓ ≥ 2``. Can be a Python ``int`` or a 0-D JAX array.

    Returns
    -------
    d2m2_ell : same type as *beta*
        The value of \(d^{\ell}_{2,\,-2}(\beta)\).
    """
    return _recurrence_value(beta, ell, 2, _d_2_m2_init, _d_2_m2_step)


@jit
def _d_2_0_ell_compute(beta: float, ell: int) -> float:
    r"""
    Evaluate the Wigner small-d matrix element \(d^{\ell}_{20}(\beta)\).

    Parameters
    ----------
    beta : float or jax.numpy.ndarray
        Polar angle in **radians**.
    ell : int or jax.numpy.ndarray
        Multipole ``ℓ ≥ 2``. Can be a Python ``int`` or a 0-D JAX array.

    Returns
    -------
    d20_ell : same type as *beta*
        The value of \(d^{\ell}_{20}(\beta)\).
    """
    return _recurrence_value(beta, ell, 2, _d_2_0_init, _d_2_0_step)


# define memoized versions of the Wigner d-matrix functions
d_0_0_ell = memoize_jax(_d_0_0_ell_compute)
d_2_2_ell = memoize_jax(_d_2_2_ell_compute)
d_2_m2_ell = memoize_jax(_d_2_m2_ell_compute)
d_2_0_ell = memoize_jax(_d_2_0_ell_compute)


@partial(jit, static_argnums=(1, 2, 3, 4))
def _recurrence_table_jit(theta, lmax, l0, init, step):
    return _recurrence_table(theta, lmax, l0, init, step)


def _make_table_compute(l0, init, step):
    """
    Build a function ``(theta, ells) -> d`` with ``d`` of shape ``(Ntheta, Nell)``.

    The recurrence is run once up to ``max(ells)`` for all angles, and the
    requested multipoles are then selected, so the cost scales as
    ``Ntheta * max(ells)``. ``ells`` must be concrete (not traced).
    """

    def compute(theta, ells):
        ells = np.asarray(ells).astype(int)
        lmax = int(np.max(ells))
        table = _recurrence_table_jit(np.atleast_1d(theta), lmax, l0, init, step)
        return table[ells].T

    return compute


# define vectorized versions of the Wigner d-matrix functions, of shape (Ntheta, Nell)
_d_0_0_vmap_compute = _make_table_compute(0, _d_0_0_init, _d_0_0_step)
_d_2_2_vmap_compute = _make_table_compute(2, _d_2_2_init, _d_2_2_step)
_d_2_m2_vmap_compute = _make_table_compute(2, _d_2_m2_init, _d_2_m2_step)
_d_2_0_vmap_compute = _make_table_compute(2, _d_2_0_init, _d_2_0_step)

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

        Parameters:
            angular_two_point (AngularTwoPoint): Object providing Cl evaluation and tracers.
            ells (jnp.ndarray): Multipole moments at which the Cl spectrum is evaluated.
            ks (jnp.ndarray): Wavenumber grid (only needed for computing Cl via angular_two_point).
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

    def build_xi_dict(self, all_xi, theta, Ntomo1, Ntomo2):
        """Return a dictionary of TwoPointCorrelationFunction objects."""
        xi_dict = {}

        tracer_info = {
            (0, 0): dict(
                label=("POS", "POS"), axis=(0,), slice=lambda i, j: all_xi[:, i, j]
            ),
            (2, 0): dict(
                label=("SHE", "POS"), axis=(1,), slice=lambda i, j: all_xi[:, :, i, j]
            ),
            (0, 2): dict(
                label=("POS", "SHE"), axis=(1,), slice=lambda i, j: all_xi[:, :, i, j]
            ),
            (2, 2): dict(
                label=("SHE", "SHE"),
                axis=(2,),
                slice=lambda i, j: all_xi[:, :, :, i, j],
            ),
        }

        key = (self.s1, self.s2)
        if key not in tracer_info:
            raise ValueError(f"Unsupported (s1, s2) combination: {key}")

        info = tracer_info[key]
        label, axis, slicer = info["label"], info["axis"], info["slice"]

        def j_range(i):
            return range(i, Ntomo2) if key in [(0, 0), (2, 2)] else range(Ntomo2)

        for i in range(Ntomo1):
            for j in j_range(i):
                xi_dict[label + (i + 1, j + 1)] = TwoPointCorrelationFunction(
                    array=slicer(i, j),
                    theta=theta,
                    axis=axis,
                )
        return xi_dict

    def get_xi(self, theta):
        """
        Compute the angular correlation function xi(theta) using the Wigner d-matrices.

        TODO: Also implement FFTLog which is likely faster
        WARNING: Currently assumes B-modes are zero, as they are not passed on from AngularTwoPoint

        Args:
            theta (jax.numpy.ndarray): Angles in radians. Should be JAX (`jax.numpy.ndarray`) ndarray.

        Returns:
            (jax.numpy.ndarray|(jax.numpy.ndarray, jax.numpy.ndarray)): Computed xi(theta) if at least one tracer is spin 0 (clustering or GGL)
                (`jax.numpy.ndarray`) or Computed xi_+(theta) and xi_-(theta) (`jax.numpy.ndarray`, `jax.numpy.ndarray`) if both tracers are spin 2 (cosmic shear)

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
        elif self.s1 == 0 and self.s2 == 2:
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
        if self.s1 == 2 and self.s2 == 2:
            xi_minus = np.zeros((Ntheta, Ntomo1, Ntomo2))

        # Vectorized computation over (theta, tomo1, tomo2)
        # Compute xi_plus

        xi_plus = np.einsum("L,LIJ,TL->TIJ", prefactor, Cl_plus, d_ell_theta_plus)

        # Compute xi_minus if we have a spin2 tracer
        if self.s1 == 2 and self.s2 == 2:
            xi_minus = (-1) ** self.s2 * np.einsum(
                "L,LIJ,TL->TIJ", prefactor, Cl_minus, d_ell_theta_minus
            )

        if self.s1 == 0 and self.s2 == 0:
            all_xi = np.zeros((Ntheta, Ntomo1, Ntomo2))
            all_xi = all_xi.at[:, :, :].set(xi_plus)

        elif (self.s1, self.s2) in [(2, 0), (0, 2)]:
            all_xi = np.zeros((2, Ntheta, Ntomo1, Ntomo2))
            all_xi = all_xi.at[0, :, :, :].set(xi_plus)

        elif self.s1 == 2 and self.s2 == 2:
            all_xi = np.zeros((2, 2, Ntheta, Ntomo1, Ntomo2))
            all_xi = all_xi.at[0, 0, :, :, :].set(xi_plus)
            all_xi = all_xi.at[1, 1, :, :, :].set(xi_minus)
        else:
            raise ValueError("Spin values not as expected")

        xi_dict = self.build_xi_dict(all_xi, theta, Ntomo1, Ntomo2)

        return xi_dict
