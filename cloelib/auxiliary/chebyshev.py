"""Module for Chebyshev polynomials and related utilities."""

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array

from cloelib.auxiliary.akima_spline import akima_interpolation


def dct_type1(f_values: Array) -> Array:
    """Compute the Discrete Cosine Transform (DCT) of type I.

    Args:
        f_values: Input array of function values.

    Returns:
        DCT type I of the input array.
    """
    # return scipy.fft.dct(f_values, type=1)
    N = f_values.shape[0]
    x_ext = jnp.concatenate([f_values, f_values[-2:0:-1]])
    X = jnp.fft.fft(x_ext)
    return jnp.real(X[:N])


def chebyshev_points(n: int) -> Array:
    """Compute the Chebyshev points of the first kind.

    Args:
        n: Number of Chebyshev points to compute.
        dtype: The desired array type (jnp.ndarray or np.ndarray).

    Returns:
        Array of Chebyshev points.
    """
    k = jnp.arange(n + 1)
    points = jnp.cos(jnp.pi * k / n)

    return points


def chebyshev_points_interval(n: int, x_start: float, x_stop: float) -> Array:
    """Compute Chebyshev points of the first kind on the interval [a, b].

    Args:
        n: Number of Chebyshev points to compute.
        a: Start of the interval.
        b: End of the interval.

    Returns:
        Array of Chebyshev points on the interval [a, b].
    """
    cheb_pts = chebyshev_points(n)
    mapped_pts = 0.5 * (x_stop - x_start) * (cheb_pts + 1) + x_start
    return mapped_pts


def chebyshev_coefficients(f_values: Array) -> Array:
    """Compute Chebyshev coefficients from function values at Chebyshev points.

    Args:
        f_values: Function values at Chebyshev points.
    Returns:
        Array of Chebyshev coefficients.
    """
    N = len(f_values)
    c = dct_type1(f_values) / (N - 1)
    c = c.at[0].set(c[0] / 2)
    c = c.at[-1].set(c[-1] / 2)
    # c[0] /= 2
    # c[-1] /= 2
    return c


def chebyshev_interpolation(x: Array, c: Array) -> Array:
    """Evaluate Chebyshev interpolation at given points.

    Args:
        c: Chebyshev coefficients.
        x: Points where to evaluate the interpolation.

    Returns:
        Array of interpolated values at points x.
    """
    x_scaled = (2 * x - (x.min() + x.max())) / (x.max() - x.min())
    return np.polynomial.chebyshev.chebval(x_scaled, c)


def clenshaws_curtis_quadrature(n: int, a: float, b: float) -> tuple[Array, Array]:
    """Compute the Clenshaw-Curtis quadrature of a function f on [a, b].

    Args:
        n: Number of Chebyshev points to use.
        a: Start of the interval.
        b: End of the interval.
    Returns:
        cheb_pts : Chebyshev points on [a, b].
        weights : Corresponding weights for Clenshaw-Curtis quadrature.
    """

    # Modified Chebyshev moments of the first kind
    # mu = jnp.array([jnp.sqrt(2),0]+[(1+(-1)**k)/(1-k**2) for k in range(2,n)])

    mu = np.zeros(n)
    for i in range(0, n, 2):
        mu[i] = 2.0 / (1 - i**2)

    w = dct_type1(mu) / (n - 1)
    w[0] /= 2
    w[-1] /= 2

    # Scale weights to the interval [a, b]
    w = (b - a) / 2 * w

    return chebyshev_points_interval(n - 1, a, b), w


def comoving_distance_to_redshift(chi, background):
    """
    Convert comoving distance chi to redshift z using interpolation given a background model.
    Arguments:
    chi : float
        Comoving distance.
    background : Background
        Background cosmology object with comoving_distance method.
    Returns:
    z : float
        Redshift corresponding to the given comoving distance.
    """
    zs = np.logspace(np.log10(1e-4), np.log10(30.0), 10000)
    chi_of_z = background.comoving_distance(zs)
    return akima_interpolation(zs, chi_of_z, chi)


def Pkl_unequaltime(
    k: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    tracer_A,
    tracer_B,
) -> jnp.ndarray:
    """
    Compute the unequal-time matter power spectrum P(k, chi1, chi2)
    using the geometric mean of the equal-time power spectra from two tracers.

    Arguments:
    k : jnp.ndarray
        Wavenumber at which to evaluate the power spectrum.
    chi1 : jnp.ndarray
        Comoving distance corresponding to the first tracer.
    chi2 : jnp.ndarray
        Comoving distance corresponding to the second tracer.
    tracer_A : Tracer
        First tracer object with perturbations attribute.
    tracer_B : Tracer
        Second tracer object with perturbations attribute.
    Returns:
    Pk : jnp.ndarray shape (len(k), len(chi1), len(chi2))
        Unequal-time matter power spectrum P(k, chi1, chi2).
    """

    z1 = comoving_distance_to_redshift(chi1, tracer_A.background)
    z2 = comoving_distance_to_redshift(chi2, tracer_B.background)

    Pk_A = tracer_A.perturbations.matter_power_spectrum(z1, k)
    Pk_B = tracer_B.perturbations.matter_power_spectrum(z2, k)

    Pk = jax.numpy.sqrt(jnp.einsum("ij,kj->jik", Pk_A, Pk_B))

    return Pk


@jax.jit
def Pkl_unequaltime_interp(Pkl, ks, k_q) -> jax.numpy.ndarray:
    """
    Interpolate the unequal-time matter power spectrum over k_q values with a fixed chi1, chi2 grid.
    Parameters
    ----------
    Pkl : jax.numpy.ndarray
        3D array of matter power spectrum values, shape (len(ks), len(chi1s), len(chi2s)).
    ks : jax.numpy.ndarray
        1D array of wavenumber grid points corresponding to the first axis of Pkl.
    k_q : jax.numpy.ndarray
        1D array of wavenumber query points.
    Returns
    -------
    jax.numpy.ndarray
        3D array of same shape as Pkl with the interpolated unequal-time matter power spectrum values.
    """
    return akima_interpolation(Pkl, ks, k_q, axis=0)


@jax.jit
def Pkl_chebyshev_coeffs(
    Pkl: jnp.ndarray,
    ks: jnp.ndarray,
    k_cheb: jnp.ndarray,
) -> jax.numpy.ndarray:
    """
    Compute the Chebyshev coefficients for the unequal-time matter power spectrum
    P(k, chi1, chi2) given a 1D array of wavenumber chebyshev points.

    Parameters:
    Pkl : jax.numpy.ndarray
        3D array of matter power spectrum values, shape (len(ks), len(chi1s), len(chi2s)).
    ks : jax.numpy.ndarray
        1D array of wavenumber grid points corresponding to the first axis of Pkl.
    k_cheb : jax.numpy.ndarray
        1D array of wavenumber Chebyshev points.
    Returns:
    jax.numpy.ndarray
        3D array of Chebyshev coefficients for P(k, chi1, chi2).
    """

    Pk = Pkl_unequaltime_interp(Pkl, ks, k_cheb)

    return jnp.apply_along_axis(chebyshev_coefficients, 0, Pk)


def w_ell(c: Array, T_tilde: Array) -> Array:
    """
    Compute the matrix contractio  to obtain w_ell from Chebyshev coefficients and T_tilde.

    Parameters:
    c : jax.numpy.ndarray
        3D array of Chebyshev coefficients. Shape: (n_k_cheb + 1, chi1_n, chi2_n)
    T_tilde : jax.numpy.ndarray
        3D array of shape (chi1_n, chi2_n, n_k_cheb + 1) representing T_tilde.

    Returns:
    jax.numpy.ndarray
        2D array of shape (chi1_n, chi2_n) representing w_ell.
    """
    return jnp.einsum("ijk,jki->jk", c, T_tilde)
