"""Module for Chebyshev polynomial."""

import scipy
import jax
import numpy as np
import jax.numpy as jnp
from jax import Array

def chebyshev_points(n: int) -> Array:
    """Compute the Chebyshev points of the first kind.

    Args:
        n: Number of Chebyshev points to compute.
        dtype: The desired array type (jnp.ndarray or np.ndarray).

    Returns:
        Array of Chebyshev points.
    """
    k = jnp.arange(n+1)
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
    mapped_pts = 0.5*(x_stop - x_start)* (cheb_pts + 1) + x_start
    return mapped_pts

def chebyshev_coefficients(f_values: Array) -> Array:
    """Compute Chebyshev coefficients from function values at Chebyshev points.

    Args:
        f_values: Function values at Chebyshev points.
    Returns:
        Array of Chebyshev coefficients.
    """
    N = len(f_values)
    c = scipy.fft.dct(f_values, type=1) / (N - 1)
    c[0] /= 2
    c[-1] /= 2
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

    w = scipy.fft.dct(mu, type=1, norm=None) / (n - 1)
    w[0] /= 2
    w[-1] /= 2
    
    # Scale weights to the interval [a, b]
    w = (b - a) / 2 * w

    return chebyshev_points_interval(n-1, a, b), w

def Pkl_unequaltime(k, z1, z2, tracer_A, tracer_B):
    """
    Compute the unequal-time matter power spectrum P(k, z1, z2)
    using the geometric mean of the equal-time power spectra from two tracers.
    
    Arguments:
    k : float
        Wavenumber at which to evaluate the power spectrum.
    z1 : float
        Redshift corresponding to the first tracer.
    z2 : float
        Redshift corresponding to the second tracer.
    tracer_A : Tracer
        First tracer object with perturbations attribute.
    tracer_B : Tracer
        Second tracer object with perturbations attribute.
    Returns:
    Pk : float
        Unequal-time matter power spectrum P(k, z1, z2).
    """

    Pk_A = tracer_A.perturbations.matter_power_spectrum(z1, k)
    Pk_B = tracer_B.perturbations.matter_power_spectrum(z2, k)

    Pk = jax.numpy.sqrt(Pk_A*Pk_B)

    return Pk

def Pkl_unequaltime_interp(k_q, chi1_q, chi2_q, ks, chi1s, chi2s, Pk) -> jax.numpy.ndarray:
    """
    Interpolate the unequal-time matter power spectrum Pkl(k, chi1, chi2) on a grid.

    Utilizes interpax's 3D interpolation with Akima method to handle non-uniform grids. 
    Extrapolation is enabled for values outside the given grid.

    Parameters:
    k_q (jax.numpy.ndarray): 1D array of query points where interpolation is desired in k.
    chi1_q (jax.numpy.ndarray): 1D array of query points where interpolation is desired in chi1.
    chi2_q (jax.numpy.ndarray): 1D array of query points where interpolation is desired in chi2.
    ks (jax.numpy.ndarray): 1D array of k values where P(k) is evaluated.
    chi1s (jax.numpy.ndarray): 1D array of chi1 values where P(k) is evaluated.
    chi2s (jax.numpy.ndarray): 1D array of chi2 values where P(k) is evaluated.
    Pk (jax.numpy.ndarray): 3D array of shape (len(ks), len(chi1s), len(chi2s)). Unequal-time matter power spectrum values at (ks, chi1s, chi2s).

    Returns:
    jax.numpy.ndarray: 1D array of shape len(k_q) Interpolated unequal-time matter power spectrum values at (k_q, chi1_q, chi2_q).
    """
    return interpax.interp3d(
        k_q, 
        chi1_q, 
        chi2_q, 
        ks, 
        chi1s, 
        chi2s, 
        Pk, 
        method='akima', extrap=True
    )

Pkl_unequaltime_interp_vmap = jax.jit(
    jax.vmap(  
        jax.vmap(  
            jax.vmap(  
                Pkl_unequaltime_interp,
                in_axes=(None, None, 0, None, None, None, None),
            ),
            in_axes=(None, 0, None, None, None, None, None),
        ),
        in_axes=(0, None, None, None, None, None, None),
    )
)

def Pkl_chebyshev_coeffs(k_min, k_max, n_k_cheb, chi1, chi2, tracer_A, tracer_B):
    """
    Compute the Chebyshev coefficients for the unequal-time matter power spectrum
    P(k, chi1, chi2) over specified ranges and number of points.

    Parameters:
    k_min : float
        Minimum wavenumber.
    k_max : float
        Maximum wavenumber.
    n_k_cheb : int
        Number of Chebyshev points in k.
    chi1 : float
        Comoving distances chi1.
    chi2 : float
        Comoving distances chi2.
    tracer_A : Tracer
        First tracer object with perturbations attribute.
    tracer_B : Tracer
        Second tracer object with perturbations attribute.

    Returns:
    jax.numpy.ndarray
        3D array of Chebyshev coefficients for P(k, chi1, chi2).
    """
    
    ks = chebyshev_points_interval(n_k_cheb, k_min, k_max)

    Pk = Pkl_unequaltime(ks, chi1, chi2, tracer_A, tracer_B)

    Pkl_coeffs = chebyshev_coefficients(Pk)

    return Pkl_coeffs

def Pkl_chebyshev_coeffs_vmap(k_min, k_max, n_k_cheb, chi1, chi2, tracer_A, tracer_B):
    """
    Vectorized computation of Chebyshev coefficients for the unequal-time matter power spectrum
    P(k, chi1, chi2) over specified ranges and number of points.

    Parameters:
    k_min : float
        Minimum wavenumber.
    k_max : float
        Maximum wavenumber.
    n_k_cheb : int
        Number of Chebyshev points in k.
    chi1 : jax.numpy.ndarray
        1D array of comoving distances chi1.
    chi2 : jax.numpy.ndarray
        1D array of comoving distances chi2.
    tracer_A : Tracer
        First tracer object with perturbations attribute.
    tracer_B : Tracer
        Second tracer object with perturbations attribute.

    Returns:
    jax.numpy.ndarray
        3D array of Chebyshev coefficients for P(k, chi1, chi2).
    """
    out = np.zeros((n_k_cheb+1, len(chi1), len(chi2)))
    for chi1_i, chi1_val in enumerate(chi1):
        for chi2_j, chi2_val in enumerate(chi2):
            out[:, chi1_i, chi2_j] = Pkl_chebyshev_coeffs(k_min, k_max, n_k_cheb, chi1_val, chi2_val, tracer_A, tracer_B)
    return out