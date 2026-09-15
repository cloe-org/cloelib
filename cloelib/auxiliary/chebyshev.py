"""Module for Chebyshev polynomials and related utilities."""

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array

from cloelib.auxiliary.akima import akima_interpolation
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.photo import PositionsTracer, ShearTracer

jax.config.update("jax_enable_x64", True)


@jax.jit
def dct_type1(f_values: Array) -> Array:
    """Compute the Discrete Cosine Transform (DCT) of type I.

    Args:
        f_values: Input array of function values.

    Returns:
        DCT type I of the input array.
    """

    internal_reversed = jnp.flip(f_values[1:-1])
    x_ext = jnp.concatenate([f_values, internal_reversed])
    X = jnp.fft.rfft(x_ext)

    return jnp.real(X[: f_values.shape[0]])


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


@jax.jit
def chebyshev_coefficients(f_values: Array) -> Array:
    """Compute Chebyshev coefficients from function values at Chebyshev points.

    Args:
        f_values: Function values at Chebyshev points.
    Returns:
        Array of Chebyshev coefficients.
    """
    N = len(f_values)
    c = dct_type1(f_values) / (N - 1)

    scale = jnp.ones(N)
    scale = scale.at[0].set(0.5)
    scale = scale.at[-1].set(0.5)

    return c * scale


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


def clenshaws_curtis_quadrature(
    n: int, a: float, b: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the Clenshaw-Curtis quadrature of a function f on [a, b].

    Args:
        n: Number of quadrature points (N+1 nodi, con N = n-1).
        a: Start of the interval.
        b: End of the interval.
    Returns:
        cheb_pts : Chebyshev points on [a, b].
        weights : Corresponding weights for Clenshaw-Curtis quadrature.
    """
    N = n - 1

    # mu moments, mu_k = 2 / (1 - k^2) if k is even, 0 if k is odd
    k = jnp.arange(n)
    mu = jnp.where(k % 2 == 0, 2.0 / (1.0 - k**2), 0.0)
    mu = mu.at[0].set(2.0)
    mu = mu.at[-1].set(mu[-1] / 2.0)

    # IDCT-I using rFFT
    internal_reversed = jnp.flip(mu[1:-1])
    mu_ext = jnp.concatenate([mu, internal_reversed])

    w = jnp.fft.rfft(mu_ext)
    w = jnp.real(w[:n]) / N
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[-1].set(w[-1] / 2.0)

    # rescling weights to the interval [a, b]
    w = w * (b - a) / 2.0

    return chebyshev_points_interval(N, a, b), w


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
    zs = np.logspace(np.log10(1e-6), np.log10(20.0), 10000)
    chi_of_z = background.comoving_distance(zs)
    return akima_interpolation(zs, chi_of_z, chi)


def Pkl_unequaltime(
    k: jnp.ndarray,
    chi: jnp.ndarray,
    R: jnp.ndarray,
    perturbation: Perturbations,
) -> jnp.ndarray:
    """
    Compute the unequal-time matter power spectrum P(k, chi1, chi2) on the grid defined by k, chi, and R = chi1/chi2,
    using the geometric mean of the equal-time power spectra from two tracers.

    Arguments:
    k : jnp.ndarray
        Wavenumber at which to evaluate the power spectrum.
    chi: jnp.ndarray
        Comoving distance.
    R : jnp.ndarray
        Ratio array corresponding to chi1/chi2.
    perturbation : Perturbations
        Perturbations object with matter_power_spectrum method.
    Returns:
    Pk : jnp.ndarray shape (len(k), len(R), len(chi))
        Unequal-time matter power spectrum P(k, R, chi).
    """

    z_of_chi = comoving_distance_to_redshift(chi, perturbation.background)
    chi_R_flatten = jnp.outer(chi, R).flatten()

    # Equal-time power spectra
    Pk_chi = perturbation.matter_power_spectrum(z_of_chi, k)
    Pk_chi_R = 10 ** akima_interpolation(
        jnp.log10(Pk_chi), chi, chi_R_flatten, axis=0
    ).reshape(len(chi), len(R), len(k))

    return jnp.sqrt(jnp.einsum("jk,jik->kji", Pk_chi, Pk_chi_R))


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

    Pk = 10 ** Pkl_unequaltime_interp(jnp.log10(Pkl), ks, k_cheb)

    return jnp.apply_along_axis(chebyshev_coefficients, 0, Pk)


@jax.jit
def w_ell(c: Array, T_tilde: Array) -> Array:
    """
    Compute the matrix contractio  to obtain w_ell from Chebyshev coefficients and T_tilde.

    Parameters:
    c : jax.numpy.ndarray
        3D array of Chebyshev coefficients. Shape: (n_k_cheb + 1, chi1_n, chi2_n)
    T_tilde : jax.numpy.ndarray
        4D array of shape (ells, chi1_n, chi2_n, n_k_cheb + 1) representing T_tilde.

    Returns:
    jax.numpy.ndarray
        3D array of shape (ells, chi1_n, chi2_n) representing w_ell.
    """
    return jnp.einsum("ijk,ljki->ljk", c, T_tilde)


def _get_kernel_array_position(tracer: PositionsTracer, chi_grid):
    """
    Computes the kernel values for a given grid based on the specified cosmological probes.
    Returns a 2D array of kernel values, where rows correspond to the number of bins and columns correspond to the grid points.

    Parameters:
    tracer: An instance of a cosmological tracer class (e.g., PositionsTracer, ShearsTracer).
    grid: A 1D array of grid points where the kernel values need to be computed.
    Returns:
    kernel_array: A 2D array of shape (n_bins, len(grid)) containing the kernel values.
    """

    z_grid = comoving_distance_to_redshift(chi_grid, tracer.background)
    kernel_values = tracer.get_window_positions(tracer.z)
    return akima_interpolation(kernel_values, tracer.z, z_grid, axis=-1)


def _get_kernel_array_shear(tracer: ShearTracer, chi_grid):
    """
    Computes the kernel values for a given grid based on the specified cosmological probes.
    Returns a 2D array of kernel values, where rows correspond to the number of bins and columns correspond to the grid points.

    Parameters:
    tracer: An instance of a cosmological tracer class (e.g., PositionsTracer, ShearsTracer).
    grid: A 1D array of grid points where the kernel values need to be computed.
    Returns:
    kernel_array: A 2D array of shape (n_bins, len(grid)) containing the kernel values.
    """

    z_grid = comoving_distance_to_redshift(chi_grid, tracer.background)
    kernel_values = tracer.get_window_lensing(tracer.z)
    return akima_interpolation(kernel_values, tracer.z, z_grid, axis=-1) / chi_grid**2


def get_kernel_array(tracer, chi_grid):
    """
    Computes the kernel values for a given grid based on the specified cosmological probes.
    Returns a 2D array of kernel values, where rows correspond to the number of bins and columns correspond to the grid points.

    Parameters:
    tracer: An instance of a cosmological tracer class (e.g., PositionsTracer, ShearsTracer).
    grid: A 1D array of grid points where the kernel values need to be computed.
    Returns:
    kernel_array: A 2D array of shape (n_bins, len(grid)) containing the kernel values.
    """
    if isinstance(tracer, PositionsTracer):
        return _get_kernel_array_position(tracer, chi_grid)
    elif isinstance(tracer, ShearTracer):
        return _get_kernel_array_shear(tracer, chi_grid)
    else:
        raise ValueError("Tracer type not supported for kernel array computation.")


def combine_kernels(tracer_1, tracer_2, chi, R):
    """
    Combine the kernels of two tracers over given chi and R grids.
    Parameters:
    tracer_1 : Tracer
        First tracer object with perturbations attribute.
    tracer_2 : Tracer
        Second tracer object with perturbations attribute.
    chi : array-like
        Array of chi values.
    R : array-like
        Array of R values.
    Returns:
    Combined kernel array of shaepe (n_bins_1, n_bins_2, len(chi), len(R)).
    """

    W1_chi = get_kernel_array(tracer_1, chi)  # shape (n_bins_1, len(chi))
    W2_chi = get_kernel_array(tracer_2, chi)  # shape (n_bins_2, len(chi))

    chi_R = jnp.outer(chi, R).flatten()

    W1_chi_R = get_kernel_array(tracer_1, chi_R).reshape(
        W1_chi.shape[0], len(chi), len(R)
    )  # shape (n_bins_1, len(chi), len(R))
    W2_chi_R = get_kernel_array(tracer_2, chi_R).reshape(
        W2_chi.shape[0], len(chi), len(R)
    )  # shape (n_bins_2, len(chi), len(R))

    return jnp.einsum("ik,jkt->ijkt", W1_chi, W2_chi_R) + jnp.einsum(
        "jk,ikt->ijkt", W2_chi, W1_chi_R
    )
