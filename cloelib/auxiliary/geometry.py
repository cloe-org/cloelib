"""Module for the geometry of non-flat FLRW spacetimes."""

import jax.numpy as jnp

from cloelib.auxiliary.units import SPEED_OF_LIGHT

_C_KM_S = SPEED_OF_LIGHT / 1000


def transverse_distance(chi, Omega_k0, H0):
    r"""Return the transverse comoving distance $f_K(\chi)$ in Mpc.

    Args:
        chi (array_like): Comoving radial distance(s) in Mpc; may be negative.
        Omega_k0 (float): Curvature density parameter today.
        H0 (float): Hubble constant in km/s/Mpc.

    Returns:
        (jax.numpy.ndarray): $f_K(\chi)$ in Mpc, same shape as ``chi``.
    """
    chi = jnp.asarray(chi)
    u = Omega_k0 * (chi * H0 / _C_KM_S) ** 2
    small = jnp.abs(u) < 1e-4
    a = jnp.sqrt(jnp.where(small, 1.0, jnp.abs(u)))
    exact = jnp.where(u > 0, jnp.sinh(a), jnp.sin(a)) / a
    series = 1 + u / 6 + u**2 / 120
    return chi * jnp.where(small, series, exact)


def radial_distance(d_M, Omega_k0, H0):
    r"""Return the comoving radial distance $\chi$ for a transverse distance $D_M$.

    Args:
        d_M (array_like): Transverse comoving distance(s) in Mpc.
        Omega_k0 (float): Curvature density parameter today.
        H0 (float): Hubble constant in km/s/Mpc.

    Returns:
        (jax.numpy.ndarray): $\chi$ in Mpc, same shape as ``d_M``.
    """
    d_M = jnp.asarray(d_M)
    u = Omega_k0 * (d_M * H0 / _C_KM_S) ** 2
    small = jnp.abs(u) < 1e-4
    a = jnp.sqrt(jnp.where(small, 1.0, jnp.abs(u)))
    exact = jnp.where(u > 0, jnp.arcsinh(a), jnp.arcsin(a)) / a
    series = 1 - u / 6 + 3 * u**2 / 40
    return d_M * jnp.where(small, series, exact)


def cos_K(chi, Omega_k0, H0):
    r"""Return $\mathrm{d}f_K/\mathrm{d}\chi$.

    Args:
        chi (array_like): Comoving radial distance(s) in Mpc.
        Omega_k0 (float): Curvature density parameter today.
        H0 (float): Hubble constant in km/s/Mpc.

    Returns:
        (jax.numpy.ndarray): Dimensionless, same shape as ``chi``.
    """
    chi = jnp.asarray(chi)
    u = Omega_k0 * (chi * H0 / _C_KM_S) ** 2
    small = jnp.abs(u) < 1e-4
    a = jnp.sqrt(jnp.where(small, 1.0, jnp.abs(u)))
    exact = jnp.where(u > 0, jnp.cosh(a), jnp.cos(a))
    series = 1 + u / 2 + u**2 / 24
    return jnp.where(small, series, exact)


def curvature(Omega_k0, H0):
    r"""Return the spatial curvature $K = -\Omega_K (H_0/c)^2$ in Mpc$^{-2}$."""
    return -Omega_k0 * (H0 / _C_KM_S) ** 2


def is_flat(Omega_k0) -> bool:
    """Return whether ``Omega_k0`` is exactly zero; a JAX-traced value is not."""
    try:
        return float(Omega_k0) == 0.0
    except TypeError:
        return False


def limber_wavenumber(ells, f_K, K):
    r"""Return the Limber wavenumber $k = \sqrt{(\nu/f_K)^2 - K}$ in Mpc$^{-1}$.

    Args:
        ells (array_like): Multipoles, shape ``(n_ell,)``.
        f_K (array_like): Transverse comoving distance in Mpc, shape ``(n_z,)``.
        K (float): Spatial curvature in Mpc$^{-2}$.

    Returns:
        (jax.numpy.ndarray): Shape ``(n_ell, n_z)``.
    """
    q2 = (jnp.expand_dims(ells + 0.5, 1) / f_K) ** 2
    return jnp.sqrt(jnp.where(q2 > K, q2 - K, q2))


def potential_response(ells, f_K, K):
    r"""Return the curved-Poisson factor $\nu^2 / (\nu^2 - 4 K f_K^2)$ of potential tracers.

    Args:
        ells (array_like): Multipoles, shape ``(n_ell,)``.
        f_K (array_like): Transverse comoving distance in Mpc, shape ``(n_z,)``.
        K (float): Spatial curvature in Mpc$^{-2}$.

    Returns:
        (jax.numpy.ndarray): Shape ``(n_ell, n_z)``.
    """
    nu2 = jnp.expand_dims(ells + 0.5, 1) ** 2
    den = nu2 - 4 * K * f_K**2
    valid = den > 0
    return jnp.where(valid, nu2 / jnp.where(valid, den, 1.0), 0.0)
