"""Common cosmology derived functions."""

# cloelib imports
from cloelib.auxiliary import units
from cloelib.auxiliary.akima import akima_interpolation

# General imports
import numpy as np

_log10_GRAVITATIONAL_CONSTANT = np.log10(units.GRAVITATIONAL_CONSTANT)


def rho_crit(background, zs: np.ndarray) -> np.ndarray:
    """
    Return the critical density as a function of redshift.

    Units: Mpc^{-3} Msun

    Args:
      background (Background): Background class containing cosmology
      zs (np.ndarray): Redshifts.

    Returns:
      (float): Critical density value at the specified redshift.
    """
    log10_h_in_seconds = np.log10(background.hubble_parameter(zs) / units.MPC_TO_KM)
    return (3.0 / 8.0 / np.pi) * 10 ** (
        2.0 * log10_h_in_seconds - _log10_GRAVITATIONAL_CONSTANT
    )


def redshift_of_comoving_distance(background, chi):
    """
    Return the redshift at which the comoving distance equals `chi`.

    Inverts `background.comoving_distance` by Akima interpolation on a
    tabulated grid of 10000 redshifts, log-spaced in [1e-6, 20].

    Args:
      background (Background): Background class containing cosmology.
      chi (np.ndarray): Comoving distances, in Mpc.

    Returns:
      (np.ndarray): Redshifts, same shape as `chi`.
    """
    zs = np.logspace(np.log10(1e-6), np.log10(20.0), 10000)
    chi_of_z = background.comoving_distance(zs)
    return akima_interpolation(zs, chi_of_z, chi)


def dV_dzdO(background, zs: np.ndarray, hubble_units=False) -> np.ndarray:
    """
    Return the volume element per redshit per solid angle at the redshift requested.

    Parameters
    ----------
    background: Background
        Background class containing cosmology
    zs :np.ndarray
        Redshifts.
    hubble_units: (Optional) bool
        Flag to specify if output in h units, defaults to False


    Returns:
      (np.ndarray): volume element in Mpc^3 h^{-3}
    """
    _dV_dzdO = (
        units.SPEED_OF_LIGHT
        / 1.0e3
        * background.comoving_distance(zs) ** 2.0
        / background.hubble_parameter(zs)
    )
    if hubble_units:
        _dV_dzdO *= (background.H0 / 100.0) ** 3.0
    return _dV_dzdO


def rdrag_fitting_function(background, neff=3.046):
    r"""Compute the sound horizon at drag epoch.

    Uses the fitting formula Eq.17
    of [1411.1074](https://arxiv.org/abs/1411.1074)

    Args:
      background (Background): Background class containing cosmology
      neff (float): Effective number of neutrinos.

    Returns:
      r_d (float): Sound horizon at drag epoch in Mpc
    """
    omega_cb = background.Omega_cdm0 * background.h**2
    omega_b = background.Omega_b0 * background.h**2
    omega_nu = background.mnu / 93.14

    r_d = (
        56.067
        * np.exp(-49.7 * (omega_nu + 0.002) ** 2)
        / (omega_cb**0.2436 * omega_b**0.128876 * (1 + (neff - 3.046) / 30.6))
    )
    return r_d


def z_star_fitting_function(background):
    r"""Compute the redshift of photon decoupling.

    Assumes a cosmology-independent z_star.

    Parameters
    ----------
    background: Background
        Background class containing cosmology

    Returns
    -------
    z_star: float
        redshift of photon decoupling.
    """
    return 1090.0
