"""Common cosmology derived functions."""

# cloelib imports
from cloelib.auxiliary import units

# General imports
import numpy as np
from scipy import optimize, integrate

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


def hubble_rate(
    lna: float, h0: float, Omega_m: float, Omega_k: float, w0: float, wa: float
) -> np.ndarray:
    """
    Analytic integral of the Hubble rate H(z) for the w0wa model, see e.g.
    Equ.(9) of 1910.09273.

    Parameters
    ----------
    lna: float
        logarithm of the scale factor

    h0: float
        Hubble parameter at z=0 in km/s/Mpc

    Omega_m: float
        matter density parameter today

    Omega_k: float
        curvature parameter today

    w0: float
        w_0 of the CPL dark energy parametrisation

    wa: float
        w_a of the CPL dark energy parametrisation

    Returns
    -------
    h_z: float
        Hubble rate at the time ln(a)
    """
    z = np.exp(-lna) - 1
    h_z = h0 * np.sqrt(
        Omega_m * (1 + z) ** 3
        + Omega_k * (1 + z) ** 2
        + (1 - Omega_m - Omega_k)
        * (1 + z) ** (3 * (1 + w0 + wa))
        * np.exp(-3 * wa * z / (1 + z))
    )
    return h_z


def growth_function_ODE_derivative(
    lna: float,
    y: np.ndarray,
    h0: float,
    Omega_m: float,
    Omega_m_geo: float,
    Omega_k: float,
    w0: float,
    wa: float,
) -> np.ndarray:
    """Calculates the derivative of the growth factor for the differential
    equation in the format for scipy.integrate.solve_ivp . Equation can be
    found in 0810.1744, Equ. (10)

    Parameters
    ----------
    lna: float
        logarithm of the scale factor

    y: np.ndarray
        current step of the ODE

    h0: float
        Hubble parameter at z=0 in km/s/Mpc

    Omega_m: float
        matter density parameter today, either geometry or growth for the split

    Omega_m_geo: float
        matter density parameter today, in the geometry regime of the split

    Omega_k: float
        curvature parameter today

    w0: float
        w_0 of the CPL dark energy parametrisation

    wa: float
        w_a of the CPL dark energy parametrisation

    Returns
    -------
    deriv: np.ndarray
        time derivative of the state y at time ln(a)
    """
    deriv = np.zeros([len(y)])
    z = np.exp(-lna) - 1

    h_rate = hubble_rate(lna, h0, Omega_m, Omega_k, w0, wa)
    h_geo = hubble_rate(lna, h0, Omega_m_geo, Omega_k, w0, wa)

    hPrime_geo = optimize.approx_fprime(
        lna, hubble_rate, 2e-8, h0, Omega_m_geo, Omega_k, w0, wa
    )
    c1 = 4 + hPrime_geo / h_geo
    c2 = 3 + hPrime_geo / h_geo - 3 / 2 * Omega_m * (1 + z) ** 3 * (h0 / h_rate) ** 2

    deriv[0] = y[1]
    deriv[1] = -c1[0] * y[1] - c2[0] * y[0]
    return deriv


def growth_function_ODE(background, zs: np.ndarray, Omega_m=-1) -> np.ndarray:
    """Calculates the differential equation using scipy.integrate.solve_ivp .
    The initial values for the ODE are from Miranda et al. 1712.04289, p. 4

    Parameters
    ----------
    background: Background
        background cosmology calculated e.g. by an Einstein-Boltzmann solver

    zs: numpy.ndarray
        redshifts

    Omega_m: float
        matter density parameter today, defaults to -1 as an implicit flag for
        users who want to use this function outside of the growth-geometry split

    Returns
    -------
    growth: np.ndarray
        growth factor G(z) at the redshifts zs
    """

    lna_vec = np.flip(np.log(1 / (1 + zs)))
    zinit = 1000

    # Used as a flag, so that the function is usable without specifying Omega_m
    if Omega_m == -1:
        Omega_m = background.Omega_cdm0 + background.Omega_b0

    Omega_m_geo = background.Omega_cdm0 + background.Omega_b0
    Omega_k = background.Omega_k0
    w0 = background.w0
    wa = background.wa
    h0 = background.H0

    e_z_init = (
        hubble_rate(np.log(1 / (1 + zinit)), h0, Omega_m, Omega_k, w0, wa) / 100 / h0
    )

    y0 = np.ones(2)

    y0[1] = (
        -6
        / 5
        * (1 - Omega_m - Omega_k)
        * (1 + zinit) ** (3 * (1 + w0 + wa))
        * np.exp(-3 * wa * zinit / (1 + zinit))
        * e_z_init ** (-2)
    )

    growth = integrate.solve_ivp(
        growth_function_ODE_derivative,
        (np.min(lna_vec), np.max(lna_vec)),
        y0,
        t_eval=lna_vec,
        args=(h0, Omega_m, Omega_m_geo, Omega_k, w0, wa),
    )
    return np.flip(growth.y[0])
