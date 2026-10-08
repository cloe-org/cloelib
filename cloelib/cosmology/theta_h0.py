"""Solve H0 from the acoustic scale for a supplied dark-energy density.

The recombination sound horizon is taken from one CAMB call. The angular
diameter distance to last scattering is then the integral of the supplied
``f_DE(z)``, so the same solver serves binned and splined ``w(z)`` and
``f_DE(z)`` models.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
import camb

# c / (100 km/s/Mpc), so H0mpc = h / C_KM_S_MPC has units of 1/Mpc.
C_KM_S_MPC = 2997.92458

# Effective massive-neutrino matter fraction G(z, mnu): the WMAP7 radiation
# multiplier (Appendix C of arXiv:2502.07185) over-estimates E(z) at z ~ 1000
# and biases H0 from theta by ~0.6 km/s/Mpc. CAMB's nu contribution to E^2 is
# well approximated as omnu * (1+z)^3 * G(z, mnu) with G -> 1 today and G -> 0
# at early times (Lesgourgues/Komatsu-style transition, fitted to CAMB).
_NU_MATTER_SCALE = 6.328
_NU_MATTER_POWER = 0.5


def _camb_recombination_at_theta(
    theta, omch2, ombh2, omega_k=0.0, mnu=0.06, TCMB=2.7255, nnu=3.044
):
    """Single CAMB call: rstar and zstar depend only weakly on H0, so fix them here."""
    original_feedback_level = camb.config.FeedbackLevel
    try:
        camb.set_feedback_level(0)
        p = camb.CAMBparams()
        p.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="ppf")
        p.set_cosmology(
            ombh2=ombh2,
            omch2=omch2,
            omk=omega_k,
            mnu=mnu,
            cosmomc_theta=theta,
            TCMB=TCMB,
            nnu=nnu,
        )
        derived = camb.get_background(p).get_derived_params()
    finally:
        camb.config.FeedbackLevel = original_feedback_level
    return derived["zstar"], derived["rstar"]


def _z_grid_to_zstar(zstar, n_low=265, n_high=512):
    z_low = np.linspace(0.0, 3.0, n_low)
    z_high = np.logspace(np.log10(3.001), np.log10(zstar), n_high)
    return np.unique(np.concatenate((z_low, z_high)))


def omega_r_h2(TCMB=2.7255, nnu=3.044):
    """Photon + massless neutrino contribution to Omega_r h^2 (CAMB convention)."""
    ogamma = 2.4728e-5 * (TCMB / 2.7255) ** 4
    return ogamma * (1.0 + (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * nnu)


def _neutrino_matter_fraction(z, mnu):
    """Fraction of omnuh2 that contributes as non-relativistic matter at redshift z."""
    z = np.asarray(z, dtype=float)
    if mnu <= 0.0:
        return np.zeros_like(z)
    z_transition = _NU_MATTER_SCALE / mnu
    return 1.0 / (1.0 + ((1.0 + z) / z_transition) ** _NU_MATTER_POWER)


def _comoving_distance_to_zstar(
    H0, z_star_array, de_density_factor, omch2, ombh2, mnu=0.06, omega_k=0.0, TCMB=2.7255, nnu=3.044
):
    h = H0 / 100.0
    H0mpc = h / C_KM_S_MPC
    omnuh2 = mnu / 93.14
    omega_b = ombh2 / h**2
    omega_c = omch2 / h**2
    omega_nu = omnuh2 / h**2
    omega_r = omega_r_h2(TCMB, nnu) / h**2
    omega_de = 1.0 - omega_b - omega_c - omega_nu - omega_r - omega_k / h**2
    nu_matter = _neutrino_matter_fraction(z_star_array, mnu)
    omega_m = omega_b + omega_c + omega_nu * nu_matter
    E_grid = np.sqrt(
        omega_r * (1.0 + z_star_array) ** 4
        + omega_m * (1.0 + z_star_array) ** 3
        + omega_de * de_density_factor
    )
    r_dimless = cumulative_trapezoid(1.0 / E_grid, z_star_array, initial=0.0)
    return r_dimless[-1] / H0mpc


def theta_to_H0(
    theta,
    omch2,
    ombh2,
    de_density,
    omega_k=0.0,
    mnu=0.06,
    TCMB=2.7255,
    nnu=3.044,
    h0_bracket=(50.0, 90.0),
):
    """Solve for H0 such that ``r_s / D_M(z_*) = theta``.

    ``theta`` is the acoustic scale itself (about 0.01), not the CosmoMC
    parameter ``100 * theta``. ``de_density(z)`` returns ``f_DE(z)`` on a
    redshift array and must not depend on H0. ``r_s`` and ``z_*`` come from
    one CAMB recombination call; only the distance integral runs inside the
    root finder.

    Returns
    -------
    float
        Hubble parameter in km/s/Mpc.
    """
    zstar, rstar = _camb_recombination_at_theta(
        theta, omch2, ombh2, omega_k, mnu, TCMB, nnu
    )
    z_star_array = _z_grid_to_zstar(zstar)
    de_density_factor = np.asarray(de_density(z_star_array), dtype=float)

    def theta_residual(H0):
        distance = _comoving_distance_to_zstar(
            H0,
            z_star_array,
            de_density_factor,
            omch2,
            ombh2,
            mnu,
            omega_k,
            TCMB,
            nnu,
        )
        return rstar / distance - theta

    return brentq(theta_residual, h0_bracket[0], h0_bracket[1])


def resolve_background_densities(
    de_density,
    H0=None,
    Omega_b0=None,
    Omega_cdm0=None,
    Omega_k0=0.0,
    mnu=0.06,
    nnu=3.044,
    Omch2=None,
    Ombh2=None,
    cosmomc_theta=None,
):
    """Hubble parameter and density parameters from either small or big omegas.

    With ``Omch2`` and ``Ombh2``, ``cosmomc_theta`` (CosmoMC's ``100 theta_*``)
    is converted to ``H0`` through ``de_density``. Otherwise ``H0`` is used as
    given. With ``Omega_b0`` and ``Omega_cdm0``, ``H0`` is required. Small
    omegas take precedence when both sets are supplied.

    Returns
    -------
    dict
        ``H0``, ``h``, ``Omega_b0``, ``Omega_cdm0``, ``Omega_m0``, ``Omega_k0``,
        ``mnu``, ``nnu``.
    """
    if Omch2 is not None and Ombh2 is not None:
        if cosmomc_theta is not None:
            H0 = theta_to_H0(
                cosmomc_theta / 100.0,
                Omch2,
                Ombh2,
                de_density,
                mnu=mnu,
                nnu=nnu,
                omega_k=Omega_k0,
            )
        elif H0 is None:
            raise ValueError(
                "H0 or cosmomc_theta must be provided with Omch2 and Ombh2."
            )
        h = H0 / 100.0
        Omega_b0 = Ombh2 / h**2
        Omega_cdm0 = Omch2 / h**2
    elif Omega_b0 is not None and Omega_cdm0 is not None:
        if H0 is None:
            raise ValueError(
                "H0 must be provided if Omega_b0 and Omega_cdm0 are provided."
            )
        h = H0 / 100.0
    else:
        raise ValueError(
            "Either Omch2 and Ombh2, or Omega_b0 and Omega_cdm0, must be provided."
        )
    return {
        "H0": H0,
        "h": h,
        "Omega_b0": Omega_b0,
        "Omega_cdm0": Omega_cdm0,
        "Omega_m0": Omega_cdm0 + Omega_b0 + mnu / 93.14 / h**2,
        "Omega_k0": Omega_k0,
        "mnu": mnu,
        "nnu": nnu,
    }
