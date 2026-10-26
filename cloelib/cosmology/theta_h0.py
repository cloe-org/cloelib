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
    # omega_k is Omega_k, the curvature density parameter, not Omega_k h^2.
    omega_de = 1.0 - omega_b - omega_c - omega_nu - omega_r - omega_k
    nu_matter = _neutrino_matter_fraction(z_star_array, mnu)
    omega_m = omega_b + omega_c + omega_nu * nu_matter
    radicand = (
        omega_r * (1.0 + z_star_array) ** 4
        + omega_m * (1.0 + z_star_array) ** 3
        + omega_k * (1.0 + z_star_array) ** 2
        + omega_de * de_density_factor
    )
    # A negative or non-finite E^2 means the universe is not expanding on this
    # grid. Returning NaN here lets a root finder wander; fail instead.
    if not np.isfinite(radicand).all() or np.any(radicand <= 0.0):
        raise ValueError("expansion history is not finite and expanding")
    E_grid = np.sqrt(radicand)
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

    Raises
    ------
    ValueError
        If the inputs, the dark-energy density, or the expansion history cannot
        produce a root inside ``h0_bracket``. Callers should treat that as an
        invalid sample. A non-finite dark-energy density is rejected before
        CAMB. A history that does not cross the target inside ``h0_bracket``
        raises before the root finder iterates.
    """
    if not np.isfinite([theta, omch2, ombh2, mnu, omega_k, TCMB, nnu]).all():
        raise ValueError("non-finite cosmological input")
    # CosmoMC theta is ~0.01. Values outside this range are prior tails, and
    # CAMB's own search does not accept them.
    if not (0.001 < theta < 0.1):
        raise ValueError("theta is outside the range CAMB can solve")
    if not (0.0 < omch2 < 0.5 and 0.005 < ombh2 < 0.1):
        raise ValueError("omega_c or omega_b is outside the solvable range")

    # f_DE does not depend on H0. Reject a broken history before the CAMB call.
    z_check = _z_grid_to_zstar(1200.0)
    de_check = np.asarray(de_density(z_check), dtype=float)
    if de_check.shape != z_check.shape or not np.isfinite(de_check).all() or np.any(de_check <= 0.0):
        raise ValueError("dark-energy density is not finite and positive")

    zstar, rstar = _camb_recombination_at_theta(
        theta, omch2, ombh2, omega_k, mnu, TCMB, nnu
    )
    z_star_array = _z_grid_to_zstar(zstar)
    de_density_factor = np.asarray(de_density(z_star_array), dtype=float)
    if (
        de_density_factor.shape != z_star_array.shape
        or not np.isfinite(de_density_factor).all()
        or np.any(de_density_factor <= 0.0)
    ):
        raise ValueError("dark-energy density is not finite and positive")

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
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("comoving distance to last scattering is not finite")
        return rstar / distance - theta

    lo, hi = float(h0_bracket[0]), float(h0_bracket[1])
    try:
        residual_lo = theta_residual(lo)
        residual_hi = theta_residual(hi)
    except ValueError:
        raise
    if not (np.isfinite(residual_lo) and np.isfinite(residual_hi)) or residual_lo * residual_hi > 0.0:
        raise ValueError(
            f"no H0 root inside ({lo}, {hi}); residuals are {residual_lo}, {residual_hi}"
        )
    return float(brentq(theta_residual, lo, hi, maxiter=50))


