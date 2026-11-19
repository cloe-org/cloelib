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


def _zstar_at_H0(
    H0, omch2, ombh2, omega_k=0.0, mnu=0.06, TCMB=2.7255, nnu=3.044, num_massive_neutrinos=1
):
    """Last-scattering redshift at fixed H0.

    Late-time dark energy barely moves recombination, so one LCDM call is enough
    for a later root find in ``w``, just as ``theta_to_H0`` freezes ``z*``
    before it solves for H0.
    """
    original_feedback_level = camb.config.FeedbackLevel
    try:
        camb.set_feedback_level(0)
        parameters = camb.CAMBparams()
        parameters.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="ppf")
        parameters.set_cosmology(
            ombh2=ombh2,
            omch2=omch2,
            omk=omega_k,
            mnu=mnu,
            H0=H0,
            TCMB=TCMB,
            nnu=nnu,
            num_massive_neutrinos=int(num_massive_neutrinos),
        )
        zstar = camb.get_background(parameters).get_derived_params()["zstar"]
    finally:
        camb.config.FeedbackLevel = original_feedback_level
    return float(zstar)


def _distance_terms(H0, z_grid, omch2, ombh2, mnu=0.06, omega_k=0.0, TCMB=2.7255, nnu=3.044):
    """Parts of ``E^2(z)`` that do not depend on the dark-energy density."""
    h = H0 / 100.0
    H0mpc = h / C_KM_S_MPC
    omnuh2 = mnu / 93.14
    omega_b = ombh2 / h**2
    omega_c = omch2 / h**2
    omega_nu = omnuh2 / h**2
    omega_r = omega_r_h2(TCMB, nnu) / h**2
    # omega_k is Omega_k, the curvature density parameter, not Omega_k h^2.
    omega_de = 1.0 - omega_b - omega_c - omega_nu - omega_r - omega_k
    nu_matter = _neutrino_matter_fraction(z_grid, mnu)
    omega_m = omega_b + omega_c + omega_nu * nu_matter
    matter_radiation = (
        omega_r * (1.0 + z_grid) ** 4
        + omega_m * (1.0 + z_grid) ** 3
        + omega_k * (1.0 + z_grid) ** 2
    )
    return H0mpc, omega_de, matter_radiation


def _chi_on_grid(z_grid, de_density_factor, H0mpc, omega_de, matter_radiation):
    """Comoving radial distance in Mpc on ``z_grid``."""
    radicand = matter_radiation + omega_de * np.asarray(de_density_factor, dtype=float)
    # A negative or non-finite E^2 means the universe is not expanding on this
    # grid. Returning NaN here lets a root finder wander; fail instead.
    if not np.isfinite(radicand).all() or np.any(radicand <= 0.0):
        raise ValueError("expansion history is not finite and expanding")
    E_grid = np.sqrt(radicand)
    r_dimless = cumulative_trapezoid(1.0 / E_grid, z_grid, initial=0.0)
    return r_dimless / H0mpc


def _comoving_distance_to_zstar(
    H0, z_star_array, de_density_factor, omch2, ombh2, mnu=0.06, omega_k=0.0, TCMB=2.7255, nnu=3.044
):
    H0mpc, omega_de, matter_radiation = _distance_terms(
        H0, z_star_array, omch2, ombh2, mnu, omega_k, TCMB, nnu
    )
    return _chi_on_grid(
        z_star_array, de_density_factor, H0mpc, omega_de, matter_radiation
    )[-1]


def equivalent_constant_w(
    H0,
    omch2,
    ombh2,
    redshifts,
    de_density,
    omega_k=0.0,
    mnu=0.06,
    TCMB=2.7255,
    nnu=3.044,
    num_massive_neutrinos=1,
    w_bracket=(-3.0, 0.0),
):
    """Constant ``w`` matching ``chi(z*) - chi(z)`` of ``de_density`` at each redshift.

    This is the same distance integral as ``theta_to_H0``. There H0 is the
    unknown and ``f_DE(z)`` is fixed. Here H0 is fixed and each sample
    redshift gets the constant ``w`` whose comoving distance to last scattering
    equals that of ``de_density``. ``de_density(z)`` returns ``f_DE(z)`` and
    must not depend on ``w``. ``z*`` comes from one CAMB call.

    Returns
    -------
    np.ndarray
        ``w`` at each redshift.

    Raises
    ------
    ValueError
        If ``f_DE`` is not finite and positive, or if no root lies inside
        ``w_bracket``.
    """
    redshifts = np.atleast_1d(np.asarray(redshifts, dtype=float))
    if redshifts.size == 0:
        return np.empty(0, dtype=float)
    if not np.isfinite(redshifts).all() or np.any(redshifts < 0.0):
        raise ValueError("redshift is not finite and non-negative")

    zstar = _zstar_at_H0(
        H0,
        omch2,
        ombh2,
        omega_k,
        mnu,
        TCMB,
        nnu,
        num_massive_neutrinos,
    )
    if np.any(redshifts > zstar):
        raise ValueError("redshift is above z*")
    z_grid = np.unique(np.concatenate((_z_grid_to_zstar(zstar), redshifts)))
    de_factor = np.asarray(de_density(z_grid), dtype=float)
    if de_factor.shape != z_grid.shape or not np.isfinite(de_factor).all() or np.any(de_factor <= 0.0):
        raise ValueError("dark-energy density is not finite and positive")

    H0mpc, omega_de, matter_radiation = _distance_terms(
        H0, z_grid, omch2, ombh2, mnu, omega_k, TCMB, nnu
    )
    chi_target = _chi_on_grid(z_grid, de_factor, H0mpc, omega_de, matter_radiation)
    nodes = np.searchsorted(z_grid, redshifts)
    distance_target = chi_target[-1] - chi_target[nodes]

    def distance_of_w(w):
        # wCDM, wa = 0: f_DE(z) = (1+z)^{3(1+w)}, equal to 1 today.
        de_w = (1.0 + z_grid) ** (3.0 * (1.0 + w))
        chi = _chi_on_grid(z_grid, de_w, H0mpc, omega_de, matter_radiation)
        return chi[-1] - chi[nodes]

    w_lo, w_hi = float(w_bracket[0]), float(w_bracket[1])
    distance_lo = distance_of_w(w_lo)
    distance_hi = distance_of_w(w_hi)
    residual_lo = distance_lo - distance_target
    residual_hi = distance_hi - distance_target

    def residual(w, index):
        return distance_of_w(w)[index] - distance_target[index]

    w_eq = np.empty(redshifts.size)
    for index in range(redshifts.size):
        left = residual_lo[index]
        right = residual_hi[index]
        if not (np.isfinite(left) and np.isfinite(right)) or left * right > 0.0:
            raise ValueError(
                "no constant-w root inside "
                f"({w_lo}, {w_hi}) at z={redshifts[index]}; residuals are {left}, {right}"
            )
        w_eq[index] = float(brentq(residual, w_lo, w_hi, args=(index,), maxiter=50))
    return w_eq


def equivalent_wcdm_growth(background, redshifts, de_density):
    """Constant ``w`` and its linear growth, matched at every sample redshift.

    ``de_density(z)`` is ``f_DE(z)`` of the model being matched. Distances use
    :func:`equivalent_constant_w`. The growth factor is ``D(a)`` of flat
    ``w``CDM (``wa = 0``) at ``a = 1/(1+z)``, from the same initial conditions
    as the binned and splined growth solvers.

    Returns
    -------
    tuple of np.ndarray
        ``w`` and ``D`` at each redshift.
    """
    redshifts = np.atleast_1d(np.asarray(redshifts, dtype=float))
    if redshifts.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    # Imported here so a distance-only caller does not load the growth solver.
    import cloelib.auxiliary.jaxmgrowth_de_descrete as mgrowth

    hubble_h = background.h
    w_eq = equivalent_constant_w(
        background.H0,
        background.Omega_cdm0 * hubble_h**2,
        background.Omega_b0 * hubble_h**2,
        redshifts,
        de_density,
        omega_k=background.Omega_k0,
        mnu=background.mnu,
        nnu=background.N_eff,
        num_massive_neutrinos=background.N_mnu,
    )
    omega_m = float(np.asarray(background.Omega_m(np.array([0.0]))).reshape(-1)[0])
    dz_wcdm = np.empty(redshifts.size)
    for index, (w_value, redshift) in enumerate(zip(w_eq, redshifts)):
        mg_cosmo = mgrowth.w0waCDM(
            {
                "Omega_m": omega_m,
                "h": hubble_h,
                "w0": float(w_value),
                "wa": 0.0,
                "a_arr": [1.0 / (1.0 + float(redshift))],
            }
        )
        growth, _ = mg_cosmo.growth_parameters()
        dz_wcdm[index] = float(np.asarray(growth).reshape(-1)[0])
    return w_eq, dz_wcdm


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


