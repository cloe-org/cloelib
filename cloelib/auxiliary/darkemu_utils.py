"""Utility functions for Dark Emulator integration with cloelib.

This module provides conversion functions between cloelib's Background cosmology
and Dark Emulator's cosmological parameter format (cparam).

Dark Emulator cparam format: (omega_b, omega_c, Omega_de, ln(10^10 As), ns, w)
where:
    - omega_b = Omega_b0 * h^2 (physical baryon density)
    - omega_c = Omega_cdm0 * h^2 (physical cold dark matter density)
    - Omega_de = 1 - Omega_m (dark energy density parameter, flat universe)
    - ln(10^10 As) = natural log of 10^10 times the scalar amplitude
    - ns = scalar spectral index
    - w = dark energy equation of state (constant, wa must be 0)

References
----------
- Nishimichi et al. (2019): Dark Quest. I. Fast and Accurate Emulation
- Miyatake et al. (2022): Cosmological inference from an emulator based
  halo model. II. Joint analysis of galaxy-galaxy weak lensing and
  galaxy clustering from HSC-Y1 and SDSS
"""

import warnings
import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from cloelib.cosmology.cosmology import Background

# Dark Emulator fixed neutrino density parameter
# Corresponds to sum(m_nu) = 0.06 eV (normal mass hierarchy lower bound)
DARKEMU_OMEGA_NU: float = 0.00064

# Parameter names in cparam array order
DARKEMU_PARAM_NAMES: List[str] = [
    "omega_b",
    "omega_c",
    "Omega_de",
    "ln10^10As",
    "ns",
    "w",
]

# Dark Emulator parameter ranges (from darkemu/cosmo_util.py)
# These are the training ranges for the emulator
DARKEMU_PARAM_RANGES = {
    "omega_b": (0.0211375, 0.0233625),
    "omega_c": (0.10782, 0.13178),
    "Omega_de": (0.54752, 0.82128),
    "ln10^10As": (2.4752, 3.7128),
    "ns": (0.916275, 1.012725),
    "w": (-1.2, -0.8),
}

# Extended ranges for linear power spectrum emulator
DARKEMU_PARAM_RANGES_LINEAR = {
    "omega_b": (0.02025, 0.02425),
    "omega_c": (0.0998, 0.1398),
    "Omega_de": (0.4594, 0.9094),
    "ln10^10As": None,  # No constraint
    "ns": (0.8645, 1.0645),
    "w": None,  # No constraint
}

# Track if neutrino warning has been shown (class-level to avoid spam in MCMC)
_neutrino_warning_shown = False


def _validate_darkemu_constraints(background: "Background") -> None:
    """Validate that background cosmology satisfies Dark Emulator constraints.

    Parameters
    ----------
    background : Background
        cloelib Background instance.

    Raises
    ------
    ValueError
        If wa != 0 or Omega_k0 != 0.
    """
    if not np.isclose(background.wa, 0.0, atol=1e-10):
        raise ValueError(
            f"Dark Emulator only supports constant w (wCDM) cosmologies. "
            f"Got wa = {background.wa}, but wa must be 0."
        )
    if not np.isclose(background.Omega_k0, 0.0, atol=1e-10):
        raise ValueError(
            f"Dark Emulator only supports flat universes (Omega_k = 0). "
            f"Got Omega_k0 = {background.Omega_k0}."
        )


def _check_neutrino_density(
    background: "Background", h2: float, neutrino_check: str
) -> None:
    """Check neutrino density consistency with Dark Emulator's fixed value.

    Parameters
    ----------
    background : Background
        cloelib Background instance.
    h2 : float
        Squared dimensionless Hubble parameter.
    neutrino_check : str
        How to handle mismatch: "warn", "raise", or "ignore".
    """
    global _neutrino_warning_shown

    if neutrino_check == "ignore":
        return

    omega_nu_input = 0.0
    if hasattr(background, "Omega_nu0") and background.Omega_nu0 is not None:
        omega_nu_input = background.Omega_nu0 * h2
    elif hasattr(background, "omega_nu") and background.omega_nu is not None:
        omega_nu_input = background.omega_nu

    if np.isclose(omega_nu_input, DARKEMU_OMEGA_NU, rtol=0.1):
        return

    msg = (
        f"Neutrino density mismatch: input omega_nu = {omega_nu_input:.6f}, "
        f"but Dark Emulator assumes fixed omega_nu = {DARKEMU_OMEGA_NU}. "
        f"Dark Emulator will use its internal fixed value regardless of input."
    )

    if neutrino_check == "raise":
        raise ValueError(msg)
    elif not _neutrino_warning_shown:
        warnings.warn(msg, UserWarning, stacklevel=3)
        _neutrino_warning_shown = True


def background_to_darkemu_cparam(
    background: "Background", neutrino_check: str = "warn"
) -> np.ndarray:
    """Convert cloelib Background cosmology to Dark Emulator cparam format.

    Parameters
    ----------
    background : Background
        cloelib Background instance containing cosmological parameters.
    neutrino_check : str, optional
        How to handle neutrino density mismatch with Dark Emulator's fixed value.
        - "warn": Issue a warning (default)
        - "raise": Raise ValueError
        - "ignore": Skip the check

    Returns
    -------
    cparam : np.ndarray
        Array of shape (6,) with Dark Emulator cosmological parameters:
        (omega_b, omega_c, Omega_de, ln(10^10 As), ns, w)

    Raises
    ------
    ValueError
        If wa != 0 (Dark Emulator only supports constant w models).
        If Omega_k0 != 0 (Dark Emulator only supports flat universes).
        If neutrino_check="raise" and omega_nu differs from Dark Emulator's fixed value.

    Notes
    -----
    Dark Emulator assumes:
    - Flat universe (Omega_k = 0)
    - Constant dark energy equation of state (wa = 0)
    - Fixed neutrino density: omega_nu = 0.00064 (corresponds to sum(mnu) = 0.06 eV)

    The Omega_de definition follows:
        Omega_de = 1 - (omega_b + omega_c + omega_nu) / h^2
    where omega_nu is fixed at 0.00064 internally by Dark Emulator.
    """
    _validate_darkemu_constraints(background)

    h2 = background.h**2
    omega_b = background.Omega_b0 * h2
    omega_c = background.Omega_cdm0 * h2

    _check_neutrino_density(background, h2, neutrino_check)

    Omega_de = 1.0 - (omega_b + omega_c + DARKEMU_OMEGA_NU) / h2
    ln_10_10_As = np.log(background.As * 1e10)

    return np.array(
        [omega_b, omega_c, Omega_de, ln_10_10_As, background.ns, background.w0]
    )


def validate_darkemu_cparam(
    cparam: np.ndarray, use_linear_range: bool = False, strict: bool = True
) -> bool:
    """Validate that cosmological parameters are within Dark Emulator's training range.

    Parameters
    ----------
    cparam : np.ndarray
        Array of shape (6,) with Dark Emulator cosmological parameters.
    use_linear_range : bool, optional
        If True, use the extended parameter range for linear power spectrum.
        Default is False (use the stricter nonlinear range).
    strict : bool, optional
        If True, raise ValueError when parameters are out of range.
        If False, return False instead. Default is True.

    Returns
    -------
    bool
        True if all parameters are within range.

    Raises
    ------
    ValueError
        If strict=True and any parameter is out of range.
    """
    param_ranges = (
        DARKEMU_PARAM_RANGES_LINEAR if use_linear_range else DARKEMU_PARAM_RANGES
    )
    out_of_range = []

    for i, name in enumerate(DARKEMU_PARAM_NAMES):
        bounds = param_ranges.get(name)
        if bounds is None:
            continue
        value = cparam[i]
        if not (bounds[0] <= value <= bounds[1]):
            out_of_range.append(
                f"{name} = {value:.6f} (valid range: [{bounds[0]:.6f}, {bounds[1]:.6f}])"
            )

    if out_of_range:
        if strict:
            raise ValueError(
                "Cosmological parameters out of Dark Emulator training range:\n"
                + "\n".join(f"  - {msg}" for msg in out_of_range)
            )
        return False

    return True


def darkemu_cparam_to_dict(cparam: np.ndarray) -> dict:
    """Convert Dark Emulator cparam array to a dictionary.

    Parameters
    ----------
    cparam : np.ndarray
        Array of shape (6,) with Dark Emulator cosmological parameters.

    Returns
    -------
    dict
        Dictionary with parameter names as keys.
    """
    return dict(zip(DARKEMU_PARAM_NAMES, cparam))


def get_darkemu_derived_params(cparam: np.ndarray) -> dict:
    """Calculate derived cosmological parameters from Dark Emulator cparam.

    Parameters
    ----------
    cparam : np.ndarray
        Array of shape (6,) with Dark Emulator cosmological parameters.

    Returns
    -------
    dict
        Dictionary containing derived parameters:
        - h: dimensionless Hubble parameter
        - Omega_m: total matter density parameter
        - As: scalar amplitude
        - H0: Hubble constant in km/s/Mpc
    """
    omega_b, omega_c, Omega_de, ln10_10_As, ns, w = cparam

    # Dark Emulator fixes omega_nu = 0.00064
    omega_m = omega_b + omega_c + DARKEMU_OMEGA_NU

    # Omega_m = 1 - Omega_de (flat universe)
    Omega_m = 1.0 - Omega_de

    # h = sqrt(omega_m / Omega_m)
    h = np.sqrt(omega_m / Omega_m)

    # As from ln(10^10 As)
    As = np.exp(ln10_10_As) / 1e10

    return {
        "h": h,
        "H0": 100.0 * h,
        "Omega_m": Omega_m,
        "Omega_b": omega_b / h**2,
        "Omega_c": omega_c / h**2,
        "As": As,
        "omega_b": omega_b,
        "omega_c": omega_c,
        "omega_nu": DARKEMU_OMEGA_NU,
    }
