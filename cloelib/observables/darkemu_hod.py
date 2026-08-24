"""Halo Occupation Distribution (HOD) model parameters for Dark Emulator.

This module provides the DarkEmuHODParameters dataclass for configuring HOD models
compatible with Dark Emulator's model_hod module.

The HOD model follows Zheng et al. (2007) with extensions for:
- Off-centering (Hikage et al. 2013)
- Incompleteness (More et al. 2015)

References
----------
- Zheng et al. (2007): ApJ 667, 760 - HOD model formulation
- Hikage et al. (2013): MNRAS 435, 2345 - Off-centering
- More et al. (2015): ApJ 806, 2 - Incompleteness model
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import numpy as np


@dataclass
class DarkEmuHODParameters:
    """HOD model parameters for Dark Emulator.

    This dataclass encapsulates all parameters needed for the Halo Occupation
    Distribution model used by Dark Emulator's model_hod module.

    Parameters
    ----------
    logMmin : float
        log10 of minimum halo mass for central galaxies [h^-1 M_sun].
        Central occupation: N_c = 0.5 * erfc((logM - logMmin) / sqrt(2*sigma_sq))
    sigma_sq : float
        Variance of the central galaxy mass distribution (sigma^2).
        Controls the sharpness of the central occupation transition.
    logM1 : float
        log10 of characteristic satellite mass [h^-1 M_sun].
        Satellite occupation: N_s = ((M - kappa*M_min) / M_1)^alpha
    alpha : float
        Power-law slope for satellite occupation.
    kappa : float, optional
        Threshold mass parameter for satellites (default 0.0).
        Satellites only appear in halos with M > kappa * M_min.
    poff : float, optional
        Fraction of off-centered central galaxies (default 0.0).
        Range: [0, 1].
    Roff : float, optional
        Off-centering scale relative to R200m (default 0.0).
        The off-center radius is Roff * R200m.
    sat_dist_type : str, optional
        Satellite distribution type (default 'emulator').
        Options:
        - 'emulator': Use emulator-based satellite profile
        - 'NFW': Use NFW profile with Diemer & Kravtsov (2015) c-M relation
    alpha_inc : float, optional
        Incompleteness power-law index (default None = no incompleteness).
        Following More et al. (2015).
    logM_inc : float, optional
        Incompleteness characteristic mass [h^-1 M_sun] (default None).
        Required if alpha_inc is set.

    Attributes
    ----------
    Mmin : float
        Minimum halo mass in h^-1 M_sun (derived from logMmin).
    M1 : float
        Characteristic satellite mass in h^-1 M_sun (derived from logM1).
    M_inc : float or None
        Incompleteness mass in h^-1 M_sun (derived from logM_inc).

    Examples
    --------
    Basic HOD parameters (Zheng+07 style):

    >>> hod = DarkEmuHODParameters(
    ...     logMmin=13.13,
    ...     sigma_sq=0.22,
    ...     logM1=14.21,
    ...     alpha=1.13,
    ...     kappa=1.25
    ... )

    With off-centering:

    >>> hod = DarkEmuHODParameters(
    ...     logMmin=13.0,
    ...     sigma_sq=0.3,
    ...     logM1=14.0,
    ...     alpha=1.0,
    ...     poff=0.2,  # 20% off-centered
    ...     Roff=0.1   # at 0.1 * R200m
    ... )

    With incompleteness:

    >>> hod = DarkEmuHODParameters(
    ...     logMmin=13.0,
    ...     sigma_sq=0.3,
    ...     logM1=14.0,
    ...     alpha=1.0,
    ...     alpha_inc=0.44,
    ...     logM_inc=13.57
    ... )

    Notes
    -----
    The mean occupation numbers are:

    Central galaxies:
        <N_c(M)> = 0.5 * erfc((logM - logMmin) / sqrt(2 * sigma_sq))

    Satellite galaxies (for M > kappa * M_min):
        <N_s(M)> = <N_c(M)> * ((M - kappa * M_min) / M_1)^alpha

    With incompleteness (More et al. 2015), occupations are modified by:
        f_inc(M) = (M / M_inc)^alpha_inc / (1 + (M / M_inc)^alpha_inc)
    """

    # Required parameters
    logMmin: float
    sigma_sq: float
    logM1: float
    alpha: float

    # Optional parameters with defaults
    kappa: float = 0.0
    poff: float = 0.0
    Roff: float = 0.0
    sat_dist_type: Literal["emulator", "NFW"] = "emulator"
    alpha_inc: Optional[float] = None
    logM_inc: Optional[float] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()

    def _validate(self):
        """Validate HOD parameter values."""
        if self.sigma_sq < 0:
            raise ValueError(f"sigma_sq must be non-negative, got {self.sigma_sq}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        if not 0 <= self.poff <= 1:
            raise ValueError(f"poff must be in [0, 1], got {self.poff}")
        if self.Roff < 0:
            raise ValueError(f"Roff must be non-negative, got {self.Roff}")
        if self.sat_dist_type not in ("emulator", "NFW"):
            raise ValueError(
                f"sat_dist_type must be 'emulator' or 'NFW', got {self.sat_dist_type}"
            )
        # Check incompleteness parameter consistency
        if (self.alpha_inc is None) != (self.logM_inc is None):
            raise ValueError(
                "Both alpha_inc and logM_inc must be set together, or both None"
            )

    @property
    def Mmin(self) -> float:
        """Minimum halo mass in h^-1 M_sun."""
        return 10**self.logMmin

    @property
    def M1(self) -> float:
        """Characteristic satellite mass in h^-1 M_sun."""
        return 10**self.logM1

    @property
    def M_inc(self) -> Optional[float]:
        """Incompleteness characteristic mass in h^-1 M_sun."""
        if self.logM_inc is None:
            return None
        return 10**self.logM_inc

    def to_darkemu_dict(self) -> Dict[str, Any]:
        """Convert to Dark Emulator gparam dictionary format.

        Returns
        -------
        gparam : dict
            Dictionary compatible with darkemu model_hod.set_galaxy().

        Examples
        --------
        >>> hod = DarkEmuHODParameters(logMmin=13.0, sigma_sq=0.3, logM1=14.0, alpha=1.0)
        >>> gparam = hod.to_darkemu_dict()
        >>> hod_model.set_galaxy(gparam)
        """
        gparam = {
            "logMmin": self.logMmin,
            "sigma_sq": self.sigma_sq,
            "logM1": self.logM1,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "poff": self.poff,
            "Roff": self.Roff,
            "sat_dist_type": self.sat_dist_type,
            # Dark Emulator requires alpha_inc and logM_inc
            # Default: no incompleteness (alpha_inc=0 gives f_inc=1)
            "alpha_inc": self.alpha_inc if self.alpha_inc is not None else 0.0,
            "logM_inc": self.logM_inc if self.logM_inc is not None else 13.0,
        }
        return gparam

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "DarkEmuHODParameters":
        """Create DarkEmuHODParameters from a dictionary.

        Parameters
        ----------
        params : dict
            Dictionary with HOD parameter names as keys.
            Extra keys are ignored.

        Returns
        -------
        DarkEmuHODParameters
            New instance with parameters from dictionary.
        """
        # Filter to only valid parameter names
        valid_keys = {
            "logMmin",
            "sigma_sq",
            "logM1",
            "alpha",
            "kappa",
            "poff",
            "Roff",
            "sat_dist_type",
            "alpha_inc",
            "logM_inc",
        }
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization/sampling.

        Returns array of [logMmin, sigma_sq, logM1, alpha, kappa,
                         poff, Roff, alpha_inc, logM_inc]
        where alpha_inc and logM_inc are np.nan if not set.

        Returns
        -------
        np.ndarray
            Array of parameter values.
        """
        return np.array(
            [
                self.logMmin,
                self.sigma_sq,
                self.logM1,
                self.alpha,
                self.kappa,
                self.poff,
                self.Roff,
                self.alpha_inc if self.alpha_inc is not None else np.nan,
                self.logM_inc if self.logM_inc is not None else np.nan,
            ]
        )

    @classmethod
    def from_array(
        cls, arr: np.ndarray, sat_dist_type: str = "emulator"
    ) -> "DarkEmuHODParameters":
        """Create DarkEmuHODParameters from numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array of length 7 or 9:
            [logMmin, sigma_sq, logM1, alpha, kappa, poff, Roff]
            or [logMmin, sigma_sq, logM1, alpha, kappa, poff, Roff,
                alpha_inc, logM_inc]
        sat_dist_type : str, optional
            Satellite distribution type (default 'emulator').

        Returns
        -------
        DarkEmuHODParameters
            New instance with parameters from array.
        """
        if len(arr) == 7:
            return cls(
                logMmin=arr[0],
                sigma_sq=arr[1],
                logM1=arr[2],
                alpha=arr[3],
                kappa=arr[4],
                poff=arr[5],
                Roff=arr[6],
                sat_dist_type=sat_dist_type,
            )
        elif len(arr) == 9:
            alpha_inc = arr[7] if not np.isnan(arr[7]) else None
            logM_inc = arr[8] if not np.isnan(arr[8]) else None
            return cls(
                logMmin=arr[0],
                sigma_sq=arr[1],
                logM1=arr[2],
                alpha=arr[3],
                kappa=arr[4],
                poff=arr[5],
                Roff=arr[6],
                sat_dist_type=sat_dist_type,
                alpha_inc=alpha_inc,
                logM_inc=logM_inc,
            )
        else:
            raise ValueError(f"Array must have length 7 or 9, got {len(arr)}")

    def copy(self, **kwargs) -> "DarkEmuHODParameters":
        """Create a copy with optional parameter overrides.

        Parameters
        ----------
        **kwargs
            Parameter values to override.

        Returns
        -------
        DarkEmuHODParameters
            New instance with updated parameters.

        Examples
        --------
        >>> hod = DarkEmuHODParameters(logMmin=13.0, sigma_sq=0.3, logM1=14.0, alpha=1.0)
        >>> hod_modified = hod.copy(logMmin=13.5)
        """
        params = {
            "logMmin": self.logMmin,
            "sigma_sq": self.sigma_sq,
            "logM1": self.logM1,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "poff": self.poff,
            "Roff": self.Roff,
            "sat_dist_type": self.sat_dist_type,
            "alpha_inc": self.alpha_inc,
            "logM_inc": self.logM_inc,
        }
        params.update(kwargs)
        return DarkEmuHODParameters.from_dict(params)


# Common HOD parameter priors for sampling
HOD_PARAM_PRIORS = {
    "logMmin": {"min": 11.0, "max": 15.0, "latex": r"$\log M_{\min}$"},
    "sigma_sq": {"min": 0.0, "max": 1.0, "latex": r"$\sigma^2$"},
    "logM1": {"min": 12.0, "max": 16.0, "latex": r"$\log M_1$"},
    "alpha": {"min": 0.5, "max": 2.0, "latex": r"$\alpha$"},
    "kappa": {"min": 0.0, "max": 2.0, "latex": r"$\kappa$"},
    "poff": {"min": 0.0, "max": 0.5, "latex": r"$p_{\rm off}$"},
    "Roff": {"min": 0.0, "max": 0.5, "latex": r"$R_{\rm off}$"},
    "alpha_inc": {"min": 0.0, "max": 1.0, "latex": r"$\alpha_{\rm inc}$"},
    "logM_inc": {"min": 12.0, "max": 15.0, "latex": r"$\log M_{\rm inc}$"},
}
