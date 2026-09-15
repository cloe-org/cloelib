"""Dark Emulator-based cosmology classes for cloelib.

This module provides Perturbations-protocol-compliant classes that use
Dark Emulator for computing:
- Nonlinear matter power spectrum P(k,z)
- Halo-matter and halo-halo correlation functions
- HOD-based galaxy statistics (ΔΣ, w_p, ξ_gg, ξ_gm)

Classes
-------
DarkEmuNonLinearPerturbations
    Base class for nonlinear P(k,z)
DarkEmuHaloPerturbations
    Adds halo statistics (ξ_hm, ξ_hh, HMF)
DarkEmuHODPerturbations
    Adds HOD-based real-space statistics (ΔΣ, w_p)

References
----------
- Nishimichi et al. (2019): Dark Quest. I. Fast and Accurate Emulation
- Miyatake et al. (2022): Cosmological inference from an emulator based
  halo model. II. Joint analysis of galaxy-galaxy weak lensing and
  galaxy clustering from HSC-Y1 and SDSS
"""

import warnings
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from scipy import interpolate

from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.darkemu_utils import (
    background_to_darkemu_cparam,
    validate_darkemu_cparam,
)

if TYPE_CHECKING:
    from cloelib.observables.darkemu_hod import DarkEmuHODParameters

# Dark Emulator imports with lazy loading pattern
_DARKEMU_AVAILABLE: bool = False
darkemu = None
model_hod = None

try:
    from dark_emulator import darkemu as _darkemu
    from dark_emulator import model_hod as _model_hod

    darkemu = _darkemu
    model_hod = _model_hod
    _DARKEMU_AVAILABLE = True
except ImportError:
    pass


def _check_darkemu_available() -> None:
    """Check if Dark Emulator is installed and raise ImportError if not."""
    if not _DARKEMU_AVAILABLE:
        raise ImportError(
            "dark_emulator package is not installed. "
            "Install it with: pip install dark_emulator"
        )


class DarkEmuNonLinearPerturbations:
    """Nonlinear perturbations using Dark Emulator.

    This class provides the Perturbations protocol interface using
    Dark Emulator's nonlinear matter power spectrum emulation.

    Parameters
    ----------
    background : Background
        cloelib Background instance with cosmological parameters.
    redshifts : np.ndarray
        Array of redshifts at which to compute perturbations.
    validate_params : bool, optional
        If True, validate that parameters are within Dark Emulator's
        training range. Default is True.

    Attributes
    ----------
    background : Background
        The background cosmology.
    z : np.ndarray
        Redshift array (sorted in ascending order).
    cparam : np.ndarray
        Dark Emulator format cosmological parameters.
    emu : darkemu.base_class
        Dark Emulator instance.

    Notes
    -----
    Dark Emulator constraints:
    - wa = 0 (constant w only)
    - Flat universe (Omega_k = 0)
    - Fixed neutrino mass (omega_nu = 0.00064)

    The emulator provides P_nl(k,z) accurate to ~2% for:
    - k < 10 h/Mpc
    - z < 1.5

    Examples
    --------
    >>> from cloelib.cosmology.class_cosmology import ClassBackground
    >>> params = {'H0': 67.0, 'Omega_cdm0': 0.25, 'Omega_b0': 0.05, ...}
    >>> bg = ClassBackground(params)
    >>> zs = np.linspace(0, 1.5, 10)
    >>> pert = DarkEmuNonLinearPerturbations(bg, zs)
    >>> ks = np.logspace(-2, 1, 100)
    >>> Pk = pert.matter_power_spectrum(zs, ks)
    """

    # Default k range for interpolation (h/Mpc)
    _K_MIN: float = 1e-4
    _K_MAX: float = 10.0
    _N_K: int = 200

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        validate_params: bool = True,
    ):
        _check_darkemu_available()

        self.background = background
        self.z = np.atleast_1d(redshifts)

        # Convert cosmology to Dark Emulator format (validates constraints internally)
        self.cparam = background_to_darkemu_cparam(background)

        if validate_params:
            validate_darkemu_cparam(self.cparam, strict=True)

        # Initialize Dark Emulator
        self.emu = darkemu.base_class()
        self.emu.set_cosmology(self.cparam)

        self._build_pk_interpolator()

    def _build_pk_interpolator(self) -> None:
        """Build 2D interpolator for P(k,z).

        Notes
        -----
        RectBivariateSpline requires:
        - Arrays must be in ascending order (monotonically increasing)
        - Pk_array shape must be (len(z), len(k))
        """
        self.k = np.logspace(np.log10(self._K_MIN), np.log10(self._K_MAX), self._N_K)

        # Sort z in ascending order (required by RectBivariateSpline)
        self.z = np.sort(self.z)

        # Ensure at least 2 redshift points for interpolation
        if len(self.z) == 1:
            z_extra = self.z[0] + 0.1 if self.z[0] < 1.4 else self.z[0] - 0.1
            self.z = np.sort([self.z[0], z_extra])

        # Compute P(k) at each redshift
        self.Pk = np.array([self.emu.get_pknl(self.k, z) for z in self.z])

        # Adjust spline order based on available points
        kx = min(3, len(self.z) - 1)
        ky = min(3, len(self.k) - 1)
        self.Pk_interp = interpolate.RectBivariateSpline(
            self.z, self.k, self.Pk, kx=kx, ky=ky
        )

    def _warn_if_extrapolating(self, zs: np.ndarray, ks: np.ndarray) -> None:
        """Warn if requested values are outside interpolation range."""
        k_min, k_max = self.k.min(), self.k.max()
        z_min, z_max = self.z.min(), self.z.max()

        if np.any(ks < k_min) or np.any(ks > k_max):
            warnings.warn(
                f"Requested k values [{ks.min():.2e}, {ks.max():.2e}] h/Mpc "
                f"outside interpolation range [{k_min:.2e}, {k_max:.2e}] h/Mpc.",
                UserWarning,
                stacklevel=3,
            )

        if np.any(zs < z_min) or np.any(zs > z_max):
            warnings.warn(
                f"Requested z values [{zs.min():.2f}, {zs.max():.2f}] "
                f"outside interpolation range [{z_min:.2f}, {z_max:.2f}].",
                UserWarning,
                stacklevel=3,
            )

    def matter_power_spectrum(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Compute the nonlinear matter power spectrum P(k,z).

        Parameters
        ----------
        zs : np.ndarray
            Redshifts at which to evaluate P(k).
        ks : np.ndarray
            Wavenumbers in h/Mpc.

        Returns
        -------
        Pk : np.ndarray
            Nonlinear matter power spectrum in (h^-1 Mpc)^3.
            Shape is (len(zs), len(ks)).
        """
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        self._warn_if_extrapolating(zs, ks)
        return self.Pk_interp(zs, ks)

    def growth_factor(
        self, zs: np.ndarray, ks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the linear growth factor D(z).

        Normalized to D(z=0) = 1.

        Parameters
        ----------
        zs : np.ndarray
            Redshifts at which to evaluate D(z).
        ks : np.ndarray, optional
            Wavenumbers (not used, included for interface compatibility).
            Growth factor is independent of k.

        Returns
        -------
        D : np.ndarray
            Linear growth factor normalized to unity at z=0.
        """
        zs = np.atleast_1d(zs)
        D = np.array([self.emu.Dgrowth_from_z(z) for z in zs])
        return D

    def growth_rate(
        self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the linear growth rate f(z) = d ln D / d ln a.

        Parameters
        ----------
        zs : np.ndarray, optional
            Redshifts at which to evaluate f(z). If None, uses self.z.
        ks : np.ndarray, optional
            Wavenumbers (not used, included for interface compatibility).

        Returns
        -------
        f : np.ndarray
            Linear growth rate.
        """
        if zs is None:
            zs = self.z
        zs = np.atleast_1d(zs)
        f = np.array([self.emu.f_from_z(z) for z in zs])
        return f

    def get_sigma8(self, z: float = 0.0) -> float:
        """Compute sigma8 at redshift z.

        Parameters
        ----------
        z : float, optional
            Redshift. Default is 0.

        Returns
        -------
        sigma8 : float
            RMS density fluctuation in 8 h^-1 Mpc spheres.
        """
        sigma8_0 = self.emu.get_sigma8()
        if z == 0.0:
            return sigma8_0
        return sigma8_0 * self.emu.Dgrowth_from_z(z)


class DarkEmuHaloPerturbations(DarkEmuNonLinearPerturbations):
    """Halo statistics using Dark Emulator.

    Extends DarkEmuNonLinearPerturbations with halo-matter and
    halo-halo correlation functions, halo mass function, and halo bias.

    Parameters
    ----------
    background : Background
        cloelib Background instance with cosmological parameters.
    redshifts : np.ndarray
        Array of redshifts at which to compute perturbations.
    validate_params : bool, optional
        If True, validate that parameters are within Dark Emulator's
        training range. Default is True.

    Examples
    --------
    >>> pert = DarkEmuHaloPerturbations(bg, zs)
    >>> r = np.logspace(-1, 2, 50)  # h^-1 Mpc
    >>> xi_hm = pert.halo_matter_correlation(r, z=0.5, M_min=1e13)
    """

    def halo_matter_correlation(
        self,
        r: np.ndarray,
        z: float,
        M_min: float,
    ) -> np.ndarray:
        """Compute halo-matter cross correlation function xi_hm(r).

        Parameters
        ----------
        r : np.ndarray
            Separations in h^-1 Mpc.
        z : float
            Redshift.
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.

        Returns
        -------
        xi_hm : np.ndarray
            Halo-matter cross correlation function.
        """
        return self.emu.get_xicross_massthreshold(r, M_min, z)

    def halo_halo_correlation(
        self,
        r: np.ndarray,
        z: float,
        M_min: float,
    ) -> np.ndarray:
        """Compute halo-halo auto correlation function xi_hh(r).

        Parameters
        ----------
        r : np.ndarray
            Separations in h^-1 Mpc.
        z : float
            Redshift.
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.

        Returns
        -------
        xi_hh : np.ndarray
            Halo-halo correlation function.
        """
        return self.emu.get_xiauto_massthreshold(r, M_min, z)

    def halo_mass_function(
        self,
        z: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the halo mass function.

        Returns the multiplicity function f(sigma) defined by:
        dn/dM = f(sigma) * rho_m / M * d ln sigma^-1 / dM

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        M : np.ndarray
            Halo masses M_200b in h^-1 M_sun.
        sigma : np.ndarray
            Mass variance sigma(M).
        f_sigma : np.ndarray
            Multiplicity function f(sigma).
        """
        return self.emu.get_f_HMF(z)

    def halo_bias(
        self,
        M_min: float,
        z: float,
    ) -> float:
        """Compute linear halo bias for mass threshold sample.

        Parameters
        ----------
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.
        z : float
            Redshift.

        Returns
        -------
        b : float
            Linear halo bias.
        """
        return self.emu.get_bias_massthreshold(M_min, z)

    def mass_to_density(
        self,
        M_min: float,
        z: float,
    ) -> float:
        """Convert halo mass threshold to cumulative number density.

        Parameters
        ----------
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.
        z : float
            Redshift.

        Returns
        -------
        n : float
            Cumulative number density in (h^-1 Mpc)^-3.
        """
        return self.emu.mass_to_dens(M_min, z)

    def density_to_mass(
        self,
        n: float,
        z: float,
    ) -> float:
        """Convert cumulative number density to halo mass threshold.

        Parameters
        ----------
        n : float
            Cumulative number density in (h^-1 Mpc)^-3.
        z : float
            Redshift.

        Returns
        -------
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.
        """
        return self.emu.dens_to_mass(n, z)

    def delta_sigma_halo(
        self,
        R: np.ndarray,
        z: float,
        M_min: float,
    ) -> np.ndarray:
        """Compute excess surface mass density ΔΣ for mass threshold halos.

        Parameters
        ----------
        R : np.ndarray
            Projected radii in h^-1 Mpc.
        z : float
            Redshift.
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.

        Returns
        -------
        delta_sigma : np.ndarray
            Excess surface mass density in h M_sun pc^-2.
        """
        return self.emu.get_DeltaSigma_massthreshold(R, M_min, z)

    def projected_correlation_halo(
        self,
        R: np.ndarray,
        z: float,
        M_min: float,
        pimax: Optional[float] = None,
    ) -> np.ndarray:
        """Compute projected halo-halo correlation function w_p(R).

        Parameters
        ----------
        R : np.ndarray
            Projected radii in h^-1 Mpc.
        z : float
            Redshift.
        M_min : float
            Minimum halo mass threshold in h^-1 M_sun.
        pimax : float, optional
            Line-of-sight integration limit in h^-1 Mpc.
            If None, integrates to infinity.

        Returns
        -------
        wp : np.ndarray
            Projected correlation function in h^-1 Mpc.
        """
        if pimax is None:
            return self.emu.get_wauto_massthreshold(R, M_min, z)
        else:
            return self.emu.get_wauto_masthreshold_cut(R, M_min, z, pimax)


class DarkEmuHODPerturbations(DarkEmuHaloPerturbations):
    """HOD-based galaxy statistics using Dark Emulator.

    Extends DarkEmuHaloPerturbations with Halo Occupation Distribution (HOD)
    modeling for galaxy-galaxy lensing (ΔΣ) and galaxy clustering (w_p, ξ_gg).

    Parameters
    ----------
    background : Background
        cloelib Background instance with cosmological parameters.
    redshifts : np.ndarray
        Array of redshifts at which to compute perturbations.
    hod_params : DarkEmuHODParameters, optional
        HOD model parameters. Can be set later with set_hod().
    validate_params : bool, optional
        If True, validate cosmological parameters. Default is True.

    Attributes
    ----------
    hod : model_hod.darkemu_x_hod
        Dark Emulator HOD model instance.
    hod_params : DarkEmuHODParameters
        Current HOD parameters.

    Notes
    -----
    HOD model follows Zheng et al. (2007) with extensions:
    - Central galaxy occupation: N_c = 0.5 * erfc((logM - logMmin) / sigma)
    - Satellite galaxy occupation: N_s = ((M - kappa*M_min) / M_1)^alpha
    - Off-centering: fraction p_off with scale R_off
    - Incompleteness: parameters alpha_inc, logM_inc (More et al. 2015)

    Examples
    --------
    >>> from cloelib.observables.darkemu_hod import DarkEmuHODParameters
    >>> hod_params = DarkEmuHODParameters(
    ...     logMmin=13.13, sigma_sq=0.22, logM1=14.21,
    ...     alpha=1.13, kappa=1.25
    ... )
    >>> pert = DarkEmuHODPerturbations(bg, zs, hod_params=hod_params)
    >>> R = np.logspace(-1, 2, 50)
    >>> ds = pert.delta_sigma(R, z=0.55)
    >>> wp = pert.projected_correlation(R, z=0.55)
    """

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        hod_params: Optional["DarkEmuHODParameters"] = None,
        validate_params: bool = True,
    ):
        super().__init__(background, redshifts, validate_params)

        # Initialize HOD model
        self.hod = model_hod.darkemu_x_hod()
        self.hod.set_cosmology(self.cparam)

        self._hod_params = None
        if hod_params is not None:
            self.set_hod(hod_params)

    def set_hod(self, hod_params: "DarkEmuHODParameters") -> None:
        """Set HOD model parameters.

        Parameters
        ----------
        hod_params : DarkEmuHODParameters
            HOD model parameters.
        """
        self._hod_params = hod_params
        gparam = hod_params.to_darkemu_dict()
        self.hod.set_galaxy(gparam)

    @property
    def hod_params(self) -> Optional["DarkEmuHODParameters"]:
        """Current HOD parameters."""
        return self._hod_params

    def _check_hod_set(self):
        """Check that HOD parameters have been set."""
        if self._hod_params is None:
            raise RuntimeError(
                "HOD parameters not set. Call set_hod() first or "
                "pass hod_params to constructor."
            )

    def delta_sigma(self, R: np.ndarray, z: float) -> np.ndarray:
        """Compute galaxy-galaxy lensing signal ΔΣ(R).

        Parameters
        ----------
        R : np.ndarray
            Projected radii in h^-1 Mpc.
        z : float
            Lens redshift.

        Returns
        -------
        delta_sigma : np.ndarray
            Excess surface mass density in h M_sun pc^-2.

        Notes
        -----
        Includes contributions from:
        - Central galaxies (including off-centering)
        - Satellite galaxies
        """
        self._check_hod_set()
        return self.hod.get_ds(R, z)

    def delta_sigma_components(self, R: np.ndarray, z: float) -> dict:
        """Compute ΔΣ broken down by component.

        Parameters
        ----------
        R : np.ndarray
            Projected radii in h^-1 Mpc.
        z : float
            Lens redshift.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'total': Total ΔΣ
            - 'central': Central galaxy contribution
            - 'central_off': Central with off-centering
            - 'satellite': Satellite contribution
        """
        self._check_hod_set()
        return {
            "total": self.hod.get_ds(R, z),
            "central": self.hod.get_ds_cen(R, z),
            "central_off": self.hod.get_ds_cen_off(R, z),
            "satellite": self.hod.get_ds_sat(R, z),
        }

    def projected_correlation(
        self,
        R: np.ndarray,
        z: float,
        pimax: Optional[float] = None,
    ) -> np.ndarray:
        """Compute projected galaxy correlation function w_p(R).

        Parameters
        ----------
        R : np.ndarray
            Projected radii in h^-1 Mpc.
        z : float
            Redshift.
        pimax : float, optional
            Line-of-sight integration limit in h^-1 Mpc.
            If None, integrates to infinity.

        Returns
        -------
        wp : np.ndarray
            Projected correlation function in h^-1 Mpc.
        """
        self._check_hod_set()
        if pimax is None:
            return self.hod.get_wp(R, z)
        else:
            return self.hod.get_wp(R, z, pimax)

    def projected_correlation_components(self, R: np.ndarray, z: float) -> dict:
        """Compute w_p broken down by halo term.

        Parameters
        ----------
        R : np.ndarray
            Projected radii in h^-1 Mpc.
        z : float
            Redshift.

        Returns
        -------
        dict
            Dictionary with keys for each term:
            - 'total': Total w_p
            - '1h_cs': 1-halo central-satellite
            - '1h_ss': 1-halo satellite-satellite
            - '2h_cc': 2-halo central-central
            - '2h_cs': 2-halo central-satellite
            - '2h_ss': 2-halo satellite-satellite
        """
        self._check_hod_set()
        return {
            "total": self.hod.get_wp(R, z),
            "1h_cs": self.hod.get_wp_1hcs(R, z),
            "1h_ss": self.hod.get_wp_1hss(R, z),
            "2h_cc": self.hod.get_wp_2hcc(R, z),
            "2h_cs": self.hod.get_wp_2hcs(R, z),
            "2h_ss": self.hod.get_wp_2hss(R, z),
        }

    def galaxy_galaxy_correlation(self, r: np.ndarray, z: float) -> np.ndarray:
        """Compute 3D galaxy-galaxy correlation function ξ_gg(r).

        Parameters
        ----------
        r : np.ndarray
            3D separations in h^-1 Mpc.
        z : float
            Redshift.

        Returns
        -------
        xi_gg : np.ndarray
            Galaxy auto-correlation function.
        """
        self._check_hod_set()
        return self.hod.get_xi_gg(r, z)

    def galaxy_matter_correlation(self, r: np.ndarray, z: float) -> np.ndarray:
        """Compute galaxy-matter cross correlation function ξ_gm(r).

        Parameters
        ----------
        r : np.ndarray
            3D separations in h^-1 Mpc.
        z : float
            Redshift.

        Returns
        -------
        xi_gm : np.ndarray
            Galaxy-matter cross correlation function.
        """
        self._check_hod_set()
        return self.hod.get_xi_gm(r, z)

    def galaxy_matter_correlation_components(self, r: np.ndarray, z: float) -> dict:
        """Compute ξ_gm broken down by component.

        Parameters
        ----------
        r : np.ndarray
            3D separations in h^-1 Mpc.
        z : float
            Redshift.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'total': Total ξ_gm
            - 'central': Central galaxy contribution
            - 'central_off': Central with off-centering
            - 'satellite': Satellite contribution
        """
        self._check_hod_set()
        return {
            "total": self.hod.get_xi_gm(r, z),
            "central": self.hod.get_xi_gm_cen(r, z),
            "central_off": self.hod.get_xi_gm_cen_off(r, z),
            "satellite": self.hod.get_xi_gm_sat(r, z),
        }
