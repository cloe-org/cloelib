"""Implementation of Background and Perturbation cosmology for binned equation of state of dark energy."""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.cosmology.cosmology import Perturbations

# General imports
import numpy as np
from typing import Optional
from scipy.integrate import quad, cumulative_trapezoid
from scipy import interpolate
import jax.numpy as jnp
import camb
from scipy.optimize import brentq


C_KM_S_MPC = 2997.92458

# Cosmology imports
try:
    import cloelib.auxiliary.jaxmgrowth_de_descrete as mgrowth
except ImportError:
    raise ImportError("MGrowth could not be imported or initialised.")

try:
    import HMcode2020Emu as hmcodeemu

    HM2020_emu = hmcodeemu.Matter_powerspectrum()
    redshift_max = HM2020_emu.emulator["linear"]["bounds"]["z"][1]
except ImportError:
    raise ImportError("HMcode2020emu could not be imported or initialised.")


def get_z_bin(z, zbin_edges):
    """
    Returns the bin index for a given redshift value.

    Parameters:
    -----------
    z : float or array-like
        Redshift value(s) to bin
    zbin_edges : array-like
        Array of bin edges (must be sorted in ascending order)

    Returns:
    --------
    bin_index : int or array
        Bin index (0-indexed). Returns -1 for values below the first edge
        and len(zbin_edges)-2 for values at or above the last edge.
    """
    # Check if input is scalar
    is_scalar = np.isscalar(z)

    # Convert to array for processing
    z = np.asarray(z)
    zbin_edges = np.asarray(zbin_edges)

    # np.digitize returns the index of the bin (1-indexed)
    # right=False means left edge inclusive, right edge exclusive
    bin_idx = np.digitize(z, zbin_edges, right=False) - 1

    # Handle edge cases:
    # - Values below first edge get -1
    # - Values at or above last edge get the last bin index
    bin_idx = np.clip(bin_idx, -1, len(zbin_edges) - 2)

    # For values exactly at the last edge, put them in the last bin
    if is_scalar:
        if z >= zbin_edges[-1]:
            bin_idx = len(zbin_edges) - 2
    else:
        bin_idx[z >= zbin_edges[-1]] = len(zbin_edges) - 2

    # Return scalar if input was scalar, otherwise return array
    return bin_idx.item() if is_scalar else bin_idx


def get_de_density(z, zbin_edges, zbin_widths, zbin_centers, w_i):
    """
    Optimized version using pre-computation and vectorization.
    """
    z = np.asarray(z)
    bins = get_z_bin(z, zbin_edges)

    # Pre-compute ratio terms for all bins (vectorized, done once)
    # These don't depend on z, so compute once and reuse
    ratios = (
        (1 + zbin_centers + (zbin_widths / 2)) / (1 + zbin_centers - (zbin_widths / 2))
    ) ** (3.0 * (1.0 + w_i))

    # Pre-compute cumulative products for all possible bin ranges
    # cumprod[i] = product of ratios[0] * ratios[1] * ... * ratios[i-1]
    cumprod = np.concatenate([[1.0], np.cumprod(ratios)])

    # Pre-compute denominator terms for all bins
    denom_terms = 1 + zbin_centers - zbin_widths / 2

    # Vectorized computation
    # For each z, get the product from cumulative products
    # Use np.maximum to handle negative bin indices
    bin_idx_safe = np.maximum(0, bins)
    products = cumprod[bin_idx_safe]

    # Compute final density vectorized
    rho = ((1 + z) / denom_terms[bin_idx_safe]) ** (
        3.0 * (1.0 + w_i[bin_idx_safe])
    ) * products

    return rho


def get_de_density_i(z, zbin_edges, zbin_widths, zbin_centers, w_i):
    bins = get_z_bin(z, zbin_edges)
    product = 1.0
    # Handle bins as scalar (get_z_bin returns scalar for scalar input)
    bin_idx = bins if isinstance(bins, (int, np.integer)) else bins[0]
    for j in range(bin_idx):
        product *= (
            (1 + zbin_centers[j] + (zbin_widths[j] / 2))
            / (1 + zbin_centers[j] - (zbin_widths[j] / 2))
        ) ** (3.0 * (1.0 + w_i[j]))
    rho = ((1 + z) / (1 + zbin_centers[bin_idx] - zbin_widths[bin_idx] / 2)) ** (
        3.0 * (1.0 + w_i[bin_idx])
    ) * product
    return rho

def _camb_recombination_at_theta(theta, omch2, ombh2, omega_k=0.0, mnu=0.06,
                                 TCMB=2.7255, nnu=3.044):
    """Single CAMB call: rstar and zstar depend only weakly on H0, so fix them here."""
    original_feedback_level = camb.config.FeedbackLevel
    try:
        camb.set_feedback_level(0)
        p = camb.CAMBparams()
        p.set_dark_energy(w=-1., wa=0., dark_energy_model="ppf")
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
    z_low = np.linspace(0.0, 3., n_low)
    z_high = np.logspace(np.log10(3.001), np.log10(zstar), n_high)
    return np.unique(np.concatenate((z_low, z_high)))


def omega_r_h2(TCMB=2.7255, nnu=3.044):
    """Photon + massless neutrino contribution to Omega_r h^2 (CAMB convention)."""
    ogamma = 2.4728e-5 * (TCMB / 2.7255) ** 4
    return ogamma * (1.0 + (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * nnu)

# Effective massive-neutrino matter fraction G(z, mnu): the WMAP7 radiation
# multiplier (Appendix C of arXiv:2502.07185) over-estimates E(z) at z ~ 1000
# and biases H0 from theta by ~0.6 km/s/Mpc.  CAMB's nu contribution to E^2 is
# well approximated as omnu * (1+z)^3 * G(z, mnu) with G -> 1 today and G -> 0
# at early times (Lesgourgues/Komatsu-style transition, fitted to CAMB).
_NU_MATTER_SCALE = 6.328
_NU_MATTER_POWER = 0.5

def _neutrino_matter_fraction(z, mnu):
    """Fraction of omnuh2 that contributes as non-relativistic matter at redshift z."""
    z = np.asarray(z, dtype=float)
    if mnu <= 0.0:
        return np.zeros_like(z)
    z_transition = _NU_MATTER_SCALE / mnu
    return 1.0 / (1.0 + ((1.0 + z) / z_transition) ** _NU_MATTER_POWER)


def _precompute_de_background(z_star_array, w_i, zbin_edges, zbin_widths, zbin_centers):
    """DE part of the Friedmann integral is independent of H0."""
    de_density_factor = get_de_density(z_star_array, zbin_edges, zbin_widths, zbin_centers, w_i)
    return de_density_factor

def _comoving_distance_analytic(H0, z_star_array, de_density_factor,
                                omch2, ombh2, mnu=0.06, omega_k=0.0,
                                TCMB=2.7255, nnu=3.044):
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
    return r_dimless[-1]/ H0mpc

def theta_to_H0(theta, omch2, ombh2,
                      w_i, zbin_edges, zbin_widths, zbin_centers,
                      omega_k=0.0, mnu=0.06, TCMB=2.7255, nnu=3.044, 
                      h0_bracket=(50.0, 90.0)):
    """Solve for H0 such that r_s / D_M(z_*) = theta using the binned-w background.

    Speed: rstar and zstar are fixed from one CAMB recombination call (they vary
    < 0.01% over the H0 bracket).  The binned-w dark-energy factor is precomputed once.
    Only the cheap distance integral runs inside brentq.
    """
    zstar, rstar = _camb_recombination_at_theta(
        theta, omch2, ombh2, omega_k, mnu, TCMB, nnu
    )
    z_star_array = _z_grid_to_zstar(zstar)
    de_density_factor = _precompute_de_background(z_star_array, w_i, zbin_edges, zbin_widths, zbin_centers)

    def theta_residual(H0):
        D_M = _comoving_distance_analytic(
            H0, z_star_array, de_density_factor,
            omch2, ombh2, mnu, omega_k, TCMB, nnu,
        )
        #print(H0, rstar / D_M - theta)
        return rstar / D_M - theta

    return brentq(theta_residual, h0_bracket[0], h0_bracket[1])


class DEBinnedEoSBackground:
    """Beyond w0wa-background cosmological calculations."""

    def __init__(
        self,
        cosmology_dict: dict,
        zbin_edges: np.ndarray,  
        w_i: np.ndarray,
    ) -> None:
        """
        Initialize the CAMBBackground instance with cosmological parameters.

        Args:
            H0 (float): Hubble parameter in [km/s/Mpc].
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0(float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (float): Total sum of neutrino mass in [eV].
            zbin_edges (np.ndarray): Array of bin edges for the redshift bins.
            w_i (np.ndarray): Array of equation of state parameters for the redshift bins.
        """
        self.zbin_edges = zbin_edges
        self.zbin_widths = np.diff(zbin_edges)
        self.zbin_centers = 0.5 * (zbin_edges[1:] + zbin_edges[:-1])
        self.w_i = w_i

        self.mnu = cosmology_dict['mnu'] if 'mnu' in cosmology_dict else 0.06
        self.nnu = cosmology_dict['nnu'] if 'nnu' in cosmology_dict else 3.044
        self.As = cosmology_dict['As']
        self.ns = cosmology_dict['ns']
        self.Omega_k0 = cosmology_dict['Omega_k0'] if 'Omega_k0' in cosmology_dict else 0.0
        # small omega's
        if 'Omch2' and 'Ombh2' in cosmology_dict:
            self.Omch2 = cosmology_dict['Omch2']
            self.Ombh2 = cosmology_dict['Ombh2']
            # 'cosmomc_theta' should be around ~1, the real theta_* is ~0.01
            if 'cosmomc_theta' in cosmology_dict:
                self.H0 = theta_to_H0(
                    cosmology_dict['cosmomc_theta']/100., self.Omch2, self.Ombh2, self.w_i, self.zbin_edges, self.zbin_widths, self.zbin_centers,
                    mnu=self.mnu, nnu=self.nnu, omega_k=self.Omega_k0
                )
            else:
                self.H0 = cosmology_dict['H0']
            self.h = self.H0 / 100

            self.Omega_b0 = self.Ombh2 / (self.h)**2
            self.Omega_cdm0 = self.Omch2 / (self.h)**2
            self.Omega_m0 = self.Omega_cdm0 + self.Omega_b0 + self.mnu / 93.14 / (self.h) ** 2
        elif 'Omega_b0' and 'Omega_cdm0' in cosmology_dict:
            self.Omega_b0 = cosmology_dict['Omega_b0']
            self.Omega_cdm0 = cosmology_dict['Omega_cdm0']
            if 'H0' in cosmology_dict:
                self.H0 = cosmology_dict['H0']
            else:
                raise ValueError("H0 must be provided if Omega_b0 and Omega_cdm0 are provided.")
            self.h = self.H0 / 100
            self.Omega_m0 = self.Omega_cdm0 + self.Omega_b0 + self.mnu / 93.14 / (self.h) ** 2
        else:
            raise ValueError("Either small omega's or Omega_b0 and Omega_cdm0 must be provided.")

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Return the Hubble parameter as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.
            units (str): Units for the Hubble parameter ('1/Mpc' or 'km/s/Mpc').

        Returns:
            np.ndarray: Hubble parameter values at specified redshifts.
        """
        DE_z_grid = get_de_density(
            zs, self.zbin_edges, self.zbin_widths, self.zbin_centers, self.w_i
        )
        E_z_grid = np.sqrt(
            self.Omega_m0 * pow(1.0 + zs, 3) + (1 - self.Omega_m0) * DE_z_grid
        )
        if units == "1/Mpc":
            return E_z_grid * self.H0 / SPEED_OF_LIGHT * 1000
        if units == "km/s/Mpc":
            return E_z_grid * self.H0
        raise ValueError(
            "Unsupported units for hubble_parameter. Choose '1/Mpc' or 'km/s/Mpc'."
        )

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the comoving distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Comoving distance values.
        """

        def r_z_int(z):
            return 1.0 / np.sqrt(
                self.Omega_m0 * pow(1.0 + z, 3)
                + (1 - self.Omega_m0)
                * get_de_density_i(
                    z, self.zbin_edges, self.zbin_widths, self.zbin_centers, self.w_i
                )
            )

        zz_ = np.hstack(([0.0], zs)) if zs[0] != 0.0 else zs
        r_z_grid = (
            np.array(
                [
                    quad(r_z_int, zz_[i], zz_[i + 1])[0] / self.H0
                    for i in range(len(zz_) - 1)
                ]
            )
            * SPEED_OF_LIGHT
            / 1000
        )  # Mpc
        r_z_total = np.cumsum(r_z_grid)
        return r_z_total

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the transverse comoving distance between two redshifts.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Transverse comoving distance values.
        """
        x = self.comoving_distance(zs)

        if self.Omega_k0 == 0.0:
            y = x
            return y
        else:
            raise ValueError(
                "Omega_k0 != 0.0 not supported for DEBackground.transverse_comoving_distance."
            )

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the angular diameter distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Angular diameter distance values.
        """
        return self.transverse_comoving_distance(zs) / (1 + zs)

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        if np.isscalar(zs):
            DE_z_grid = get_de_density_i(
                zs, self.zbin_edges, self.zbin_widths, self.zbin_centers, self.w_i
            )
        else:
            DE_z_grid = get_de_density(
                zs, self.zbin_edges, self.zbin_widths, self.zbin_centers, self.w_i
            )
        E2_z_grid = self.Omega_m0 * pow(1.0 + zs, 3) + (1 - self.Omega_m0) * DE_z_grid
        return self.Omega_m0 * pow(1.0 + zs, 3) / E2_z_grid

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Baryonic density values at specified redshifts.
        """
        if np.isscalar(zs):
            DE_z_grid = get_de_density_i(
                zs, self.zbin_edges, self.zbin_widths, self.zbin_centers, self.w_i
            )
        else:
            DE_z_grid = get_de_density(
                zs, self.zbin_edges, self.zbin_widths, self.zbin_centers, self.w_i
            )
        E2_z_grid = self.Omega_m0 * pow(1.0 + zs, 3) + (1 - self.Omega_m0) * DE_z_grid
        return self.Omega_b0 * pow(1.0 + zs, 3) / E2_z_grid

    @property
    def rdrag(self) -> float:
        """Sound horizon radius at last scattering in Mpc."""
        w_nu = 0.0107
        Obh2 = self.Omega_b0 * self.h**2
        Och2 = self.Omega_cdm0 * self.h**2
        rdrag = (
            55.154
            * np.exp(-72.3 * (w_nu * self.mnu + 0.0006) ** 2)
            / (Obh2**0.12807 * (Obh2 + Och2) ** 0.25351)
        )
        return rdrag


class DEBinnedEoSLinearPerturbations:
    """Class for linear perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self, background: DEBinnedEoSBackground, base_linear_perturbations: Perturbations
    ) -> None:
        assert background.Omega_k0 == 0, "Non flat geometries not supported"
        # must be binned-DE
        self.background = background
        # must be LCDM
        assert (
            base_linear_perturbations.background.w0 == -1.0
            and base_linear_perturbations.background.wa == 0.0
        ), "Base linear perturbations must be LCDM"
        self.base = base_linear_perturbations

        # is called later in kernels, has to match
        self.z = self.base.z

        # Sort scale factors and redshifts in ascending order (early to late times)
        # hardcoded
        self.z_sorted = np.linspace(0.0, 5.0, 256, endpoint=True)
        # Get scale factors
        self.a = 1.0 / (1.0 + self.z_sorted)
        self.a_sorted = self.a[::-1]

        # hardcoded to save time for f(R) by setting 128 k-values and interpolating later
        self.k = np.logspace(
            np.log10(self.base.k[0]), np.log10(self.base.k[-1]), 128, endpoint=True
        )  # in 1/Mpc
        self.k_len = len(self.k)

        self.zbin_edges = background.zbin_edges
        self.zbin_widths = background.zbin_widths
        self.zbin_centers = background.zbin_centers
        self.w_i = background.w_i
        self.a_grid = jnp.linspace(1e-4, 1.0, 512)
        z_grid = 1.0 / np.linspace(1e-4, 1.0, 512) - 1.0
        bins = get_z_bin(z_grid, background.zbin_edges)
        w_vals = [self.w_i[bins_i] for bins_i in bins]
        self.w_vals = jnp.array(w_vals)

        # Build background dict for MGrowth
        background_mgrowth = {
            "Omega_m": self.background.Omega_m(np.array([0.0]))[0],
            "h": self.background.h,
            "w0": -1.0,
            "wa": 0.0,
            "a_arr": self.a_sorted,
        }
        # Compute MG and LCDM growth and assign linear growth parameters
        self._compute_growth(background_mgrowth)

    def _compute_growth(self, bg_dict):
        """Handle model-specific MGrowth and LCDM growth evaluation."""
        # D_mg, f_mg = self._compute_growth_w_interp()
        cosmo = mgrowth.w_a(bg_dict)
        D_raw, f_raw = cosmo.growth_parameters(self.a_grid, self.w_vals, use_fast=False)
        D_mg, f_mg = (
            np.repeat(D_raw[::-1, None], self.k_len, axis=1),
            np.repeat(f_raw[::-1, None], self.k_len, axis=1),
        )
        # D_mg, f_mg = D_raw[::-1], f_raw[::-1]

        # Get LCDM growth and construct array over k to be applied as normalisation
        base_cosmo = mgrowth.LCDM(bg_dict)
        D_base_raw, _ = base_cosmo.growth_parameters()
        D_base = D_base_raw[::-1]

        # Interpolate D(z, k) and f(z, k)
        self.dz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg, kx=1, ky=1
        )
        self.dz_norm_dz0_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_mg[0, :], kx=1, ky=1
        )
        self.fz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, f_mg, kx=1, ky=1
        )
        self.dz_norm_lcdm_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_base[0, None], kx=1, ky=1
        )
        self.dz_norm_lcdm_z_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_base[:, None], kx=1, ky=1
        )

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Calculate the growth factor D(z, k) normalized to D(0).

        Parameters
        ----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns
        -------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        return self.dz_norm_dz0_interp(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """Calculate the growth rate f(z, k) for given redshifts and wavenumbers.

        Parameters
        ----------
        zs : array_like
            Redshifts at which to calculate the growth rate.
        ks : array_like
            Wavenumbers at which to calculate the growth rate.

        Returns
        -------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """
        return self.fz_interp(zs, ks)

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Compute the linear matter power spectrum.

        Parameters
        ----------
        ks : numpy.ndarray
            Wavenumber in h Mpc^{-1}.
        zs : numpy.ndarray
            Redshifts.

        Returns
        -------
        np.ndarray
            Linear matter power spectrum at the specified redshifts and scales.
        """
        ps_base = self.base.matter_power_spectrum(0.0, ks)
        Dz_div_Dlcdmz0 = self.dz_norm_lcdm_interp(zs, ks)
        return Dz_div_Dlcdmz0**2 * ps_base

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        Dz_div_Dlcdmz0 = self.dz_norm_lcdm_interp(0.0, 0.01)
        return self.base.sigma8_0() * Dz_div_Dlcdmz0


class DEBinnedEoSNonlinearPerturbations:
    """Class for nonlinear (pseudo) perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: DEBinnedEoSBackground,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the HMemuNonLinearPerturbations intance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        redshift_max = HM2020_emu.emulator["nonlinear"]["bounds"]["z"][1]

        self.z = redshifts[redshifts <= redshift_max]
        # binned DE background
        self.background = background
        # binned DE linear perturbations
        self.de_lin = linearperturbations

        self.params_hm_emu = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            # modify for pseudo-power spectrum
            "As": self.background.As
            * self.de_lin.dz_norm_lcdm_z_interp(self.z, np.full(len(self.z), 0.01))[
                :, 0
            ]
            ** 2,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": self.background.mnu,
            "w0": -1.0,
            "wa": 0.0,
        }
        baryonic_boost = log10TAGN is not None

        if baryonic_boost:
            self.params_hm_emu["log10TAGN"] = log10TAGN

        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if key != "As":
                if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                    raise ValueError("HMcode 2020 NL emulator out of range.")
                else:
                    self.params_hm_emu[key] = np.tile(
                        self.params_hm_emu[key], len(self.z)
                    )
            else:
                check = np.array(
                    [
                        np.prod(As_i - hm_bounds["As"])
                        for As_i in self.params_hm_emu["As"]
                    ]
                )
                if (check > 0).any():
                    raise ValueError("As re-scaled is out of range.")

        self.params_hm_emu["z"] = self.z

        _, Pk = HM2020_emu.get_nonlinear_pk(
            nonu=False, **self.params_hm_emu, baryonic_boost=baryonic_boost
        )

        k_emu = HM2020_emu.emulator["nonlinear"]["k"] * self.background.h

        # Low-k extrapolation.
        # Done this way to use Pk array instead of calling an interpolator
        # This only works if the redshift array is exactly the same within
        # range. This should be, but we should probably make sure in some way
        Pk_lin_mask_k = linearperturbations.k < k_emu[0]
        Pk_lin_mask_z = linearperturbations.z <= redshift_max
        Pk_lin = linearperturbations.matter_power_spectrum(
            linearperturbations.z, linearperturbations.k
        )[Pk_lin_mask_z][:, Pk_lin_mask_k]
        k_all = np.concatenate((linearperturbations.k[Pk_lin_mask_k], k_emu))
        Pk_all = np.concatenate((Pk_lin, self.background.h**-3 * Pk), axis=1)

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            k_all,
            self.z,
            Pk_all,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out

        pk_interp = interpolate.RectBivariateSpline(self.z, self.k, self.Pk, kx=1, ky=1)

        self.Pk_interp = pk_interp

    def matter_power_spectrum(self, zs, ks):
        r"""Compute the linear matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift

        """
        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.


        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns:
        --------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        return self.de_lin.growth_factor(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        return self.de_lin.growth_rate(zs, ks)

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""

        return self.de_lin.sigma8_0()
