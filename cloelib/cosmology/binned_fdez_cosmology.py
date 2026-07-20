"""Implementation of Background and Perturbation cosmology for binned density of dark energy."""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.cosmology.cosmology import Perturbations

# General imports
import numpy as np
from typing import Optional
from scipy.integrate import quad
from scipy import interpolate
import jax.numpy as jnp

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


def get_de_density(z, zbin_edges, fde_i):
    """
    Optimized version using pre-computation and vectorization.
    """
    z = np.asarray(z)
    bins = get_z_bin(z, zbin_edges)

    # Vectorized computation
    # For each z, get the product from cumulative products
    # Use np.maximum to handle negative bin indices
    bin_idx_safe = np.maximum(0, bins)


    # Compute final density
    rho = np.array([fde_i[bin_idx_safe_i] for bin_idx_safe_i in bin_idx_safe])

    return rho


def get_de_density_i(z, zbin_edges, fde_i):
    bins = get_z_bin(z, zbin_edges)
    # Handle bins as scalar (get_z_bin returns scalar for scalar input)
    bin_idx = bins if isinstance(bins, (int, np.integer)) else bins[0]
    return fde_i[bin_idx]


class DEBinnedDensityBackground:
    """Beyond w0wa-background cosmological calculations."""

    def __init__(
        self,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        mnu: float,
        zbin_edges: np.ndarray,  # zbin_widths: np.ndarray, zbin_centers: np.ndarray,
        fde_i: np.ndarray,
        binning: str = "step",
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
            fde_i (np.ndarray): Array of density of dark energy parameters for the redshift bins.
            binning (str): Type of binning used for the redshift bins.
        """
        self.H0 = H0
        self.h = self.H0 / 100
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.mnu = mnu
        self.Omega_m0 = Omega_cdm0 + Omega_b0 + self.mnu / 93.14 / (self.h) ** 2
        self.zbin_edges = zbin_edges
        self.zbin_widths = np.diff(zbin_edges)
        self.zbin_centers = 0.5 * (zbin_edges[1:] + zbin_edges[:-1])
        self.fde_i = fde_i

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
            zs, self.zbin_edges, self.fde_i
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
                    z, self.zbin_edges, self.fde_i
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
        raise self.transverse_comoving_distance(zs) / (1 + zs)

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
                zs, self.zbin_edges, self.fde_i
            )
        else:
            DE_z_grid = get_de_density(
                zs, self.zbin_edges, self.fde_i
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
                zs, self.zbin_edges, self.fde_i
            )
        else:
            DE_z_grid = get_de_density(
                zs, self.zbin_edges, self.fde_i
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


class DEBinnedDensityLinearPerturbations:
    """Class for linear perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self, background: DEBinnedDensityBackground, base_linear_perturbations: Perturbations
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
        self.fde_i = background.fde_i
        self.a_grid = jnp.linspace(1e-4, 1.0, 512)
        z_grid = 1.0 / np.linspace(1e-4, 1.0, 512) - 1.0
        bins = get_z_bin(z_grid, background.zbin_edges)
        fde_vals = [self.fde_i[bins_i] for bins_i in bins]
        self.fde_vals = jnp.array(fde_vals)

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
        cosmo = mgrowth.fde_a(bg_dict)
        # growth expects per-bin amplitudes (len = n_bins), not the a-grid samples
        D_raw, f_raw = cosmo.growth_parameters(
            jnp.asarray(self.fde_i), jnp.asarray(self.zbin_edges)
        )
        D_mg, f_mg = (
            np.repeat(D_raw[::-1, None], self.k_len, axis=1),
            np.repeat(f_raw[::-1, None], self.k_len, axis=1),
        )


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


class DEBinnedDensityNonlinearPerturbations:
    """Class for nonlinear (pseudo) perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: DEBinnedDensityBackground,
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
