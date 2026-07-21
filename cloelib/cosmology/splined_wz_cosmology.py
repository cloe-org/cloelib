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
from scipy.interpolate import make_interp_spline, CubicSpline

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



def w_of_z(z, z_i, w_i):
    """
    Returns the equation of state as a function of redshift.

    Outside the knot range, w is held constant at the endpoint values:
    w_i[0] for z < z_i[0] and w_i[-1] for z > z_i[-1].
    """
    z = np.asarray(z)
    #spl_i = make_interp_spline(z_i, w_i)
    spl_i = CubicSpline(z_i, w_i)
    return spl_i(np.clip(z, z_i[0], z_i[-1]))


def get_de_density(z, z_i, w_i):
    """
    Optimized version using pre-computation and vectorization.
    """
    z = np.asarray(z)
    w_grid = np.asarray(w_of_z(z, z_i, w_i), dtype=float)
    de_integrand = (1.0 + w_grid) / (1.0 + z)
    de_int = cumulative_trapezoid(de_integrand, z, initial=0.0)
    return np.exp(3.0 * de_int)


class DESplinedEoSBackground:
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
        z_i: np.ndarray, 
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
            z_i (np.ndarray): Array of redshift values for the spline.
            w_i (np.ndarray): Array of equation of state parameters for the redshift bins.
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
        self.z_i = z_i
        self.w_i = w_i

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
            zs, self.z_i, self.w_i
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
        prepend_z0 = zs[0] != 0.0
        zz_ = np.hstack(([0.0], zs)) if prepend_z0 else zs
        DE_z_grid = get_de_density(zz_, self.z_i, self.w_i)
        E_z_grid = np.sqrt(
            self.Omega_m0 * pow(1.0 + zz_, 3) + (1 - self.Omega_m0) * DE_z_grid
        )
        # Dimensionless integral of c/H0 / E(z); convert to Mpc
        r_z_total = (
            cumulative_trapezoid(1.0 / E_z_grid, zz_, initial=0.0)
            * SPEED_OF_LIGHT
            / 1000
            / self.H0
        )
        return r_z_total[1:] if prepend_z0 else r_z_total

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
        DE_z_grid = get_de_density(
                zs, self.z_i, self.w_i
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
        DE_z_grid = get_de_density(
                zs, self.z_i, self.w_i
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


class DESplinedEoSLinearPerturbations:
    """Class for linear perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self, background: DESplinedEoSBackground, base_linear_perturbations: Perturbations
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

        self.z_i = background.z_i
        self.w_i = background.w_i
        self.a_grid = jnp.linspace(1e-4, 1.0, 512)
        z_grid = 1.0 / np.linspace(1e-4, 1.0, 512) - 1.0
        w_vals = w_of_z(z_grid, background.z_i, background.w_i)
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


class DESplinedEoSNonlinearPerturbations:
    """Class for nonlinear (pseudo) perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: DESplinedEoSBackground,
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
