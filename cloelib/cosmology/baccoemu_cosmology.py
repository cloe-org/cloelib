"""Implementation of Background and Perturbation cosmology using BACCOemu (similarly to what done with HMcode2020emu)."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

from scipy import interpolate

# General imports
import numpy as np
from typing import Optional

# Cosmology imports
try:
    import baccoemu
except ImportError:
    raise ImportError("BACCOemu could not be imported")


class BACCOemuLinearPerturbations:
    """Class for perturbations cosmology using BACCOemu, inheriting from Perturbations parent class."""

    def __init__(
        self, background: Background, redshifts: np.ndarray, cold: bool = False
    ):
        """Intialize the HMemuLinearPerturbations instance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        self.emu = baccoemu.Matter_powerspectrum(
            verbose=False, nonlinear_boost=False, baryonic_boost=False
        )
        redshift_max = 1 / self.emu.emulator["linear"]["bounds"][-1][0].item() - 1

        self.z = redshifts[redshifts <= redshift_max]
        self.background = background
        self.cold = cold

        self.params_emu = {
            "omega_cold": self.background.Omega_cdm0 + self.background.Omega_b0,
            "omega_baryon": self.background.Omega_b0,
            "A_s": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": self.background.mnu,
            "w0": self.background.w0,
            "wa": self.background.wa,
        }

        self.params_emu["expfactor"] = 1 / (1 + self.z)

        k_emu, Pk = self.emu.get_linear_pk(cold=self.cold, **self.params_emu)

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            k_emu * self.background.h,
            self.z,
            Pk * self.background.h**-3,
            flag_range=True,
            option_wavenumber="logk2",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out

        pk_interp = interpolate.RectBivariateSpline(self.z, self.k, Pk_out, kx=1, ky=1)

        self.Pk_interp = pk_interp

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Compute the linear matter power spectrum.

        Args:
            ks (numpy.ndarray): Wave number in h Mpc^{-1}
            zs (numpy.ndarray): redshifts

        Returns:
            pk (numpy.ndarray): Linear matter power spectrum at the specified scale and redshift

        """
        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        $$
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}
        $$

        and normalizes as for $D(z)/D(0)$.

        Args:
            zs (array_like): Redshifts at which to calculate the growth factor.
            ks (array_like): Wavenumbers at which to calculate the growth factor.

        Returns:
            (np.ndarray): The growth factor as a function of redshift and wavenumber.
        """
        if hasattr(self, "Pk_interp") and self.Pk_interp is not None:
            D_z_k = np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        return D_z_k

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        return self.emu.get_sigma8(cold=self.cold, **self.params_emu)

    def sigma12_0(self) -> float:
        """
        Calculate the sigma12 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        return self.emu.get_sigma12(cold=self.cold, **self.params_emu)


class BACCOemuNonLinearPerturbations:
    """Class for non linear perturbations cosmology using BACCOemu, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
        cold: Optional[bool] = False,
        nonlinear_model_name: Optional[str] = "Arico2023",
        baryonic_boost: Optional[str] = None,
        baryonic_model_name: Optional[str] = "Burger2025",
        M_c: Optional[float] = None,
        eta: Optional[float] = None,
        beta: Optional[float] = None,
        M1_z0_cen: Optional[float] = None,
        theta_out: Optional[float] = None,
        theta_inn: Optional[float] = None,
        M_inn: Optional[float] = None,
    ):
        """Initialize the HMemuNonLinearPerturbations intance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        self.emu = baccoemu.Matter_powerspectrum(
            verbose=False,
            nonlinear_model_name=nonlinear_model_name,
            baryonic_boost=baryonic_boost,
            baryonic_model_name=baryonic_model_name,
        )

        redshift_max = 1 / self.emu.emulator["nonlinear"]["bounds"][-1][0].item() - 1

        self.z = redshifts[redshifts <= redshift_max]
        self.background = background

        self.params_emu = {
            "omega_cold": self.background.Omega_cdm0 + self.background.Omega_b0,
            "omega_baryon": self.background.Omega_b0,
            "A_s": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": self.background.mnu,
            "w0": self.background.w0,
            "wa": self.background.wa,
        }

        if baryonic_boost:
            self.params_emu["M_c"] = M_c
            self.params_emu["eta"] = eta
            self.params_emu["beta"] = beta
            self.params_emu["M1_z0_cen"] = M1_z0_cen
            self.params_emu["theta_out"] = theta_out
            self.params_emu["theta_inn"] = theta_inn
            self.params_emu["M_inn"] = M_inn

        self.params_emu["expfactor"] = 1 / (1 + self.z)

        # Low-k extrapolation taken care by baccoemu (nonlinear boosts tends to 1)
        low_k = (
            self.emu.emulator["linear"]["k"] < self.emu.emulator["nonlinear"]["k"][0]
        )
        all_k = np.concatenate(
            (
                self.emu.emulator["linear"]["k"][low_k],
                self.emu.emulator["nonlinear"]["k"],
            )
        )

        _, Pk = self.emu.get_nonlinear_pk(
            cold=cold, baryonic_boost=baryonic_boost, k=all_k, **self.params_emu
        )

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            all_k * self.background.h,
            self.z,
            Pk * self.background.h**-3,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        # Aleternative method using interpolators
        # Pk_lin = linearperturbations.Pk_interp(self.z, k_emu)
        # k_out, z_out, boost_out = \
        #     extend_spectra(k_emu, self.z, self.background.h ** -3 * Pk / Pk_lin,
        #                    flag_range=True,
        #                    option_wavenumber="power_law",
        #                    option_redshift="power_law", extrap_z = redshifts,
        #                    option_cosmo="const", ns=self.background.ns)
        # self.Pk = linearperturbations.Pk_interp(self.z, k_out) * boost_out

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out

        pk_interp = interpolate.RectBivariateSpline(self.z, self.k, self.Pk, kx=1, ky=1)

        self.Pk_interp = pk_interp

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Compute the linear matter power spectrum.

        Args:
            ks (numpy.ndarray): Wave number in h Mpc^{-1}
            zs (numpy.ndarray): redshifts

        Returns:
            pk (numpy.ndarray): Linear matter power spectrum at the specified scale and redshift

        """
        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        $$
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\
        $$

        and normalizes as for $D(z)/D(0)$.

        Args:
            zs (array_like): Redshifts at which to calculate the growth factor.
            ks (array_like): Wavenumbers at which to calculate the growth factor.

        Returns:
            (np.ndarray): The growth factor as a function of redshift and wavenumber.
        """
        if hasattr(self, "Pk_interp") and self.Pk_interp is not None:
            D_z_k = np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        return D_z_k

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        return self.emu.get_sigma8(cold=self.cold, **self.params_emu)

    def sigma12_0(self) -> float:
        """
        Calculate the sigma12 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        return self.emu.get_sigma12(cold=self.cold, **self.params_emu)
