"""Implementation of Background and Perturbation cosmology using HMcode2020Emu."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, BaryonBoostMixin, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

from scipy import interpolate

# General imports
import numpy as np
from typing import Optional, Sequence

# Cosmology imports
try:
    import HMcode2020Emu as hmcodeemu

    HM2020_emu = hmcodeemu.Matter_powerspectrum()
    redshift_max = HM2020_emu.emulator["linear"]["bounds"]["z"][1]
except ImportError:
    raise ImportError("HMcode2020emu could not be imported or initialised.")


class HMemuLinearPerturbations:
    """Class for perturbations cosmology using HMemu, compatibly with the Perturbations protocol."""

    def __init__(self, background: Background, redshifts: np.ndarray):
        """Intialize the HMemuLinearPerturbations instance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        self.z = redshifts[redshifts <= redshift_max]
        self.background = background

        self.params_hm_emu = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": _set_neutrino_masses(self.background),
            "w0": self.background.w0,
            "wa": self.background.wa,
        }

        hm_bounds = HM2020_emu.emulator["linear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 lin emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key], len(self.z))

        self.params_hm_emu["z"] = self.z

        _, Pk = HM2020_emu.get_linear_pk(**self.params_hm_emu)

        k_emu = HM2020_emu.emulator["linear"]["k"] * self.background.h

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            k_emu,
            self.z,
            self.background.h**-3 * Pk,
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

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        Returns:
            (np.ndarray): The growth rate as a function of redshift and wavenumber.
        """
        self.sigma8, self.fsigma8 = HM2020_emu.get_sigma8(**self.params_hm_emu)

        return self.fsigma8 / self.sigma8


class HMemuNonLinearPerturbations:
    """Class for non linear perturbations cosmology using HMemu,  compatibly with the Perturbations protocol."""

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the HMemuNonLinearPerturbations instance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        redshift_max = HM2020_emu.emulator["nonlinear"]["bounds"]["z"][1]

        self.z = redshifts[redshifts <= redshift_max]
        self.background = background

        self.params_hm_emu = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": _set_neutrino_masses(self.background),
            "w0": self.background.w0,
            "wa": self.background.wa,
        }
        baryonic_boost = log10TAGN is not None

        if baryonic_boost:
            self.params_hm_emu["log10TAGN"] = log10TAGN

        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 NL emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key], len(self.z))

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
        Pk_lin = linearperturbations.Pk[Pk_lin_mask_z][:, Pk_lin_mask_k]
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

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        Returns:
            (np.ndarray): The growth rate as a function of redshift and wavenumber.
        """
        self.sigma8, self.fsigma8 = HM2020_emu.get_sigma8(**self.params_hm_emu)

        return self.fsigma8 / self.sigma8

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        self.params_hm_emu["z"] = np.insert(self.z, 0, 0.0)
        max_len = len(self.params_hm_emu["z"])
        for k, v in self.params_hm_emu.items():
            if len(v) < max_len:
                pad_size = max_len - len(v)
                # Repeat last element to match length
                self.params_hm_emu[k] = np.pad(v, (0, pad_size), mode="edge")
        self.sigma8_0, _ = HM2020_emu.get_sigma8(**self.params_hm_emu)
        return self.sigma8_0[0]


def _set_neutrino_masses(background: Background) -> float:
    r"""Set neutrino masses in the parameters dictionary.

    This method adds neutrino masses to the provided dictionary.
    It also ensures consistency with the background cosmology.
    HMcode2020Emu only supports a single species massive of neutrinos, so this method
    throws an error if multiple massive neutrino species are provided.

    Parameters
    ----------
    background: Background
        Background class containing cosmology and background quantities

    Returns
    -------
    float
        The total neutrino mass in eV.
    """
    if background.N_mnu > 1:
        raise ValueError(
            "HMcode2020Emu only supports a single species of neutrinos. "
            "Set N_mnu=1 in the Background class."
        )
    if not np.isclose(background.N_ur, 2.0308, rtol=1e-4):
        raise ValueError(
            "HMcode2020Emu only supports a fixed number of relativistic species (N_ur=2.0308). "
            "Set N_ur=2.0308 in the Background class."
            "[Note that HMcode2020Emu actually sets N_ur=2.0328,"
            "this will be fixed in a future release.]"
        )
    if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
        raise ValueError(
            "HMcode2020Emu only supports a fixed number of effective"
            f"relativistic species (N_eff=3.044). Found {background.N_eff} "
            "Ensure that N_eff=3.044 in the Background class."
            "[Note that HMcode2020Emu actually sets N_eff=3.046,"
            "this will be fixed in a future release.]"
        )
    if isinstance(background.mnu, Sequence) or isinstance(background.mnu, np.ndarray):
        raise ValueError(
            "HMcode2020Emu only supports a single species of neutrinos. "
            "Set N_mnu=1 in the Background class."
        )
    else:
        mnu_arg = float(background.mnu)
    # returns the neutrino mass in eV
    return mnu_arg


class HMcode2020BaryonBoostMixin(BaryonBoostMixin):
    """Mixin providing baryonic suppression via an HMcode2020-computed P_baryon/P_dmo ratio.

    Expects the host class to have set ``self._baryon_ratio_interp``,
    ``self._baryon_k_nl_min``, and ``self.background``.
    For scales below the nonlinear emulator k range, the suppression factor is 1.
    """

    def baryonic_suppression(self, zs, ks, k_hunit: bool = False) -> np.ndarray:
        """Return B(z, k) = P_baryon(z, k) / P_dmo(z, k).

        Parameters
        ----------
        zs : array_like
            Redshifts.
        ks : array_like
            Wavenumbers. By default in 1/Mpc; pass k_hunit=True for h/Mpc.
        k_hunit : bool
            If True, convert ks from h/Mpc to 1/Mpc before evaluating.

        Returns
        -------
        np.ndarray
            Baryonic suppression factor, shape (nz, nk).
            Returns 1.0 for scales below the nonlinear emulator k range.
        """
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        if k_hunit:
            ks = ks * self.background.h
        result = np.ones((len(zs), len(ks)))
        in_range = ks >= self._baryon_k_nl_min
        if np.any(in_range):
            result[:, in_range] = self._baryon_ratio_interp(zs, ks[in_range])
        return result


class HMemuNonLinearBaryonicPerturbations(
    HMcode2020BaryonBoostMixin, HMemuNonLinearPerturbations
):
    """HMcode2020 nonlinear perturbations with baryonic suppression applied.

    Initialises the parent class without log10TAGN (DMO), then runs the
    emulator a second time with ``log10TAGN`` to pre-compute
    B(z, k) = P_baryon / P_dmo as a bivariate spline over the nonlinear k range.
    Scales below the nonlinear k range have suppression factor = 1.
    """

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
        log10TAGN: float,
    ):
        """Initialise HMemuNonLinearBaryonicPerturbations."""
        super().__init__(background, linearperturbations, redshifts, log10TAGN=None)

        # Validate log10TAGN against emulator bounds
        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]
        if np.prod(log10TAGN - hm_bounds["log10TAGN"]) > 0:
            raise ValueError(
                f"HMcode 2020 NL emulator: log10TAGN={log10TAGN} is out of bounds "
                f"{hm_bounds['log10TAGN']}."
            )

        # Emulator z grid is stored in params_hm_emu (not rebound by extend_spectra)
        z_emu = self.params_hm_emu["z"]
        nz = len(z_emu)

        # DMO spectrum at the nonlinear k grid
        _, Pk_dmo = HM2020_emu.get_nonlinear_pk(
            nonu=False, **self.params_hm_emu, baryonic_boost=False
        )

        # Baryonic spectrum with log10TAGN
        params_with_tagn = {
            **self.params_hm_emu,
            "log10TAGN": np.tile(log10TAGN, nz),
        }
        _, Pk_baryon = HM2020_emu.get_nonlinear_pk(
            nonu=False, **params_with_tagn, baryonic_boost=True
        )

        ratio = Pk_baryon / Pk_dmo  # shape (nz, nk_nl)
        k_nl_phys = (
            HM2020_emu.emulator["nonlinear"]["k"] * self.background.h
        )  # h/Mpc -> 1/Mpc
        self._baryon_k_nl_min = k_nl_phys[0]
        self._baryon_ratio_interp = interpolate.RectBivariateSpline(
            z_emu, k_nl_phys, ratio, kx=1, ky=1
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Total matter power spectrum with baryonic suppression applied."""
        return super().matter_power_spectrum(zs, ks) * self.baryonic_suppression(zs, ks)
