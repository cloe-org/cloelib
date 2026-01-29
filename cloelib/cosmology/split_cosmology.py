# cloelib imports
from cloelib.auxiliary import extrapolator
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.derived_cosmology import growth_function_ODE

import numpy as np
from scipy import interpolate
import copy
from typing import Optional, Sequence

# Cosmology imports
try:
    from classy import Class  # type: ignore
except ImportError as e:
    raise ImportError("classy could not be imported.") from e

try:
    import HMcode2020Emu as hmcodeemu

    HM2020_emu = hmcodeemu.Matter_powerspectrum()
    redshift_max = HM2020_emu.emulator["linear"]["bounds"]["z"][1]
except ImportError:
    raise ImportError("HMcode2020emu could not be imported or initialised.")


class SplitLinearPerturbations:
    """Class to output the rescaled linear matter power spectrum for the
    growth-geometry split, inheriting from the Perturbations parent class
    and using CLASS."""

    def __init__(
        self, background: Background, omega_m_growth: float,
        redshifts: np.ndarray
    ):
        """Initialise SplitLinearPerturbations."""
        self.background = background
        self.omega_m_growth = omega_m_growth
        self.z = redshifts
        self.kmax = 100
        self.results = None  # Store CLASS results
        self.hubble_units = False

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background.interface_args)
        self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
        self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
        self.interface_args["CLASSparams"]["k_per_decade_for_bao"] = 70
        self.interface_args["CLASSparams"]["k_per_decade_for_pk"] = 10
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        self.interface_args["CLASSparams"]["non linear"] = "none"
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args["CLASSparams"])
        self.results.compute()

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the growth-geometry split CLASS linear matter power
        spectrum. This implies a rescaling with the growth function of the
        matter power spectrum and also sigma_8, as in 2301.03694

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk_linear_EBS: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift from the Einstein-Boltzmann solver. This is needed to
            compute the boost factor in the class SplitNonLinearPerturbations.

        pk_linear: numpy.ndarray
            Rescaled linear matter power spectrum at the specified scale
            and redshift.
        """

        self.k = ks

        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        self.Pk_EBS = np.array([[self.results.pk(k, z) for k in ks] for z in zs])  # type:ignore[union-attr]

        omega_m_geo = self.background.Omega_cdm0 + self.background.Omega_b0

        # Compute the growth factor
        g_z_geo = growth_function_ODE(self.background, zs, omega_m_geo)
        g_z_growth = growth_function_ODE(self.background, zs, self.omega_m_growth)

        self.Pk = np.zeros_like(self.Pk_EBS)

        # Rescale the matter power spectrum with G(z)
        for i in range(len(zs)):
            rescale_fac = g_z_growth[i] ** 2 / g_z_geo[i] ** 2
            self.Pk[i, :] = rescale_fac * self.Pk_EBS[i, :]

        # Rescale sigma_8
        self.sigma8_0 = g_z_growth[i] / g_z_geo[i] * self.sigma8_0_EBS()
        return self.Pk

    def sigma8_0_EBS(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology from the EBS.

        Returns:
        --------
        float
            The sigma8 value.
        """
        self.sigma8_0 = self.results.sigma8()  # type: ignore[union-attr]
        return self.sigma8_0


class SplitNonLinearPerturbations:
    """Class to output the rescaled non-linear matter power spectrum for the
    growth-geometry split, inheriting from the Perturbations parent class
    and using HMcode2020Emulator."""

    def __init__(
        self,
        background: Background,
        linperturbations: Perturbations,
        omega_m_growth: float,
        redshifts: np.ndarray,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the HMemuNonLinearPerturbations intance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        redshift_max = HM2020_emu.emulator["nonlinear"]["bounds"]["z"][1]
        
        self.background = background
        self.omega_m_growth = omega_m_growth
        self.redshifts = redshifts
        self.z = self.redshifts[self.redshifts <= redshift_max]

        self.params_hm_emu = {
            "omega_cdm": omega_m_growth - self.background.Omega_b0,
            "omega_baryon": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": _set_neutrino_masses(self.background),
            "w0": self.background.w0,
            "wa": self.background.wa,
        }
        self.baryonic_boost = log10TAGN is not None
        if self.baryonic_boost:
            self.params_hm_emu["log10TAGN"] = log10TAGN

        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 NL emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key], len(self.z))

        self.params_hm_emu["z"] = self.z

        _, Pk_lin_emu = HM2020_emu.get_linear_pk(**self.params_hm_emu)
        _, Pk = HM2020_emu.get_nonlinear_pk(
            nonu=False, **self.params_hm_emu,
            baryonic_boost=self.baryonic_boost)

        k_emu_lin = HM2020_emu.emulator["linear"]["k"] * self.background.h
        k_emu = HM2020_emu.emulator["nonlinear"]["k"] * self.background.h

        # Low-k extrapolation.
        # Done this way to use Pk array instead of calling an interpolator
        # This only works if the redshift array is exactly the same within
        # range. This should be, but we should probably make sure in some way
        Pk_lin_mask_k = k_emu_lin < k_emu[0]
        Pk_lin_mask_z = self.z <= redshift_max
        Pk_lin = Pk_lin_emu[Pk_lin_mask_z][:, Pk_lin_mask_k]
        k_all = np.concatenate((k_emu_lin[Pk_lin_mask_k], k_emu))
        Pk_all = np.concatenate((self.background.h**-3 * Pk_lin,
                                 self.background.h**-3 * Pk),
                                 axis=1)

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extrapolator.extend_spectra(
            k_all,
            self.z,
            Pk_all,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=self.redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out

        pk_lin_rescale = linperturbations.matter_power_spectrum(self.z,
                                                                k_emu_lin)

        pk_lin_rescale_interp = interpolate.RectBivariateSpline(
            self.z, k_emu_lin, pk_lin_rescale, kx=1, ky=1)
        pk_lin_emu_interp = interpolate.RectBivariateSpline(
            self.z, k_emu_lin, Pk_lin_emu * self.background.h**-3, kx=1, ky=1)
        pk_interp = interpolate.RectBivariateSpline(
            self.z, self.k, self.Pk, kx=1, ky=1)

        self.Pk_lin_rescale_interp = pk_lin_rescale_interp
        self.Pk_lin_emu_interp = pk_lin_emu_interp
        self.Pk_interp = pk_interp

    def matter_power_spectrum(
        self, zs, ks,
    ) -> np.ndarray:
        """Calculate the split non-linear boost using HMemu2020.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk: numpy.ndarray
            Non-linear matter power spectrum at the specified scale
            and redshift
        """
        Pk_lin_rescale = self.Pk_lin_rescale_interp(zs, ks)

        # Compute the boost factor from the standard power spectra
        self.boost = self.Pk_interp(zs, ks) / self.Pk_lin_emu_interp(zs, ks)

        self.Pk_nonlinear = np.zeros_like(self.boost)
        # Add the boost to the rescaled power spectrum (instead of EBS)
        for i in range(len(zs)):
            self.Pk_nonlinear[i, :] = self.boost[i, :] * Pk_lin_rescale[i, :]

        return self.Pk_nonlinear

def _set_neutrino_masses(background: Background) -> float:
    r"""Set neutrino masses in the parameters dictionary.

    This method adds neutrino masses to the provided dictionary.
    It also ensures consistency with the background cosmology.
    HMcode2020Emu only supports a single species massive of neutrinos, so this method
    throws an error if multiple massive neutrino species are provided.

    Parametersplt.loglog(ks / h, nl_3_3_boost[0, :],
           label="nonlinear hmemu boost only $\\Omega_{\\rm m}=0.3$")
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
