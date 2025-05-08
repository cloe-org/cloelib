# cloelib imports
from cloelib.auxiliary import units
from cloelib.cosmology.cosmology import Background

# General imports
import numpy as np
from typing import Tuple, Optional

# Cosmology imports
try:
    import camb  # type: ignore
    from camb import model  # type: ignore
except ImportError as e:
    raise ImportError("camb could not be imported.") from e

"""
## Notes:

- This implementation interfaces with CAMB while adhering to the
Background, LinearPerturbations and NonLinearPerturbations Protocols.
"""


class CAMBBackground:
    """
    A wrapper for CAMB background cosmological calculations.
    """

    def __init__(self, H0: float, Omega_b0: float, Omega_cdm0: float, Omega_k0: float,
                 As: float, ns: float, mnu: float,
                 w0: float, wa: float, gamma_MG: float) -> None:
        """
        Initializes the CAMBBackground class with cosmological parameters.

        Args:
            H0 (float): Hubble parameter in [km/s/Mpc].
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0(float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (float): Total sum of neutrino mass in [eV].
            w0 (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the dark energy equation of state.
            gamma_MG (float): Modified gravity growth parameter.
        """
        self.H0 = H0
        self.h = self.H0 / 100
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG
        self.mnu = mnu

        # Initialize CAMB parameters
        self.interface_args = {'CAMBparams': camb.CAMBparams()}
        self.interface_args['CAMBparams'].set_cosmology(
            H0=self.H0,
            ombh2=self.Omega_b0 * (self.h) ** 2,
            omch2=self.Omega_cdm0 * (self.h) ** 2,
            omk=self.Omega_k0,
            mnu = self.mnu
        )
        self.interface_args['CAMBparams'].set_dark_energy(w=self.w0, wa=self.wa)
        self.interface_args['CAMBparams'].InitPower.set_params(As=self.As, ns=self.ns)

        # Call CAMB to compute the background
        self.results = camb.get_background(self.interface_args['CAMBparams'])

    @property
    def _interface_args(self) -> dict:
        """
        Save internal structure format of interface codes
        """
        return self.interface_args

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Returns the Hubble parameter as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.
            units (str): Units for the Hubble parameter ('1/Mpc' or 'km/s/Mpc').

        Returns:
            np.ndarray: Hubble parameter values at specified redshifts.
        """
        if units == "1/Mpc":
            return self.results.h_of_z(zs)
        if units == "km/s/Mpc":
            return self.results.hubble_parameter(zs)
        raise ValueError("Unsupported units for hubble_parameter. Choose '1/Mpc' or 'km/s/Mpc'.")

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the comoving distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Comoving distance values.
        """
        return self.results.comoving_radial_distance(zs)

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the transverse comoving distance between two redshifts.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Transverse comoving distance values.
        """
        x = self.comoving_distance(zs)

        if self.Omega_k0 == 0.0:
            y = x
        elif self.Omega_k0 > 0.0:
            y = np.sinh(np.sqrt(self.Omega_k0) * x) / np.sqrt(self.Omega_k0)
        else:
            y = np.sin(np.sqrt(-self.Omega_k0) * x) / np.sqrt(-self.Omega_k0)

        return y

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the angular diameter distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Angular diameter distance values.
        """
        return self.results.angular_diameter_distance(zs)

    def Omega_m(self, zs: np.ndarray, nonu: bool = False) -> np.ndarray:
        """
        Returns the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.
            nonu (bool): if True, massive neutrinos are not included
                         in the density parameter summation.

        Returns:
            np.ndarray: Matter density values.
        """
        Om = self.results.get_Omega("cdm", z=zs) + self.results.get_Omega(
            "baryon", z=zs
        )
        if nonu:
            return Om
        return Om + self.results.get_Omega("nu", z=zs)

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Baryonic density values at specified redshifts.
        """
        return self.results.get_Omega("baryon", z=zs)

    def rho_crit(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the critical density as a function of redshift.

        Units: Mpc^{-3} Msun

        Args:
            zs (np.ndarray): Redshifts.

        Returns:
            float: Critical density value at the specified redshift.
        """
        h_in_seconds = self.hubble_parameter(zs) / units.MPC_TO_KM
        return 3.0 * h_in_seconds**2.0 / (8.0 * np.pi * units.GRAVITATIONAL_CONSTANT)

    def dV_dzdO(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the volume element per redshit per solid angle
        at the redshift requested.


        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: volume element in Mpc^3 h^{-3}
        """
        return (
            units.SPEED_OF_LIGHT
            / 1.0e3
            * self.comoving_distance(zs) ** 2.0
            * self.hubble_parameter(zs)
        )

    def rdrag(
        self,
    ) -> float:
        """
        Returns the Sound horizon radius at last scattering.

        Returns:
            float: Sound horizon radius at last scattering
        """
        return self.results.get_derived_params()["rdrag"]


class CAMBLinearPerturbations:
    """
    A wrapper for CAMB linear perturbation calculations.
    """

    def __init__(self, background: Background, redshifts: np.ndarray) -> None:
        """
        Initializes the CAMBLinearPerturbations class with a background instance.

        Args:
            background (Background): A CAMBBackground instance.
            redshifts (np.ndarray): Array of redshifts for the calculations.
        """
        self.background = background

        self.kmax = 300.
        self.z = redshifts

        self.background.interface_args['CAMBparams'].set_matter_power(
            redshifts=redshifts, kmax=self.kmax)
        self.results = camb.get_results(self.background.interface_args['CAMBparams'])

        self.k, _, self.Pk = self.results.get_linear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)

    def matter_power_spectrum(self, zs, ks, hubble_units=False,
                              k_hunit=False, delta='delta_tot') -> np.ndarray:
        r"""Computes the linear matter power spectrum.

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

        delta: (Optional) str
            Variable for which to compute the power spectrum

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift
        """
        pk_values = camb.get_matter_power_interpolator(
            self.background.interface_args['CAMBparams'],
            nonlinear=False, extrap_kmax=self.kmax,
            hubble_units=hubble_units, k_hunit=k_hunit,
            var1=delta, var2=delta).P(zs, ks)
        return pk_values

    def growth_rate(self) -> np.ndarray:
        """
        Calculates growth rate.

        Returns:
            np.ndarray: growth rate.
        """
        f_z = self.results.get_fsigma8()/self.results.get_sigma8()
        # Reversing array because camb re-sorts redshifts when power spectrum is computed
        return f_z[::-1]

    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        Returns:
        --------
        np.ndarray
            The growth factor at the specified redshift and wavenumber.
        """
        D_z_k = np.sqrt(self.matter_power_spectrum(zs, ks) / \
                        self.matter_power_spectrum(0.0, ks))

        return D_z_k


class CAMBNonLinearPerturbations:
    """
    A wrapper for CAMB nonlinear perturbation calculations.
    """

    def __init__(self, background: Background, redshifts: np.ndarray,
                 nonlinear_model: Optional[str] = None) -> None:
        """
        Initializes the CAMBNonLinearPerturbations class with linear perturbation data.

        Args:
            linear_perturbations (LinearPerturbations): An instance of the LinearPerturbations class.
            redshifts (np.ndarray): Array of redshifts for the calculations.
            nonlinear_model (Optional[str]): The nonlinear model to use (e.g., "takahashi").
                Defaults to None, which uses the CAMB default model.
        """

        self.background = background
        self.kmax = 500
        self.z = redshifts

        # Configure CAMB parameters for nonlinear calculations
        self.background.interface_args['CAMBparams'].NonLinear = model.NonLinear_both

        if nonlinear_model:
            self.background.interface_args['CAMBparams'].NonLinearModel.set_params(halofit_version=nonlinear_model)

        self.background.interface_args['CAMBparams'].set_matter_power(redshifts=redshifts, kmax=self.kmax)

        # Compute nonlinear perturbations
        self.results = camb.get_results(self.background.interface_args['CAMBparams'])

        self.k, _, self.Pk = self.results.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)


    def matter_power_spectrum(self, zs, ks, hubble_units=False,
                              k_hunit=False, delta='delta_tot') -> np.ndarray:
        r"""Computes the nonlinear matter power spectrum.

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

        delta: (Optional) str
            Variable for which to compute the power spectrum

        Returns
        -------
        pk: numpy.ndarray
            Nonlinear matter power spectrum at the specified scale
            and redshift
        """
        pk_values = self.results.get_matter_power_interpolator(
            nonlinear=True, extrap_kmax=self.kmax,
            hubble_units=hubble_units, k_hunit=k_hunit,
            var1=delta, var2=delta).P(zs, ks)
        return pk_values

    def growth_rate(self) -> np.ndarray:
        """
        Calculates growth rate.

        Returns:
            np.ndarray: growth rate.
        """
        f_z = self.results.get_fsigma8()/self.results.get_sigma8()
        # Reversing array because camb re-sorts redshifts when power spectrum is computed
        return f_z[::-1]

    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        Returns:
        --------
        np.ndarray
            The growth factor at the specified redshift and wavenumber.
        """
        D_z_k = np.sqrt(self.matter_power_spectrum(zs, ks) / \
                        self.matter_power_spectrum(0.0, ks))
        return D_z_k
