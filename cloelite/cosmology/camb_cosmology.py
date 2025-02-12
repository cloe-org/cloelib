# cloelite imports
from cloelite.auxiliary.units import SPEED_OF_LIGHT
from cloelite.cosmology.cosmology import Background

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
                 As: float, ns: float, 
                 w0: float, wa: float, gamma_MG: float) -> None:
        """
        Initializes the CAMBBackground class with cosmological parameters.

        Args:
            H0 (float): Hubble parameter at z=0 in km/s/Mpc.
            Omb (float): Baryonic matter density parameter.
            Omc (float): Cold dark matter density parameter.
            Omk (float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            w (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the equation of state.
            gamma_MG (float): Modified gravity growth parameter.
        """
        self.H0 = H0
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG

        # Initialize CAMB parameters
        self.interface_args = {'CAMBparams': camb.CAMBparams()}
        self.interface_args['CAMBparams'].set_cosmology(
            H0=self.H0,
            ombh2=self.Omega_b0 * (self.H0 / 100) ** 2,
            omch2=self.Omega_cdm0 * (self.H0 / 100) ** 2,
            omk=self.Omega_k0
        )
        self.interface_args['CAMBparams'].set_dark_energy(w=self.w0, wa=self.wa)
        self.interface_args['CAMBparams'].InitPower.set_params(As=self.As, ns=self.ns)

        # Call CAMB to compute the background
        self.results = camb.get_background(self.interface_args['CAMBparams'])

    def hubble_parameter(self, zs: np.ndarray, units: str = "1/Mpc") -> np.ndarray:
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
        c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s
        delta_z = self.comoving_distance(zs)[None, :] - self.comoving_distance(zs)[:, None]
        x = delta_z * self.H0 / c_0

        if self.Omega_k0 == 0.0:
            y = x
        elif self.Omega_k0 > 0.0:
            y = np.sinh(np.sqrt(self.Omega_k0) * x) / np.sqrt(self.Omega_k0)
        else:
            y = np.sin(np.sqrt(-self.Omega_k0) * x) / np.sqrt(-self.Omega_k0)

        return y * (c_0 / self.H0)

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the angular diameter distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Angular diameter distance values.
        """
        return self.results.angular_diameter_distance(zs)

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return (
            self.results.get_Omega("cdm", z=zs)
            + self.results.get_Omega("baryon", z=zs)
            + self.results.get_Omega("nu", z=zs)
        )


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
        
        self.kmax = 100
        self.z = redshifts

        self.background.interface_args['CAMBparams'].set_matter_power(redshifts=redshifts, kmax=self.kmax)
        self.results = camb.get_results(self.background.interface_args['CAMBparams'])

    def matter_power_spectrum(self) -> np.ndarray:
        """
        Calculates the linear matter power spectrum.

        Returns:
            np.ndarray: Linear power spectrum values \(P(k)\).
        """
        k_values, z_values, pk_values = self.results.get_linear_matter_power_spectrum(
            hubble_units=False, k_hunit=False
        )
        self.k = k_values
        self.z = z_values
        return pk_values

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
        self.kmax = 100
        self.z = redshifts

        # Configure CAMB parameters for nonlinear calculations
        self.background.interface_args['CAMBparams'].NonLinear = model.NonLinear_both

        if nonlinear_model:
            self.background.interface_args['CAMBparams'].NonLinearModel.set_params(halofit_version=nonlinear_model)

        self.background.interface_args['CAMBparams'].set_matter_power(redshifts=redshifts, kmax=self.kmax)

        # Compute nonlinear perturbations
        self.results = camb.get_results(self.background.interface_args['CAMBparams'])


    def matter_power_spectrum(self) -> np.ndarray:
        """
        Calculates the nonlinear matter power spectrum.

        This function uses CAMB to compute the nonlinear matter power spectrum \( P(k) \) 
        as a function of wavenumber \( k \) and redshift \( z \).

        Units of 1/Mpc

        Returns:
            np.ndarray: Nonlinear power spectrum values \( P(k) \).
        """
        k_values, z_values, pk_values = self.results.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False
        )
        self.k = k_values
        self.z = z_values
        return pk_values

      