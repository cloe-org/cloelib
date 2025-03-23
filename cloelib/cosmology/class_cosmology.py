# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.units import SPEED_OF_LIGHT

# General imports
import numpy as np
import copy
from typing import Tuple, Optional
import interpax
from scipy.interpolate import UnivariateSpline

# Cosmology imports
try:
    from classy import Class # type: ignore
except ImportError as e:
    raise ImportError("classy could not be imported.") from e

class CLASSBackground:
    """
    A wrapper for CLASS background cosmological calculations.
    """
    c0 = SPEED_OF_LIGHT/1000
    def __init__(self, H0: float, Omega_b0: float, Omega_cdm0: float, Omega_k0: float,
                 As: float, ns: float,
                 w0: float, wa: float, gamma_MG: float) -> None:
        """
        Initializes the CLASSBackground class with cosmological parameters.

        Args:
            H0 (float): Hubble parameter at z=0 in km/s/Mpc.
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0 (float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            w0 (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the equation of state.
            gamma_MG (float): Modified gravity growth parameter (not directly used in CLASS, but kept for protocol compliance).
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
        self.gamma_MG = gamma_MG  # Kept for protocol, but CLASS doesn't directly use it

        # Initialize CLASS parameters
        self.interface_args = {'CLASSparams': {}}  # Use a dictionary for CLASS parameters
        self.interface_args['CLASSparams']['H0'] = self.H0
        self.interface_args['CLASSparams']['omega_b'] = self.Omega_b0 * (self.h)**2
        self.interface_args['CLASSparams']['omega_cdm'] = self.Omega_cdm0 * (self.h)**2
        self.interface_args['CLASSparams']['Omega_k'] = self.Omega_k0
        self.interface_args['CLASSparams']['n_s'] = self.ns
        self.interface_args['CLASSparams']['A_s'] = self.As
        self.interface_args['CLASSparams']['w0_fld'] = self.w0 # or w0
        self.interface_args['CLASSparams']['wa_fld'] = self.wa # or wa
        # To get correct perturbations for w0wa
        self.interface_args['CLASSparams']['use_ppf'] = "yes"
        # To avoid using a cosmological constant
        self.interface_args['CLASSparams']['Omega_Lambda'] = 0. 

        # Initialize CLASS
        self.results = Class()
        self.results.set(self.interface_args['CLASSparams'])
        self.results.compute()

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
        H = np.array([self.results.Hubble(z) for z in zs])  # CLASS returns H in 1/Mpc
        if units == "km/s/Mpc":
            return H * CLASSBackground.c0  # Convert to km/s/Mpc
        elif units == "1/Mpc":
            return H
        else:
            raise ValueError("Unsupported units.  Must be 'km/s/Mpc' or '1/Mpc'")

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the comoving distance as a function of redshift.
        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Comoving distance values.
        """
        return np.array([self.results.comoving_distance(z) for z in zs])

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the transverse comoving distance between two redshifts.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Transverse comoving distance values.
        """
        delta_z = self.comoving_distance(zs)[None, :] - self.comoving_distance(zs)[:, None]
        x = delta_z * self.H0 / CLASSBackground.c0

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
        return np.array([self.results.angular_distance(z) for z in zs])

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Returns the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return np.array([self.results.Om_m(z) for z in zs])

class CLASSLinearPerturbations:
    def __init__(self, background : Background, redshifts: np.ndarray):
        r"""
        A class to define perturbations cosmology using CLASS
        and inheriting from Perturbations parent class
        """
        self.background = background
        self.z = redshifts
        self.kmax = 100
        self.results = None  # Store CLASS results

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background._interface_args)
        self.interface_args['CLASSparams']['output'] = 'mPk, mTk'
        self.interface_args['CLASSparams']['P_k_max_1/Mpc'] = 100
        self.interface_args['CLASSparams']['k_per_decade_for_bao'] = 70
        self.interface_args['CLASSparams']['k_per_decade_for_pk'] = 10
        self.interface_args['CLASSparams']['z_max_pk'] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args['CLASSparams'])
        self.results.compute()

    @property
    def _interface_args(self) -> dict:
        """
        Save internal structure format of interface codes
        """
        return self.interface_args
    
    def matter_power_spectrum(self) -> np.ndarray:
        """
        Calculates the CLASS linear matter power spectrum.

        Returns:
            np.ndarray: Linear power spectrum values \(P(k)\).
        """
        Pk_l, k_values, z_values = self.results.get_pk_and_k_and_z(nonlinear=False)
        self.k = k_values
        self.z = z_values[::-1]
        self.Pk_linear = (Pk_l.T)[::-1, :]
        # To match array convention of CAMB
        return self.Pk_linear

    def growth_factor(self) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Returns:
        --------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        if hasattr(self, 'Pk_linear') and self.Pk_linear is not None:
            self.D_z_k = np.sqrt(self.Pk_linear / self.Pk_linear[0, :])
        else:
            self.matter_power_spectrum()
            self.D_z_k = np.sqrt(self.Pk_linear / self.Pk_linear[0, :])
        return self.D_z_k
    
    def f_deriv(self, k_fix=True, fixed_k=1e-3) -> np.ndarray:
        """
        Calculate the growth rate f(z,k).

        Parameters
        ----------
        k_fix : bool, optional
            If True, use a fixed k value (default is False).
        fixed_k : float, optional
            Fixed k value to use if k_fix is True (default is 1e-3).

        Returns
        -------
        tuple
            Growth rate f(z,k) and the corresponding z array.
        """
        if not hasattr(self, 'D_z_k') or self.D_z_k is None:
            self.growth_factor()
        if k_fix:
            k_array = np.full((len(self.k)), fixed_k)
        ## Generates interpolators D(z) for varying k values
        D_zk_interp = np.array([UnivariateSpline(self.z, self.D_z_k[:, ki], s=0) for ki in range(len(k_array))])
        ## Generates arrays f(z) for varying k values
        self.f_zk = np.array(
            [-(1 + self.z) / D_z(self.z) * (D_z.derivative())(self.z) for D_z in D_zk_interp]
        )
        return self.f_zk
    
    def growth_rate(self, k_fix=True) -> np.ndarray:
        """
        Calculates the growth rate f(z,k).

        Returns
        -------
        np.ndarray
            If k_fix is True, returns the growth rate as a 1D array over redshift.
            Otherwise, returns the full 2D growth rate array.
        """
        # Ensure that the growth rate has been computed
        if not hasattr(self, 'f_zk'):
            self.f_deriv(k_fix=k_fix)

        if k_fix:
            # Return the redshift part (the first row), since all k's are equal by construction
            return self.f_zk[0, :]
        else:
            # Return the full 2D array (varying over z and k)
            return self.f_zk

class CLASSNonLinearPerturbations:
    def __init__(self, background : Background, 
                 linear_perturbations: CLASSLinearPerturbations, 
                 redshifts: np.ndarray,
                 nonlinear_model: Optional[str] = None):
        r"""
        A class to define non-linear perturbations cosmology using CLASS
        and inheriting from Perturbations parent class
        """
        self.background = background
        self.linear_perturbations = linear_perturbations
        self.z = redshifts
        self.kmax = 100
        self.results = linear_perturbations.results  # Store CLASS results

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.linear_perturbations._interface_args)
        self.interface_args['CLASSparams']['nonlinear_min_k_max'] = 50
        self.interface_args['CLASSparams']['hmcode_tol_sigma'] = 1e-8
        self.interface_args['CLASSparams']['non linear'] = nonlinear_model
        self.interface_args['CLASSparams']['z_max_pk'] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args['CLASSparams'])
        self.results.compute()

    def matter_power_spectrum(self) -> np.ndarray:
        """
        Calculates the CLASS linear matter power spectrum.

        Returns:
            np.ndarray: Linear power spectrum values \(P(k)\).
        """
        Pk_nl, k_values, z_values = self.results.get_pk_and_k_and_z(nonlinear=True)
        self.k = k_values
        self.z = z_values[::-1]
        self.Pk_nonlinear = (Pk_nl.T)[::-1, :]
        return self.Pk_nonlinear
    
    def growth_factor(self) -> np.ndarray:
        """
        Calculates growth factor from linear perturbations
        This method is here to follow Protocol definition.
        """ 
        return self.linear_perturbations.growth_factor()
    
    def growth_rate(self, k_fix=True) -> np.ndarray:
        """
        Calculates growth rate from linear perturbations
        This method is here to follow Protocol definition.
        """ 
        return self.linear_perturbations.growth_rate(k_fix=k_fix)