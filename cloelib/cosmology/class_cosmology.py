"""Implementation of Background and Perturbation cosmology using CLASS."""
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
    """A wrapper for CLASS background cosmological calculations."""

    c0 = SPEED_OF_LIGHT/1000
    def __init__(self, H0: float, Omega_b0: float, Omega_cdm0: float, Omega_k0: float,
                 As: float, ns: float, mnu: float, 
                 w0: float, wa: float, gamma_MG: float) -> None:
        """
        Initialize the CLASSBackground instance with cosmological parameters.

        Args:
            H0 (float): Hubble parameter at z=0 in km/s/Mpc.
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0 (float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (float): Sum of neutrino masses in [eV].
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
        self.mnu = mnu

        # Initialize CLASS parameters
        self.interface_args = {'CLASSparams': {}}  # Use a dictionary for CLASS parameters
        self.interface_args['CLASSparams']['H0'] = self.H0
        self.interface_args['CLASSparams']['omega_b'] = self.Omega_b0 * (self.h)**2
        self.interface_args['CLASSparams']['omega_cdm'] = self.Omega_cdm0 * (self.h)**2
        self.interface_args['CLASSparams']['Omega_k'] = self.Omega_k0
        self.interface_args['CLASSparams']['n_s'] = self.ns
        self.interface_args['CLASSparams']['m_ncdm'] = self.mnu
        self.interface_args['CLASSparams']['A_s'] = self.As
        self.interface_args['CLASSparams']['w0_fld'] = self.w0 # or w0
        self.interface_args['CLASSparams']['wa_fld'] = self.wa # or wa
        # To get correct perturbations for w0wa
        self.interface_args['CLASSparams']['use_ppf'] = "yes"
        # To avoid using a cosmological constant
        self.interface_args['CLASSparams']['Omega_Lambda'] = 0. 
        self.interface_args['CLASSparams']['N_ncdm'] = 1 

        # Initialize CLASS
        self.results = Class()
        self.results.set(self.interface_args['CLASSparams'])
        self.results.compute()

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Return the Hubble parameter as a function of redshift.

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
        Return the comoving distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Comoving distance values.
        """
        return np.array([self.results.comoving_distance(z) for z in zs])

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
        elif self.Omega_k0 > 0.0:
            y = np.sinh(np.sqrt(self.Omega_k0) * x) / np.sqrt(self.Omega_k0)
        else:
            y = np.sin(np.sqrt(-self.Omega_k0) * x) / np.sqrt(-self.Omega_k0)

        return y 

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the angular diameter distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Angular diameter distance values.
        """
        return np.array([self.results.angular_distance(z) for z in zs])

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return np.array([self.results.Om_m(z) for z in zs])
    
    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return np.array([self.results.Om_b(z) for z in zs])

    @property
    def rdrag(self) -> float:
        """
        Sound horizon radius at last scattering.
        """
        return self.results.rs_drag()


class CLASSLinearPerturbations:
    """Class for perturbations cosmology using CLASS, inheriting from Perturbations parent class."""

    def __init__(self, background : Background, redshifts: np.ndarray):
        """Initialize the CLASSLinearPerturbation instance."""
        self.background = background
        self.z = redshifts
        self.kmax = 100
        self.results = None  # Store CLASS results

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background._interface_args)
        self.interface_args['CLASSparams']['output'] = 'mPk, mTk'
        self.interface_args['CLASSparams']['P_k_max_1/Mpc'] = self.kmax
        self.interface_args['CLASSparams']['k_per_decade_for_bao'] = 70
        self.interface_args['CLASSparams']['k_per_decade_for_pk'] = 10
        self.interface_args['CLASSparams']['z_max_pk'] = np.max(self.z)
        self.interface_args['CLASSparams']['non linear'] = "none"
        self.interface_args['CLASSparams']['z_max_pk'] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args['CLASSparams'])
        self.results.compute()

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args
    
    def matter_power_spectrum(self, zs, ks, hubble_units=False,
                              k_hunit=False) -> np.ndarray:
        """Calculate the CLASS linear matter power spectrum.
        
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
            Linear matter power spectrum at the specified scale
            and redshift
        """
        if hubble_units == True or k_hunit == True:
            raise ValueError("This CLASS method does not yet support h-units")
        self.Pk_linear = np.array([[self.results.pk(ki, zi) for ki in ks] for zi in zs])
        # To match array convention of CAMB
        return self.Pk_linear

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

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
                        self.matter_power_spectrum(np.zeros_like(zs), ks))

        return D_z_k
    
    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        np.ndarray
            Scale-independent growth rate f(z)
        """
        return [self.results.scale_independent_growth_factor_f(zi)
                for zi in self.z]

class CLASSNonLinearPerturbations:
    """Class for non-linear perturbations cosmology using CLASS, inheriting from Perturbations parent class."""

    def __init__(self, background : Background, 
                 redshifts: np.ndarray,
                 nonlinear_model: Optional[str] = None):
        """Initialize the CLASSNonLinearPerturbation instance."""
        self.background = background
        self.z = redshifts
        self.kmax = 100

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background._interface_args)
        self.interface_args['CLASSparams']['output'] = 'mPk, mTk'
        self.interface_args['CLASSparams']['P_k_max_1/Mpc'] = self.kmax
        self.interface_args['CLASSparams']['k_per_decade_for_bao'] = 70
        self.interface_args['CLASSparams']['k_per_decade_for_pk'] = 10
        self.interface_args['CLASSparams']['z_max_pk'] = np.max(self.z)
        self.interface_args['CLASSparams']['nonlinear_min_k_max'] = 50
        self.interface_args['CLASSparams']['hmcode_tol_sigma'] = 1e-8
        self.interface_args['CLASSparams']['non linear'] = nonlinear_model
        self.interface_args['CLASSparams']['z_max_pk'] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args['CLASSparams'])
        self.results.compute()

    def matter_power_spectrum(self, zs, ks, hubble_units=False,
                              k_hunit=False) -> np.ndarray:
        """Calculate the CLASS non-linear matter power spectrum.
        
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
        if hubble_units == True or k_hunit == True:
            raise ValueError("This CLASS method does not yet support h-units")
        self.Pk_nonlinear = np.array([[self.results.pk(ki, zi) for ki in ks] for zi in zs])
        # To match array convention of CAMB
        return self.Pk_nonlinear

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

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
    
    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        np.ndarray
            Scale-independent growth rate f(z)
        """
        return [self.results.scale_independent_growth_factor_f(zi)
                for zi in self.z]
