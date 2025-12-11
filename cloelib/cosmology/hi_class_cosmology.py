"""Implementation of Background and Perturbation cosmology using hi_class."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.units import SPEED_OF_LIGHT

# General imports
import numpy as np
import copy
from typing import Optional, Union, Sequence

# Cosmology imports
try:
    from classy import Class  # type: ignore
except ImportError as e:
    raise ImportError("classy could not be imported.") from e


class hi_classBackground:
    """A wrapper for hi_class background cosmological calculations."""

    c0 = SPEED_OF_LIGHT / 1000

    def __init__(
        self,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        mnu: Union[float, Sequence[float], np.ndarray],
        w0: float,
        wa: float,
        gamma_MG: float,
        N_mnu: int,
        N_ur: Optional[float] = None,
    ) -> None:
        """
        Initialize the hi_classBackground instance with cosmological parameters.

        Args:
            H0 (float): Hubble parameter at z=0 in km/s/Mpc.
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0 (float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (Union[float, Sequence[float], np.ndarray]): Total neutrino mass in eV.
                Can be a single float for degenerate masses, an array (or a sequence of floats) for individual species.
            w0 (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the equation of state.
            gamma_MG (float): Modified gravity growth parameter (not directly used in hi_class, but kept for protocol compliance).
            N_mnu (int): Number of massive neutrino species.
            N_ur (Optional[float]): Effective number of ultra-relativistic species.
                If not provided, it will be inferred from N_mnu such that N_eff = 3.044.
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
        self.gamma_MG = gamma_MG  # Kept for protocol, but hi_class doesn't directly use it
        self.mnu = mnu
        self.N_mnu = N_mnu
        # We can set N_ur to a default value if not provided
        self._provided_N_ur = N_ur

        if np.sum(self.mnu) > 0 and self.N_mnu == 0:
            raise ValueError("If mnu is provided, N_mnu must be greater than 0.")
        if self.N_mnu > 0 and np.sum(self.mnu) == 0:
            raise ValueError("If N_mnu is provided, mnu must be greater than 0.")

        # Initialize hi_class parameters
        self.interface_args: dict = {
            "hi_classparams": {}
        }  # Use a dictionary for hi_class parameters
        self.interface_args["hi_classparams"]["H0"] = self.H0
        self.interface_args["hi_classparams"]["omega_b"] = self.Omega_b0 * (self.h) ** 2
        self.interface_args["hi_classparams"]["omega_cdm"] = (
            self.Omega_cdm0 * (self.h) ** 2
        )
        self.interface_args["hi_classparams"]["Omega_k"] = self.Omega_k0
        self.interface_args["hi_classparams"]["n_s"] = self.ns
        self.interface_args["hi_classparams"]["A_s"] = self.As
        self.interface_args["hi_classparams"]["w0_fld"] = self.w0  # or w0
        self.interface_args["hi_classparams"]["wa_fld"] = self.wa  # or wa
        # To get correct perturbations for w0wa
        self.interface_args["hi_classparams"]["use_ppf"] = "yes"
        # To avoid using a cosmological constant
        self.interface_args["hi_classparams"]["Omega_Lambda"] = 0.0

        # Set neutrino parameters
        if self.N_mnu > 0:
            self.interface_args["hi_classparams"]["m_ncdm"] = self._set_neutrino_masses()
        self.interface_args["hi_classparams"]["N_ncdm"] = self.N_mnu
        self.interface_args["hi_classparams"]["N_ur"] = self.N_ur

        # Initialize hi_class
        self.results = Class()
        self.results.set(self.interface_args["hi_classparams"])
        self.results.compute()

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    @property
    def N_ur(self) -> float:
        """Effective number of ultra-relativistic species.

        If the user gave one, return it; otherwise infer from other parameters such that
        N_eff = 3.044 for the standard model of cosmology.
        """
        if self._provided_N_ur is not None:
            return self._provided_N_ur

        # If N_ur is not provided, we assume the standard model of cosmology
        # where N_eff = 3.044 (including photons, neutrinos, and their contributions)
        # This is a common assumption in cosmology.
        # Values are taken from the hi_class documentation.
        if self.N_mnu == 0:
            return 3.044
        elif self.N_mnu == 1:
            return 2.0308
        elif self.N_mnu == 2:
            return 1.0176
        elif self.N_mnu == 3:
            return 0.0044
        else:
            raise ValueError(
                f"Unsupported number of massive neutrino species: {self.N_mnu}. "
                "N_ur can only be inferred for 0, 1, 2, or 3 massive neutrino species."
            )

    @property
    def N_eff(self) -> float:
        """
        Return the effective number of relativistic species.

        Assumes a standard value of T_ncdm = 0.71611 K for neutrinos.
        """
        return self.results.Neff()

    def _set_neutrino_masses(self) -> str:
        """Set the neutrino masses in the hi_class parameters.

        This is a helper method to ensure that the neutrino masses are set correctly.
        """
        # neutrino parameters require more care
        if isinstance(self.mnu, float) and self.N_mnu > 1:
            # user gave a total mnu but wants to use a degenerate mass case
            per_mass = self.mnu / self.N_mnu
            m_ncdm_str = ",".join(f"{per_mass:g}" for _ in range(self.N_mnu))
            return m_ncdm_str
        elif isinstance(self.mnu, float) and self.N_mnu == 1:
            # single species case
            return f"{self.mnu:g}"
        elif isinstance(self.mnu, (np.ndarray, Sequence)):
            # user passed an explicit list/array of masses
            if len(self.mnu) != self.N_mnu:
                raise ValueError(
                    f"Expected {self.N_mnu} individual neutrino masses, "
                    f"but got {len(self.mnu)}: {self.mnu}"
                )

            m_ncdm_str = ",".join(f"{mass:g}" for mass in self.mnu)
            return m_ncdm_str
        else:
            raise TypeError("mnu must be a float, numpy.ndarray or Sequence of floats")

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Return the Hubble parameter as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.
            units (str): Units for the Hubble parameter ('1/Mpc' or 'km/s/Mpc').

        Returns:
            np.ndarray: Hubble parameter values at specified redshifts.
        """
        H = np.array([self.results.Hubble(z) for z in zs])  # hi_class returns H in 1/Mpc
        if units == "km/s/Mpc":
            return H * hi_classBackground.c0  # Convert to km/s/Mpc
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
        """Sound horizon radius at last scattering in Mpc."""
        return self.results.rs_drag()


class hi_classLinearPerturbations:
    """Class for perturbations cosmology using hi_class, inheriting from Perturbations parent class."""

    def __init__(self, background: Background, redshifts: np.ndarray):
        """Initialize the hi_classLinearPerturbation instance."""
        self.background = background
        self.z = redshifts
        self.kmax = 100
        self.results = None  # Store hi_class results

        # Ensure hi_class is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background.interface_args)
        self.interface_args["hi_classparams"]["output"] = "mPk, mTk"
        self.interface_args["hi_classparams"]["P_k_max_1/Mpc"] = self.kmax
        self.interface_args["hi_classparams"]["k_per_decade_for_bao"] = 70
        self.interface_args["hi_classparams"]["k_per_decade_for_pk"] = 10
        self.interface_args["hi_classparams"]["z_max_pk"] = np.max(self.z)
        self.interface_args["hi_classparams"]["non linear"] = "none"
        self.interface_args["hi_classparams"]["z_max_pk"] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args["hi_classparams"])
        self.results.compute()

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the hi_class linear matter power spectrum.

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
        if hubble_units or k_hunit:
            raise ValueError("This hi_class method does not yet support h-units")
        self.Pk_linear = np.array([[self.results.pk(ki, zi) for ki in ks] for zi in zs])  # type: ignore[union-attr]
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
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks)
            / self.matter_power_spectrum(np.zeros_like(zs), ks)
        )

        return D_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        np.ndarray
            Scale-independent growth rate f(z)
        """
        arr = [self.results.scale_independent_growth_factor_f(zi) for zi in self.z]  # type: ignore[union-attr]
        return np.array(arr)


class hi_classNonLinearPerturbations:
    """Class for non-linear perturbations cosmology using hi_class, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        nonlinear_model: Optional[str] = None,
    ):
        """Initialize the hi_classNonLinearPerturbation instance."""
        self.background = background
        self.z = redshifts
        self.kmax = 100

        if nonlinear_model is None:
            nonlinear_model = "none"

        # Ensure hi_class is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background.interface_args)
        self.interface_args["hi_classparams"]["output"] = "mPk, mTk"
        self.interface_args["hi_classparams"]["P_k_max_1/Mpc"] = self.kmax
        self.interface_args["hi_classparams"]["k_per_decade_for_bao"] = 70
        self.interface_args["hi_classparams"]["k_per_decade_for_pk"] = 10
        self.interface_args["hi_classparams"]["z_max_pk"] = np.max(self.z)
        self.interface_args["hi_classparams"]["nonlinear_min_k_max"] = 50
        self.interface_args["hi_classparams"]["hmcode_tol_sigma"] = 1e-8
        self.interface_args["hi_classparams"]["non linear"] = nonlinear_model
        self.interface_args["hi_classparams"]["z_max_pk"] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args["hi_classparams"])
        self.results.compute()

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the hi_class non-linear matter power spectrum.

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
        if hubble_units or k_hunit:
            raise ValueError("This hi_class method does not yet support h-units")
        self.Pk_nonlinear = np.array(
            [[self.results.pk(ki, zi) for ki in ks] for zi in zs]
        )
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
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks)
            / self.matter_power_spectrum(np.zeros_like(zs), ks)
        )

        return D_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        np.ndarray
            Scale-independent growth rate f(z)
        """
        arr = [self.results.scale_independent_growth_factor_f(zi) for zi in self.z]  # type: ignore[union-attr]
        return np.array(arr)