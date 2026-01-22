"""Implementation of Background and Perturbation cosmology using CAMB."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.math_utils import ensure_z_zero_included

# General imports
import numpy as np
from typing import Optional, Union, Sequence

# Cosmology imports
try:
    import camb  # type: ignore
    from camb import model  # type: ignore
except ImportError as e:
    raise ImportError("camb could not be imported.") from e


class CAMBBackground:
    """A wrapper for CAMB background cosmological calculations."""

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
        Initialize the CAMBBackground instance with cosmological parameters.

        Args:
            H0 (float): Hubble parameter in [km/s/Mpc].
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0(float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (Union[float, Sequence[float], np.ndarray]): Total neutrino mass in eV.
                Can be a single float for degenerate masses, an array (or a sequence of floats) for individual species.
            w0 (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the dark energy equation of state.
            gamma_MG (float): Modified gravity growth parameter.
            N_mnu (int): Number of massive neutrino species.
            N_ur (Optional[float]): Extra number of ultra-relativistic species.
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
        self.gamma_MG = gamma_MG
        self.mnu = mnu
        self.N_mnu = N_mnu
        # We can set N_ur to a default value if not provided
        self._provided_N_ur = N_ur

        # camb does not accept mnu as an array, so we need to handle it
        if isinstance(self.mnu, Sequence) or isinstance(self.mnu, np.ndarray):
            mnu_arg = float(np.sum(self.mnu))
        else:
            mnu_arg = float(self.mnu)

        if mnu_arg > 0 and self.N_mnu == 0:
            raise ValueError("If mnu is provided, N_mnu must be greater than 0.")
        if self.N_mnu > 0 and np.sum(self.mnu) == 0:
            raise ValueError("If N_mnu is provided, mnu must be greater than 0.")

        # Initialize CAMB parameters
        self.interface_args: dict = {"CAMBparams": camb.CAMBparams()}

        self.interface_args["CAMBparams"].set_cosmology(
            H0=self.H0,
            ombh2=self.Omega_b0 * (self.h) ** 2,
            omch2=self.Omega_cdm0 * (self.h) ** 2,
            omk=self.Omega_k0,
            mnu=mnu_arg,
            num_massive_neutrinos=self.N_mnu,
        )

        # setting the neutrino parameters
        self.interface_args["CAMBparams"].share_delta_neff = True
        self._set_neutrino_parameters()
        self.interface_args["CAMBparams"].num_nu_massless = self.N_eff - self.N_mnu

        # Set initial conditions and dark energy
        self.interface_args["CAMBparams"].set_dark_energy(
            w=self.w0, wa=self.wa, dark_energy_model="ppf"
        )
        self.interface_args["CAMBparams"].InitPower.set_params(As=self.As, ns=self.ns)

        # Call CAMB to compute the background
        self.results = camb.get_background(self.interface_args["CAMBparams"])

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
        # Values are taken from the CLASS documentation.
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

        Assumes a standard value of T_ncdm = 0.71611 for neutrinos.
        """
        T_ncdm = 0.71611  # Standard value for neutrino temperature in K
        return self.N_ur + self.N_mnu * np.power(T_ncdm, 4.0) * np.power(
            4.0 / 11, -4.0 / 3
        )

    def _set_neutrino_parameters(self) -> None:
        """Set the neutrino mass parameters in the CAMB interface arguments.

        This method handles both degenerate and non-degenerate neutrino mass cases.
        If the non-degenerate case is used (mnu is an array),
        it will set accurate_massive_neutrinos = True.
        """
        if isinstance(self.mnu, float) and self.N_mnu >= 1:
            # user gave a total mnu but wants to use a degenerate mass case
            self.interface_args["CAMBparams"].nu_mass_eigenstates = self.N_mnu
            mass_fraction = 1.0 / self.N_mnu
            self.interface_args["CAMBparams"].nu_mass_fractions = [
                mass_fraction
            ] * self.N_mnu
            self.interface_args["CAMBparams"].nu_mass_degeneracies = [1.0] * self.N_mnu
            self.interface_args["CAMBparams"].nu_mass_numbers = [1] * self.N_mnu
        elif isinstance(self.mnu, (np.ndarray, Sequence)):
            # non-degenerate case
            sum_mnu = np.sum(self.mnu)
            if len(self.mnu) != self.N_mnu:
                raise ValueError(
                    f"Expected {self.N_mnu} individual neutrino masses, "
                    f"but got {len(self.mnu)}: {self.mnu}"
                )
            self.interface_args["CAMBparams"].nu_mass_eigenstates = self.N_mnu
            self.interface_args["CAMBparams"].Transfer.accurate_massive_neutrinos = True
            self.interface_args["CAMBparams"].nu_mass_fractions = [
                mass / sum_mnu for mass in self.mnu
            ]
            self.interface_args["CAMBparams"].nu_mass_degeneracies = [1.0] * self.N_mnu
            self.interface_args["CAMBparams"].nu_mass_numbers = [1] * self.N_mnu
        elif isinstance(self.mnu, float) and self.N_mnu == 0:
            # no neutrinos, set to zero
            self.interface_args["CAMBparams"].nu_mass_eigenstates = 0
            self.interface_args["CAMBparams"].nu_mass_fractions = []
            self.interface_args["CAMBparams"].nu_mass_degeneracies = []
            self.interface_args["CAMBparams"].nu_mass_numbers = []
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
        if units == "1/Mpc":
            return self.results.h_of_z(zs)
        if units == "km/s/Mpc":
            return self.results.hubble_parameter(zs)
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
        return self.results.comoving_radial_distance(zs)

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
        return self.results.angular_diameter_distance(zs)

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the matter density as a function of redshift.

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

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Baryonic density values at specified redshifts.
        """
        return self.results.get_Omega("baryon", z=zs)

    @property
    def rdrag(self) -> float:
        """Sound horizon radius at last scattering in Mpc."""
        return self.results.get_derived_params()["rdrag"]

    @property
    def z_star(self) -> float:
        """Redshift of recombination."""
        return self.results.get_derived_params()["zstar"]


class CAMBLinearPerturbations:
    """A wrapper for CAMB linear perturbation calculations."""

    def __init__(self, background: Background, redshifts: np.ndarray) -> None:
        """
        Initialize the CAMBLinearPerturbations class with a background instance.

        Args:
            background (Background): A CAMBBackground instance.
            redshifts (np.ndarray): Array of redshifts for the calculations.
        """
        self.background = background

        # Avoid unnecessary computations
        self.background.interface_args["CAMBparams"].WantCls = False
        self.background.interface_args["CAMBparams"].DoLensing = False
        self.background.interface_args["CAMBparams"].Want_CMB = False
        self.background.interface_args["CAMBparams"].Want_CMB_lensing = False
        self.background.interface_args["CAMBparams"].Want_cl_2D_array = False
        self.background.interface_args["CAMBparams"].WantTransfer = True

        self.kmax = 300.0

        # Ensure z=0 is included for sigma8(z=0) computation and proper interpolation
        self.z = ensure_z_zero_included(redshifts)

        self.background.interface_args["CAMBparams"].set_matter_power(
            redshifts=self.z, kmax=self.kmax
        )
        self.results = camb.get_results(self.background.interface_args["CAMBparams"])
        self.k, _, self.Pk = self.results.get_linear_matter_power_spectrum(
            hubble_units=False, k_hunit=False
        )

    def matter_power_spectrum(
        self, zs: np.ndarray, ks: np.ndarray, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        r"""Compute the linear matter power spectrum.

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
        pk_values = camb.get_matter_power_interpolator(
            self.background.interface_args["CAMBparams"],
            nonlinear=False,
            extrap_kmax=self.kmax,
            hubble_units=hubble_units,
            k_hunit=k_hunit,
            var1="delta_tot",
            var2="delta_tot",
        ).P(zs, ks)
        return pk_values

    def growth_rate(self) -> np.ndarray:
        """
        Calculate growth rate.

        Returns:
            np.ndarray: growth rate.
        """
        f_z = self.results.get_fsigma8() / self.results.get_sigma8()
        # Reversing array because camb re-sorts redshifts when power spectrum is computed
        return f_z[::-1]

    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
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
            / self.matter_power_spectrum(np.array([0.0]), ks)[0]
        )

        return D_z_k

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""

        return self.results.get_sigma8().max()


class CAMBNonLinearPerturbations:
    """A wrapper for CAMB nonlinear perturbation calculations."""

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        nonlinear_model: Optional[str] = None,
        log10TAGN: Optional[float] = None,
    ) -> None:
        """
        Initialize the CAMBNonLinearPerturbations class with linear perturbation data.

        Args:
            linear_perturbations (LinearPerturbations): An instance of the LinearPerturbations class.
            redshifts (np.ndarray): Array of redshifts for the calculations.
            nonlinear_model (Optional[str]): The nonlinear model to use (e.g., "takahashi").
                Defaults to None, which uses the CAMB default model.
        """
        self.background = background
        self.kmax = 500

        # Ensure z=0 is included for sigma8(z=0) computation and proper interpolation
        self.z = ensure_z_zero_included(redshifts)

        # Configure CAMB parameters for nonlinear calculations
        self.background.interface_args["CAMBparams"].NonLinear = model.NonLinear_both

        # Avoid unnecessary computations
        self.background.interface_args["CAMBparams"].WantCls = False
        self.background.interface_args["CAMBparams"].DoLensing = False
        self.background.interface_args["CAMBparams"].Want_CMB = False
        self.background.interface_args["CAMBparams"].Want_CMB_lensing = False
        self.background.interface_args["CAMBparams"].Want_cl_2D_array = False
        self.background.interface_args["CAMBparams"].WantTransfer = True

        if nonlinear_model is not None:
            self.background.interface_args["CAMBparams"].NonLinearModel.set_params(
                halofit_version=nonlinear_model
            )
            if log10TAGN is not None:
                self.background.interface_args["CAMBparams"].NonLinearModel.set_params(
                    halofit_version=nonlinear_model, HMCode_logT_AGN=log10TAGN
                )
        else:
            self.background.interface_args["CAMBparams"].NonLinearModel.set_params()

        self.background.interface_args["CAMBparams"].set_matter_power(
            redshifts=self.z, kmax=self.kmax
        )

        # Compute nonlinear perturbations
        self.results = camb.get_results(self.background.interface_args["CAMBparams"])

        self.k, _, self.Pk = self.results.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False
        )

    def matter_power_spectrum(
        self, zs: np.ndarray, ks: np.ndarray, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        r"""Compute the nonlinear matter power spectrum.

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
            Nonlinear matter power spectrum at the specified scale
            and redshift
        """
        pk_values = self.results.get_matter_power_interpolator(
            nonlinear=True,
            extrap_kmax=self.kmax,
            hubble_units=hubble_units,
            k_hunit=k_hunit,
            var1="delta_tot",
            var2="delta_tot",
        ).P(zs, ks)
        return pk_values

    def growth_rate(self) -> np.ndarray:
        """
        Calculate growth rate.

        Returns:
            np.ndarray: growth rate.
        """
        f_z = self.results.get_fsigma8() / self.results.get_sigma8()
        # Reversing array because camb re-sorts redshifts when power spectrum is computed
        return f_z[::-1]

    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
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
            / self.matter_power_spectrum(np.array([0.0]), ks)[0]
        )
        return D_z_k

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""

        return self.results.get_sigma8().max()
