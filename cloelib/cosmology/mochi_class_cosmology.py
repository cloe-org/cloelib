"""Implementation of Background and Perturbation cosmology using MOCHI_CLASS (https://github.com/mcataneo/mochi_class_public)."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.units import SPEED_OF_LIGHT
import sys

# General imports
import numpy as np
import copy
from typing import Optional, Union, Sequence

np.set_printoptions(threshold=sys.maxsize)

# Cosmology imports (make sure it's the version of mochi_class that's being imported not the standard CLASS!)
try:
    import mochi_classy
    from mochi_classy import Class  # type: ignore

    print(f"Loaded mochi_classy from {mochi_classy.__file__}")
except ImportError as e:
    raise ImportError("mochi_classy could not be imported.") from e

####################################
############ BACKGROUND ############
####################################


class mochiCLASSBackground:
    """A wrapper for MOCHI_CLASS background cosmological calculations."""

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
        mg_stable_basis_on: bool,
        stable_MG_dict: dict,
        mg_background_model: str,
        gamma_MG: float,
        N_mnu: int,
        N_ur: Optional[float] = None,
    ) -> None:
        """
        Initialize the mochiCLASSBackground instance with cosmological parameters.

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
            gamma_MG (float): Modified gravity growth parameter (not directly used in CLASS, but kept for protocol compliance).
            mg_stable_basis_on (bool): Flag to indicate if stable basis for modified gravity is used.
            stable_MG_dict (dict) : Dictionary of stable basis parameters lna_smg, Delta_M2, D_kin, cs2 and alpha_B0 for input into mochi_class.
            mg_background_model (str) : Desired background expansion model ('lcdm', 'wowa') for modified gravity in mochi_class.
            N_mnu (int): Number of massive neutrino species.
            N_ur (Optional[float]): Effective number of ultra-relativistic species.
                If not provided, it will be inferred from N_mnu such that N_eff = 3.044.
        """
        # TO-DO: ADD MOCHI_CLASS MG PARAMS
        self.H0 = H0
        self.h = self.H0 / 100
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = (
            gamma_MG  # Kept for protocol, but mochi_CLASS doesn't directly use it
        )
        self.mnu = mnu
        self.N_mnu = N_mnu
        self.mg_stable_basis_on = mg_stable_basis_on
        self.stable_MG_dict = stable_MG_dict
        self.s = stable_MG_dict["s"]
        self.a0 = stable_MG_dict["a0"]
        self.a1 = stable_MG_dict["a1"]
        self.b = stable_MG_dict["b"]
        self.mg_background_model = mg_background_model
        # We can set N_ur to a default value if not provided
        self._provided_N_ur = N_ur

        if np.sum(self.mnu) > 0 and self.N_mnu == 0:
            raise ValueError("If mnu is provided, N_mnu must be greater than 0.")
        if self.N_mnu > 0 and np.sum(self.mnu) == 0:
            raise ValueError("If N_mnu is provided, mnu must be greater than 0.")

        # Initialize CLASS parameters (TO-DO: ADD PARAMS HERE AS WELL)
        self.interface_args: dict = {
            "CLASSparams": {}
        }  # Use a dictionary for CLASS parameters
        self.interface_args["CLASSparams"]["H0"] = self.H0
        self.interface_args["CLASSparams"]["omega_b"] = self.Omega_b0 * (self.h) ** 2
        self.interface_args["CLASSparams"]["omega_cdm"] = (
            self.Omega_cdm0 * (self.h) ** 2
        )
        self.interface_args["CLASSparams"]["Omega_k"] = self.Omega_k0
        self.interface_args["CLASSparams"]["n_s"] = self.ns
        self.interface_args["CLASSparams"]["A_s"] = self.As

        # Set neutrino parameters
        if self.N_mnu > 0:
            self.interface_args["CLASSparams"]["m_ncdm"] = self._set_neutrino_masses()
        self.interface_args["CLASSparams"]["N_ncdm"] = self.N_mnu
        self.interface_args["CLASSparams"]["N_ur"] = self.N_ur

        ########### CHECK BACKGROUND MODEL AND MG SWITCH ################
        # WITH MG
        if mg_stable_basis_on:
            # MOCHI CLASS GENERAL PARAMETERS
            file_mochiclass_params_stable_general = {  # combines common_hiclass_params and mochiclass_params from alpha_B0
                "gravity_model": "stable_params",
                "method_gr_smg": "on",
                "z_gr_smg": 99.0,
                "skip_stability_tests_smg": "yes",  # cause stable parameters
                "a_min_stability_test_smg": 1e-2,
                "skip_math_stability_smg": "no",  ######### <---------------------- ADAPT FOR TESTING
                "exp_rate_smg": 1.0,
                "pert_initial_conditions_smg": "zero",
                "pert_ic_ini_z_ref_smg": 1e10,
                "pert_ic_tolerance_smg": 2e-2,
                "pert_ic_regulator_smg": 1e-15,
                "pert_qs_ic_tolerance_test_smg": 10,
                "method_qs_smg": "automatic",
                "z_fd_qs_smg": 0.0,
                "trigger_mass_qs_smg": 1.0e2,
                "trigger_rad_qs_smg": 1.0e2,
                "eps_s_qs_smg": 0.01,
                "n_min_qs_smg": 100,
                "n_max_qs_smg": 10000,
            }
            for key, value in file_mochiclass_params_stable_general.items():
                self.interface_args["CLASSparams"][key] = value
            # LOAD MG PARAMETRISATIONS
            lna_smg = self.stable_MG_dict["lna_smg"]
            Delta_Mpl = self.stable_MG_dict["Delta_M2"]
            Dkin = self.stable_MG_dict["D_kin"]
            cs2 = self.stable_MG_dict["cs2"]
            alpha_B0 = self.stable_MG_dict["alpha_B0"]
            if isinstance(alpha_B0, np.ndarray):
                # convert array to float
                alpha_B0_str = (
                    np.array2string(alpha_B0, separator=",", precision=16)
                    .replace("\n", "")
                    .strip("[]")
                )
            elif isinstance(alpha_B0, (int, float, np.number)):
                # convert to string
                alpha_B0_str = str(alpha_B0)
            else:
                raise ValueError("alpha_B0 must be a number or a 1D array")
            mochiclass_stable_basis_dict = {
                # omega settings here assume that MG (stable basis) is loaded
                "Omega_Lambda": 0.0,
                "Omega_fld": 0.0,
                "Omega_smg": -1.0,  # fractional density scalar field today (0: no smg, negative: specify both Omega_Lambda and Omega_fld, infer Omega_smg, 0<...<1: specify both Omega_Lambda and Omega_smg, infer Omega_fld)
                "lna_smg": np.array2string(lna_smg, separator=",", precision=16)
                .replace("\n", "")
                .strip("[]"),
                "Delta_M2": np.array2string(Delta_Mpl, separator=",", precision=16)
                .replace("\n", "")
                .strip("[]"),
                "D_kin": np.array2string(Dkin, separator=",", precision=16)
                .replace("\n", "")
                .strip("[]"),
                "cs2": np.array2string(cs2, separator=",", precision=16)
                .replace("\n", "")
                .strip("[]"),
                "parameters_smg": alpha_B0_str,
            }
            # Check expansion model for MG
            if self.mg_background_model == "lcdm":
                mochiclass_stable_basis_dict["expansion_model"] = "lcdm"
            elif self.mg_background_model == "wowa":
                w0 = self.w0
                wa = self.wa
                mochiclass_stable_basis_dict["expansion_model"] = "w0wa"
                mochiclass_stable_basis_dict["expansion_smg"] = (
                    np.array2string(
                        np.array([0.5, w0, wa]), separator=",", precision=16
                    )
                    .replace("\n", "")
                    .strip("[]")
                )  # for wowa: expansion_smg = Omega_smg, w0, wa
            elif self.mg_background_model == "rho_de":
                lna_de = self.stable_MG_dict["lna_de"]
                rho_de = self.stable_MG_dict["rho_de"]
                mochiclass_stable_basis_dict["expansion_model"] = "rho_de"
                mochiclass_stable_basis_dict["expansion_smg"] = 0.5
                mochiclass_stable_basis_dict["lna_de"] = (
                    np.array2string(lna_de, separator=",", precision=16)
                    .replace("\n", "")
                    .strip("[]")
                )
                mochiclass_stable_basis_dict["de_evo"] = (
                    np.array2string(rho_de, separator=",", precision=16)
                    .replace("\n", "")
                    .strip("[]")
                )
            else:
                raise ValueError(
                    "mg_background_model must be 'lcdm', 'wowa' or 'rho_de'"
                )

        # ########## NO MG TURNED ON ##########
        else:
            # Check expansion model for MG
            if self.mg_background_model == "lcdm":
                mochiclass_stable_basis_dict = {"Omega_fld": 0.0, "Omega_smg": 0.0}
            elif self.mg_background_model == "wowa":
                mochiclass_stable_basis_dict = {
                    "Omega_Lambda": 0,
                    "Omega_scf": 0,
                    # "Omega_fld": 0.0,
                    # "Omega_smg": 0.0,
                    "use_ppf": "yes",
                    "c_gamma_over_c_fld": 0.4,
                    "fluid_equation_of_state": "CLP",
                    "w0_fld": self.w0,
                    "wa_fld": self.wa,
                    "cs2_fld": 1,
                }
            elif (
                self.mg_background_model == "rho_de"
            ):  # no MG (activate stable basis, but LCDM values)
                lna_smg = self.stable_MG_dict["lna_smg"]
                Delta_Mpl = self.stable_MG_dict["Delta_M2"]
                Dkin = self.stable_MG_dict["D_kin"]
                cs2 = self.stable_MG_dict["cs2"]
                lna_de = self.stable_MG_dict["lna_de"]
                rho_de = self.stable_MG_dict["rho_de"]
                mochiclass_stable_basis_dict = {
                    "Omega_Lambda": 0.0,
                    "Omega_fld": 0.0,
                    "Omega_smg": -1.0,
                    "expansion_model": "rho_de",
                    "expansion_smg": 0.5,
                    "lna_de": np.array2string(lna_de, separator=",", precision=16)
                    .replace("\n", "")
                    .strip("[]"),
                    "de_evo": np.array2string(rho_de, separator=",", precision=16)
                    .replace("\n", "")
                    .strip("[]"),
                    "lna_smg": np.array2string(lna_smg, separator=",", precision=16)
                    .replace("\n", "")
                    .strip("[]"),
                    "Delta_M2": np.array2string(
                        np.zeros_like(lna_smg), separator=",", precision=16
                    )
                    .replace("\n", "")
                    .strip("[]"),
                    "D_kin": np.array2string(
                        np.ones_like(lna_smg) * 1e-8, separator=",", precision=16
                    )
                    .replace("\n", "")
                    .strip("[]"),
                    "cs2": np.array2string(
                        np.ones_like(lna_smg), separator=",", precision=16
                    )
                    .replace("\n", "")
                    .strip("[]"),
                    "parameters_smg": "0.0",
                }
            else:
                raise ValueError(
                    "mg_background_model must be 'lcdm', 'wowa' or 'rho_de'"
                )

        # load the mochi_class stable basis dictionary
        for key, value in mochiclass_stable_basis_dict.items():
            self.interface_args["CLASSparams"][key] = value
        # Initialize CLASS
        self.results = Class()
        self.results.set(self.interface_args["CLASSparams"])
        try:
            self.results.compute()
        except mochi_classy._classy.CosmoComputationError as e:
            raise RuntimeError(f"Error computing CLASS results: {e}")

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

        Assumes a standard value of T_ncdm = 0.71611 K for neutrinos.
        """
        return self.results.Neff()

    def _set_neutrino_masses(self) -> str:
        """Set the neutrino masses in the CLASS parameters.

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
        H = np.array([self.results.Hubble(z) for z in zs])  # CLASS returns H in 1/Mpc
        if units == "km/s/Mpc":
            return H * mochiCLASSBackground.c0  # Convert to km/s/Mpc
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

    # TO-DO: Get alpha parameters and M?
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
        return self.results.Om_m(zs)

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return np.array([self.results.Om_b(z) for z in zs])

    def Om_smg(self, zs: np.ndarray) -> np.ndarray:
        """
        Calculate dark energy density fraction Om_smg(z)
        (exactly, the ratio of quantities defined by Class
        as index_bg_rho_smg and index_bg_rho_crit in the background module)

        requires mochi_class v3.3.3 - 12/11/2025

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns
        -------
        np.ndarray
            dark energy density fraction Om_smg(z)
        """
        arr = [self.results.Om_smg(zi) for zi in zs]
        return np.array(arr)

    def get_background(self) -> dict:
        """
        Return all background quantities

        Return a dictionary of background quantities at all times.
        The name and list of quantities in the returned dictionary are
        defined in CLASS, in background_output_titles() and
        background_output_data(). The keys of the dictionary refer to
        redshift 'z', proper time 'proper time [Gyr]', conformal time
        'conf. time [Mpc]', and many quantities such as the Hubble
        rate, distances, densities, pressures, or growth factors. For
        each key, the dictionary contains an array of values
        corresponding to each sampled value of time.

        This function works for whatever request in the 'output'
        field, and even if 'output' was not passed or left blank.

        Parameters
        ----------
        None

        Returns
        -------
        background : dict
            Dictionary of all background quantities at each time
        """
        return self.results.get_background()

    @property
    def rdrag(self) -> float:
        """Sound horizon radius at last scattering in Mpc."""
        return self.results.rs_drag()


####################################
############## LINEAR ##############
####################################


class mochiCLASSLinearPerturbations:
    """Class for perturbations cosmology using MOCHI_CLASS, inheriting from Perturbations parent class."""

    def __init__(self, background: Background, redshifts: np.ndarray, ks: np.ndarray):
        """Initialize the CLASSLinearPerturbation instance."""
        self.background = background
        self.z = redshifts
        self.k = ks
        self.kmax = ks[-1]
        self.results = None  # Store CLASS results

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background.interface_args)
        self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
        self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
        self.interface_args["CLASSparams"]["k_per_decade_for_bao"] = 70
        self.interface_args["CLASSparams"]["k_per_decade_for_pk"] = 10
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        self.interface_args["CLASSparams"]["non_linear"] = "none"
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        self.results = Class()
        self.results.set(self.interface_args["CLASSparams"])
        try:
            self.results.compute()
        except mochi_classy._classy.CosmoComputationError as e:
            raise RuntimeError(f"Error computing CLASS results: {e}")

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
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
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        # ks /= self.background.h
        self.Pk_linear = np.array(
            [[self.results.pk_lin(ki, zi) for ki in ks] for zi in zs]
        )  # type: ignore[union-attr]
        # To match array convention of CAMB
        return self.Pk_linear  # * (self.background.h) ** 3

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

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        sigma8_0 = self.results.get_current_derived_parameters(["sigma8"])["sigma8"]
        return sigma8_0


####################################
############ NONLINEAR ############
####################################


class mochiCLASSNonLinearPerturbations:
    """Class for non-linear perturbations cosmology using MOCHI_CLASS, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        ks: np.ndarray,
        nonlinear_model: Optional[str] = None,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the CLASSNonLinearPerturbation instance."""
        self.background = background
        self.k = ks
        self.z = redshifts
        self.kmax = ks[-1]

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background.interface_args)
        self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
        self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
        self.interface_args["CLASSparams"]["k_per_decade_for_bao"] = 70
        self.interface_args["CLASSparams"]["k_per_decade_for_pk"] = 10
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        self.interface_args["CLASSparams"]["nonlinear_min_k_max"] = 50
        self.interface_args["CLASSparams"]["hmcode_tol_sigma"] = 1e-8
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        if background.mg_stable_basis_on is False:
            self.interface_args["CLASSparams"]["non_linear"] = nonlinear_model
            if nonlinear_model == "hmcode" and log10TAGN is not None:
                self.interface_args["CLASSparams"]["hmcode_version"] = (
                    "2020_baryonic_feedback"
                )
                self.interface_args["CLASSparams"]["log10T_heat_hmcode"] = log10TAGN
        else:
            self.interface_args["CLASSparams"]["non_linear"] = "None"

        self.results = Class()
        self.results.set(self.interface_args["CLASSparams"])
        try:
            self.results.compute()
        except mochi_classy._classy.CosmoComputationError as e:
            raise RuntimeError(f"Error computing CLASS results: {e}")

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the MOCHI_CLASS non-linear matter power spectrum.

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
            raise ValueError("This CLASS method does not yet support h-units")
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

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """

        sigma8_0 = self.results.get_current_derived_parameters(["sigma8"])["sigma8"]
        return sigma8_0
