"""Implementation of Background and Perturbation cosmology using CLASS."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.extrapolator import extend_spectra
from scipy import interpolate

# General imports
import os
import numpy as np
import copy
from typing import Optional, Union, Sequence
import DMemu

# Cosmology imports
try:
    from classy import Class  # type: ignore
except ImportError as e:
    raise ImportError("classy could not be imported.") from e

#cosmopower import
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

class obDDMBackground:
    """A wrapper for CLASS background cosmological calculations."""

    c0 = SPEED_OF_LIGHT / 1000
    invGYR_TO_KMS_MPC = 977.792 # to convert from 1/Gyr to km/s/Mpc
 
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
        f_dcdm: float,
#        Gamma_dcdm: float,
        Gamma_times_f: float,
        N_ur: Optional[float] = None,
        use_emulator: bool = True,
    ) -> None:
        """
        Initialize the obDDMBackground instance with cosmological parameters.

        Args:
            H0 (float): Hubble parameter at z=0 in km/s/Mpc.
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): TOTAL Cold dark matter density parameter (stable + decay)
            Omega_k0 (float): Curvature density parameter.
            As (float): scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (Union[float, Sequence[float], np.ndarray]): Total neutrino mass in eV.
                Can be a single float for degenerate masses, an array (or a sequence of floats) for individual species.
            w0 (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the equation of state.
            gamma_MG (float): Modified gravity growth parameter (not directly used in CLASS, but kept for protocol compliance).
            N_mnu (int): Number of massive neutrino species (i.e. N_ncdm in CLASS). For a degenerate
                mass case, pass mnu as the total mass and N_mnu as the number of species; _set_neutrino_masses
                will distribute mnu/N_mnu to each species.
            N_ur (Optional[float]): Effective number of ultra-relativistic species.
                If not provided, it will be inferred from N_mnu such that N_eff = 3.044.
            f_dcdm (float): fraction of the total cold dark matter that decays into dark radiation.
            Gamma_times_f (float): Decay rate of the dcdm component (in units of 1/Gyr) times f_dcdm.
            use_emulator (bool): If True (default), use the CosmoPower-JAX 1bDDM background
                and global emulators instead of CLASS for distances and sigma8/r_d.
        """
        self.H0 = H0
        self.h = self.H0 / 100
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0*(1.0 - f_dcdm) #stable component
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG  # Kept for protocol, but CLASS doesn't directly use it
        self.mnu = mnu
        self.N_mnu = N_mnu
        self._provided_N_ur = N_ur

        if np.sum(self.mnu) > 0 and self.N_mnu == 0:
            raise ValueError("If mnu is provided, N_mnu must be greater than 0.")
        if self.N_mnu > 0 and np.sum(self.mnu) == 0:
            raise ValueError("If N_mnu is provided, mnu must be greater than 0.")
            
        # set DCDM parameters
        self.f_dcdm = f_dcdm
        self.Omega_ini_dcdm = Omega_cdm0*f_dcdm 
        self.Gamma_dcdm = Gamma_times_f/f_dcdm
        self.Gamma_times_f = Gamma_times_f
        assert f_dcdm >= 0. and f_dcdm <= 1., "f is not within (0,1), chosen f is: {}".format(f_dcdm) # well-defined f

        self.use_emulator = use_emulator

        # Initialize CLASS parameters
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
        self.interface_args["CLASSparams"]["w0_fld"] = self.w0  
        self.interface_args["CLASSparams"]["wa_fld"] = self.wa  
        # To get correct perturbations for w0wa
        self.interface_args["CLASSparams"]["use_ppf"] = "yes"
        # To avoid using a cosmological constant
        self.interface_args["CLASSparams"]["Omega_Lambda"] = 0.0

        # Set neutrino parameters
        if self.N_mnu > 0:
            self.interface_args["CLASSparams"]["m_ncdm"] = self._set_neutrino_masses()
        self.interface_args["CLASSparams"]["N_ncdm"] = self.N_mnu
        self.interface_args["CLASSparams"]["N_ur"] = self.N_ur

        self.interface_args["CLASSparams"]["omega_ini_dcdm"] = (
            self.Omega_ini_dcdm * (self.h) ** 2
        )
        if (f_dcdm != 0):
            self.interface_args["CLASSparams"]["Gamma_dcdm"] = self.Gamma_dcdm*obDDMBackground.invGYR_TO_KMS_MPC

        # Fix YHe to standard BBN value to avoid interpolation failure at extreme omega_b
        self.interface_args["CLASSparams"]["YHe"] = 0.2454006

        if self.use_emulator:
            assert self.Omega_k0 == 0.0, "The 1bDDM background emulator only supports flat geometries."
            _emu_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emulator-data-jax")
            self._cp_distances = CosmoPowerJAX(
                probe="custom_log",
                filepath=os.path.join(_emu_dir, "ddm-1body-distances.npz"),
                verbose=False,
            )
            cp_global = CosmoPowerJAX(
                probe="custom",
                filepath=os.path.join(_emu_dir, "ddm-1body-global.npz"),
                verbose=False,
            )
            _, Omega_m0_emu, rdrag_emu = np.array(cp_global.predict(self._emulator_params())).squeeze()
            self._Omega_m0_emu = float(Omega_m0_emu)
            self._rdrag_emu = float(rdrag_emu)
        else:
            # Initialize CLASS
            self.results = Class()
            self.results.set(self.interface_args["CLASSparams"])
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
        if self.use_emulator:
            return 3.044
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

    def _emulator_params(self, zs: Optional[np.ndarray] = None) -> dict:
        """Build the parameter dictionary for the DDM-1body background emulators.

        Args:
            zs (Optional[np.ndarray]): Array of redshifts. If None, a single
                entry is returned (for emulators without a redshift input).

        Returns:
            dict: Parameter dictionary in the format expected by CosmoPowerJAX.
        """
        n = 1 if zs is None else len(zs)
        params = {
            "omega_b": np.full(n, self.Omega_b0 * self.h ** 2),
            "omega_cdm_tot": np.full(n, (self.Omega_cdm0 + self.Omega_ini_dcdm) * self.h ** 2),
            "h": np.full(n, self.h),
            "n_s": np.full(n, self.ns),
            "ln10^{10}A_s": np.full(n, np.log(1e10 * self.As)),
            "tau_reio": np.full(n, 0.054),
            "f_dcdm": np.full(n, self.f_dcdm),
            "Gamma_times_f": np.full(n, self.Gamma_times_f),
        }
        if zs is not None:
            params["z"] = np.asarray(zs)
        return params

    def _emulator_distances(self, zs: np.ndarray) -> np.ndarray:
        """Return (N, 3) array of [H(z), D_A(z), D_L(z)] with correct z=0 limits.

        The distances emulator is trained for z >= Z_EMU_MIN = 0.02.  Below that
        limit we linearly interpolate to the exact z=0 boundary values:
            H(0) = H0  [1/Mpc],  D_A(0) = 0 [Mpc],  D_L(0) = 0 [Mpc].
        """
        Z_EMU_MIN = 0.02
        H0_inv_Mpc = self.h * 100.0 / obDDMBackground.c0
        boundary = np.array([H0_inv_Mpc, 0.0, 0.0])

        mask_low = zs < Z_EMU_MIN
        out = np.empty((len(zs), 3))

        z_high = zs[~mask_low]
        if z_high.size > 0:
            out[~mask_low] = np.array(self._cp_distances.predict(self._emulator_params(z_high)))

        if mask_low.any():
            pred_min = np.array(self._cp_distances.predict(self._emulator_params(np.array([Z_EMU_MIN]))))
            at_min = pred_min[0]
            t = (zs[mask_low] / Z_EMU_MIN)[:, None]
            out[mask_low] = boundary + (at_min - boundary) * t

        return out

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Return the Hubble parameter as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.
            units (str): Units for the Hubble parameter ('1/Mpc' or 'km/s/Mpc').

        Returns:
            np.ndarray: Hubble parameter values at specified redshifts.
        """
        if self.use_emulator:
            zs = np.atleast_1d(zs)
            H = self._emulator_distances(zs)[:, 0]
        else:
            H = np.array([self.results.Hubble(z) for z in zs])  # CLASS returns H in 1/Mpc
        if units == "km/s/Mpc":
            return H * obDDMBackground.c0  # Convert to km/s/Mpc
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
        if self.use_emulator:
            zs = np.atleast_1d(zs)
            return self.angular_diameter_distance(zs) * (1.0 + zs)
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
        if self.use_emulator:
            zs = np.atleast_1d(zs)
            return self._emulator_distances(zs)[:, 1]
        return np.array([self.results.angular_distance(z) for z in zs])

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        zs = np.atleast_1d(zs)
        if self.use_emulator:
            if not np.all(zs == 0.0):
                raise ValueError(
                    "The 1bDDM background emulator only provides Omega_m at z=0. "
                    "Pass zs=0.0 or set use_emulator=False for z-dependent Omega_m."
                )
            return self._Omega_m0_emu if len(zs) > 1 else float(self._Omega_m0_emu)
        result = np.array([self.results.Om_m(z) for z in zs])
        return result if len(result) > 1 else result[0]

    @property
    def rdrag(self) -> float:
        """Sound horizon radius at last scattering in Mpc."""
        if self.use_emulator:
            return self._rdrag_emu
        return self.results.rs_drag()


class obDDMLinearPerturbations:
    """Class for perturbations cosmology using CLASS, inheriting from Perturbations parent class."""

    def __init__(self, background: Background, redshifts: np.ndarray, use_emulator: bool = True):
        """Initialize the obDDMLinearPerturbations instance."""
        self.background = background
        self.z = redshifts
        self.kmax = 49 #maximum k at which linear emulator is trained
        self.use_emulator = use_emulator

        if self.use_emulator == True:
            self.h = self.background.h
            self.wb = self.background.Omega_b0*self.h**2
            self.wdm = (self.background.Omega_cdm0 + self.background.Omega_ini_dcdm)*self.h**2
            self.log_As = np.log(1e10*self.background.As)
            self.ns = self.background.ns
            self.f     = self.background.f_dcdm
            self.Gamma_times_f = self.background.Gamma_times_f #in 1/Gyr
            # Load cosmopower emulator (path relative to this file, works in any install location)
            _emu_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emulator-data-jax")
            EMU_PATH = os.path.join(_emu_dir, "ddm-1body-linear.npz")
            DATA_PATH = os.path.join(_emu_dir, "small-k-modes.txt")
            cp = CosmoPowerJAX(probe="custom_log", filepath=EMU_PATH, verbose=False)
            self.k = np.loadtxt(DATA_PATH)
            # Pre-compute emulator predictions for all redshifts in self.z
            pk_emu = np.zeros((len(self.z), len(self.k)))
            for i, zi in enumerate(self.z):
                params = {
                    "omega_b":       np.array([self.wb]),
                    "omega_cdm_tot": np.array([self.wdm]),
                    "h":             np.array([self.h]),
                    "n_s":           np.array([self.ns]),
                    "ln10^{10}A_s":  np.array([self.log_As]),
                    "tau_reio":      np.array([0.054]),
                    "f_dcdm":        np.array([self.f]),
                    "Gamma_times_f": np.array([self.Gamma_times_f]),
                    "z":             np.array([zi]),
                }
                pk_emu[i, :] = np.array(cp.predict(params)).squeeze()
            # Extend k range to 500 1/Mpc to prevent Akima blow-up in AngularTwoPoint
            # at low z (z~1e-4) where Limber k >> k_max_emu ~50 1/Mpc
            k_out, z_out, Pk_out = extend_spectra(
                self.k, self.z, pk_emu,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=self.z,
                option_cosmo="const",
                ns=self.ns,
                extrap_kmax=500.0,
            )
            self.k = k_out
            self.Pk_int = interpolate.RectBivariateSpline(z_out, k_out, Pk_out, kx=1, ky=1)

            # Load global emulator for sigma8 (no z dependence)
            GLOBAL_EMU_PATH = os.path.join(_emu_dir, "ddm-1body-global.npz")
            cp_global = CosmoPowerJAX(probe="custom", filepath=GLOBAL_EMU_PATH, verbose=False)
            global_params = {
                "omega_b":       np.array([self.wb]),
                "omega_cdm_tot": np.array([self.wdm]),
                "h":             np.array([self.h]),
                "n_s":           np.array([self.ns]),
                "ln10^{10}A_s":  np.array([self.log_As]),
                "tau_reio":      np.array([0.054]),
                "f_dcdm":        np.array([self.f]),
                "Gamma_times_f": np.array([self.Gamma_times_f]),
            }
            sigma8_emu, _, _ = np.array(cp_global.predict(global_params)).squeeze()
            self._sigma8_emu = float(sigma8_emu)
        else:
            # Ensure CLASS is initialized with necessary parameters
            self.interface_args = copy.deepcopy(self.background.interface_args)
            self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
            self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
            self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
            self.interface_args["CLASSparams"]["non linear"] = "none"
            # Precision settings matching the emulator training.  
            if False:
                emulator_accuracy_settings = {
                    "YHe":                                        0.2454006,
                    "T_cmb":                                      2.7255,
                    "perturbations_sampling_stepsize":            0.05,
                    "ur_fluid_approximation":                     2,
                    "ur_fluid_trigger_tau_over_tau_k":            130.,
                    "radiation_streaming_approximation":          2,
                    "radiation_streaming_trigger_tau_over_tau_k": 240.,
                    "hyper_flat_approximation_nu":                7000.,
                    "transfer_neglect_delta_k_S_t0":              0.17,
                    "transfer_neglect_delta_k_S_t1":              0.05,
                    "transfer_neglect_delta_k_S_t2":              0.17,
                    "transfer_neglect_delta_k_S_e":               0.17,
                    "start_small_k_at_tau_c_over_tau_h":          0.0004,
                    "start_large_k_at_tau_h_over_tau_k":          0.05,
                    "tight_coupling_trigger_tau_c_over_tau_h":    0.005,
                    "tight_coupling_trigger_tau_c_over_tau_k":    0.008,
                    "start_sources_at_tau_c_over_tau_h":          0.006,
                    # Neutrino precision settings
                    "tol_ncdm_synchronous":                       1.e-5,
                    "ncdm_fluid_trigger_tau_over_tau_k":          100,
                    "ncdm_fluid_approximation":                   3,
                }
                self.interface_args["CLASSparams"].update(emulator_accuracy_settings)
                # Neutrino sector: replace deg_ncdm shorthand with 3 explicit species matching emulator training
                self.interface_args["CLASSparams"].pop("deg_ncdm", None)
                self.interface_args["CLASSparams"]["N_ur"]   = 0.00441
                self.interface_args["CLASSparams"]["N_ncdm"] = 3
                self.interface_args["CLASSparams"]["m_ncdm"] = "0.02,0.02,0.02"
                self.interface_args["CLASSparams"]["T_ncdm"] = "0.71611,0.71611,0.71611"

            self.results = Class()
            self.results.set(self.interface_args["CLASSparams"])
            self.results.compute()
            # GFA, I added this line in order to retrieve the wavenumber grid (in 1/Mpc) used by CLASS to compute Pk
            _, self.k, _ = self.results.get_pk_and_k_and_z(nonlinear=False, only_clustering_species = False, h_units=False)

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

        if self.use_emulator == True:
            self.Pk_linear = self.Pk_int(zs, ks)
        else:
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
        arr = [self.background.scale_independent_growth_factor_f(zi) for zi in self.z]  # type: ignore[union-attr]
        return np.array(arr)

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns
        -------
        float
            The sigma8 value.
        """
        if self.use_emulator:
            return self._sigma8_emu
        return self.results.sigma8()


class obDDMNonLinearPerturbations:
    """Class for non-linear perturbations cosmology using CLASS, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
        use_emulator: bool = False,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the obDDMNonLinearPerturbations instance."""
        self.background = background
        self.linearperturbations = linearperturbations
        self.z = redshifts
        self.kmax = 40
        self.use_emulator = use_emulator
        self.log10TAGN = log10TAGN if log10TAGN is not None else 7.6
        # These are only to check if the parameter is in a range where low error is expected.
        # Fit can still function well outside this range if the values are not extreme
        self.h = self.background.h
        self.wb = self.background.Omega_b0*self.h**2
        self.wdm = (self.background.Omega_cdm0 + self.background.Omega_ini_dcdm)*self.h**2
        self.wm = self.wb + self.wdm
        self.f     = self.background.f_dcdm
        self.Gamma = self.background.Gamma_dcdm #in 1/Gyr
        if (self.h < 0.6 or self.h > 0.8):
            print("You have chosen h={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(self.h))
        if(self.wb < 0.019 or self.wb > 0.026):
            print("You have chosen omega_b={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(self.wb))
        if(self.wm < 0.09 or self.wm > 0.28):
            print("You have chosen omega_m={}!\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(self.wm))
        if(self.Gamma >= 0.0316455696):
            print("You have chosen a short lifetime of {} Gyr<31.6 Gyr\n-> the fit could be unaccurate with this choice! (error might be > 10%)".format(self.Gamma**-1.))

        if not self.use_emulator:
        # CLASS params for equivalent LCDM (DDM params removed, total CDM restored)
            self.interface_args = copy.deepcopy(self.background.interface_args)
            self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
            self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
            self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
            self.interface_args["CLASSparams"].pop("omega_ini_dcdm", None)
            self.interface_args["CLASSparams"].pop("Gamma_dcdm", None)
            self.interface_args["CLASSparams"]["omega_cdm"] = self.wdm            
            self.interface_args["CLASSparams"]["non linear"] = "halofit"
            self.results = Class()
            self.results.set(self.interface_args["CLASSparams"])
            self.results.compute()
            _, self.k, _ = self.results.get_pk_and_k_and_z(nonlinear=True, only_clustering_species=False, h_units=False)
        else:
            # Emulator path: cosmopower-jax emulators for both NL and linear equiv LCDM Pk.

            from cloelib.cosmology.cosmopower_jax_cosmology import (
                emulator_data, load_pk_emulator, k_modes_path
            )

            cp_NL = load_pk_emulator(emulator_data("w0wa-3degen-nonlinear.npz"))
            k_emu = np.loadtxt(k_modes_path)

            mnu_total = self.background.mnu

            params_nl = {
                "ombh2":    np.tile(self.wb,                              len(self.z)),
                "omch2":    np.tile(self.wdm,                             len(self.z)),
                "H0":       np.tile(self.background.H0,                   len(self.z)),
                "ns":       np.tile(self.background.ns,                   len(self.z)),
                "lnAs":     np.tile(np.log(self.background.As * 1e10),    len(self.z)),
                "w0":       np.tile(self.background.w0,                   len(self.z)),
                "wa":       np.tile(self.background.wa,                   len(self.z)),
                "mnu":      np.tile(mnu_total,                            len(self.z)),
                "logT_AGN": np.tile(self.log10TAGN,                        len(self.z)),
                "z":        self.z,
            }

            Pk_nonlin = np.array(cp_NL.predict(params_nl))
            k_out_nl, z_out_nl, Pk_out_nl = extend_spectra(
                k_emu, self.z, Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=self.z,
                option_cosmo="const",
                ns=self.background.ns,
                extrap_kmax=500.0,
            )
            self.Pk_int_lcdm = interpolate.RectBivariateSpline(z_out_nl, k_out_nl, Pk_out_nl, kx=1, ky=1)

            # Linear LCDM Pk: w0wa-3degen-linear.npz emulator (same params, no logT_AGN)
            cp_LIN = load_pk_emulator(emulator_data("w0wa-3degen-linear.npz"))

            params_lin = {
                "ombh2": np.tile(self.wb,                           len(self.z)),
                "omch2": np.tile(self.wdm,                          len(self.z)),
                "H0":    np.tile(self.background.H0,                len(self.z)),
                "ns":    np.tile(self.background.ns,                len(self.z)),
                "lnAs":  np.tile(np.log(self.background.As * 1e10), len(self.z)),
                "w0":    np.tile(self.background.w0,                len(self.z)),
                "wa":    np.tile(self.background.wa,                len(self.z)),
                "mnu":   np.tile(mnu_total,                         len(self.z)),
                "z":     self.z,
            }

            Pk_lin = np.array(cp_LIN.predict(params_lin))
            k_out_lin, z_out_lin, Pk_out_lin = extend_spectra(
                k_emu, self.z, Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=self.z,
                option_cosmo="const",
                ns=self.background.ns,
                extrap_kmax=500.0,
            )
            self.Pk_lin_int_lcdm = interpolate.RectBivariateSpline(z_out_lin, k_out_lin, Pk_out_lin, kx=1, ky=1)
            self.k = k_out_nl  # extended k grid, shared by all three splines

    def eps_lin(self, z) -> float:
        """Calculate the function which describes the redshift evolution of the 1bDDM suppression, fit developed in Hubert et al. (2104.07675)
        
        Parameters
        ----------
        z: float
           redshift

        Returns
        -------
        eps_lin: float
                 "linear" part of 1bDDM suppression at given redshift
         """
        u = self.wb/0.02216
        v = self.h/0.6776
        w = self.wm/0.1412
        
        eps1 = 5.323 - 1.4644*u - 1.391*v + (-2.055 +1.329*u + 0.8672*v)*w + (0.2682 - 0.3509*u)*w*w
        eps2 = 0.9260 + (0.05735 - 0.02690*v)*w + (-0.01373 + 0.006713*v)*w*w
        eps3 = (9.553 - 0.7860*v) + (0.4884 + 0.1754*v)*w + (-0.2512 + 0.07558*v)*w*w
        
        eps_lin = self.f*eps1*((self.Gamma)**eps2)*((1./(1.+z*0.105))**eps3)
        
        return eps_lin
    
    def eps_nonlin(self, z, k) -> float:
        """Calculate the function which describes the non-linear 1bDDM suppression, fit developed in Hubert et al. (2104.07675)
        
        Parameters
        ----------
        z: float
           redshift
        k: float
           wavenumber

        Returns
        -------
        eps_nonlin: float
                   "non-linear" 1bDDM suppression at given redshift and wavenumber
         """
        a = 0.7208 + 2.027*self.Gamma + (3.431 - 0.4)*(1./(1.+z*1.1)) - 0.18
        b = 0.0120 + 2.786*self.Gamma + (0.6499 + 0.02)*(1./(1.+z*1.1)) - 0.09
        p = 1.045 + 1.225*self.Gamma + (0.2207)*(1./(1.+z*1.1)) - 0.099
        q = 0.9922 + 1.735*self.Gamma + (0.2154)*(1./(1.+z*1.1)) - 0.056
        
        correction_k =  (1.+a*(k**p))/(1.+b*(k**q))
        eps_nonlin = self.eps_lin(z)*correction_k 
        
        return eps_nonlin

    def boost_1bDDM(self, z, k) -> float:
        """Calculate the boost factor for the 1bDDM suppression, as in eq. 16 of Lesgourgues et al. (2406.18274)
        
        Parameters
        ----------
        z: float
           redshift
        k: float
           wavenumber

        Returns
        -------
        S_1bDDM: float
                boost factor for the 1bDDM suppression at given redshift and wavenumber
        """
        pk_1bDDM_lin = self.linearperturbations.matter_power_spectrum(np.array([z]), np.array([k]))[0, 0]
        if self.use_emulator:
            pk_LCDM_lin = self.Pk_lin_int_lcdm(np.array([z]), np.array([k]))[0, 0]
        else:
            pk_LCDM_lin = self.results.pk_lin(k, z)

        factor1 = pk_1bDDM_lin/pk_LCDM_lin
        factor2 = (1.0 - self.eps_nonlin(z,k)) / (1.0 - self.eps_lin(z))
        S_1bDDM = factor1*factor2
        
        return S_1bDDM


    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
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
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")

        # These are only to check if the parameter is in a range where low error is expected.
#        if(np.any(ks > 10)):
#            print("You have chosen k>10 1/Mpc; \n-> the fit extrapolation could be inacurate")
#        if(np.any(zs > 2.35)):
#            print("You have chosen z>2.35; \n-> the fit could be unaccurate with this choice!")

        # NL LCDM Pk: CLASS halofit (default) or cosmopower emulator (use_emulator=True)
        if self.use_emulator:
            pk_LCDM_nl = self.Pk_int_lcdm(zs, ks)
        else:
            pk_LCDM_nl = np.array([[self.results.pk(ki, zi) for ki in ks] for zi in zs])

        # Linear LCDM Pk: emulator spline (use_emulator=True) or CLASS (default)
        if self.use_emulator:
            pk_LCDM_lin = self.Pk_lin_int_lcdm(zs, ks)
        else:
            pk_LCDM_lin = np.array([[self.results.pk_lin(ki, zi) for ki in ks] for zi in zs])

        # Compute 1bDDM linear Pk grid in one vectorized call (critical for emulator performance)
        pk_1bDDM_lin = self.linearperturbations.matter_power_spectrum(zs, ks)

        # Vectorized boost factor: eq. 16 of Lesgourgues et al. (2406.18274)
        eps_lin    = np.array([self.eps_lin(zi) for zi in zs])                              # shape (nz,)
        eps_nonlin = np.array([[self.eps_nonlin(zi, ki) for ki in ks] for zi in zs])        # shape (nz, nk)
        boost = (pk_1bDDM_lin / pk_LCDM_lin) * (1.0 - eps_nonlin) / (1.0 - eps_lin[:, None])

        self.Pk_nonlinear = pk_LCDM_nl * boost
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
            self.linearperturbations.matter_power_spectrum(zs, ks)
            / self.linearperturbations.matter_power_spectrum(np.zeros_like(zs), ks)
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
        arr = [self.background.scale_independent_growth_factor_f(zi) for zi in self.z]  # type: ignore[union-attr]
        return np.array(arr)

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns
        -------
        float
            The sigma8 value.
        """
        return self.linearperturbations.sigma8_0()
