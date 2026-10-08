"""CosmoPower-JAX emulators for the linear and nonlinear matter power spectrum.

Uses ``cosmopower_jax`` (rather than the TensorFlow-based ``cosmopower``) for fast
JAX-accelerated predictions. Covers LCDM, wCDM and w0waCDM, plus curvature and
running-spectral-index extensions, each over neutrino configurations N_mnu = 0-3.
Every cosmology exposes ``Linear`` and ``NonLinear`` inner classes; the baseline
models additionally offer halofit nonlinear variants. Each provides both the
total-matter spectrum (``matter_power_spectrum``) and the CDM+baryon spectrum
(``matter_power_spectrum_cb``, built at construction from the paired cb emulator),
along with ``sigma8``, ``fsigma8``, ``growth_factor``, ``growth_factor_cb`` and
``growth_rate``, and
validates inputs against ``CP_EMULATOR_BOUNDS``.
"""

from cloelib.cosmology.cosmology import (
    Background,
    Perturbations,
    growth_rate_on_redshifts,
)
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.auxiliary.math_utils import ensure_z_zero_included

import numpy as np
from scipy import interpolate
import os
import urllib.request
import warnings
from typing import Optional


# GitHub repository hosting the emulator files, downloaded and cached on first use.
# This is the raw-file base path; each .npz is fetched individually, not the whole repo.
GITHUB_EMULATOR_URL = (
    "https://raw.githubusercontent.com/cosmopower-organization/"
    "Euclid-DR1-matter-emulators/main/emulators"
)


def _emulator_url_for(filename: str) -> str:
    """Return the GitHub raw base URL, including subfolder, that hosts ``filename``.

    Emulators are organised by cosmology family and nonlinear prescription:
    ``<model>/hmcode``, ``<model>/halofit`` and ``extended/<variant>``. The
    subfolder is derived from the filename prefix.
    """
    if filename.startswith("halofit-"):
        subdir = f"{filename.split('-')[1]}/halofit"
    elif filename.startswith("nrun-"):
        subdir = "extended/running"
    elif filename.startswith("curvature-"):
        subdir = "extended/curvature"
    else:
        subdir = f"{filename.split('-')[0]}/hmcode"
    return f"{GITHUB_EMULATOR_URL}/{subdir}"


def emulator_data(filename: str, base_url: Optional[str] = None) -> str:
    """Download the emulator data file if it does not exist.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    base_url : str, optional
        The base URL from which to download the file. Defaults to the GitHub
        location derived from the filename.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    if base_url is None:
        base_url = _emulator_url_for(filename)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "emulator-data-jax")
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        url = f"{base_url}/{filename}"
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, file_path)

    return file_path


def load_pk_emulator(filepath: str):
    """Load a CosmoPower-JAX P(k) emulator from an .npz file.

    Uses probe='custom_log' since P(k) emulators predict log10(P(k)).

    Parameters
    ----------
    filepath : str
        Full path to the .npz emulator file.

    Returns
    -------
    CosmoPowerJAX
        The loaded emulator object.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

        return CosmoPowerJAX(probe="custom_log", filepath=filepath, verbose=False)


def load_sigma_emulator(filepath: str):
    """Load a CosmoPower-JAX sigma8/fsigma8 emulator from an .npz file.

    Uses probe='custom' since sigma8 emulators predict values directly (not log).

    Parameters
    ----------
    filepath : str
        Full path to the .npz emulator file.

    Returns
    -------
    CosmoPowerJAX
        The loaded emulator object.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

        return CosmoPowerJAX(probe="custom", filepath=filepath, verbose=False)


CP_EMULATOR_BOUNDS = {
    "ombh2": np.array([0.001, 0.1]),
    "omch2": np.array([0.05, 0.9]),
    "H0": np.array([20.0, 100.0]),
    "ns": np.array([0.6, 1.3]),
    "lnAs": np.array([1.61, 5.0]),
    "mnu": np.array([0.0, 1.0]),
    "logT_AGN": np.array([7.3, 8.5]),
    "w0": np.array([-3.0, 1.0]),
    "wa": np.array([-3.0, 3.0]),
    "w": np.array([-3.0, 0.0]),
    "omk": np.array([-0.3, 0.3]),
    "alpha_s": np.array([-0.3, 0.3]),
}


def check_emulator_bounds(params: dict) -> None:
    """Validate emulator input parameters against their training-box bounds.

    Parameters
    ----------
    params : dict
        Mapping of parameter name to its scalar value. Keys not present in
        ``CP_EMULATOR_BOUNDS`` (e.g. ``z``) are skipped.

    Raises
    ------
    ValueError
        If any parameter lies outside its emulator training range.
    """
    for key, value in params.items():
        if key not in CP_EMULATOR_BOUNDS:
            continue
        if np.prod(np.asarray(value) - CP_EMULATOR_BOUNDS[key]) > 0:
            raise ValueError(f"Parameter {key} out of emulator range.")


def _extended_pk_cb(cb_file, params, z_emu, extrap_z, ns):
    """Predict and extend the cb P(k, z) grid from a CosmoPower-JAX cb emulator.

    Built with the same ``extend_spectra`` options as the total-matter spectrum,
    so the returned grid lands on the same ``(z, k)`` grid as ``Pk``.
    """
    emu = load_pk_emulator(emulator_data(cb_file))
    Pk = np.array(emu.predict(params))
    _, _, Pk_out = extend_spectra(
        np.asarray(emu.modes),
        z_emu,
        Pk,
        flag_range=True,
        option_wavenumber="logk2",
        option_redshift="power_law",
        extrap_z=extrap_z,
        option_cosmo="const",
        ns=ns,
    )
    return Pk_out


# cb (CDM+baryon) emulator file for each total-matter class, keyed by N_mnu.
# Used to build the ``Pk_cb`` grid and ``matter_power_spectrum_cb`` at construction.
_CB_EMU_FILES = {
    "CosmoPowerJAXw0waCDMPerturbations.Linear": {
        0: "w0wa-0mass-cb-linear.npz",
        1: "w0wa-1mass-cb-linear.npz",
        2: "w0wa-2degen-cb-linear.npz",
        3: "w0wa-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXw0waCDMPerturbations.NonLinear": {
        0: "w0wa-0mass-cb-nonlinear.npz",
        1: "w0wa-1mass-cb-nonlinear.npz",
        2: "w0wa-2degen-cb-nonlinear.npz",
        3: "w0wa-3degen-cb-nonlinear.npz",
    },
    "CosmoPowerJAXw0waCDMPerturbations.NonLinearHalofit": {
        0: "halofit-w0wa-0mass-cb-nonlinear.npz",
        1: "halofit-w0wa-1mass-cb-nonlinear.npz",
        2: "halofit-w0wa-2mass-cb-nonlinear.npz",
        3: "halofit-w0wa-3mass-cb-nonlinear.npz",
    },
    "CosmoPowerJAXwCDMPerturbations.Linear": {
        0: "wcdm-0mass-cb-linear.npz",
        1: "wcdm-1mass-cb-linear.npz",
        2: "wcdm-2degen-cb-linear.npz",
        3: "wcdm-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXwCDMPerturbations.NonLinear": {
        0: "wcdm-0mass-cb-nonlinear.npz",
        1: "wcdm-1mass-cb-nonlinear.npz",
        2: "wcdm-2degen-cb-nonlinear.npz",
        3: "wcdm-3degen-cb-nonlinear.npz",
    },
    "CosmoPowerJAXwCDMPerturbations.NonLinearHalofit": {
        0: "halofit-wcdm-0mass-cb-nonlinear.npz",
        1: "halofit-wcdm-1mass-cb-nonlinear.npz",
        2: "halofit-wcdm-2mass-cb-nonlinear.npz",
        3: "halofit-wcdm-3mass-cb-nonlinear.npz",
    },
    "CosmoPowerJAXLCDMPerturbations.Linear": {
        0: "lcdm-0mass-cb-linear.npz",
        1: "lcdm-1mass-cb-linear.npz",
        2: "lcdm-2degen-cb-linear.npz",
        3: "lcdm-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXLCDMPerturbations.NonLinear": {
        0: "lcdm-0mass-cb-nonlinear.npz",
        1: "lcdm-1mass-cb-nonlinear.npz",
        2: "lcdm-2degen-cb-nonlinear.npz",
        3: "lcdm-3degen-cb-nonlinear.npz",
    },
    "CosmoPowerJAXLCDMPerturbations.NonLinearHalofit": {
        0: "halofit-lcdm-0mass-cb-nonlinear.npz",
        1: "halofit-lcdm-1mass-cb-nonlinear.npz",
        2: "halofit-lcdm-2mass-cb-nonlinear.npz",
        3: "halofit-lcdm-3mass-cb-nonlinear.npz",
    },
    "CosmoPowerJAXLCDMCurvaturePerturbations.Linear": {
        0: "curvature-lcdm-0mass-cb-linear.npz",
        1: "curvature-lcdm-1mass-cb-linear.npz",
        3: "curvature-lcdm-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXLCDMCurvaturePerturbations.NonLinear": {
        0: "curvature-lcdm-0mass-cb-nonlinear.npz",
        1: "curvature-lcdm-1mass-cb-nonlinear.npz",
        3: "curvature-lcdm-3degen-cb-nonlinear.npz",
    },
    "CosmoPowerJAXw0waCurvaturePerturbations.Linear": {
        0: "curvature-w0wa-0mass-cb-linear.npz",
        1: "curvature-w0wa-1mass-cb-linear.npz",
        3: "curvature-w0wa-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXw0waCurvaturePerturbations.NonLinear": {
        0: "curvature-w0wa-0mass-cb-nonlinear.npz",
        1: "curvature-w0wa-1mass-cb-nonlinear.npz",
        3: "curvature-w0wa-3degen-cb-nonlinear.npz",
    },
    "CosmoPowerJAXLCDMRunningIndexPerturbations.Linear": {
        0: "nrun-lcdm-0mass-cb-linear.npz",
        1: "nrun-lcdm-1mass-cb-linear.npz",
        3: "nrun-lcdm-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXLCDMRunningIndexPerturbations.NonLinear": {
        0: "nrun-lcdm-0mass-cb-nonlinear.npz",
        1: "nrun-lcdm-1mass-cb-nonlinear.npz",
        3: "nrun-lcdm-3degen-cb-nonlinear.npz",
    },
    "CosmoPowerJAXw0waRunningIndexPerturbations.Linear": {
        0: "nrun-w0wa-0mass-cb-linear.npz",
        1: "nrun-w0wa-1mass-cb-linear.npz",
        3: "nrun-w0wa-3degen-cb-linear.npz",
    },
    "CosmoPowerJAXw0waRunningIndexPerturbations.NonLinear": {
        0: "nrun-w0wa-0mass-cb-nonlinear.npz",
        1: "nrun-w0wa-1mass-cb-nonlinear.npz",
        3: "nrun-w0wa-3degen-cb-nonlinear.npz",
    },
}


class CosmoPowerJAXw0waCDMPerturbations:
    """
    Class for w0waCDM cosmology perturbations using CosmoPower-JAX emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the w0waCDM cosmology.

        Uses CosmoPower-JAX for fast JAX-accelerated predictions.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("w0wa-0mass-linear.npz")
                cp_file_sigma = emulator_data("w0wa-0mass-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("w0wa-1mass-linear.npz")
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("w0wa-2degen-linear.npz")
                cp_file_sigma = emulator_data("w0wa-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("w0wa-3degen-linear.npz")
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)

            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]

            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            redshift_max = 5
            self.z = ensure_z_zero_included(redshifts[redshifts <= redshift_max])

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))

            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
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

            pk_int = interpolate.RectBivariateSpline(self.z, self.k, Pk_out, kx=1, ky=1)
            self.Pk_interp = pk_int
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are two massive neutrinos, with a total mass sum described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )
            else:
                return (
                    f"Cosmopower-JAX linear Pk module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """
        Emulator for the nonlinear matter power spectrum in the w0waCDM cosmology.

        This class uses a Cosmopower-JAX neural network to emulate the nonlinear power spectrum
        for a w0waCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        Nonlinear corrections are applied using the mead2020 model in CAMB, with baryonic feedback
        regulated using the `log10TAGN` parameter.
        """

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file_pk = emulator_data("w0wa-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-0mass-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data("w0wa-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file_pk = emulator_data("w0wa-2degen-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data("w0wa-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file_pk)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)

            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]

            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            redshift_max = 5
            self.z = ensure_z_zero_included(redshifts[redshifts <= redshift_max])

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))

            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
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

            pk_int = interpolate.RectBivariateSpline(self.z, self.k, Pk_out, kx=1, ky=1)
            self.Pk_interp = pk_int
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower-JAX nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are two massive neutrinos, with a total mass sum described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            else:
                return (
                    f"Cosmopower-JAX nonlinear Pk module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinearHalofit:
        """
        Nonlinear matter power spectrum in the w0waCDM cosmology using the
        halofit (Takahashi 2012) prescription.

        Dark-matter-only: there is no baryonic feedback, so ``log10TAGN`` is
        accepted for interface compatibility with the HMcode ``NonLinear`` class
        but ignored. Automatically selects the emulator from ``background.N_mnu``.
        """

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinearHalofit instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("halofit-w0wa-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-w0wa-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("halofit-w0wa-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-w0wa-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("halofit-w0wa-2mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-w0wa-2mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("halofit-w0wa-3mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-w0wa-3mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            # The sigma8/fsigma8 emulator is shared with the HMcode modules and
            # expects logT_AGN; sigma8/fsigma8 are (near) linear quantities, so we
            # evaluate them at the fiducial logT_AGN=7.6.
            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX nonlinear P(k) module (halofit) for w0waCDM cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"w0={self.background.w0}, wa={self.background.wa}. "
                "Dark-matter-only halofit (no baryonic feedback)."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]


class CosmoPowerJAXwCDMPerturbations:
    """
    Class for wCDM cosmology perturbations using CosmoPower-JAX emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the wCDM cosmology.

        This class uses a Cosmopower-JAX neural network to emulate the linear power spectrum
        for a wCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-0mass-linear.npz")
                cp_file_sigma = emulator_data("wcdm-0mass-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("wcdm-1mass-linear.npz")
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("wcdm-2degen-linear.npz")
                cp_file_sigma = emulator_data("wcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("wcdm-3degen-linear.npz")
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are two massive neutrinos, with a total mass sum described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )
            else:
                return (
                    f"Cosmopower-JAX linear Pk module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """Emulator for the nonlinear matter power spectrum in wCDM cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-0mass-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("wcdm-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("wcdm-2degen-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("wcdm-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    "Cosmopower-JAX nonlinear P(k) module for wCDM cosmology.\n"
                    "Configuration: N_mnu=0, no massive neutrinos."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu=1, 1 massive neutrino with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu=2, 2 massive neutrinos with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu=3, 3 degenerate neutrinos with mnu={self.background.mnu}eV."
                )
            else:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinearHalofit:
        """
        Nonlinear matter power spectrum in the wCDM cosmology using the halofit
        (Takahashi 2012) prescription.

        Dark-matter-only: ``log10TAGN`` is accepted for interface compatibility
        with the HMcode ``NonLinear`` class but ignored. Automatically selects the
        emulator from ``background.N_mnu``.
        """

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinearHalofit instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("halofit-wcdm-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-wcdm-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("halofit-wcdm-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-wcdm-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("halofit-wcdm-2mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-wcdm-2mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("halofit-wcdm-3mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-wcdm-3mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            # sigma8/fsigma8 emulator is shared with the HMcode modules and expects
            # logT_AGN; evaluate these (near) linear quantities at fiducial 7.6.
            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX nonlinear P(k) module (halofit) for wCDM cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, w={self.background.w0}. "
                "Dark-matter-only halofit (no baryonic feedback)."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]


class CosmoPowerJAXLCDMPerturbations:
    """Class for LCDM cosmology perturbations using CosmoPower-JAX emulators."""

    class Linear:
        """Emulator for the linear matter power spectrum in LCDM cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-0mass-linear.npz")
                cp_file_sigma = emulator_data("lcdm-0mass-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("lcdm-1mass-linear.npz")
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("lcdm-2degen-linear.npz")
                cp_file_sigma = emulator_data("lcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("lcdm-3degen-linear.npz")
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    "Cosmopower-JAX linear P(k) module for LCDM cosmology.\n"
                    "Configuration: N_mnu=0, no massive neutrinos."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX linear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=1, 1 massive neutrino with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX linear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=2, 2 massive neutrinos with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX linear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=3, 3 degenerate neutrinos with mnu={self.background.mnu}eV."
                )
            else:
                return (
                    f"Cosmopower-JAX linear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """Emulator for the nonlinear matter power spectrum in LCDM cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-0mass-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("lcdm-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("lcdm-2degen-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("lcdm-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    "Cosmopower-JAX nonlinear P(k) module for LCDM cosmology.\n"
                    "Configuration: N_mnu=0, no massive neutrinos."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=1, 1 massive neutrino with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=2, 2 massive neutrinos with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=3, 3 degenerate neutrinos with mnu={self.background.mnu}eV."
                )
            else:
                return (
                    f"Cosmopower-JAX nonlinear P(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinearHalofit:
        """
        Nonlinear matter power spectrum in the LCDM cosmology using the halofit
        (Takahashi 2012) prescription.

        Dark-matter-only: ``log10TAGN`` is accepted for interface compatibility
        with the HMcode ``NonLinear`` class but ignored. Automatically selects the
        emulator from ``background.N_mnu``.
        """

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinearHalofit instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("halofit-lcdm-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-lcdm-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("halofit-lcdm-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-lcdm-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("halofit-lcdm-2mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-lcdm-2mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("halofit-lcdm-3mass-nonlinear.npz")
                cp_file_sigma = emulator_data("halofit-lcdm-3mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            # sigma8/fsigma8 emulator is shared with the HMcode modules and expects
            # logT_AGN; evaluate these (near) linear quantities at fiducial 7.6.
            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX nonlinear P(k) module (halofit) for LCDM cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}. "
                "Dark-matter-only halofit (no baryonic feedback)."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]


class CosmoPowerJAXLCDMCurvaturePerturbations:
    """Class for LCDM+curvature cosmology perturbations using CosmoPower-JAX.

    Emulators cover spatial curvature Omega_k0.
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    class Linear:
        """linear matter power spectrum in LCDM+curvature cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("curvature-lcdm-0mass-linear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-lcdm-0mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("curvature-lcdm-1mass-linear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-lcdm-1mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("curvature-lcdm-3degen-linear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-lcdm-3degen-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "omk": self.background.Omega_k0,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for LCDM+curvature cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"omk={self.background.Omega_k0}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """nonlinear matter power spectrum in LCDM+curvature cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("curvature-lcdm-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-lcdm-0mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("curvature-lcdm-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-lcdm-1mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("curvature-lcdm-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-lcdm-3degen-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "omk": self.background.Omega_k0,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for LCDM+curvature cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"omk={self.background.Omega_k0}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]


class CosmoPowerJAXw0waCurvaturePerturbations:
    """Class for w0waCDM+curvature cosmology perturbations using CosmoPower-JAX.

    Emulators cover spatial curvature Omega_k0 together with a w0waCDM dark energy background (free w0, wa).
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    class Linear:
        """linear matter power spectrum in w0waCDM+curvature cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("curvature-w0wa-0mass-linear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-w0wa-0mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("curvature-w0wa-1mass-linear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-w0wa-1mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("curvature-w0wa-3degen-linear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-w0wa-3degen-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "omk": self.background.Omega_k0,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for w0waCDM+curvature cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"omk={self.background.Omega_k0}, "
                f"w0={self.background.w0}, wa={self.background.wa}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """nonlinear matter power spectrum in w0waCDM+curvature cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("curvature-w0wa-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-w0wa-0mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("curvature-w0wa-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-w0wa-1mass-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("curvature-w0wa-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data(
                    "curvature-w0wa-3degen-combined-s8-fs8.npz"
                )
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "omk": self.background.Omega_k0,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for w0waCDM+curvature cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"omk={self.background.Omega_k0}, "
                f"w0={self.background.w0}, wa={self.background.wa}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]


class CosmoPowerJAXLCDMRunningIndexPerturbations:
    """Class for LCDM+running spectral index cosmology perturbations using CosmoPower-JAX.

    Emulators cover running spectral index alpha_s.
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    class Linear:
        """linear matter power spectrum in LCDM+running spectral index cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("nrun-lcdm-0mass-linear.npz")
                cp_file_sigma = emulator_data("nrun-lcdm-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("nrun-lcdm-1mass-linear.npz")
                cp_file_sigma = emulator_data("nrun-lcdm-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("nrun-lcdm-3degen-linear.npz")
                cp_file_sigma = emulator_data("nrun-lcdm-3degen-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "alpha_s": self.background.alpha_s,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for LCDM+running spectral index cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"alpha_s={self.background.alpha_s}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """nonlinear matter power spectrum in LCDM+running spectral index cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("nrun-lcdm-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("nrun-lcdm-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("nrun-lcdm-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("nrun-lcdm-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("nrun-lcdm-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data("nrun-lcdm-3degen-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "alpha_s": self.background.alpha_s,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for LCDM+running spectral index cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"alpha_s={self.background.alpha_s}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]


class CosmoPowerJAXw0waRunningIndexPerturbations:
    """Class for w0waCDM+running spectral index cosmology perturbations using CosmoPower-JAX.

    Emulators cover running spectral index alpha_s together with a w0waCDM dark energy background (free w0, wa).
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    class Linear:
        """linear matter power spectrum in w0waCDM+running spectral index cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            """Initialize the Linear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("nrun-w0wa-0mass-linear.npz")
                cp_file_sigma = emulator_data("nrun-w0wa-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("nrun-w0wa-1mass-linear.npz")
                cp_file_sigma = emulator_data("nrun-w0wa-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("nrun-w0wa-3degen-linear.npz")
                cp_file_sigma = emulator_data("nrun-w0wa-3degen-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_LIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "alpha_s": self.background.alpha_s,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))

            Pk_lin = np.array(self.cp_LIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_lin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for w0waCDM+running spectral index cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"alpha_s={self.background.alpha_s}, "
                f"w0={self.background.w0}, wa={self.background.wa}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the linear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Linear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]

    class NonLinear:
        """nonlinear matter power spectrum in w0waCDM+running spectral index cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            """Initialize the NonLinear instance."""
            if background.N_mnu == 0:
                cp_file = emulator_data("nrun-w0wa-0mass-nonlinear.npz")
                cp_file_sigma = emulator_data("nrun-w0wa-0mass-combined-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("nrun-w0wa-1mass-nonlinear.npz")
                cp_file_sigma = emulator_data("nrun-w0wa-1mass-combined-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("nrun-w0wa-3degen-nonlinear.npz")
                cp_file_sigma = emulator_data("nrun-w0wa-3degen-combined-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.asarray(self.cp_NONLIN.modes)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            # Retained for downstream consumers that need the *linear* P(k) back
            # from a tracer's (nonlinear) `perturbations` - same attribute/pattern
            # as HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
            self.linearperturbations = linearperturbations

            self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "alpha_s": self.background.alpha_s,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            check_emulator_bounds(self.params)

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(self.z))
            self.params["z"] = self.z

            Pk_nonlin = np.array(self.cp_NONLIN.predict(self.params))
            k_out, z_out, Pk_out = extend_spectra(
                self.k_emu,
                self.z,
                Pk_nonlin,
                flag_range=True,
                option_wavenumber="logk2",
                option_redshift="power_law",
                extrap_z=redshifts,
                option_cosmo="const",
                ns=self.background.ns,
            )
            self.k, self.z, self.Pk = k_out, z_out, Pk_out
            self.Pk_interp = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )
            self.Pk_cb = _extended_pk_cb(
                _CB_EMU_FILES[type(self).__qualname__][background.N_mnu],
                self.params,
                self.params["z"],
                redshifts,
                self.background.ns,
            )
            self.Pk_cb_interp = interpolate.RectBivariateSpline(
                self.z, self.k, self.Pk_cb, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self) -> str:
            """Return emulator description."""
            return (
                "Cosmopower-JAX P(k) module for w0waCDM+running spectral index cosmology.\n"
                f"Configuration: N_mnu={self.background.N_mnu}, "
                f"alpha_s={self.background.alpha_s}, "
                f"w0={self.background.w0}, wa={self.background.wa}."
            )

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear matter power spectrum P(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear total-matter power spectrum.
            """
            return self.Pk_interp(zs, ks)

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum P_cb(k, z).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon (no neutrino) power spectrum.
            """
            return self.Pk_cb_interp(zs, ks)

        def growth_factor_cb(self, zs, ks) -> np.ndarray:
            """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
            """
            return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in Mpc^-1.

            Returns:
                (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
            """
            return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        def growth_rate(
            self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
        ) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
                    emulator was evaluated at. Defaults to that grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            return growth_rate_on_redshifts(
                self.params["z"], self.fsigma8 / self.sigma8, zs, ks
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0.

            Returns:
                float: The rms matter fluctuation sigma8 at z=0.
            """
            return self.sigma8[0]
