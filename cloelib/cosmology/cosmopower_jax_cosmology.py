"""
This module provides CosmoPower-JAX-based emulators for linear and nonlinear matter power spectra.

Uses cosmopower_jax instead of tensorflow-based cosmopower for faster JAX-accelerated predictions.

Supported models include:
- w0waCDM with mass of the neutrino 0
- w0waCDM with one massive neutrino
- w0waCDM with 2 massive neutrinos
- w0waCDM with three degenerate massive neutrinos
- wCDM with mass of the neutrino 0
- wCDM with one massive neutrino
- wCDM with 2 massive neutrinos
- wCDM with three massive neutrinos
- LCDM with mass of the neutrinos 0
- LCDM with one massive neutrino
- LCDM with 2 massive neutrinos
- LCDM with three massive neutrinos

- LCDM with curvature
- LCDM with running of the spectral index
"""

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

import numpy as np
from scipy import interpolate
import os
import urllib.request
import warnings
from typing import Optional


# Zenodo URL for emulator files
ZENODO_URL = "https://zenodo.org/records/19678842/files"


def emulator_data(filename: str, zenodo_url: str = None) -> str:
    """Download the emulator data file if it does not exist.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    zenodo_url : str, optional
        The base URL from which to download the file. Defaults to ZENODO_URL.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    if zenodo_url is None:
        zenodo_url = ZENODO_URL

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "emulator-data-jax")
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        url = f"{zenodo_url}/{filename}"
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


# Download k-modes files at module load time
k_modes_path = emulator_data("k-modes.txt", ZENODO_URL)
k_modes_curvature_path = emulator_data("curvature-kmodes.txt", ZENODO_URL)


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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object.
            redshifts : np.ndarray
                Array of redshift values.
            """
            if background.N_mnu == 0:
                cp_file = emulator_data("w0wa-linear.npz")
                cp_file_sigma = emulator_data("w0wa-s8-fs8.npz")
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

            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]

            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            redshift_max = 5
            self.z = redshifts[redshifts <= redshift_max]

            cp_bounds = {
                "ombh2": np.array([0.001, 0.1]),
                "omch2": np.array([0.05, 0.9]),
                "H0": np.array([20, 100]),
                "ns": np.array([0.6, 1.3]),
                "lnAs": np.array([1.61, 5]),
                "w0": np.array([-3.0, -0.33]),
                "wa": np.array([-3, 3]),
                "z": np.array([0.0, 5.0]),
            }
            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

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

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError(f"Parameter {key} out of emulator range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = pk_int

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
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

        def matter_power_spectrum(self, zs, ks):
            """Compute the linear matter power spectrum P(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Linear matter power spectrum in (Mpc/h)^3.
            """
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks) -> np.ndarray:
            r"""
            Calculate the growth factor for given redshifts and wavenumbers.

            .. math::
                D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
                /P_{\rm \delta\delta}(z=0, k)}\\

            and normalizes as for :math:`D(z)/D(0)`.

            Parameters:
            -----------
            zs : array_like
                Redshifts at which to calculate the growth factor.
            ks : array_like
                Wavenumbers at which to calculate the growth factor.

            Returns:
            --------
            np.ndarray
                The growth factor as a function of redshift and wavenumber.
            """
            if hasattr(self, "Pk_int") and self.Pk_int is not None:
                D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))
            return D_z_k

        def growth_rate(self) -> np.ndarray:
            """
            Calculate the growth rate f(z) = d ln D / d ln a.

            This is computed as f = fsigma8 / sigma8.

            Returns
            -------
            np.ndarray
                The growth rate as a function of redshift.
            """
            return self.fsigma8 / self.sigma8

        def sigma8_0(self) -> float:
            """
            Calculate the sigma8 value.

            Returns:
            --------
            float
                The sigma8 value.
            """
            return self.sigma8[0]

    class LinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the w0waCDM cosmology.

        This class uses a Cosmopower-JAX neural network to emulate the cb linear power spectrum
        for a w0waCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            if background.N_mnu == 0:
                cp_file = emulator_data("w0wa-cb-linear.npz")
                cp_file_sigma = emulator_data("w0wa-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("w0wa-1mass-cb-linear.npz")
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("w0wa-2degen-cb-linear.npz")
                cp_file_sigma = emulator_data("w0wa-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("w0wa-3degen-cb-linear.npz")
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)

            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]

            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            redshift_max = 5
            self.z = redshifts[redshifts <= redshift_max]

            cp_bounds = {
                "ombh2": np.array([0.001, 0.1]),
                "omch2": np.array([0.05, 0.9]),
                "H0": np.array([20, 100]),
                "ns": np.array([0.6, 1.3]),
                "lnAs": np.array([1.61, 5]),
                "w0": np.array([-3.0, -0.33]),
                "wa": np.array([-3, 3]),
                "z": np.array([0.0, 5.0]),
            }
            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

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

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError(f"Parameter {key} out of emulator range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = pk_int

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower-JAX cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are two massive neutrinos, with a total mass sum described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )
            else:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb linear matter power spectrum P_cb(k, z)."""
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Calculate the growth factor D(z, k)."""
            if hasattr(self, "Pk_int") and self.Pk_int is not None:
                D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))
            return D_z_k

        def growth_rate(self) -> np.ndarray:
            """Calculate the growth rate f(z) = fsigma8 / sigma8."""
            return self.fsigma8 / self.sigma8

        def sigma8_0(self) -> float:
            """Return sigma8 at z=0."""
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
            if background.N_mnu == 0:
                cp_file_pk = emulator_data("w0wa-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-s8-fs8.npz")
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

            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]

            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            redshift_max = 5
            self.z = redshifts[redshifts <= redshift_max]

            cp_bounds = {
                "ombh2": np.array([0.001, 0.1]),
                "omch2": np.array([0.05, 0.9]),
                "H0": np.array([20, 100]),
                "ns": np.array([0.6, 1.3]),
                "lnAs": np.array([1.61, 5]),
                "w0": np.array([-3.0, -0.33]),
                "wa": np.array([-3, 3]),
                "z": np.array([0.0, 5.0]),
                "logT_AGN": np.array([7.3, 8.5]),
            }
            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

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

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError(f"Parameter {key} out of emulator range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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
            self.Pk_int = pk_int

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
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

        def matter_power_spectrum(self, zs, ks):
            """Compute the nonlinear matter power spectrum P(k, z)."""
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Calculate the growth factor D(z, k)."""
            if hasattr(self, "Pk_int") and self.Pk_int is not None:
                D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))
            return D_z_k

        def growth_rate(self) -> np.ndarray:
            """Calculate the growth rate f(z) = fsigma8 / sigma8."""
            return self.fsigma8 / self.sigma8

        def sigma8_0(self) -> float:
            """Return sigma8 at z=0."""
            return self.sigma8[0]

    class NonLinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] nonlinear matter power spectrum in the w0waCDM cosmology.

        This class uses a Cosmopower-JAX neural network to emulate the cb nonlinear power spectrum
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
            if background.N_mnu == 0:
                cp_file = emulator_data("w0wa-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("w0wa-1mass-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("w0wa-2degen-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("w0wa-3degen-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower-JAX cb nonlinear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
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
                    f"Cosmopower-JAX nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are two massive neutrinos, with a total mass sum described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
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
                    f"Cosmopower-JAX nonlinear P_cb(k) module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb nonlinear matter power spectrum P_cb(k, z)."""
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            """Calculate the growth factor D(z, k)."""
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            """Calculate the growth rate f(z) = fsigma8 / sigma8."""
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            """Return sigma8 at z=0."""
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
            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-linear.npz")
                cp_file_sigma = emulator_data("wcdm-s8-fs8.npz")
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
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
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

        def matter_power_spectrum(self, zs, ks):
            """Compute the linear matter power spectrum P(k, z)."""
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            """Calculate the growth factor D(z, k)."""
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            """Calculate the growth rate f(z) = fsigma8 / sigma8."""
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            """Return sigma8 at z=0."""
            return self.sigma8[0]

    class LinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the wCDM cosmology.

        This class uses a Cosmopower-JAX neural network to emulate the cb linear power spectrum
        for a wCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-cb-linear.npz")
                cp_file_sigma = emulator_data("wcdm-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("wcdm-1mass-cb-linear.npz")
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("wcdm-2degen-cb-linear.npz")
                cp_file_sigma = emulator_data("wcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("wcdm-3degen-cb-linear.npz")
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower-JAX cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are two massive neutrinos, with a total mass sum described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )
            else:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb linear matter power spectrum P_cb(k, z)."""
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
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
            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("wcdm-1mass-nolinear.npz")
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
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
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

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class NonLinearCB:
        """Emulator for the cb nonlinear matter power spectrum in wCDM cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("wcdm-1mass-cb-nolinear.npz")
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("wcdm-2degen-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("wcdm-3degen-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            if self.background.N_mnu == 0:
                return (
                    "Cosmopower-JAX nonlinear P_cb(k) module for wCDM cosmology.\n"
                    "Configuration: N_mnu=0, no massive neutrinos."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu=1, 1 massive neutrino with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu=2, 2 massive neutrinos with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu=3, 3 degenerate neutrinos with mnu={self.background.mnu}eV."
                )
            else:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]


class CosmoPowerJAXLCDMPerturbations:
    """Class for LCDM cosmology perturbations using CosmoPower-JAX emulators."""

    class Linear:
        """Emulator for the linear matter power spectrum in LCDM cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-linear.npz")
                cp_file_sigma = emulator_data("lcdm-s8-fs8.npz")
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
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
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

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class LinearCB:
        """Emulator for the cb linear matter power spectrum in LCDM cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-cb-linear.npz")
                cp_file_sigma = emulator_data("lcdm-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("lcdm-1mass-cb-linear.npz")
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("lcdm-2degen-cb-linear.npz")
                cp_file_sigma = emulator_data("lcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("lcdm-3degen-cb-linear.npz")
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            if self.background.N_mnu == 0:
                return (
                    "Cosmopower-JAX linear P_cb(k) module for LCDM cosmology.\n"
                    "Configuration: N_mnu=0, no massive neutrinos."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=1, 1 massive neutrino with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=2, 2 massive neutrinos with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=3, 3 degenerate neutrinos with mnu={self.background.mnu}eV."
                )
            else:
                return (
                    f"Cosmopower-JAX linear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
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
            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-s8-fs8.npz")
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
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
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

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class NonLinearCB:
        """Emulator for the cb nonlinear matter power spectrum in LCDM cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-s8-fs8.npz")
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("lcdm-1mass-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 2:
                cp_file = emulator_data("lcdm-2degen-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-2degen-s8-fs8.npz")
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("lcdm-3degen-cb-nonlinear.npz")
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.npz")
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported: 0, 1, 2, 3"
                )

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            self.z = redshifts[redshifts <= 5]
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

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            if self.background.N_mnu == 0:
                return (
                    "Cosmopower-JAX nonlinear P_cb(k) module for LCDM cosmology.\n"
                    "Configuration: N_mnu=0, no massive neutrinos."
                )
            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=1, 1 massive neutrino with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 2:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=2, 2 massive neutrinos with mnu={self.background.mnu}eV."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu=3, 3 degenerate neutrinos with mnu={self.background.mnu}eV."
                )
            else:
                return (
                    f"Cosmopower-JAX nonlinear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]


class CosmoPowerJAXCurvaturePerturbations:
    """Class for LCDM+curvature cosmology perturbations using CosmoPower-JAX emulators.

    Supports non-zero spatial curvature (Omega_k0 != 0). Neutrino mass is fixed at
    mnu=0.06 eV during training and is not a free parameter of these emulators.
    Uses a dedicated k-mode grid (curvature-kmodes.txt) and emulator files:
    - lcdm-curvature-linear.npz
    - lcdm-curvature-nonlinear.npz
    - lcdm-curvature-s8-fs8.npz

    Emulator parameter ranges:
        ombh2    in [0.019, 0.025]
        omch2    in [0.09,  0.15]
        H0       in [60,    80]
        ns       in [0.8,   1.2]
        lnAs     in [1.6,   4.0]
        z        in [0,     5]
        logT_AGN in [7.3,   8.5]  (nonlinear and sigma8/fsigma8 emulators only)
        omk      in [-0.1,  0.1]
    """

    class Linear:
        """Emulator for the linear matter power spectrum in LCDM+curvature cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            cp_file = emulator_data("lcdm-curvature-linear.npz")
            cp_file_sigma = emulator_data("lcdm-curvature-s8-fs8.npz")

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_curvature_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "omk": self.background.Omega_k0,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX linear P(k) module for LCDM+curvature cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), Omega_k0={self.background.Omega_k0}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class NonLinear:
        """Emulator for the nonlinear matter power spectrum in LCDM+curvature cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            cp_file = emulator_data("lcdm-curvature-nonlinear.npz")
            cp_file_sigma = emulator_data("lcdm-curvature-s8-fs8.npz")

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_curvature_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
                "omk": self.background.Omega_k0,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX nonlinear P(k) module for LCDM+curvature cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), Omega_k0={self.background.Omega_k0}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class LinearCB:
        """Emulator for the cb linear matter power spectrum in LCDM+curvature cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            cp_file = emulator_data("lcdm-curvature-cb-linear.npz")
            cp_file_sigma = emulator_data("lcdm-curvature-s8-fs8.npz")

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_curvature_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "omk": self.background.Omega_k0,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX linear P_cb(k) module for LCDM+curvature cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), Omega_k0={self.background.Omega_k0}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class NonLinearCB:
        """Emulator for the cb nonlinear matter power spectrum in LCDM+curvature cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            cp_file = emulator_data("lcdm-curvature-cb-nonlinear.npz")
            cp_file_sigma = emulator_data("lcdm-curvature-s8-fs8.npz")

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_curvature_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "omk": self.background.Omega_k0,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX nonlinear P_cb(k) module for LCDM+curvature cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), Omega_k0={self.background.Omega_k0}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]


class CosmoPowerJAXRunningIndexPerturbations:
    """Class for LCDM+running spectral index cosmology perturbations using CosmoPower-JAX emulators.

    Supports a running spectral index alpha_s = d ns / d ln k. Neutrino mass is fixed at
    mnu=0.06 eV during training and is not a free parameter of these emulators.
    Uses the standard k-mode grid (k-modes.txt) and emulator files:
    - lcdm-running-linear.npz
    - lcdm-running-nonlinear.npz
    - lcdm-running-s8-fs8.npz

    Emulator parameter ranges:
        ombh2    in [0.019, 0.025]
        omch2    in [0.09,  0.15]
        H0       in [60,    80]
        ns       in [0.8,   1.2]
        lnAs     in [1.6,   4.0]
        z        in [0,     5]
        alpha_s  in [-0.1,  0.1]
        logT_AGN in [7.3,   8.5]  (nonlinear and sigma8/fsigma8 emulators only)
    """

    class Linear:
        """Emulator for the linear matter power spectrum in LCDM+running spectral index cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            cp_file = emulator_data("lcdm-nrun-linear.npz")
            cp_file_sigma = emulator_data("lcdm-nrun-s8-fs8.npz")

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "alpha_s": self.background.alpha_s,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX linear P(k) module for LCDM+running spectral index cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), alpha_s={self.background.alpha_s}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class NonLinear:
        """Emulator for the nonlinear matter power spectrum in LCDM+running spectral index cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            cp_file = emulator_data("lcdm-nrun-nonlinear.npz")
            cp_file_sigma = emulator_data("lcdm-nrun-s8-fs8.npz")

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "alpha_s": self.background.alpha_s,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX nonlinear P(k) module for LCDM+running spectral index cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), alpha_s={self.background.alpha_s}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class LinearCB:
        """Emulator for the cb linear matter power spectrum in LCDM+running spectral index cosmology."""

        def __init__(self, background: Background, redshifts: np.ndarray):
            cp_file = emulator_data("lcdm-nrun-cb-linear.npz")
            cp_file_sigma = emulator_data("lcdm-nrun-s8-fs8.npz")

            self.cp_LIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "alpha_s": self.background.alpha_s,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.sigma_params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX linear P_cb(k) module for LCDM+running spectral index cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), alpha_s={self.background.alpha_s}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]

    class NonLinearCB:
        """Emulator for the cb nonlinear matter power spectrum in LCDM+running spectral index cosmology."""

        def __init__(
            self,
            background: Background,
            linearperturbations: Perturbations,
            redshifts: np.ndarray,
            log10TAGN: Optional[float] = None,
        ):
            cp_file = emulator_data("lcdm-nrun-cb-nonlinear.npz")
            cp_file_sigma = emulator_data("lcdm-nrun-s8-fs8.npz")

            self.cp_NONLIN = load_pk_emulator(cp_file)
            self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
            self.k_emu = np.loadtxt(k_modes_path)
            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]
            self.background = background

            self.z = redshifts[redshifts <= 5]
            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "alpha_s": self.background.alpha_s,
                "logT_AGN": log10TAGN if log10TAGN is not None else 7.6,
            }

            for key in self.params.keys():
                self.params[key] = np.tile(self.params[key], len(redshifts))
            self.params["z"] = redshifts

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
            self.Pk_int = interpolate.RectBivariateSpline(
                self.z, self.k, Pk_out, kx=1, ky=1
            )

            sigma_predictions = np.array(self.cp_SIGMA.predict(self.params))
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            return (
                "Cosmopower-JAX nonlinear P_cb(k) module for LCDM+running spectral index cosmology.\n"
                f"Configuration: mnu=0.06 eV (fixed), alpha_s={self.background.alpha_s}."
            )

        def matter_power_spectrum(self, zs, ks):
            return self.Pk_int(zs, ks)

        def growth_factor(self, zs, ks):
            return np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        def growth_rate(self):
            return self.fsigma8 / self.sigma8

        def sigma8_0(self):
            return self.sigma8[0]