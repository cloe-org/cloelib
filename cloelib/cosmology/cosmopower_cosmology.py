"""
This module provides Cosmopower-based emulators for linear and nonlinear matter power spectra in various cosmological models.

Supported models include:
- w0waCDM with mass of the neutrino 0
- w0waCDM with one massive neutrino
- wCDM with mass of the neutrino 0
- wCDM with one massive neutrino
- LCDM with mass of the neutrinos 0
- LCDM with one massive neutrino
"""

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

import numpy as np
from scipy import interpolate
import os
import urllib.request
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam
tf.get_logger().setLevel("ERROR")

import cosmopower as cp  # noqa: E402


def emulator_data(filename: str, url_base: str) -> str:
    """Download the emulator data file if it does not exist.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    url_base : str
        The base URL from which to download the file.
    Returns
    -------
    str
        The path to the downloaded file.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "emulator-data")
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        url = f"{url_base.rstrip('/')}/{filename}"
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, file_path)
    else:
        pass

    if filename.endswith(".pkl"):
        file_path = file_path[:-4]
    else:
        file_path = file_path

    return file_path


zenodo_path = "https://zenodo.org/records/17643593/files"
k_modes_path = emulator_data("k-modes.txt", zenodo_path)


class CosmoPowerw0waCDMPerturbations:
    """
    Class for w0waCDM cosmology perturbations using Cosmopower emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the w0waCDM cosmology with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a w0waCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file = emulator_data("w0wa-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("w0wa-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("w0wa-1mass-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("w0wa-3degen-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

            Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )

            else:
                return (
                    f"Cosmopower linear Pk module for w0waCDM cosmology.\n"
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

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a w0waCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file = emulator_data("w0wa-pcb-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("w0wa-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data(
                    "w0wa-1mass-pcb-linear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data(
                    "w0wa-3degen-pcb-linear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported neutrino configuration: N_mnu={background.N_mnu}. "
                    f"Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

            Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""

            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )

            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )

            else:
                return (
                    f"Cosmopower linear P_cb(k) module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb linear matter power spectrum P_cb(k, z).

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

    class NonLinear:
        """
        Emulator for the nonlinear matter power spectrum in the w0waCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the nonlinear power spectrum
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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            linearperturbations : Perturbations
                Linear perturbations object (kept for API consistency with cloelib).
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.
            log10TAGN : float, optional
                Logarithm (base 10) of the AGN feedback temperature in Kelvin.
                Controls baryonic feedback effects. If None, uses default value.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file_pk = emulator_data("w0wa-nonlinear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("w0wa-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data(
                    "w0wa-1mass-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data(
                    "w0wa-3degen-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_NONLIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file_pk)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "logT_AGN": np.array([7.6, 8.5]),
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
                "logT_AGN": log10TAGN,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            Pk_nonlin = self.cp_NONLIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
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
                    f"Cosmopower nonlinear Pk module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the nonlinear matter power spectrum P(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Nonlinear matter power spectrum in (Mpc/h)^3.
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

    class NonLinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] nonlinear matter power spectrum in the w0waCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the cb nonlinear power spectrum
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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            linearperturbations : Perturbations
                Linear perturbations object (kept for API consistency with cloelib).
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.
            log10TAGN : float, optional
                Logarithm (base 10) of the AGN feedback temperature in Kelvin.
                Controls baryonic feedback effects. If None, uses default value.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file_pk = emulator_data(
                    "w0wa-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data(
                    "w0wa-1mass-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data(
                    "w0wa-3degen-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("w0wa-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported neutrino configuration: N_mnu={background.N_mnu}. "
                    f"Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_NONLIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file_pk)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "logT_AGN": np.array([7.6, 8.5]),
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
                "logT_AGN": log10TAGN,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            Pk_nonlin = self.cp_NONLIN.ten_to_predictions_np(self.params)

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

            predictions = self.cp_SIGMA.predictions_np(self.params)
            self.sigma8 = predictions[:, 0]
            self.fsigma8 = predictions[:, 1]

        def __str__(self):
            """Return emulator description."""

            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower cb nonlinear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a w0waCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
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
                    f"Cosmopower nonlinear P_cb(k) module for w0waCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb nonlinear matter power spectrum P_cb(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Nonlinear matter power spectrum in (Mpc/h)^3.
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


class CosmoPowerwCDMPerturbations:
    """
    Class for wCDM cosmology perturbations using Cosmopower emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the wCDM cosmology with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a wCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("wcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("wcdm-1mass-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("wcdm-3degen-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "w": np.array([-3.0, 0]),
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
                "w": self.background.w0,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

            Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )

            else:
                return (
                    f"Cosmopower linear Pk module for wCDM cosmology.\n"
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
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the wCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a wCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file = emulator_data("wcdm-pcb-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("wcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data(
                    "wcdm-1mass-pcb-linear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data(
                    "wcdm-3degen-pcb-linear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported neutrino configuration: N_mnu={background.N_mnu}. "
                    f"Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "w": np.array([-3.0, -0.33]),
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
                "w": self.background.w0,
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

            Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""

            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )

            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )

            else:
                return (
                    f"Cosmopower linear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb linear matter power spectrum P_cb(k, z).

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

    class NonLinear:
        """
        Emulator for the nonlinear matter power spectrum in the wCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the nonlinear power spectrum
        for a wCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            linearperturbations : Perturbations
                Linear perturbations object (kept for API consistency with cloelib).
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.
            log10TAGN : float, optional
                Logarithm (base 10) of the AGN feedback temperature in Kelvin.
                Controls baryonic feedback effects. If None, uses default value.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file_pk = emulator_data("wcdm-nonlinear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("wcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data(
                    "wcdm-1mass-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data(
                    "wcdm-3degen-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_NONLIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file_pk)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "w": np.array([-3.0, 0]),
                "z": np.array([0.0, 5.0]),
                "logT_AGN": np.array([7.6, 8.5]),
            }

            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
                "logT_AGN": log10TAGN,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            Pk_nonlin = self.cp_NONLIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            else:
                return (
                    f"Cosmopower nonlinear Pk module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the nonlinear matter power spectrum P(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Nonlinear matter power spectrum in (Mpc/h)^3.
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
            Calculate the sigma8 value..

            Returns:
            --------
            float
                The sigma8 value.
            """
            return self.sigma8[0]

    class NonLinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] nonlinear matter power spectrum in the wCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the cb nonlinear power spectrum
        for a wCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            linearperturbations : Perturbations
                Linear perturbations object (kept for API consistency with cloelib).
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.
            log10TAGN : float, optional
                Logarithm (base 10) of the AGN feedback temperature in Kelvin.
                Controls baryonic feedback effects. If None, uses default value.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file_pk = emulator_data(
                    "wcdm-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data(
                    "wcdm-1mass-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data(
                    "wcdm-3degen-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("wcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported neutrino configuration: N_mnu={background.N_mnu}. "
                    f"Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_NONLIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file_pk)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "w": np.array([-3.0, -0.33]),
                "z": np.array([0.0, 5.0]),
                "logT_AGN": np.array([7.6, 8.5]),
            }

            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
                "logT_AGN": log10TAGN,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            Pk_nonlin = self.cp_NONLIN.ten_to_predictions_np(self.params)

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

            predictions = self.cp_SIGMA.predictions_np(self.params)
            self.sigma8 = predictions[:, 0]
            self.fsigma8 = predictions[:, 1]

        def __str__(self):
            """Return emulator description."""

            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower cb nonlinear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a wCDM cosmology, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            else:
                return (
                    f"Cosmopower nonlinear P_cb(k) module for wCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb nonlinear matter power spectrum P_cb(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Nonlinear matter power spectrum in (Mpc/h)^3.
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


class CosmoPowerLCDMPerturbations:
    """
    Class for LCDM cosmology perturbations using Cosmopower emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the LCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a LCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("1mass-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("3degen-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

            Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower linear Pk module. Computes the linear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )

            else:
                return (
                    f"Cosmopower linear Pk module for LCDM cosmology.\n"
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
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the LCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a LCDM cosmology. Automatically selects the appropriate emulator based on neutrino configuration.
        """

        def __init__(self, background: Background, redshifts: np.ndarray):
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file = emulator_data("lcdm-pcb-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file = emulator_data("1mass-pcb-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file = emulator_data("3degen-pcb-linear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported neutrino configuration: N_mnu={background.N_mnu}. "
                    f"Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
            }
            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            self.sigma_params = self.params.copy()
            self.sigma_params["logT_AGN"] = np.tile(7.6, len(redshifts))

            Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""

            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter."
                )

            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
                )

            else:
                return (
                    f"Cosmopower linear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb linear matter power spectrum P_cb(k, z).

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

    class NonLinear:
        """
        Emulator for the nonlinear matter power spectrum in the LCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the nonlinear power spectrum
        for a LCDM cosmology with w = -1. Automatically selects the appropriate emulator based on neutrino configuration.
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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            linearperturbations : Perturbations
                Linear perturbations object (kept for API consistency with cloelib).
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.
            log10TAGN : float, optional
                Logarithm (base 10) of the AGN feedback temperature in Kelvin.
                Controls baryonic feedback effects. If None, uses default value.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file_pk = emulator_data("lcdm-nonlinear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data("1mass-nonlinear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data("3degen-nonlinear-spectra.pkl", zenodo_path)
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported N_mnu={background.N_mnu}. Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_NONLIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file_pk)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "z": np.array([0.0, 5.0]),
                "logT_AGN": np.array([7.6, 8.5]),
            }

            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "logT_AGN": log10TAGN,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            Pk_nonlin = self.cp_NONLIN.ten_to_predictions_np(self.params)

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

            sigma_predictions = self.cp_SIGMA.predictions_np(self.params)
            self.sigma8 = sigma_predictions[:, 0]
            self.fsigma8 = sigma_predictions[:, 1]

        def __str__(self):
            """Return emulator description."""
            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )
            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower nonlinear Pk module. Computes the nonlinear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            else:
                return (
                    f"Cosmopower nonlinear Pk module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the nonlinear matter power spectrum P(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Nonlinear matter power spectrum in (Mpc/h)^3.
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

    class NonLinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] nonlinear matter power spectrum in the LCDM cosmology.

        This class uses a Cosmopower-trained neural network to emulate the cb nonlinear power spectrum
        for a LCDM cosmology with w = -1. Automatically selects the appropriate emulator based on neutrino configuration.
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
            """
            Initialize the emulator with a given cosmological background and redshift array.

            Parameters
            ----------
            background : Background
                Background cosmology object, providing all necessary cosmological parameters.
            linearperturbations : Perturbations
                Linear perturbations object (kept for API consistency with cloelib).
            redshifts : np.ndarray
                Array of redshift values for which the power spectrum should be computed.
            log10TAGN : float, optional
                Logarithm (base 10) of the AGN feedback temperature in Kelvin.
                Controls baryonic feedback effects. If None, uses default value.

            Raises
            ------
            AssertionError
                If the geometry is not flat (Omega_k0 != 0).
            ValueError
                If any parameter lies outside the bounds supported by the emulator.
            """

            if background.N_mnu == 0:
                cp_file_pk = emulator_data(
                    "lcdm-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("lcdm-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = False
            elif background.N_mnu == 1:
                cp_file_pk = emulator_data(
                    "1mass-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("lcdm-1mass-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            elif background.N_mnu == 3:
                cp_file_pk = emulator_data(
                    "3degen-pcb-nonlinear-spectra.pkl", zenodo_path
                )
                cp_file_sigma = emulator_data("lcdm-3degen-s8-fs8.pkl", zenodo_path)
                self.has_neutrinos = True
            else:
                raise ValueError(
                    f"Unsupported neutrino configuration: N_mnu={background.N_mnu}. "
                    f"Supported values: 0 (massless), 1 (single massive), 3 (degenerate)"
                )

            self.cp_NONLIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file_pk)
            self.cp_SIGMA = cp.cosmopower_NN(
                restore=True, restore_filename=cp_file_sigma
            )

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
                "z": np.array([0.0, 5.0]),
                "logT_AGN": np.array([7.6, 8.5]),
            }

            if self.has_neutrinos:
                cp_bounds["mnu"] = np.array([0.00, 1.0])

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "logT_AGN": log10TAGN,
            }

            if self.has_neutrinos:
                self.params["mnu"] = self.background.mnu

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

            Pk_nonlin = self.cp_NONLIN.ten_to_predictions_np(self.params)

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

            predictions = self.cp_SIGMA.predictions_np(self.params)
            self.sigma8 = predictions[:, 0]
            self.fsigma8 = predictions[:, 1]

        def __str__(self):
            """Return emulator description."""

            if self.background.N_mnu == 0:
                return (
                    f"Cosmopower cb nonlinear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"There are no massive neutrinos in this model. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 1:
                return (
                    f"Cosmopower nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Casas et al. 2023. "
                    f"There is one massive neutrino, with a total mass described by the `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            elif self.background.N_mnu == 3:
                return (
                    f"Cosmopower nonlinear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] nonlinear power spectrum "
                    f"for a LCDM cosmology with w = -1, using input cosmological parameters:\n"
                    f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu', 'logT_AGN']\n"
                    f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                    f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
                    f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter. "
                    f"Nonlinear corrections are applied using the mead2020 model in CAMB, "
                    f"with baryonic feedback regulated using the `logT_AGN` parameter."
                )

            else:
                return (
                    f"Cosmopower nonlinear P_cb(k) module for LCDM cosmology.\n"
                    f"Configuration: N_mnu={self.background.N_mnu} (unsupported in __str__)"
                )

        def matter_power_spectrum(self, zs, ks):
            """Compute the cb nonlinear matter power spectrum P_cb(k, z).

            Parameters
            ----------
            zs : np.ndarray
                Redshifts at which to evaluate the power spectrum.
            ks : np.ndarray
                Wavenumbers in units of Mpc^-1.

            Returns
            -------
            np.ndarray
                Nonlinear matter power spectrum in (Mpc/h)^3.
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
