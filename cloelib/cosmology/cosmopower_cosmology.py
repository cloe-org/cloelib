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

from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.extrapolator import extend_spectra

import numpy as np
from scipy import interpolate
import os
import urllib.request

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


zenodo_path = "https://zenodo.org/records/17570978/files"
k_modes_path = emulator_data("k-modes.txt", zenodo_path)


class CosmoPowerw0waCDMPerturbations:
    """
    Class for w0waCDM cosmology perturbations using Cosmopower emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the w0waCDM cosmology with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a w0waCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
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

            cp_file = emulator_data("w0wa-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear Pk module. Computes the linear power spectrum "
                f"for an w0waCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"There are no massive neutrinos in this model. "
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

    class LinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the w0waCDM cosmology with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a w0waCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
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

            cp_file = emulator_data("w0wa-pcb-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                f"for an w0waCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"There are no massive neutrinos in this model. "
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

    class Linear_1mass:
        """
        Emulator for the linear matter power spectrum in the w0waCDM cosmology with one massive neutrino.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a w0waCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino controled with `m_nu` parameter.
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

            cp_file = emulator_data("w0wa-1mass-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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
                "mnu": np.array([0.00, 1]),
            }

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "mnu": self.background.mnu,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Parameters out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear Pk module. Computes the linear power spectrum "
                f"for an w0waCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa' 'mnu'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"Neutrinos are modeled as in Casas et al. 2023."
                f"There are is one massive neutrino, with a total mass described by the `mnu` parameter."
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

    class LinearCB_1mass:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the w0waCDM cosmology with one massive neutrino.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a w0waCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino cotroled with `m_nu` parameter.
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

            cp_file = emulator_data("w0wa-1mass-pcb-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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
                "mnu": np.array([0.00, 1]),
            }

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w0": self.background.w0,
                "wa": self.background.wa,
                "mnu": self.background.mnu,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Parameters out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                f"for an w0waCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa', 'mnu'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"Neutrinos are modeled as in Casas et al. 2023."
                f"There are is one massive neutrino, with a total mass described by the `mnu` parameter."
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


class CosmoPowerwCDMPerturbations:
    """
    Class for wCDM cosmology perturbations using Cosmopower emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the w0waCDM cosmology with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a wCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
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

            cp_file = emulator_data("wcdm-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear Pk module. Computes the linear power spectrum "
                f"for an wCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs','w','z' ] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"There are no massive neutrinos in this model. "
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

    class LinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the w0waCDM cosmology with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a wCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
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

            cp_file = emulator_data("wcdm-pcb-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                f"for an wCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'w', 'z'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"There are no massive neutrinos in this model. "
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

    class Linear_1mass:
        """
        Emulator for the linear matter power spectrum in the wCDM cosmology with one massive neutrino.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a wCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino controled with `m_nu` parameter.
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

            cp_file = emulator_data("wcdm-1mass-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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
                "mnu": np.array([0.00, 1]),
            }

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
                "mnu": self.background.mnu,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Parameters out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear Pk module. Computes the linear power spectrum "
                f"for an wCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w', 'mnu'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"Neutrinos are modeled as in Casas et al. 2023."
                f"There are is one massive neutrino, with a total mass described by the `mnu` parameter."
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

    class LinearCB_1mass:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the w0waCDM cosmology with one massive neutrino.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a wCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino cotroled with `m_nu` parameter.
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

            cp_file = emulator_data("wcdm-1mass-pcb-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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
                "mnu": np.array([0.00, 1]),
            }

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "w": self.background.w0,
                "mnu": self.background.mnu,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Parameters out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear P_cb(k) module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                f"for an wCDM cosmology, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'mnu'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"Neutrinos are modeled as in Casas et al. 2023."
                f"There are is one massive neutrino, with a total mass described by the `mnu` parameter."
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


class CosmoPowerLCDMPerturbations:
    """
    Class for LCDM cosmology perturbations using Cosmopower emulators.
    """

    class Linear:
        """
        Emulator for the linear matter power spectrum in the LCDM cosmology (w is set to -1) with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a LCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
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

            cp_file = emulator_data("lcdm-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower emulator out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear Pk module. Computes the linear power spectrum "
                f"for an LCDM cosmology with w = -1, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"There are no massive neutrinos in this model. "
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

    class LinearCB:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the LCDM cosmology (w is set to -1) with no massive neutrinos.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a LCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
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

            cp_file = emulator_data("lcdm-pcb-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Cosmopower linear Ivan out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                f"for an LCDM cosmology with w = -1, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"There are no massive neutrinos in this model. "
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

    class Linear_1mass:
        """
        Emulator for the linear matter power spectrum in the LCDM cosmology (w is set to -1) with one massive neutrino.

        This class uses a Cosmopower-trained neural network to emulate the linear power spectrum
        for a LCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino cotroled with `m_nu` parameter.
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

            cp_file = emulator_data("1mass-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

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
                "mnu": np.array([0.00, 1]),
            }

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "mnu": self.background.mnu,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Parameters out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower linear Pk module. Computes the linear power spectrum "
                f"for an LCDM cosmology with w = -1, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"Neutrinos are modeled as in Casas et al. 2023."
                f"There are is one massive neutrino, with a total mass described by the `mnu` parameter."
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

    class LinearCB_1mass:
        """
        Emulator for the cb [cold dark matter (c) + baryon (b)] linear matter power spectrum in the LCDM cosmology (w is set to -1) with one massive neutrino.

        This class uses a Cosmopower-trained neural network to emulate the cb linear power spectrum
        for a LCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino controled with `m_nu` parameter.
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

            cp_file = emulator_data("1mass-pcb-linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

            self.k_emu = np.loadtxt(k_modes_path)

            self.k_min = self.k_emu[0]
            self.k_max = self.k_emu[-1]

            self.background = background
            assert background.Omega_k0 == 0, "Non flat geometries not supported"

            redshift_max = 5
            self.z = redshifts[redshifts <= redshift_max]

            cp_bounds = {
                "ombh2": np.array([0.001, 0.1]),
                "omch2": np.array([0.05, 0.8]),
                "H0": np.array([20, 100]),
                "ns": np.array([0.6, 1.3]),
                "lnAs": np.array([1.61, 5]),
                "z": np.array([0.0, 5.0]),
                "mnu": np.array([0.00, 1]),
            }

            self.params = {
                "ombh2": self.background.Omega_b0 * self.background.h**2,
                "omch2": self.background.Omega_cdm0 * self.background.h**2,
                "H0": self.background.H0,
                "ns": self.background.ns,
                "lnAs": np.log(self.background.As * 1e10),
                "mnu": self.background.mnu,
            }

            for key in self.params.keys():
                if np.product(self.params[key] - cp_bounds[key]) > 0:
                    raise ValueError("Parameters out of range.")
                else:
                    self.params[key] = np.tile(self.params[key], len(redshifts))

            self.params["z"] = redshifts

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

        def __str__(self):
            """Return emulator description."""
            return (
                f"Cosmopower cb linear Pk module. Computes the cb [cold dark matter (c) + baryon (b)] linear power spectrum "
                f"for an LCDM cosmology with w = -1, using input cosmological parameters:\n"
                f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu'] \n"
                f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
                f"Neutrinos are modeled as in Casas et al. 2023."
                f"There are is one massive neutrino, with a total mass described by the `mnu` parameter."
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
