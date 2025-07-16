"""
This module provides Cosmopower-based emulators for linear and nonlinear matter power spectra in various cosmological models.

Supported models include:
- w0waCDM
- w0waCDM with one massive neutrino
- w0waCDM with three degenerate massive neutrinos
"""

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

import numpy as np
import warnings
from scipy import interpolate
import os
import urllib.request
from typing import Tuple, Optional
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    DATA_DIR = os.path.join(BASE_DIR, 'emulator-data')
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        url = f"{url_base.rstrip('/')}/{filename}"
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, file_path)
    else:
        pass

    if filename.endswith('.pkl'):
        file_path = file_path[:-4]
    else:
        file_path = file_path

    return file_path


zenodo_path="https://zenodo.org/records/15537463/files"
k_modes_path = emulator_data("k-modes.txt", zenodo_path)


class w0waCDM_3degen_Linear:
    """
    Emulator for the linear matter power spectrum in the w0waCDM cosmology with three degenerate massive neutrinos.

    This class uses a Cosmopower-trained neural network to emulate the linear power spectrum 
    for a w0waCDM cosmology with three degenerate neutrinos. Neutrinos are modeled as in Archidiacono et al. (2024). The total mass
    sum is described by the `mnu` parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp


            cp_file = emulator_data("w0wa+3degen_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)


        self.k_emu = np.loadtxt(k_modes_path)
        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        self.background = background         
        assert background.Omega_k0 == 0, 'Non flat geometries not supported' 

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        cp_bounds = {'ombh2': np.array([0.015, 0.035]),
                     'omch2': np.array([0.05,  0.2]),
                     'H0': np.array([60, 80]),
                     'ns': np.array([0.9, 1.05]),
                     'lnAs': np.array([2.5, 3.5]),
                     'w0': np.array([-1.0, 0.0]),
                     'wa': np.array([-0.5, 0.5]),
                     'z': np.array([0.0, 5.0]),
                     'mnu': np.array([0.00, 0.09]),
          }
        

        self.params = {'ombh2': self.background.Omega_b0 * self.background.h**2,
                  'omch2': self.background.Omega_cdm0 * self.background.h**2,
                    'H0': self.background.H0,
                    'ns': self.background.ns,
                    'lnAs': np.log(self.background.As * 1e10),
                    'w0': self.background.w0,
                    'wa': self.background.wa,
                    'mnu': self.background.mnu,
                  }
        
        for key in self.params.keys():
            if np.product(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Parameters out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))
        
        self.params['z'] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(self.k_emu, self.z , Pk_lin,flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)
        
        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out
        
        pk_int = interpolate.RectBivariateSpline( self.z , self.k, Pk_out, kx=1, ky=1)
        self.Pk_int = pk_int
        



    def __str__(self):
        """Return emulator description."""
        return (
            f"Cosmopower linear Pk module. Computes the linear power spectrum "
            f"for an w0waCDM cosmology, using input cosmological parameters:\n"
            f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa' 'mnu'] \n"
            f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
            f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
            f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
        )
    
    def matter_power_spectrum(self,zs,ks):
        """
        Compute the linear matter power spectrum P(k, z).

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
        if hasattr(self, 'Pk_int') and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k


class w0waCDM_1mass_Linear:
    """
    Emulator for the linear matter power spectrum in the w0waCDM cosmology with one massive neutrino.

    This class uses a Cosmopower-trained neural network to emulate the linear power spectrum 
    for a w0waCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino cotroled with `m_nu` parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp


            cp_file = emulator_data("w0wa+1mass_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)


        self.k_emu = np.loadtxt(k_modes_path)


        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        self.background = background         
        assert background.Omega_k0 == 0, 'Non flat geometries not supported'

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        cp_bounds = {'ombh2': np.array([0.015, 0.035]),
                     'omch2': np.array([0.05,  0.2]),
                     'H0': np.array([60, 80]),
                     'ns': np.array([0.9, 1.05]),
                     'lnAs': np.array([2.5, 3.5]),
                     'w0': np.array([-1.0, 0.0]),
                     'wa': np.array([-0.5, 0.5]),
                     'z': np.array([0.0, 5.0]),
                     'mnu': np.array([0.00, 0.09]),
          }
        

        self.params = {'ombh2': self.background.Omega_b0 * self.background.h**2,
                  'omch2': self.background.Omega_cdm0 * self.background.h**2,
                    'H0': self.background.H0,
                    'ns': self.background.ns,
                    'lnAs': np.log(self.background.As * 1e10),
                    'w0': self.background.w0,
                    'wa': self.background.wa,
                    'mnu': self.background.mnu,
                  }
        
        for key in self.params.keys():
            if np.product(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Parameters out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))
        
        self.params['z'] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(self.k_emu, self.z , Pk_lin,flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)
        
        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out
        
        pk_int = interpolate.RectBivariateSpline( self.z , self.k, Pk_out, kx=1, ky=1)
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

    def matter_power_spectrum(self,zs,ks):
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
        if hasattr(self, 'Pk_int') and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k



class w0waCDM_Linear:
    """
    Emulator for the linear matter power spectrum in the w0waCDM cosmology with no massive neutrinos.

    This class uses a Cosmopower-trained neural network to emulate the linear power spectrum 
    for a w0waCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp


            cp_file = emulator_data("w0wa_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)


        self.k_emu =np.loadtxt(k_modes_path)


        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        self.background = background         
        assert background.Omega_k0 == 0, 'Non flat geometries not supported'

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        cp_bounds = {'ombh2': np.array([0.009, 0.058]),
                     'omch2': np.array([0.02,  0.69]),
                     'H0': np.array([55., 91.]),
                     'ns': np.array([0.87, 1.07]),
                     'lnAs': np.array([0.5, 5]),
                     'w0': np.array([-1.0, 0]),
                     'wa': np.array([-0.5, 0.5]),
                     'z': np.array([0.0, 5.0]),
          }
        

        self.params = {'ombh2': self.background.Omega_b0 * self.background.h**2,
                  'omch2': self.background.Omega_cdm0 * self.background.h**2,
                    'H0': self.background.H0,
                    'ns': self.background.ns,
                    'lnAs': np.log(self.background.As * 1e10),
                    'w0': self.background.w0,
                    'wa': self.background.wa,
                  }
        
        for key in self.params.keys():
            if np.product(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Cosmopower linear Ivan out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))
        
        self.params['z'] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(self.k_emu, self.z , Pk_lin,flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)
        
        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out
        
        pk_int = interpolate.RectBivariateSpline( self.z , self.k, Pk_out, kx=1, ky=1)
        self.Pk_int = pk_int
        



    def __str__(self):
        """Return emulator description."""
        return (
            f"Cosmopower linear Pk module. Computes the linear power spectrum "
            f"for an w0waCDM cosmology, using input cosmological parameters:\n"
            f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'w0', 'wa' 'mnu'] \n"
            f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
            f"There are no massive neutrinos in this model. " 
        )
    
    def matter_power_spectrum(self,zs,ks):
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
        if hasattr(self, 'Pk_int') and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k
    


class LCDM_3degen_Linear:
    """
    Emulator for the linear matter power spectrum in the LCDM cosmology (w is set to -1) with three degenerate massive neutrinos.

    This class uses a Cosmopower-trained neural network to emulate the linear power spectrum 
    for a LCDMCDM cosmology with three degenerate neutrinos. Neutrinos are modeled as in Archidiacono et al. (2024). The total mass
    sum is described by the `mnu` parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp


            cp_file = emulator_data("3degen_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)


        self.k_emu = np.loadtxt(k_modes_path)
        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        self.background = background         
        assert background.Omega_k0 == 0, 'Non flat geometries not supported' 

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        cp_bounds = {'ombh2': np.array([0.015, 0.035]),
                     'omch2': np.array([0.05,  0.2]),
                     'H0': np.array([60, 80]),
                     'ns': np.array([0.9, 1.05]),
                     'lnAs': np.array([2.5, 3.5]),
                     'z': np.array([0.0, 5.0]),
                     'mnu': np.array([0.00, 0.09]),
          }
        

        self.params = {'ombh2': self.background.Omega_b0 * self.background.h**2,
                  'omch2': self.background.Omega_cdm0 * self.background.h**2,
                    'H0': self.background.H0,
                    'ns': self.background.ns,
                    'lnAs': np.log(self.background.As * 1e10),
                    'mnu': self.background.mnu,
                  }
        
        for key in self.params.keys():
            if np.product(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Parameters out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))
        
        self.params['z'] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(self.k_emu, self.z , Pk_lin,flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)
        
        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out
        
        pk_int = interpolate.RectBivariateSpline( self.z , self.k, Pk_out, kx=1, ky=1)
        self.Pk_int = pk_int
        



    def __str__(self):
        """Return emulator description."""
        return (
            f"Cosmopower linear Pk module. Computes the linear power spectrum "
            f"for an LCDM cosmology with w = -1, using input cosmological parameters:\n"
            f"Inputs: ['ombh2', 'omch2', 'H0', 'ns', 'lnAs', 'z', 'mnu'] \n"
            f"Output: P(k) evaluated between k_min={self.k_min} and k_max={self.k_max}.\n"
            f"Neutrinos are modeled as in Archidiacono et al. (2024). There are three "
            f"degenerate massive neutrinos, with a total mass sum described by `mnu` parameter."
        )
    
    def matter_power_spectrum(self,zs,ks):
        """
        Compute the linear matter power spectrum P(k, z).

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
        if hasattr(self, 'Pk_int') and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k


class LCDM_1mass_Linear:
    """
    Emulator for the linear matter power spectrum in the LCDM cosmology (w is set to -1) with one massive neutrino.

    This class uses a Cosmopower-trained neural network to emulate the linear power spectrum 
    for a w0waCDM cosmology with one massive neutrino. The neutrinos are modeled as in Casas et al. (2023), with the mass of the neutrino cotroled with `m_nu` parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp


            cp_file = emulator_data("1mass_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)


        self.k_emu = np.loadtxt(k_modes_path)


        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        self.background = background         
        assert background.Omega_k0 == 0, 'Non flat geometries not supported'

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        cp_bounds = {'ombh2': np.array([0.015, 0.035]),
                     'omch2': np.array([0.05,  0.2]),
                     'H0': np.array([60, 80]),
                     'ns': np.array([0.9, 1.05]),
                     'lnAs': np.array([2.5, 3.5]),
                     'z': np.array([0.0, 5.0]),
                     'mnu': np.array([0.00, 0.09]),
          }
        

        self.params = {'ombh2': self.background.Omega_b0 * self.background.h**2,
                  'omch2': self.background.Omega_cdm0 * self.background.h**2,
                    'H0': self.background.H0,
                    'ns': self.background.ns,
                    'lnAs': np.log(self.background.As * 1e10),
                    'mnu': self.background.mnu,
                  }
        
        for key in self.params.keys():
            if np.product(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Parameters out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))
        
        self.params['z'] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(self.k_emu, self.z , Pk_lin,flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)
        
        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out
        
        pk_int = interpolate.RectBivariateSpline( self.z , self.k, Pk_out, kx=1, ky=1)
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

    def matter_power_spectrum(self,zs,ks):
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
        if hasattr(self, 'Pk_int') and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k



class LCDM_Linear:
    """
    Emulator for the linear matter power spectrum in the LCDM cosmology (w is set to -1) with no massive neutrinos.

    This class uses a Cosmopower-trained neural network to emulate the linear power spectrum 
    for a w0waCDM cosmology with no massive neutrinos (neutrino mass is set to zero).
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp


            cp_file = emulator_data("LCDM_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)


        self.k_emu =np.loadtxt(k_modes_path)


        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        self.background = background         
        assert background.Omega_k0 == 0, 'Non flat geometries not supported'

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        cp_bounds = {'ombh2': np.array([0.009, 0.058]),
                     'omch2': np.array([0.02,  0.69]),
                     'H0': np.array([55., 91.]),
                     'ns': np.array([0.87, 1.07]),
                     'lnAs': np.array([0.5, 5]),
                     'z': np.array([0.0, 5.0]),
          }
        

        self.params = {'ombh2': self.background.Omega_b0 * self.background.h**2,
                  'omch2': self.background.Omega_cdm0 * self.background.h**2,
                    'H0': self.background.H0,
                    'ns': self.background.ns,
                    'lnAs': np.log(self.background.As * 1e10),
                  }
        
        for key in self.params.keys():
            if np.product(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Cosmopower linear Ivan out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))
        
        self.params['z'] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(self.k_emu, self.z , Pk_lin,flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)
        
        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out
        
        pk_int = interpolate.RectBivariateSpline( self.z , self.k, Pk_out, kx=1, ky=1)
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
    
    def matter_power_spectrum(self,zs,ks):
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
        if hasattr(self, 'Pk_int') and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k
    
