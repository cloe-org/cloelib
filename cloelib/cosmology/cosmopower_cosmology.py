from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

import numpy as np
import warnings
from scipy import interpolate
import os
import urllib.request


##########################################################################
# This module contains different emulators for linear and non-linear matter power spectra. 
# The emulators are based on the Cosmopower emulator package (Mancini et al. 2022). Different models include: LCDM, LCDM+one massive neutrino, LCDM + 3 degenerate massive neutrinos, w0wa,  w0wa + one massive neutrino, w0wa + 3 degenerate massive neutrinos.
##########################################################################

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
        print(f"{filename} already exists at {file_path}")

    if filename.endswith('.pkl'):
        file_path = file_path[:-4]
    else:
        file_path = file_path

    return file_path


zenodo_path="https://zenodo.org/records/15537463/files"
k_modes_path = emulator_data("k-modes.txt", zenodo_path)


class w0wa3degenLinear:
    """
    Class to compute the linear matter power spectrum using the Cosmopower emulator. Computes the linear power spectrum for the w0wa cosmology. Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("w0wa+3degen_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)
        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module.  Computes the linear power spectrum for the w0wa cosmology between  k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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


class w0waOnemassLinear:
    """
    Class to compute the linear matter power spectrum using the Cosmopower emulator. Computes the linear power spectrum for the w0wa cosmology. Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("w0wa+1mass_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the linear power spectrum for the w0wa cosmology between k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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

class LCDMOnemassLinear:
    """
    Class to compute the linear matter power spectrum using the Cosmopower emulator. Computes the linear power spectrum for the LCDM cosmology (w is set to -1). Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("1mass_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the linear power spectrum for the LCDM cosmology (w is set to -1), between  k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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

class LCDM3degenLinear:
    """
    Class to compute the linear matter power spectrum using the Cosmopower emulator. Computes the linear power spectrum for the LCDM cosmology (w is set to -1). Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("3degen_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)
        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the linear power spectrum for the LCDM cosmology (w is set to -1) between k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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


class LCDMLinear:
    """
    Class to compute the linear power spectrum for the LCDM cosmology. The mass of the neutrinos is set to zero.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("LCDM_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the linear power spectrum for the LCDM cosmology between k_min={self.k_min} and k_max={self.k_max}. The mass of the neutrinos is set to zero and the w parameter is set to -1."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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
        

class w0waLinear:
    """
    Class to compute the linear power spectrum for the w0wa cosmology. Neutrino mass is set to zero.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("w0wa_linear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu =np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the linear power spectrum for the w0wa cosmology between k_min={self.k_min} and k_max={self.k_max}. Neutrino mass is set to zero."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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
    

class w0wa3degenNonLinear:
    """
    Class to compute the nonlinear matter power spectrum using the Cosmopower emulator. Computes the nonlinear power spectrum for the w0wa cosmology. Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("w0wa+3degen_nonlinear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the nonlinear power spectrum for the w0wa cosmology between k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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


class w0waOnemassNonLinear:
    """
    Class to compute the nonlinear matter power spectrum using the Cosmopower emulator. Computes the nonlinear power spectrum for the w0wa cosmology. Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("w0wa+1mass_nonlinear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan linear Pk module. Computes the nonlinear power spectrum for the w0wa cosmology between k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB. "
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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

class LCDMOnemassNonLinear:
    """
    Class to compute the nonlinear matter power spectrum using the Cosmopower emulator. Computes the nonlinear power spectrum for the LCDM cosmology (w is set to -1). Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("1mass_nonlinear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan nonlinear Pk module. Computes the nonlinear power spectrum for the LCDM cosmology (w is set to -1) between  k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Casas et al. 2023. There are is one massive neutrino, with a total mass described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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

class LCDM3degenNonLinear:
    """
    Class to compute the nonlinear matter power spectrum using the Cosmopower emulator. Computes the nonlinear power spectrum for the LCDM cosmology (w is set to -1). Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("3degen_nonlinear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan nonlinear Pk module. Computes the nonlinear power spectrum for the LCDM cosmology (w is set to -1) between k_min={self.k_min} and k_max={self.k_max}. Neutrinos are modeled as in Archidiacono et al. (2024). There are three degenerate massive neutrinos, with a total mass sum described by the mnu parameter. Nonlinear corrections are applied using the mead2020 model in CAMB."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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


class LCDMNonLinear:
    """
    Class to compute the nonlinear power spectrum for the LCDM cosmology. The mass of the neutrinos is set to zero. Nonlinear corrections are applied using the mead2020 model in CAMB.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("LCDM_nonlinear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan nonlinear Pk module,computes nonlinear matter power spectrum between k_min={self.k_min} and k_max={self.k_max}.  The mass of the neutrinos is set to zero. Nonlinear corrections are applied using the mead2020 model in CAMB."
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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
        

class w0waNonLinear:
    """
    Class to compute the nonlinear power spectrum for the w0wa cosmology. Neutrino mass is set to zero. Nonlinear corrections are applied using the mead2020 model in CAMB.
    """

    def __init__(self, background : Background, redshifts: np.ndarray):
        """
        Initialize the Cosmopower emulator.

        Parameters
        ----------
        background : Background
            The background cosmology.
        redshifts : np.ndarray
            The redshift values for which to compute the power spectrum.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cosmopower as cp

            # Load trained emulator from file
            cp_file = emulator_data("w0wa_nonlinear-spectra.pkl", zenodo_path)
            self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        # Load k-modes for the emulator
        self.k_emu = np.loadtxt(k_modes_path)

        # Define min/max k-ranges for safety
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
        """Return emulator description"""
        return f"CosmopowIvan nonlinear Pk module. Computs the nonlinear power spectrum for the w0wa cosmology between k_min={self.k_min}, k_max={self.k_max}.  Neutrino mass is set to zero. Nonlinear corrections are applied using the mead2020 model in CAMB"
    
    def matter_power_spectrum(self,zs,ks):
        """
        Return the linear matter power spectrum for the given redshifts

        Returns
        -------
        k : numpy.ndarray
            The k-modes in Mpc^{-1}
        Pk_lin : numpy.ndarray
            The linear power spectrum in (Mpc/h)^3
        """

        return self.Pk_int(zs, ks)
    
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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
    
