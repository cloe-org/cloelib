# cloelite imports
from cloelite.cosmology.cosmology import Background
# General imports
import numpy as np 
# Cosmology imports
import camb

"""
## Author:
    **Name**: G. Canas-Herrera & M. Bonici  
    **Date**: June 9, 2024

## Notes:

- Make sufficiently general to interface with CAMB keeping the structure by Cosmology

"""

class CAMBBackground(Background):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, sigma8: float, ns: float,
                 w: float, wa: float, gamma_MG: float):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class
        """
        
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.sigma8 = float(sigma8)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)
        self.h = float(self.H0 / 100)
        self.ombh2 = float(self.Omb * self.h**2)
        self.omch2 = float(self.Omc * self.h**2)

        # Define CAMB params
        self.CAMBparams = camb.CAMBparams()
        self.CAMBparams.set_cosmology(H0=H0, ombh2=self.ombh2, omch2=self.omch2, 
                                      mnu=0.0, neutrino_hierarchy='degenerate', num_massive_neutrinos=0.0, YHe=0.2454 , nnu=3.046)
        self.CAMBparams.set_dark_energy(w=self.w, wa=self.wa) #re-set defaults
        
        # Get background cosmology
        self.CAMBresults = camb.get_background(self.CAMBparams)

        # Update attributes with derived parameters
        self.Omm = self.CAMBparams.omegam
        self.Omnu = self.CAMBparams.omeganu

    def hubble_parameter(self, zs, units = '1/Mpc') -> np.ndarray:
            r"""
            Retrieves the hubble parameter as
            a function of redshift

            .. math::
                H(z) = \sqrt

            Parameters
            ----------
            zs: numpy.ndarray
                Redshifts for the matter density
            units: str
                Used units to return H(z)
                Options are: km/s/Mpc and 1/Mpc

            Returns
            -------
            Hubble parameter: numpy.ndarray
                hubble parameter as a function of redshift

            """

            if units == '1/Mpc':
                return self.CAMBresults.h_of_z(zs)
            elif units == 'km/s/Mpc':
                return self.CAMBresults.hubble_parameter(zs)

    def comoving_distance(self, zs) -> np.ndarray:
        """
        Calculates the comoving distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the comoving distance.

        Returns:
        --------
        np.ndarray
            The comoving distance as a function of redshift.
        """

        return self.CAMBresults.comoving_radial_distance(zs)

    def angular_diameter_distance(self, zs) -> np.ndarray:
        """
        Calculates the angular diameter distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the angular diameter distance.

        Returns:
        --------
        np.ndarray
            The angular diameter distance as a function of redshift.
        """

        return self.CAMBresults.angular_diameter_distance(zs)

    def matter_density(self, zs) -> np.ndarray:
        r"""
        Computes the matter density as

        .. math::
            \Omega_{\rm m}(z) = \Omega_{{\rm m},0}(1+z)^3H_0^2/H^2(z)

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts at which to calculate the matter density

        Returns
        -------
        Matter density parameter: numpy.ndarray
            Matter density as a function of redshift

        """

        omegam_z = self.CAMBresults.get_Omega('cdm', z=zs) + \
            self.CAMBresults.get_Omega('baryon', z=zs) + \
            self.CAMBresults.get_Omega('neutrino', z=zs) + \
            self.CAMBresults.get_Omega('nu', z=zs)
        return self.CAMBresults.get_Omega('tot', z=zs)
    
    def transverse_comoving_distance(self, zs) -> np.ndarray:
        """
        Calculates the transverse comoving distance beetween two redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the transverse comoving distance.

        Returns:
        --------
        np.ndarray
            The transverse comoving distance as a function of redshift.
        """
        c_0 = 2.99792458e5
        int_z1z2 = ((self.comoving_distance(z)[None, :] - self.comoving_distance(z)[:, None]) *
                    self.H0 / c_0)
        if self.Omk == 0.0:
            y_int = int_z1z2
        elif self.Omk > 0.0:
            y_int = (np.sinh(np.sqrt(self.Omk) * int_z1z2) /
                     np.sqrt(self.Omk))
        else:
            y_int = (np.sin(np.sqrt(-self.Omk) * int_z1z2) /
                     np.sqrt(-self.Omk))
        y_int *= (self.cosmo_dic['c'] / self.H0)

        return y_int