# cloelite imports
from cosmology import Cosmology

# General imports
import jax.numpy as np

"""
## Author:
    **Name**: M. Bonici & G. Canas-Herrea
    **Date**: June 9, 2024

## Notes:

- Make it completely differentiable

"""

class JAXCosmology(Cosmology):
    """
    A class to define background cosmology using JAX
    and inheriting from Cosmology parent class

    """
    @property
    def hubble_parameter(self, zs) -> np.ndarray:
        r"""
        Retrieves the hubble parameter as
        a function of redshift

        .. math::
            H(z) = \sqrt

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts for the matter density

        Returns
        -------
        Hubble parameter: numpy.ndarray
            hubble parameter as a function of redshift

        """

        return 

    @property
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

        return 
    
    @property
    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

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
        return 
    
    @property
    def growth_rate(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.
        ks : array_like
            Wavenumbers at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """
        return 
    
    @property
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
        return 

    @property
    def transverse_comoving_distance(self, zs) -> np.ndarray:
        """
        Calculates the transverse comoving distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the transverse comoving distance.

        Returns:
        --------
        np.ndarray
            The transverse comoving distance as a function of redshift.
        """
        return 

    @property
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
        return 
