# cloelite imports
from cloelite.cosmology.cosmology import Background

# General imports
import jax.numpy as np

"""
## Author:
    **Name**: M. Bonici & G. Canas-Herrea
    **Date**: June 9, 2024

## Notes:

- Make it completely differentiable

"""

class JAXBackground(Background):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, gamma_MG: float):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class

        """
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.As = float(As)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)

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
        return self.H0 * np.sqrt((self.Omb+self.Omc)*np.power(1+zs, 3) +
                                 (self.Omk)*np.power(1+zs, 2) +
                                 (1-self.Omb-self.Omc-self.Omk) * np.power(1+zs, 3*(1+self.w+self.wa))*np.exp(-3*self.wa*zs/(1+zs)))

    #@property
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
        c_0 = 2.99792458e5 #please, put all the constanst in a single place
        fun = lambda x: 1/self.hubble_parameter(x)
        y = simps(fun, 0, zs, N=512) * c_0
        return y

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

    #@property
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
        return self.comoving_distance(zs)
    #TODO add the lax conditionals to account for the curvature!

    #@property
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
        return self.transverse_comoving_distance(zs)/(1+zs)

#function takes from JAXCosmo. Should likely be moved to an utils.py
def simps(f, a, b, N=128):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S
