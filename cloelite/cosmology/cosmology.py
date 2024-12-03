# General imports
import numpy as np
from abc import ABC, abstractmethod

"""

## Notes:

- Fix cosmology.py from original CLOE by offering a more flexible
approach to link to other codes and make it Cobaya independent

"""

class Background(ABC):
    def __init__(self, H0: float, ombh2: float, omch2: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, sigma8 : float, gamma_MG: float):
        self.H0 = float(H0)
        self.ombh2 = float(ombh2)
        self.omch2 = float(omch2)
        self.Omk = float(Omk)
        self.As = float(As)
        self.sigma8 = float(sigma8)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)

        self.h = H0 / 100.0
        self.Omb = ombh2 / self.h**2
        self.Omc = omch2 / self.h**2

    @abstractmethod
    def hubble_parameter(self, zs, units = '1/Mpc'):
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
        pass

    @abstractmethod
    def comoving_distance(self, zs):
        r"""
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
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

class LinearPerturbations(ABC):
    def __init__(self, background : Background):
        self.background = background

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def linear_matter_power_spectrum(self):
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
    pass

class NonLinearPerturbations(ABC):
    def __init__(self, linearperturbations : LinearPerturbations):
        self.linearperturbations = linearperturbations

    @abstractmethod
    def nonlinear_matter_power_spectrum(self):
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
    pass


class Cosmology:
    """
    A general parent class that collects cosmological background quantities
    and the linear matter power spectrum from different backends.

    CAMB
    CLASS (work-in-progress)
    JAX
    general emulator (work-in-progress)

    """

    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, gamma_MG: float, background_backend: str,
                 perturbations_backend: str):
        """
        Initializes the CosmologyParameters class with the provided parameters.

        Parameters:
        -----------
        H0 : float
            Present-day Hubble constant
            :math:`{\rm (km·s^{-1}·Mpc^{-1})}`
        Omb : float
            Present-day baryon energy density
            :math:`\Omega_{\rm baryon}`
        Omc : float
            Present-day CDM energy density
            :math:`\Omega_{\rm CDM}`
        Omk : float
            Present-day curvature energy density
            :math:`\Omega_{\rm k}`
        As : float
            Amplitude of the primordial power spectrum
            :math:`A_{\rm s}`
        ns : float
            Spectral tilt of the primordial power spectrum
            :math:`n_{\rm s}`
        w : float
            Dark energy equation of state parameter
            :math:`w_0`
        wa : float
            Dark energy equation of state parameter
            :math:`w_a`
        gamma_MG: float
           Linder Modified Gravity parameter
           :math:`\gamma_{\rm g}`
        backend : str
            Selected backend to calculate cosmological quantities.
            Choose between: CAMB, JAX
        """
        # Ensure the attributes are floats
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.As = float(As)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)
        self.backend = {"background": background_backend,
                        "perturbations": perturbations_backend}

        # Dynamically instantiate the appropriate child class based on the backend
        if self.backend["background"] == 'CAMB':
            from camb_cosmology import CAMBBackground
            self.background_backend = CAMBBackground(H0, Omb, Omc, Omk, As, ns, w, wa, gamma_MG)
        elif self.backend["background"] == 'JAX':
            from cloelite.cosmology.jax_cosmology import JAXBackground
            self.background_backend = JAXBackground(H0, Omb, Omc, Omk, As, ns, w, wa, gamma_MG)
        else:
            raise ValueError(f"Unsupported background backend: {background_backend}. Choose between: CAMB, JAX")

        if self.backend["perturbations"] == 'CAMB':
            from camb_cosmology import CAMBPerturbations
            self.perturbations_backend = CAMBPerturbations(self.background_backend)
        elif self.backend["perturbations"] == 'JAX':
            from cloelite.cosmology.jax_cosmology import JAXPerturbations
            self.perturbations_backend = JAXPerturbations(self.background_backend)
        else:
            raise ValueError(f"Unsupported perturbations backend: {perturbations_backend}. Choose between: CAMB, JAX")
