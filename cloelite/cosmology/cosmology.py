# General imports
import numpy as np
from abc import ABC, abstractmethod

"""
## Author:
    **Name**: G. Canas-Herrera & M. Bonici
    **Date**: June 9, 2024

## Notes:

- Fix cosmology.py from original CLOE by offering a more flexible
approach to link to other codes and make it Cobaya independent

"""

class Background(ABC):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, gamma_MG: float):
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.As = float(As)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)

    @abstractmethod
    def hubble_parameter(self):
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

class Perturbations(ABC):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, gamma_MG: float):
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.As = float(As)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)

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



class Cosmology(self):
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
            from cloelite.cosmology.jax import JAXBackground
            self.background_backend = JAXBackground(H0, Omb, Omc, Omk, As, ns, w, wa, gamma_MG)
        else:
            raise ValueError(f"Unsupported background backend: {self.backend["background"]}. Choose between: CAMB, JAX")

        if self.backend["perturbations"] == 'CAMB':
            from camb_cosmology import CAMBPerturbations
            self.perturbations_backend = CAMBPerturbations(H0, Omb, Omc, Omk, As, ns, w, wa, gamma_MG)
        elif self.backend["perturbations"] == 'JAX':
            from cloelite.cosmology.jax import JAXPerturbations
            self.perturbations_backend = JAXPerturbations(H0, Omb, Omc, Omk, As, ns, w, wa, gamma_MG)
        else:
            raise ValueError(f"Unsupported perturbations backend: {self.backend["perturbations"]}. Choose between: CAMB, JAX")
