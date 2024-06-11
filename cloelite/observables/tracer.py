# General imports
import numpy as np
from abc import ABC, abstractmethod

"""
## Author:
    **Name**: G. Canas-Herrera & M. Bonici
    **Date**: June 11, 2024

## Notes:

- Introducing the Tracer abstract class to implement different window functions
- Idea behind: Tracer should be sufficiently generic to have flexibility (!)
- SPIRIT: we do not classify among probes. All probes have their reason & place to be
- Lesson learnt: the cosmological analysis is a common endevaour

"""

class Tracer(ABC):
    def __init__(self, perturbations: object, **args):
        # The only common ingredient to all the tracers is 
        # perturbations and cosmological background 
        # (inherited on perturbations too)
        
        # args should be whatever n(z)

        self.perturbations = perturbations
        self.background = perturbations.background

    @abstractmethod
    def _window_integrad(self, z, zprime):
        r"""Window integrand.

        Calculates generic integrand for windows such as
        cosmic shear or magnification bias kernels

        .. math::
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm A}(z^{\prime})
            \frac{f_{K}\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}
            }

        This method is private. Not recommended to call directly, but possible

        Parameters
        ----------
        zprime: float or numpy.ndarray
            Redshift parameter that will be integrated over
        z: float
            Redshift at which kernel is being evaluated

        Returns
        -------
        window_integrand: np.ndarray
        """

        pass
    
    @abstractmethod
    def _get_prefactor(self, ell):
        r"""Computes the needed prefactor in Limber approximation.

        Parameters
        ----------
        ell: float or numpy.ndarray of float
           :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        Pre-factor: float or numpy.ndarray of float
           Value(s) of the prefactor at the given :math:`\ell`
        """

        pass

    @abstractmethod
    def get_window(self, z):
        r"""Window

        Computes general window(s) given the selected tracer

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        all windows
        """

        pass

