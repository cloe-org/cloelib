# General imports
import numpy as np
from abc import ABC, abstractmethod

# cloelite imports
from cloelite.cosmology.cosmology import LinearPerturbations
from cloelite.cosmology.cosmology import NonLinearPerturbations

"""

## Notes:

- Tracer abstract class to implement different window functions

"""

class Tracer(ABC):
    def __init__(self, perturbations: {LinearPerturbations, NonLinearPerturbations}):
        # The only common ingredients to all the tracers are
        # perturbations and cosmological background
        # (inherited from perturbations too)

        self.perturbations = perturbations
        self.background = perturbations.background

    @abstractmethod
    def _window_integrand(self, z, zprime):
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
