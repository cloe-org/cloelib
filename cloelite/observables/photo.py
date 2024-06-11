# cloelite imports
from cloelite.observables.tracer import Tracer

# General imports
import jax.numpy as np

"""
## Author:
    **Name**: G. Canas-Herrera & M. Bonici  
    **Date**: June 11, 2024

## Notes:

- Make sufficiently general to interface with CAMB keeping the structure by Cosmology

"""

class ShearTracer(Tracer):
    def __init__(self, perturbations: object, dndz_she: np.ndarray,
                 intrinsic_aligment_model: str, nuisance_params: dict):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class
        """
        
        self.dndz_she = dndz_she
        self.flags = {'intrinsic_aligment_model': intrinsic_aligment_model}
        self.nuisance_params = nuisance_params

    def get_window_IA(self, z):
        r"""Window integrand.

        Calculates IA window

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated

        Returns
        -------
        window_IA: np.ndarray
        """

        pass

    def get_window_shear(self, z):
        r"""Window integrand.

        Calculates shear window

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated

        Returns
        -------
        window_shear: np.ndarray
        """

        pass

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

    def get_window(self, z):
        r"""Window

        Computes general window given the selected tracer

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """

        pass

class PositionsTracer(Tracer):
    def __init__(self, perturbations: object, dndz_pos: np.ndarray,
                 galaxy_bias_model: str, magnification_bias_model: str,
                 nuisance_params: dict):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class
        """
        
        self.dndz_pos = dndz_pos
        self.flags = {'galaxy_bias_model': galaxy_bias_model,
                      'magnification_bias_model': magnification_bias_model}
        self.nuisance_params = nuisance_params


    def get_window_IA(self, z):
        r"""Window integrand.

        Calculates IA window

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated

        Returns
        -------
        window_IA: np.ndarray
        """

        pass

    def get_window_shear(self, z):
        r"""Window integrand.

        Calculates shear window

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated

        Returns
        -------
        window_shear: np.ndarray
        """

        pass

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

    def get_window(self, z):
        r"""Window

        Computes general window given the selected tracer

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """

        pass
