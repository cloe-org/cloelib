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
    def __init__(self, perturbations: object, dndz: np.ndarray,
                 intrinsic_aligment_model: str, nuisance_params: dict):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class
        """
        
        super().__init__(self)
        self.dndz = dndz
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
    def __init__(self, perturbations: object, dndz: np.ndarray,
                 galaxy_bias_model: str, magnification_bias_model: str,
                 nuisance_params: dict):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class
        """

        super().__init__(self)
        self.dndz_pos = dndz
        self.flags = {'galaxy_bias_model': galaxy_bias_model,
                      'magnification_bias_model': magnification_bias_model}
        self.nuisance_params = nuisance_params

    def _get_prefactor(self, ell):
        pass

    def get_window_positions(self, z):
        r"""GC window.

        Implements the galaxy clustering photometric window function.

        .. math::
            W_i^G(z) &= \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\

        Parameters
        ----------
        z: numpy.ndarray of float or float
           Redshift at which to evaluate distribution

        Returns
        -------
        GCphot window function: float
           Window function for photometric galaxy clustering
        """

        # check how we normalize
        # Think on interpolators for dndz

        window_GC = self.dndz * self.theory['H_z_func_Mpc'](z)

        return window_GC


    def get_window_RSD(self, z):
        r"""GC window RSD.

        Implements the RSD correction to the galaxy clustering photometric
        window function in an array-like format, modulo the Limber and
        full sky prefactor,

        .. math::
            W_i^{\rm{G,RSD}}(z,\ell) =
            \frac{1}{c \,b_{\mathrm{g},i}^\mathrm{photo}} \
            \left[H(z_{\rm m})f(z_{\rm m})\
            \frac{n_i(z)}{\bar{n_i}}\right]_{\rm m}

        where :math:`m` assumes the values (-1,0,+1).

        Parameters
        ----------
        z: numpy.ndarray or float
            Redshift at which to evaluate the window function
        ell: numpy.ndarray or float
            Multipole at which to evaluate the window function
        bin_i: int
            Index of desired tomographic bin. Tomographic bin
            indices start from 1

        Returns
        -------
        RSD GCphot window function: numpy.ndarray
            Window function for RSD component of photometric galaxy clustering.
        """
        #if isinstance(ell, (int, float)):
        #    ell = [ell]
        #if isinstance(z, (int, float)):
        #    z = [z]

        #tdist = self.theory['f_K_z_func'](z)
        #zm_arr = np.array([[self.z_minus1(ll, tdist) for ll in ell],
        #                   np.full((len(ell), len(z)), z),
        #                   [self.z_plus1(ll, tdist) for ll in ell]])

        #Hzm_arr = self.theory['H_z_func_Mpc'](zm_arr)
        #fzm_arr = self.theory['f_z'](zm_arr)
        #nzm_arr = self.nz_GC.evaluates_n_i_z(bin_i, zm_arr)

        #if self.theory['bias_model'] == 2:
        #    bias = self.photobias[bin_i - 1]
        #elif self.theory['bias_model'] in [1, 3]:
        #    bias = self.theory['b1_inter'](z)

        #return Hzm_arr * fzm_arr * nzm_arr / bias

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
