# cloelite imports
from cloelite.observables.tracer import Tracer
from cloelite.cosmology.cosmology import LinearPerturbations
from cloelite.cosmology.cosmology import NonLinearPerturbations

# General imports
import jax.numpy as np
from scipy.interpolate import RectBivariateSpline
import jax
import interpax


"""
**Date**: June 11, 2024

## Notes:

- Make sufficiently general to interface with CAMB keeping the structure by Cosmology

"""

class ShearTracer(Tracer):
    def __init__(self, perturbations: {LinearPerturbations, NonLinearPerturbations}, dndz: np.ndarray, z: np.ndarray,
                 intrinsic_aligment_model: str, nuisance_params: dict):
        r"""
        A class to define the kernel for Cosmic Shear.

        Initialize the class with given perturbations, redshift distribution,
        intrinsic aligment models, and nuisance parameters.

        Parameters
        ----------
        perturbations : object
            An object from NonLinearPerturbations class
        dndz : np.ndarray
            A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
        z : np.ndarray
            A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
        intrinsic_aligment_model : str
            A string specifying the model used to describe intrinsic aligments
        nuisance_params : dict
            A dictionary containing additional parameters that are not directly related to the cosmological model but may affect the observations.
        """

        super().__init__(perturbations)
        self.dndz = dndz
        self.z = z
        self.nuisance_params = nuisance_params
        self.flags = {'intrinsic_aligment_model': intrinsic_aligment_model}

    def _get_prefactor(self, ell):
        return 0

    @jax.jit
    def _window_integrand(self, z, n_z):
        r"""Window integrand.

        Calculates generic integrand for windows such as
        lensing or magnification bias kernels

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
        n_z: numpy.ndarray
            Redshift bin distribution

        Returns
        -------
        window_integrand: np.ndarray
        """

        chi = self.background.comoving_distance(z)
        weights = np.ones(len(chi))

        mat_jax = jax.vmap(get_simpsons_weights_jit, in_axes=(0,))


        for i, redshift in enumerate(z):
            np.einsum('ij, j, j, jz -> iz', n_z, 1 - chi[i]/chi, mat_jax)

        return

    def get_window_shear(self, z):
        r"""Weak Lensing shear kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        :obj:`np.trapz` on the array along one axis.

        .. math::
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z) \Sigma(z, k)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\

        Parameters
        ----------
        z: numpy.ndarray of float
            Redshift at which weight is evaluated.
        bin_i: int
            Index of desired tomographic bin.
            Tomographic bin indices start from 1
        k: float
            Wavenumber at which to evaluate the Modified Gravity
            :math:`\Sigma(z,k)` function

        Returns
        -------
        Shear kernel: numpy.ndarray
            1-D Numpy array of shear kernel values for specified bin
            at specified scale for the redshifts defined in z
        """

        c_0 = 2.99792458e5
        win_int = self._window_integrand(z, self.dndz)

        W_val = (1.5 * self.background.H0 * self.background.Omm * \
                 (1.0 + z) * self.background.comoving_distance(z) *
                  ( c_0/ self.background.H0)) * win_int

        return W_val

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

    def lensing_efficiency_bin(self, z, bin_idx):
        interpolator = interpax.Akima1DInterpolator(self.z, self.dndz[bin_idx,:])
        #f1 = lambda x, y: interpolator(x)*(1-tracer_she.background.comoving_distance(y)/tracer_she.background.comoving_distance(x))
        f1 = jax.jit(lambda x: interpolator(x))
        f2 = jax.jit(lambda x: interpolator(x)/self.background.comoving_distance(x))
        integral_1 =  simps(f1, z, 3.)
        integral_2 =  simps(f2, z, 3.)
        efficiency = integral_1 - integral_2*self.background.comoving_distance(z)
        return efficiency

    def lensing_efficiency(self, z):
        n_bins = self.dndz.shape[0]
        efficiency = self.lensing_efficiency_bin(z, 0)
        for i in np.arange(1,n_bins):
            efficiency = np.vstack([efficiency, self.lensing_efficiency_bin(z, i)])

        return efficiency

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
    def __init__(self, perturbations: {LinearPerturbations, NonLinearPerturbations}, dndz: np.ndarray, z: np.ndarray,
                 galaxy_bias_model: str, magnification_bias_model: str,
                 nuisance_params: dict):
        r"""
        A class to define the kernel for angular (galaxy) clustering

        Initialize the cosmology class with given perturbations, redshift distribution,
        galaxy and magnification bias models, and nuisance parameters.

        Parameters
        ----------
        perturbations : :class:`LinearPerturbations` or :class:`NonLinearPerturbations`
            An object from NonLinearPerturbations class
        dndz : np.ndarray
            A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
        z : np.ndarray
            A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
        galaxy_bias_model : str
            A string specifying the model used to describe the galaxy bias
        magnification_bias_model : str
            A string specifying the model used to describe the magnification bias
        nuisance_params : dict
            A dictionary containing additional parameters that are not directly related to the cosmological model but may affect the observations.
        """

        super().__init__(perturbations)
        self.dndz = dndz#np.vstack(list(dndz.values()))
        self.z = z
        self.nuisance_params = nuisance_params
        self.flags = {'galaxy_bias_model': galaxy_bias_model, 'magnification_bias_model': magnification_bias_model}

    def _get_prefactor(self, ell):
        pass

    def get_window_positions(self, z):
        r"""Galaxy Position window.

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
           Window function for angular photometric galaxy clustering
        """

        # check how we normalize
        # Think on interpolators for dndz

        window_positions = self.dndz * \
            self.background.hubble_parameter(z)

        return window_positions


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

    def get_window(self, z):
        return self.get_window_positions(z)
    #gonna add the other contributes here!

def simps(f, a, b, N=128):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S
