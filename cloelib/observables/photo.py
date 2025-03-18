# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson

# General imports
import jax.numpy as np # type: ignore
from scipy.interpolate import RectBivariateSpline # type: ignore
import jax # type: ignore
import interpax # type: ignore


"""

## Notes:

- Make sufficiently general to interface with CAMB keeping the structure by Cosmology
- Make Tracer a protocol

"""

# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s

class ShearTracer:
    def __init__(self, perturbations: Perturbations, dndz: np.ndarray, z: np.ndarray,
                 nuisance_params: dict):
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
        """
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.dndz = dndz
        self.z = z
        self.nuisance_params = nuisance_params
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 1

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

        Args:
            zprime: float or numpy.ndarray
                Redshift parameter that will be integrated over
            z: float
                Redshift at which kernel is being evaluated
            n_z: numpy.ndarray
                Redshift bin distribution

        Returns:
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

        win_int = self._window_integrand(z, self.dndz)

        W_val = (1.5 * self.background.H0 *
                 (self.background.Omega_b0 + self.background.Omega_cdm0) * \
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
        Hz = self.perturbations.background.hubble_parameter(z)
        Dz = self.perturbations.growth_factor(self.perturbations.z, self.perturbations.k)[:,1]
        #TODO discuss whether we want growth factor to output a 1D or a 2D array
        A_IA = self.nuisance_params["AIA"]
        C_IA = self.nuisance_params["CIA"]
        Eta_IA = self.nuisance_params["EtaIA"]
        factor = -Hz/c_0*A_IA*C_IA*(self.background.Omega_b0 + self.background.Omega_cdm0)*(1+z)**Eta_IA/Dz
        return np.einsum('ij, j->ij', self.dndz, factor)

    def get_lensing_efficiency_bin(self, z, bin_idx):
        interpolator = interpax.Akima1DInterpolator(self.z, self.dndz[bin_idx,:])
        #horrible quick fix
        #to make it right, we need the triangular matrix of simpson weight for the
        #lensing efficiency. This will also significantly improve performance!
        x = np.linspace(0., 4, 200)
        y = self.background.comoving_distance(x)
        rx_interp = interpax.Akima1DInterpolator(x, y)
        #f1 = lambda x, y: interpolator(x)*(1-tracer_she.background.comoving_distance(y)/tracer_she.background.comoving_distance(x))
        f1 = jax.jit(lambda x: interpolator(x))
        f2 = jax.jit(lambda x: interpolator(x)/rx_interp(x))
        integral_1 =  simps(f1, z, 3.)
        integral_2 =  simps(f2, z, 3.)
        efficiency = integral_1 - integral_2*self.background.comoving_distance(z)
        return efficiency

    def get_lensing_efficiency_check(self, z):
        n_bins = self.dndz.shape[0]
        efficiency = self.get_lensing_efficiency_bin(z, 0)
        for i in np.arange(1,n_bins):
            efficiency = np.vstack([efficiency, self.get_lensing_efficiency_bin(z, i)])

        return efficiency

    def get_lensing_efficiency(self, z):
        dndz = self.dndz
        dz = z[1]-z[0]#assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz,1/rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum('ik, jk, jk->ij', dndz, rzrz, w_matrix)*dz
        return result

    def get_lensing_window_check(self, z):
        factor = 3/2*(self.background.H0/c_0)**2*(self.background.Omega_b0 + self.background.Omega_cdm0)\
        *(1+z)*self.background.comoving_distance(z)
        efficiency = self.get_lensing_efficiency_check(z)
        return np.einsum('ij, j->ij', efficiency, factor)

    def get_lensing_window(self, z):
        factor = 3/2*(self.background.H0/c_0)**2*(self.background.Omega_b0 + self.background.Omega_cdm0)\
        *(1+z)*self.background.comoving_distance(z)
        efficiency = self.get_lensing_efficiency(z)
        return np.einsum('ij, j->ij', efficiency, factor)

    def get_window_check(self, z):
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
        return self.get_lensing_window_check(z) + self.get_window_IA(z)

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
        return self.get_lensing_window(z) + self.get_window_IA(z)

class PositionsTracer:
    def __init__(self, perturbations: Perturbations, dndz: np.ndarray, z: np.ndarray):
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

        self.perturbations = perturbations
        self.dndz = dndz
        self.z = z
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 1

        #self.nuisance_params = nuisance_params
        #self.flags = {'galaxy_bias_model': galaxy_bias_model, 'magnification_bias_model': magnification_bias_model}

    def get_window_positions(self, z) -> np.ndarray:
        r"""Galaxy Positions window function

        Implements the galaxy clustering photometric window function.

        .. math::
            W_i^G(z) &= \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\

        Parameters
        ----------
        z: numpy.ndarray of float or float
           Redshift at which to evaluate distribution

        Returns
        -------
        window_positions: numpy.ndarray
           Window function for angular photometric galaxy clustering
        """
        window_positions = self.dndz * \
            self.perturbations.background.hubble_parameter(z) / c_0

        return window_positions

    def get_window(self, z) -> np.ndarray:
        """
        Computes the angular photometric galaxy clustering window function.

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """
        # add more contributions below
        return self.get_window_positions(z)


def simps(f, a, b, N=128):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S
