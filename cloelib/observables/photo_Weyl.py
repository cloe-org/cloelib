"""
Module with two classes for each observable tracer type: shear and galaxy positions.

Both classes are compatible with the Tracer protocol.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson, simps
from cloelib.auxiliary.systematics import shift_dndz_jax

# General imports
import jax.numpy as np  # type: ignore
import jax  # type: ignore
import interpax  # type: ignore
import jax.lax as lx


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s

# NOTE WEYL PROJECT: No changes to ShearTracer class --> import from usual photo.py file 
    

class PositionsTracer_Weyl_GC:
    """Class to define the kernel for angular (galaxy) clustering.""" #when used in the context of the Weyl potential
    # For the Weyl measurement, we define parameters bhat*sigma_8 per bin --> we keep the per bin case and remove the other bias implementations
    
    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        #galaxy_bias_model: str, #Not needed for Weyl measurement (we only use the per-bin case)
        nuisance_params: dict,
    ):
        r"""
        Initialize the class instance.

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
        nuisance_params : dict
            A dictionary containing additional parameters that are not directly related to the cosmological model but may affect the observations.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        
        # WEYL: Add z_ini 
        # WEYL: For this to work, perturbations should be initialized with redshifts = np.array([z_ini])
        self.z_ini = self.perturbations.z[0]
        
        
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 0

        self.nuisance_params = nuisance_params
        self.dz_pos_i = [
            self.nuisance_params[f"dz_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for dz_pos
        self.dndz_shifted = shift_dndz_jax(dndz, z, self.dz_pos_i)
        #self.flags = {"galaxy_bias_model": galaxy_bias_model}
        self.n_z_bins = dndz.shape[0]
        self.magnification_bias = [
            self.nuisance_params[f"magnification_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]
        

        # Using dict.get so I can provide a default since lax has to compile every branch of the conditional
        # WEYL: Only use per bin case; We use bhat(z) = b(z)*sigma_8(z) instead of just b(z) as parameters
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("bhat_bin%d" % bin, 1.0) # WEYL: Renamed parameter b1_photo -> bhat
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        self.bias_array = per_bin_case()

    def get_window_positions(self, z) -> np.ndarray:
        # WEYL: This will be the positions window function when used for galaxy clustering
        
        r"""Galaxy Positions window function.    

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
        # WEYL: We need to act factor (1/sigma_8(z_ini))**2
        # The following will work with a CAMB Background, assuming it is initialized with redshifts = np.array([z_ini])
        # Will need to adjust this to make it consistently work with any background
        sigma_8ini = self.perturbations.results.get_sigma8()[0]
        #sigma_8ini = self.background.sigma8*self.perturbations.growth_factor(self.z_ini,1) #growth factor already normalized to 1 today

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / c_0
                / sigma_8ini
            )
            return window

        window_positions = per_bin_case()

        return window_positions

    def get_magnification_efficiency(self, z):
        r"""
        Compute the magnification efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Parameters
        ----------
        z : np.ndarray
            1D array of redshift values (must be evenly spaced). Used to compute comoving distances
            and define integration domain.

        Returns
        -------
        np.ndarray
            2D array of shape (N_bins, len(z)) representing the lensing efficiency kernel W(z)
            for each redshift bin over the evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz` is expected to have shape (N_bins, len(z)) and be normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_magnification(self, z):         
        r"""Magnification photometric galaxy kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        :obj:`np.trapz` on the array along one axis.

        .. math::
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} b_{\rm mag, i} (1 + z)
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
        
        # WEYL: Added factor D_1(z)/D_1(z_\ini)
        
        growth_factor = self.perturbations.growth_factor(z,1)/self.perturbations.growth_factor(self.z_ini,1)
        
        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
            * growth_factor 
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_magnification_efficiency(z)
        return (
            np.einsum("ij, j->ij", efficiency, factor)
            * np.array(self.magnification_bias)[:, None]
        )

    def get_window(self, z) -> np.ndarray:
        #WEYL: This will be the total positions window function used for galaxy clustering; no modifications are needed here, as long as get_window_positions and get_window_magnification are properly modified
        """
        Compute the angular photometric galaxy clustering window function.

        This function combines the galaxy clustering window and the magnification
        bias window to produce the final window function.

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """
        window = self.get_window_positions(z) + self.get_window_magnification(z)
        return window
    

    
class PositionsTracer_Weyl_GGL:
    """Class to define the positions kernel for galaxy-galaxy lensing, including the Weyl potential.""" #when used in the context of the Weyl potential
    
    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        #galaxy_bias_model: str, #Not needed for Weyl measurement (we only use the per-bin case)
        nuisance_params: dict,
        Jhat_params: dict, #New parameter for Weyl project
    ):
        r"""
        Initialize the class instance.

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
        nuisance_params : dict
            A dictionary containing additional parameters that are not directly related to the cosmological model but may affect the observations.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        
        # WEYL: Add z_ini
        self.z_ini = self.perturbations.z[0]
        
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 0

        self.nuisance_params = nuisance_params
        self.dz_pos_i = [
            self.nuisance_params[f"dz_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for dz_pos
        self.dndz_shifted = shift_dndz_jax(dndz, z, self.dz_pos_i)
        #self.flags = {"galaxy_bias_model": galaxy_bias_model}
        self.n_z_bins = dndz.shape[0]
        self.magnification_bias = [
            self.nuisance_params[f"magnification_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]
        
        # Using dict.get so I can provide a default since lax has to compile every branch of the conditional
    
        # WEYL: Only use per bin case; We use bhat(z) = b(z)*sigma_8(z) instead of just b(z) as parameters
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("bhat_bin%d" % bin, 1.0) # WEYL: Renamed parameter b1_photo -> bhat
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))
        
        #Same as above but for Jhat instead of bias
        
        def per_bin_Jhat():
            bias_array = np.asarray(
                [
                    Jhat_params.get("Jhat_bin%d" % bin, 1.0) 
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))
        

        self.bias_array = per_bin_case()
        self.Jhat_array = per_bin_Jhat()

    def get_window_positions(self, z) -> np.ndarray:
    #This will be the positions window function used in case of galaxy-galaxy lensing, sensitive to both Jhat and bhat
        
        r"""Galaxy Positions window function.    

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

        # WEYL: Multiplied with Jhat; Removed factor (self.background.H0 / c_0) ** 2* Omega_m0* (1 + z)
        Omega_m0 = self.background.Omega_m(0.0)
        
        
        # Weyl: We need to act factor (1/sigma_8(z_ini))**2
        # The following will work with a CAMB Background, assuming it is initialized with redshifts = np.array([z_ini])
        # Will need to adjust this to make it consistently work with any background
        sigma_8ini = self.perturbations.results.get_sigma8()[0]
        #sigma_8ini = self.background.sigma8*self.perturbations.growth_factor(self.z_ini,1) #growth factor already normalized to 1 today
        
        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.Jhat_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / (c_0* (self.background.H0 / c_0) ** 2* Omega_m0* (1 + z))
                / sigma_8ini**2
            )
            return window

        window_positions = per_bin_case()

        return window_positions
    

    def get_magnification_efficiency(self, z):
        r"""
        Compute the magnification efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Parameters
        ----------
        z : np.ndarray
            1D array of redshift values (must be evenly spaced). Used to compute comoving distances
            and define integration domain.

        Returns
        -------
        np.ndarray
            2D array of shape (N_bins, len(z)) representing the lensing efficiency kernel W(z)
            for each redshift bin over the evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz` is expected to have shape (N_bins, len(z)) and be normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_magnification(self, z):
        r"""Magnification photometric galaxy kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        :obj:`np.trapz` on the array along one axis.

        .. math::
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} b_{\rm mag, i} (1 + z)
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
        
        # Weyl project: added growth factor
        growth_factor = self.perturbations.growth_factor(z,1)/self.perturbations.growth_factor(self.z_ini,1)

        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
            * growth_factor**2 
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_magnification_efficiency(z)
        return (
            np.einsum("ij, j->ij", efficiency, factor)
            * np.array(self.magnification_bias)[:, None]
        )
    
    def get_window(self, z) -> np.ndarray:
         # WEYL: This will be the total positions window function used for GGL; no modifications are needed here, as long as get_window_positions and get_window_magnification have been changed appropriately
        """
        Compute the angular photometric galaxy clustering window function.

        This function combines the galaxy clustering window and the magnification
        bias window to produce the final window function.

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """
        window = self.get_window_positions(z) + self.get_window_magnification(z)
        return window
    