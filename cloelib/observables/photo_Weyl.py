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
    

class PositionsTracer_Weyl:
    """Class to define the kernel for angular (galaxy) clustering.""" #when used in the context of the Weyl potential
    # TO DO FOR WEYL PROJECT: Implement the parameters Jhat and bhat; remove current implementation of galaxy bias;
    
    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        galaxy_bias_model: str,
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
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 0

        self.nuisance_params = nuisance_params
        self.dz_pos_i = [
            self.nuisance_params[f"dz_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for dz_pos
        self.dndz_shifted = shift_dndz_jax(dndz, z, self.dz_pos_i)
        self.flags = {"galaxy_bias_model": galaxy_bias_model}
        self.n_z_bins = dndz.shape[0]
        self.magnification_bias = [
            self.nuisance_params[f"magnification_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]

        # Using dict.get so I can provide a default since lax has to compile every branch of the conditional
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        def per_bin_int_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            index_max_nz = np.argmax(dndz, axis=1)
            z_nz_max = jax.vmap(
                lambda i: lx.dynamic_index_in_dim(self.z, i, keepdims=False)
            )(index_max_nz)
            return interpax.interp1d(self.z, z_nz_max, bias_array, extrap=True)

        def poly_case():
            poly_order = 3
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_poly%d" % bin, 1.0)
                    for bin in range(poly_order + 1)
                ]
            )
            return (
                bias_array[0]
                + bias_array[1] * z
                + bias_array[2] * z**2
                + bias_array[3] * z**3
            )

        conditions = np.array(
            [
                self.flags["galaxy_bias_model"] == "per_bin",
                self.flags["galaxy_bias_model"] == "per_bin_int",
                self.flags["galaxy_bias_model"] == "poly",
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        self.bias_array = [per_bin_case, per_bin_int_case, poly_case][index]()

    def get_window_positions(self, z) -> np.ndarray:
        # WEYL PROJECT: This will be the positions window function when used for galaxy clustering
        # TO DO: Remove current implementation of bias; Implement Jhat and bhat parameters
        
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

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / c_0
            )
            return window

        def z_func_case():
            window = (
                self.bias_array[None, :]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / c_0
            )
            return window

        conditions = np.array(
            [
                self.flags["galaxy_bias_model"] == "per_bin",
                self.flags["galaxy_bias_model"] in ["per_bin_int", "poly"],
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        window_positions = [per_bin_case, z_func_case][index]()

        return window_positions
    
    def get_window_positions_GGL(self, z) -> np.ndarray: 
        # TO DO WEYL: This will be the positions window function used in case of galaxy-galaxy lensing, sensitive to both Jhat and bhat
        raise NotImplementedError("get_window_positions_GGL is not yet implemented")
        return window_positions_GGl

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
        #TO DO WEYL: This will need to be adjusted as well, by multiplying with the matter growth function squared
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
        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
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
        # NOTE WEYL PROJECT: This will be the total positions window function used for galaxy clustering; no modifications are needed here, as long as get_window_positions and get_window_magnification are properly modified
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
    
      def get_window_GGL(self, z) -> np.ndarray:
        # NOTE WEYL PROJECT: This will be the total positions window function used for galaxy clustering; no modifications are needed here, as long as get_window_positions_GGL(z) has been implemented
        window = self.get_window_positions_GGL(z) + self.get_window_magnification(z)
        return window

    