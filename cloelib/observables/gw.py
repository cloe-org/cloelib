"""
Module with two classes for gravitational-wave observables: GW number counts and
GW weak lensing.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson, simps
from cloelib.auxiliary.systematics import shift_dndz_jax, stretch_dndz_jax

# General imports
import jax.numpy as np  # type: ignore
import jax  # type: ignore
import interpax  # type: ignore
import jax.lax as lx


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


class GWNumberCountsTracer:
    """Class to define the kernel for angular GW number counts."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        gw_bias_model: str,
        nuisance_params: dict,
    ):
        r"""
        Initialize the class instance.

        Parameters:
          perturbations (Perturbations): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density
            distribution of GW sources as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values
            corresponding to the `dndz` array.
          gw_bias_model (str): A string specifying the model used to describe
            the GW source bias.
          nuisance_params (dict): A dictionary containing additional parameters
            that are not directly related to the cosmological model but may
            affect the GW observations.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        if dndz.ndim != 2:
            raise ValueError("dndz must have shape (n_bins, n_z).")
        if dndz.shape[1] != z.shape[0]:
            raise ValueError("The last dimension of dndz must match the z grid.")
        if gw_bias_model not in ("per_bin", "per_bin_int", "poly"):
            raise ValueError(
                "gw_bias_model must be 'per_bin', 'per_bin_int', or 'poly'."
            )
        if dndz.shape[0] > z.shape[0]:
            raise ValueError("The number of tomographic bins cannot exceed len(z).")

        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        # GW number counts are scalar, like galaxy positions.
        self.prefact_toggle = 0
        self.nuisance_params = nuisance_params
        self.dz_gw_i = [
            self.nuisance_params[f"dz_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_gw_i = [
            self.nuisance_params[f"width_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for width_gw.
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_gw_i)
        # Correct dndz_stretched for dz_gw.
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_gw_i)
        self.flags = {"gw_bias_model": gw_bias_model}
        self.n_z_bins = dndz.shape[0]

        # Use the same bias-model structure and defaults as PositionsTracer.
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_GW_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax requires the same size for all cases, so pad here and use
            # only the first n_z_bins values later.
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        def per_bin_int_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_GW_bin%d" % bin, 1.0)
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
                    nuisance_params.get("b1_GW_poly%d" % bin, 1.0)
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
                self.flags["gw_bias_model"] == "per_bin",
                self.flags["gw_bias_model"] == "per_bin_int",
                self.flags["gw_bias_model"] == "poly",
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        self.bias_array = [per_bin_case, per_bin_int_case, poly_case][index]()

    def get_window_number_counts(self, z) -> np.ndarray:
        r"""GW number-count window function.

        Implements the GW number-count source-density window, analogous to
        `PositionsTracer.get_window_positions`, with the galaxy bias replaced
        by the GW source bias.

        $$
            W_i^{\rm GW-NC}(z) =
            b_i^{\rm GW}(z)\,n_i^{\rm GW}(z)\frac{H(z)}{c}
        $$

        Parameters:
          z (numpy.ndarray|float): Redshift at which to evaluate distribution
            (array of `float` or `float`)

        Returns:
          window_number_counts (np.ndarray): Window function for angular GW
            number counts
        """

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.background.hubble_parameter(z)
                / c_0
            )
            return window

        def z_func_case():
            window = (
                self.bias_array[None, :]
                * self.dndz_shifted
                * self.background.hubble_parameter(z)
                / c_0
            )
            return window

        conditions = np.array(
            [
                self.flags["gw_bias_model"] == "per_bin",
                self.flags["gw_bias_model"] in ["per_bin_int", "poly"],
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        window_number_counts = [per_bin_case, z_func_case][index]()

        return window_number_counts

    def get_window(self, z) -> np.ndarray:
        """
        Compute the angular GW number-count window function.

        Parameters:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        return self.get_window_number_counts(z)


class GWWeakLensingTracer:
    """Class for the kernel for GW weak lensing."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
    ):
        r"""
        Initialize the class instance.

        Parameters:
          perturbations (Perturbations): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density
            distribution of GW sources as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values
            corresponding to the `dndz` array.
          nuisance_params (dict): A dictionary containing additional parameters
            that are not directly related to the cosmological model but may
            affect the GW observations.

        Notes
        -----
        This tracer uses the same geometric lensing-efficiency structure as
        `ShearTracer.get_lensing_efficiency`. It does not use the galaxy
        magnification-bias factor from `PositionsTracer`, and it does not apply
        shear intrinsic-alignment or multiplicative-bias terms, because GW-WL is
        the scalar GW convergence field.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        if dndz.ndim != 2:
            raise ValueError("dndz must have shape (n_bins, n_z).")
        if dndz.shape[1] != z.shape[0]:
            raise ValueError("The last dimension of dndz must match the z grid.")

        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        self.nuisance_params = nuisance_params
        # GW convergence is scalar, unlike spin-2 galaxy shear.
        self.prefact_toggle = 0
        self.dz_gw_i = [
            self.nuisance_params[f"dz_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_gw_i = [
            self.nuisance_params[f"width_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.n_z_bins = dndz.shape[0]
        self.dndz = dndz
        # Correct dndz for width_gw.
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_gw_i)
        # Correct dndz_stretched for dz_gw.
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_gw_i)

    def get_lensing_efficiency_bin(self, z, bin_idx):
        """Compute the GW lensing efficiency in a redshift bin."""
        interpolator = interpax.Akima1DInterpolator(
            self.z, self.dndz_shifted[bin_idx, :]
        )
        x = np.linspace(0.0, 4, 200)
        y = self.background.comoving_distance(x)
        rx_interp = interpax.Akima1DInterpolator(x, y)
        f1 = jax.jit(lambda x: interpolator(x))
        f2 = jax.jit(lambda x: interpolator(x) / rx_interp(x))
        integral_1 = simps(f1, z, 3.0)
        integral_2 = simps(f2, z, 3.0)
        efficiency = integral_1 - integral_2 * self.background.comoving_distance(z)
        return efficiency

    def get_lensing_efficiency(self, z) -> np.ndarray:
        r"""
        Compute the GW lensing efficiency kernel for each redshift bin.

        This function calculates the geometric convergence kernel for a given
        redshift grid ``z``. The observable-dependent response is applied in
        :meth:`get_window_lensing`.

        Parameters:
          z (np.ndarray): 1D array of redshift values (must be evenly spaced).
            Used to compute comoving distances and define integration domain.

        Returns:
          (np.ndarray): 2D array of shape (N_bins, len(z)) representing the
            lensing efficiency kernel for each GW redshift bin over the
            evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`)
          for integration.
        - `self.dndz` is expected to have shape (N_bins, len(z)) and be
          normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        return np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz

    def get_window_lensing(self, z) -> np.ndarray:
        r"""GW weak-lensing convergence kernel.

        Calculates the GW weak-lensing kernel for a given tomographic bin
        distribution. The underlying geometry is the scalar convergence
        kernel defined in the paper. There is no galaxy magnification-bias,
        intrinsic-alignment, or multiplicative-shear factor in this tracer.

        $$
            W_i^{\rm GW-WL}(z) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm GW}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\
        $$

        Parameters:
          z (numpy.ndarray): Redshift at which weight is evaluated (array of
            `float`).

        Returns:
          (numpy.ndarray): 1-D Numpy array of GW weak-lensing kernel values for
            specified bin at specified scale for the redshifts defined in z
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
        efficiency = self.get_lensing_efficiency(z)
        return np.einsum("ij, j->ij", efficiency, factor)

    def get_window(self, z) -> np.ndarray:
        """
        Compute the angular GW weak-lensing window function.

        Parameters:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        return self.get_window_lensing(z)
