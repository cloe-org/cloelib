"""
Module with two classes for gravitational-wave observables: GW number counts and
GW weak lensing.

Both classes are compatible with the Tracer protocol.
"""

from cloelib.auxiliary.math_utils import cached_stacked_simpson
from cloelib.auxiliary.systematics import shift_dndz_jax
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations

import jax.numpy as np  # type: ignore


c_0 = SPEED_OF_LIGHT / 1000


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
        if gw_bias_model not in ("per_bin", "poly"):
            raise ValueError("gw_bias_model must be 'per_bin' or 'poly'.")

        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        self.prefact_toggle = 0
        self.nuisance_params = nuisance_params
        self.dz_gw_i = [
            self.nuisance_params.get(f"dz_gw_{i + 1}", 0.0)
            for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        self.dndz_shifted = shift_dndz_jax(dndz, z, np.asarray(self.dz_gw_i))
        self.flags = {"gw_bias_model": gw_bias_model}
        self.n_z_bins = dndz.shape[0]

        # Using dict.get so I can provide a default since lax has to compile every branch of the conditional
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get(
                        "b1_GW_bin%d" % bin,
                        nuisance_params.get("gw_bias_%d" % (bin + 1), 1.0),
                    )
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        def poly_case():
            poly_order = 3
            bias_array = np.asarray(
                [
                    nuisance_params.get(
                        "b1_GW_poly%d" % bin,
                        nuisance_params.get(
                            "b%d_poly_GW" % bin,
                            nuisance_params.get("gw_bias_poly%d" % bin, 1.0),
                        ),
                    )
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
                self.flags["gw_bias_model"] == "poly",
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        self.bias_array = [per_bin_case, poly_case][index]()

    def _window_integrand(self, z, zprime) -> np.ndarray:
        r"""
        Window integrand.

        Returns the shifted GW source redshift distribution evaluated at
        `zprime`. This is the source-distribution part of the GW number-count
        window, analogous to the density part of `PositionsTracer`.

        Parameters:
          z (float): Redshift at which kernel is being evaluated
          zprime (float): Redshift parameter that will be integrated over

        Returns:
          window_integrand (np.ndarray):
        """
        del z
        return np.asarray(
            [np.interp(zprime, self.z, dndz_i) for dndz_i in self.dndz_shifted]
        )

    def _get_prefactor(self, ells) -> np.ndarray:
        r"""
        Compute the needed prefactor in Limber approximation.

        GW number counts are scalar observables, so no spin-dependent shear
        prefactor is applied.

        Parameters:
          ells (np.ndarray): Multipoles at which the prefactor is evaluated

        Returns:
          prefactor (np.ndarray):
        """
        return np.ones_like(ells)

    def get_window_number_counts(self, z) -> np.ndarray:
        r"""GW number-count window function.

        Implements the GW number-count source-density window, analogous to
        `PositionsTracer.get_window_positions`, with the galaxy bias replaced
        by the GW source bias.

        $$
            W_i^{\rm GWC}(z) =
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
                self.flags["gw_bias_model"] == "poly",
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
        shear intrinsic-alignment or multiplicative-bias terms, because GWL is
        the scalar GW amplitude-lensing field.
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
        self.prefact_toggle = 0
        self.dz_gw_i = [
            self.nuisance_params.get(f"dz_gw_{i + 1}", 0.0)
            for i in range(dndz.shape[0])
        ]
        self.n_z_bins = dndz.shape[0]
        self.dndz = dndz
        self.dndz_shifted = shift_dndz_jax(dndz, z, np.asarray(self.dz_gw_i))

    def _window_integrand(self, z, zprime) -> np.ndarray:
        r"""
        Window integrand.

        Returns the source distribution times the lensing geometry entering the
        GW weak-lensing efficiency. This is analogous to the geometric
        efficiency used by `ShearTracer`, but for the scalar GW amplitude
        lensing observable.

        Parameters:
          z (float): Redshift at which kernel is being evaluated
          zprime (float): Redshift parameter that will be integrated over

        Returns:
          window_integrand (np.ndarray):
        """
        dndz_primes = np.asarray(
            [np.interp(zprime, self.z, dndz_i) for dndz_i in self.dndz_shifted]
        )
        chi = self.background.comoving_distance(np.asarray([z]))[0]
        chi_prime = self.background.comoving_distance(np.asarray([zprime]))[0]
        geometry = np.where(zprime > z, (chi_prime - chi) / chi_prime, 0.0)
        return dndz_primes * geometry

    def _get_prefactor(self, ells) -> np.ndarray:
        r"""
        Compute the needed prefactor in Limber approximation.

        GW weak lensing is treated as a scalar amplitude-lensing observable, not
        as spin-2 galaxy shear, so no spin-dependent shear prefactor is applied.

        Parameters:
          ells (np.ndarray): Multipoles at which the prefactor is evaluated

        Returns:
          prefactor (np.ndarray):
        """
        return np.ones_like(ells)

    def get_lensing_efficiency(self, z) -> np.ndarray:
        r"""
        Compute the GW lensing efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel, using the same
        integration structure as `ShearTracer.get_lensing_efficiency`, for a
        given redshift grid `z`.

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
        r"""GW weak-lensing amplitude kernel.

        Calculates the GW weak-lensing kernel for a given tomographic bin
        distribution. This has the same geometric convergence kernel used by
        `ShearTracer.get_window_lensing`, but it is returned as a scalar GWL
        observable. There is no `magnification_bias` multiplier here; the
        magnification-bias factor in `PositionsTracer` belongs to galaxy number
        counts, not to the GW amplitude-lensing observable.

        $$
            W_i^{\rm GWL}(z) =
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
