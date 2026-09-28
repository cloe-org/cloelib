"""GW weak-lensing tracer: `GWWeakLensingTracer` and its Contributions.

Compatible with the Tracer protocol. Counterpart to `gw.number_counts`, which
holds `GWNumberCountsTracer`.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson, simps
from cloelib.auxiliary.systematics import shift_dndz_jax, stretch_dndz_jax

# General imports
import jax.numpy as np
import jax
import interpax

# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


class GWWeakLensingContribution:
    """Weak-lensing convergence kernel term of `GWWeakLensingTracer.get_window()`."""

    def __init__(self, tracer: "GWWeakLensingTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window_lensing(z)


class GWWeakLensingTracer:
    """Tracer for GW weak lensing."""

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
          perturbations (Perturbations): Perturbation backend providing a
            compatible background cosmology.
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of GW sources as a function of redshift, with shape
            `(n_bins, n_z)`. It is expected to be normalised.
          z (np.ndarray): A 1 dimensional array representing the evenly sampled, non-zero redshift grid with shape
            `(n_z,)` corresponding to the `dndz` array.
          nuisance_params (dict): Redshift-shift and width parameters for every bin.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        self.nuisance_params = nuisance_params
        # The radial kernel is scalar convergence, so only the GW
        # weak-lensing field response applies in AngularTwoPoint.
        self.prefact_toggle = 0
        self.gw_prefact_toggle = 1
        self.dz_gw_i = [
            self.nuisance_params[f"dz_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_gw_i = [
            self.nuisance_params[f"width_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.n_z_bins = dndz.shape[0]
        self.dndz = dndz
        # Correct dndz for width_gw
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_gw_i)
        # Correct dndz_stretched for dz_gw
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_gw_i)

        self.lensing = GWWeakLensingContribution(self)

    def get_contributions(self):
        """Return this tracer's window as its separable Contribution terms.

        Returns:
          contributions (tuple): `(self.lensing,)`.
        """
        return (self.lensing,)

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

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Parameters:
          z (np.ndarray): 1D array of redshift values (must be evenly spaced). Used to compute comoving distances
            and define integration domain.

        Returns:
          (np.ndarray): 2D array of shape (N_bins, len(z)) representing the lensing efficiency kernel W(z)
            for each redshift bin over the evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz_shifted` is expected to have shape (N_bins, len(z)) and be normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_lensing(self, z) -> np.ndarray:
        r"""GW weak-lensing convergence kernel.

        Calculates the GW weak-lensing kernel for a given tomographic bin
        distribution. The underlying geometry is the scalar convergence
        kernel. The observable-dependent harmonic response is applied in
        `AngularTwoPoint`. There is no magnification bias,
        intrinsic alignment, or multiplicative shear factor in this tracer.

        $$
            W_{i}^{\kappa}(z) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z)
            \chi(z)
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm GW}(z^{\prime})
            \frac{\chi(z^{\prime}) - \chi(z)}
            {\chi(z^{\prime})}}\\
        $$

        Parameters:
          z (numpy.ndarray): One-dimensional redshift grid matching the source distributions.

        Returns:
          (numpy.ndarray): Numpy array of convergence kernel values of shape (n_bins, n_z)
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
          z (np.ndarray): Redshift grid at which the window kernel is being evaluated.

        Returns:
          window (np.ndarray): GW weak-lensing windows with shape
            `(n_bins, n_z)`.
        """
        total_window = sum(c.compute_kernel(z) for c in self.get_contributions())
        return total_window
