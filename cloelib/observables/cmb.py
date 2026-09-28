"""
Module implementing CMB lensing.

This class is compatible with the Tracer protocol.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations

# General imports
import jax.numpy as np  # type: ignore


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


class CMBLensingTracer:
    """Class for the kernel for CMB Lensing convergence."""

    def __init__(
        self,
        perturbations: Perturbations,
        z: np.ndarray,
    ):
        r"""
        Initialize the class instance.

        Parameters
        ----------
        perturbations : object
            An object from NonLinearPerturbations class
        z : np.ndarray
            A 1-dimensional array used to perform line-of-sight integration.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        self.n_z_bins = 1
        # This is to add the necessary prefactor to shear
        self.prefact_toggle = 0

    def get_window(self, z):
        r"""Compute the Window.

        Computes CMB lensing window function

        .. math::
            W^{\kappa}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z)
            f_K\left[\tilde{r}(z)\right]
            \frac{f_K\left[\tilde{r}(z_*) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z_*)\right]}\\

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
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
        rz = self.background.comoving_distance(z)
        z_star = self.background.z_star
        rz_star = self.background.comoving_distance(z_star)
        efficiency = 1 - rz / rz_star
        result = factor * efficiency
        return np.expand_dims(result, 0)
