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


class CMBLensingContribution:
    """CMB-lensing convergence kernel term of `CMBLensingTracer.get_window()`.

    Lets `CMBLensingTracer` participate in the generalized
    per-contribution-pair spectrum engine
    (`cloelib.observables.photo.spectrum_engine`) whenever it's paired with
    a contribution that itself declares extra `SpectrumRequest`s (e.g.
    `NonLinearGalaxyBiasContribution`, `TATTContribution`). Without this,
    `getattr(tracer, "get_contributions", lambda: ())()` returns an empty
    tuple for `CMBLensingTracer`, which silently disables the generalized
    engine for any pairing involving CMB lensing (an empty
    `contributions2` means `spectrum_engine.needs_generalized_engine`'s
    `for c2 in contributions2` loop never runs) - falling back to the
    legacy path's plain `get_window`/matter-Pk integral even when the other
    side needs its own effective P(k,z) (e.g. `NonLinearGalaxyBiasContribution`'s
    amplitude-free kernel, which relies on `get_effective_pk` to supply the
    bias amplitude at all). Carries no `get_spectrum_requests` of its own -
    CMB lensing needs nothing beyond the plain matter Pk (or, paired with a
    density contribution, that contribution's own galaxy-matter cross
    spectrum) - so pairing CMB lensing with anything that itself declares
    no extra requests still takes the untouched legacy path, exactly as
    before this class existed.
    """

    def __init__(self, tracer: "CMBLensingTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window(z)


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
        self.convergence = CMBLensingContribution(self)

    def get_contributions(self):
        """Return this tracer's window as its separable Contribution terms.

        Returns:
          contributions (tuple): `(self.convergence,)`.
        """
        return (self.convergence,)

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
