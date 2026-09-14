"""Tracer protocol for different window functions."""

# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
from typing import Protocol, Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp  # type: ignore

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class Tracer(Protocol):
    """Tracer protocol to implement window functions."""

    @property
    def perturbations(self) -> Perturbations:
        """Store perturbations obj."""
        ...

    @property
    def n_z_bins(self) -> int:
        """Number of tomographic redshift bins carried by this tracer.

        Not used by the tracer itself, but required by
        `AngularTwoPoint.get_Cl` to size its output. Declared here so it is
        part of the documented contract rather than an implicit assumption.
        """
        ...

    @property
    def z(self) -> T:
        """Redshift grid this tracer was built on.

        Required by `AngularTwoPoint.get_Cl`, which reads `tracer1.z` as the
        integration grid for the Limber integral.
        """
        ...

    @property
    def prefact_toggle(self) -> int:
        """Whether the spin-2-to-convergence Limber prefactor applies (1) or not (0).

        Required by `AngularTwoPoint.get_Cl`. `ShearTracer`/`CMBLensingTracer`
        set this to 1, `PositionsTracer` to 0.
        """
        ...

    def _window_integrand(self, z: T, zprime: T) -> T:
        """
        Window integrand method.

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
        ...

    def _get_prefactor(self, ell: T) -> T:
        r"""
        Compute the needed prefactor in Limber approximation.

        Parameters
        ----------
        ell: float or numpy.ndarray of float
           :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        Pre-factor: float or numpy.ndarray of float
           Value(s) of the prefactor at the given :math:`\ell`
        """
        ...

    def get_window(self, z: T) -> T:
        """
        Compute general window(s) given the selected tracer.

        Args:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        ...
