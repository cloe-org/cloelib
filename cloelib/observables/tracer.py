"""Tracer protocol for different window functions."""
# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
from typing import Protocol, Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp # type: ignore

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class Tracer(Protocol):
    """Tracer protocol to implement window functions."""

    @property
    def perturbations(self) -> Perturbations:
        """Store perturbations obj."""
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

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """
        ...
