"""SpectroPower protocol."""

# cloelib imports
from cloelib.cosmology.cosmology import Background

# General imports
from typing import Protocol, Union, TypeVar
import numpy as np
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

class SpectroPower(Protocol):
    r"""Protocol to define the $P(k,\mu)$ interface."""

    @property
    def background(self) -> Background:
        """Attribute to store background object."""
        ...

    def Pk2d_rsd(self, k: T, mu: T, **args) -> T:
        r"""2D power spectrum from couplings of density and velocity fields.

        Parameters:
          k (ndarray): Wavenumber. Can be NumPy (`numpy.ndarray`) or JAX (`jax.numpy.ndarray`) ndarray.
          mu (numpy.ndarray|jax.numpy.ndarray): Angle (cosinus) to the line of sight
        
        Returns:
          Pk2d_rsd (numpy.ndarray|jax.numpy.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        ...

    def Pk2d_term_rsd(self, k: T, mu: T, **args) -> T:
        r"""2D power spectrum for a subset of specific term of the loop expansion.

        Parameters:
          k (numpy.ndarray|jax.numpy.ndarray): Wavenumber
          mu (numpy.ndarray|jax.numpy.ndarray): Angle (cosinus) to the line of sight

        Returns: 
          Pk2d_term_rsd (numpy.ndarray|jax.numpy.ndarray): 2D power spectrum of specific terms
        """
        ...
