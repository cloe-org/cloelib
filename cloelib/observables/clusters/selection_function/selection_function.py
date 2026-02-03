# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np  # type: ignore

"""
- Introducing a protocol for the halo statistics part that we might have many versions of it.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class SelectionFunction(Protocol):
    def prob_true_richness_given_mass(self, z: T, M: T, Lambda: T) -> T:
        r"""
        Proxy - mass relation PDF.

        Computes the theoretical richness probability distribution
        at the requested true mass, redshift, and richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.
        Lambda: numpy.ndarray
            True richness points.

        Returns
        -------
        prob_true_richness_given_mass: numpy.ndarray
            prob_true_richness_given_mass[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        ...
