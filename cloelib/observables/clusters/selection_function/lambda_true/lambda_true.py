# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np  # type: ignore

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class LambdaTrue(Protocol):
    def prob_richness_given_mass(self, z, M, lambda_true):
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
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        prob_richness_given_mass: numpy.ndarray
            prob_richness_given_mass[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        ...
