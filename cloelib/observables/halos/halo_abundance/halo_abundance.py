"""
- Introducing a protocol for the halo statistics part that we might have many versions of it.
"""

# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np  # type: ignore

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class HaloAbundance(Protocol):
    def f_sigma_nu(self, z: T, M: T) -> T:
        r"""Computes the halo multiplicity function."""
        ...

    def bias(self, z: T, M: T) -> T:
        r"""Computes the halo bias."""
        ...
