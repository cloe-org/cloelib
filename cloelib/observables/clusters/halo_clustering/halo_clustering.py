# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np  # type: ignore

"""
- Introducing a protocol for the halo statistics part that we might have many versions of it.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class HaloClustering(Protocol):
    def power_spectrum_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        r"""Computes the halo multiplicity function."""
        ...
