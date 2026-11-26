"""
- Introducing a protocol for the halo statistics part that we might have many versions of it.
"""

# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class HaloClustering(Protocol):
    def power_spectrum_monopole_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        r"""Compute the halo power-spectrum monopole."""
        ...

    def power_spectrum_quadrupole_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        r"""Compute the halo power-spectrum quadrupole."""
        ...

    def power_spectrum_hexadecapole_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        r"""Compute the halo power-spectrum hexadecapole."""
        ...

    def power_spectrum_RSD_amplitude(self, z, k, z_obs_scatter, b_eff, mu):
        r"""Compute the halo redshift-space amplitude at fixed mu."""
        ...
