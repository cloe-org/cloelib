"""Module for functions related to redshifts."""

import jax
import numpy as np
from functools import lru_cache


def ensure_z_zero_included(redshifts: np.ndarray) -> np.ndarray:
    """
    Ensure z=0 is in redshift array for sigma8(z=0) computation.

    This utility function checks if z=0 is present in the redshift array,
    and adds it if missing. This is necessary for computing sigma8(z=0)
    which is required for cosmological analyses.

    Parameters
    ----------
    redshifts : np.ndarray
        Array of redshift values

    Returns
    -------
    np.ndarray
        Redshift array with z=0 included (sorted)
    """
    z_min = redshifts.min()
    if z_min > 0.001:
        return np.sort(np.append(redshifts, 0.0))
    else:
        return redshifts


def check_nonzero_redshifts(z):
    """Raise `ValueError` if the redshift grid `z` contains zero.

    A zero redshift breaks the Limber integration. The check runs in plain
    NumPy on a concrete grid, so a tracer constructor holding it can still be
    traced by `jax.jit`, where even a constant `jnp` array would otherwise be
    compared as a tracer. A traced `z` is not checked.
    """
    if isinstance(z, jax.core.Tracer):
        return
    if np.any(np.asarray(z) == 0.0):
        raise ValueError(
            "One of the z array elements is equal to zero, breaking Limber integration."
        )
