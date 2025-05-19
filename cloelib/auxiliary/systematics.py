import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from typing import Union, TypeVar

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

@jit
def shift_dndz_jax(dndz: T, z: T, dz: T) -> T:
    """
    Shift redshift distributions by a bin-specific offset using JAX-compatible interpolation.

    This function interpolates each redshift distribution in `dndz` from a shifted redshift grid
    `z - dz[i]` back onto the original `z` grid, allowing redshift bin shifts in auto-diff pipelines.

    Parameters
    ----------
    dndz : Union[jnp.ndarray, np.ndarray]
        Array of shape (N_bins, N_z), representing redshift distributions for each bin.
    z : Union[jnp.ndarray, np.ndarray]
        1D array of redshift values corresponding to the columns of `dndz`.
    dz : Union[jnp.ndarray, np.ndarray]
        1D array of length N_bins specifying redshift shift per bin.

    Returns
    -------
    Union[jnp.ndarray, np.ndarray]
        Shifted redshift distributions, same shape as `dndz`, interpolated and zero-padded where needed.

    Notes
    -----
    - Interpolation outside bounds is filled with zero.
    - `dndz` is being normalized _by this function_ .
    - JAX-compatible and JIT-compiled for use in differentiable models.
    """
    def interp_single_bin(i):
        z_shifted = z - dz[i]
        return jnp.interp(z, z_shifted, dndz[i], left=0.0, right=0.0)

    bins = dndz.shape[0]
    shifted = jnp.stack([interp_single_bin(i) for i in range(bins)], axis=0)
    #do the cumulative trapezoidal integral (notice the different treatment of first and last point)
    normalization = (-0.5*(shifted[0]+shifted[-1])+jnp.sum(shifted))*(z[1]-z[0])
    return shifted/normalization
