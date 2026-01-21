import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import Union, TypeVar

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@jit
def stretch_dndz_jax(dndz: T, z: T, width: T) -> T:
    """
    Stretch redshift distributions by a bin-specific stretch parameter using JAX-compatible interpolation.

    This function interpolates each redshift distribution in `dndz` from a stretched redshift grid
    `widths[i]*(z-zmean) + zmean` back onto the original `z` grid, allowing redshift bin stretches in auto-diff pipelines.

    Parameters
    ----------
    dndz : Union[jnp.ndarray, np.ndarray]
        Array of shape (N_bins, N_z), representing redshift distributions for each bin.
    z : Union[jnp.ndarray, np.ndarray]
        1D array of redshift values corresponding to the columns of `dndz`.
    width : Union[jnp.ndarray, np.ndarray]
        1D array of length N_bins specifying redshift stretches per bin.

    Returns
    -------
    Union[jnp.ndarray, np.ndarray]
        Stretched redshift distributions, same shape as `dndz`, interpolated and zero-padded where needed.

    Notes
    -----
    - Interpolation outside bounds is filled with zero.
    - `dndz` is being normalized _by this function_ .
    - JAX-compatible and JIT-compiled for use in differentiable models.
    """

    def interp_single_bin(i):
        nz_i = dndz[i]
        width_i = width[i]
        norm_nz = jnp.sum(nz_i)
        z_mean = jnp.sum(z * nz_i) / (norm_nz + 1e-10)
        z_stretched = width_i * (z - z_mean) + z_mean
        return jnp.interp(z, z_stretched, nz_i, left=0.0, right=0.0)

    bins = dndz.shape[0]
    stretched = jnp.stack([interp_single_bin(i) for i in range(bins)], axis=0)
    # do the cumulative trapezoidal integral (notice the different treatment of first and last point)
    normalization = (
        -0.5 * (stretched[:, 0] + stretched[:, -1]) + jnp.sum(stretched, axis=1)
    ) * (z[1] - z[0])
    # normalization = jnp.where(normalization==0.0, 1.0, normalization)
    return stretched / (normalization[:, None])
