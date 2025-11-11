"""
Module with one class to compute the BNT matrix.
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from typing import List, Sequence
from cloelib.cosmology.cosmology import Background


def _pick_xp(arr):
    """Return numpy or jax.numpy based on the array type (defaults to numpy)."""
    if jnp is not None and isinstance(arr, jnp.ndarray):
        return jnp
    return np


def _is_jax_array(arr) -> bool:
    return jnp is not None and isinstance(arr, jnp.ndarray)


def _trapz(xp, y, x, axis=-1):
    """Backend-agnostic trapezoidal integration."""
    if xp is np:
        return np.trapz(y, x, axis=axis)
    else:
        # JAX version
        dx = x[1:] - x[:-1]
        # reshape dx for broadcasting
        sl = [None] * y.ndim
        sl[axis] = slice(None)
        dx = dx[tuple(sl)]
        return xp.sum((y[..., 1:] + y[..., :-1]) * dx * 0.5, axis=axis)


class BNT:
    """Class to compute the BNT matrix."""

    def __init__(
        self,
        dndz_list: List[object],  # accepts np.ndarray or jnp.ndarray
        z: object,  # accepts np.ndarray or jnp.ndarray
        fid_parameters: dict,
        background: Background,
    ):
        r"""
        Initialize the class instance.

        Parameters
        ----------
        dndz_list : List[np.ndarray or jnp.ndarray]
            A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            Each list member is expected to be normalised.
        z : np.ndarray or jnp.ndarray
            A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
        fid_parameters : dict
            A dictionary of the fiducial cosmology parameters.
        background : Background
            Cosmological background instance.
        """

        # --- backend selection & consistency check ---
        xp = _pick_xp(z)
        z_is_jax = _is_jax_array(z)
        list_is_jax = all(_is_jax_array(nz) for nz in dndz_list)
        list_is_np = all(isinstance(nz, np.ndarray) for nz in dndz_list)

        # If mixed types across z and dndz_list, force user to make them consistent
        if z_is_jax and not list_is_jax:
            raise TypeError(
                "Mixed array backends detected: z is JAX but some dndz are NumPy."
            )
        if (not z_is_jax) and not list_is_np:
            raise TypeError(
                "Mixed array backends detected: z is NumPy but some dndz are JAX."
            )

        self.dndz_list = dndz_list
        self.z = z
        self.fid_parameters = fid_parameters
        self.background = background
        self.nbins = len(self.dndz_list)
        self._xp = xp  # remember backend

        if len(self.dndz_list) < 3:
            raise ValueError(
                "BNT requires at least 3 tomographic bins to compute the matrix."
            )
        if xp.any(self.z == 0.0):
            raise ValueError(
                "One of the z array elements is equal to zero, breaking the BNT computation."
            )
        if len(self.z.shape) != len(self.dndz_list[0].shape):
            raise ValueError(
                "The shape of the z array does not match the shape of the dndz array."
            )

        # background typically returns NumPy; convert to selected backend
        self.chi = self.background.comoving_distance(z)
        # self.chi = chi_np if xp is np else jnp.asarray(chi_np)  # type: ignore

    def get_matrix(self):
        """Compute the BNT matrix."""
        xp = self._xp

        A_list: Sequence[float] = []
        B_list: Sequence[float] = []
        for i in range(self.nbins):
            nz = self.dndz_list[i]
            A_list += [_trapz(xp, nz, self.z)]
            B_list += [_trapz(xp, nz / self.chi, self.z)]

        BNT_matrix = xp.eye(
            self.nbins,
            dtype=(
                self.dndz_list[0].dtype if hasattr(self.dndz_list[0], "dtype") else None
            ),
        )
        # JAX arrays are immutable; use .at for assignment
        if jnp is not None and xp is jnp:
            BNT_matrix = BNT_matrix.at[1, 0].set(-1.0)
        else:
            BNT_matrix[1, 0] = -1.0

        for i in range(2, self.nbins):
            mat = xp.array(
                [[A_list[i - 1], A_list[i - 2]], [B_list[i - 1], B_list[i - 2]]]
            )
            A = -1.0 * xp.array([A_list[i], B_list[i]])
            # Use solve instead of inv for numerical stability
            soln = xp.linalg.solve(mat, A)

            if jnp is not None and xp is jnp:
                BNT_matrix = BNT_matrix.at[i, i - 1].set(soln[0])
                BNT_matrix = BNT_matrix.at[i, i - 2].set(soln[1])
            else:
                BNT_matrix[i, i - 1] = soln[0]
                BNT_matrix[i, i - 2] = soln[1]

        return BNT_matrix
