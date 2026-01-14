"""
Module with one class to compute the BNT matrix.
"""

import numpy as np
import jax.numpy as jnp
from cloelib.cosmology.cosmology import Background
from typing import List, TypeVar, Union

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class BNTMatrixCalculator:
    """Class to compute the BNT matrix."""

    def __init__(
        self,
        dndz_list: List[T],
        z: T,
        background: Background,
    ):
        r"""
        Initialize the class instance.

        Parameters
        ----------
        dndz_list : List[np.ndarray or jnp.ndarray]
            A list of 1d-arrays representing the number density distribution of galaxies as a function of redshift.
            Each list member is expected to be normalised.
        z : np.ndarray or jnp.ndarray
            A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
        background : Background
            Cosmological background instance.
        """

        # Convert everything to NumPy for internal computations
        self.z = np.asarray(z)
        self.dndz_list = [np.asarray(nz) for nz in dndz_list]

        self.background = background
        self.nbins = len(self.dndz_list)

        if len(self.dndz_list) < 3:
            raise ValueError(
                "BNTMatrixCalculator requires at least 3 tomographic bins to compute the matrix."
            )
        if np.any(self.z == 0.0):
            raise ValueError(
                "One of the z array elements is equal to zero, breaking the BNT computation."
            )
        for nz in self.dndz_list:
            if nz.shape != self.z.shape:
                raise ValueError(
                    f"Each dndz array must have shape {self.z.shape}, "
                    f"but found {nz.shape}."
                )

        self.chi = self.background.comoving_distance(self.z)

    def get_bnt_matrix(self) -> np.ndarray:
        """
        Compute the BNT matrix.

        The BNT matrix is a lower–triangular linear transformation acting on tomographic
        weak-lensing kernels, designed to construct linear combinations of shear kernels
        that are localised in comoving distance.

        Returns
        -------
        np.ndarray
            The BNT transformation matrix with shape ``(nbins, nbins)``, always
            lower triangular with ones on the diagonal.
        """

        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        nz_arr = np.stack(self.dndz_list, axis=0)
        A = trapz(nz_arr, self.z, axis=1)
        B = trapz(nz_arr / self.chi[None, :], self.z, axis=1)

        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1, 0] = -1.0

        for i in range(2, self.nbins):
            mat = np.array([[A[i - 1], A[i - 2]], [B[i - 1], B[i - 2]]])
            rhs = -np.array([A[i], B[i]])
            try:
                soln = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "BNT matrix construction failed: non-invertible 2x2 "
                    f"system encountered at tomographic bin index {i}."
                ) from exc
            BNT_matrix[i, i - 1] = soln[0]
            BNT_matrix[i, i - 2] = soln[1]

        return BNT_matrix
