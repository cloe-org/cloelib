# file: bnt.py
import numpy as np
from cloelib.cosmology.cosmology import Background
from typing import List

class BNT:
    """Class to compute the BNT matrix."""

    def __init__(
        self,
        dndz_list : List[np.ndarray],
        z : np.ndarray,
        fid_parameters: dict,
        background : Background,
    ):
        r"""
        Initialize the class instance.

        Parameters
        ----------
        dndz_list : List[np.ndarray]
            A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            Each list member is expected to be normalised.
        z : np.ndarray
            A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
        fid_parameters : dict
            A dictionary of the fiducial cosmology parameters.
        background : Background
            Cosmological background instance.
        """

        self.dndz_list = dndz_list
        self.z = z
        self.fid_parameters = fid_parameters
        self.background = background
        self.nbins = len(self.dndz_list)

        if len(self.dndz_list) < 3:
            raise ValueError(
                "BNT requires at least 3 tomographic bins to compute the matrix."
            )
        if np.any(self.z == 0.0):
            raise ValueError(
                "One of the z array elements is equal to zero, breaking the BNT computation."
            )
        if len(self.z.shape) != len(self.dndz_list[0].shape):
            raise ValueError(
                "The shape of the z array does not match the shape of the dndz array."
            )
        self.chi  = self.background.comoving_distance(z)
        

    def get_matrix(self):
        """Compute the BNT matrix."""
        A_list = []
        B_list = []
        for i in range(self.nbins):
            nz = self.dndz_list[i]
            A_list += [np.trapz(nz, self.z)]
            B_list += [np.trapz(nz / self.chi, self.z)]


        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1,0] = -1.

        for i in range(2,self.nbins):
            mat = np.array([ [A_list[i-1], A_list[i-2]], [B_list[i-1], B_list[i-2]] ])
            A = -1. * np.array( [A_list[i], B_list[i]] )
            soln = np.dot(np.linalg.inv(mat), A)
            BNT_matrix[i,i-1] = soln[0]
            BNT_matrix[i,i-2] = soln[1]
        
        return BNT_matrix
















