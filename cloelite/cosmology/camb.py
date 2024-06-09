# cloelite imports
from cosmology import Cosmology

# General imports
import numpy as np 

"""
## Author:
    **Name**: G. Canas-Herrera & M. Bonici  
    **Date**: June 9, 2024

## Notes:

- We make it sufficiently general to interface with CAMB keeping the structure by Cosmology

"""

class CAMBCosmology(Cosmology):
    """
    A class to interface with CAMB inheriting from Cosmology parent class

    """

    @property
    def hubble_parameter(self, zs) -> np.ndarray:
        """
        Retrieves the Hubble parameter as a function of redshift using CAMB.

        Parameters:
        -----------
        zs: numpy.ndarray
            Redshifts for the matter density

        Returns:
        --------
        Hubble parameter: numpy.ndarray
            Hubble parameter as a function of redshift
        """
        # Use CAMB to calculate the Hubble parameter
        # Replace the following line with actual CAMB calculation
        return 