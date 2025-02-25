# cloelite imports
from cloelib.cosmology.cosmology import Background

# General imports
from typing import Protocol, Union, TypeVar, Optional
import numpy as np  # type: ignore
from copy import deepcopy

# external codes imports
#try:
    #import PBJ
#except ImportError:
#    raise ImportError("PBJ could not be imported or initialised.")

"""

## Notes:

- Interface of Legendre Multiples with Comet

"""

class PBJSpectroPower:
    def __init__(self, 
                 background: Background, #this will initialise the cosmo
                 background_fiducial: Background, #this will initialise the cosmo
                 # model
                 redshifts: np.ndarray,
                 NLmodel: str,
                 **args):
        
        self.background = background
        self.background_fiducial = background_fiducial
        self.NLmodel = NLmodel
        self.parameters = args
        # do the translations to your names below

        #create instance of PBJ
        #self.pbj_inst = 

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        Pk2d_rsd: np.ndarray
            2D power spectrum from couplings of density and velocity fields
        """
        ...
        
        #return whatever

        
    def Pk2d_X_rsd(self, k: np.ndarray, mu: np.ndarray, X: str) -> np.ndarray:
        r"""2D power spectrum for the specific diagram X
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        X: str
            Identifier of loop diagram
        Returns
        -------
        PX2d_rsd: np.ndarray
            2D power spectrum of term X
        """
        ...
        #return whatever      