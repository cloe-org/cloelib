# cloelite imports
from cloelite.cosmology.cosmology import Background

# General imports
from typing import Protocol, Union, TypeVar, Optional
import numpy as np  # type: ignore
from copy import deepcopy

# Cosmology imports
try:
    from comet import comet
except ImportError:
    raise ImportError("Comet could not be imported or initialised.")

"""

## Notes:

- Interface of Legendre Multiples with Comet

"""

class CometSpectroPower:
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
        self.parameters['z'] = redshifts
        self.parameters['ns'] = self.background.ns
        self.parameters['As'] = self.background.As/1e-9
        self.parameters['w0'] = self.background.w0
        self.parameters['wa'] = self.background.wa
        self.parameters['h'] = self.background.h
        self.parameters['wc'] = self.background.Omega_cdm0*self.background.h**2
        self.parameters['wb'] = self.background.Omega_b0*self.background.h**2
        self.parameters['cnlo'] = 0

        #initialise Comet
        self.comet_inst = comet(model = self.NLmodel, bias_basis='AssBauGre')

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

        if self.NLmodel == 'VDG_infty':
            f = self.comet_inst.params['f']
            sigmav = self.comet_inst.params['sv']
            num = (f * k * mu)**2
            den = 1.0 + num * self.parameters['avir']**2
            W_damping = 1.0 / np.sqrt(den) * np.exp(
                -num * sigmav**2 / den)
        else:
            W_damping = 1.0
        return self.comet_inst.Pk2d(k=k, mu=mu, params=self.parameters,
                                    de_model='w0wa') * W_damping

        
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
        if self.NLmodel == 'VDG_infty':
            f = self.comet_inst.params['f']
            sigmav = self.comet_inst.params['sv']
            num = (f * k * mu)**2
            den = 1.0 + num * self.parameters['avir']**2
            W_damping = 1.0 / np.sqrt(den) * np.exp(
                -num * sigmav**2 / den)
        else:
            W_damping = 1.0
        return self.comet_inst.PX2d(k=k, mu=mu, params=self.parameters, X=X,
                                    de_model='w0wa')        