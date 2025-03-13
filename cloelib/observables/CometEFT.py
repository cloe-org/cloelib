# cloelib imports
from cloelib.cosmology.cosmology import Background

# General imports
from typing import Protocol, Union, TypeVar, Optional
import numpy as np  # type: ignore
from copy import deepcopy

# Cosmology imports
try:
    from comet import comet # type: ignore
    comet_inst = comet(model='EFT', use_Mpc=True, bias_basis='AssBauGre')
except ImportError:
    raise ImportError("Comet could not be imported or initialised.")

"""

## Notes:

- Interface of Legendre Multiples with Comet

"""

class CometEFT_SpectroPower:
    def __init__(self, background: Background, RSD_parameters: dict,
                 redshift: float):

        self.background = background

        self.parameters = {}
        self.parameters['wc'] = self.background.Omega_cdm0 * self.background.h**2
        self.parameters['wb'] = self.background.Omega_b0 * self.background.h**2
        self.parameters['ns'] = self.background.ns
        self.parameters['h'] = self.background.h
        self.parameters['As'] = self.background.As * 1e9
        self.parameters['w0'] = self.background.w0
        self.parameters['wa'] = self.background.wa
        self.parameters.update(RSD_parameters)
        self.parameters['z'] = redshift

        self.redshift = redshift

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
        return comet_inst.Pk2d(k=k, mu=mu, params=self.parameters,
                               de_model='w0wa')

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
        return self.comet_inst.PX2d(k=k, mu=mu, params=self.parameters, X=X,
                                    de_model='w0wa')
