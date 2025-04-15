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
    r"""Class to retrieve :math:`P(k,\mu)` (including redshift-space
    distortions) with the EFT model from COMET

    Parameters
    ----------
    background: Background
        Background class containing cosmology and background distances
    RSD_parameters: dict
        Dictionary containing bias and counterterm parameters
    redshift: float
        Redshift at which to evaluate :math:`P(k,\mu)`
    """

    def __init__(self, background: Background, RSD_parameters: dict,
                 redshift: float):
        r"""Class constructor
        """
        self.background = background

        self.parameters = {}
        self.parameters['wc'] = self.background.Omega_cdm0 * self.background.h**2
        self.parameters['wb'] = self.background.Omega_b0 * self.background.h**2
        self.parameters['Mnu'] = self.background.mnu
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
        Returns
        -------
        Pk2d_rsd: np.ndarray
            2D power spectrum from couplings of density and velocity fields
        """
        return np.squeeze(
            comet_inst.P2d_nostoch(k=k[:, :, np.newaxis], mu=mu[:, np.newaxis],
                                   params=self.parameters, de_model='w0wa'))

    def Pk2d_X_rsd(self, k: np.ndarray, mu: np.ndarray, X_list: list) -> np.ndarray:
        r"""2D power spectrum for the specific diagram X of the loop expansion
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        X: list
            Identifiers of loop diagrams
        Returns
        -------
        PX2d_rsd: np.ndarray
            2D power spectrum of term X
        """
        Pk2d = comet_inst.PX_2d(k=k[:, :, np.newaxis], mu=mu[:, np.newaxis],
                                params=self.parameters, X_list=X_list,
                                de_model='w0wa')
        return np.squeeze(Pk2d)
