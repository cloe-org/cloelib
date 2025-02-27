# cloelib imports
from cloelib.cosmology.cosmology import Background 

# General imports
from typing import Protocol, Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp # type: ignore

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

class SpectroPower(Protocol):
    @property
    def background(self) -> Background: #dependency on Background
        """
        Stores background obj
        """
        ...

    @property
    def background_fiducial(self) -> Background: #dependency on Background
        """
        Stores background fiducial obj
        """
        ...

    def Pk2d_rsd(self, k: T, mu: T, **args) -> T: 
        #for a class to be compatible with this protocol
        # it must always return _Pk2d_rsd
        r"""2D power spectrum from couplings of density and velocity fields
        Parameters
        ----------
        k: numpy.ndarray or jax.numpy.ndarray
            Wavenumber
        mu: numpy.ndarray or jax.numpy.ndarray
            Angle (cosinus) to the line of sight
        **args
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        Pk2d_rsd: numpy.ndarray or jax.numpy.ndarray
            2D power spectrum from couplings of density and velocity fields
        """
        ...

    def Pk2d_X_rsd(self, k: T, mu: T, **args) -> T: 
        #for a class to be compatible with this protocol
        # it must always return _Pk2d_rsd
        r"""2D power spectrum for the specific diagram X
        Parameters
        ----------
        k: numpy.ndarray or jax.numpy.ndarray
            Wavenumber
        mu: numpy.ndarray or jax.numpy.ndarray
            Angle (cosinus) to the line of sight
        **args
            Ensemble of cosmological and nuisance parameters
        X: str
            Identifier of loop diagram
        Returns
        -------
        Pk2d_X_rsd: numpy.ndarray or jax.numpy.ndarray
            2D power spectrum of term X
        """
        ...
