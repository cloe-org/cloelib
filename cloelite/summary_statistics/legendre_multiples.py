# cloelite imports
from cloelite.cosmology.cosmology import Background, Perturbations

# General imports
from typing import Protocol, Union, TypeVar, Optional, Generic
import numpy as np  # type: ignore
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

"""

## Notes:

- Legendre multiples protocol to implement different spectro external codes

"""

class LegendreMultipoles(Protocol, Generic[T]):
    @property
    def linear_perturbations(self) -> Perturbations:
        """
        Stores perturbations obj
        """
        ...

    @property
    def background_fiducial(self) -> Background:
        """
        Stores background obj
        """
        ...

    @property
    def redshifts(self) -> T:
        """
        Set mean redshifts bins
        """
        ...
    
    @property
    def sigmaz(self) -> T:
        """
        Set redshifts bin error
        """
        ...

    @property
    def fout(self) -> T:
        """
        Set redshifts bin error
        """
        ...

    @property
    def nbar(self) -> T:
        """
        Set redshifts bin error
        """
        ...

    def power_multipoles(self, k: T,
                         ells: T, **args) -> T:
        r"""Power spectrum Legendre multipoles
        Parameters
        ----------
        k: jax.numpy.ndarray or numpy.ndarray
            Wavenumber
        ells: jax.numpy.ndarray or numpy.ndarray
            Legendre multipole order
        args: EFTofLSS parameters

        Returns
        -------
        multipoles: jax.numpy.ndarray or numpy.ndarray
            Power spectrum Legendre multipoles
        """
        ...