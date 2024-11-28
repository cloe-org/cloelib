# cloelite imports
from cloelite.cosmology.cosmology import Perturbations

# General imports
from typing import Protocol, Union
import numpy as np # type: ignore

"""

## Notes:

- Tracer protocol to implement different window functions

"""

class Tracer(Protocol):
    perturbations: Perturbations

    def _window_integrand(self, z: float, zprime: Union[float, np.ndarray]) -> np.ndarray:
        """
        Window integrand method.

        Parameters
        ----------
        zprime: float or numpy.ndarray
            Redshift parameter that will be integrated over
        z: float
            Redshift at which kernel is being evaluated

        Returns
        -------
        window_integrand: np.ndarray
        """
        ...

    def _get_prefactor(self, ell: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the needed prefactor in Limber approximation.

        Parameters
        ----------
        ell: float or numpy.ndarray of float
           :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        Pre-factor: float or numpy.ndarray of float
           Value(s) of the prefactor at the given :math:`\ell`
        """
        ...

    def get_window(self, z: float) -> np.ndarray:
        """
        Computes general window(s) given the selected tracer.

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """
        ...
