# General imports
from typing import Protocol
import numpy as np  # type: ignore

"""
## Notes:
 
- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

class Background(Protocol):
    H0: float
    Omb: float
    Omc: float
    Omk: float
    As: float
    ns: float
    w: float
    wa: float
    gamma_MG: float
    # Variable to be able to keep args from background codes to perturbations
    # in reality, will we use it beyond CAMB/CLASS?
    # If it is not an array, not auto-diff
    interface_args: None

    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, ns: float, As: float, w: float, wa: float, gamma_MG: float):
        """
        A protocol to define background cosmology
        """
        ...

    def hubble_parameter(self, zs: np.ndarray, units: str = '1/Mpc') -> np.ndarray:
        """
        Retrieves the hubble parameter as a function of redshift.
        """
        ...

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Calculates the comoving distance for given redshifts.
        """
        ...

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Calculates the transverse comoving distance for given redshifts.
        """
        ...

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Calculates the angular diameter distance for given redshifts.
        """
        ...

    def matter_density(self, zs: np.ndarray) -> np.ndarray:
        """
        Computes the matter density as a function of redshift.
        """
        ...

class Perturbations(Protocol):
    background: Background

    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.
        """
        ...

    def growth_rate(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.
        """
        ...

    def matter_power_spectrum(self) -> np.ndarray:
        """
        Retrieves the linear matter power spectrum.
        """
        ...