# General imports
from typing import Protocol, Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp


"""
## Notes:
 
- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


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
    # double check if there is a better option than this
    interface_args: None

    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, ns: float, As: float, w: float, wa: float, gamma_MG: float):
        """
        A protocol to define background cosmology
        """
        ...
    
    def Omega_b(self, zs: T) -> T:
        """
        Computes the matter density as a function of redshift.
        """
        ...

    def Omega_m(self, zs: T) -> T:
        """
        Computes the matter density as a function of redshift.
        """
        ...

    def hubble_parameter(self, zs: T, units: str = '1/Mpc') -> T:
        """
        Retrieves the hubble parameter as a function of redshift.
        """
        ...

    def comoving_distance(self, zs: T) -> T:
        """
        Calculates the comoving distance for given redshifts.
        """
        ...

    def transverse_comoving_distance(self, zs: T) -> T:
        """
        Calculates the transverse comoving distance for given redshifts.
        """
        ...

    def angular_diameter_distance(self, zs: T) -> T:
        """
        Calculates the angular diameter distance for given redshifts.
        """
        ...

class Perturbations(Protocol):
    background: Background

    def growth_factor(self, zs: T, ks: T) -> T:
        """
        Calculates the growth factor for given redshifts and wavenumbers.
        """
        ...

    def growth_rate(self, zs: T, ks: T) -> T:
        """
        Calculates the growth rate for given redshifts and wavenumbers.
        """
        ...

    def matter_power_spectrum(self, zs: T, ks: T) -> T:
        """
        Retrieves the matter power spectrum.
        """
        ...
