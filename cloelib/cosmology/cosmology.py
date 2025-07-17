# General imports
from typing import Protocol, Union, TypeVar, runtime_checkable

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

@runtime_checkable
class Background(Protocol):

    @property
    def H0(self) -> float:
        """
        Hubble parameter at redshift 0 in km s-1 Mpc-1.
        """
        ...
    
    @property
    def h(self) -> float:
        """
        Dimensionless Hubble constant
        """
        ...
    
    @property
    def Omega_b0(self) -> float:
        """
        Omega baryon; the baryon density/critical density at z=0.
        """
        ...

    @property
    def Omega_cdm0(self) -> float:
        """
        Omega cold dark matter; the cold dark matter density/critical density at z=0.
        """
        ...

    @property
    def mnu(self) -> float:
        """
        Total neutrino mass in eV.
        """
        ...

    @property
    def Omega_k0(self) -> float:
        """
        Omega curvature; the effective curvature density/critical density at z=0.
        """
        ...

    @property
    def As(self) -> float:
        """
        Amplitude of the primordial power spectrum
        """
        ...

    @property
    def ns(self) -> float:
        """
        scalar index of the primordial power spectrum
        """
        ...

    @property
    def w0(self) -> float:
        """
        dark energy parameter
        """
        ...

    @property
    def wa(self) -> float:
        """
        dark energy parameter
        """
        ...

    @property
    def gamma_MG(self) -> float:
        """
        Modified gravity Linder parameter
        """
        ...

    @property
    def _interface_args(self) -> dict:
        """
        Save internal structure format of possible interface codes
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

    def hubble_parameter(self, zs: T, units: str = "km/s/Mpc") -> T:
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

    @property
    def rdrag(self) -> float:
        """
        Sound horizon radius at last scattering
        """
        ...


@runtime_checkable
class Perturbations(Protocol):

    @property
    def background(self) -> Background:
        """
        Stores background obj
        """
        ...

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
