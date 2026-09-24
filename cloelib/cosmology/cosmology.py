"""Protocols for Background and Perturbation cosmology classes.

## Notes:

- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

# General imports
from typing import Protocol, Union, Sequence, TypeVar, Optional, runtime_checkable

import numpy as np
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class Background(Protocol):
    """Protocol for Background cosmology class."""

    @property
    def H0(self) -> float:
        """Hubble parameter at redshift 0 in km s-1 Mpc-1."""
        ...

    @property
    def h(self) -> float:
        """Dimensionless Hubble constant."""
        ...

    @property
    def Omega_b0(self) -> float:
        """Omega baryon; the baryon density/critical density at z=0."""
        ...

    @property
    def Omega_cdm0(self) -> float:
        """Omega cold dark matter; the cold dark matter density/critical density at z=0."""
        ...

    @property
    def mnu(self) -> Union[float, Sequence[float], T]:
        """Total neutrino mass in eV (float) or an array of individual neutrino masses in eV."""
        ...

    @property
    def N_ur(self) -> float:
        """Effective number of ultra-relativistic species. As defined by CLASS."""
        ...

    @property
    def N_eff(self) -> float:
        """Effective number of relativistic species."""
        ...

    @property
    def N_mnu(self) -> int:
        """Integer number of massive neutrino species."""
        ...

    @property
    def Omega_k0(self) -> float:
        """Omega curvature; the effective curvature density/critical density at z=0."""
        ...

    @property
    def As(self) -> float:
        """Amplitude of the primordial power spectrum."""
        ...

    @property
    def ns(self) -> float:
        """Scalar index of the primordial power spectrum."""
        ...

    @property
    def alpha_s(self) -> float:
        """Running of the scalar spectral index (d ns / d ln k)."""
        ...

    @property
    def w0(self) -> float:
        """Dark energy parameter."""
        ...

    @property
    def wa(self) -> float:
        """Dark energy parameter."""
        ...

    @property
    def gamma_MG(self) -> float:
        """Return the modified gravity Linder parameter."""
        ...

    @property
    def interface_args(self) -> dict:
        """Save internal structure format of possible interface codes."""
        ...

    def Omega_b(self, zs: T) -> T:
        """Compute the matter density as a function of redshift."""
        ...

    def Omega_m(self, zs: Union[T, float]) -> T:
        """Compute the matter density as a function of redshift.

        A bare Python float is also accepted for a single redshift (e.g.
        `Omega_m(0.0)` to get Omega_m0), matching the convention used
        throughout cloelib's observables code; implementations return a
        scalar in that case.
        """
        ...

    def Omega_cb(self, zs: np.ndarray) -> np.ndarray:
        """Computes the cold dark matter + baryons (no neutrinos) as a function of redshift."""
        ...

    def hubble_parameter(self, zs: T, units: str = "km/s/Mpc") -> T:
        """Retrieve the hubble parameter as a function of redshift."""
        ...

    def comoving_distance(self, zs: T) -> T:
        """Calculate the comoving distance for given redshifts."""
        ...

    def transverse_comoving_distance(self, zs: T) -> T:
        """Calculate the transverse comoving distance for given redshifts."""
        ...

    def angular_diameter_distance(self, zs: T) -> T:
        """Calculate the angular diameter distance for given redshifts."""
        ...

    @property
    def rdrag(self) -> float:
        """Sound horizon radius at last scattering in Mpc."""
        ...

    @property
    def z_star(self) -> float:
        """Redshift of photon decoupling."""
        ...


@runtime_checkable
class Perturbations(Protocol):
    """Protocol for Perturbation cosmology class.

    Note: some consumers (e.g. `ShearTracer.get_window_IA`,
    `AngularTwoPoint.get_Cl`) informally read `.k`/`.z` attributes off a
    `Perturbations` instance for the wavenumber/redshift grid it was built
    on. These are deliberately *not* part of this Protocol: not every
    backend sets them (e.g. the JAX backends ignore the `ks` argument to
    `growth_factor` and never set `self.k`), and several tests assert
    `isinstance(instance, Perturbations)` for those backends. Code reading
    `.k`/`.z` off an arbitrary `Perturbations` must use
    `getattr(perturbations, "k", None)` / handle their absence, not assume
    they exist.
    """

    @property
    def background(self) -> Background:
        """Store background obj."""
        ...

    def growth_factor(self, zs: T, ks: T) -> T:
        """Calculate the growth factor for given redshifts and wavenumbers."""
        ...

    def growth_rate(self, zs: Optional[T] = None, ks: Optional[T] = None) -> T:
        """Calculate the growth rate for given redshifts and wavenumbers."""
        ...

    def matter_power_spectrum(self, zs: T, ks: T) -> T:
        """Retrieve the matter power spectrum."""
        ...

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        """Retrieves matter power spectrum of cold dark matter + baryons (no neutrinos)."""
        ...

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""
        ...


@runtime_checkable
class WithWavenumberGrid(Protocol):
    """Structural protocol for a `linearperturbations`-like object exposing a `k` grid.

    `.k` is deliberately not part of the `Perturbations` protocol (see its
    docstring). Boost-style implementations that need the wavenumber grid
    their `linearperturbations` argument was built on (e.g.
    `EE2NonLinearPerturbations`, `MGemuNonlinearBoost`,
    `TabulatedNonlinearBoost`) type that argument against this narrower
    protocol instead of the full `Perturbations`, since not every
    `Perturbations` implementation sets `.k`.
    """

    @property
    def k(self) -> np.ndarray:
        """Wavenumber grid in 1/Mpc."""
        ...


@runtime_checkable
class WithLinearSpectrumGrid(WithWavenumberGrid, Protocol):
    """Structural protocol for a `linearperturbations` exposing cached P(k, z) grids.

    Used by implementations (e.g. `HMemuNonLinearPerturbations`) that stitch
    a cached low-k linear tail onto a high-k nonlinear/emulated spectrum,
    and so need direct array access to the grid `linearperturbations` was
    built on rather than going through `matter_power_spectrum`/
    `matter_power_spectrum_cb`. `.z`/`.Pk`/`.Pk_cb` are, like `.k`,
    deliberately not part of the `Perturbations` protocol.
    """

    @property
    def z(self) -> np.ndarray:
        """Redshift grid the linear spectrum was computed on."""
        ...

    @property
    def Pk(self) -> np.ndarray:
        """Cached linear total-matter power spectrum on the (z, k) grid."""
        ...

    @property
    def Pk_cb(self) -> np.ndarray:
        """Cached linear cdm+baryon power spectrum on the (z, k) grid."""
        ...
