"""Protocols for Background and Perturbation cosmology classes.

## Notes:

- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

# General imports
from typing import Protocol, Union, Sequence, TypeVar, Optional, runtime_checkable

import numpy as np  # type: ignore
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

    def Omega_m(self, zs: T) -> T:
        """Compute the matter density as a function of redshift."""
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


@runtime_checkable
class Perturbations(Protocol):
    """Protocol for Perturbation cosmology class."""

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
class BaryonBoostMixin(Protocol):
    """Protocol for mixins that add a baryonic suppression factor.

    Any class that provides ``baryonic_suppression`` satisfies this protocol,
    regardless of inheritance.  Used for static type-checking only — never
    instantiated directly.

    Concrete implementations live in backend-specific files and are composed
    into ``Perturbations`` subclasses to override ``matter_power_spectrum``::

        class FlamingoBaryonBoostMixin(BaryonBoostMixin):
            def baryonic_suppression(self, zs, ks, k_hunit=False): ...

        class CAMBNonLinearFLAMINGOPerturbations(
            FlamingoBaryonBoostMixin, CAMBNonLinearPerturbations
        ):
            def matter_power_spectrum(self, zs, ks, ...):
                return super().matter_power_spectrum(...) * self.baryonic_suppression(...)
    """

    def baryonic_suppression(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Return the multiplicative baryonic suppression factor P_hydro/P_DMO.

        Parameters
        ----------
        zs:
            Redshifts, shape (nz,).
        ks:
            Wavenumbers, shape (nk,).

        Returns
        -------
        np.ndarray, shape (nz, nk)
        """
        ...


def with_baryon_boost(NonLinearClass: type, BaryonMixinClass: type) -> type:
    """Compose a nonlinear perturbations class with a baryonic-boost mixin.

    Returns a new class that:

    * Places ``BaryonMixinClass`` as the **left** parent (MRO priority).
    * Wires ``__init__`` to call ``NonLinearClass.__init__`` first (so that
      emulator state such as ``self.emu`` and ``self.params_*`` is available)
      and then ``BaryonMixinClass.__init__`` with ``baryon_kwargs``.
    * Overrides ``matter_power_spectrum`` and ``matter_power_spectrum_cb`` to
      multiply P(k, z) by ``baryonic_suppression``.

    Usage
    -----
    ::

        from cloelib.cosmology.cosmology import with_baryon_boost
        from cloelib.cosmology.baccoemu_cosmology import BACCOemuNonLinearPerturbations
        from cloelib.cosmology.FlamingoBaryonResponseEmulator_cosmology import (
            FlamingoBaryonBoostMixin,
        )

        BACCOemuFLAMINGO = with_baryon_boost(
            BACCOemuNonLinearPerturbations, FlamingoBaryonBoostMixin
        )
        pert = BACCOemuFLAMINGO(
            background, linear_pert, redshifts,
            nonlinear_model_name="Arico2023",
            baryon_kwargs=dict(fgas_sigma=0.0, Mstar_sigma=0.0, jet_fraction=0.0),
        )

    For HMcode2020::

        BACCOemuHM = with_baryon_boost(
            BACCOemuNonLinearPerturbations, HMcode2020BaryonBoostMixin
        )
        pert = BACCOemuHM(
            background, linear_pert, redshifts,
            baryon_kwargs=dict(log10TAGN=7.8),
        )

    Parameters
    ----------
    NonLinearClass:
        A concrete nonlinear perturbations class (e.g.
        ``BACCOemuNonLinearPerturbations``, ``CAMBNonLinearPerturbations``).
    BaryonMixinClass:
        A concrete :class:`BaryonBoostMixin` subclass (e.g.
        ``FlamingoBaryonBoostMixin``, ``BACCOemuBaryonBoostMixin``,
        ``HMcode2020BaryonBoostMixin``).

    Returns
    -------
    type
        A new class named
        ``"{NonLinearClass.__name__}With{BaryonMixinClass.__name__}"``.
    """

    class Combined(BaryonMixinClass, NonLinearClass):  # type: ignore[misc]
        def __init__(self, *args, baryon_kwargs=None, **kwargs):
            NonLinearClass.__init__(self, *args, **kwargs)
            BaryonMixinClass.__init__(self, **(baryon_kwargs or {}))

        def matter_power_spectrum(self, zs, ks, **kwargs):
            pk = NonLinearClass.matter_power_spectrum(self, zs, ks, **kwargs)
            return (
                pk
                * self.baryonic_suppression(
                    zs, ks, k_hunit=kwargs.get("k_hunit", False)
                )
            ).squeeze()

        def matter_power_spectrum_cb(self, zs, ks, **kwargs):
            pk = NonLinearClass.matter_power_spectrum_cb(self, zs, ks, **kwargs)
            return (
                pk
                * self.baryonic_suppression(
                    zs, ks, k_hunit=kwargs.get("k_hunit", False)
                )
            ).squeeze()

    Combined.__name__ = f"{NonLinearClass.__name__}With{BaryonMixinClass.__name__}"
    Combined.__qualname__ = Combined.__name__
    return Combined
