"""Protocols for Background and Perturbation cosmology classes.

## Notes:

- Refactored cosmology.py from the original CLOE to provide a more flexible framework,
  enabling seamless integration with external cosmological codes while removing dependency on Cobaya.

- Introduced the use of protocols to standardize external code interfaces,
  providing a unified and extensible template for interaction.
"""

# General imports
import inspect
from typing import Any, Protocol, Union, Sequence, TypeVar, Optional, runtime_checkable

import numpy as np
import numpy.typing as npt
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

# Prescriptions for the cb x matter cross power spectrum, declared by every
# backend providing `matter_power_spectrum_cb` as the class attribute
# `cb_cross_prescription` and read by `SpectraBank` (`photo.spectrum_engine`).
# Each is exact under a condition on the backend's own non-linear spectra, and
# a backend declares the one it satisfies:
#
# - CB_CROSS_CORRELATED: `P_cb,m = sqrt(P_cb P_mm)`, exact if the backend
#   applies the same non-linear boost to cb and total matter,
#   `P_cb^NL / P_cb^L = P_mm^NL / P_mm^L` (always true for linear spectra).
# - CB_CROSS_LINEAR_NU: `P_cb,m = f_cb (P_cb^NL - P_cb^L) + sqrt(P_cb^L P_mm^L)`,
#   exact if neutrinos and the cb x nu spectrum stay linear,
#   `P_mm^NL - P_mm^L = f_cb^2 (P_cb^NL - P_cb^L)`. Needs the backend's
#   `linearperturbations`.
#
# Backends satisfying neither (e.g. CLASS, which runs its non-linear model on
# P_cb separately) use the closest one; see `docs/code_structure/perturbations.md`.
CB_CROSS_CORRELATED = "correlated"
CB_CROSS_LINEAR_NU = "linear_neutrinos"
CB_CROSS_PRESCRIPTIONS = (CB_CROSS_CORRELATED, CB_CROSS_LINEAR_NU)


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
    def mnu(self) -> Union[float, Sequence[float], np.ndarray, jnp.ndarray]:
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
        """Evaluate the dimensionless growth factor at redshifts and wavenumbers.

        Args:
            zs (T): One-dimensional NumPy or JAX array of redshifts, with
                shape ``(nz,)``.
            ks (T): One-dimensional NumPy or JAX array of wavenumbers in
                ``1/Mpc``, with shape ``(nk,)``.

        Returns:
            T: NumPy or JAX array of growth-factor values. Scale-dependent
            backends generally return shape ``(nz, nk)``; scale-independent
            backends may return shape ``(nz,)``. The growth factor is
            conventionally normalized to unity at redshift zero. Consult the
            concrete backend when relying on a particular shape or
            normalization convention.
        """
        ...

    def growth_rate(self, zs: Optional[T] = None, ks: Optional[T] = None) -> T:
        """Calculate the growth rate for given redshifts and wavenumbers.

        Args:
            zs (Optional[T]): One-dimensional NumPy or JAX array of redshifts,
                with shape ``(nz,)``. If ``None``, implementations use the
                redshift grid they were built on; implementations without a
                default grid may raise ``ValueError``.
            ks (Optional[T]): One-dimensional NumPy or JAX array of wavenumbers
                in ``1/Mpc``, with shape ``(nk,)``. If ``None``, return the
                growth rate as a function of redshift only.

        Returns:
            T: NumPy or JAX array of growth-rate values. With ``ks=None``, the
            shape is ``(nz,)``; with wavenumbers supplied, the shape is
            ``(nz, nk)``. Scale-independent growth rates are broadcast along
            the wavenumber axis.
        """
        ...

    def matter_power_spectrum(self, zs: T, ks: T) -> T:
        """Retrieve the matter power spectrum."""
        ...

    def matter_power_spectrum_cb(self, zs: T, ks: T) -> T:
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


@runtime_checkable
class BaryonBoostMixin(Protocol):
    """Protocol for mixins that add a baryonic suppression factor.

    Any class that provides ``baryonic_suppression`` satisfies this protocol,
    regardless of inheritance.  Used for static type-checking only — never
    instantiated directly.

    Concrete implementations live in backend-specific files and are composed
    with a nonlinear perturbations class through :func:`with_baryon_boost`,
    which multiplies ``matter_power_spectrum`` and ``matter_power_spectrum_cb``
    by ``baryonic_suppression``::

        CAMBNonLinearFLAMINGOPerturbations = with_baryon_boost(
            CAMBNonLinearPerturbations, FlamingoBaryonBoostMixin
        )

    Composing by hand instead requires overriding both spectra, and making
    sure the cb one is boosted once also when the nonlinear class derives it
    from the total one; `with_baryon_boost` takes care of this.
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


def hmcode_version_includes_baryons(hmcode_version: Optional[str]) -> bool:
    """Whether a CLASS ``hmcode_version`` selects HMcode2020 with baryonic feedback.

    CLASS selects the baryonic version for any value containing ``20baryon``
    or ``20_baryon`` (e.g. ``"2020_baryonic_feedback"`` or ``"2020_baryonic"``).
    """
    return hmcode_version is not None and (
        "20baryon" in hmcode_version or "20_baryon" in hmcode_version
    )


def class_hmcode_includes_baryons(
    nonlinear_model: Optional[str], hmcode_version: Optional[str]
) -> bool:
    """Whether CLASS options select HMcode2020 with baryonic feedback.

    CLASS selects HMcode for any ``non_linear`` value containing ``hmcode``,
    whatever its capitalisation, and the baryonic version as described in
    `hmcode_version_includes_baryons`. Shared by the CLASS-based backends.
    """
    return (
        nonlinear_model is not None
        and "hmcode" in nonlinear_model.lower()
        and hmcode_version_includes_baryons(hmcode_version)
    )


def perturbations_include_baryons(perturbations: Any) -> bool:
    """Whether the nonlinear spectrum of ``perturbations`` includes baryonic feedback.

    True if it reports ``includes_baryons`` (see `ensure_gravity_only`) or
    carries a baryonic mixin, however composed. Used by the classes that wrap
    another nonlinear spectrum, e.g. to apply a modified-gravity boost.
    """
    return bool(getattr(perturbations, "includes_baryons", False)) or isinstance(
        perturbations, BaryonBoostMixin
    )


def ensure_gravity_only(
    perturbations: Any, mixin_name: str, perturbations_name: Optional[str] = None
) -> None:
    """Refuse a baryonic mixin on a nonlinear spectrum that includes baryons.

    Nonlinear perturbations whose spectrum already includes baryonic feedback
    set ``includes_baryons = True``: CAMB with ``log10TAGN`` or the
    ``mead2020_feedback`` model, CLASS and mochi_class with HMcode2020 and
    its baryonic feedback (e.g. through ``log10TAGN``), HMcode2020emu with
    ``log10TAGN``, BACCOemu with ``baryonic_boost``, the HMcode2020 sector of
    CosmoPower-JAX, the classes wrapping one of those (the modified-gravity
    boosts), and classes composed by `with_baryon_boost`. Every baryonic
    mixin calls this at the start of its ``__init__``, so the check holds
    however the mixin is composed.

    Args:
        perturbations: The nonlinear perturbations to check.
        mixin_name: Name of the baryonic mixin to be added, for the message.
        perturbations_name: Name of the nonlinear perturbations class, for the
            message. Defaults to the class of ``perturbations``.

    Raises:
        ValueError: If ``perturbations.includes_baryons`` is true.
    """
    if getattr(perturbations, "includes_baryons", False):
        name = perturbations_name or type(perturbations).__name__
        raise ValueError(
            f"{name} already includes baryonic feedback "
            f"in its nonlinear spectrum. Adding {mixin_name} on top would apply "
            "baryonic feedback twice. Use a gravity-only nonlinear prescription."
        )


def _native_baryon_options(kwargs: dict) -> list[str]:
    """Return the nonlinear constructor options that already switch on baryons.

    Several nonlinear backends can include a baryonic boost themselves
    (BACCOemu via `baryonic_boost`, HMcode2020emu, CAMB and CLASS via
    `log10TAGN` or their HMcode2020 feedback model). Composing a baryonic
    mixin on top of one of them would apply baryonic feedback twice.
    """
    options = []
    if kwargs.get("baryonic_boost"):
        options.append(f"baryonic_boost={kwargs['baryonic_boost']!r}")
    if kwargs.get("log10TAGN") is not None:
        options.append(f"log10TAGN={kwargs['log10TAGN']!r}")
    if kwargs.get("nonlinear_model") == "mead2020_feedback":
        options.append("nonlinear_model='mead2020_feedback'")
    if hmcode_version_includes_baryons(kwargs.get("hmcode_version")):
        options.append(f"hmcode_version={kwargs['hmcode_version']!r}")
    return options


def _k_hunit_argument(method: Any, *args: Any, **kwargs: Any) -> bool:
    """Return the ``k_hunit`` argument of a call to ``method``, however passed.

    Backends pass it by keyword or by position, or accept it through
    ``**kwargs``; it defaults to ``False`` (wavenumbers in 1/Mpc).
    """
    signature = inspect.signature(method)
    arguments = signature.bind(*args, **kwargs).arguments
    if "k_hunit" in arguments:
        return bool(arguments["k_hunit"])
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return bool(arguments.get(parameter.name, {}).get("k_hunit", False))
    return False


def with_baryon_boost(NonLinearClass: Any, BaryonMixinClass: Any) -> type:
    """Compose a nonlinear perturbations class with a baryonic-boost mixin.

    Returns a new class that:

    * Places ``BaryonMixinClass`` as the **left** parent (MRO priority).
    * Wires ``__init__`` to call ``NonLinearClass.__init__`` first (so that
      ``self.background`` and ``self.z`` are available) and then
      ``BaryonMixinClass.__init__`` with ``baryon_kwargs``.
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

    Raises
    ------
    TypeError
        If ``NonLinearClass`` already carries a baryonic mixin, since stacking
        two baryonic models would apply baryonic feedback twice.
    ValueError
        At construction, if the nonlinear options passed already switch on
        the backend's own baryonic boost (e.g. ``log10TAGN`` or
        ``baryonic_boost``), or if the constructed nonlinear perturbations
        report ``includes_baryons = True`` (e.g. the CosmoPower-JAX HMcode2020
        sector, which always includes baryonic feedback), for the same reason.

    Notes
    -----
    While the nonlinear class computes ``matter_power_spectrum_cb``, any call
    it makes to ``self.matter_power_spectrum`` returns the gravity-only
    spectrum, so that backends deriving the cb spectrum from the total one are
    boosted only once. The composed instances report ``includes_baryons =
    True``, so that no further baryonic mixin is added on top of them, e.g.
    through a class wrapping them.
    """
    if issubclass(NonLinearClass, BaryonBoostMixin):
        raise TypeError(
            f"{NonLinearClass.__name__} already includes a baryonic boost; "
            f"composing {BaryonMixinClass.__name__} on top would apply baryonic "
            "feedback twice. Start from a nonlinear class without baryons."
        )

    class Combined(BaryonMixinClass, NonLinearClass):
        # True while the nonlinear class computes the cb spectrum, see below.
        _gravity_only = False

        def __init__(self, *args, baryon_kwargs=None, **kwargs):
            native = _native_baryon_options(kwargs)
            if native:
                raise ValueError(
                    f"{NonLinearClass.__name__} was given {', '.join(native)}, so "
                    "its matter_power_spectrum already includes a baryonic boost. "
                    f"Composing {BaryonMixinClass.__name__} on top would apply "
                    "baryonic feedback twice. Drop those options and use the mixin, "
                    "or drop the mixin and use the base class on its own."
                )
            NonLinearClass.__init__(self, *args, **kwargs)
            # The mixins check this themselves; repeated here for mixins that
            # do not.
            ensure_gravity_only(
                self, BaryonMixinClass.__name__, NonLinearClass.__name__
            )
            BaryonMixinClass.__init__(self, **(baryon_kwargs or {}))
            self.includes_baryons = True

        def matter_power_spectrum(self, zs, ks, *args, **kwargs):
            pk = NonLinearClass.matter_power_spectrum(self, zs, ks, *args, **kwargs)
            if self._gravity_only:
                return pk
            k_hunit = _k_hunit_argument(
                NonLinearClass.matter_power_spectrum, self, zs, ks, *args, **kwargs
            )
            return pk * self.baryonic_suppression(zs, ks, k_hunit=k_hunit)

        def matter_power_spectrum_cb(self, zs, ks, *args, **kwargs):
            # The cdm+baryon spectrum is given the *same* boost as total matter.
            # The backends differ slightly in what they predict -- FLAMINGO's
            # response is (b + cdm + nu) / (cdm + nu), BACCOemu's is the cold
            # (b + cdm) / (cdm + nu) -- and one could refine this by adding the
            # neutrino contribution at linear order, as BACCOemu does for its
            # nonlinear boost. In practice the difference is ~1e-5 even for
            # mnu = 0.4 eV, and FLAMINGO simulations show total and cold boosts
            # to be equivalent at that level, so the same boost is used for both.
            # Some backends build the cb spectrum from `self.matter_power_spectrum`
            # (e.g. CLASS without massive neutrinos, or JAX), which must then be
            # gravity-only for the boost to be applied once.
            previous, self._gravity_only = self._gravity_only, True
            try:
                pk = NonLinearClass.matter_power_spectrum_cb(
                    self, zs, ks, *args, **kwargs
                )
            finally:
                self._gravity_only = previous
            k_hunit = _k_hunit_argument(
                NonLinearClass.matter_power_spectrum_cb, self, zs, ks, *args, **kwargs
            )
            return pk * self.baryonic_suppression(zs, ks, k_hunit=k_hunit)

    Combined.__name__ = f"{NonLinearClass.__name__}With{BaryonMixinClass.__name__}"
    Combined.__qualname__ = Combined.__name__
    return Combined


B_contra = TypeVar("B_contra", bound=Background, contravariant=True)
L_contra = TypeVar("L_contra", contravariant=True)


class LinearPerturbationsFactory(Protocol[B_contra]):
    """Constructor interface shared by every linear `Perturbations` class.

    A linear perturbations class is built as
    `cls(background=..., redshifts=...)`. `B_contra` is the background type
    the implementation needs (e.g. `mochiCLASSBackground`), so a class that
    requires a specific background is only accepted where that background
    type is promised.
    """

    def __call__(self, background: B_contra, redshifts: np.ndarray) -> Perturbations:
        """Build the linear perturbations."""
        ...


class NonLinearPerturbationsFactory(Protocol[B_contra, L_contra]):
    """Constructor interface shared by every nonlinear `Perturbations` class.

    A nonlinear perturbations class is built as
    `cls(background=..., linearperturbations=..., redshifts=...)`, followed
    by optional, backend-specific keyword arguments (e.g. `nonlinear_model`,
    or `log10TAGN` for the HMcode-based backends) that are deliberately not
    part of this interface.

    `L_contra` is the type of `linearperturbations` the caller provides, so
    implementations that need more than the `Perturbations` interface from
    it (e.g. `HMemuNonLinearPerturbations` reads the cached `Pk`/`Pk_cb`
    grid, see `WithLinearSpectrumGrid`) are only accepted for linear
    perturbations that provide it.
    """

    def __call__(
        self,
        background: B_contra,
        linearperturbations: L_contra,
        redshifts: np.ndarray,
    ) -> Perturbations:
        """Build the nonlinear perturbations."""
        ...


def growth_rate_on_redshifts(
    z_grid: npt.ArrayLike,
    f_grid: npt.ArrayLike,
    zs: Optional[npt.ArrayLike] = None,
    ks: Optional[npt.ArrayLike] = None,
) -> np.ndarray:
    """Evaluate a tabulated, scale-independent growth rate following `Perturbations.growth_rate`.

    Used by implementations that compute the growth rate on their own
    redshift grid. Requested redshifts are linearly interpolated on that
    grid and clamped to its edge values outside of it.

    Args:
        z_grid (array_like): Redshifts at which `f_grid` is tabulated.
        f_grid (array_like): Growth rate at `z_grid`.
        zs (Optional[array_like]): Redshifts at which to evaluate the growth
            rate. Defaults to `z_grid`, returning `f_grid` unchanged.
        ks (Optional[array_like]): Wavenumbers used to broadcast the growth rate.

    Returns:
        np.ndarray: The growth rate, with shape (nz,) if `ks` is None and
        (nz, nk) otherwise.
    """
    f = np.asarray(f_grid)
    if zs is not None:
        z = np.asarray(z_grid)
        order = np.argsort(z)
        f = np.interp(np.atleast_1d(np.asarray(zs, dtype=float)), z[order], f[order])
    if ks is None:
        return f
    return np.tile(f[:, None], (1, np.size(ks)))
