import weakref
from typing import Protocol, TypeVar, Union

import numpy as np
import jax.numpy as jnp


def unit_sigma(z):
    """Default MG lensing parameter Sigma(z) = 1 for GR / LCDM perturbations.

    Every lensing kernel (shear, magnification bias, CMB lensing) multiplies by
    ``perturbations.Sigma(z)`` when the perturbations provide it, and by this
    otherwise.
    """
    return jnp.ones_like(z)


T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

# Density fields a Contribution can trace. The base power spectrum of a pair of
# contributions follows from their two fields: (MATTER, MATTER) -> P_mm,
# (CB, CB) -> P_cb, mixed -> the cb x matter cross spectrum.
MATTER = "m"
CB = "cb"


def contribution_field(contribution) -> str:
    """`contribution.field` if defined, else `MATTER`.

    Only contributions that can trace the cold dark matter + baryons field
    (e.g. `GalaxyBiasContribution` of a `PositionsTracer(use_Pcb=True)`)
    define `field`; everything else (lensing, magnification, intrinsic
    alignments, ...) traces the total matter and needs no change.
    """
    return getattr(contribution, "field", MATTER)


class Contribution(Protocol):
    """A single additive term of a tracer's window function.

    `spectrum_engine.py` and `photo.shear`'s `TATTContribution` extend this
    with three *optional* methods, deliberately left out of this Protocol
    so every simpler `Contribution` is unaffected:

    - `get_spectrum_requests() -> Sequence[SpectrumRequest]`: derived power
      spectra this contribution needs beyond the plain matter Pk (default,
      for anything not defining it: none).
    - `get_requirements_for_interaction(other) -> Sequence[SpectrumRequest]`:
      context-aware version of the above - e.g. TATT needs fewer terms when
      paired with a non-IA contribution than with another IA one (default:
      `get_spectrum_requests()`, unconditionally).
    - `get_effective_pk(other, bank) -> Optional[Array]`: the effective
      P(k,z) this contribution's pairing with `other` should be integrated
      against, shape `(len(bank.zs), len(bank.ks))` (default: `None`,
      meaning "use the plain matter Pk", today's behavior for everything).

    `spectrum_engine.py` looks these up via `getattr(obj, name, None)`
    rather than requiring them here, so nothing below needs to change.

    An optional `field` attribute (`MATTER` or `CB`, read through
    `contribution_field`) states which density field the contribution
    traces; it selects the base power spectrum of each contribution pair
    in `AngularTwoPoint` (default: `MATTER`).
    """

    def compute_kernel(self, z: T) -> T:
        """Return this contribution's kernel, shape (n_bins, len(z))."""
        ...


class IntrinsicAlignmentContribution:
    """Base class for every intrinsic-alignment model - the general "this is
    an IA contribution" type. `NLAContribution`, `TATTContribution` and
    `TATTMContribution` (`photo/shear.py`) are its concrete models today; a
    future IA model would be another subclass here, not an unrelated name.

    Lets a contribution ask "is the other side of this pairing also IA?" -
    e.g. TATT needs its II-only one-loop terms (Eq. 14 of Navarro-Gironés
    et al. 2026) only when paired with another IA contribution, not with a
    density/lensing one - via `isinstance(other,
    IntrinsicAlignmentContribution)`. Mirrors `toy_cloelib.contributions.
    AbstractIAContribution`, which the same TATT pruning logic there relies
    on. Carries no behavior of its own: a plain base class, not a
    `typing.Protocol` - an empty `@runtime_checkable` Protocol matches
    *any* object under `isinstance` (nothing to structurally check for),
    which would silently break the II-vs-GI pruning this class exists for.
    `NLAContribution` and `LensingContribution` both satisfy the same
    `Contribution` Protocol (`compute_kernel(z)`), so "is this IA" is a
    nominal question with no structural feature to key a Protocol off.

    The concrete photometric-probe Contributions (`LensingContribution`,
    `NLAContribution`, `TATTContribution`, ... for shear;
    `GalaxyBiasContribution`, `MagnificationContribution` for positions)
    live alongside their owning tracer in `cloelib.observables.photo.shear`
    / `.positions` - this module holds only the generic, tracer-agnostic
    base types `spectrum_engine.py` and both submodules share.
    """


class TracerBackRefMixin:
    """Mixin for a Contribution that needs its owning tracer: keeps
    `self._tracer` as a *weak* reference.

    A tracer owns its contributions (e.g. `ShearTracer.lensing`), and each
    contribution calls back into the tracer for its kernel
    (`self._tracer.get_window_lensing(z)`). A plain attribute would make the
    pair a reference cycle, which reference counting cannot free: every
    discarded tracer - and, through it, its perturbations, background and
    their P(k) grids / CAMB results - would stay in memory until Python's
    cyclic garbage collector happens to run (cloelib#714, cloerun#18).

    Use it by listing it among the bases; `self._tracer = tracer` and every
    `self._tracer....` access then work unchanged. The tracer must outlive
    its contributions, as it does whenever they are reached through it
    (`tracer.get_contributions()`, `tracer.lensing`, ...).
    """

    @property
    def _tracer(self):
        tracer = self._tracer_ref()
        if tracer is None:
            raise ReferenceError(
                f"the tracer owning this {type(self).__name__} no longer exists: "
                "keep a reference to the tracer while using its contributions."
            )
        return tracer

    @_tracer.setter
    def _tracer(self, tracer):
        self._tracer_ref = weakref.ref(tracer)
