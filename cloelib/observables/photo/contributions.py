from typing import Protocol, TypeVar, Union

import numpy as np  # type: ignore
import jax.numpy as jnp  # type: ignore

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


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
    """

    def compute_kernel(self, z: T) -> T:
        """Return this contribution's kernel, shape (n_bins, len(z))."""
        ...


class IntrinsicAlignmentContribution:
    """Base class for every intrinsic-alignment model - the general "this is
    an IA contribution" type. `NLAContribution` and `TATTContribution`
    (`photo/shear.py`) are its two concrete models today; a future IA model
    would be a third subclass here, not a fourth unrelated name.

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
