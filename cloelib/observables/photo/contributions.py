from typing import Any, Optional, Protocol, TypeVar, Union

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
# Fields only a halo model provides (e.g. `GODMAXNonLinearPerturbations`): HOD
# galaxies and Compton-y. A pair involving one of them is integrated against
# `perturbations.power_spectrum(field1, field2, zs, ks)`.
GALAXY = "g"
COMPTON_Y = "y"
HALO_MODEL_FIELDS = (GALAXY, COMPTON_Y)


def contribution_field(contribution) -> str:
    """`contribution.field` if defined, else `MATTER`.

    Only contributions that can trace the cold dark matter + baryons field
    (e.g. `GalaxyBiasContribution` of a `PositionsTracer(use_Pcb=True)`) or
    a halo-model field (`GALAXY` for `HODGalaxyContribution`, `COMPTON_Y`
    for `ComptonYContribution`) define `field`; everything else (lensing,
    magnification, RSD, intrinsic alignments, ...) traces the total matter
    and needs no change.
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


def check_halo_model_field(perturbations: Any, field: str, needed_for: str) -> None:
    """Raise if `perturbations` cannot provide halo-model spectra of `field`.

    Args:
      perturbations: The perturbations object of a tracer.
      field: `GALAXY` (HOD galaxies) or `COMPTON_Y` (Compton-y).
      needed_for: What requires the field, used in the error message.
    """
    spectra = getattr(perturbations, "spectra", None)
    if spectra is None or (field, field) not in spectra:
        option = {GALAXY: "galaxies=...", COMPTON_Y: "model_tSZ=True"}[field]
        raise ValueError(
            f"{needed_for} needs halo-model power spectra: build the perturbations "
            "with cloelib.cosmology.godmax_cosmology.GODMAXNonLinearPerturbations"
            f"(..., {option})."
        )


def check_supported_model(perturbations: Any, kind: str, model: Optional[str]) -> None:
    """Raise if `perturbations` declares that it does not support `model`.

    A backend may declare `supported_ia_models` / `supported_galaxy_bias_models`
    (e.g. `GODMAXNonLinearPerturbations`); backends that declare nothing
    support every model.

    Args:
      perturbations: The perturbations object of a tracer.
      kind: `"ia"` or `"galaxy_bias"`.
      model: The requested model name (`None` for no IA).
    """
    supported = getattr(perturbations, f"supported_{kind}_models", None)
    if supported is not None and model not in supported:
        raise NotImplementedError(
            f"{type(perturbations).__name__} does not support {kind}_model="
            f"{model!r}; supported: {list(supported)}."
        )
