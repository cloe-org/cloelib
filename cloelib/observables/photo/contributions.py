from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, TypeVar, Union

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

    Likewise optional, and read the same way, is the class attribute
    `nonlimber_spec: NonLimberSpec`: what the beyond-Limber engine needs to
    know about this contribution (see `NonLimberSpec`). A contribution
    without it simply isn't supported beyond Limber.
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


# ---------------------------------------------------------------------------
# Beyond-Limber metadata
# ---------------------------------------------------------------------------
def ell_prefactor_one(ells):
    """No multipole dependence (galaxy clustering)."""
    return jnp.ones(jnp.shape(ells))


def ell_prefactor_magnification(ells):
    """ell (ell + 1): from the two angular derivatives of the lensing potential."""
    return ells * (ells + 1)


def ell_prefactor_spin2(ells):
    """sqrt((ell + 2)! / (ell - 2)!): shear from the lensing potential (spin 2)."""
    return jnp.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))


def ell_prefactor_rsd(ells):
    """-1, kept here (not in the integrand) to avoid a subtraction there."""
    return -jnp.ones(jnp.shape(ells))


@dataclass(frozen=True)
class NonLimberSpec:
    """What the beyond-Limber engine needs to know about one `Contribution`.

    Purely static (no arrays), so it is hashable and safe to read while
    tracing under `jax.jit`.

    Attributes:
      power: the $p$ in the name of the T-tilde matrix a pair needs
        (`T_{p_a + p_b}_..`, see `t_tilde_names`): $+1$ for galaxy clustering
        and RSD, $-1$ for lensing, magnification and IA.
      derivative: order of the Bessel-function derivative the kernel
        multiplies ($0$ for all but RSD, which has $2$).
      ell_prefactor: `ell_prefactor(ells) -> Array`, the kernel's own
        $\\ell$-dependent factor.
      divide_by_chi2: whether the kernel is divided by the squared comoving
        distance it is evaluated at, $\\chi^2$ or $(R\\chi)^2$
        (`nonlimber_kernels.kernel_on_chi_R`).
    """

    power: int
    derivative: int
    ell_prefactor: Callable
    divide_by_chi2: bool


def _power_tag(power: int) -> str:
    return f"minus{-power}" if power < 0 else str(power)


def t_tilde_names(spec_a: NonLimberSpec, spec_b: NonLimberSpec) -> Tuple[str, str]:
    """Names of the two T-tilde matrices a pair of contributions needs.

    The name is `T_{p_a + p_b}_{d_a d_b}`, with a negative power written
    `minusN` (e.g. `T_minus2_00`, `T_2_02`). The first element is for
    the ordering (A at $\\chi$, B at $R\\chi$), the second for the swapped
    one (derivative digits exchanged); they only differ when the two
    derivative orders do (i.e. for pairs involving RSD).

    Returns:
      tuple[str, str]: `(name_12, name_21)`.
    """
    tag = _power_tag(spec_a.power + spec_b.power)
    d_a, d_b = spec_a.derivative, spec_b.derivative
    return f"T_{tag}_{d_a}{d_b}", f"T_{tag}_{d_b}{d_a}"


def nonlimber_spec_of(contribution) -> NonLimberSpec:
    """`contribution.nonlimber_spec`, or a clear error if it has none."""
    spec = getattr(contribution, "nonlimber_spec", None)
    if spec is None:
        raise NotImplementedError(
            f"{type(contribution).__name__} defines no `nonlimber_spec`, so it "
            "is not supported by the beyond-Limber engine."
        )
    return spec


def nonlimber_contributions(tracer):
    """The contributions a tracer enters the beyond-Limber integral with.

    `tracer.get_nonlimber_contributions()` if it defines one (e.g.
    `PositionsTracer`, which adds RSD there but not to the Limber window),
    otherwise the same `get_contributions()` as the Limber computation.
    """
    getter = getattr(tracer, "get_nonlimber_contributions", None)
    return getter() if getter is not None else tracer.get_contributions()
