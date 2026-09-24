"""Generalized per-contribution-pair spectrum engine.

`contributions.py` gives each tracer's window terms their own `Contribution`
object. Most of them only ever need the plain matter power spectrum P(k,z),
so `AngularTwoPoint.get_Cl` can pair one shared Pk grid against the
tracer-level window sum. TATT breaks that assumption: its `II` (and `GI`)
power spectra are sums of several *distinct* k-dependent terms (Eqs. 13-15
of Navarro-Gironés et al. 2026, arXiv:2602.16448, following Blazek et al.
2019), each needing its own one-loop kernel, not just P(k,z). This module
is the minimal machinery that lets a `Contribution` declare "I need these
extra derived spectra" and get them computed once (and only if actually
needed for the pairing at hand) before the Limber integral runs - the same
principle `toy_cloelib` demonstrates (see its `engine.py` /
`ARCHITECTURE_REVIEW.md` Sect. 1.2), adapted here without adding new
dependencies (no `equinox`/`plum`): plain dataclasses and a
`getattr(..., default)` lookup, matching how `AngularTwoPoint` already
dispatches on tracer type (`tracer_rules`, `angular_two_point.py`).

Nothing here changes behavior for any `Contribution` that doesn't declare
extra requests: `needs_generalized_engine` returns `False` for them, and
`AngularTwoPoint.get_Cl` keeps using its original, untouched fast path in
that case.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class SpectrumRequest:
    """One derived power-spectrum grid a Contribution needs beyond P(k,z).

    Attributes:
      name: unique key this grid is stored/looked up under in a
        `SpectraBank`. Two contributions declaring the same `name` share
        one computation (memoized by `SpectraBank`).
      compute: `compute(matter_pk, ks, zs) -> Array`, shape `(len(zs),
        len(ks))` or `(len(ks),)` for a pure k-kernel - whatever shape the
        requesting Contribution's `get_effective_pk` expects back. This is
        the "SpectrumComputer": the math that turns the base matter power
        spectrum into the derived one. Swapping the physics (e.g. a
        PBJ-backed computer instead of a placeholder) means passing a
        different `compute` callable, not touching the Contribution.
    """

    name: str
    compute: Callable[..., jnp.ndarray]


class SpectraBank:
    """Memoized store of derived power-spectrum grids for one `get_Cl` call.

    Not shared across calls - built fresh each time `get_Cl` needs the
    generalized engine, exactly like the plain matter Pk grid it already
    rebuilds every call today.
    """

    def __init__(self, matter_pk: jnp.ndarray, ks: jnp.ndarray, zs: jnp.ndarray):
        self.matter_pk = matter_pk
        self.ks = ks
        self.zs = zs
        self._grids: Dict[str, jnp.ndarray] = {}

    def get(self, request: SpectrumRequest) -> jnp.ndarray:
        """Return the named grid, computing and caching it on first use."""
        if request.name not in self._grids:
            self._grids[request.name] = request.compute(
                self.matter_pk, self.ks, self.zs
            )
        return self._grids[request.name]


def _spectrum_requests(contribution) -> Sequence[SpectrumRequest]:
    """`contribution.get_spectrum_requests()` if defined, else `()`.

    The plain (non-TATT) `Contribution`s (`LensingContribution`,
    `NLAContribution`, `GalaxyBiasContribution`,
    `MagnificationContribution`) have no such method - by design, so this
    always returns `()` for them without requiring any change to those
    classes.
    """
    getter = getattr(contribution, "get_spectrum_requests", None)
    return getter() if getter is not None else ()


def _requirements_for_interaction(contribution, other) -> Sequence[SpectrumRequest]:
    """`contribution.get_requirements_for_interaction(other)` if defined.

    Lets a Contribution declare *fewer* requests for one interaction than
    another (e.g. TATT dropping its II-only kernels when paired with a
    non-IA contribution) - mirrors `toy_cloelib`'s
    `Contribution.get_requirements_for_interaction`. Falls back to the
    unconditional `get_spectrum_requests()` when a Contribution doesn't
    override this (correct default: "the same requests regardless of
    pairing").
    """
    getter = getattr(contribution, "get_requirements_for_interaction", None)
    if getter is not None:
        return getter(other)
    return _spectrum_requests(contribution)


def needs_generalized_engine(contributions1, contributions2) -> bool:
    """Whether any contribution-pair in this tracer pair needs extra spectra.

    Pure Python over Contribution *types*/declared requests, evaluated
    before any Limber integration - safe to call from plain (non-jitted)
    `get_Cl` code, and cheap enough to call on every `get_Cl` invocation.
    """
    for c1 in contributions1:
        for c2 in contributions2:
            if _requirements_for_interaction(c1, c2) or _requirements_for_interaction(
                c2, c1
            ):
                return True
    return False


def build_spectra_bank(
    contributions1, contributions2, matter_pk, ks, zs
) -> SpectraBank:
    """Populate a `SpectraBank` with every request either tracer's
    contributions declare for their pairing with the other tracer's
    contributions - the minimal set the generalized engine needs, mirroring
    `toy_cloelib.engine.setup_engine`'s plan-building (without a separate
    static-plan object, since here the bank is rebuilt fresh per call
    anyway).
    """
    bank = SpectraBank(matter_pk, ks, zs)
    for c1 in contributions1:
        for c2 in contributions2:
            for req in _requirements_for_interaction(c1, c2):
                bank.get(req)
            for req in _requirements_for_interaction(c2, c1):
                bank.get(req)
    return bank


def get_effective_pk(c1, c2, bank: SpectraBank):
    """The effective P(k,z) for the pairing `(c1, c2)`, or `None`.

    Tries `c1.get_effective_pk(c2, bank)` first, then `c2.get_effective_pk(
    c1, bank)` - the P(k,z) two contributions should be integrated against
    is a property of the *pairing*, not of either side alone (e.g. a plain
    `LensingContribution` never defines this, but a `TATTContribution` it's
    paired with does, regardless of argument order: P_deltaI(k,z) is the
    same scalar grid whichever contribution is asked). `None` means neither
    side has anything special to say - use the plain matter Pk grid, i.e.
    every non-TATT pairing.
    """
    for a, b in ((c1, c2), (c2, c1)):
        getter = getattr(a, "get_effective_pk", None)
        if getter is not None:
            pk = getter(b, bank)
            if pk is not None:
                return pk
    return None


def compute_effective_pk(c1, c2, matter_pk, ks, zs):
    """One-call convenience wrapper: `get_effective_pk` for a pairing that
    doesn't already have a `SpectraBank` lying around.

    `AngularTwoPoint._compute_cl_generalized` builds a `SpectraBank` and
    calls `get_effective_pk` per contribution pair as part of computing a
    full `Cl`, but that bank is a local variable, not something callers can
    get at directly - anyone who wants just the effective P(k,z) itself
    (e.g. to plot or validate it, without integrating a full `Cl`) would
    otherwise have to reconstruct a `SpectraBank` by hand. This does that
    reconstruction for you, in one call:

    ```python
    P_II = compute_effective_pk(tracer.ia, tracer.ia, matter_pk, ks, zs)
    P_deltaI = compute_effective_pk(tracer.ia, tracer.lensing, matter_pk, ks, zs)
    ```

    `matter_pk` must already be evaluated on `(zs, ks)` (e.g.
    `perturbations.matter_power_spectrum(zs, ks)`) - this doesn't compute
    it for you, since which `perturbations` object to use is caller
    context this function has no way to guess. Returns `None` for any
    pairing with nothing special to say (e.g. two plain, non-IA
    contributions) - same as `get_effective_pk`.
    """
    bank = SpectraBank(matter_pk, ks, zs)
    return get_effective_pk(c1, c2, bank)


@dataclass(frozen=True)
class PkTerm:
    """One additive term of a contribution-pair's Limber integrand:
    `Cl += Cl_integration(kernel1, kernel2, pk, ...)`.

    Generalizes `get_effective_pk` (a single `(compute_kernel, compute_
    kernel, effective_pk)` triple per pairing) to a *sum* of triples with
    their own, *different* kernels - needed whenever a contribution's bias
    amplitudes vary per tomographic bin (e.g.
    `NonLinearGalaxyBiasContribution`'s per-bin b1/b2/bs2/b3nl/bk2): its
    P_gg(k,z) is a sum of terms like `0.5*(b1_i*b2_j + b1_j*b2_i)*Pd1d2(k)`,
    which isn't expressible as one shared kernel pair times one shared
    effective Pk (the standard `get_effective_pk` contract) because the
    bin-dependence lives in *different bias-weighted kernels* per term, not
    in one amplitude-free kernel times one Pk. Contributions whose
    amplitudes are global (not per-bin), e.g. `TATTContribution`, have no
    need for this and keep using `get_effective_pk`.

    Attributes:
      kernel1: window kernel for the first tracer, shape `(n_bin1, len(zs))` -
        not necessarily this term's own contribution's `compute_kernel(z)`
        output.
      kernel2: window kernel for the second tracer, shape `(n_bin2, len(zs))` -
        same caveat as `kernel1`.
      pk: the power spectrum this term should be integrated against,
        shape `(len(bank.zs), len(bank.ks))` - as `get_effective_pk`
        returns, but scoped to just this one additive term.
    """

    kernel1: jnp.ndarray
    kernel2: jnp.ndarray
    pk: jnp.ndarray


def get_pk_terms(c1, c2, zs, bank: SpectraBank):
    """The `PkTerm`s for the pairing `(c1, c2)`, or `None`.

    Tries `c1.get_pk_terms(c2, zs, bank)` first, then `c2.get_pk_terms(c1,
    zs, bank)` - mirrors `get_effective_pk`'s own order-dependent
    resolution (see its docstring), generalized to a list of terms instead
    of one. A `Contribution.get_pk_terms(other, zs, bank)` implementation
    must return its terms in `(self, other)` order regardless of which
    tracer position it's called from - this resolver swaps `kernel1`/
    `kernel2` back into `(c1, c2)` order when `c2` is the side that
    actually defined the terms, so callers never need to know which side
    supplied them.

    `None` means neither side needs this - callers should fall back to
    `get_effective_pk` (still `None` for a pairing needing neither).
    """
    getter1 = getattr(c1, "get_pk_terms", None)
    if getter1 is not None:
        terms = getter1(c2, zs, bank)
        if terms is not None:
            return terms
    getter2 = getattr(c2, "get_pk_terms", None)
    if getter2 is not None:
        terms = getter2(c1, zs, bank)
        if terms is not None:
            return tuple(PkTerm(t.kernel2, t.kernel1, t.pk) for t in terms)
    return None
