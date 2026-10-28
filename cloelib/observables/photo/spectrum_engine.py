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

The same bank also holds what the beyond-Limber computation shares between
tracer pairs: the linear transfer function of each traced field, the
Chebyshev coefficients of the unequal-time *linear* power spectrum of each
pair of fields and, per pair of fields and T-tilde matrix, the projected
density `w_ell` (`SpectraBank.nonlimber_transfer`,
`SpectraBank.nonlimber_cheb`, `SpectraBank.w_ell`). They are cosmology
dependent but tracer independent, so a bank shared by the pairs of a 3x2pt
computation evaluates each of them once.

Nothing here changes behavior for any `Contribution` that doesn't declare
extra requests: `needs_generalized_engine` returns `False` for them, and
`AngularTwoPoint.get_Cl` keeps using its original, untouched fast path in
that case.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax.numpy as jnp

from cloelib.auxiliary.chebyshev import (
    chebyshev_coefficients,
    comoving_distance_to_redshift,
)
from cloelib.auxiliary.math_utils import safe_sqrt
from cloelib.auxiliary.nonlimber_kernels import unequal_time_pk, w_ell
from cloelib.auxiliary.t_tildes import TTildeStore
from cloelib.cosmology.cosmology import (
    CB_CROSS_CORRELATED,
    CB_CROSS_LINEAR_NU,
    CB_CROSS_PRESCRIPTIONS,
    Perturbations,
)
from cloelib.observables.photo.contributions import CB, MATTER


def _is_known_linear_perturbations(perturbations) -> bool:
    """Whether `perturbations` is itself one of cloelib's own linear-only
    backends (`HMemuLinearPerturbations`, `CAMBLinearPerturbations`, ...).

    Identified by cloelib's own `*LinearPerturbations` naming convention
    (consistently followed across every cosmology backend module) rather
    than an `isinstance` check against every concrete class: several
    backends (CAMB, CLASS, BACCOemu, MGCLASS, mochi_class, hi_class, ...)
    are optional dependencies, so importing all of them here just to check
    would make this module require every optional extra. Used only as the
    fallback when `perturbations` has no `.linearperturbations` attribute
    of its own - see `linear_source` and `PBJTATTLoopComputer.__init__`.

    Deliberately not a plain `.endswith("LinearPerturbations")`: a
    `*NonLinearPerturbations` class name (e.g. `CAMBNonLinearPerturbations`)
    also ends with that suffix (`"NonLinearPerturbations"` itself ends with
    `"LinearPerturbations"`) - excluding any name containing `"NonLinear"`
    is what actually distinguishes the two conventions.
    """
    name = type(perturbations).__name__
    return name.endswith("LinearPerturbations") and "NonLinear" not in name


def linear_source(perturbations) -> Optional[Perturbations]:
    """The linear perturbations behind `perturbations`, or `None`.

    `perturbations.linearperturbations` when the backend exposes one;
    otherwise `perturbations` itself if it is one of cloelib's linear
    backends (`_is_known_linear_perturbations`); otherwise `None`.
    """
    linear = getattr(perturbations, "linearperturbations", None)
    if linear is not None:
        return linear
    if _is_known_linear_perturbations(perturbations):
        return perturbations
    return None


def _check_fields(*fields: str) -> None:
    """Raise for a traced field other than `MATTER` and `CB`."""
    unknown = set(fields) - {MATTER, CB}
    if unknown:
        raise ValueError(f"Unknown traced field(s) {sorted(unknown)}.")


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
    """Memoized store of the power-spectrum grids for one `get_Cl` call.

    Holds two kinds of grids on the `(zs, ks)` grid, each computed at most
    once and only when first asked for:

    - base spectra, `base(field1, field2)`: `P_mm`, `P_cb` and the cb x
      matter cross spectrum, combined from the backend's `P_mm` and `P_cb`
      (and their linear counterparts when the cross spectrum needs them);
    - derived spectra, `get(request)`: e.g. TATT's one-loop kernels,
      computed from `matter_pk` (`P_mm`).

    Not shared across calls - `AngularTwoPoint` builds one per `get_Cl` with
    `from_perturbations`. Built directly from an already evaluated
    `matter_pk` instead (`SpectraBank(matter_pk, ks, zs)`), only the total
    matter is available: `base` raises for any pair involving cb.

    When configured for it (`t_tildes`, and a linear spectrum: see
    `linear_perturbations` below), it also holds the grids of the beyond-Limber integration
    (`nonlimber_transfer`, `nonlimber_cheb`, `w_ell`, built from the
    *linear* spectra and the T-tilde matrices),
    computed lazily as well. These depend on the cosmology only, not on the
    tracers: one bank can serve every tracer pair of a 3x2pt computation,
    so that what pairs have in common (the transfer functions, the
    Chebyshev coefficients, each `w_ell`) is computed once.

    Args:
      matter_pk: matter power spectrum on `(zs, ks)`, shape
        `(len(zs), len(ks))`, or `None` to evaluate it from `perturbations`.
      ks: wavenumber grid of every spectrum.
      zs: redshift grid of every spectrum.
      perturbations: backend the base spectra are evaluated from.
      linear_perturbations: optional override of the linear `Perturbations`
        the beyond-Limber spectra (and the background that maps comoving
        distance to redshift) come from. By default they are resolved from
        `perturbations` (`linear_source`): its `linearperturbations`, or
        itself if it is a linear backend. Needed only for backends that do
        not expose their linear spectrum (e.g. `CAMBNonLinearPerturbations`).
      t_tildes: the T-tilde matrices, and the grids they were computed on.
        Needed for `nonlimber_cheb`/`w_ell` only.
    """

    def __init__(
        self,
        matter_pk: jnp.ndarray | None,
        ks: jnp.ndarray,
        zs: jnp.ndarray,
        perturbations=None,
        *,
        linear_perturbations: Optional[Perturbations] = None,
        t_tildes: Optional[TTildeStore] = None,
    ):
        if matter_pk is None and perturbations is None:
            raise ValueError("SpectraBank needs `matter_pk` or `perturbations`.")
        self.ks = ks
        self.zs = zs
        self.perturbations = perturbations
        self.linear_perturbations = linear_perturbations
        self.t_tildes = t_tildes
        self._spectra: dict[str, jnp.ndarray] = {}
        if matter_pk is not None:
            self._spectra["P_mm"] = matter_pk
        self._grids: dict[str, jnp.ndarray] = {}

    @classmethod
    def from_perturbations(
        cls,
        perturbations,
        ks,
        zs,
        *,
        linear_perturbations: Optional[Perturbations] = None,
        t_tildes: Optional[TTildeStore] = None,
    ) -> "SpectraBank":
        """Bank evaluating every spectrum lazily from `perturbations`."""
        return cls(
            None,
            ks,
            zs,
            perturbations=perturbations,
            linear_perturbations=linear_perturbations,
            t_tildes=t_tildes,
        )

    @property
    def matter_pk(self) -> jnp.ndarray:
        """The total matter power spectrum `P_mm`, shape `(len(zs), len(ks))`."""
        return self.base(MATTER, MATTER)

    def _require_perturbations(self, name: str):
        """`perturbations`, or a clear error for a bank built from `matter_pk`."""
        perturbations = self.perturbations
        if perturbations is None:
            raise ValueError(
                f"This SpectraBank was built from `matter_pk` alone and "
                f"cannot provide {name}; use `SpectraBank.from_perturbations`."
            )
        return perturbations

    def _spectrum(self, name: str) -> jnp.ndarray:
        """One of the backend's spectra, evaluated on first use.

        `P_mm`/`P_cb` come from `perturbations` itself, `P_mm_lin`/`P_cb_lin`
        from its `linearperturbations`.
        """
        if name not in self._spectra:
            perturbations = self._require_perturbations(name)
            source = (
                perturbations.linearperturbations
                if name.endswith("_lin")
                else perturbations
            )
            method = (
                source.matter_power_spectrum_cb
                if name.startswith("P_cb")
                else source.matter_power_spectrum
            )
            self._spectra[name] = method(self.zs, self.ks)
        return self._spectra[name]

    def _cb_cross_matter(self) -> jnp.ndarray:
        """cb cross matter power spectrum.

        The formula is the one the backend declares in its
        `cb_cross_prescription` class attribute (see
        `cloelib.cosmology.cosmology`):

        - `CB_CROSS_CORRELATED`: cb and the total matter perfectly
          correlated, `sqrt(P_cb P_mm)`;
        - `CB_CROSS_LINEAR_NU`: with linear neutrinos,
          `P_cb,m = f_cb P_cb + f_nu P_cb,nu` with a linear `P_cb,nu`, i.e.
          `f_cb (P_cb^NL - P_cb^L) + sqrt(P_cb^L P_mm^L)`, with the linear
          spectra from the backend's `linearperturbations`.

        The prescription is a static Python string, so the choice is safe
        under `jax.jit`.

        Raises:
            ValueError: if the backend does not declare a known prescription,
                or declares `CB_CROSS_LINEAR_NU` without `linearperturbations`.
        """
        perturbations = self._require_perturbations("the cb x matter spectrum")
        prescription = getattr(perturbations, "cb_cross_prescription", None)
        if prescription not in CB_CROSS_PRESCRIPTIONS:
            raise ValueError(
                f"{type(perturbations).__name__} must declare the class attribute "
                f"`cb_cross_prescription` as one of {CB_CROSS_PRESCRIPTIONS} to "
                f"provide the cb x matter spectrum (got {prescription!r})."
            )
        if prescription == CB_CROSS_CORRELATED:
            return safe_sqrt(self._spectrum("P_cb") * self._spectrum("P_mm"))
        if getattr(perturbations, "linearperturbations", None) is None:
            raise ValueError(
                f"{type(perturbations).__name__} declares "
                f"cb_cross_prescription={CB_CROSS_LINEAR_NU!r}, which needs its "
                f"`linearperturbations`, but it has none."
            )
        background = perturbations.background
        f_cb = jnp.squeeze(background.Omega_cb(0.0)) / jnp.squeeze(
            background.Omega_m(0.0)
        )
        Pcb_l = self._spectrum("P_cb_lin")
        return f_cb * (self._spectrum("P_cb") - Pcb_l) + safe_sqrt(
            Pcb_l * self._spectrum("P_mm_lin")
        )

    def base(self, field1: str, field2: str) -> jnp.ndarray:
        """Base power spectrum for a pair of traced fields.

        `P_mm` for `(MATTER, MATTER)`, `P_cb` for `(CB, CB)` and the cb x
        matter cross spectrum for a mixed pair, with the fields given by
        `contributions.contribution_field`. The fields are static Python
        strings, so the choice is resolved at trace time and is safe under
        `jax.jit`/`jax.grad`.

        Returns:
            Array: shape `(len(zs), len(ks))`.
        """
        _check_fields(field1, field2)
        fields = {field1, field2}
        if fields == {MATTER}:
            return self._spectrum("P_mm")
        if fields == {CB}:
            return self._spectrum("P_cb")
        if "P_cbxm" not in self._spectra:
            self._spectra["P_cbxm"] = self._cb_cross_matter()
        return self._spectra["P_cbxm"]

    def get(self, request: SpectrumRequest) -> jnp.ndarray:
        """Return the named derived grid, computing and caching it on first use."""
        if request.name not in self._grids:
            self._grids[request.name] = request.compute(
                self.matter_pk, self.ks, self.zs
            )
        return self._grids[request.name]

    # -- beyond Limber -------------------------------------------------------
    def _require_nonlimber(self) -> tuple[Perturbations, TTildeStore]:
        """The linear perturbations and T-tilde matrices of the beyond-Limber grids.

        The linear perturbations are the `linear_perturbations` override if
        given, else resolved from `perturbations` (`linear_source`).
        """
        if self.t_tildes is None:
            raise ValueError(
                "Beyond-Limber quantities need a SpectraBank built with `t_tildes`."
            )
        linear = self.linear_perturbations
        if linear is None and self.perturbations is not None:
            linear = linear_source(self.perturbations)
        if linear is None:
            name = type(self.perturbations).__name__
            raise ValueError(
                "Beyond-Limber quantities need the linear power spectrum, but "
                + (
                    "this SpectraBank has no `perturbations`"
                    if self.perturbations is None
                    else f"{name!r} exposes no `.linearperturbations` and is not "
                    "itself a linear backend"
                )
                + "; pass it with `linear_perturbations=`."
            )
        return linear, self.t_tildes

    def _memoized(self, key: str, compute: Callable[[], jnp.ndarray]) -> jnp.ndarray:
        if key not in self._grids:
            self._grids[key] = compute()
        return self._grids[key]

    def z_of_chi(self) -> jnp.ndarray:
        """Redshift of every comoving distance of the T-tilde grid, shape `(n_chi,)`."""
        linear, store = self._require_nonlimber()
        return self._memoized(
            "nonlimber:z_of_chi",
            lambda: comoving_distance_to_redshift(store.grid.chi, linear.background),
        )

    def nonlimber_transfer(self, field: str) -> jnp.ndarray:
        r"""Linear transfer function of a traced field on the T-tilde grid.

        $T_X(k, \chi) = \sqrt{P^{\rm lin}_{XX}(k, z(\chi))}$, at the
        redshifts $z(\chi)$ and at the Chebyshev nodes in $k$; the primordial
        spectrum is absorbed into $T$ (Eqs. 14-15 of arXiv:2609.01855).
        `P_mm` for `MATTER`, `P_cb` for `CB`.

        Returns:
          Array: shape `(n_chi, n_cheb)`.
        """
        _check_fields(field)
        linear, store = self._require_nonlimber()

        def compute():
            method = (
                linear.matter_power_spectrum_cb
                if field == CB
                else linear.matter_power_spectrum
            )
            return jnp.sqrt(method(self.z_of_chi(), 10.0**store.grid.k_cheb))

        return self._memoized(f"nonlimber:T:{field}", compute)

    def nonlimber_cheb(
        self, field_A: str = MATTER, field_B: str = MATTER
    ) -> jnp.ndarray:
        r"""Chebyshev coefficients, in $\log k$, of the unequal-time *linear* Pk.

        The spectrum is $P_{AB}(k, \chi, R\chi) = T_A(k, \chi)\,T_B(k, R\chi)$
        (`unequal_time_pk`), with field A at $\chi$ and field B at $R\chi$,
        expanded along $k$. For $A \neq B$ the two orderings differ, and
        each is computed (and memoized) on its own. Shared by every pair of
        kernels tracing these fields: this is the cosmology-dependent part
        of the beyond-Limber computation that the pairs have in common.

        Returns:
          Array: shape `(n_cheb, n_chi, n_R)`, the layout `w_ell` expects.
        """
        _check_fields(field_A, field_B)
        _, store = self._require_nonlimber()

        def compute():
            grid = store.grid
            Pk = unequal_time_pk(
                self.nonlimber_transfer(field_A),
                self.nonlimber_transfer(field_B),
                grid.chi,
                grid.R,
            )
            return jnp.apply_along_axis(chebyshev_coefficients, 0, Pk)

        return self._memoized(f"nonlimber:cheb:{field_A}:{field_B}", compute)

    def w_ell(
        self, t_name: str, field_A: str = MATTER, field_B: str = MATTER
    ) -> jnp.ndarray:
        r"""Projected density $w_\ell(\chi, R)$ for the T-tilde matrix `t_name`.

        Contracts `nonlimber_cheb(field_A, field_B)` with the matrix (e.g.
        `"T_minus2_00"`); computed once per `(field_A, field_B, t_name)`, and
        the matrix is only read from disk when a name is first asked for.
        Field A is the one at $\chi$, field B the one at $R\chi$.

        Returns:
          Array: shape `(n_ell, n_chi, n_R)`.
        """
        _check_fields(field_A, field_B)
        _, store = self._require_nonlimber()
        return self._memoized(
            f"nonlimber:w_ell:{field_A}:{field_B}:{t_name}",
            lambda: w_ell(self.nonlimber_cheb(field_A, field_B), store.get(t_name)),
        )


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


def get_effective_pk(c1, c2, bank: SpectraBank):
    """The effective P(k,z) for the pairing `(c1, c2)`, or `None`.

    Tries `c1.get_effective_pk(c2, bank)` first, then `c2.get_effective_pk(
    c1, bank)` - the P(k,z) two contributions should be integrated against
    is a property of the *pairing*, not of either side alone (e.g. a plain
    `LensingContribution` never defines this, but a `TATTContribution` it's
    paired with does, regardless of argument order: P_deltaI(k,z) is the
    same scalar grid whichever contribution is asked). `None` means neither
    side has anything special to say - use the pair's base spectrum
    (`SpectraBank.base`), i.e. every non-TATT pairing.
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
