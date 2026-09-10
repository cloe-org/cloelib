# Contribution-Based Tracer Architecture

**Scope:** How `cloelib` supports TATT intrinsic alignment (and, by the same
mechanism, future non-linear galaxy bias / galaxy evolution effects)
without changing the public `ShearTracer`/`PositionsTracer`/`CMBLensingTracer`/
`AngularTwoPoint` interface for anyone not opting in.
**Companion:** `toy_cloelib` (`toy_cloelib/ARCHITECTURE_REVIEW.md`,
`toy_cloelib/STATUS.md`) - the sandbox project this design's central idea
(`Contribution`/`SpectrumComputer`) is adapted from.

---

## 1. Why this exists

`Tracer`, `Background`, `Perturbations` are `typing.Protocol` classes
(`cloelib/observables/photo/tracer.py`, `cloelib/cosmology/cosmology.py`) -
concrete classes satisfy them structurally, with no inheritance. The real
obstacle to adding TATT wasn't a class hierarchy; it was that physics used
to be pre-fused: `ShearTracer.get_window()` added the lensing window and a
hardcoded NLA intrinsic-alignment window together before anything
downstream ever saw them, and `AngularTwoPoint.get_Cl` paired whatever
`get_window()` returned against a single shared matter power spectrum grid.
TATT needs several correlators (density-density, density-IA, IA-IA) with
different k/z-scaling, combined at the _contribution_ level, not the
_window_ level - "sum windows, then integrate once" can't express that.

## 2. Design

### 2.1 `Contribution`

A `Contribution` (`cloelib/observables/photo/contributions.py`) is one
additive term of what used to be fused into a tracer's `get_window`. Each
tracer holds a small, fixed set of them and sums their kernels:

```python
class Contribution(Protocol):
    def compute_kernel(self, z) -> Array: ...
```

`ShearTracer` holds `(LensingContribution, ia_contribution)`;
`PositionsTracer` holds `(bias_contribution, MagnificationContribution)`.
`get_window(z)` is exactly `sum(c.compute_kernel(z) for c in
self.get_contributions())` (plus the multiplicative-bias factor for shear) -
for every contribution combination that exists, this reproduces the
pre-Contribution formula exactly, which is what
`tests/test_photo_tracers_characterization.py` pins bit-for-bit.

Three _optional_ methods extend this for models that need more than the
plain matter power spectrum (TATT is the only one that uses them today):

- `get_spectrum_requests() -> Sequence[SpectrumRequest]` - derived power
  spectra this contribution needs beyond P(k,z).
- `get_requirements_for_interaction(other) -> Sequence[SpectrumRequest]` -
  context-aware pruning (e.g. TATT drops its II-only terms when paired
  with a non-IA contribution).
- `get_effective_pk(other, bank) -> Optional[Array]` - the effective
  P(k,z) this contribution's pairing with `other` should be integrated
  against.

`spectrum_engine.py` looks these up via `getattr(obj, name, None)` rather
than requiring them on the base `Contribution` Protocol, so nothing about
the simpler contributions (`LensingContribution`,
`NLAContribution`, `GalaxyBiasContribution`,
`MagnificationContribution`) needs to change.

`IntrinsicAlignmentContribution` (`contributions.py`) is the general base
every IA model shares: `NLAContribution` and `TATTContribution`
(`photo/shear.py`) both subclass it, so the _specific_ model is always the
class name a reader sees (`NLAContribution`, `TATTContribution`), and the
_general_ "this is some IA model" question is a single, unambiguous
`isinstance(x, IntrinsicAlignmentContribution)` check - not a name that
could be mistaken for the general concept while actually only being one
specific model (`NLAContribution` used to be called
`IntrinsicAlignmentContribution` itself, which was exactly that trap).

It's a plain base class, not a `Protocol`, deliberately:
`TATTContribution.get_requirements_for_interaction`/`get_effective_pk` use
`isinstance(other, IntrinsicAlignmentContribution)` to ask "is the _other_
side of this pairing also IA?", and that's a nominal question, not a
structural one - `LensingContribution` and `NLAContribution` both expose
exactly `compute_kernel(z)` (they satisfy the same `Contribution`
Protocol), so no structural feature distinguishes "IA" from "not IA" for a
`Protocol`-based `isinstance` check to key off. An empty
`@runtime_checkable` Protocol matches _any_ object under `isinstance` (it
has no members to check for), which would make the II-vs-GI pruning this
base class exists for silently apply to everything - a concrete, checkable
failure mode, not a style preference. Nominal typing (a real base class) is
correct here.

### 2.2 The generalized spectrum engine

`spectrum_engine.py` is the machinery a `Contribution` uses to declare "I
need these extra derived spectra" and get them computed once, only if
actually needed for the pairing at hand, before the Limber integral runs -
the same principle `toy_cloelib.engine` demonstrates, adapted here without
new dependencies (no `equinox`/`plum`): plain dataclasses and a
`getattr(..., default)` lookup, the same pattern `AngularTwoPoint` already
uses for its `tracer_rules` type-pair dispatch.

`AngularTwoPoint.get_Cl` forks on a cheap, static check,
`needs_generalized_engine(contributions1, contributions2)`:

```python
def get_Cl_tensor(self, ells, nl, ks) -> jax.numpy.ndarray:
    if needs_generalized_engine(contributions1, contributions2):
        return self._compute_cl_generalized(...)
    return self._compute_cl_legacy(...)              # original, untouched

def get_Cl(self, ells, nl, ks) -> dict:
    C_ell_calc = self.get_Cl_tensor(ells, nl, ks)
    return self._package_cl(C_ell_calc, ells)         # unchanged either way
```

`_compute_cl_legacy` is the original `get_Cl` body: one shared Pk grid,
tracer-level windows, the existing jitted `Cl_integration`/`Pkl_interp`
kernels. It runs, unmodified, for every configuration that doesn't opt into
a generalized-engine-aware contribution - which is `False` for any tracer
built with today's defaults (`ia_model=None` - no IA contribution - or
`ia_model="NLA"`, plus linear galaxy bias). Only
`ia_model="TATT"` (or a future non-linear-bias contribution) takes
`_compute_cl_generalized`, which sums `Cl_integration(...)` over every
`(c1, c2)` contribution pair, each against its own effective P(k,z)
(`spectrum_engine.get_effective_pk`, falling back to the plain matter Pk
when a pair has nothing special to say).

The generalized path's effective spectra are generically _signed_ (TATT's
`C1(z)` carries a minus sign, so its GI cross-term is legitimately
negative) - `Pkl_interp_signed` (`angular_two_point.py`) handles this by
interpolating `log10(|Pk|)` and `sign(Pk)` separately and recombining, and
clamps its query k to the grid's own domain rather than extrapolating (see
its docstring: a real, steep, sign-changing kernel has no well-defined
extrapolation law, unlike the smooth matter Pk `Pkl_interp` extrapolates).
The original log-log `Pkl_interp` is untouched, still used by the legacy
path.

### 2.3 Differentiability: `get_Cl_tensor`

`get_Cl`'s own return value - a `dict` of `cosmolib.data.photo.
AngularPowerSpectrum` objects - can never be differentiated via
`jax.grad`: that dataclass's `__post_init__` unconditionally does
`np.asarray(self.array, dtype=float)`, a plain NumPy cast in an external
package with no notion of JAX, which raises `TracerArrayConversionError`
the moment a traced value reaches it. This is true regardless of whether
the _physics_ itself is differentiable - and with `cloelib.cosmology.
jax_cosmology` (`JAXBackground`/`JAXLinearPerturbations`/
`JAXNonLinearPerturbations`, the one cosmology backend that's pure JAX
end-to-end), it is: the Limber integral, window functions, growth-factor
ODE solve, and halofit all differentiate cleanly, for both NLA and TATT's
own amplitude parameters (confirmed against finite differences in
`tests/test_get_cl_tensor.py` and `playground/tutorials/observables/
photo_autodiff.ipynb`). TATT's one-loop kernels themselves are the one
hard exception: `PBJTATTLoopComputer` calls FAST-PT (plain NumPy/SciPy),
so gradients through cosmological parameters that would affect the
kernels' _shape_ aren't available - only through TATT's own amplitude
parameters, which multiply the (fixed) kernel values.

`get_Cl_tensor(ells, nl, ks)` exposes the exact same computation `get_Cl`
runs - literally the same `C_ell_calc` tensor - without the packaging step
that breaks it, as a genuine public method (not a workaround via private
methods). `get_Cl` itself is unchanged: it now calls `get_Cl_tensor`
internally and packages the result, same as before. Fixing this at the
root would mean either changing `cosmolib`'s dataclass (external,
out of cloelib's control) or changing `get_Cl`'s return type for every
existing caller - `get_Cl_tensor` sidesteps the problem instead of
"fixing" it, which is why it exists as an addition rather than a change
to `get_Cl`.

## 3. TATT

`TATTContribution` (`cloelib/observables/photo/shear.py`) implements Eqs.
(9)-(16) of Navarro-Gironés et al. (2026, arXiv:2602.16448, "Euclid
preparation. CIV. Impact of galaxy intrinsic alignment modelling choices on
Euclid 3x2pt cosmology"), which follows Blazek et al. (2019) for the TATT
model and one-loop kernel definitions. It declares ten one-loop
`SpectrumRequest`s, pruned from 10 (needed for II, the shear-shear
self-correlation) to 4 (needed for GI, the position-shear cross-correlation)
depending on whether the paired contribution is IA-like.

Selected via one new, optional `ShearTracer` kwarg:

```python
ShearTracer(perturbations, dndz, z, nuisance_params, ia_model="TATT")
```

reading `nuisance_params["AIA"/"A2IA"/"bTA"]` (`=A1`/`A2`/`b_TA`) and
optionally `["EtaIA"/"Eta2IA"/"z0IA"]`. `ia_model=None` (the default) means
no intrinsic-alignment contribution at all; `ia_model="NLA"` reproduces the
pre-existing `NLAContribution` behavior exactly; passing a `Contribution`
instance directly is also supported, for any other model.

### Kernel backend

The ten one-loop kernel _values_ come from a required `tatt_loop_computer`
(`TATTContribution` has no illustrative default - `ShearTracer` raises
`ValueError` if `ia_model="TATT"` is used without one, rather than silently
reporting made-up numbers). **`PBJTATTLoopComputer`** is the real-physics
implementation: it calls `fastpt.FASTPT.IA_ta`/`.IA_tt`/`.IA_mix` (the
`fast-pt` PyPI package, an optional dependency -
`pip install cloelib[fastpt]`) on the linear matter power spectrum, with
the same extrapolation settings and `C_window` production CLOE uses. This
is exactly how TATT was implemented in production CLOE
(`github.com/cloe-org/CLOE`, `cloe/non_linear/miscellanous.py::
ia_tatt_terms`, `cloe/non_linear/pLL_phot.py::Pii_ee_halo_tatt`/
`Pdeltai_halo_tatt`) - verified directly against that source: same three
FAST-PT calls, same kernel names, same `c1**2*Pdd + 2*c1*c1d*D**4*
(a00e+c00e) + ...` assembly `TATTContribution.get_effective_pk` implements.
`cloelib`'s own `pbjcosmo`-based PBJ interface (`spectro/PBJ_spectro.py`)
has no IA/TATT support of its own (verified against `pbjcosmo` 1.6.1's
published source: its PT wrapper class subclasses a `fastpt` variant
without `IA_*` methods), but `pbjcosmo` itself depends on this same
`fast-pt` package as its PT engine - `PBJTATTLoopComputer` goes straight to
`fast-pt`, independent of whether `pbjcosmo` is installed. Any other object
exposing the same `.compute(name) -> callable(matter_pk, ks, zs)`
interface is also accepted, for a different PT backend.

Pass it via the same constructor call - no manual
`tracer.ia = TATTContribution(...)` rebuild, no separately-tracked linear
`Perturbations` object:

```python
from cloelib.observables.photo.shear import PBJTATTLoopComputer

tracer = ShearTracer(
    perturbations=perturbations,  # the tracer's usual (possibly nonlinear) backend
    dndz=dndz, z=z,
    nuisance_params={**nuisance_params, "A2IA": 0.4, "bTA": -0.83},
    ia_model="TATT",
    tatt_loop_computer=PBJTATTLoopComputer(perturbations),
)
```

`PBJTATTLoopComputer` resolves the actual _linear_ source itself (FAST-PT's
one-loop integrals are only valid starting from linear P(k)) via
`perturbations.linearperturbations` when that attribute is present - every
nonlinear backend built from a separate linear one sets it
(`HMemuNonLinearPerturbations`, `EE2NonLinearPerturbations`,
`BACCOemuNonLinearPerturbations`, `EmantisFofrNonLinearPerturbations`,
`JAXNonLinearPerturbations`) - falling back to using the given object
directly if it has no such attribute (i.e. it's already linear). Passing
`tatt_loop_computer` with any `ia_model` other than `"TATT"` raises
`ValueError` rather than silently ignoring it.

### Known limitations

- **`CAMBNonLinearPerturbations` gap.** It does not (yet) set
  `.linearperturbations`, and its own `matter_power_spectrum` is always
  nonlinear - passing one to `PBJTATTLoopComputer` today silently uses the
  nonlinear Pk as FAST-PT input, which is physically wrong. Flagged in
  `PBJTATTLoopComputer`'s docstring, not fixed here.
- **B-modes.** TATT's real `P_II^BB` (Eq. 14) is computed internally by
  `get_effective_pk` when both sides are IA-like, but `AngularTwoPoint`'s
  SHE-SHE output packaging (`she_she_rule`) still hardcodes B-mode output
  slots to zero. Wiring real B-modes into that output means changing what
  that dict's B-mode slot means for TATT-enabled tracers without breaking
  it for everyone else - not done here.
- **Cross-population TATT pairing** (two _different_ `TATTContribution`
  instances with different `A1`/`A2`/`b_TA`) uses the natural bilinear
  generalization of Eq. 13, not something the reference paper states
  explicitly (it only writes the single-population form). Reduces to
  Eq. 13 exactly when both sides are the same instance; the general case is
  this codebase's own extrapolation, not a citation.
- **RSD.** `PositionsTracer(include_rsd=True)` paired with a
  generalized-engine-requiring contribution raises `NotImplementedError` -
  a documented gap, not silently wrong physics.

## 4. Module layout

```
cloelib/observables/
├── photo/                  # ShearTracer, PositionsTracer, and everything they build on
│   ├── shear.py              # ShearTracer + its IA models: LensingContribution,
│   │                          # NLAContribution, TATTContribution,
│   │                          # PBJTATTLoopComputer
│   ├── positions.py           # PositionsTracer + GalaxyBiasContribution,
│   │                          # MagnificationContribution, RSD helpers
│   ├── tracer.py              # the Tracer protocol
│   ├── contributions.py       # the Contribution protocol, IntrinsicAlignmentContribution base
│   └── spectrum_engine.py     # SpectrumRequest, SpectraBank, needs_generalized_engine, ...
├── spectro/                 # SpectroPower and its backends
│   ├── spectro.py             # the SpectroPower protocol
│   ├── PBJ_spectro.py
│   ├── CometEFT_spectro.py
│   └── CometVDG_spectro.py
└── cmb.py                   # CMBLensingTracer
```

`ShearTracer`/`PositionsTracer`/`SpectroPower` are re-exported at
`cloelib.observables.photo`/`cloelib.observables.spectro` respectively, so
`from cloelib.observables.photo import ShearTracer, PositionsTracer` and
`from cloelib.observables.spectro import SpectroPower` are the stable,
public import paths. Everything IA-model-specific for shear (including
both TATT backends) lives together in `photo.shear` - one place to import
from, rather than scattered across several top-level files.

## 5. Compatibility guarantees

- `PositionsTracer(perturbations, dndz, z, galaxy_bias_model,
nuisance_params, include_rsd=False)`, `CMBLensingTracer(perturbations, z)`
  signatures are unchanged. `ShearTracer`'s `ia_model` kwarg is additive and
  optional, but its default is **not** compatibility-preserving: `ia_model=
  None` (the default) means no intrinsic-alignment contribution at all -
  callers that want NLA (the historical, implicit behavior) must now pass
  `ia_model="NLA"` explicitly. This was a deliberate choice: an implicit
  default that silently ran a physics model was considered more surprising
  than requiring it be named. `tatt_loop_computer` is only meaningful (and
  required) with `ia_model="TATT"`; passing it otherwise raises
  `ValueError`.
- `AngularTwoPoint(tracer1, tracer2).get_Cl(ells, nl, ks)`'s return schema
  (`dict` keyed by `("POS"|"SHE"|"CMBL", ..., i, j)` →
  `AngularPowerSpectrum`) and its packaging logic (`_package_cl`,
  `tracer_rules`) are untouched and shared by both the legacy and
  generalized `Cl` computation paths.
- `tests/test_photo_tracers_characterization.py` pins `get_window`/`get_Cl`
  output for `ShearTracer`/`PositionsTracer`/`CMBLensingTracer` bit-for-bit
  as a regression gate; `tests/test_contributions.py` pins that the
  Contribution decomposition (`get_contributions()`) reproduces the
  directly-summed window methods exactly.
