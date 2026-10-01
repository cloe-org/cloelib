r"""Cosmic shear tracer: `ShearTracer`, its Contributions, and its
intrinsic-alignment models.

Compatible with the Tracer protocol. Counterpart to `photo.positions`,
which holds `PositionsTracer`.

Everything needed to build and use a shear tracer - including its
intrinsic-alignment options - lives in this one module:

- `ShearTracer` itself. By default (`ia_model=None`) it has no
  intrinsic-alignment contribution at all - only `LensingContribution`.
  Pass `ia_model="NLA"` explicitly for `NLAContribution` (the nonlinear
  alignment model), `ia_model="TATT"` for `TATTContribution`, or
  `ia_model="TATT-M"` for `TATTMContribution`.
- `LensingContribution`, `NLAContribution`: two of the `Contribution`s a
  `ShearTracer` can be built from.
- `TATTContribution` (Eqs. 9-16 of Navarro-Gironés et al. 2026,
  arXiv:2602.16448, "Euclid preparation. CIV. Impact of galaxy intrinsic
  alignment modelling choices on Euclid 3x2pt cosmology"): an alternative
  IA model, selected via `ShearTracer(..., ia_model="TATT",
  tatt_loop_computer=...)`. Its ten one-loop kernels come from a required
  `loop_computer` - `PBJTATTLoopComputer`: real physics, via
  `fastpt.FASTPT.IA_ta`/`.IA_tt`/`.IA_mix` (the `fast-pt` PyPI package - an
  optional dependency, `pip install cloelib[fastpt]`) on the linear matter
  power spectrum. The same three FAST-PT calls, kernel names, and
  `c1**2*Pdd + 2*c1*c1d*D**4*(a00e+c00e) + ...` assembly production CLOE
  used (github.com/cloe-org/CLOE, `cloe/non_linear/miscellanous.py`/
  `pLL_phot.py`) - verified directly against that source. `cloelib`'s own
  `pbjcosmo`-based PBJ interface (`spectro/PBJ_spectro.py`) has no IA/TATT
  support of its own (verified against pbjcosmo 1.6.1's published source:
  its own PT wrapper class subclasses a `fastpt` variant without `IA_*`
  methods), but `pbjcosmo` itself depends on this same `fast-pt` package as
  its PT engine - so this goes straight to `fast-pt`, independent of
  whether `pbjcosmo` is installed. `loop_computer` is a required argument
  (not defaulted to anything illustrative): `TATTContribution` only ever
  reports kernel values it can stand behind as real physics - passing any
  other object exposing the same `.compute(name) -> callable(matter_pk,
  ks, zs)` interface is also supported, for a different PT backend.
- `TATTMContribution` (Herle et al. 2026, arXiv:2601.15851): ties TATT's
  C1/C2/C1delta amplitudes to per-tomographic-bin halo mass and
  red-galaxy fraction instead of `TATTContribution`'s single global
  A1/A2/bTA. See `TATTMContribution`'s docstring for why this needs three
  internal rank-1 "component" Contributions rather than one bin-aware
  `TATTContribution`-like object.

Pass `loop_computer=PBJTATTLoopComputer(perturbations)` directly to
`ShearTracer(..., ia_model="TATT"` (or `"TATT-M"`) `, tatt_loop_computer=
...)` - one constructor call, not build-then-replace `.ia`.

See `CONTRIBUTION_ARCHITECTURE.md` for the design this all
follows.
"""

from typing import Dict, Optional

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson, simps
from cloelib.auxiliary.systematics import shift_dndz_jax, stretch_dndz_jax
from cloelib.observables.photo.contributions import (
    IntrinsicAlignmentContribution,
    Contribution,
)
from cloelib.observables.photo.spectrum_engine import (
    SpectraBank,
    SpectrumRequest,
    compute_effective_pk,
)

# General imports
import jax.numpy as np  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp
import interpax  # type: ignore
import numpy as _numpy
from scipy import interpolate as _scipy_interpolate


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s
# Same convention: SPEED_OF_LIGHT is in m/s.
_C_KM_S = SPEED_OF_LIGHT / 1000


class LensingContribution:
    """Weak-lensing shear kernel term of `ShearTracer.get_window()`."""

    def __init__(self, tracer: "ShearTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window_lensing(z)


class NLAContribution(IntrinsicAlignmentContribution):
    """NLA (nonlinear alignment model) intrinsic-alignment kernel term of
    `ShearTracer.get_window()`.

    Implemented by `ShearTracer.get_window_IA` - selected via
    `ia_model="NLA"` on `ShearTracer` (the default, `ia_model=None`, has no
    IA contribution at all). `ia_model="TATT"` swaps in `TATTContribution`
    here instead - a different intrinsic-alignment model, hence the
    distinct name; `get_window()` and `AngularTwoPoint` don't need to
    change either way.
    """

    def __init__(self, tracer: "ShearTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window_IA(z)


# ---------------------------------------------------------------------------
# TATT: the ten one-loop kernel names appearing in Eqs. (13)-(15), matching
# the paper's own subscript notation. Each is a pure function of k (no
# separate z-dependence - that lives entirely in the C1/C1delta/C2
# amplitudes and the explicit D(z)**4 prefactor).
# ---------------------------------------------------------------------------
_EE_ONLY_KERNELS = ("tatt_A_0_0E", "tatt_C_0_0E", "tatt_A_0E_0E", "tatt_A_E2_E2")
_SHARED_KERNELS = ("tatt_A_0_E2", "tatt_B_0_E2")
_BB_ONLY_KERNELS = (
    "tatt_D_0E_E2",
    "tatt_A_0B_0B",
    "tatt_A_B2_B2",
    "tatt_D_0B_B2",
)
_GI_KERNELS = ("tatt_A_0_0E", "tatt_C_0_0E", "tatt_A_0_E2", "tatt_B_0_E2")
_ALL_KERNELS = _EE_ONLY_KERNELS + _SHARED_KERNELS + _BB_ONLY_KERNELS


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
    of its own - see `PBJTATTLoopComputer.__init__`.

    Deliberately not a plain `.endswith("LinearPerturbations")`: a
    `*NonLinearPerturbations` class name (e.g. `CAMBNonLinearPerturbations`)
    also ends with that suffix (`"NonLinearPerturbations"` itself ends with
    `"LinearPerturbations"`) - excluding any name containing `"NonLinear"`
    is what actually distinguishes the two conventions.
    """
    name = type(perturbations).__name__
    return name.endswith("LinearPerturbations") and "NonLinear" not in name


class PBJTATTLoopComputer:
    r"""FAST-PT-backed computer for the ten TATT one-loop kernels.

    Calls `fastpt.FASTPT.IA_ta`/`.IA_tt`/`.IA_mix` on the *linear* matter
    power spectrum at z=0, with the same extrapolation settings and
    `C_window` production CLOE uses (see module docstring). The ten
    outputs are pure functions of k (no z-dependence - `TATTContribution`
    supplies that separately via `D(z)**4` and its C1/C1delta/C2
    amplitudes, exactly matching Eqs. 13-15's own separation of scales), so
    they're computed once per k-grid, cached, and reused across the ten
    named `SpectrumRequest`s (and across `get_Cl` calls, as long as the
    k-grid doesn't change) instead of re-running FAST-PT per kernel name.

    Takes the *same* `perturbations` object you'd pass to `ShearTracer` -
    typically nonlinear (a halo model/emulator backend, for the tree-level
    P_dd term), but FAST-PT's one-loop integrals are only valid starting
    from the linear power spectrum (the same distinction production CLOE
    draws between `Pk_delta` and `Pk_halomodel_recipe`), so this class
    resolves the actual linear source itself: `perturbations.
    linearperturbations` if that attribute exists (every nonlinear backend
    that's built *from* a separate linear one sets it -
    `HMemuNonLinearPerturbations`, `EE2NonLinearPerturbations`,
    `BACCOemuNonLinearPerturbations`, `EmantisFofrNonLinearPerturbations`,
    `JAXNonLinearPerturbations`), else `perturbations` itself *if and only
    if* it's itself a recognized `*LinearPerturbations` backend (see
    `_is_known_linear_perturbations`) - raises `ValueError` otherwise (PR
    #569 review): the absence of a `.linearperturbations` attribute does
    not by itself mean the object passed in *is* linear, and silently
    treating an unrecognized nonlinear backend as linear would be a
    physical error, not a numerical one (FAST-PT would still return
    finite-looking, wrong spectra) - the kind that shouldn't fail silently.
    `CAMBNonLinearPerturbations` is the concrete case this currently
    excludes: it does not set `.linearperturbations` (its own
    `matter_power_spectrum` is always nonlinear), so it now raises here
    instead of being misused - pass `CAMBLinearPerturbations` directly
    instead.

    TODO (PR #569 review): standardize access to the corresponding linear
    perturbations object across nonlinear cosmology backends (e.g. also
    wiring it up for `CAMBNonLinearPerturbations`), so PT-based observables
    can retrieve a linear P(k,z) through one common interface instead of
    this getattr-plus-name-check fallback.

    FAST-PT requires its input k-grid to be evenly log-spaced (an FFTLog
    requirement); `TATTContribution`'s own `ks` (whatever grid the calling
    `AngularTwoPoint.get_Cl` was invoked with, e.g. `perturbations.k` from
    an emulator's extended, non-uniform grid) generally isn't. Production
    CLOE handles this by running FAST-PT on its own dedicated log-uniform
    `k_win` grid and interpolating the results onto whatever `wavenumber`
    is actually needed (`Misc.ia_tatt_terms`'s `interp1d(..., kind=
    'linear', fill_value='extrapolate')`); this class does the same -
    builds a log-uniform grid spanning the requested `ks`' own range,
    runs FAST-PT there, and linearly interpolates (extrapolating past the
    edges, matching production CLOE) back onto `ks`.

    Requires the optional `fast-pt` dependency (`pip install
    cloelib[fastpt]`); raises `ImportError` with install instructions at
    construction time.
    """

    #: Same low/high-k extrapolation as production CLOE's
    #: `Misc.update_dic` (`fpt.FASTPT(..., low_extrap=-5, high_extrap=3)`).
    _LOW_EXTRAP = -5
    _HIGH_EXTRAP = 3
    #: Same C_window as production CLOE's `Misc.ia_tatt_terms` (tuned there
    #: to suppress FFTLog ringing).
    _C_WINDOW = 0.75

    def __init__(self, perturbations: Perturbations) -> None:
        try:
            import fastpt as fpt
        except ImportError as e:
            raise ImportError(
                "fastpt (the 'fast-pt' PyPI package - the perturbation-"
                "theory engine pbjcosmo's own PT backend, pbjcosmo.fptplus"
                f".FASTPTPlus, is itself built on) could not be imported: {e}"
                ". Install it with `pip install fast-pt` (or `pip install "
                "cloelib[fastpt]`)."
            )
        self._fpt = fpt
        # See class docstring: use the nonlinear backend's own linear
        # source if it has one; else, only accept `perturbations` itself if
        # it's a recognized linear backend - fail loudly otherwise rather
        # than silently treating an unrecognized object as linear.
        linear = getattr(perturbations, "linearperturbations", None)
        if linear is not None:
            self.linear_perturbations = linear
        elif _is_known_linear_perturbations(perturbations):
            self.linear_perturbations = perturbations
        else:
            raise ValueError(
                "PBJTATTLoopComputer requires a linear matter-power-"
                f"spectrum source, but {type(perturbations).__name__!r} "
                "exposes no `.linearperturbations` attribute and isn't "
                "itself a recognized *LinearPerturbations backend. Pass a "
                "linear perturbations object directly (e.g. "
                "CAMBLinearPerturbations), or a nonlinear backend that "
                "sets `.linearperturbations` (e.g. "
                "HMemuNonLinearPerturbations)."
            )
        self._cached_ks: Optional[_numpy.ndarray] = None
        self._cached_kernels: Optional[Dict[str, _numpy.ndarray]] = None

    def _kernels_for(self, ks) -> Dict[str, _numpy.ndarray]:
        ks_np = _numpy.asarray(ks)
        cached = self._cached_kernels
        if (
            cached is not None
            and self._cached_ks is not None
            and self._cached_ks.shape == ks_np.shape
            and _numpy.allclose(self._cached_ks, ks_np)
        ):
            return cached

        # FAST-PT (FFTLog) requires an evenly log-spaced k-grid of even
        # length; `ks_np` generally satisfies neither (e.g. an emulator's
        # extended grid), so build a dedicated one spanning the same range -
        # same pattern as production CLOE's separate `k_win` grid (see class
        # docstring).
        n_win = len(ks_np) + (len(ks_np) % 2)
        k_win = _numpy.logspace(
            _numpy.log10(ks_np.min()), _numpy.log10(ks_np.max()), n_win
        )

        # Backend-dependent return shape for a length-1 `zs` (same class of
        # inconsistency `_growth_factor_1d` already works around for
        # `growth_factor`): CAMB's `matter_power_spectrum` keeps an
        # explicit (1, n_k) z-axis, HMemu's `.squeeze()`s it away to (n_k,).
        # `reshape(-1)` normalizes either to the flat (n_k,) FAST-PT needs.
        p_lin_z0 = _numpy.reshape(
            _numpy.asarray(
                self.linear_perturbations.matter_power_spectrum(
                    _numpy.array([0.0]), k_win
                )
            ),
            (-1,),
        )

        f_pt = self._fpt.FASTPT(
            k_win,
            to_do=["IA"],
            low_extrap=self._LOW_EXTRAP,
            high_extrap=self._HIGH_EXTRAP,
            n_pad=n_win,
        )

        a00e, c00e, a0e0e, a0b0b = f_pt.IA_ta(
            p_lin_z0, P_window=None, C_window=self._C_WINDOW
        )
        ae2e2, ab2b2 = f_pt.IA_tt(p_lin_z0, P_window=None, C_window=self._C_WINDOW)
        a0e2, b0e2, d0ee2, d0bb2 = f_pt.IA_mix(
            p_lin_z0, P_window=None, C_window=self._C_WINDOW
        )

        raw_kernels = {
            "tatt_A_0_0E": a00e,
            "tatt_C_0_0E": c00e,
            "tatt_A_0E_0E": a0e0e,
            "tatt_A_0B_0B": a0b0b,
            "tatt_A_E2_E2": ae2e2,
            "tatt_A_B2_B2": ab2b2,
            "tatt_A_0_E2": a0e2,
            "tatt_B_0_E2": b0e2,
            "tatt_D_0E_E2": d0ee2,
            "tatt_D_0B_B2": d0bb2,
        }
        kernels = {
            name: _scipy_interpolate.interp1d(
                k_win, values, kind="linear", fill_value="extrapolate"
            )(ks_np)
            for name, values in raw_kernels.items()
        }
        self._cached_ks = ks_np
        self._cached_kernels = kernels
        return kernels

    def compute(self, name: str):
        def _compute(matter_pk, ks, zs):
            del matter_pk, zs  # pure k-kernel; z-dependence lives elsewhere
            return jnp.asarray(self._kernels_for(ks)[name])

        return _compute


def _growth_factor_1d(perturbations, z, k_pivot_hmpc: float = 0.03):
    """D(z) as a 1D array, robust to backend-dependent `growth_factor` shape.

    Same handling `ShearTracer.get_window_IA` uses: some
    backends (e.g. CAMB) return D(z, k) with an explicit k-axis; others
    (the JAX backends) ignore `ks` and return a scale-independent D(z), and
    don't set a `.k` attribute at all.

    When a k-axis is present, picks the tabulated k closest to a fixed
    physical pivot scale (`k_pivot_hmpc`, in h/Mpc, converted to the
    backend's own Mpc^-1 units via `background.h`) rather than a fixed
    grid index - growth is mildly scale-dependent once neutrinos are
    massive, and a fixed grid index (e.g. "index 1") silently drifts to a
    different physical k, and hence a slightly different D(z), if the
    tabulated k-grid's resolution or minimum changes. Matches CosmoSIS's
    own growth-pivot convention in tatt_interface.py (its `k_h > 0.03`
    cut), which otherwise disagrees with this one at the ~0.1% level for
    m_nu > 0.
    """
    ks = getattr(perturbations, "k", None)
    d_raw = perturbations.growth_factor(z, ks)
    if getattr(d_raw, "ndim", 1) != 2:
        return d_raw
    k_pivot = k_pivot_hmpc * perturbations.background.h
    pivot_idx = int(_numpy.argmin(_numpy.abs(_numpy.asarray(ks) - k_pivot)))
    return d_raw[:, pivot_idx]


class TATTContribution(IntrinsicAlignmentContribution):
    r"""TATT intrinsic-alignment contribution (Eqs. 9-16 of Navarro-Gironés
    et al. 2026, arXiv:2602.16448), see module docstring.

        C1(z)      = -A1 * C_IA * Omega_m0 * ((1+z)/(1+z0))**eta1 / D(z)
        C1delta(z) = b_TA * C1(z)
        C2(z)      = 5*A2 * C_IA * Omega_m0 / D(z)**2 * ((1+z)/(1+z0))**eta2

        P_II^EE(z,k) = C1(z)**2 * P_dd(z,k)
                     + 2*C1(z)*C1delta(z)*D(z)**4 * [A_0_0E(k) + C_0_0E(k)]
                     + C1delta(z)**2 * D(z)**4 * A_0E_0E(k)
                     + C2(z)**2 * D(z)**4 * A_E2_E2(k)
                     + 2*C1(z)*C2(z)*D(z)**4 * [A_0_E2(k) + B_0_E2(k)]
                     + 2*C1delta(z)*C2(z)*D(z)**4 * D_0E_E2(k)

        P_deltaI(z,k) = C1(z)*P_dd(z,k)
                      + C1delta(z)*D(z)**4 * [A_0_0E(k) + C_0_0E(k)]
                      + C2(z)*D(z)**4 * [A_0_E2(k) + B_0_E2(k)]

    `C_IA` bundles the paper's `C_bar_1 * rho_crit` product (the standard
    IA literature convention; `ShearTracer.get_window_IA`'s NLA
    implementation already uses one constant this way with the same
    default value, 0.0134 - see Brown et al. 2002).

    Args:
      tracer: the owning `ShearTracer`.
      A1: tidal-alignment amplitude (Table 2 of the reference paper).
      A2: tidal-torquing amplitude (Table 2).
      b_TA: tidal-alignment-of-galaxy-bias-tracers amplitude (Table 2).
      eta1: redshift-evolution index for `A1`'s (1+z)/(1+z0) scaling
        (Table 2; the full `zTATT` model uses all five of these).
      eta2: redshift-evolution index for `A2`'s (1+z)/(1+z0) scaling
        (Table 2).
      z0: pivot redshift for the (1+z)/(1+z0) scaling. Fixed at 0.62 in the
        reference paper's fiducial setup (their Table 3).
      C_IA: the `C_bar_1 * rho_crit` normalisation constant (see above);
        same convention and default (0.0134) as `ShearTracer`'s NLA model.
      loop_computer: object exposing `.compute(name) -> callable(matter_pk,
        ks, zs)` for each of the ten kernel names in `_ALL_KERNELS` - the
        "SpectrumComputer" for the one-loop terms. Required: pass
        `PBJTATTLoopComputer(perturbations)` (or build a `ShearTracer`
        directly with `ia_model="TATT",
        tatt_loop_computer=PBJTATTLoopComputer(...)`, which does this for
        you), or any other object with the same interface. No default -
        `TATTContribution` only ever reports kernel values it can stand
        behind as real physics.
    """

    def __init__(
        self,
        tracer: "ShearTracer",
        A1: float,
        A2: float,
        b_TA: float,
        loop_computer: object,
        eta1: float = 0.0,
        eta2: float = 0.0,
        z0: float = 0.62,
        C_IA: float = 0.0134,
    ) -> None:
        self._tracer = tracer
        self.A1 = A1
        self.A2 = A2
        self.b_TA = b_TA
        self.eta1 = eta1
        self.eta2 = eta2
        self.z0 = z0
        self.C_IA = C_IA
        self._loop_computer = loop_computer

    def compute_kernel(self, z):
        """Amplitude-free IA weighting kernel: n_i(z) * H(z)/c.

        Unlike NLA (where a single scalar C1(z) factor can be baked
        straight into the window because P_II = C1(z)**2 * P_dd(z,k) shares
        P_dd's k-shape), TATT's extra terms have k-shapes (the one-loop
        kernels) that differ from P_dd - so the *amplitude* weighting
        (C1/C1delta/C2) has to move into `get_effective_pk`'s P(k,z)
        assembly instead of living in this kernel.

        The H(z)/c factor is not part of the TATT amplitude functions
        themselves - it's the same "per unit z to per unit comoving
        distance" Jacobian `ShearTracer.get_window_IA`'s NLA implementation
        already folds into its own window (`factor = -Hz/c_0 * ...`).
        Leaving it out here isn't just a simplification: without it, this
        contribution's effective normalisation differs from NLA's by
        (Hz/c_0) per side. Keeping the Jacobian in the kernel (applied
        identically to every contribution pairing) rather than folding it
        into C1/C1delta/C2 separately keeps `get_effective_pk` matching the
        paper's equations exactly, unencumbered by this pipeline-specific
        normalisation detail.
        """
        h_over_c = self._tracer.perturbations.background.hubble_parameter(z) / _C_KM_S
        return self._tracer.dndz_shifted * h_over_c[None, :]

    def _C1(self, zs):
        omega_m0 = self._tracer.background.Omega_m(0.0)
        d = _growth_factor_1d(self._loop_computer.linear_perturbations, zs)
        return (
            -self.A1
            * self.C_IA
            * omega_m0
            * ((1 + zs) / (1 + self.z0)) ** self.eta1
            / d
        )

    def _C2(self, zs):
        omega_m0 = self._tracer.background.Omega_m(0.0)
        d = _growth_factor_1d(self._loop_computer.linear_perturbations, zs)
        return (
            5
            * self.A2
            * self.C_IA
            * omega_m0
            / d**2
            * ((1 + zs) / (1 + self.z0)) ** self.eta2
        )

    def _C1delta(self, zs):
        return self.b_TA * self._C1(zs)

    def _D4(self, zs):
        return _growth_factor_1d(self._loop_computer.linear_perturbations, zs) ** 4

    def get_spectrum_requests(self):
        return tuple(
            SpectrumRequest(name=name, compute=self._loop_computer.compute(name))
            for name in _ALL_KERNELS
        )

    def get_requirements_for_interaction(self, other):
        """Drop the II-only (EE and BB) kernels unless `other` is IA too.

        Mirrors `toy_cloelib.contributions.TATTModel.get_requirements_for_
        interaction`, which drops its analogous `pk_beta` term for the same
        reason: those terms only enter the II auto/cross correlation, never
        the matter-intrinsic GI cross-correlation - computing them for a
        GI-only analysis would be wasted one-loop-integral cost for terms
        that get multiplied into a total no GI computation ever reads.
        """
        if isinstance(other, IntrinsicAlignmentContribution):
            return self.get_spectrum_requests()
        return tuple(r for r in self.get_spectrum_requests() if r.name in _GI_KERNELS)

    def get_effective_pk(self, other, bank: SpectraBank):
        """P_II^EE(k,z) or P_deltaI(k,z), as appropriate.

        Bilinear in `self`'s and `other`'s own amplitude functions: reduces
        exactly to the II form when `other is self` (or another
        `TATTContribution` with identical parameters), and is the natural
        generalisation for a genuine cross-population II term otherwise.
        `other`'s C1/C1delta/C2 are used when it exposes them (i.e. it's
        also a `TATTContribution`); a plain `NLAContribution`
        (NLA) has no C1delta/C2 of its own, so it's treated as
        C1delta=C2=0 - the correct NLA limit, just cross-correlated against
        this contribution's full TATT terms rather than assuming both sides
        are identical.

        Returns `None` (defers to the plain matter Pk) only if this call is
        somehow reached with neither side being IA-like, which shouldn't
        happen given `get_requirements_for_interaction`.
        """
        zs = bank.zs
        d4 = self._D4(zs)[:, None]
        c1_self = self._C1(zs)
        matter_pk = bank.matter_pk

        def _kernel(name):
            return bank.get(
                SpectrumRequest(name=name, compute=self._loop_computer.compute(name))
            )[None, :]

        if isinstance(other, IntrinsicAlignmentContribution):
            c1_other = other._C1(zs) if hasattr(other, "_C1") else c1_self
            c1d_self = self._C1delta(zs)
            c1d_other = other._C1delta(zs) if hasattr(other, "_C1delta") else 0.0
            c2_self = self._C2(zs)
            c2_other = other._C2(zs) if hasattr(other, "_C2") else 0.0

            term_00e = _kernel("tatt_A_0_0E") + _kernel("tatt_C_0_0E")
            term_0e2 = _kernel("tatt_A_0_E2") + _kernel("tatt_B_0_E2")

            return (
                (c1_self * c1_other)[:, None] * matter_pk
                + (c1_self * c1d_other + c1_other * c1d_self)[:, None] * d4 * term_00e
                + (c1d_self * c1d_other)[:, None] * d4 * _kernel("tatt_A_0E_0E")
                + (c2_self * c2_other)[:, None] * d4 * _kernel("tatt_A_E2_E2")
                + (c1_self * c2_other + c1_other * c2_self)[:, None] * d4 * term_0e2
                + (c1d_self * c2_other + c1d_other * c2_self)[:, None]
                * d4
                * _kernel("tatt_D_0E_E2")
            )

        # Matter/density-intrinsic cross term.
        c1d_self = self._C1delta(zs)
        c2_self = self._C2(zs)
        term_00e = _kernel("tatt_A_0_0E") + _kernel("tatt_C_0_0E")
        term_0e2 = _kernel("tatt_A_0_E2") + _kernel("tatt_B_0_E2")
        return (
            c1_self[:, None] * matter_pk
            + c1d_self[:, None] * d4 * term_00e
            + c2_self[:, None] * d4 * term_0e2
        )


# ---------------------------------------------------------------------------
# TATT-M (Herle et al. 2026, arXiv:2601.15851): per-tomographic-bin
# IA amplitudes tied to halo mass and red-galaxy fraction, rather than
# TATTContribution's single global A1/A2/bTA. See TATTMContribution's
# docstring for why this needs three separate rank-1 Contributions instead
# of one bin-aware TATTContribution.
# ---------------------------------------------------------------------------

_TA_II_KERNELS = ("tatt_A_0_0E", "tatt_C_0_0E", "tatt_A_0_E2", "tatt_B_0_E2")
_TB_GI_KERNELS = ("tatt_A_0_0E", "tatt_C_0_0E")
_TB_II_KERNELS = _TB_GI_KERNELS + ("tatt_A_0E_0E", "tatt_D_0E_E2")
_TC_GI_KERNELS = ("tatt_A_0_E2", "tatt_B_0_E2")
_TC_II_KERNELS = _TC_GI_KERNELS + ("tatt_D_0E_E2", "tatt_A_E2_E2")


def _tatt_m_per_bin_amplitudes(
    nuisance_params, n_bins, alpha_M, beta_M, k1, k2, k3, log10_M0
):
    """Per-bin (A1, A2, bTA) -> (a1, a2, a1delta) amplitude arrays for
    TATT-M, shape `(n_bins,)` each.

    `n_bins` is a plain Python int (`ShearTracer.n_z_bins`), not a traced
    value, so this list comprehension over `range(n_bins)` is the same
    non-traced pattern `ShearTracer.__init__`'s own `m_bias`/`dz_shear_i`
    (this module) and `positions.py`'s `per_bin_case` already use - it runs
    once at Contribution-construction time, not inside any jitted call.

        A1^i  = alphaM * (Mh^i/M0)^betaM
        A2^i  = k1 * A1^i
        bTA^i = k2 * log10(Mh^i/M0) - k3

    `Mh^i`/`M0` (i.e. `log10_Mh_{i}`/`log10_M0`) are in Msun/h,
    - `Mh^i/M0` is a dimensionless ratio, so
    nothing here enforces that unit choice, but `log10_Mh_{i}` and
    `log10_M0` must be given in the *same* units as each other for the
    ratio (and hence `A1^i`/`bTA^i`) to mean what the paper's own fiducial
    values (e.g. `log10_M0=13.5`) assume.

    `a1 = f_r*A1`, `a2 = f_r*A2`, `a1delta = bTA*f_r*A1` (= `bTA*a1`, i.e.
    C1delta's own amplitude, since C1delta^i(z) = bTA^i * C1^i(z) and
    C1^i(z) = a1^i * shape(z) - see `TATTMContribution`'s docstring) are the
    three per-bin scalars `_TATTMTaContribution`/`_TATTMTbContribution`/
    `_TATTMTcContribution` bake into their own kernels.

    Reads `nuisance_params[f"log10_Mh_{i+1}"]` and
    `f"f_r_{i+1}"]` (1-indexed, matching this module's other per-bin
    nuisance params - `multiplicative_bias_1`, `dz_shear_1`, `width_shear_1`
    - rather than `positions.py`'s 0-indexed `b1_photo_bin0` convention).
    Required, no default: these are core physics inputs (halo mass, red
    fraction), not incidental systematics.
    """
    log10_mh = jnp.asarray(
        [nuisance_params[f"log10_Mh_{i + 1}"] for i in range(n_bins)]
    )
    f_r = jnp.asarray([nuisance_params[f"f_r_{i + 1}"] for i in range(n_bins)])
    delta = log10_mh - log10_M0
    A1 = alpha_M * 10.0 ** (delta * beta_M)
    A2 = k1 * A1
    b_ta = k2 * delta - k3
    a1 = f_r * A1
    a2 = f_r * A2
    a1delta = b_ta * a1
    return a1, a2, a1delta


def _tatt_m_kernel_grid(bank, loop_computer, name):
    """One named TATT one-loop kernel evaluated on `bank`'s grid, shape
    `(1, n_k)` - same convention `TATTContribution.get_effective_pk`'s own
    `_kernel` closure uses, factored out here since it's shared by all
    three TATT-M components below. `SpectraBank.get` memoizes by `name`
    (checks its cache before calling `compute`), so requesting the same
    name from more than one component never re-runs FAST-PT.
    """
    return bank.get(SpectrumRequest(name=name, compute=loop_computer.compute(name)))[
        None, :
    ]


def _tatt_m_d4(loop_computer, zs):
    return _growth_factor_1d(loop_computer.linear_perturbations, zs) ** 4


def _tatt_m_cached_term(bank, name, compute_term):
    """Memoize a derived TATT-M term grid (`D(z)**4` times one or more
    FAST-PT kernels) in `bank`, keyed by `name` - reuses `SpectraBank`'s
    existing per-name cache (the same mechanism the raw FAST-PT kernels
    already use via `_tatt_m_kernel_grid`) so repeated calls with the same
    `bank` - e.g. both `(Ta,Tb)` and `(Tb,Ta)` in
    `_compute_cl_generalized`'s Cartesian double loop ask for this exact
    term - recompute it only once, not once per visit.
    """

    def _compute(matter_pk, ks, zs_arg):
        del matter_pk, ks
        return compute_term(zs_arg)

    return bank.get(SpectrumRequest(name=name, compute=_compute))


def _tatt_m_term_00e(bank, loop_computer):
    def _term(zs):
        d4 = _tatt_m_d4(loop_computer, zs)[:, None]
        return d4 * (
            _tatt_m_kernel_grid(bank, loop_computer, "tatt_A_0_0E")
            + _tatt_m_kernel_grid(bank, loop_computer, "tatt_C_0_0E")
        )

    return _tatt_m_cached_term(bank, "tatt_m_term_00e", _term)


def _tatt_m_term_0e0e(bank, loop_computer):
    def _term(zs):
        d4 = _tatt_m_d4(loop_computer, zs)[:, None]
        return d4 * _tatt_m_kernel_grid(bank, loop_computer, "tatt_A_0E_0E")

    return _tatt_m_cached_term(bank, "tatt_m_term_0e0e", _term)


def _tatt_m_term_e2e2(bank, loop_computer):
    def _term(zs):
        d4 = _tatt_m_d4(loop_computer, zs)[:, None]
        return d4 * _tatt_m_kernel_grid(bank, loop_computer, "tatt_A_E2_E2")

    return _tatt_m_cached_term(bank, "tatt_m_term_e2e2", _term)


def _tatt_m_term_0e2(bank, loop_computer):
    def _term(zs):
        d4 = _tatt_m_d4(loop_computer, zs)[:, None]
        return d4 * (
            _tatt_m_kernel_grid(bank, loop_computer, "tatt_A_0_E2")
            + _tatt_m_kernel_grid(bank, loop_computer, "tatt_B_0_E2")
        )

    return _tatt_m_cached_term(bank, "tatt_m_term_0e2", _term)


def _tatt_m_term_d0ee2(bank, loop_computer):
    def _term(zs):
        d4 = _tatt_m_d4(loop_computer, zs)[:, None]
        return d4 * _tatt_m_kernel_grid(bank, loop_computer, "tatt_D_0E_E2")

    return _tatt_m_cached_term(bank, "tatt_m_term_d0ee2", _term)


def _tatt_m_effective_pk_for_bin_pair(tatt_m, bank, bin_i, bin_j):
    """`P_II^{bin_i,bin_j}(k,z)` and `P_deltaI^{bin_i}(k,z)` for one
    specific tomographic bin pair, from a `TATTMContribution` coordinator.

    `AngularTwoPoint._compute_cl_generalized` never materializes this
    directly - it sums Ta/Tb/Tc pairwise once, letting each bin's own
    window function (not a per-bin-pair Pk) carry the bin dependence (see
    `TATTMContribution`'s docstring). This reconstructs one specific
    bin pair's grid explicitly, from the same per-bin amplitude arrays
    (`tatt_m.ta/.tb/.tc.amplitude`) and shared shape grids `get_effective_pk`
    uses internally - for `ShearTracer.get_ia_effective_spectra`'s
    inspection/plotting use case only.

    `bin_i`/`bin_j` are 1-indexed, matching every other per-bin convention
    in this module. `deltaI` depends only on `bin_i` (the density-IA GI
    term involves one IA bin, not a pair); `bin_j` is still required
    alongside it so callers always name a definite bin pair for `II`.

    Note `tatt_m.ta/.tb/.tc.amplitude` are only the bin-scalar *prefactors*
    (`a1^i = f_r^i*A1^i`, etc.) - the actual C1^i(z)/C1delta^i(z)/C2^i(z)
    also need each component's shared `_shape(z)` function (the D(z)
    dependence): `C1^i(z) = amplitude[i] * _shape(z)`. Multiplying the
    bare `amplitude` scalars straight into the bilinear formula below
    (omitting `_shape(z)`) would silently drop that z-dependence entirely -
    caught by `test_get_ia_effective_spectra_bin_pair_matches_collapsed_
    tatt_z` in `tests/test_tatt_m.py`, which compares against
    `TATTContribution`'s already-validated formula directly.
    """
    i, j = bin_i - 1, bin_j - 1
    ta, tb, tc = tatt_m.ta, tatt_m.tb, tatt_m.tc
    loop_computer = ta._loop_computer  # shared across ta/tb/tc
    zs = bank.zs

    shape1 = ta._shape(zs)  # == tb._shape(zs): C1/C1delta share one shape
    shape2 = tc._shape(zs)
    c1_i, c1_j = ta.amplitude[i] * shape1, ta.amplitude[j] * shape1
    c1d_i, c1d_j = tb.amplitude[i] * shape1, tb.amplitude[j] * shape1
    c2_i, c2_j = tc.amplitude[i] * shape2, tc.amplitude[j] * shape2

    pdd = bank.matter_pk
    term_00e = _tatt_m_term_00e(bank, loop_computer)
    term_0e0e = _tatt_m_term_0e0e(bank, loop_computer)
    term_e2e2 = _tatt_m_term_e2e2(bank, loop_computer)
    term_0e2 = _tatt_m_term_0e2(bank, loop_computer)
    term_d0ee2 = _tatt_m_term_d0ee2(bank, loop_computer)

    p_ii = (
        (c1_i * c1_j)[:, None] * pdd
        + (c1_i * c1d_j + c1_j * c1d_i)[:, None] * term_00e
        + (c1d_i * c1d_j)[:, None] * term_0e0e
        + (c2_i * c2_j)[:, None] * term_e2e2
        + (c1_i * c2_j + c1_j * c2_i)[:, None] * term_0e2
        + (c1d_i * c2_j + c1d_j * c2_i)[:, None] * term_d0ee2
    )
    p_delta_i = (
        c1_i[:, None] * pdd + c1d_i[:, None] * term_00e + c2_i[:, None] * term_0e2
    )
    return {"II": p_ii, "deltaI": p_delta_i}


def _tatt_m_requests(loop_computer, names):
    return tuple(
        SpectrumRequest(name=n, compute=loop_computer.compute(n)) for n in names
    )


def _tatt_m_cross_model_error(self_name, other):
    return ValueError(
        f"{self_name} cannot be paired with {type(other).__name__}: "
        "TATT-M does not support cross-correlating against a different "
        "intrinsic-alignment model (a legacy TATTContribution/"
        "NLAContribution, or a TATT-M component from a different "
        "coordinator) in this version - build both tracers with the same "
        "ia_model. (Note: if the *other* side is a legacy TATTContribution "
        "and it is checked first by spectrum_engine.get_effective_pk, its "
        "own duck-typed `hasattr(other, '_C1')` fallback silently uses its "
        "own amplitude instead of raising - a pre-existing quirk this class "
        "does not patch, since existing TATT/NLA behavior must not change.)"
    )


class _TATTMComponentContribution(IntrinsicAlignmentContribution):
    """Shared base for TATT-M's three rank-1 components (`_TATTMTaContribution`
    /`_TATTMTbContribution`/`_TATTMTcContribution`) - see
    `TATTMContribution`'s docstring for the physics and the architectural
    reason this decomposition exists. Each carries a genuine per-bin
    amplitude array (shape `(n_bins,)`) and a shared, bin-independent
    z-shape function, factoring C1^i(z)/C1delta^i(z)/C2^i(z) as
    `amplitude^i * shape(z)` - baked directly into `compute_kernel` (NLA's
    own convention), not into `get_effective_pk`.
    """

    def __init__(self, tracer: "ShearTracer", amplitude, loop_computer, C_IA: float):
        self._tracer = tracer
        self.amplitude = amplitude
        self._loop_computer = loop_computer
        self.C_IA = C_IA

    def _shape(self, zs):
        """`s_1(z) = -C_IA*Omega_m0/D(z)` - the shape shared by `C1^i(z)`
        and `C1delta^i(z)` (since `C1delta^i(z) = bTA^i * C1^i(z)` reuses
        `C1^i(z)`'s own shape exactly). The default here, inherited
        unchanged by `_TATTMTaContribution`/`_TATTMTbContribution` -
        `_TATTMTcContribution` overrides this with its own `s_2(z)`, since
        `C2^i(z)` has a different D(z) power.
        """
        d = _growth_factor_1d(self._loop_computer.linear_perturbations, zs)
        return -self.C_IA * self._tracer.background.Omega_m(0.0) / d

    def compute_kernel(self, z):
        h_over_c = self._tracer.perturbations.background.hubble_parameter(z) / _C_KM_S
        shape = self._shape(z)
        return (
            (self.amplitude[:, None] * shape[None, :])
            * self._tracer.dndz_shifted
            * h_over_c[None, :]
        )


class _TATTMTaContribution(_TATTMComponentContribution):
    """C1-weighted rank-1 term: per-bin amplitude `a1^i = f_r^i*A1^i` times
    the shape shared by every bin, `-C_IA*Omega_m0/D(z)`. Paired against
    itself -> Pdd (the II C1*C1 term); against `_TATTMTbContribution`/
    `_TATTMTcContribution` -> the matching one-loop shape grid; against
    anything non-IA (lensing/density) -> Pdd (the GI `C1^i(z)*Pdd` term).
    Uses the base class's default `_shape` (`s_1(z)`) unchanged.

    The two Pdd cases return `None`, not `bank.matter_pk` directly: `None`
    is `spectrum_engine.get_effective_pk`'s own convention for "use the
    plain matter Pk" (see its docstring), and the two conventions are
    physically identical here - but `_compute_cl_generalized` interpolates
    a `None` pk_eff through the cheaper always-positive `Pkl_interp_vmap`,
    while any *other* non-`None` return value (even one numerically equal
    to `bank.matter_pk`) goes through the pricier sign-aware
    `Pkl_interp_signed_vmap` (needed for TATT-M's genuinely signed terms,
    but wasted work here since Pdd never changes sign).
    """

    def get_requirements_for_interaction(self, other):
        if isinstance(other, IntrinsicAlignmentContribution):
            return _tatt_m_requests(self._loop_computer, _TA_II_KERNELS)
        return ()

    def get_effective_pk(self, other, bank: SpectraBank):
        if isinstance(other, _TATTMTaContribution):
            return None
        if isinstance(other, _TATTMTbContribution):
            return _tatt_m_term_00e(bank, self._loop_computer)
        if isinstance(other, _TATTMTcContribution):
            return _tatt_m_term_0e2(bank, self._loop_computer)
        if isinstance(other, IntrinsicAlignmentContribution):
            raise _tatt_m_cross_model_error("_TATTMTaContribution", other)
        return None


class _TATTMTbContribution(_TATTMComponentContribution):
    """C1delta-weighted rank-1 term: per-bin amplitude `a1delta^i =
    bTA^i*a1^i`, same shared shape as `_TATTMTaContribution` (C1delta^i(z) =
    bTA^i * C1^i(z)) - uses the base class's default `_shape` unchanged."""

    def get_requirements_for_interaction(self, other):
        names = (
            _TB_II_KERNELS
            if isinstance(other, IntrinsicAlignmentContribution)
            else _TB_GI_KERNELS
        )
        return _tatt_m_requests(self._loop_computer, names)

    def get_effective_pk(self, other, bank: SpectraBank):
        if isinstance(other, _TATTMTaContribution):
            return _tatt_m_term_00e(bank, self._loop_computer)
        if isinstance(other, _TATTMTbContribution):
            return _tatt_m_term_0e0e(bank, self._loop_computer)
        if isinstance(other, _TATTMTcContribution):
            return _tatt_m_term_d0ee2(bank, self._loop_computer)
        if isinstance(other, IntrinsicAlignmentContribution):
            raise _tatt_m_cross_model_error("_TATTMTbContribution", other)
        return _tatt_m_term_00e(bank, self._loop_computer)


class _TATTMTcContribution(_TATTMComponentContribution):
    """C2-weighted rank-1 term: per-bin amplitude `a2^i = f_r^i*A2^i`,
    shared shape `5*C_IA*Omega_m0/D(z)**2`."""

    def _shape(self, zs):
        d = _growth_factor_1d(self._loop_computer.linear_perturbations, zs)
        return 5 * self.C_IA * self._tracer.background.Omega_m(0.0) / d**2

    def get_requirements_for_interaction(self, other):
        names = (
            _TC_II_KERNELS
            if isinstance(other, IntrinsicAlignmentContribution)
            else _TC_GI_KERNELS
        )
        return _tatt_m_requests(self._loop_computer, names)

    def get_effective_pk(self, other, bank: SpectraBank):
        if isinstance(other, _TATTMTaContribution):
            return _tatt_m_term_0e2(bank, self._loop_computer)
        if isinstance(other, _TATTMTbContribution):
            return _tatt_m_term_d0ee2(bank, self._loop_computer)
        if isinstance(other, _TATTMTcContribution):
            return _tatt_m_term_e2e2(bank, self._loop_computer)
        if isinstance(other, IntrinsicAlignmentContribution):
            raise _tatt_m_cross_model_error("_TATTMTcContribution", other)
        return _tatt_m_term_0e2(bank, self._loop_computer)


class TATTMContribution(IntrinsicAlignmentContribution):
    r"""TATT-M intrinsic-alignment contribution (Herle et al. 2026,
    arXiv:2601.15851): ties TATT's C1/C2/C1delta amplitudes to per-
    tomographic-bin halo mass and red-galaxy fraction, rather than
    `TATTContribution`'s single global A1/A2/bTA.

        A1^i  = alphaM * (Mh^i/M0)^betaM
        A2^i  = k1 * A1^i
        bTA^i = k2 * log10(Mh^i/M0) - k3

        C1^i(z)      = -f_r^i * A1^i * C_IA * Omega_m0 / D(z)
        C2^i(z)      =  5 * f_r^i * A2^i * C_IA * Omega_m0 / D(z)**2
        C1delta^i(z) =  bTA^i * C1^i(z)

    Unlike `TATTContribution`, there is no `(1+z)^eta` redshift evolution
    here - the per-bin halo mass `Mh^i` (and fixed red fraction `f_r^i`) is
    the only source of amplitude variation across bins.

    Implemented as three rank-1 "component" Contributions
    (`get_components()` below) rather than one monolithic bin-aware object,
    because the Limber engine (`AngularTwoPoint._compute_cl_generalized`/
    `Cl_integration`) has no bin-*pair* axis on its effective power spectrum
    grid - only on the two window functions it integrates against
    (`Cl_integration`'s `einsum("iz,jz,lz,...->lij", ...)`: the bin indices
    `i,j` come only from the window terms, never from `Pkl`). A single
    per-bin-amplitude object cannot express bin-pair `(i,j)` cross terms
    (e.g. `C1^i(z)*C1^j(z)`) through that grid - `get_effective_pk` returns
    one grid shared by every bin pair, today correct only because
    `TATTContribution`'s amplitude is bin-independent.

    Since `alphaM`/`betaM`/`k1`/`k2`/`k3` are global (not per-bin),
    `C1^i(z)`/`C1delta^i(z)`/`C2^i(z)` each factor exactly as
    `(per-bin scalar amplitude) * (shape shared by every bin)`. Splitting
    the contribution into three rank-1 terms - each with its genuine
    per-bin amplitude baked into its own kernel (NLA's own convention - see
    `NLAContribution`) and paired via the *same* bin-independent one-loop
    shape grids `TATTContribution.get_effective_pk` already computes - lets
    the existing Cartesian double loop over `contributions1 x contributions2`
    in `_compute_cl_generalized` reconstruct the exact per-bin-pair
    bilinear formula, with zero changes to `spectrum_engine.py`/
    `angular_two_point.py`.

    Cross-correlating a TATT-M tracer against a tracer using the legacy
    scalar `TATTContribution`/`NLAContribution` IA model is not supported
    in this version: pairing one of this class's
    components against any *other* `IntrinsicAlignmentContribution` type
    raises `ValueError` (see `_tatt_m_cross_model_error`) - not needed for
    the Euclid DR1 analysis this model was built for.

    Args:
      tracer: the owning `ShearTracer`.
      alpha_M: normalisation of the halo-mass-A1 relation (global; read from
        `nuisance_params["alphaM"]` by `ShearTracer._build_ia_contribution`).
      beta_M: power-law index of the halo-mass-A1 relation (global;
        `nuisance_params["betaM"]`).
      loop_computer: shared one-loop kernel backend for all three
        components - same contract as `TATTContribution`'s own
        `loop_computer` (e.g. `PBJTATTLoopComputer(perturbations)`). Shared
        (not one per component) so `PBJTATTLoopComputer`'s own FAST-PT
        cache and linear-perturbations source stay consistent across all
        three.
      k1: A2/A1 ratio (global; default 0.073, Herle et al.'s simple
        iterative-inertia-tensor value - `nuisance_params.get("k1", 0.073)`
        - but sampled the same way as any other nuisance parameter if
        freed).
      k2: slope of bTA vs `log10(Mh/M0)` (global; default 0.268, Herle et
        al. - `nuisance_params.get("k2", 0.268)`).
      k3: intercept of bTA vs `log10(Mh/M0)` (global; default -0.038,
        Herle et al. - `nuisance_params.get("k3", -0.038)`).
      log10_M0: log10 of the halo-mass pivot in Msun/h (default 13.5;
        `nuisance_params.get("log10_M0", 13.5)`) - same units as
        `log10_Mh_{i}`.
      C_IA: same `C_bar_1*rho_crit` normalisation convention as
        `NLAContribution`/`TATTContribution` (default 0.0134).

    Validation status: checked internally against the TATT-z collapse
    limit - with every tomographic bin given the same halo mass/red
    fraction, this reproduces `TATTContribution`'s own (already-
    validated-against-CosmoSIS) P_II/P_deltaI formula exactly, and its
    angular power spectrum (Cl) closely (see `tests/test_tatt_m.py`).
    Not yet cross-checked against an external code: no CosmoSIS (or
    other) implementation of TATT-M exists to validate the halo-mass-
    dependent amplitude itself against.
    """

    def __init__(
        self,
        tracer: "ShearTracer",
        alpha_M: float,
        beta_M: float,
        loop_computer: object,
        k1: float = 0.073,
        k2: float = 0.268,
        k3: float = -0.038,
        log10_M0: float = 13.5,
        C_IA: float = 0.0134,
    ) -> None:
        self._tracer = tracer
        self.alpha_M = alpha_M
        self.beta_M = beta_M
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.log10_M0 = log10_M0
        self.C_IA = C_IA
        a1, a2, a1delta = _tatt_m_per_bin_amplitudes(
            tracer.nuisance_params,
            tracer.n_z_bins,
            alpha_M,
            beta_M,
            k1,
            k2,
            k3,
            log10_M0,
        )
        self.ta = _TATTMTaContribution(tracer, a1, loop_computer, C_IA)
        self.tb = _TATTMTbContribution(tracer, a1delta, loop_computer, C_IA)
        self.tc = _TATTMTcContribution(tracer, a2, loop_computer, C_IA)

    def get_components(self):
        """The three rank-1 Contributions `ShearTracer.get_contributions()`
        flattens into the tracer's own top-level contribution tuple."""
        return (self.ta, self.tb, self.tc)

    def compute_kernel(self, z):
        """Sum of the three components' kernels - convenience for direct
        `tracer.ia.compute_kernel(z)` callers (matches every other IA
        model's `compute_kernel` contract). Never what
        `_compute_cl_generalized` actually calls once `get_components()`
        has flattened this contribution away - see
        `ShearTracer.get_contributions`.
        """
        return (
            self.ta.compute_kernel(z)
            + self.tb.compute_kernel(z)
            + self.tc.compute_kernel(z)
        )


class ShearTracer:
    """Class for the kernel for Cosmic Shear."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        ia_model: "str | Contribution | None" = None,
        tatt_loop_computer: Optional[object] = None,
    ):
        r"""
        Initialize the class instance.

        Args:
          perturbations (object): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
          ia_model (str | Contribution | None): Which intrinsic-alignment
            contribution to use. `None` (default) means no intrinsic-alignment
            contribution at all - `get_window()`/`get_Cl` use only the lensing
            term. `"NLA"` builds `NLAContribution` (`get_window_IA`, reading
            `nuisance_params["AIA"/"CIA"/"EtaIA"]`). `"TATT"` builds a
            `TATTContribution` from `nuisance_params["AIA"]` (=A1),
            `nuisance_params["A2IA"]`, `nuisance_params["bTA"]`, and optionally
            `nuisance_params["Eta2IA"]`/`["z0IA"]` (default 0.0/0.62).
            `"TATT-M"` builds a `TATTMContribution` (Herle et al. 2026,
            arXiv:2601.15851 - per-bin IA amplitude tied to halo mass/red
            fraction) from `nuisance_params["alphaM"]`, `["betaM"]`,
            optionally `.get("k1", 0.073)`/`.get("k2", 0.268)`/
            `.get("k3", -0.038)`/`.get("log10_M0", 13.5)` (Herle et al.'s
            fiducial values, freeable like any other nuisance parameter),
            and per-bin `nuisance_params[f"log10_Mh_{i+1}"]`/
            `[f"f_r_{i+1}"]` for every tomographic bin - requires
            `tatt_loop_computer` like `"TATT"` does. Advanced use: pass a
            `Contribution` instance directly to use any other model.
          tatt_loop_computer: required when `ia_model="TATT"` or `"TATT-M"`
            (raises `ValueError` if omitted) - the one-loop kernel backend
            (see `TATTContribution`'s docstring). Pass
            `PBJTATTLoopComputer(perturbations)` for real FAST-PT-computed
            kernels, or any other object with the same `.compute(name)`
            interface. Passing this with any other `ia_model` raises
            `ValueError`.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        self.nuisance_params = nuisance_params
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 1
        # Set multiplicative bias (m_bias)
        self.m_bias = [
            self.nuisance_params[f"multiplicative_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]
        self.dz_shear_i = [
            self.nuisance_params[f"dz_shear_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_shear_i = [
            self.nuisance_params[f"width_shear_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.n_z_bins = dndz.shape[0]
        self.dndz = dndz
        # Correct dndz for width_shear
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_shear_i)
        # Correct dndz_stretched for dz_shear
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_shear_i)

        self.lensing = LensingContribution(self)
        self.ia = self._build_ia_contribution(
            ia_model, nuisance_params, tatt_loop_computer
        )

    def get_contributions(self):
        """Return this tracer's window as its separable Contribution terms.

        Returns:
          contributions (tuple): `(self.lensing,)` if `ia_model=None` (the
            default - no intrinsic-alignment contribution). If `self.ia`
            defines the optional `get_components()` hook (currently only
            `TATTMContribution` - see its docstring for why), its
            components are flattened in directly:
            `(self.lensing,) + self.ia.get_components()`. Otherwise (`"NLA"`,
            `"TATT"`, or a custom `Contribution`): `(self.lensing, self.ia)`.
        """
        if self.ia is None:
            return (self.lensing,)
        get_components = getattr(self.ia, "get_components", None)
        if get_components is not None:
            return (self.lensing,) + tuple(get_components())
        return (self.lensing, self.ia)

    def get_ia_effective_spectra(
        self,
        ks: Optional[np.ndarray] = None,
        zs: Optional[np.ndarray] = None,
        bin_i: Optional[int] = None,
        bin_j: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """The IA model's own effective power spectra - `P_II(k,z)` and
        `P_deltaI(k,z)` - the same grids `AngularTwoPoint._compute_cl_
        generalized` integrates internally, exposed directly for
        inspection/plotting/validation (e.g. against an external code's own
        tabulated TATT spectra) without building a `SpectraBank` by hand.

        Raises `ValueError` for `ia_model=None` (no IA contribution) or
        `ia_model="NLA"` (whose P_II/P_deltaI share the matter Pk's
        k-shape by construction: the amplitude is applied directly inside
        `compute_kernel`, not exposed as a separate grid) - neither has
        any effective-Pk grid to return, with or without `bin_i`/`bin_j`.

        For `ia_model="TATT"` (bin-independent amplitude): `bin_i`/`bin_j`
        must be left `None` (raises otherwise - this model has one grid,
        shared by every bin pair, so naming a pair doesn't apply).

        For `ia_model="TATT-M"` (per-bin amplitude): `bin_i`/`bin_j`
        (1-indexed) are required (raises if either is omitted) - there is
        no single grid once amplitudes vary per bin (see
        `TATTMContribution`'s docstring), so you must say which
        tomographic bin pair's `P_II^{bin_i,bin_j}(k,z)`/
        `P_deltaI^{bin_i}(k,z)` to return.

        Args:
          ks: wavenumber grid to evaluate on. Defaults to
            `self.perturbations.k`.
          zs: redshift grid to evaluate on. Defaults to
            `self.perturbations.z`.
          bin_i: first tomographic bin (1-indexed). Required for
            `ia_model="TATT-M"`; must be omitted otherwise.
          bin_j: second tomographic bin (1-indexed, may equal `bin_i` for
            the auto-correlation). Required for `ia_model="TATT-M"`; must
            be omitted otherwise.

        Returns:
          dict: `{"II": P_II(k,z), "deltaI": P_deltaI(k,z)}`, each shape
            `(len(zs), len(ks))`.
        """
        get_components = getattr(self.ia, "get_components", None)
        has_single_grid = (
            self.ia is not None
            and getattr(self.ia, "get_effective_pk", None) is not None
        )

        if get_components is None and not has_single_grid:
            ia_name = type(self.ia).__name__ if self.ia is not None else None
            raise ValueError(
                "get_ia_effective_spectra() needs an IA model that defines "
                "an effective power spectrum (currently ia_model='TATT' or "
                f"'TATT-M'); this tracer's ia is {ia_name!r}, which has "
                "none - either no IA contribution at all, or NLA "
                "(amplitude applied directly inside compute_kernel, not "
                "exposed as a separate grid)."
            )

        if get_components is not None:
            # Per-bin IA model (currently only TATTMContribution): no
            # single grid exists, so bin_i/bin_j are mandatory.
            if bin_i is None or bin_j is None:
                raise ValueError(
                    f"{type(self.ia).__name__} varies its IA amplitude per "
                    "tomographic bin, so get_ia_effective_spectra() needs "
                    "bin_i and bin_j (1-indexed) to know which bin pair's "
                    "P_II(k,z)/P_deltaI(k,z) to return "
                    f"(got bin_i={bin_i!r}, bin_j={bin_j!r})."
                )
            ks = getattr(self.perturbations, "k", None) if ks is None else ks
            zs = getattr(self.perturbations, "z", None) if zs is None else zs
            matter_pk = self.perturbations.matter_power_spectrum(zs, ks)
            bank = SpectraBank(matter_pk, ks, zs)
            return _tatt_m_effective_pk_for_bin_pair(self.ia, bank, bin_i, bin_j)

        # Single, bin-independent grid (currently only TATTContribution).
        if bin_i is not None or bin_j is not None:
            raise ValueError(
                f"{type(self.ia).__name__} doesn't vary its IA amplitude "
                "per bin, so get_ia_effective_spectra() doesn't take "
                f"bin_i/bin_j (got bin_i={bin_i!r}, bin_j={bin_j!r})."
            )
        ks = getattr(self.perturbations, "k", None) if ks is None else ks
        zs = getattr(self.perturbations, "z", None) if zs is None else zs
        matter_pk = self.perturbations.matter_power_spectrum(zs, ks)
        return {
            "II": compute_effective_pk(self.ia, self.ia, matter_pk, ks, zs),
            "deltaI": compute_effective_pk(self.ia, self.lensing, matter_pk, ks, zs),
        }

    def _build_ia_contribution(self, ia_model, nuisance_params, tatt_loop_computer):
        """Resolve the `ia_model` constructor argument into a Contribution.

        `None` (default) means no intrinsic-alignment contribution at all;
        `"NLA"` builds `NLAContribution`; `"TATT"` builds a
        `TATTContribution`, which requires `tatt_loop_computer`
        (no illustrative default - see `TATTContribution`'s docstring);
        `"TATT-M"` builds a `TATTMContribution` (also requires
        `tatt_loop_computer`); anything else must already be a
        `Contribution` instance, used as-is. `tatt_loop_computer` is only
        meaningful for `"TATT"`/`"TATT-M"`.
        """
        if ia_model not in ("TATT", "TATT-M") and tatt_loop_computer is not None:
            raise ValueError(
                "tatt_loop_computer is only used when ia_model='TATT' or "
                f"'TATT-M' (got ia_model={ia_model!r})."
            )
        if ia_model is None:
            return None
        if ia_model == "NLA":
            return NLAContribution(self)
        if ia_model == "TATT":
            if tatt_loop_computer is None:
                raise ValueError(
                    "ia_model='TATT' requires tatt_loop_computer (e.g. "
                    "PBJTATTLoopComputer(perturbations), from "
                    "cloelib.observables.photo.shear) - TATTContribution "
                    "has no illustrative default."
                )
            return TATTContribution(
                self,
                A1=nuisance_params["AIA"],
                A2=nuisance_params["A2IA"],
                b_TA=nuisance_params["bTA"],
                loop_computer=tatt_loop_computer,
                eta1=nuisance_params.get("EtaIA", 0.0),
                eta2=nuisance_params.get("Eta2IA", 0.0),
                z0=nuisance_params.get("z0IA", 0.62),
                C_IA=nuisance_params.get("CIA", 0.0134),
            )
        if ia_model == "TATT-M":
            if tatt_loop_computer is None:
                raise ValueError(
                    "ia_model='TATT-M' requires tatt_loop_computer (e.g. "
                    "PBJTATTLoopComputer(perturbations), from "
                    "cloelib.observables.photo.shear) - TATTMContribution "
                    "has no illustrative default, same as ia_model='TATT'."
                )
            return TATTMContribution(
                self,
                alpha_M=nuisance_params["alphaM"],
                beta_M=nuisance_params["betaM"],
                loop_computer=tatt_loop_computer,
                k1=nuisance_params.get("k1", 0.073),
                k2=nuisance_params.get("k2", 0.268),
                k3=nuisance_params.get("k3", -0.038),
                log10_M0=nuisance_params.get("log10_M0", 13.5),
                C_IA=nuisance_params.get("CIA", 0.0134),
            )
        if isinstance(ia_model, str):
            raise ValueError(
                f"Unknown ia_model {ia_model!r}: expected 'NLA', 'TATT', "
                "'TATT-M', or a Contribution instance."
            )
        return ia_model

    def get_window_IA(self, z):
        r"""Window integrand.

        Calculates IA window

        Args:
          z (float): Redshift at which kernel is being evaluated

        Returns:
          window_IA (np.ndarray):
        """
        Omega_m0 = self.background.Omega_m(0.0)
        Hz = self.perturbations.background.hubble_parameter(z)
        # `growth_factor` is backend-dependent: some backends (e.g. CAMB)
        # return D(z, k) with an explicit k-axis; others (e.g. the JAX
        # backends) ignore `ks` entirely and return a scale-independent
        # D(z), and don't set a `.k` attribute at all. NLA treats growth as
        # ~scale-independent, so when a k-axis is present we pick a single
        # representative column (unchanged from the historical behavior);
        # when it isn't, the backend's own 1D D(z) is used directly.
        # TODO discuss whether we want growth factor to output a 1D or a 2D array
        ks = getattr(self.perturbations, "k", None)
        Dz_raw = self.perturbations.growth_factor(z, ks)
        Dz = Dz_raw[:, 1] if getattr(Dz_raw, "ndim", 1) == 2 else Dz_raw
        A_IA = self.nuisance_params["AIA"]
        C_IA = self.nuisance_params["CIA"]
        Eta_IA = self.nuisance_params["EtaIA"]
        factor = -Hz / c_0 * A_IA * C_IA * Omega_m0 * (1 + z) ** Eta_IA / Dz
        return np.einsum("ij, j->ij", self.dndz_shifted, factor)

    def get_lensing_efficiency_bin(self, z, bin_idx):
        """Compute the lensing efficiency in a redshift bin."""
        interpolator = interpax.Akima1DInterpolator(
            self.z, self.dndz_shifted[bin_idx, :]
        )
        x = np.linspace(0.0, 4, 200)
        y = self.background.comoving_distance(x)
        rx_interp = interpax.Akima1DInterpolator(x, y)
        f1 = jax.jit(lambda x: interpolator(x))
        f2 = jax.jit(lambda x: interpolator(x) / rx_interp(x))
        integral_1 = simps(f1, z, 3.0)
        integral_2 = simps(f2, z, 3.0)
        efficiency = integral_1 - integral_2 * self.background.comoving_distance(z)
        return efficiency

    def get_lensing_efficiency(self, z):
        r"""
        Compute the lensing efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Args:
          z (np.ndarray): 1D array of redshift values (must be evenly spaced). Used to compute comoving distances
            and define integration domain.

        Returns:
          (np.ndarray): 2D array of shape (N_bins, len(z)) representing the lensing efficiency kernel W(z)
            for each redshift bin over the evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz` is expected to have shape (N_bins, len(z)) and be normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_lensing(self, z):
        r"""Weak Lensing shear kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        `np.trapz` on the array along one axis.

        $$
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\
        $$

        Args:
          z (numpy.ndarray): Redshift at which weight is evaluated (`float` type).

        Returns:
          (numpy.ndarray): 1-D Numpy array of shear kernel values for specified bin
            at specified scale for the redshifts defined in z
        """
        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_lensing_efficiency(z)
        return np.einsum("ij, j->ij", efficiency, factor)

    def get_window(self, z):
        r"""Compute the Window.

        Computes general window given the selected tracer

        Parameters:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        total_window = sum(c.compute_kernel(z) for c in self.get_contributions())
        # Apply multiplicative bias
        total_window *= 1 + np.array(self.m_bias)[:, None]
        return total_window
