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
  A1/A2/bTA. Its per-bin amplitudes reach the Limber engine through
  `get_pk_terms` (`spectrum_engine.PkTerm`).

Pass `loop_computer=PBJTATTLoopComputer(perturbations)` directly to
`ShearTracer(..., ia_model="TATT"` (or `"TATT-M"`) `, tatt_loop_computer=
...)` - one constructor call, not build-then-replace `.ia`.

See `CONTRIBUTION_ARCHITECTURE.md` for the design this all
follows.
"""

from typing import Any, Callable, Dict, Optional, Protocol

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.geometry import cos_K, transverse_distance
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import (
    cached_stacked_simpson,
    check_nonzero_redshifts,
    simps,
)
from cloelib.auxiliary.systematics import shift_dndz_jax, stretch_dndz_jax
from cloelib.observables.photo.contributions import (
    IntrinsicAlignmentContribution,
    Contribution,
    contribution_field,
    unit_sigma,
)
from cloelib.observables.photo.spectrum_engine import (
    PkTerm,
    SpectraBank,
    SpectrumRequest,
    compute_effective_pk,
)

# General imports
import jax.numpy as np
import jax
import jax.numpy as jnp
import interpax
import numpy as _numpy
from scipy import interpolate as _scipy_interpolate


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


# Same convention: SPEED_OF_LIGHT is in m/s.
_C_KM_S = SPEED_OF_LIGHT / 1000


class LensingContribution:
    """Weak-lensing shear kernel term of `ShearTracer.get_window()`."""

    traces_potential = True

    def __init__(self, tracer: "ShearTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        """Return the uncalibrated lensing kernel, shape ``(n_bin, n_z)``."""
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
        """Return the uncalibrated NLA kernel, shape ``(n_bin, n_z)``."""
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


class TATTLoopComputer(Protocol):
    """Interface `TATTContribution` needs from its one-loop kernel backend.

    `PBJTATTLoopComputer` is the production implementation; any other
    object with these two members can be passed as `tatt_loop_computer`.
    """

    @property
    def linear_perturbations(self) -> Perturbations:
        """Linear perturbations used for the D(z) factors in C1/C2/D**4."""
        ...

    def compute(self, name: str) -> Callable[..., jnp.ndarray]:
        """Return a `SpectrumRequest.compute` callable for kernel `name`."""
        ...


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
        # `growth_factor`): most backends keep an explicit (1, n_k) z-axis,
        # but not all are guaranteed to. `reshape(-1)` normalizes either to
        # the flat (n_k,) FAST-PT needs.
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
        """Return the FAST-PT callable for one named TATT kernel.

        ``name`` must be present in ``_ALL_KERNELS``. The returned
        ``SpectrumRequest.compute(matter_pk, ks, zs)`` ignores ``matter_pk``
        and ``zs`` because the cached kernel depends only on k, and returns
        shape ``(len(ks),)``.
        """

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
        loop_computer: TATTLoopComputer,
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
        """Return requests for the ten TATT kernels.

        ``get_requirements_for_interaction`` retains only the GI kernels when
        paired with a non-IA contribution; an IA-IA pairing needs all ten.
        """
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

        The `P_dd(z,k)` terms use the base spectrum of the fields the two
        sides trace (`bank.base`): `P_mm` against IA or lensing, the cb x
        matter cross spectrum against the galaxy-bias term of a
        `PositionsTracer(use_Pcb=True)`.

        Returns `None` (defers to the plain matter Pk) only if this call is
        somehow reached with neither side being IA-like, which shouldn't
        happen given `get_requirements_for_interaction`.
        """
        zs = bank.zs
        d4 = self._D4(zs)[:, None]
        c1_self = self._C1(zs)
        matter_pk = bank.base(contribution_field(self), contribution_field(other))

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
# TATT-M (Herle et al. 2026, arXiv:2601.15851): TATT with per-tomographic-bin
# amplitudes set by halo mass and red-galaxy fraction.
# ---------------------------------------------------------------------------

# One-loop kernels TATT-M reads: the E-mode subset of `_ALL_KERNELS`.
_TATT_M_KERNELS = _EE_ONLY_KERNELS + _SHARED_KERNELS + ("tatt_D_0E_E2",)

# Shared one-loop shapes of the bilinear TATT formula (see `TATTContribution`),
# each `D(z)**4` times the sum of the named kernels.
_TATT_M_TERMS = {
    "00e": ("tatt_A_0_0E", "tatt_C_0_0E"),
    "0e2": ("tatt_A_0_E2", "tatt_B_0_E2"),
    "0e0e": ("tatt_A_0E_0E",),
    "e2e2": ("tatt_A_E2_E2",),
    "d0ee2": ("tatt_D_0E_E2",),
}


def _tatt_m_per_bin_amplitudes(
    nuisance_params, n_bins, alpha_M, beta_M, k1, k2, k3, log10_M0
):
    """Per-bin amplitudes `(a1, a2, a1delta)`, each shape `(n_bins,)`.

        A1^i  = alphaM * (Mh^i/M0)^betaM
        bTA^i = k2 * log10(Mh^i/M0) - k3
        a1^i = f_r^i * A1^i,  a2^i = k1 * a1^i,  a1delta^i = bTA^i * a1^i

    Reads the 1-indexed `log10_Mh_{i}` and `f_r_{i}` from `nuisance_params`.
    `log10_Mh_{i}` and `log10_M0` must be in the same units (Msun/h).
    """
    log10_mh = jnp.asarray(
        [nuisance_params[f"log10_Mh_{i + 1}"] for i in range(n_bins)]
    )
    f_r = jnp.asarray([nuisance_params[f"f_r_{i + 1}"] for i in range(n_bins)])
    delta = log10_mh - log10_M0
    a1 = f_r * alpha_M * 10.0 ** (delta * beta_M)
    b_ta = k2 * delta - k3
    return a1, k1 * a1, b_ta * a1


def _tatt_m_term(bank, loop_computer, key):
    """`D(z)**4 * sum(kernels)` for `_TATT_M_TERMS[key]` on `bank`'s grid.

    Memoized in `bank` under `tatt_m_<key>` (and `D(z)**4` under
    `tatt_m_d4`), so every pairing that needs a term gets the same array
    object and the engine interpolates it once.
    """

    def _d4(matter_pk, ks, zs):
        return _growth_factor_1d(loop_computer.linear_perturbations, zs) ** 4

    def _compute(matter_pk, ks, zs):
        d4 = bank.get(SpectrumRequest(name="tatt_m_d4", compute=_d4))
        total = sum(
            bank.get(SpectrumRequest(name=n, compute=loop_computer.compute(n)))
            for n in _TATT_M_TERMS[key]
        )
        return d4[:, None] * total[None, :]

    return bank.get(SpectrumRequest(name=f"tatt_m_{key}", compute=_compute))


class TATTMContribution(IntrinsicAlignmentContribution):
    r"""TATT-M intrinsic alignments (Herle et al. 2026, arXiv:2601.15851).

    TATT with per-bin amplitudes set by halo mass and red fraction:

        A1^i     = alphaM * (Mh^i/M0)^betaM
        A2^i(z)  = k1 * D(z) * A1^i
        bTA^i    = k2 * log10(Mh^i/M0) - k3

        C1^i(z)      = -f_r^i * A1^i * C_IA * Omega_m0 / D(z)
        C2^i(z)      =  5 * f_r^i * k1 * A1^i * C_IA * Omega_m0 / D(z)
        C1delta^i(z) =  bTA^i * C1^i(z)

    The growth factor in `A2^i(z)` keeps `C2/C1 = -5*k1` constant in
    redshift (as found for the quadratic shape biases in simulations, e.g.
    Akitsu, Li & Okumura 2023). There is no `(1+z)^eta` evolution: bins
    differ only through `Mh^i` and `f_r^i`. `D(0) = 1`.

    P_II and P_deltaI are TATT's bilinear formulas (see `TATTContribution`)
    with per-bin C's. Because the amplitudes vary per bin, they are handed
    to the Limber engine as `PkTerm`s (`get_pk_terms`): one per-bin
    weighted kernel pair per term, as `NonLinearGalaxyBiasContribution`
    does for its per-bin biases.

    Pairing with a different IA model (NLA, TATT) raises `ValueError`.

    Args:
      tracer: the owning `ShearTracer`.
      alpha_M: normalisation of the A1-halo mass relation.
      beta_M: slope of the A1-halo mass relation.
      loop_computer: one-loop kernel backend, as for `TATTContribution`.
      k1: A2/A1 ratio at z=0 (Herle et al. default).
      k2: slope of bTA in log10(Mh/M0) (Herle et al. default).
      k3: intercept of bTA (Herle et al. default).
      log10_M0: halo-mass pivot, log10(Msun/h).
      C_IA: IA normalisation, as for `NLAContribution`/`TATTContribution`.
    """

    def __init__(
        self,
        tracer: "ShearTracer",
        alpha_M: float,
        beta_M: float,
        loop_computer: TATTLoopComputer,
        k1: float = 0.073,
        k2: float = 0.268,
        k3: float = -0.038,
        log10_M0: float = 13.5,
        C_IA: float = 0.0134,
    ) -> None:
        self._tracer = tracer
        self._loop_computer = loop_computer
        self.alpha_M = alpha_M
        self.beta_M = beta_M
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.log10_M0 = log10_M0
        self.C_IA = C_IA
        self.a1, self.a2, self.a1delta = _tatt_m_per_bin_amplitudes(
            tracer.nuisance_params,
            tracer.n_z_bins,
            alpha_M,
            beta_M,
            k1,
            k2,
            k3,
            log10_M0,
        )

    def _C(self, zs):
        """`(C1, C1delta, C2)` per bin, each shape `(n_bins, len(zs))`."""
        d = _growth_factor_1d(self._loop_computer.linear_perturbations, zs)
        shape = -self.C_IA * self._tracer.background.Omega_m(0.0) / d
        c1 = self.a1[:, None] * shape[None, :]
        c1d = self.a1delta[:, None] * shape[None, :]
        c2 = -5.0 * self.a2[:, None] * shape[None, :]
        return c1, c1d, c2

    def compute_kernel(self, z):
        """Amplitude-free IA kernel `n_i(z) * H(z)/c`, as `TATTContribution`'s.

        The per-bin amplitudes enter through `get_pk_terms` instead.
        """
        h_over_c = self._tracer.perturbations.background.hubble_parameter(z) / _C_KM_S
        return self._tracer.dndz_shifted * h_over_c[None, :]

    def _weighted_kernels(self, zs, bank: SpectraBank):
        """`C * compute_kernel` for C1, C1delta, C2 on `zs`.

        Memoized in `bank`, not on `self`: the bank lives for one `get_Cl`
        call, so a cached value can never outlive a `jax.jit`/`jax.grad`
        trace and leak a tracer into a later call.
        """

        def _compute(matter_pk, ks, bank_zs):
            del matter_pk, ks, bank_zs
            base = self.compute_kernel(zs)
            return tuple(c * base for c in self._C(zs))

        name = f"tatt_m_kernels_{id(self)}_{id(zs)}"
        return bank.get(SpectrumRequest(name=name, compute=_compute))

    def get_spectrum_requests(self):
        return tuple(
            SpectrumRequest(name=name, compute=self._loop_computer.compute(name))
            for name in _TATT_M_KERNELS
        )

    def get_requirements_for_interaction(self, other):
        """All E-mode kernels against IA; only the GI ones otherwise."""
        if isinstance(other, IntrinsicAlignmentContribution):
            return self.get_spectrum_requests()
        return tuple(r for r in self.get_spectrum_requests() if r.name in _GI_KERNELS)

    def get_pk_terms(self, other, zs, bank: SpectraBank):
        """`PkTerm`s for P_II (against TATT-M) or P_deltaI (anything non-IA).

        A `None` kernel or pk means the engine uses that side's own
        `compute_kernel` or the pair's base spectrum (see `PkTerm`).
        """
        if isinstance(other, IntrinsicAlignmentContribution) and not isinstance(
            other, TATTMContribution
        ):
            raise ValueError(
                f"TATTMContribution cannot be paired with {type(other).__name__}: "
                "build both shear tracers with ia_model='TATT-M'."
            )

        def term(key):
            return _tatt_m_term(bank, self._loop_computer, key)

        k1, k1d, k2 = self._weighted_kernels(zs, bank)

        if not isinstance(other, TATTMContribution):
            return (
                PkTerm(k1, None, None),
                PkTerm(k1d, None, term("00e")),
                PkTerm(k2, None, term("0e2")),
            )

        o1, o1d, o2 = other._weighted_kernels(zs, bank)
        return (
            PkTerm(k1, o1, None),
            PkTerm(k1, o1d, term("00e")),
            PkTerm(k1d, o1, term("00e")),
            PkTerm(k1d, o1d, term("0e0e")),
            PkTerm(k2, o2, term("e2e2")),
            PkTerm(k1, o2, term("0e2")),
            PkTerm(k2, o1, term("0e2")),
            PkTerm(k1d, o2, term("d0ee2")),
            PkTerm(k2, o1d, term("d0ee2")),
        )

    def effective_pk_for_bin_pair(self, bank: SpectraBank, bin_i: int, bin_j: int):
        """`{"II": P_II^{ij}(k,z), "deltaI": P_deltaI^{i}(k,z)}` on `bank`'s grid.

        For inspection only; `bin_i`/`bin_j` are 1-indexed.
        """
        c1, c1d, c2 = self._C(bank.zs)
        i, j = bin_i - 1, bin_j - 1
        pdd = bank.matter_pk

        def term(key):
            return _tatt_m_term(bank, self._loop_computer, key)

        p_ii = (
            (c1[i] * c1[j])[:, None] * pdd
            + (c1[i] * c1d[j] + c1[j] * c1d[i])[:, None] * term("00e")
            + (c1d[i] * c1d[j])[:, None] * term("0e0e")
            + (c2[i] * c2[j])[:, None] * term("e2e2")
            + (c1[i] * c2[j] + c1[j] * c2[i])[:, None] * term("0e2")
            + (c1d[i] * c2[j] + c1d[j] * c2[i])[:, None] * term("d0ee2")
        )
        p_delta_i = (
            c1[i][:, None] * pdd
            + c1d[i][:, None] * term("00e")
            + c2[i][:, None] * term("0e2")
        )
        return {"II": p_ii, "deltaI": p_delta_i}


class ShearTracer:
    """Class for the kernel for Cosmic Shear."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        ia_model: "str | Contribution | None" = None,
        tatt_loop_computer: Optional[TATTLoopComputer] = None,
    ):
        r"""Construct a tomographic cosmic-shear tracer.

        ``dndz`` is expected to have shape ``(n_bin, len(z))`` with
        normalized distributions. Per-bin width transformations are applied
        before redshift shifts. ``z`` must not contain zero.

        Args:
          perturbations: Backend supplying background and perturbation
            quantities used by the kernels.
          dndz: Normalized tomographic distributions, shape
            ``(n_bin, len(z))``.
          z: One-dimensional redshift grid corresponding to the second axis
            of ``dndz``.
          nuisance_params: Mapping requiring one-based keys
            ``multiplicative_bias_{i+1}``, ``dz_shear_{i+1}``, and
            ``width_shear_{i+1}`` for each bin.
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
            interface. Passing this with any other `ia_model` raises `ValueError`.

        Raises:
          ValueError: If ``z`` contains zero, ``ia_model`` is unknown, or the
            TATT loop-computer argument is inconsistent with ``ia_model``.
        """
        check_nonzero_redshifts(z)
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
            default - no intrinsic-alignment contribution), otherwise
            `(self.lensing, self.ia)`.
        """
        if self.ia is None:
            return (self.lensing,)
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
        (1-indexed) are required (raises if either is omitted), since each
        bin pair has its own `P_II^{bin_i,bin_j}(k,z)`/`P_deltaI^{bin_i}(k,z)`.

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
        per_bin = isinstance(self.ia, TATTMContribution)
        has_single_grid = (
            self.ia is not None
            and getattr(self.ia, "get_effective_pk", None) is not None
        )

        if not per_bin and not has_single_grid:
            ia_name = type(self.ia).__name__ if self.ia is not None else None
            raise ValueError(
                "get_ia_effective_spectra() needs an IA model that defines "
                "an effective power spectrum (currently ia_model='TATT' or "
                f"'TATT-M'); this tracer's ia is {ia_name!r}, which has "
                "none - either no IA contribution at all, or NLA "
                "(amplitude applied directly inside compute_kernel, not "
                "exposed as a separate grid)."
            )

        if per_bin:
            if bin_i is None or bin_j is None:
                raise ValueError(
                    f"{type(self.ia).__name__} varies its IA amplitude per "
                    "tomographic bin, so get_ia_effective_spectra() needs "
                    "bin_i and bin_j (1-indexed) to know which bin pair's "
                    "P_II(k,z)/P_deltaI(k,z) to return "
                    f"(got bin_i={bin_i!r}, bin_j={bin_j!r})."
                )
            # `.k`/`.z` aren't part of the `Perturbations` protocol (see its
            # docstring), so read them defensively and fail with a clear error.
            ks = getattr(self.perturbations, "k", None) if ks is None else ks
            zs = getattr(self.perturbations, "z", None) if zs is None else zs
            if ks is None or zs is None:
                raise ValueError(
                    "get_ia_effective_spectra() needs explicit ks/zs when "
                    f"{type(self.perturbations).__name__} does not expose its "
                    "own `.k`/`.z` grid."
                )
            matter_pk = self.perturbations.matter_power_spectrum(zs, ks)
            bank = SpectraBank(matter_pk, ks, zs)
            return self.ia.effective_pk_for_bin_pair(bank, bin_i, bin_j)

        # Single, bin-independent grid (currently only TATTContribution).
        if bin_i is not None or bin_j is not None:
            raise ValueError(
                f"{type(self.ia).__name__} doesn't vary its IA amplitude "
                "per bin, so get_ia_effective_spectra() doesn't take "
                f"bin_i/bin_j (got bin_i={bin_i!r}, bin_j={bin_j!r})."
            )
        # `.k`/`.z` aren't part of the `Perturbations` protocol (see its
        # docstring), so read them defensively and fail with a clear error.
        ks = getattr(self.perturbations, "k", None) if ks is None else ks
        zs = getattr(self.perturbations, "z", None) if zs is None else zs
        if ks is None or zs is None:
            raise ValueError(
                "get_ia_effective_spectra() needs explicit ks/zs when "
                f"{type(self.perturbations).__name__} does not expose its "
                "own `.k`/`.z` grid."
            )
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
        r"""Return the NLA intrinsic-alignment kernel on ``z``.

        The per-bin kernel is ``-H(z)/c * AIA * CIA * Omega_m0 *
        (1+z)**EtaIA / D(z) * dndz_shifted``. If the backend returns a
        scale-dependent growth factor, the second tabulated k column is used;
        otherwise its scale-independent result is used. TATT amplitudes are
        represented in effective spectra rather than by this NLA window.

        Args:
            z (array_like): Redshift evaluation grid.

        Returns a two-dimensional NLA kernel with one row per tomographic
        bin and one column per redshift.
        """
        Omega_m0 = self.background.Omega_m(0.0)
        Hz = self.perturbations.background.hubble_parameter(z)
        ks: Any = getattr(self.perturbations, "k", None)
        Dz_raw = self.perturbations.growth_factor(z, ks)
        Dz = Dz_raw[:, 1] if getattr(Dz_raw, "ndim", 1) == 2 else Dz_raw
        A_IA = self.nuisance_params["AIA"]
        C_IA = self.nuisance_params["CIA"]
        Eta_IA = self.nuisance_params["EtaIA"]
        factor = -Hz / c_0 * A_IA * C_IA * Omega_m0 * (1 + z) ** Eta_IA / Dz
        return np.einsum("ij, j->ij", self.dndz_shifted, factor)

    def get_lensing_efficiency_bin(self, z, bin_idx):
        """Compute the legacy single-bin lensing efficiency.

        Uses Akima interpolation of the shifted bin distribution and a fixed
        integration grid from redshift 0 to 4. The vectorized production
        window path uses :meth:`get_lensing_efficiency` instead.
        """
        interpolator = interpax.Akima1DInterpolator(
            self.z, self.dndz_shifted[bin_idx, :]
        )
        Ok, H0 = self.background.Omega_k0, self.background.H0
        x = np.linspace(0.0, 4, 200)
        y = self.background.comoving_distance(x)
        rx_interp = interpax.Akima1DInterpolator(x, transverse_distance(y, Ok, H0))
        cx_interp = interpax.Akima1DInterpolator(x, cos_K(y, Ok, H0))
        f1 = jax.jit(lambda x: interpolator(x))
        f2 = jax.jit(lambda x: interpolator(x) * cx_interp(x) / rx_interp(x))
        integral_1 = simps(f1, z, 3.0)
        integral_2 = simps(f2, z, 3.0)
        rz = self.background.comoving_distance(z)
        efficiency = cos_K(rz, Ok, H0) * integral_1 - integral_2 * transverse_distance(
            rz, Ok, H0
        )
        return efficiency

    def get_lensing_efficiency(self, z):
        r"""
        Compute the discretized lensing efficiency for every bin.

        The efficiency uses ``dndz_shifted`` and the grid distance-ratio
        factor ``f_K(chi(z_source) - chi(z_lens)) / f_K(chi(z_source))``,
        contracted with cached
        stacked Simpson weights. The supplied redshift grid is used as both
        the lens and source grid; this implementation assumes uniform spacing.

        Args:
          z (np.ndarray): 1D, evenly spaced redshift evaluation grid.

        Returns a two-dimensional lensing efficiency with one row per
        tomographic bin and one column per redshift.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz_shifted` has shape ``(n_bin, len(z))``.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        Ok, H0 = self.background.Omega_k0, self.background.H0
        rzrz = (
            transverse_distance(rz[None, :] - rz[:, None], Ok, H0)
            / transverse_distance(rz, Ok, H0)[None, :]
        )
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_lensing(self, z):
        r"""Return the radial weak-lensing kernel for the supplied redshifts.

        It multiplies the discretized efficiency from
        :meth:`get_lensing_efficiency` by
        ``3/2 * (H0/c)**2 * Omega_m0 * (1+z) * f_K(chi(z))``. The harmonic shear
        response is applied by ``AngularTwoPoint``.

        Args:
          z (numpy.ndarray): Redshift evaluation grid.

        Returns a two-dimensional kernel with one row per bin and one column
        per redshift.
        """
        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * transverse_distance(
                self.background.comoving_distance(z),
                self.background.Omega_k0,
                self.background.H0,
            )
        )
        efficiency = self.get_lensing_efficiency(z)
        # MG lensing modification Sigma(z); unity for GR / LCDM perturbations.
        Sigma = getattr(self.perturbations, "Sigma", unit_sigma)(z)
        return np.einsum("ij, j->ij", efficiency, factor) * Sigma[np.newaxis, :]

    def get_window(self, z):
        r"""Sum lensing and optional IA kernels, then apply multiplicative bias.

        Args:
          z (array_like): Redshift evaluation grid.

        Returns a two-dimensional window with one row per bin and one column
        per redshift. Each bin is multiplied once by
        ``1 + multiplicative_bias_{i+1}``.
        """
        total_window = sum(c.compute_kernel(z) for c in self.get_contributions())
        # Apply multiplicative bias
        total_window *= 1 + np.array(self.m_bias)[:, None]
        return total_window
