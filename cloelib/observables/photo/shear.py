r"""Cosmic shear tracer: `ShearTracer`, its Contributions, and its
intrinsic-alignment models.

Compatible with the Tracer protocol. Counterpart to `photo.positions`,
which holds `PositionsTracer`.

Everything needed to build and use a shear tracer - including its
intrinsic-alignment options - lives in this one module:

- `ShearTracer` itself.
- `LensingContribution`, `NLAContribution` (the default IA model): the two
  `Contribution`s a plain `ShearTracer` is built from.
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

Pass `loop_computer=PBJTATTLoopComputer(perturbations)` directly to
`ShearTracer(..., ia_model="TATT", tatt_loop_computer=...)` - one
constructor call, not build-then-replace `.ia`.

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
from cloelib.observables.photo.spectrum_engine import SpectraBank, SpectrumRequest

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

    Implemented by `ShearTracer.get_window_IA` - the default
    `ia_model="NLA"` on `ShearTracer`. `ia_model="TATT"` swaps in
    `TATTContribution` here instead - a different intrinsic-alignment
    model, hence the distinct name; `get_window()` and `AngularTwoPoint`
    don't need to change either way.
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
    `JAXNonLinearPerturbations`), else `perturbations` itself (assumed
    already linear - true if you pass a `*LinearPerturbations` object
    directly). No separate linear-perturbations variable to track and pass
    alongside the tracer's own `perturbations`.

    Known gap: `CAMBNonLinearPerturbations` does not (yet) set
    `.linearperturbations`, and its own `matter_power_spectrum` is always
    nonlinear - passing one here silently uses the *nonlinear* Pk as
    FAST-PT input instead of raising, which is physically wrong. Not
    fixed here; flagged rather than guessed at.

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
        # source if it has one, else assume `perturbations` is linear.
        self.linear_perturbations = getattr(
            perturbations, "linearperturbations", perturbations
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


def _growth_factor_1d(perturbations, z):
    """D(z) as a 1D array, robust to backend-dependent `growth_factor` shape.

    Same handling `ShearTracer.get_window_IA` uses: some
    backends (e.g. CAMB) return D(z, k) with an explicit k-axis; others
    (the JAX backends) ignore `ks` and return a scale-independent D(z), and
    don't set a `.k` attribute at all.
    """
    ks = getattr(perturbations, "k", None)
    d_raw = perturbations.growth_factor(z, ks)
    return d_raw[:, 1] if getattr(d_raw, "ndim", 1) == 2 else d_raw


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
        d = _growth_factor_1d(self._tracer.perturbations, zs)
        return (
            -self.A1
            * self.C_IA
            * omega_m0
            * ((1 + zs) / (1 + self.z0)) ** self.eta1
            / d
        )

    def _C2(self, zs):
        omega_m0 = self._tracer.background.Omega_m(0.0)
        d = _growth_factor_1d(self._tracer.perturbations, zs)
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
        return _growth_factor_1d(self._tracer.perturbations, zs) ** 4

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


class ShearTracer:
    """Class for the kernel for Cosmic Shear."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        ia_model: "str | Contribution" = "NLA",
        tatt_loop_computer: Optional[object] = None,
    ):
        r"""
        Initialize the class instance.

        Args:
          perturbations (object): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
          ia_model (str | Contribution): Which intrinsic-alignment contribution to
            use. `"NLA"` (default) reproduces the exact pre-existing behavior of
            this class (`NLAContribution`/`get_window_IA`, reading
            `nuisance_params["AIA"/"CIA"/"EtaIA"]`). `"TATT"` builds a
            `TATTContribution` from `nuisance_params["AIA"]` (=A1),
            `nuisance_params["A2IA"]`, `nuisance_params["bTA"]`, and optionally
            `nuisance_params["Eta2IA"]`/`["z0IA"]` (default 0.0/0.62). Advanced
            use: pass a `Contribution` instance directly to use any other model.
          tatt_loop_computer: required when `ia_model="TATT"` (raises
            `ValueError` if omitted) - the `TATTContribution`'s one-loop
            kernel backend (see `TATTContribution`'s docstring). Pass
            `PBJTATTLoopComputer(perturbations)` for real FAST-PT-computed
            kernels, or any other object with the same `.compute(name)`
            interface. Passing this with any `ia_model` other than
            `"TATT"` raises `ValueError`.
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

        Both currently delegate to `get_window_lensing`/`get_window_IA`, so
        this is a no-op decomposition today - the seam a future TATT
        contribution would occupy in place of `self.ia`.

        Returns:
          contributions (tuple): `(self.lensing, self.ia)`.
        """
        return (self.lensing, self.ia)

    def _build_ia_contribution(self, ia_model, nuisance_params, tatt_loop_computer):
        """Resolve the `ia_model` constructor argument into a Contribution.

        `"NLA"` (default) preserves the exact pre-existing behavior; `"TATT"`
        builds a `TATTContribution`, which requires `tatt_loop_computer`
        (no illustrative default - see `TATTContribution`'s docstring);
        anything else must already be a `Contribution` instance, used as-is.
        `tatt_loop_computer` is only meaningful for `"TATT"`.
        """
        if ia_model != "TATT" and tatt_loop_computer is not None:
            raise ValueError(
                "tatt_loop_computer is only used when ia_model='TATT' "
                f"(got ia_model={ia_model!r})."
            )
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
        if isinstance(ia_model, str):
            raise ValueError(
                f"Unknown ia_model {ia_model!r}: expected 'NLA', 'TATT', or a "
                "Contribution instance."
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
