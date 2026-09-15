r"""Angular (galaxy) clustering tracer: `PositionsTracer` and its Contributions.

Compatible with the Tracer protocol. Counterpart to `photo.shear`, which
holds `ShearTracer`.

`NonLinearGalaxyBiasContribution` (McDonald & Roy 2009, arXiv:0902.0991,
"Clustering of Dark Matter Tracers: Generalizing Bias for the Coming Era of
Precision LSS", Eqs. 39-40, extended with a `bk2` non-local/counterterm
bias - see the class docstring): an alternative to the three purely-linear
`galaxy_bias_model` options `GalaxyBiasContribution` implements, selected
via `PositionsTracer(..., galaxy_bias_model="nonlinear",
nl_bias_loop_computer=...)`. Its one-loop kernels come from a required
`loop_computer` - `PBJNonlinearBiasLoopComputer`: real physics, via
`fastpt.FASTPT.one_loop_dd_bias_b3nl` (the `fast-pt` PyPI package - an
optional dependency, `pip install cloelib[fastpt]`) on the linear matter
power spectrum. This is the same "closest available real physics via the
public `fast-pt` package" choice `TATTContribution` (`photo/shear.py`)
makes for intrinsic alignments.

Grounded in production CLOE (github.com/cloe-org/CLOE) where CLOE itself
provides the relevant recipe, and flagged plainly where it doesn't:
- The *photometric* galaxy-bias module `cloe/non_linear/pgg_phot.py`/
  `pgL_phot.py` computes an equivalent set of terms, but in a *different*
  basis - the Lagrangian `bG2`/`bG3` "Gamma"-operator basis, via CLOE's own
  non-public `FASTPTPlus` subclass (`cloe/non_linear/eft.py::EFTofLSS`) -
  and has no `bk2`/counterterm bias at all. `NonLinearGalaxyBiasContribution`
  uses the public-`fast-pt`-native Eulerian basis (`b1`, `b2`, `bs2`,
  `b3nl`) instead, the same basis-choice tradeoff `TATTContribution` already
  makes for IA (real physics via the public package's own API, rather than
  reimplementing CLOE's private engine).
- CLOE's own "1-loop perturbation theory" counterterms (Euclid
  Collaboration: Joudaki et al. 2026, "CLOE. 2. Code implementation",
  arXiv:2603.22475, Sect. 3.7 - "linear, quadratic, and two non-local
  galaxy biases, along with four counter-terms based on 1-loop
  perturbation theory") are part of the *spectroscopic* (redshift-space)
  power spectrum module instead, a distinct RSD-specific counterterm set,
  not a photometric `bk2`. `bk2`'s own `k^2 P_{\delta\delta}(k,z)` form
  here follows the standard EFT-of-LSS treatment that same CLOE paper cites
  for its own counterterms (Baumann et al. 2012, arXiv:1004.2488; Carrasco,
  Hertzberg & Senatore 2012, arXiv:1206.2926) - see the class docstring.

See `CONTRIBUTION_ARCHITECTURE.md` for the design this all follows.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson
from cloelib.auxiliary.systematics import shift_dndz_jax, stretch_dndz_jax
from cloelib.observables.photo.contributions import IntrinsicAlignmentContribution
from cloelib.observables.photo.spectrum_engine import (
    PkTerm,
    SpectraBank,
    SpectrumRequest,
)

# General imports
import jax.numpy as np  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax  # type: ignore
import interpax  # type: ignore
import jax.lax as lx
import numpy as _numpy
from scipy import interpolate as _scipy_interpolate

# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s

# The seven terms `fastpt.FASTPT.one_loop_dd_bias_b3nl` returns beyond the
# plain matter loop (`P_1loop`, `Ps` - already available as
# `perturbations.matter_power_spectrum`), naming them after FAST-PT's own
# return-value names for direct traceability to its docstring/source.
_NLBIAS_KERNELS = (
    "nlbias_Pd1d2",
    "nlbias_Pd2d2",
    "nlbias_Pd1s2",
    "nlbias_Pd2s2",
    "nlbias_Ps2s2",
    "nlbias_sig4",
    "nlbias_sig3nl",
)


@jax.jit
def _L_coeffs(ells):
    ell = ells.astype(np.float64)
    # Avoid invalid sqrt for ell<2 by using a safe ell in the algebra
    ell_s = np.maximum(ell, 2.0)

    Lm1 = (
        -ell_s
        * (ell_s - 1.0)
        / ((2.0 * ell_s - 1.0) * np.sqrt((2.0 * ell_s - 3.0) * (2.0 * ell_s + 1.0)))
    )
    L0 = (2.0 * ell_s**2 + 2.0 * ell_s - 1.0) / (
        (2.0 * ell_s - 1.0) * (2.0 * ell_s + 3.0)
    )
    Lp1 = (
        -(ell_s + 1.0)
        * (ell_s + 2.0)
        / ((2.0 * ell_s + 3.0) * np.sqrt((2.0 * ell_s + 1.0) * (2.0 * ell_s + 5.0)))
    )

    mask = ell >= 2.0
    Lm1 = np.where(mask, Lm1, 0.0)
    L0 = np.where(mask, L0, 0.0)
    Lp1 = np.where(mask, Lp1, 0.0)
    return Lm1, L0, Lp1


@jax.jit
def _alpha_coeffs(ells):
    ell = ells.astype(np.float64)
    denom = 2.0 * ell + 1.0
    am1 = (2.0 * ell - 3.0) / denom
    a0 = np.ones_like(ell)
    ap1 = (2.0 * ell + 5.0) / denom

    # For ell<2 the L's are zero anyway; keep alphas harmless.
    mask = ell >= 2.0
    am1 = np.where(mask, am1, 1.0)
    ap1 = np.where(mask, ap1, 1.0)
    return am1, a0, ap1


@jax.jit
def _interp_linear_2d_queries(chi, y_bz, xq_lz):
    """
    Linear 2D interpolator for the RSD window

    """
    Z = chi.shape[0]
    idx = np.searchsorted(chi, xq_lz, side="right") - 1
    idx = np.clip(idx, 0, Z - 2)

    idx0 = idx[None, :, :]
    idx1 = (idx + 1)[None, :, :]

    y0 = np.take_along_axis(y_bz[:, None, :], idx0, axis=2)
    y1 = np.take_along_axis(y_bz[:, None, :], idx1, axis=2)

    x0 = np.take_along_axis(chi[None, :], idx, axis=1)
    x1 = np.take_along_axis(chi[None, :], idx + 1, axis=1)

    t = (xq_lz - x0) / (x1 - x0)
    t = t[None, :, :]

    out = y0 + t * (y1 - y0)

    # zero outside bounds
    oob = (xq_lz < chi[0]) | (xq_lz > chi[-1])
    out = np.where(oob[None, :, :], 0.0, out)

    return np.transpose(out, (1, 0, 2))


@jax.jit
def get_photo_rsd(ells, chi, S_bin_z):
    Lm1, L0, Lp1 = _L_coeffs(ells)
    am1, _, ap1 = _alpha_coeffs(ells)

    chi_q_m1 = am1[:, None] * chi[None, :]
    chi_q_p1 = ap1[:, None] * chi[None, :]

    S_m1 = _interp_linear_2d_queries(chi, S_bin_z, chi_q_m1)
    S_0 = np.broadcast_to(S_bin_z[None, :, :], S_m1.shape)
    S_p1 = _interp_linear_2d_queries(chi, S_bin_z, chi_q_p1)

    return (
        Lm1[:, None, None] * S_m1 + L0[:, None, None] * S_0 + Lp1[:, None, None] * S_p1
    )


class GalaxyBiasContribution:
    """Galaxy-bias-weighted positions kernel term of `PositionsTracer.get_window()`.

    One of the three linear-bias models selected by `galaxy_bias_model`
    (`PositionsTracer.get_window_positions`). `galaxy_bias_model="nonlinear"`
    swaps in `NonLinearGalaxyBiasContribution` here instead - a different
    galaxy-bias model, hence the distinct name; `get_window()`/
    `AngularTwoPoint` don't need to change either way.
    """

    def __init__(self, tracer: "PositionsTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window_positions(z)


class MagnificationContribution:
    """Magnification-bias kernel term of `PositionsTracer.get_window()`."""

    def __init__(self, tracer: "PositionsTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window_magnification(z)


class PBJNonlinearBiasLoopComputer:
    r"""FAST-PT-backed computer for the seven McDonald & Roy (2009) one-loop
    galaxy-bias kernels used by `NonLinearGalaxyBiasContribution`.

    Calls `fastpt.FASTPT.one_loop_dd_bias_b3nl` on the *linear* matter power
    spectrum at z=0, with the same extrapolation settings and `C_window`
    production CLOE's own FAST-PT-backed code uses (see module docstring).
    The seven outputs are pure functions of k (no z-dependence -
    `NonLinearGalaxyBiasContribution` supplies that separately via its
    constant-in-z b1/b2/bs2/b3nl amplitudes), so they're computed once per
    k-grid, cached, and reused across the seven named `SpectrumRequest`s
    (and across `get_Cl` calls, as long as the k-grid doesn't change)
    instead of re-running FAST-PT per kernel name - the same caching
    `PBJTATTLoopComputer` (`photo/shear.py`) uses, which this class mirrors
    closely; see that class's docstring for the rationale behind each of
    the shared choices below (linear-source resolution, log-uniform k-grid
    construction, edge interpolation).

    Takes the *same* `perturbations` object you'd pass to `PositionsTracer` -
    typically nonlinear, so this class resolves the actual linear source
    itself: `perturbations.linearperturbations` if that attribute exists,
    else `perturbations` itself (assumed already linear). Same known gap as
    `PBJTATTLoopComputer`: `CAMBNonLinearPerturbations` does not (yet) set
    `.linearperturbations`.

    `sig4` (FAST-PT's `_get_sig4`) is a single k-independent normalisation
    constant (a UV-sensitive loop integral, not itself a function of k),
    broadcast to a constant array shaped like the requested `ks` so it fits
    the same `compute(name) -> array(len(ks))` contract as every other
    kernel.

    Requires the optional `fast-pt` dependency (`pip install
    cloelib[fastpt]`); raises `ImportError` with install instructions at
    construction time.
    """

    #: Same low/high-k extrapolation as `PBJTATTLoopComputer`.
    _LOW_EXTRAP = -5
    _HIGH_EXTRAP = 3
    #: Same C_window as `PBJTATTLoopComputer`.
    _C_WINDOW = 0.75

    def __init__(self, perturbations: Perturbations) -> None:
        try:
            import fastpt as fpt
        except ImportError as e:
            raise ImportError(
                "fastpt (the 'fast-pt' PyPI package - the perturbation-"
                "theory engine pbjcosmo's own PT backend is itself built "
                f"on) could not be imported: {e}. Install it with `pip "
                "install fast-pt` (or `pip install cloelib[fastpt]`)."
            )
        self._fpt = fpt
        # See class docstring: use the nonlinear backend's own linear
        # source if it has one, else assume `perturbations` is linear.
        self.linear_perturbations = getattr(
            perturbations, "linearperturbations", perturbations
        )
        self._cached_ks: _numpy.ndarray | None = None
        self._cached_kernels: dict[str, _numpy.ndarray] | None = None

    def _kernels_for(self, ks) -> dict[str, _numpy.ndarray]:
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
        # length; `ks_np` generally satisfies neither - see
        # `PBJTATTLoopComputer._kernels_for`'s docstring for why a
        # dedicated grid is built and interpolated back onto `ks_np`.
        n_win = len(ks_np) + (len(ks_np) % 2)
        k_win = _numpy.logspace(
            _numpy.log10(ks_np.min()), _numpy.log10(ks_np.max()), n_win
        )

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
            low_extrap=self._LOW_EXTRAP,
            high_extrap=self._HIGH_EXTRAP,
            n_pad=n_win,
        )

        (
            _p_1loop,
            _ps,
            pd1d2,
            pd2d2,
            pd1s2,
            pd2s2,
            ps2s2,
            sig4,
            sig3nl,
        ) = f_pt.one_loop_dd_bias_b3nl(p_lin_z0, P_window=None, C_window=self._C_WINDOW)

        raw_kernels = {
            "nlbias_Pd1d2": pd1d2,
            "nlbias_Pd2d2": pd2d2,
            "nlbias_Pd1s2": pd1s2,
            "nlbias_Pd2s2": pd2s2,
            "nlbias_Ps2s2": ps2s2,
            "nlbias_sig3nl": sig3nl,
            "nlbias_sig4": _numpy.full_like(k_win, float(sig4)),
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
            del matter_pk, zs  # pure k-kernel; no z-dependence at all
            return jnp.asarray(self._kernels_for(ks)[name])

        return _compute


class NonLinearGalaxyBiasContribution:
    r"""Non-linear galaxy-bias kernel term of `PositionsTracer.get_window()`,
    see module docstring for basis/citation notes.

    Standard McDonald & Roy (2009, arXiv:0902.0991) Eulerian one-loop
    expansion in `(b1, b2, bs2, b3nl)`, extended with a `bk2` non-local
    ("higher-derivative") counterterm bias in the standard EFT-of-LSS form
    `bk2 * k^2 * P_{\delta\delta}(k,z)` (Baumann et al. 2012, Carrasco,
    Hertzberg & Senatore 2012 - see module docstring). For two (possibly
    different) instances `self`/`other` - the same tomographic bin's own
    auto-spectrum when `other is self`, a genuine cross-population term
    otherwise:

        P_gg(z, k)      = b1_a b1_b P_dd(z, k)
                         + (1/2)(b1_a b2_b + b1_b b2_a) Pd1d2(k,z)
                         + (1/4) b2_a b2_b [Pd2d2(k,z) - 2 sig4(z)]
                         + (1/2)(b1_a bs2_b + b1_b bs2_a) Pd1s2(k,z)
                         + (1/4)(b2_a bs2_b + b2_b bs2_a) [Pd2s2(k,z) - (4/3) sig4(z)]
                         + (1/4) bs2_a bs2_b [Ps2s2(k,z) - (8/9) sig4(z)]
                         + (1/2)(b1_a b3nl_b + b1_b b3nl_a) sig3nl(k,z)
                         + (1/2)(b1_a bk2_b + b1_b bk2_a) k^2 P_dd(k,z)

        P_g,delta(z, k) = b1 P_dd(z, k)
                         + (1/2) b2 Pd1d2(k,z)
                         + (1/2) bs2 Pd1s2(k,z)
                         + (1/2) b3nl sig3nl(k,z)
                         + (1/2) bk2 k^2 P_dd(k,z)

    where `Pd1d2`/`Pd2d2`/`Pd1s2`/`Pd2s2`/`Ps2s2`/`sig3nl` are FAST-PT's
    `one_loop_dd_bias_b3nl` kernels (pure functions of k, computed once at
    z=0 by `PBJNonlinearBiasLoopComputer`) scaled to redshift z via the
    growth factor `D(z)^4` (`_growth4`) - the same z-scaling
    `TATTContribution._D4` applies to its own one-loop kernels, and for the
    same reason: FAST-PT's one-loop integrals are only valid from the
    *linear* Pk, so the z-dependence of a term quadratic in the linear
    density field has to be supplied separately as `D(z)^4`, not read off
    `matter_pk` (which is `P_dd` itself, i.e. `D(z)^2` for the *linear*
    part but includes nonlinear corrections beyond that). `k^2 P_dd(k,z)`
    instead uses `matter_pk` directly (already fully z- and k-dependent, no
    separate growth-factor scaling needed).

    At `b2=bs2=b3nl=bk2=0`, `P_gg`/`P_g,delta` collapse exactly to
    `b1_a*b1_b * matter_pk`/`b1 * matter_pk` - every other term is
    multiplied by a zero amplitude - independent of the (real, but
    then-irrelevant) kernel *values* this class's own tests check this
    against.

    `P_g,delta` is used for every pairing with a non-density contribution
    (`LensingContribution`, `MagnificationContribution`, a CMB-lensing
    kernel, ...) except an `IntrinsicAlignmentContribution` (NLA or TATT):
    there, only the *linear*-bias term `b1 P_dd` is used (`get_pk_terms`
    returns a single `b1 * matter_pk` term, deferring the rest of the
    amplitude to the IA side's own kernel) - matching production CLOE's
    `Pgi_phot_halo_nla`/`_tatt` (`cloe/non_linear/pgL_phot.py`,
    github.com/cloe-org/CLOE), which likewise never applies its own
    `_add_NLbias_contributions_gL` to the galaxy-intrinsic cross (a
    documented CLOE limitation, not something specific to this class): the
    higher-order bias corrections aren't modelled for that cross yet.

    Unlike `TATTContribution`'s global C1/C1delta/C2 amplitudes, `b1`/`b2`/
    `bs2`/`b3nl`/`bk2` here are genuinely **per tomographic bin** (arrays of
    length `tracer.n_z_bins`, matching `GalaxyBiasContribution`'s per-bin
    `b1_photo_bin{i}`) - achieved via `get_pk_terms` (`spectrum_engine.
    PkTerm`), which lets this class hand the generalized Limber engine
    several *per-bin-weighted-kernel* pairs per contribution pairing
    instead of the single shared-kernel/shared-Pk pair `get_effective_pk`
    assumes (see `PkTerm`'s docstring for why the ordinary `get_effective_pk`
    mechanism can't express a per-bin amplitude). Redshift dependence
    *within* a bin is not supported (each bias parameter is one constant
    per bin, matching `GalaxyBiasContribution`'s `"per_bin"` model, not its
    `"per_bin_int"`/`"poly"` ones) - a real, deliberate scope limit.

    Args:
      tracer: the owning `PositionsTracer`.
      b1: linear bias (McDonald & Roy 2009 convention), shape `(n_z_bins,)`.
      loop_computer: object exposing `.compute(name) -> callable(matter_pk,
        ks, zs)` for each of the seven kernel names in `_NLBIAS_KERNELS` -
        the "SpectrumComputer" for the one-loop terms. Required: pass
        `PBJNonlinearBiasLoopComputer(perturbations)` (or build a
        `PositionsTracer` directly with `galaxy_bias_model="nonlinear",
        nl_bias_loop_computer=PBJNonlinearBiasLoopComputer(...)`, which does
        this for you), or any other object with the same interface. No
        default - like `TATTContribution`, this only ever reports kernel
        values it can stand behind as real physics.
      b2, bs2, b3nl, bk2: quadratic, tidal (`s^2`), third-order non-local,
        and non-local/counterterm biases respectively, each shape
        `(n_z_bins,)`.
    """

    def __init__(
        self,
        tracer: "PositionsTracer",
        b1,
        loop_computer: object,
        b2,
        bs2,
        b3nl,
        bk2,
    ) -> None:
        self._tracer = tracer
        self.b1 = b1
        self.b2 = b2
        self.bs2 = bs2
        self.b3nl = b3nl
        self.bk2 = bk2
        self._loop_computer = loop_computer

    def _bias_kernel(self, bias_array, z):
        """Per-bin bias-weighted density kernel: `bias_i * n_i(z) * H(z)/c`."""
        h_over_c = self._tracer.perturbations.background.hubble_parameter(z) / c_0
        return bias_array[:, None] * self._tracer.dndz_shifted * h_over_c[None, :]

    def compute_kernel(self, z):
        """The `b1` (linear-bias) part of the window only - diagnostic/
        `get_window()` use (e.g. plotting), *not* what the generalized
        engine uses for `get_Cl` (that goes through `get_pk_terms`, which
        needs the b2/bs2/b3nl/bk2-weighted kernels too, each paired with
        its own k-shape - see class docstring). Matches
        `GalaxyBiasContribution.compute_kernel`'s per-bin convention.
        """
        return self._bias_kernel(self.b1, z)

    def _growth4(self, zs):
        """`D(z)^4`, robust to backend-dependent `growth_factor` shape -
        same handling `shear.py`'s `_growth_factor_1d`/`TATTContribution.
        _D4` use, and for the same reason (see class docstring).
        """
        ks = getattr(self._tracer.perturbations, "k", None)
        d_raw = self._tracer.perturbations.growth_factor(zs, ks)
        d = d_raw[:, 1] if getattr(d_raw, "ndim", 1) == 2 else d_raw
        return d**4

    def get_spectrum_requests(self):
        return tuple(
            SpectrumRequest(name=name, compute=self._loop_computer.compute(name))
            for name in _NLBIAS_KERNELS
        )

    def get_requirements_for_interaction(self, other):
        """Drop every one-loop kernel when paired with an IA contribution.

        Paired with `IntrinsicAlignmentContribution`, `get_pk_terms` uses
        only `b1 * matter_pk` (see class docstring) - no one-loop kernel is
        needed for that pairing at all. Mirrors `TATTContribution.
        get_requirements_for_interaction`'s pairing-aware pruning, in the
        opposite direction (fewer terms for the IA pairing here, rather
        than for the non-IA one there).
        """
        if isinstance(other, IntrinsicAlignmentContribution):
            return ()
        return self.get_spectrum_requests()

    def _loop_pk(self, bank: SpectraBank, name: str):
        """One FAST-PT one-loop kernel, scaled to redshift via `D(z)^4` and
        broadcast to the full `(len(bank.zs), len(bank.ks))` grid `PkTerm.
        pk` must have.

        Deliberately uses `bank.zs` - *not* whatever z-grid the window
        kernels (`_bias_kernel`) were evaluated on - because `PkTerm.pk`
        feeds straight into `Pkl_interp_signed_vmap(k_lz, zs_calc, ks,
        pert_zs, term.pk.T)` in `AngularTwoPoint._compute_cl_generalized`,
        which treats it as a grid on `pert_zs` (`bank.zs`) and interpolates
        it onto the query grid `zs_calc` itself; only `PkTerm.kernel1`/
        `kernel2` (fed directly to `Cl_integration`, no interpolation) need
        to match `zs_calc`. `TATTContribution.get_effective_pk` makes the
        same choice for its own C1(z)-type amplitudes, for the same reason
        - it reads `bank.zs`, not a separately-passed z.
        """
        raw = bank.get(
            SpectrumRequest(name=name, compute=self._loop_computer.compute(name))
        )
        return self._growth4(bank.zs)[:, None] * raw[None, :]

    def _pd1k2(self, bank: SpectraBank):
        """`k^2 P_dd(k,z)` - the `bk2` counterterm's own Pk shape. Not a
        FAST-PT loop term (no convolution integral involved), so it needs
        no `SpectrumRequest`/`loop_computer` call at all - just `bank`'s
        already-available `matter_pk`/`ks`.
        """
        return bank.matter_pk * (bank.ks**2)[None, :]

    def get_pk_terms(self, other, zs, bank: SpectraBank):
        """The `PkTerm`s for `P_gg(k,z)` or `P_g,delta(k,z)`, as
        appropriate - see class docstring for both formulas and when each
        applies. Returned in `(self, other)` order, per `spectrum_engine.
        get_pk_terms`'s contract.

        `zs` (the tracer's own window/dndz z-grid, `zs_calc` in
        `AngularTwoPoint._compute_cl_generalized`) is used only for the
        window kernels (`_bias_kernel`/`compute_kernel`) - the returned
        `PkTerm.pk` grids are built on `bank.zs` instead (see `_loop_pk`'s
        docstring for why these must be two different z-grids).
        """
        matter_pk = bank.matter_pk
        k_b1 = self._bias_kernel(self.b1, zs)

        if isinstance(other, IntrinsicAlignmentContribution):
            return (PkTerm(k_b1, other.compute_kernel(zs), matter_pk),)

        pd1d2 = self._loop_pk(bank, "nlbias_Pd1d2")
        pd1s2 = self._loop_pk(bank, "nlbias_Pd1s2")
        sig3nl = self._loop_pk(bank, "nlbias_sig3nl")
        pd1k2 = self._pd1k2(bank)

        k_b2 = self._bias_kernel(self.b2, zs)
        k_bs2 = self._bias_kernel(self.bs2, zs)
        k_b3nl = self._bias_kernel(self.b3nl, zs)
        k_bk2 = self._bias_kernel(self.bk2, zs)

        if isinstance(other, NonLinearGalaxyBiasContribution):
            pd2d2 = self._loop_pk(bank, "nlbias_Pd2d2")
            pd2s2 = self._loop_pk(bank, "nlbias_Pd2s2")
            ps2s2 = self._loop_pk(bank, "nlbias_Ps2s2")
            sig4 = self._loop_pk(bank, "nlbias_sig4")

            o_k_b1 = other._bias_kernel(other.b1, zs)
            o_k_b2 = other._bias_kernel(other.b2, zs)
            o_k_bs2 = other._bias_kernel(other.bs2, zs)
            o_k_b3nl = other._bias_kernel(other.b3nl, zs)
            o_k_bk2 = other._bias_kernel(other.bk2, zs)

            return (
                PkTerm(k_b1, o_k_b1, matter_pk),
                PkTerm(k_b1, o_k_b2, 0.5 * pd1d2),
                PkTerm(k_b2, o_k_b1, 0.5 * pd1d2),
                PkTerm(k_b2, o_k_b2, 0.25 * (pd2d2 - 2.0 * sig4)),
                PkTerm(k_b1, o_k_bs2, 0.5 * pd1s2),
                PkTerm(k_bs2, o_k_b1, 0.5 * pd1s2),
                PkTerm(k_b2, o_k_bs2, 0.25 * (pd2s2 - (4.0 / 3.0) * sig4)),
                PkTerm(k_bs2, o_k_b2, 0.25 * (pd2s2 - (4.0 / 3.0) * sig4)),
                PkTerm(k_bs2, o_k_bs2, 0.25 * (ps2s2 - (8.0 / 9.0) * sig4)),
                PkTerm(k_b1, o_k_b3nl, 0.5 * sig3nl),
                PkTerm(k_b3nl, o_k_b1, 0.5 * sig3nl),
                PkTerm(k_b1, o_k_bk2, 0.5 * pd1k2),
                PkTerm(k_bk2, o_k_b1, 0.5 * pd1k2),
            )

        # Galaxy-matter cross (lensing, magnification, CMB lensing, ...).
        other_kernel = other.compute_kernel(zs)
        return (
            PkTerm(k_b1, other_kernel, matter_pk),
            PkTerm(k_b2, other_kernel, 0.5 * pd1d2),
            PkTerm(k_bs2, other_kernel, 0.5 * pd1s2),
            PkTerm(k_b3nl, other_kernel, 0.5 * sig3nl),
            PkTerm(k_bk2, other_kernel, 0.5 * pd1k2),
        )


class PositionsTracer:
    """Class to define the kernel for angular (galaxy) clustering."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        galaxy_bias_model: str,
        nuisance_params: dict,
        include_rsd: bool = False,
        nl_bias_loop_computer: object | None = None,
    ):
        r"""
        Initialize the class instance.

        ### This docstring should be checked. I replaced LinearPerturbations or NonLinearPerturbations with Perturbations as the type of perturbations in the parameters list doc.

        Parameters:
          perturbations (Perturbations): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
          galaxy_bias_model (str): One of `"per_bin"`, `"per_bin_int"`, `"poly"`
            (`GalaxyBiasContribution`'s three linear models, read from
            `nuisance_params["b1_photo_bin{i}"/"b1_photo_poly{i}"]`) or
            `"nonlinear"` (`NonLinearGalaxyBiasContribution`, the standard
            McDonald & Roy 2009 one-loop expansion plus a `bk2` counterterm -
            requires `nl_bias_loop_computer`, and reads, per tomographic bin
            `i` (`i = 0..dndz.shape[0]-1`),
            `nuisance_params["b1_photo_nl_bin{i}"/"b2_photo_nl_bin{i}"/
            "bs2_photo_nl_bin{i}"/"b3nl_photo_nl_bin{i}"/"bk2_photo_nl_bin{i}"]`,
            each defaulting to the value that drops it from the expansion -
            see `NonLinearGalaxyBiasContribution`'s docstring for what it
            does and does not support).
          nuisance_params (dict): A dictionary containing additional parameters that are not directly related to the cosmological model but may affect the observations.
          nl_bias_loop_computer: required when `galaxy_bias_model="nonlinear"`
            (raises `ValueError` if omitted) - `NonLinearGalaxyBiasContribution`'s
            one-loop kernel backend. Pass
            `PBJNonlinearBiasLoopComputer(perturbations)` for real
            FAST-PT-computed kernels, or any other object with the same
            `.compute(name)` interface. Passing this with any
            `galaxy_bias_model` other than `"nonlinear"` raises `ValueError`.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 0

        self.nuisance_params = nuisance_params
        self.dz_pos_i = [
            self.nuisance_params[f"dz_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_pos_i = [
            self.nuisance_params[f"width_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for width_pos
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_pos_i)
        # Correct dndz for dz_pos
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_pos_i)
        self.flags = {"galaxy_bias_model": galaxy_bias_model}
        self.n_z_bins = dndz.shape[0]
        self.magnification_bias = [
            self.nuisance_params[f"magnification_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]
        self.include_rsd = include_rsd

        # Using dict.get so I can provide a default since lax has to compile every branch of the conditional
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        def per_bin_int_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            index_max_nz = np.argmax(dndz, axis=1)
            z_nz_max = jax.vmap(
                lambda i: lx.dynamic_index_in_dim(self.z, i, keepdims=False)
            )(index_max_nz)
            return interpax.interp1d(self.z, z_nz_max, bias_array, extrap=True)

        def poly_case():
            poly_order = 3
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_poly%d" % bin, 1.0)
                    for bin in range(poly_order + 1)
                ]
            )
            return (
                bias_array[0]
                + bias_array[1] * z
                + bias_array[2] * z**2
                + bias_array[3] * z**3
            )

        if galaxy_bias_model == "nonlinear":
            if nl_bias_loop_computer is None:
                raise ValueError(
                    "galaxy_bias_model='nonlinear' requires "
                    "nl_bias_loop_computer (e.g. "
                    "PBJNonlinearBiasLoopComputer(perturbations), from "
                    "cloelib.observables.photo.positions) - "
                    "NonLinearGalaxyBiasContribution has no illustrative "
                    "default."
                )

            def nlbias_bin_array(prefix, default):
                return np.asarray(
                    [
                        nuisance_params.get(f"{prefix}_bin{bin}", default)
                        for bin in range(self.n_z_bins)
                    ]
                )

            self.bias_array = None  # unused: see NonLinearGalaxyBiasContribution
            self.bias = NonLinearGalaxyBiasContribution(
                self,
                b1=nlbias_bin_array("b1_photo_nl", 1.0),
                loop_computer=nl_bias_loop_computer,
                b2=nlbias_bin_array("b2_photo_nl", 0.0),
                bs2=nlbias_bin_array("bs2_photo_nl", 0.0),
                b3nl=nlbias_bin_array("b3nl_photo_nl", 0.0),
                bk2=nlbias_bin_array("bk2_photo_nl", 0.0),
            )
        else:
            if nl_bias_loop_computer is not None:
                raise ValueError(
                    "nl_bias_loop_computer is only used when "
                    "galaxy_bias_model='nonlinear' (got "
                    f"galaxy_bias_model={galaxy_bias_model!r})."
                )
            conditions = np.array(
                [
                    self.flags["galaxy_bias_model"] == "per_bin",
                    self.flags["galaxy_bias_model"] == "per_bin_int",
                    self.flags["galaxy_bias_model"] == "poly",
                ]
            )
            index = np.argwhere(conditions, size=1).squeeze()

            self.bias_array = [per_bin_case, per_bin_int_case, poly_case][index]()
            self.bias = GalaxyBiasContribution(self)

        self.magnification = MagnificationContribution(self)

    def get_contributions(self):
        """Return this tracer's window as its separable Contribution terms.

        Returns:
          contributions (tuple): `(self.bias, self.magnification)` -
            `self.bias` is a `GalaxyBiasContribution` for the three linear
            `galaxy_bias_model`s, or a `NonLinearGalaxyBiasContribution` for
            `galaxy_bias_model="nonlinear"`.
        """
        return (self.bias, self.magnification)

    def get_window_positions(self, z) -> np.ndarray:
        r"""Galaxy Positions window function.

        Implements the galaxy clustering photometric window function.

        $$
            W_i^G(z) = \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\
        $$

        Parameters:
          z (numpy.ndarray|float): Redshift at which to evaluate distribution (array of `float` or `float`)

        Returns:
          window_positions (numpy.ndarray): Window function for angular photometric galaxy clustering
        """

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / c_0
            )
            return window

        def z_func_case():
            window = (
                self.bias_array[None, :]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / c_0
            )
            return window

        conditions = np.array(
            [
                self.flags["galaxy_bias_model"] == "per_bin",
                self.flags["galaxy_bias_model"] in ["per_bin_int", "poly"],
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        window_positions = [per_bin_case, z_func_case][index]()

        return window_positions

    def get_window_rsd(self, ells, H, f, chi) -> np.ndarray:
        r"""
        Linear photo-RSD window :math:`W^{\rm RSD}_i(\ell,z)` tabulated on the internal z-grid.

        We use the shifted-distance (extended Limber) approximation, where the RSD contribution
        is written as a linear combination of the source term evaluated at shifted redshifts:

        $$
        W^{\rm RSD}_i(\ell,z)
        =
        A_\ell\,S_i(z)
        +B_\ell\,S_i\!\left(z_{-2}(\ell,z)\right)
        +C_\ell\,S_i\!\left(z_{+2}(\ell,z)\right),
        $$

        with
        $$
        S_i(z)=\frac{H(z)\,f(z)}{c}\,n_i(z),
        $$

        and the shifted redshifts defined implicitly via comoving distance (here :math:`\chi \equiv r`):
        $$
        \chi\!\left(z_m(\ell,z)\right)=
        \frac{\ell+m+\tfrac12}{\ell+\tfrac12}\,\chi(z),
        \qquad m\in\{-2,+2\}.
        $$

        The coefficients are
        $$
        A_\ell=\frac{2\ell^2+2\ell-1}{(2\ell-1)(2\ell+3)},\quad
        B_\ell=-\frac{\ell(\ell-1)}{(2\ell-1)(2\ell+1)},\quad
        C_\ell=-\frac{(\ell+1)(\ell+2)}{(2\ell+1)(2\ell+3)}.
        $$

        Notes
        -----
        - `chi` is used purely as an interpolation coordinate so `get_photo_rsd` can evaluate
          the shifted arguments efficiently.

        Parameters
        ----------
        ells : array_like
            Multipoles ell.
        H, f : array_like
            H(z) and growth rate f(z), sampled on the same z-grid as `self.dndz_shifted`.
        chi : array_like
            Comoving distance χ(z)=r(z), sampled on the same z-grid.

        Returns
        -------
        ndarray
            The RSD window sampled on the z-grid (shape as returned by `get_photo_rsd`).
        """
        # S_i(z) = H(z) f(z) n_i(z) / c
        S = (H[None, :] * f[None, :] / c_0) * self.dndz_shifted
        return get_photo_rsd(ells, chi, S)

    def get_magnification_efficiency(self, z):
        r"""
        Compute the magnification efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Parameters:
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

    def get_window_magnification(self, z):
        r"""Magnification photometric galaxy kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        `np.trapz` on the array along one axis.

        $$
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} b_{\rm mag, i} (1 + z)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\
        $$

        Parameters:
          z (numpy.ndarray): Redshift at which weight is evaluated (array of `float`).

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
        efficiency = self.get_magnification_efficiency(z)
        return (
            np.einsum("ij, j->ij", efficiency, factor)
            * np.array(self.magnification_bias)[:, None]
        )

    def get_window(self, z) -> np.ndarray:
        """
        Compute the angular photometric galaxy clustering window function.

        This function combines the galaxy clustering window and the magnification
        bias window to produce the final window function.

        Parameters:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        window = sum(c.compute_kernel(z) for c in self.get_contributions())
        return window
