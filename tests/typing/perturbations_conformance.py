"""Static checks that every cosmology backend implements the shared interfaces.

This module is never imported: `tests/test_perturbations_interface.py` runs
`ty check` on it, so each assignment below is verified by the type checker
rather than at runtime. It fails if a `Perturbations`/`Background`
implementation drifts from its protocol, or if a constructor stops matching
`LinearPerturbationsFactory`/`NonLinearPerturbationsFactory`.

Known, intentional exceptions are marked with `# ty: ignore[invalid-assignment]`
and a reason. The test enables `unused-ignore-comment`, so an exception that
stops being one must have its ignore comment removed. So that an ignore does
not also hide unrelated drift, each exception is additionally checked against
`Perturbations` without only the member it lacks (`PerturbationsWithoutCb`,
`PerturbationsWithoutBackground`), with no ignore.
"""

from typing import Optional, Protocol

from cloelib.cosmology.cosmology import (
    Background,
    LinearPerturbationsFactory,
    NonLinearPerturbationsFactory,
    Perturbations,
    T,
)
from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)
from cloelib.cosmology.class_cosmology import (
    CLASSBackground,
    CLASSLinearPerturbations,
    CLASSNonLinearPerturbations,
)
from cloelib.cosmology.hi_class_cosmology import (
    hi_classBackground,
    hi_classLinearPerturbations,
    hi_classNonLinearPerturbations,
)
from cloelib.cosmology.mgclass_cosmology import (
    MGCLASSBackground,
    MGCLASSLinearPerturbations,
    MGCLASSNonLinearPerturbations,
)
from cloelib.cosmology.mochi_class_cosmology import (
    mochiCLASSBackground,
    mochiCLASSLinearPerturbations,
    mochiCLASSNonLinearPerturbations,
)
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelib.cosmology.baccoemu_cosmology import (
    BACCOemuLinearPerturbations,
    BACCOemuNonLinearPerturbations,
)
from cloelib.cosmology.EE2_cosmology import EE2NonLinearPerturbations
from cloelib.cosmology.emantis_cosmology import EmantisFofrNonLinearPerturbations
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)
from cloelib.cosmology.ReACTEmu_cosmology import BoostedPerturbations
from cloelib.cosmology.TabulatedBoost_cosmology import TabulatedBoostedPerturbations
from cloelib.cosmology.cosmopower_jax_cosmology import (
    CosmoPowerJAXLCDMCurvaturePerturbations,
    CosmoPowerJAXLCDMPerturbations,
    CosmoPowerJAXLCDMRunningIndexPerturbations,
    CosmoPowerJAXw0waCDMPerturbations,
    CosmoPowerJAXw0waCurvaturePerturbations,
    CosmoPowerJAXw0waRunningIndexPerturbations,
    CosmoPowerJAXwCDMPerturbations,
)
from cloelib.cosmology.mg_cosmopower_jax_cosmology import (
    MGLinearPerturbations,
    MGNonLinearPerturbations,
)


# Copies of `Perturbations` without one member. Their member names are kept in
# sync with `Perturbations` by `test_partial_protocols_match_perturbations`.
class PerturbationsWithoutCb(Protocol):
    """`Perturbations` without `matter_power_spectrum_cb`."""

    @property
    def background(self) -> Background: ...

    def growth_factor(self, zs: T, ks: T) -> T: ...

    def growth_rate(self, zs: Optional[T] = None, ks: Optional[T] = None) -> T: ...

    def matter_power_spectrum(self, zs: T, ks: T) -> T: ...

    def sigma8_0(self) -> float: ...


class PerturbationsWithoutBackground(Protocol):
    """`Perturbations` without `background`."""

    def growth_factor(self, zs: T, ks: T) -> T: ...

    def growth_rate(self, zs: Optional[T] = None, ks: Optional[T] = None) -> T: ...

    def matter_power_spectrum(self, zs: T, ks: T) -> T: ...

    def matter_power_spectrum_cb(self, zs: T, ks: T) -> T: ...

    def sigma8_0(self) -> float: ...


def backgrounds(
    camb: CAMBBackground,
    class_: CLASSBackground,
    hi_class: hi_classBackground,
    mgclass: MGCLASSBackground,
    mochi: mochiCLASSBackground,
    jax: JAXBackground,
) -> None:
    """Every background implements `Background`."""
    _camb: Background = camb
    _class: Background = class_
    _hi_class: Background = hi_class
    _mgclass: Background = mgclass
    _mochi: Background = mochi
    # `JAXBackground.N_ur` is not implemented and returns None (see its FIXME).
    _jax: Background = jax  # ty: ignore[invalid-assignment]


def perturbations(
    mg_lin: MGLinearPerturbations,
    mg_nl: MGNonLinearPerturbations,
    camb_lin: CAMBLinearPerturbations,
    camb_nl: CAMBNonLinearPerturbations,
    class_lin: CLASSLinearPerturbations,
    class_nl: CLASSNonLinearPerturbations,
    hi_class_lin: hi_classLinearPerturbations,
    hi_class_nl: hi_classNonLinearPerturbations,
    mgclass_lin: MGCLASSLinearPerturbations,
    mgclass_nl: MGCLASSNonLinearPerturbations,
    mochi_lin: mochiCLASSLinearPerturbations,
    mochi_nl: mochiCLASSNonLinearPerturbations,
    hmemu_lin: HMemuLinearPerturbations,
    hmemu_nl: HMemuNonLinearPerturbations,
    bacco_lin: BACCOemuLinearPerturbations,
    bacco_nl: BACCOemuNonLinearPerturbations,
    ee2_nl: EE2NonLinearPerturbations,
    emantis_nl: EmantisFofrNonLinearPerturbations,
    jax_lin: JAXLinearPerturbations,
    jax_nl: JAXNonLinearPerturbations,
    react: BoostedPerturbations,
    tabulated: TabulatedBoostedPerturbations,
) -> None:
    """Every perturbations class implements `Perturbations`."""
    _camb_lin: Perturbations = camb_lin
    _camb_nl: Perturbations = camb_nl
    _class_lin: Perturbations = class_lin
    _class_nl: Perturbations = class_nl
    _hi_class_lin: Perturbations = hi_class_lin
    _hi_class_nl: Perturbations = hi_class_nl
    _mgclass_lin: Perturbations = mgclass_lin
    _mgclass_nl: Perturbations = mgclass_nl
    _mochi_lin: Perturbations = mochi_lin
    _mochi_nl: Perturbations = mochi_nl
    _hmemu_lin: Perturbations = hmemu_lin
    _hmemu_nl: Perturbations = hmemu_nl
    # The MG boost classes are module-level bases bound by `mg_perturbations`.
    _mg_lin: Perturbations = mg_lin
    _mg_nl: Perturbations = mg_nl
    _bacco_lin: Perturbations = bacco_lin
    _bacco_nl: Perturbations = bacco_nl
    _ee2_nl: Perturbations = ee2_nl
    _emantis_nl: Perturbations = emantis_nl
    # Only through `JAXBackground`, see `backgrounds` above.
    _jax_lin: Perturbations = jax_lin  # ty: ignore[invalid-assignment]
    _jax_nl: Perturbations = jax_nl  # ty: ignore[invalid-assignment]
    _jax_lin_partial: PerturbationsWithoutBackground = jax_lin
    _jax_nl_partial: PerturbationsWithoutBackground = jax_nl
    # The boost wrappers do not provide `matter_power_spectrum_cb`.
    _react: Perturbations = react  # ty: ignore[invalid-assignment]
    _tabulated: Perturbations = tabulated  # ty: ignore[invalid-assignment]
    _react_partial: PerturbationsWithoutCb = react
    _tabulated_partial: PerturbationsWithoutCb = tabulated


def cosmopower_perturbations(
    w0wa_lin: CosmoPowerJAXw0waCDMPerturbations.Linear,
    w0wa_nl: CosmoPowerJAXw0waCDMPerturbations.NonLinear,
    w0wa_nl_hf: CosmoPowerJAXw0waCDMPerturbations.NonLinearHalofit,
    wcdm_lin: CosmoPowerJAXwCDMPerturbations.Linear,
    wcdm_nl: CosmoPowerJAXwCDMPerturbations.NonLinear,
    wcdm_nl_hf: CosmoPowerJAXwCDMPerturbations.NonLinearHalofit,
    lcdm_lin: CosmoPowerJAXLCDMPerturbations.Linear,
    lcdm_nl: CosmoPowerJAXLCDMPerturbations.NonLinear,
    lcdm_nl_hf: CosmoPowerJAXLCDMPerturbations.NonLinearHalofit,
    lcdm_curv_lin: CosmoPowerJAXLCDMCurvaturePerturbations.Linear,
    lcdm_curv_nl: CosmoPowerJAXLCDMCurvaturePerturbations.NonLinear,
    w0wa_curv_lin: CosmoPowerJAXw0waCurvaturePerturbations.Linear,
    w0wa_curv_nl: CosmoPowerJAXw0waCurvaturePerturbations.NonLinear,
    lcdm_run_lin: CosmoPowerJAXLCDMRunningIndexPerturbations.Linear,
    lcdm_run_nl: CosmoPowerJAXLCDMRunningIndexPerturbations.NonLinear,
    w0wa_run_lin: CosmoPowerJAXw0waRunningIndexPerturbations.Linear,
    w0wa_run_nl: CosmoPowerJAXw0waRunningIndexPerturbations.NonLinear,
) -> None:
    """The CosmoPower-JAX classes implement the full `Perturbations` protocol.

    Each total-matter class provides both `matter_power_spectrum` and
    `matter_power_spectrum_cb` (the cb spectrum, built at construction from the paired cb
    emulator), so they conform to `Perturbations` like the other backends.
    """
    _w0wa_lin: Perturbations = w0wa_lin
    _w0wa_nl: Perturbations = w0wa_nl
    _w0wa_nl_hf: Perturbations = w0wa_nl_hf
    _wcdm_lin: Perturbations = wcdm_lin
    _wcdm_nl: Perturbations = wcdm_nl
    _wcdm_nl_hf: Perturbations = wcdm_nl_hf
    _lcdm_lin: Perturbations = lcdm_lin
    _lcdm_nl: Perturbations = lcdm_nl
    _lcdm_nl_hf: Perturbations = lcdm_nl_hf
    _lcdm_curv_lin: Perturbations = lcdm_curv_lin
    _lcdm_curv_nl: Perturbations = lcdm_curv_nl
    _w0wa_curv_lin: Perturbations = w0wa_curv_lin
    _w0wa_curv_nl: Perturbations = w0wa_curv_nl
    _lcdm_run_lin: Perturbations = lcdm_run_lin
    _lcdm_run_nl: Perturbations = lcdm_run_nl
    _w0wa_run_lin: Perturbations = w0wa_run_lin
    _w0wa_run_nl: Perturbations = w0wa_run_nl


# Linear constructors: `cls(background=..., redshifts=...)`.
_camb_lin: LinearPerturbationsFactory[CAMBBackground] = CAMBLinearPerturbations
_class_lin: LinearPerturbationsFactory[CLASSBackground] = CLASSLinearPerturbations
_hi_class_lin: LinearPerturbationsFactory[hi_classBackground] = (
    hi_classLinearPerturbations
)
_mgclass_lin: LinearPerturbationsFactory[MGCLASSBackground] = MGCLASSLinearPerturbations
_mochi_lin: LinearPerturbationsFactory[mochiCLASSBackground] = (
    mochiCLASSLinearPerturbations
)
_hmemu_lin: LinearPerturbationsFactory[CAMBBackground] = HMemuLinearPerturbations
_bacco_lin: LinearPerturbationsFactory[CAMBBackground] = BACCOemuLinearPerturbations

# Nonlinear constructors:
# `cls(background=..., linearperturbations=..., redshifts=...)`.
_camb_nl: NonLinearPerturbationsFactory[CAMBBackground, CAMBLinearPerturbations] = (
    CAMBNonLinearPerturbations
)
_class_nl: NonLinearPerturbationsFactory[CLASSBackground, CLASSLinearPerturbations] = (
    CLASSNonLinearPerturbations
)
_hi_class_nl: NonLinearPerturbationsFactory[
    hi_classBackground, hi_classLinearPerturbations
] = hi_classNonLinearPerturbations
_mgclass_nl: NonLinearPerturbationsFactory[
    MGCLASSBackground, MGCLASSLinearPerturbations
] = MGCLASSNonLinearPerturbations
_mochi_nl: NonLinearPerturbationsFactory[
    mochiCLASSBackground, mochiCLASSLinearPerturbations
] = mochiCLASSNonLinearPerturbations
_hmemu_nl: NonLinearPerturbationsFactory[CAMBBackground, HMemuLinearPerturbations] = (
    HMemuNonLinearPerturbations
)
_bacco_nl: NonLinearPerturbationsFactory[
    CAMBBackground, BACCOemuLinearPerturbations
] = BACCOemuNonLinearPerturbations
_ee2_nl: NonLinearPerturbationsFactory[CAMBBackground, CAMBLinearPerturbations] = (
    EE2NonLinearPerturbations
)

# The backends that ignore `linearperturbations` accept any linear perturbations.
_camb_nl_any: NonLinearPerturbationsFactory[
    CAMBBackground, HMemuLinearPerturbations
] = CAMBNonLinearPerturbations
_class_nl_none: NonLinearPerturbationsFactory[CLASSBackground, None] = (
    CLASSNonLinearPerturbations
)

# Known exceptions. Formatting is disabled below so that each ignore comment
# stays on the line ty reports.
CAMBNonLinearFactory = NonLinearPerturbationsFactory[
    CAMBBackground, CAMBLinearPerturbations
]
CLASSNonLinearFactory = NonLinearPerturbationsFactory[
    CLASSBackground, CLASSLinearPerturbations
]

# fmt: off
# HMcode2020Emu stitches the cached `Pk`/`Pk_cb` grid of its linear
# perturbations (`WithLinearSpectrumGrid`), which CAMB's does not provide.
_hmemu_on_camb: CAMBNonLinearFactory = HMemuNonLinearPerturbations  # ty: ignore[invalid-assignment]

# mochi_class requires its own background.
_mochi_on_class: CLASSNonLinearFactory = mochiCLASSNonLinearPerturbations  # ty: ignore[invalid-assignment]

# e-MANTIS also needs the LCDM nonlinear perturbations and fR0.
_emantis_nl: CAMBNonLinearFactory = EmantisFofrNonLinearPerturbations  # ty: ignore[invalid-assignment]
# fmt: on
