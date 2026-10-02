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
    CosmoPowerJAXCurvaturePerturbations,
    CosmoPowerJAXLCDMPerturbations,
    CosmoPowerJAXRunningIndexPerturbations,
    CosmoPowerJAXw0waCDMPerturbations,
    CosmoPowerJAXwCDMPerturbations,
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
    w0wa_lin_cb: CosmoPowerJAXw0waCDMPerturbations.LinearCB,
    w0wa_nl: CosmoPowerJAXw0waCDMPerturbations.NonLinear,
    w0wa_nl_cb: CosmoPowerJAXw0waCDMPerturbations.NonLinearCB,
    wcdm_lin: CosmoPowerJAXwCDMPerturbations.Linear,
    wcdm_lin_cb: CosmoPowerJAXwCDMPerturbations.LinearCB,
    wcdm_nl: CosmoPowerJAXwCDMPerturbations.NonLinear,
    wcdm_nl_cb: CosmoPowerJAXwCDMPerturbations.NonLinearCB,
    lcdm_lin: CosmoPowerJAXLCDMPerturbations.Linear,
    lcdm_lin_cb: CosmoPowerJAXLCDMPerturbations.LinearCB,
    lcdm_nl: CosmoPowerJAXLCDMPerturbations.NonLinear,
    lcdm_nl_cb: CosmoPowerJAXLCDMPerturbations.NonLinearCB,
    curvature_lin: CosmoPowerJAXCurvaturePerturbations.Linear,
    curvature_lin_cb: CosmoPowerJAXCurvaturePerturbations.LinearCB,
    curvature_nl: CosmoPowerJAXCurvaturePerturbations.NonLinear,
    curvature_nl_cb: CosmoPowerJAXCurvaturePerturbations.NonLinearCB,
    running_lin: CosmoPowerJAXRunningIndexPerturbations.Linear,
    running_lin_cb: CosmoPowerJAXRunningIndexPerturbations.LinearCB,
    running_nl: CosmoPowerJAXRunningIndexPerturbations.NonLinear,
    running_nl_cb: CosmoPowerJAXRunningIndexPerturbations.NonLinearCB,
) -> None:
    """The CosmoPower-JAX classes implement `Perturbations` except for the cb spectrum.

    These backends split the total and cb spectra into separate classes (e.g.
    `Linear` and `LinearCB`), neither providing `matter_power_spectrum_cb`.
    """
    _lcdm_lin: Perturbations = lcdm_lin  # ty: ignore[invalid-assignment]
    _w0wa_lin: PerturbationsWithoutCb = w0wa_lin
    _w0wa_lin_cb: PerturbationsWithoutCb = w0wa_lin_cb
    _w0wa_nl: PerturbationsWithoutCb = w0wa_nl
    _w0wa_nl_cb: PerturbationsWithoutCb = w0wa_nl_cb
    _wcdm_lin: PerturbationsWithoutCb = wcdm_lin
    _wcdm_lin_cb: PerturbationsWithoutCb = wcdm_lin_cb
    _wcdm_nl: PerturbationsWithoutCb = wcdm_nl
    _wcdm_nl_cb: PerturbationsWithoutCb = wcdm_nl_cb
    _lcdm_lin_partial: PerturbationsWithoutCb = lcdm_lin
    _lcdm_lin_cb: PerturbationsWithoutCb = lcdm_lin_cb
    _lcdm_nl: PerturbationsWithoutCb = lcdm_nl
    _lcdm_nl_cb: PerturbationsWithoutCb = lcdm_nl_cb
    _curvature_lin: PerturbationsWithoutCb = curvature_lin
    _curvature_lin_cb: PerturbationsWithoutCb = curvature_lin_cb
    _curvature_nl: PerturbationsWithoutCb = curvature_nl
    _curvature_nl_cb: PerturbationsWithoutCb = curvature_nl_cb
    _running_lin: PerturbationsWithoutCb = running_lin
    _running_lin_cb: PerturbationsWithoutCb = running_lin_cb
    _running_nl: PerturbationsWithoutCb = running_nl
    _running_nl_cb: PerturbationsWithoutCb = running_nl_cb


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
