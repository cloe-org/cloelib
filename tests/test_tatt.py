"""
Tests for the generalized spectrum engine and TATT (see
`CONTRIBUTION_ARCHITECTURE.md`).

These check the *architecture*: that the generalized engine is a true no-op
for every configuration that doesn't need it (already covered exhaustively
by `test_photo_tracers_characterization.py`/`test_contributions.py`, since
`ia_model="NLA"` is the default and unchanged), that TATT's request pruning
and NLA-reduction limit are correct, and that the generalized Cl path
produces finite, correctly-shaped output - with both
`PlaceholderTATTLoopComputer` (no extra dependency, illustrative kernels)
and, when `fast-pt` is installed, `PBJTATTLoopComputer` (real FAST-PT
`IA_ta`/`IA_tt`/`IA_mix` kernels, matching production CLOE's
`cloe/non_linear/miscellanous.py::ia_tatt_terms` - see
`cloelib/observables/photo/shear.py`).
"""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo.contributions import AbstractIAContribution
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.shear import (
    IntrinsicAlignmentContribution,
    TATTContribution,
    _ALL_KERNELS,
    _GI_KERNELS,
)
from cloelib.observables.photo.spectrum_engine import needs_generalized_engine
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

import importlib.util

_FASTPT_INSTALLED = importlib.util.find_spec("fastpt") is not None
if _FASTPT_INSTALLED:
    from cloelib.observables.photo.shear import PBJTATTLoopComputer


@pytest.fixture(scope="module")
def cosmo_setup():
    H0 = 67.7
    h = H0 / 100.0
    background = CAMBBackground(
        H0=H0,
        Omega_b0=0.022 / h**2,
        Omega_cdm0=0.12 / h**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        alpha_s=0.0,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )
    z_auto = np.linspace(0.01, 1100.0, 100)
    z = np.linspace(0.2, 2.0, 15)
    perturbations = CAMBNonLinearPerturbations(background, None, z_auto)
    return perturbations, z


@pytest.fixture(scope="module")
def linear_perturbations(cosmo_setup):
    """A *linear* Perturbations object, as `PBJTATTLoopComputer` requires
    (FAST-PT's one-loop integrals are only valid starting from the linear
    Pk - see its docstring) - `cosmo_setup`'s own `perturbations` is
    nonlinear.
    """
    perturbations, z = cosmo_setup
    return CAMBLinearPerturbations(perturbations.background, perturbations.z)


def _shear_nuisance(n_z_bins, **extra):
    return {
        **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(n_z_bins)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(n_z_bins)},
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": -0.41,
        **extra,
    }


def test_ia_model_nla_default_gives_identical_contribution_type(cosmo_setup):
    """`ia_model="NLA"` (the default) must build the pre-existing NLA class."""
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    default_tracer = ShearTracer(
        perturbations=perturbations, dndz=dndz, z=z, nuisance_params=_shear_nuisance(1)
    )
    explicit_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    assert isinstance(default_tracer.ia, IntrinsicAlignmentContribution)
    assert isinstance(explicit_tracer.ia, IntrinsicAlignmentContribution)
    assert not needs_generalized_engine(
        default_tracer.get_contributions(), default_tracer.get_contributions()
    )


def test_ia_model_tatt_builds_tatt_contribution(cosmo_setup):
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
    )
    assert isinstance(tracer.ia, TATTContribution)
    assert isinstance(tracer.ia, AbstractIAContribution)
    assert tracer.ia.A1 == 1.0
    assert tracer.ia.A2 == 0.4
    assert tracer.ia.b_TA == -0.83


def test_unknown_ia_model_string_raises(cosmo_setup):
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    with pytest.raises(ValueError):
        ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            nuisance_params=_shear_nuisance(1),
            ia_model="bogus",
        )


def test_requirements_pruned_for_gi_vs_ii(cosmo_setup):
    """TATT must request all 10 kernels for II, only the 4 GI ones for GI/gI.

    Mirrors toy_cloelib's TATT `pk_beta`-dropping test
    (`demo_syren_3x2pt.py`): the pruning is what makes the "avoid computing
    unneeded one-loop terms" optimization real, not just an unused method.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
    )
    tatt = tracer.ia
    lensing = tracer.lensing

    ii_names = {r.name for r in tatt.get_requirements_for_interaction(tatt)}
    gi_names = {r.name for r in tatt.get_requirements_for_interaction(lensing)}

    assert ii_names == set(_ALL_KERNELS)
    assert gi_names == set(_GI_KERNELS)
    assert gi_names < ii_names  # strictly fewer terms for GI than II

    contribs = tracer.get_contributions()  # (lensing, ia)
    assert needs_generalized_engine(contribs, contribs)


def test_tatt_reduces_to_nla_form_when_a2_and_bta_zero(cosmo_setup):
    """When A2=b_TA=0, TATT's C1delta=C2=0, so Eqs. (13)/(15) collapse to
    Eq. (10): P_II = C1**2 * P_dd, P_deltaI = C1 * P_dd - independent of the
    (unvalidated) one-loop kernel *values*, since every kernel term is
    multiplied by a zero amplitude. This is checkable exactly, without
    needing a real PT backend.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=0.0, bTA=0.0),
        ia_model="TATT",
    )
    tatt = tracer.ia

    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    matter_pk = perturbations.matter_power_spectrum(pert_zs, ks)

    from cloelib.observables.photo.spectrum_engine import SpectraBank

    bank = SpectraBank(matter_pk, ks, pert_zs)

    p_ii = tatt.get_effective_pk(tatt, bank)
    p_di = tatt.get_effective_pk(tracer.lensing, bank)

    c1 = tatt._C1(pert_zs)[:, None]
    np.testing.assert_allclose(
        np.asarray(p_ii), np.asarray(c1**2 * matter_pk), rtol=1e-10
    )
    np.testing.assert_allclose(np.asarray(p_di), np.asarray(c1 * matter_pk), rtol=1e-10)


def test_generalized_cl_finite_and_correctly_shaped(cosmo_setup):
    """End-to-end: ShearTracer(ia_model="TATT") x PositionsTracer through
    AngularTwoPoint.get_Cl produces finite output with the same shape
    contract as the legacy (NLA) path - the architecture actually runs.
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
    )
    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params={
            **{f"dz_pos_{i + 1}": 0.0 for i in range(n_z_bins)},
            **{f"width_pos_{i + 1}": 1.0 for i in range(n_z_bins)},
            **{f"magnification_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
            "b1_photo_bin0": 1.1,
            "b1_photo_bin1": 1.4,
        },
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    she_she = AngularTwoPoint(shear_tracer, shear_tracer).get_Cl(ells, 0, ks)
    pos_she = AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells, 0, ks)

    assert ("SHE", "SHE", 1, 1) in she_she
    for key, spectrum in {**she_she, **pos_she}.items():
        arr = np.asarray(spectrum.array)
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"

    assert she_she[("SHE", "SHE", 1, 1)].array.shape == (2, 2, len(ells))
    assert pos_she[("POS", "SHE", 1, 1)].array.shape == (2, len(ells))


def test_tatt_matches_legacy_nla_end_to_end_at_z0_zero(cosmo_setup):
    """Strongest available check: with A2=b_TA=0 and z0=0 (matching legacy
    NLA's implicit z0=0 pivot, see `tatt.py`'s `compute_kernel` docstring),
    `ia_model="TATT"` must reproduce `ia_model="NLA"`'s `get_Cl` output
    bit-for-bit - through the *entire* pipeline (two independent code
    paths: `_compute_cl_legacy` vs `_compute_cl_generalized`), not just the
    isolated `get_effective_pk` formula `test_tatt_reduces_to_nla_form_...`
    already checks.

    This caught a real bug during development: `TATTContribution.
    compute_kernel` initially omitted the H(z)/c Jacobian
    `ShearTracer.get_window_IA`'s NLA implementation folds into its window
    - without it, TATT's Cl differed from NLA's by several orders of
    magnitude despite being analytically the same quantity at A2=b_TA=0.

    Tolerance is 5e-3, not exact: the legacy path interpolates `log10(Pk)`
    directly (`Pkl_interp`), the generalized path interpolates
    `log10(|Pk|)` and `sign(Pk)` separately then recombines
    (`Pkl_interp_signed`, needed because TATT's other terms are generically
    signed - see its docstring) - algebraically equivalent when `Pk > 0`
    everywhere (true here), but not bit-identical, since they're genuinely
    different numerical procedures (observed max relative difference
    ~1.6e-3, likely from akima's boundary-derivative estimation not being
    perfectly invariant to the extra sign-array interpolation).
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    nla_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins),
        ia_model="NLA",
    )
    tatt_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, A2IA=0.0, bTA=0.0, z0IA=0.0),
        ia_model="TATT",
    )

    cl_nla = AngularTwoPoint(nla_tracer, nla_tracer).get_Cl(ells, 0, ks)
    cl_tatt = AngularTwoPoint(tatt_tracer, tatt_tracer).get_Cl(ells, 0, ks)

    np.testing.assert_allclose(
        np.asarray(cl_tatt[("SHE", "SHE", 1, 1)].array),
        np.asarray(cl_nla[("SHE", "SHE", 1, 1)].array),
        rtol=5e-3,
    )


def test_rsd_with_generalized_engine_raises_not_implemented(cosmo_setup):
    perturbations, z = cosmo_setup
    n_z_bins = 1
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
    )
    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params={
            "dz_pos_1": 0.0,
            "width_pos_1": 1.0,
            "magnification_bias_1": 0.0,
            "b1_photo_bin0": 1.1,
        },
        include_rsd=True,
    )
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    with pytest.raises(NotImplementedError):
        AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells, 0, ks)


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_loop_computer_kernels_finite_and_correctly_shaped(
    linear_perturbations,
):
    """The real FAST-PT backend must return all ten named kernels, each
    finite and shaped like the requested k-grid - `TATTContribution`'s
    `get_effective_pk` relies on this shape contract (same as
    `PlaceholderTATTLoopComputer`'s).
    """
    computer = PBJTATTLoopComputer(linear_perturbations)
    ks = np.logspace(-3, 1, 40)
    zs = np.linspace(0.0, 2.0, 5)
    matter_pk = np.ones((len(zs), len(ks)))  # unused by this computer

    for name in _ALL_KERNELS:
        values = np.asarray(computer.compute(name)(matter_pk, ks, zs))
        assert values.shape == ks.shape, name
        assert np.all(np.isfinite(values)), name


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_loop_computer_caches_across_kernel_names(linear_perturbations):
    """Ten kernel names share one (expensive) FAST-PT call per k-grid,
    not one FAST-PT call each - `_kernels_for` must be memoized.
    """
    computer = PBJTATTLoopComputer(linear_perturbations)
    ks = np.logspace(-3, 1, 40)
    zs = np.linspace(0.0, 2.0, 5)
    matter_pk = np.ones((len(zs), len(ks)))

    computer.compute(_ALL_KERNELS[0])(matter_pk, ks, zs)
    cached_after_first = computer._cached_kernels
    computer.compute(_ALL_KERNELS[1])(matter_pk, ks, zs)
    assert computer._cached_kernels is cached_after_first


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_reduces_to_nla_form_when_a2_and_bta_zero(
    cosmo_setup, linear_perturbations
):
    """Same check as `test_tatt_reduces_to_nla_form_when_a2_and_bta_zero`,
    but with the real FAST-PT loop computer wired in instead of the
    placeholder - since every kernel term is multiplied by a zero
    amplitude at A2=b_TA=0, the assembled P_II/P_deltaI must still collapse
    to the exact NLA form regardless of which backend computed the kernels.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=0.0, bTA=0.0),
        ia_model="TATT",
    )
    tracer.ia = TATTContribution(
        tracer,
        A1=tracer.ia.A1,
        A2=tracer.ia.A2,
        b_TA=tracer.ia.b_TA,
        eta1=tracer.ia.eta1,
        eta2=tracer.ia.eta2,
        z0=tracer.ia.z0,
        C_IA=tracer.ia.C_IA,
        loop_computer=PBJTATTLoopComputer(linear_perturbations),
    )
    tatt = tracer.ia

    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    matter_pk = perturbations.matter_power_spectrum(pert_zs, ks)

    from cloelib.observables.photo.spectrum_engine import SpectraBank

    bank = SpectraBank(matter_pk, ks, pert_zs)

    p_ii = tatt.get_effective_pk(tatt, bank)
    p_di = tatt.get_effective_pk(tracer.lensing, bank)

    c1 = tatt._C1(pert_zs)[:, None]
    np.testing.assert_allclose(
        np.asarray(p_ii), np.asarray(c1**2 * matter_pk), rtol=1e-10
    )
    np.testing.assert_allclose(np.asarray(p_di), np.asarray(c1 * matter_pk), rtol=1e-10)


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_generalized_cl_finite_with_real_fastpt(
    cosmo_setup, linear_perturbations
):
    """End-to-end regression test for a real numerical-stability bug found
    while developing `PBJTATTLoopComputer`: the real FAST-PT kernels are
    steep and sign-changing near their k-grid's edges (unlike
    `PlaceholderTATTLoopComputer`'s deliberately smooth, tame shape), and
    the Limber grid `k_l = (ell+0.5)/chi` reaches well past that k-grid at
    the low-z/high-ell corner (chi -> 0). Naively extrapolating such a
    kernel there (`Pkl_interp_signed`'s akima extrapolation) sent the
    interpolated log-magnitude to hundreds, overflowing `10**(...)` to
    +-inf and producing a NaN `Cl` - fixed by clamping the query k to the
    grid's own domain before interpolating (see `Pkl_interp_signed`'s
    docstring in `angular_two_point.py`). This test would have caught that
    bug directly (the placeholder-based tests above do not, since the
    placeholder never triggers it).
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
    )
    shear_tracer.ia = TATTContribution(
        shear_tracer,
        A1=shear_tracer.ia.A1,
        A2=shear_tracer.ia.A2,
        b_TA=shear_tracer.ia.b_TA,
        eta1=shear_tracer.ia.eta1,
        eta2=shear_tracer.ia.eta2,
        z0=shear_tracer.ia.z0,
        C_IA=shear_tracer.ia.C_IA,
        loop_computer=PBJTATTLoopComputer(linear_perturbations),
    )
    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params={
            **{f"dz_pos_{i + 1}": 0.0 for i in range(n_z_bins)},
            **{f"width_pos_{i + 1}": 1.0 for i in range(n_z_bins)},
            **{f"magnification_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
            "b1_photo_bin0": 1.1,
            "b1_photo_bin1": 1.4,
        },
    )

    ells = np.logspace(1.0, np.log10(3000), 40)  # wide range - exercises chi~0 edge
    ks = np.asarray(perturbations.k)

    she_she = AngularTwoPoint(shear_tracer, shear_tracer).get_Cl(ells, 0, ks)
    pos_she = AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells, 0, ks)

    for key, spectrum in {**she_she, **pos_she}.items():
        arr = np.asarray(spectrum.array)
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_resolves_linear_source_from_nonlinear_perturbations(
    linear_perturbations,
):
    """`PBJTATTLoopComputer` must take the *same* `perturbations` object a
    tracer already has - not a separately-tracked linear one - and resolve
    the actual linear source itself via `.linearperturbations` when present
    (every nonlinear backend built from a separate linear one sets it:
    `HMemuNonLinearPerturbations`, `EE2NonLinearPerturbations`,
    `BACCOemuNonLinearPerturbations`, `EmantisFofrNonLinearPerturbations`,
    `JAXNonLinearPerturbations`), falling back to using the given object
    directly when it has no such attribute (i.e. it's already linear).
    Uses lightweight stand-ins rather than a real nonlinear backend so this
    stays fast and independent of which cosmology backends happen to be
    installed.
    """

    class _FakeNonLinear:
        def __init__(self, linear):
            self.linearperturbations = linear

    computer_with_nonlinear = PBJTATTLoopComputer(_FakeNonLinear(linear_perturbations))
    assert computer_with_nonlinear.linear_perturbations is linear_perturbations

    computer_with_linear_directly = PBJTATTLoopComputer(linear_perturbations)
    assert computer_with_linear_directly.linear_perturbations is linear_perturbations
