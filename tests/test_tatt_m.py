"""Tests for TATT-M (Herle et al. 2026, arXiv:2601.15851): per-bin IA
amplitude tied to halo mass/red fraction, built as three rank-1
"component" Contributions on top of the existing TATT one-loop machinery -
see `TATTMContribution`'s docstring in
`cloelib/observables/photo/shear.py` for why.

`cosmo_setup`/`linear_perturbations` (used throughout as fixture
parameters) come from `conftest.py`, shared with `test_tatt.py` - no
import needed. `_StubTATTLoopComputer`/`_shear_nuisance` are reused from
`test_tatt.py` directly (plain helpers, not fixtures, so a normal import
works for them).
"""

import importlib.util

import numpy as np
import pytest

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import IntrinsicAlignmentContribution
from cloelib.observables.photo.shear import (
    TATTMContribution,
    _TATTMTaContribution,
    _TATTMTbContribution,
    _TATTMTcContribution,
    _tatt_m_per_bin_amplitudes,
)
from cloelib.observables.photo.spectrum_engine import SpectraBank
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from test_tatt import _StubTATTLoopComputer, _shear_nuisance

_FASTPT_INSTALLED = importlib.util.find_spec("fastpt") is not None
if _FASTPT_INSTALLED:
    from cloelib.observables.photo.shear import PBJTATTLoopComputer


def _tatt_m_nuisance(n_z_bins, log10_mh, f_r, **extra):
    """`_shear_nuisance` plus TATT-M's own global + per-bin keys.

    `log10_mh`/`f_r` are length-`n_z_bins` sequences, one value per
    tomographic bin (1-indexed keys, matching `ShearTracer`'s own
    `multiplicative_bias_i`/`dz_shear_i` convention).
    """
    base = _shear_nuisance(n_z_bins)
    base.update(
        {
            "alphaM": 1.0,
            "betaM": 0.5,
            "k1": 0.073,
            "k2": 0.268,
            **{f"log10_Mh_{i + 1}": log10_mh[i] for i in range(n_z_bins)},
            **{f"f_r_{i + 1}": f_r[i] for i in range(n_z_bins)},
        }
    )
    base.update(extra)
    return base


def _tatt_m_tracer(perturbations, dndz, z, n_z_bins, log10_mh, f_r, **extra):
    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_tatt_m_nuisance(n_z_bins, log10_mh, f_r, **extra),
        ia_model="TATT-M",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )


def test_tatt_m_per_bin_amplitudes_match_hand_formula():
    """Direct arithmetic check of the helper, independent of ShearTracer."""
    nuisance = {
        "log10_Mh_1": 13.5,
        "f_r_1": 1.0,
        "log10_Mh_2": 14.0,
        "f_r_2": 0.5,
        "log10_Mh_3": 12.5,
        "f_r_3": 0.0,
    }
    alpha_M, beta_M, k1, k2, k3, log10_M0 = 1.0, 0.5, 0.073, 0.268, -0.038, 13.5

    a1, a2, ad1 = _tatt_m_per_bin_amplitudes(
        nuisance, 3, alpha_M, beta_M, k1, k2, k3, log10_M0
    )

    log10_mh = np.array([13.5, 14.0, 12.5])
    f_r = np.array([1.0, 0.5, 0.0])
    delta = log10_mh - log10_M0
    expected_A1 = alpha_M * 10.0 ** (delta * beta_M)
    expected_A2 = k1 * expected_A1
    expected_bTA = k2 * delta - k3
    expected_a1 = f_r * expected_A1
    expected_a2 = f_r * expected_A2
    expected_ad1 = expected_bTA * expected_a1

    np.testing.assert_allclose(np.asarray(a1), expected_a1, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(a2), expected_a2, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ad1), expected_ad1, rtol=1e-12)


def test_ia_model_tatt_m_builds_tatt_m_contribution(cosmo_setup):
    """`ia_model="TATT-M"` must build a `TATTMContribution` (itself an
    `IntrinsicAlignmentContribution`, like TATT/NLA) wrapping exactly one
    `_TATTMTa/Tb/TcContribution` each - the three rank-1 components the
    coordinator is built from.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((2, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    tracer = _tatt_m_tracer(
        perturbations, dndz, z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    assert isinstance(tracer.ia, TATTMContribution)
    assert isinstance(tracer.ia, IntrinsicAlignmentContribution)
    assert isinstance(tracer.ia.ta, _TATTMTaContribution)
    assert isinstance(tracer.ia.tb, _TATTMTbContribution)
    assert isinstance(tracer.ia.tc, _TATTMTcContribution)


def test_tatt_m_get_contributions_flattens_components(cosmo_setup):
    """`get_contributions()` must flatten TATT-M's 3 components in, but
    leave TATT/NLA (which don't define `get_components`) unchanged.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((2, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    tatt_m_tracer = _tatt_m_tracer(
        perturbations, dndz, z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    contribs = tatt_m_tracer.get_contributions()
    assert contribs == (
        tatt_m_tracer.lensing,
        tatt_m_tracer.ia.ta,
        tatt_m_tracer.ia.tb,
        tatt_m_tracer.ia.tc,
    )

    tatt_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(2, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )
    assert tatt_tracer.get_contributions() == (tatt_tracer.lensing, tatt_tracer.ia)


def test_get_ia_effective_spectra_bin_pair_matches_collapsed_tatt_z(cosmo_setup):
    """`get_ia_effective_spectra(bin_i, bin_j)`'s per-bin-pair reconstruction
    must match `TATTContribution`'s own (already-validated) formula exactly
    when every bin is engineered to share the same (A1, A2, bTA) - this
    compares the raw effective-Pk grids directly (no Limber interpolation
    involved, unlike the Cl-level collapse test), so the tolerance can be
    tight.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    alpha_M, beta_M, k1, k2, k3, log10_M0 = 1.0, 0.5, 0.073, 0.268, -0.038, 13.5
    log10_mh, f_r = 14.0, 1.0
    delta = log10_mh - log10_M0
    A1 = alpha_M * 10.0 ** (delta * beta_M)
    A2 = k1 * A1
    b_ta = k2 * delta - k3

    tatt_m_tracer = _tatt_m_tracer(
        perturbations,
        dndz,
        z,
        1,
        log10_mh=[log10_mh],
        f_r=[f_r],
        alphaM=alpha_M,
        betaM=beta_M,
        k1=k1,
        k2=k2,
        k3=k3,
        log10_M0=log10_M0,
    )
    tatt_z_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=A2, bTA=b_ta, EtaIA=0.0, AIA=A1),
        ia_model="TATT",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )

    ks = np.logspace(-3, 1, 20)
    zs = perturbations.z
    spectra_m = tatt_m_tracer.get_ia_effective_spectra(ks=ks, zs=zs, bin_i=1, bin_j=1)
    spectra_z = tatt_z_tracer.get_ia_effective_spectra(ks=ks, zs=zs)

    np.testing.assert_allclose(
        np.asarray(spectra_m["II"]), np.asarray(spectra_z["II"]), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(spectra_m["deltaI"]), np.asarray(spectra_z["deltaI"]), rtol=1e-10
    )


def test_get_ia_effective_spectra_bin_pair_differs_across_bins(cosmo_setup):
    """A genuinely different pair (bin 1 vs bin 2, different Mh/f_r) must
    give a different P_II than the (1,1) auto-pair - the whole point of
    needing bin_i/bin_j at all.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((2, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    tracer = _tatt_m_tracer(
        perturbations, dndz, z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    ks = np.logspace(-3, 1, 20)
    zs = perturbations.z
    spectra_11 = tracer.get_ia_effective_spectra(ks=ks, zs=zs, bin_i=1, bin_j=1)
    spectra_12 = tracer.get_ia_effective_spectra(ks=ks, zs=zs, bin_i=1, bin_j=2)

    assert np.all(np.isfinite(np.asarray(spectra_11["II"])))
    assert not np.allclose(np.asarray(spectra_11["II"]), np.asarray(spectra_12["II"]))


def test_get_ia_effective_spectra_bin_args_required_or_forbidden(cosmo_setup):
    """bin_i/bin_j must be given for TATT-M (raises otherwise) and must be
    omitted for TATT/NLA (raises if given) - checked both ways.
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    tatt_m_tracer = _tatt_m_tracer(
        perturbations, dndz, z, 1, log10_mh=[13.5], f_r=[1.0]
    )
    with pytest.raises(ValueError, match="needs bin_i and bin_j"):
        tatt_m_tracer.get_ia_effective_spectra()
    with pytest.raises(ValueError, match="needs bin_i and bin_j"):
        tatt_m_tracer.get_ia_effective_spectra(bin_i=1)

    tatt_z_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )
    with pytest.raises(ValueError, match="doesn't take bin_i/bin_j"):
        tatt_z_tracer.get_ia_effective_spectra(bin_i=1, bin_j=1)
    tatt_z_tracer.get_ia_effective_spectra()  # no bin args: still works

    nla_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    with pytest.raises(ValueError, match="defines an effective power spectrum"):
        nla_tracer.get_ia_effective_spectra()


def test_tatt_m_requirements_pruned_per_component(cosmo_setup):
    """Union across all 3 components must be exactly the 7 kernels TATT-M's
    formula actually reads (not all 10 - the 3 B-mode-only ones,
    `tatt_A_0B_0B`/`tatt_A_B2_B2`/`tatt_D_0B_B2`, are never consumed by any
    component's `get_effective_pk`).
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    tracer = _tatt_m_tracer(perturbations, dndz, z, 1, log10_mh=[13.5], f_r=[1.0])
    ta, tb, tc = tracer.ia.ta, tracer.ia.tb, tracer.ia.tc

    ta_ii = {r.name for r in ta.get_requirements_for_interaction(ta)}
    ta_gi = {r.name for r in ta.get_requirements_for_interaction(tracer.lensing)}
    tb_ii = {r.name for r in tb.get_requirements_for_interaction(tb)}
    tb_gi = {r.name for r in tb.get_requirements_for_interaction(tracer.lensing)}
    tc_ii = {r.name for r in tc.get_requirements_for_interaction(tc)}
    tc_gi = {r.name for r in tc.get_requirements_for_interaction(tracer.lensing)}

    assert ta_gi == set()
    assert ta_ii == {"tatt_A_0_0E", "tatt_C_0_0E", "tatt_A_0_E2", "tatt_B_0_E2"}
    assert tb_gi == {"tatt_A_0_0E", "tatt_C_0_0E"}
    assert tb_ii == tb_gi | {"tatt_A_0E_0E", "tatt_D_0E_E2"}
    assert tc_gi == {"tatt_A_0_E2", "tatt_B_0_E2"}
    assert tc_ii == tc_gi | {"tatt_D_0E_E2", "tatt_A_E2_E2"}

    union = ta_ii | tb_ii | tc_ii
    assert union == {
        "tatt_A_0_0E",
        "tatt_C_0_0E",
        "tatt_A_0E_0E",
        "tatt_A_E2_E2",
        "tatt_A_0_E2",
        "tatt_B_0_E2",
        "tatt_D_0E_E2",
    }
    assert len(union) == 7


def test_tatt_m_spectra_bank_memoized_across_components(cosmo_setup):
    """The same kernel name requested by more than one of Ta/Tb/Tc must be
    computed once, not once per component - `SpectraBank.get` checks its
    cache before calling `compute`.
    """
    perturbations, z = cosmo_setup
    call_counts = {}

    class _CountingLoopComputer(_StubTATTLoopComputer):
        def compute(self, name):
            base_compute = super().compute(name)

            def _compute(matter_pk, ks, zs):
                call_counts[name] = call_counts.get(name, 0) + 1
                return base_compute(matter_pk, ks, zs)

            return _compute

    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_tatt_m_nuisance(1, log10_mh=[13.5], f_r=[1.0]),
        ia_model="TATT-M",
        tatt_loop_computer=_CountingLoopComputer(perturbations),
    )
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    AngularTwoPoint(tracer, tracer).get_Cl(ells, 0, ks)

    assert call_counts, "expected at least one kernel to have been requested"
    assert all(count == 1 for count in call_counts.values()), call_counts


def test_tatt_m_derived_terms_cached_not_just_raw_kernels(cosmo_setup):
    """`_tatt_m_term_00e`/etc. must memoize the *combined* D(z)**4-times-
    kernel(s) grid in `bank` itself, not just the raw named FAST-PT kernels
    underneath it - otherwise every visit to a pair that needs the same
    term (e.g. both `(Ta,Tb)` and `(Tb,Ta)` in `_compute_cl_generalized`'s
    Cartesian loop) redundantly recomputes the growth factor and the
    kernel sum even though no new FAST-PT call is involved.
    """
    from cloelib.observables.photo.shear import _tatt_m_term_00e

    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations,
        np.ones((1, len(z))) / np.trapezoid(np.ones((1, len(z))), z, axis=1)[:, None],
        z,
        1,
        log10_mh=[14.0],
        f_r=[1.0],
    )
    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    matter_pk = perturbations.matter_power_spectrum(pert_zs, ks)
    bank = SpectraBank(matter_pk, ks, pert_zs)
    loop_computer = tracer.ia.ta._loop_computer

    first = _tatt_m_term_00e(bank, loop_computer)
    second = _tatt_m_term_00e(bank, loop_computer)
    assert first is second, "expected the cached array, not a freshly computed one"
    assert "tatt_m_term_00e" in bank._grids


def test_tatt_m_collapses_to_tatt_z_when_bins_identical(cosmo_setup):
    """With per-bin Mh/f_r engineered so every bin gets the SAME (A1, A2,
    bTA), TATT-M must reproduce plain TATT (eta1=eta2=0, matching TATT-M's
    lack of any (1+z)^eta factor) bin-by-bin, tightly - both go through
    `_compute_cl_generalized`, so there's no legacy-vs-generalized numerical
    gap to tolerate (unlike the existing TATT-vs-NLA comparison).
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    # Same Mh/f_r for every bin => same (A1, A2, bTA) for every bin.
    alpha_M, beta_M, k1, k2, k3, log10_M0 = 1.0, 0.5, 0.073, 0.268, -0.038, 13.5
    log10_mh = 14.0
    f_r = 1.0
    delta = log10_mh - log10_M0
    A1 = alpha_M * 10.0 ** (delta * beta_M)
    A2 = k1 * A1
    b_ta = k2 * delta - k3

    tatt_m_tracer = _tatt_m_tracer(
        perturbations,
        dndz,
        z,
        n_z_bins,
        log10_mh=[log10_mh] * n_z_bins,
        f_r=[f_r] * n_z_bins,
        alphaM=alpha_M,
        betaM=beta_M,
        k1=k1,
        k2=k2,
        k3=k3,
        log10_M0=log10_M0,
    )
    tatt_z_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, A2IA=A2, bTA=b_ta, EtaIA=0.0, AIA=A1),
        ia_model="TATT",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )

    cl_tatt_m = AngularTwoPoint(tatt_m_tracer, tatt_m_tracer).get_Cl_tensor(ells, 0, ks)
    cl_tatt_z = AngularTwoPoint(tatt_z_tracer, tatt_z_tracer).get_Cl_tensor(ells, 0, ks)

    # rtol is not machine-precision-tight despite both paths going through
    # the *same* generalized engine: TATT-M's 9 component pairs (Ta/Tb/Tc x
    # Ta/Tb/Tc) each get their own `Pkl_interp_signed_vmap` call (signed-log
    # Akima interpolation - a nonlinear operation) *before* being summed by
    # `Cl_integration`, whereas TATT-Z's single `TATTContribution` sums all
    # terms into one P_II/P_deltaI grid *first* and interpolates that sum
    # once. "Interpolate-then-sum" and "sum-then-interpolate" are not
    # identical once a nonlinear (log-space) interpolation sits in between -
    # empirically ~1.6% here, an expected footprint of this design (see
    # TATTMContribution's docstring), not a bug.
    np.testing.assert_allclose(np.asarray(cl_tatt_m), np.asarray(cl_tatt_z), rtol=2e-2)


def test_tatt_m_f_r_scaling(cosmo_setup):
    """f_r=0 for one bin must zero every IA contribution from that bin:
    that bin's II auto-spectrum and its GI cross with lensing/density both
    vanish (down to whatever raw matter-Pk baseline the lensing-lensing
    term itself contributes - checked here directly on `get_effective_pk`
    /`compute_kernel`, not the full Cl, to isolate the claim).
    """
    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations,
        np.ones((2, len(z))) / np.trapezoid(np.ones((2, len(z))), z, axis=1)[:, None],
        z,
        2,
        log10_mh=[14.0, 14.0],
        f_r=[0.0, 1.0],
    )
    kernel_ta = np.asarray(tracer.ia.ta.compute_kernel(z))
    kernel_tb = np.asarray(tracer.ia.tb.compute_kernel(z))
    kernel_tc = np.asarray(tracer.ia.tc.compute_kernel(z))
    # Bin 0 (f_r=0) must have a zero IA kernel; bin 1 (f_r=1) must not.
    np.testing.assert_allclose(kernel_ta[0], 0.0)
    np.testing.assert_allclose(kernel_tb[0], 0.0)
    np.testing.assert_allclose(kernel_tc[0], 0.0)
    assert np.any(kernel_ta[1] != 0.0)


def test_tatt_m_halo_mass_scaling(cosmo_setup):
    """A1^i must scale as (Mh^i/M0)^betaM, and bTA^i linearly in
    log10(Mh^i/M0) - checked via the per-bin amplitude helper directly.
    """
    alpha_M, beta_M, k1, k2, k3, log10_M0 = 2.0, 0.7, 0.073, 0.268, -0.038, 13.5
    log10_mh = np.array([13.0, 13.5, 14.0, 14.5])
    f_r = np.ones_like(log10_mh)
    nuisance = {
        **{f"log10_Mh_{i + 1}": v for i, v in enumerate(log10_mh)},
        **{f"f_r_{i + 1}": v for i, v in enumerate(f_r)},
    }
    a1, a2, ad1 = _tatt_m_per_bin_amplitudes(
        nuisance, len(log10_mh), alpha_M, beta_M, k1, k2, k3, log10_M0
    )
    A1 = np.asarray(a1)  # f_r=1 everywhere, so a1 == A1
    delta = log10_mh - log10_M0
    expected_A1 = alpha_M * 10.0 ** (delta * beta_M)
    np.testing.assert_allclose(A1, expected_A1, rtol=1e-10)

    bTA = np.asarray(ad1) / A1  # ad1 = bTA * a1 = bTA * A1 here
    expected_bTA = k2 * delta - k3
    np.testing.assert_allclose(bTA, expected_bTA, rtol=1e-10)


def test_tatt_m_cross_model_partner_raises(cosmo_setup):
    """Pairing a TATT-M component against a legacy `TATTContribution`/
    `NLAContribution` must raise - not silently reuse the wrong amplitude
    (see `_tatt_m_cross_model_error`'s docstring for the caveat about
    pairing order with a legacy `TATTContribution` as the first argument).
    """
    perturbations, z = cosmo_setup
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    tatt_m_tracer = _tatt_m_tracer(
        perturbations, dndz, z, 1, log10_mh=[13.5], f_r=[1.0]
    )
    nla_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    matter_pk = perturbations.matter_power_spectrum(pert_zs, ks)
    bank = SpectraBank(matter_pk, ks, pert_zs)

    with pytest.raises(ValueError, match="cannot be paired with"):
        tatt_m_tracer.ia.ta.get_effective_pk(nla_tracer.ia, bank)
    with pytest.raises(ValueError, match="cannot be paired with"):
        tatt_m_tracer.ia.tb.get_effective_pk(nla_tracer.ia, bank)
    with pytest.raises(ValueError, match="cannot be paired with"):
        tatt_m_tracer.ia.tc.get_effective_pk(nla_tracer.ia, bank)


def test_tatt_m_no_speed_regression_vs_tatt(cosmo_setup):
    """TATT-M (3 components) must not be dramatically slower than plain
    TATT (1 contribution) for the same bin count, once both are warmed up
    (post-jit-compile) - a coarse order-of-magnitude guard, not a tight
    performance benchmark (no existing timing-test precedent in this repo
    to match tighter tolerances against).
    """
    import time

    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    tatt_m_tracer = _tatt_m_tracer(
        perturbations,
        dndz,
        z,
        n_z_bins,
        log10_mh=[13.5, 14.0],
        f_r=[1.0, 0.5],
    )
    tatt_z_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )

    def _timed_call(tracer, n_reps=3):
        ap = AngularTwoPoint(tracer, tracer)
        ap.get_Cl(ells, 0, ks)  # warm-up / jit-compile
        start = time.perf_counter()
        for _ in range(n_reps):
            ap.get_Cl(ells, 0, ks)
        return (time.perf_counter() - start) / n_reps

    t_tatt_m = _timed_call(tatt_m_tracer)
    t_tatt_z = _timed_call(tatt_z_tracer)

    assert t_tatt_m < 10 * t_tatt_z + 1e-3, (t_tatt_m, t_tatt_z)


def test_tatt_m_generalized_cl_finite_and_correctly_shaped_with_positions(cosmo_setup):
    """End-to-end, packaged-`get_Cl` regression, mirroring `test_tatt.py`'s
    `test_generalized_cl_finite_and_correctly_shaped` (which only exercises
    TATT-z): ShearTracer(ia_model="TATT-M") x PositionsTracer must produce
    finite SHE-SHE *and* POS-SHE output with the expected shape contract.

    Nothing in the memoization/collapse/jit tests above actually runs
    TATT-M's GI term (`_TATTMTb`/`_TATTMTc` paired against a real
    galaxy-bias contribution) through the full dict-keyed `get_Cl` pipeline
    - this is the one place that combination is checked.
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    shear_tracer = _tatt_m_tracer(
        perturbations,
        dndz,
        z,
        n_z_bins,
        log10_mh=[13.5, 14.0],
        f_r=[1.0, 0.5],
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


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_m_generalized_cl_finite_with_real_fastpt(
    cosmo_setup, linear_perturbations
):
    """Same real-FAST-PT, wide-ell-range numerical-stability check as
    `test_tatt.py`'s `test_pbj_tatt_generalized_cl_finite_with_real_fastpt`,
    but for TATT-M specifically.

    That TATT-z test caught a real bug: real FAST-PT kernels are steep and
    sign-changing near their k-grid's edges, and at high ell / low z (chi ->
    0) the Limber k_l = (ell+0.5)/chi grid reaches past that k-grid, so a
    naive akima extrapolation there overflowed to +-inf and produced NaN Cl
    - fixed by clamping the query k to the grid's own domain
    (`Pkl_interp_signed`). TATT-z's fix lives in shared code
    (`Pkl_interp_signed_vmap`), so it protects TATT-M too in principle - but
    TATT-M reaches that code 9 separate times per `Cl` (once per
    Ta/Tb/Tc x Ta/Tb/Tc component pair, each with its own interpolated
    grid) rather than TATT-z's single pre-summed one, so this exercises the
    real kernels through that multi-grid path directly rather than assuming
    the fix transfers.
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = np.ones((n_z_bins, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_tatt_m_nuisance(
            n_z_bins, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
        ),
        ia_model="TATT-M",
        tatt_loop_computer=PBJTATTLoopComputer(linear_perturbations),
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
