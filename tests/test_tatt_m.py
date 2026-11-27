"""Tests for TATT-M (Herle et al. 2026, arXiv:2601.15851): TATT with
per-bin amplitudes set by halo mass and red-galaxy fraction.

`cosmo_setup`/`linear_perturbations` come from `conftest.py`.
"""

import importlib.util

import numpy as np
import pytest

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import IntrinsicAlignmentContribution
from cloelib.observables.photo.shear import (
    TATTMContribution,
    _growth_factor_1d,
    _tatt_m_per_bin_amplitudes,
    _tatt_m_term,
)
from cloelib.observables.photo.spectrum_engine import SpectraBank, get_pk_terms
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from test_tatt import _StubTATTLoopComputer, _shear_nuisance

_FASTPT_INSTALLED = importlib.util.find_spec("fastpt") is not None
if _FASTPT_INSTALLED:
    from cloelib.observables.photo.shear import PBJTATTLoopComputer


def _tatt_m_nuisance(n_z_bins, log10_mh, f_r, **extra):
    """`_shear_nuisance` plus TATT-M's global and per-bin keys."""
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


def _dndz(n_z_bins, z):
    dndz = np.ones((n_z_bins, len(z)))
    return dndz / np.trapezoid(dndz, z, axis=1)[:, None]


def _pos_nuisance(n_z_bins, **extra):
    return {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(n_z_bins)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(n_z_bins)},
        **{f"magnification_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
        **extra,
    }


def _with_c2_times_growth(tracer):
    """Give a plain-TATT tracer TATT-M's `A2(z) = A2 * D(z)` convention."""
    ia, c2 = tracer.ia, tracer.ia._C2
    ia._C2 = lambda zs: c2(zs) * _growth_factor_1d(
        ia._loop_computer.linear_perturbations, zs
    )
    return tracer


def _collapsed_pair(perturbations, dndz, z, n_z_bins):
    """A TATT-M tracer with identical bins and the equivalent TATT tracer."""
    alpha_M, beta_M, k1, k2, k3, log10_M0 = 1.0, 0.5, 0.073, 0.268, -0.038, 13.5
    log10_mh, f_r = 14.0, 1.0
    delta = log10_mh - log10_M0
    A1 = alpha_M * 10.0 ** (delta * beta_M)
    tatt_m = _tatt_m_tracer(
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
    tatt = _with_c2_times_growth(
        ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            nuisance_params=_shear_nuisance(
                n_z_bins, AIA=A1, A2IA=k1 * A1, bTA=k2 * delta - k3, EtaIA=0.0
            ),
            ia_model="TATT",
            tatt_loop_computer=_StubTATTLoopComputer(perturbations),
        )
    )
    return tatt_m, tatt


def test_tatt_m_per_bin_amplitudes_match_hand_formula():
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
    expected_a1 = f_r * alpha_M * 10.0 ** (delta * beta_M)
    np.testing.assert_allclose(np.asarray(a1), expected_a1, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(a2), k1 * expected_a1, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ad1), (k2 * delta - k3) * expected_a1, rtol=1e-12
    )


def test_ia_model_tatt_m_builds_tatt_m_contribution(cosmo_setup):
    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    assert isinstance(tracer.ia, TATTMContribution)
    assert isinstance(tracer.ia, IntrinsicAlignmentContribution)
    assert tracer.get_contributions() == (tracer.lensing, tracer.ia)


def test_get_ia_effective_spectra_bin_pair_matches_collapsed_tatt_z(cosmo_setup):
    """With identical bins, the per-bin-pair grids equal TATT's exactly."""
    perturbations, z = cosmo_setup
    tatt_m, tatt = _collapsed_pair(perturbations, _dndz(1, z), z, 1)

    ks = np.logspace(-3, 1, 20)
    zs = perturbations.z
    spectra_m = tatt_m.get_ia_effective_spectra(ks=ks, zs=zs, bin_i=1, bin_j=1)
    spectra_z = tatt.get_ia_effective_spectra(ks=ks, zs=zs)

    for key in ("II", "deltaI"):
        np.testing.assert_allclose(
            np.asarray(spectra_m[key]), np.asarray(spectra_z[key]), rtol=1e-10
        )


def test_get_ia_effective_spectra_bin_pair_differs_across_bins(cosmo_setup):
    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    ks = np.logspace(-3, 1, 20)
    zs = perturbations.z
    spectra_11 = tracer.get_ia_effective_spectra(ks=ks, zs=zs, bin_i=1, bin_j=1)
    spectra_12 = tracer.get_ia_effective_spectra(ks=ks, zs=zs, bin_i=1, bin_j=2)

    assert np.all(np.isfinite(np.asarray(spectra_11["II"])))
    assert not np.allclose(np.asarray(spectra_11["II"]), np.asarray(spectra_12["II"]))


def test_get_ia_effective_spectra_bin_args_required_or_forbidden(cosmo_setup):
    """bin_i/bin_j are required for TATT-M and rejected for TATT/NLA."""
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)

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
    tatt_z_tracer.get_ia_effective_spectra()

    nla_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    with pytest.raises(ValueError, match="defines an effective power spectrum"):
        nla_tracer.get_ia_effective_spectra()


def test_tatt_m_requirements_pruned(cosmo_setup):
    """II needs the 7 E-mode kernels (no B-mode ones); GI only the 4 GI ones."""
    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations, _dndz(1, z), z, 1, log10_mh=[13.5], f_r=[1.0]
    )
    ia = tracer.ia

    ii = {r.name for r in ia.get_requirements_for_interaction(ia)}
    gi = {r.name for r in ia.get_requirements_for_interaction(tracer.lensing)}

    assert gi == {"tatt_A_0_0E", "tatt_C_0_0E", "tatt_A_0_E2", "tatt_B_0_E2"}
    assert ii == gi | {"tatt_A_0E_0E", "tatt_A_E2_E2", "tatt_D_0E_E2"}


def test_tatt_m_kernels_computed_once_per_cl(cosmo_setup):
    """Each one-loop kernel is computed once per `get_Cl`, however many
    terms use it."""
    perturbations, z = cosmo_setup
    call_counts = {}

    class _CountingLoopComputer(_StubTATTLoopComputer):
        def compute(self, name):
            base_compute = super().compute(name)

            def _compute(matter_pk, ks, zs):
                call_counts[name] = call_counts.get(name, 0) + 1
                return base_compute(matter_pk, ks, zs)

            return _compute

    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=_dndz(1, z),
        z=z,
        nuisance_params=_tatt_m_nuisance(1, log10_mh=[13.5], f_r=[1.0]),
        ia_model="TATT-M",
        tatt_loop_computer=_CountingLoopComputer(perturbations),
    )
    ells = np.logspace(1.0, np.log10(200), 6)
    AngularTwoPoint(tracer, tracer).get_Cl(ells, 0, np.asarray(perturbations.k))

    assert call_counts
    assert all(count == 1 for count in call_counts.values()), call_counts


def test_tatt_m_derived_terms_cached_in_bank(cosmo_setup):
    """Derived terms are memoized in the bank, so pairings share one array
    (and the engine interpolates it once)."""
    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations, _dndz(1, z), z, 1, log10_mh=[14.0], f_r=[1.0]
    )
    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    bank = SpectraBank(perturbations.matter_power_spectrum(pert_zs, ks), ks, pert_zs)

    first = _tatt_m_term(bank, tracer.ia._loop_computer, "00e")
    assert first is _tatt_m_term(bank, tracer.ia._loop_computer, "00e")
    assert "tatt_m_00e" in bank._grids


def test_tatt_m_collapses_to_tatt_z_when_bins_identical(cosmo_setup):
    """With identical bins, TATT-M's Cl reproduces TATT's (eta=0, C2 times
    D(z)). Not exact: TATT interpolates its summed P_II once, TATT-M
    interpolates each term separately (and P_dd log-log) before summing."""
    perturbations, z = cosmo_setup
    tatt_m, tatt = _collapsed_pair(perturbations, _dndz(2, z), z, 2)
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    cl_tatt_m = AngularTwoPoint(tatt_m, tatt_m).get_Cl_tensor(ells, 0, ks)
    cl_tatt_z = AngularTwoPoint(tatt, tatt).get_Cl_tensor(ells, 0, ks)

    np.testing.assert_allclose(np.asarray(cl_tatt_m), np.asarray(cl_tatt_z), rtol=2e-2)


def test_tatt_m_c2_over_c1_is_redshift_independent(cosmo_setup):
    """`A2^i(z) = k1*D(z)*A1^i` gives `C2/C1 = -5*k1` at every bin and z."""
    perturbations, z = cosmo_setup
    k1 = 0.073
    tracer = _tatt_m_tracer(
        perturbations,
        _dndz(3, z),
        z,
        3,
        log10_mh=[13.0, 13.5, 14.2],
        f_r=[1.0, 0.5, 0.3],
        k1=k1,
    )
    c1, _, c2 = tracer.ia._C(perturbations.z)
    np.testing.assert_allclose(np.asarray(c2) / np.asarray(c1), -5 * k1, rtol=1e-12)


def test_tatt_m_f_r_zero_switches_off_bin(cosmo_setup):
    """f_r=0 zeroes every IA amplitude of that bin."""
    perturbations, z = cosmo_setup
    tracer = _tatt_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[14.0, 14.0], f_r=[0.0, 1.0]
    )
    c1, c1d, c2 = (np.asarray(c) for c in tracer.ia._C(z))
    for c in (c1, c1d, c2):
        np.testing.assert_allclose(c[0], 0.0)
    assert np.any(c1[1] != 0.0)


def test_tatt_m_halo_mass_scaling():
    """A1 scales as (Mh/M0)^betaM and bTA linearly in log10(Mh/M0)."""
    alpha_M, beta_M, k1, k2, k3, log10_M0 = 2.0, 0.7, 0.073, 0.268, -0.038, 13.5
    log10_mh = np.array([13.0, 13.5, 14.0, 14.5])
    nuisance = {
        **{f"log10_Mh_{i + 1}": v for i, v in enumerate(log10_mh)},
        **{f"f_r_{i + 1}": 1.0 for i in range(len(log10_mh))},
    }
    a1, _, ad1 = _tatt_m_per_bin_amplitudes(
        nuisance, len(log10_mh), alpha_M, beta_M, k1, k2, k3, log10_M0
    )
    A1 = np.asarray(a1)
    delta = log10_mh - log10_M0
    np.testing.assert_allclose(A1, alpha_M * 10.0 ** (delta * beta_M), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(ad1) / A1, k2 * delta - k3, rtol=1e-10)


def test_tatt_m_cross_model_partner_raises(cosmo_setup):
    """Pairing TATT-M with NLA or TATT raises, in either order."""
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)
    tatt_m = _tatt_m_tracer(perturbations, dndz, z, 1, log10_mh=[13.5], f_r=[1.0])
    nla = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    tatt = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1, A2IA=0.4, bTA=-0.83),
        ia_model="TATT",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )
    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    bank = SpectraBank(perturbations.matter_power_spectrum(pert_zs, ks), ks, pert_zs)

    for other in (nla.ia, tatt.ia):
        with pytest.raises(ValueError, match="cannot be paired with"):
            get_pk_terms(tatt_m.ia, other, z, bank)
        with pytest.raises(ValueError, match="cannot be paired with"):
            get_pk_terms(other, tatt_m.ia, z, bank)


def test_tatt_m_with_nonlinear_bias_uses_tatt_m_gi_terms(cosmo_setup):
    """Against `galaxy_bias_model="nonlinear"`, the GI cross comes from
    TATT-M's own terms with the b1 kernel, whichever side is asked first."""
    perturbations, z = cosmo_setup

    class _StubNLBias:
        def compute(self, name):
            return lambda matter_pk, ks, zs: 1e-2 * matter_pk[0]

    tatt_m = _tatt_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    pos = PositionsTracer(
        perturbations=perturbations,
        dndz=_dndz(2, z),
        z=z,
        galaxy_bias_model="nonlinear",
        nuisance_params=_pos_nuisance(
            2, b1_photo_nl_bin0=1.7, b1_photo_nl_bin1=2.1, b2_photo_nl_bin0=0.3
        ),
        nl_bias_loop_computer=_StubNLBias(),
    )
    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    bank = SpectraBank(perturbations.matter_power_spectrum(pert_zs, ks), ks, pert_zs)

    own = tatt_m.ia.get_pk_terms(pos.bias, z, bank)
    resolved = get_pk_terms(pos.bias, tatt_m.ia, z, bank)
    assert len(resolved) == len(own) == 3
    for r, o in zip(resolved, own):
        assert r.kernel1 is None  # -> the b1 kernel of `pos.bias`
        assert r.kernel2 is o.kernel1
        assert r.pk is o.pk

    ells = np.logspace(1.0, np.log10(200), 6)
    cl = AngularTwoPoint(pos, tatt_m).get_Cl_tensor(
        ells, 0, np.asarray(perturbations.k)
    )
    assert np.all(np.isfinite(np.asarray(cl)))


def test_tatt_m_no_speed_regression_vs_tatt(cosmo_setup):
    """Coarse guard: warmed-up TATT-M is not an order of magnitude slower
    than TATT."""
    import time

    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    tatt_m_tracer = _tatt_m_tracer(
        perturbations, dndz, z, n_z_bins, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
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
    """End-to-end `get_Cl` for SHE-SHE and POS-SHE with TATT-M."""
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)

    shear_tracer = _tatt_m_tracer(
        perturbations, dndz, z, n_z_bins, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_bin0=1.1, b1_photo_bin1=1.4),
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    she_she = AngularTwoPoint(shear_tracer, shear_tracer).get_Cl(ells, 0, ks)
    pos_she = AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells, 0, ks)

    assert ("SHE", "SHE", 1, 1) in she_she
    for key, spectrum in {**she_she, **pos_she}.items():
        assert np.all(np.isfinite(np.asarray(spectrum.array))), key

    assert she_she[("SHE", "SHE", 1, 1)].array.shape == (2, 2, len(ells))
    assert pos_she[("POS", "SHE", 1, 1)].array.shape == (2, len(ells))


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_tatt_m_generalized_cl_finite_with_real_fastpt(
    cosmo_setup, linear_perturbations
):
    """Real FAST-PT kernels over a wide ell range (reaching chi -> 0) give
    finite Cl, as `test_pbj_tatt_generalized_cl_finite_with_real_fastpt`
    checks for TATT."""
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)

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
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_bin0=1.1, b1_photo_bin1=1.4),
    )

    ells = np.logspace(1.0, np.log10(3000), 40)
    ks = np.asarray(perturbations.k)

    she_she = AngularTwoPoint(shear_tracer, shear_tracer).get_Cl(ells, 0, ks)
    pos_she = AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells, 0, ks)

    for key, spectrum in {**she_she, **pos_she}.items():
        assert np.all(np.isfinite(np.asarray(spectrum.array))), key
