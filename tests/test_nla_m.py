"""Tests for NLA-M (Wright et al. 2025, arXiv:2503.19441): the A2 = bTA = 0
limit of TATT-M, i.e. NLA with per-bin amplitudes set by halo mass and
red-galaxy fraction.

`cosmo_setup`/`linear_perturbations` come from `conftest.py`.
"""

import jax
import numpy as np
import pytest

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import IntrinsicAlignmentContribution
from cloelib.observables.photo.shear import (
    NLAMContribution,
    _growth_factor_1d,
    _nla_m_per_bin_amplitudes,
)
from cloelib.observables.photo.spectrum_engine import (
    SpectraBank,
    get_pk_terms,
    needs_generalized_engine,
)
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from test_nonlinear_bias import _StubNLBiasLoopComputer
from test_tatt import _StubTATTLoopComputer, _shear_nuisance
from test_tatt_m import _dndz, _pos_nuisance, _tatt_m_nuisance


def _nla_m_nuisance(n_z_bins, log10_mh, f_r, **extra):
    """`_shear_nuisance` plus NLA-M's global and per-bin keys.

    Unlike `_tatt_m_nuisance`, no k1/k2/k3: NLA-M does not read them. The
    NLA keys `_shear_nuisance` sets (`AIA`, `EtaIA`) are ignored by NLA-M.
    """
    base = _shear_nuisance(n_z_bins)
    base.update(
        {
            "alphaM": 1.0,
            "betaM": 0.5,
            **{f"log10_Mh_{i + 1}": log10_mh[i] for i in range(n_z_bins)},
            **{f"f_r_{i + 1}": f_r[i] for i in range(n_z_bins)},
        }
    )
    base.update(extra)
    return base


def _nla_m_tracer(perturbations, dndz, z, n_z_bins, log10_mh, f_r, **extra):
    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_nla_m_nuisance(n_z_bins, log10_mh, f_r, **extra),
        ia_model="NLA-M",
    )


def _tatt_m_zero_higher_order_tracer(
    perturbations, loop_perturbations, dndz, z, n_z_bins, log10_mh, f_r
):
    """TATT-M with k1 = k2 = k3 = 0, i.e. A2^i = bTA^i = 0."""
    return ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_tatt_m_nuisance(
            n_z_bins, log10_mh, f_r, k1=0.0, k2=0.0, k3=0.0
        ),
        ia_model="TATT-M",
        tatt_loop_computer=_StubTATTLoopComputer(loop_perturbations),
    )


def _bank(perturbations):
    ks = np.logspace(-3, 1, 20)
    zs = perturbations.z
    return SpectraBank(perturbations.matter_power_spectrum(zs, ks), ks, zs)


def test_nla_m_per_bin_amplitudes_match_hand_formula():
    nuisance = {
        "log10_Mh_1": 13.5,
        "f_r_1": 1.0,
        "log10_Mh_2": 14.0,
        "f_r_2": 0.5,
        "log10_Mh_3": 12.5,
        "f_r_3": 0.0,
    }
    alpha_M, beta_M, log10_M0 = 2.0, 0.7, 13.5

    a1 = _nla_m_per_bin_amplitudes(nuisance, 3, alpha_M, beta_M, log10_M0)

    log10_mh = np.array([13.5, 14.0, 12.5])
    f_r = np.array([1.0, 0.5, 0.0])
    delta = log10_mh - log10_M0
    expected = f_r * alpha_M * 10.0 ** (delta * beta_M)
    np.testing.assert_allclose(np.asarray(a1), expected, rtol=1e-12)


def test_ia_model_nla_m_builds_nla_m_contribution(cosmo_setup):
    """NLA-M is a plain kernel model: no extra spectra, so the engine never
    needs one-loop kernels for it."""
    perturbations, z = cosmo_setup
    tracer = _nla_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    assert isinstance(tracer.ia, NLAMContribution)
    assert isinstance(tracer.ia, IntrinsicAlignmentContribution)
    assert tracer.get_contributions() == (tracer.lensing, tracer.ia)
    assert not needs_generalized_engine(
        tracer.get_contributions(), tracer.get_contributions()
    )


def test_nla_m_rejects_tatt_loop_computer(cosmo_setup):
    """`tatt_loop_computer` is only accepted with ia_model='TATT' or
    'TATT-M'. NLA-M needs no one-loop kernels, so passing one raises, as
    it does for NLA. `_StubTATTLoopComputer` is the fake loop computer the
    TATT/TATT-M tests use instead of the real FAST-PT one; any non-None
    object would trigger the same error."""
    perturbations, z = cosmo_setup
    with pytest.raises(ValueError, match="only used when ia_model='TATT' or 'TATT-M'"):
        ShearTracer(
            perturbations=perturbations,
            dndz=_dndz(1, z),
            z=z,
            nuisance_params=_nla_m_nuisance(1, [13.5], [1.0]),
            ia_model="NLA-M",
            tatt_loop_computer=_StubTATTLoopComputer(perturbations),
        )


def _nla_growth(perturbations, z):
    """D(z) as NLA and NLA-M take it: second k-grid column if k-dependent."""
    d_raw = np.asarray(
        perturbations.growth_factor(z, getattr(perturbations, "k", None))
    )
    return d_raw[:, 1] if d_raw.ndim == 2 else d_raw


def test_nla_m_kernel_matches_tatt_m_c1_kernel(cosmo_setup, linear_perturbations):
    """NLA-M's kernel is TATT-M's C1-weighted kernel, and TATT-M's C1delta
    and C2 kernels vanish when k1 = k2 = k3 = 0.

    The two differ only in D(z): NLA-M uses NLA's convention (second
    k-grid column of the tracer's perturbations), TATT-M linear
    perturbations at k = 0.03 h/Mpc (`_growth_factor_1d`). Rescaling by
    the ratio of the two makes the comparison exact; the ratio itself
    (massive-neutrino scale dependence) is checked to be within 0.5%."""
    perturbations, z = cosmo_setup
    log10_mh, f_r = [13.0, 13.5, 14.2], [1.0, 0.5, 0.3]
    nla_m = _nla_m_tracer(perturbations, _dndz(3, z), z, 3, log10_mh, f_r)
    tatt_m = _tatt_m_zero_higher_order_tracer(
        perturbations, linear_perturbations, _dndz(3, z), z, 3, log10_mh, f_r
    )

    base = np.asarray(tatt_m.ia.compute_kernel(z))
    c1, c1d, c2 = (np.asarray(c) for c in tatt_m.ia._C(z))

    d_nla_m = _nla_growth(perturbations, z)
    d_tatt_m = np.asarray(_growth_factor_1d(linear_perturbations, z))
    np.testing.assert_allclose(d_nla_m / d_tatt_m, 1.0, rtol=5e-3)

    # The kernel goes as 1/D(z), so NLA-M = TATT-M's C1 kernel * D_tatt_m / D_nla_m.
    np.testing.assert_allclose(
        np.asarray(nla_m.ia.compute_kernel(z)),
        c1 * base * (d_tatt_m / d_nla_m)[None, :],
        rtol=1e-10,
    )
    np.testing.assert_array_equal(c1d, 0.0)
    np.testing.assert_array_equal(c2, 0.0)


def test_nla_m_cl_matches_tatt_m_with_zero_higher_order(
    cosmo_setup, linear_perturbations
):
    """SHE-SHE and POS-SHE Cl of NLA-M equal TATT-M's with k1 = k2 = k3 = 0,
    with TATT-M's growth taken from linear perturbations (as in production).

    Not exact: the two D(z) conventions differ by up to ~0.2% (see
    `test_nla_m_kernel_matches_tatt_m_c1_kernel`), so the IA part differs by
    up to ~0.4%, diluted in the total by the lensing term."""
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)
    log10_mh, f_r = [13.5, 14.0], [1.0, 0.5]
    nla_m = _nla_m_tracer(perturbations, dndz, z, n_z_bins, log10_mh, f_r)
    tatt_m = _tatt_m_zero_higher_order_tracer(
        perturbations, linear_perturbations, dndz, z, n_z_bins, log10_mh, f_r
    )
    pos = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_bin0=1.1, b1_photo_bin1=1.4),
    )
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    for other_nla_m, other_tatt_m in ((nla_m, tatt_m), (pos, pos)):
        cl_nla_m = AngularTwoPoint(other_nla_m, nla_m).get_Cl_tensor(ells, 0, ks)
        cl_tatt_m = AngularTwoPoint(other_tatt_m, tatt_m).get_Cl_tensor(ells, 0, ks)
        np.testing.assert_allclose(
            np.asarray(cl_nla_m), np.asarray(cl_tatt_m), rtol=1e-3
        )


def test_nla_m_collapses_to_nla_when_bins_identical(cosmo_setup):
    """With identical bins, f_r = 1 and EtaIA = 0, NLA-M is exactly NLA with
    AIA = A1: both take D(z) the same way (`get_window_IA`'s convention)."""
    perturbations, z = cosmo_setup
    alpha_M, beta_M, log10_M0, log10_mh = 1.0, 0.5, 13.5, 14.0
    delta = log10_mh - log10_M0
    A1 = alpha_M * 10.0 ** (delta * beta_M)
    dndz = _dndz(2, z)

    nla_m = _nla_m_tracer(perturbations, dndz, z, 2, [log10_mh] * 2, [1.0] * 2)
    nla = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(2, AIA=A1, EtaIA=0.0),
        ia_model="NLA",
    )

    np.testing.assert_allclose(
        np.asarray(nla_m.ia.compute_kernel(z)),
        np.asarray(nla.ia.compute_kernel(z)),
        rtol=1e-10,
    )


def test_nla_m_f_r_zero_switches_off_bin(cosmo_setup):
    """f_r^i multiplies the whole amplitude of
    bin i, so f_r = 0 must give an identically zero IA kernel for that bin
    while leaving the other bins (here f_r = 1) non-zero."""
    perturbations, z = cosmo_setup
    tracer = _nla_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[14.0, 14.0], f_r=[0.0, 1.0]
    )
    kernel = np.asarray(tracer.ia.compute_kernel(z))
    np.testing.assert_array_equal(kernel[0], 0.0)
    assert np.any(kernel[1] != 0.0)


def test_nla_m_tatt_m_pairing_raises(cosmo_setup):
    """Pairing NLA-M with TATT-M raises TATT-M's ValueError, in either order
    and through the full `get_Cl`."""
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)
    nla_m = _nla_m_tracer(perturbations, dndz, z, 1, [13.5], [1.0])
    tatt_m = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_tatt_m_nuisance(1, [13.5], [1.0]),
        ia_model="TATT-M",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )
    bank = _bank(perturbations)

    with pytest.raises(ValueError, match="cannot be paired with NLAMContribution"):
        get_pk_terms(tatt_m.ia, nla_m.ia, z, bank)
    with pytest.raises(ValueError, match="cannot be paired with NLAMContribution"):
        get_pk_terms(nla_m.ia, tatt_m.ia, z, bank)

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    for t1, t2 in ((nla_m, tatt_m), (tatt_m, nla_m)):
        with pytest.raises(ValueError, match="cannot be paired with NLAMContribution"):
            AngularTwoPoint(t1, t2).get_Cl(ells, 0, ks)


def test_nla_m_with_nonlinear_bias_uses_b1_times_nla_m_kernel(cosmo_setup):
    """Against `galaxy_bias_model="nonlinear"`, the galaxy-intrinsic cross is
    a single term: the b1 kernel times NLA-M's own (amplitude-carrying)
    kernel times P_dd, in either tracer order. b2 is non-zero to check that
    higher-order bias is ignored for this cross."""
    perturbations, z = cosmo_setup
    nla_m = _nla_m_tracer(
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
        nl_bias_loop_computer=_StubNLBiasLoopComputer(),
    )
    bank = _bank(perturbations)
    b1_kernel = np.asarray(pos.bias.compute_kernel(z))  # b1_i * n_i(z) * H/c
    nla_m_kernel = np.asarray(nla_m.ia.compute_kernel(z))

    terms = get_pk_terms(pos.bias, nla_m.ia, z, bank)
    assert len(terms) == 1
    np.testing.assert_allclose(np.asarray(terms[0].kernel1), b1_kernel)
    np.testing.assert_allclose(np.asarray(terms[0].kernel2), nla_m_kernel)
    assert terms[0].pk is bank.matter_pk

    swapped = get_pk_terms(nla_m.ia, pos.bias, z, bank)
    assert len(swapped) == 1
    np.testing.assert_allclose(np.asarray(swapped[0].kernel1), nla_m_kernel)
    np.testing.assert_allclose(np.asarray(swapped[0].kernel2), b1_kernel)
    assert swapped[0].pk is bank.matter_pk


def test_nla_m_pos_she_nonlinear_bias_matches_linear_bias(cosmo_setup):
    """End-to-end: with only b1 non-zero, POS-SHE against NLA-M is the same
    for nonlinear bias (generalized engine, `get_pk_terms`) and linear
    `per_bin` bias with the same b1 (legacy engine, which integrates
    b1 kernel x NLA-M kernel x P_dd directly). Mirrors
    `test_pos_she_matches_legacy_linear_bias_when_higher_order_zero` for
    NLA in `test_nonlinear_bias.py`, with the same tolerance."""
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)
    nla_m = _nla_m_tracer(
        perturbations, dndz, z, n_z_bins, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )
    linear = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_bin0=1.8, b1_photo_bin1=1.3),
    )
    nonlinear = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="nonlinear",
        nuisance_params=_pos_nuisance(
            n_z_bins, b1_photo_nl_bin0=1.8, b1_photo_nl_bin1=1.3
        ),
        nl_bias_loop_computer=_StubNLBiasLoopComputer(),
    )
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    cl_linear = AngularTwoPoint(linear, nla_m).get_Cl(ells, 0, ks)
    cl_nonlinear = AngularTwoPoint(nonlinear, nla_m).get_Cl(ells, 0, ks)

    for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        np.testing.assert_allclose(
            np.asarray(cl_nonlinear[("POS", "SHE", i, j)].array),
            np.asarray(cl_linear[("POS", "SHE", i, j)].array),
            rtol=5e-3,
        )


def test_nla_m_get_ia_effective_spectra_raises(cosmo_setup):
    """`get_ia_effective_spectra` returns a model's separate P_II/P_deltaI
    grids. NLA-M, like NLA, has none: its amplitude is in the kernel and the
    spectrum is plain P_dd, so the method raises instead of returning one."""
    perturbations, z = cosmo_setup
    tracer = _nla_m_tracer(perturbations, _dndz(1, z), z, 1, [13.5], [1.0])
    with pytest.raises(ValueError, match="defines an effective power spectrum"):
        tracer.get_ia_effective_spectra()


def test_nla_m_kernel_differentiable_in_alpha_m(cosmo_setup):
    """`jax.grad` flows through `NLAMContribution` (amplitudes and kernel)
    and gives the right value. The kernel is proportional to alphaM, so
    d/d(alphaM) of its sum is the sum of the alphaM = 1 kernel."""
    perturbations, z = cosmo_setup
    tracer = _nla_m_tracer(
        perturbations, _dndz(2, z), z, 2, log10_mh=[13.5, 14.0], f_r=[1.0, 0.5]
    )

    def total_kernel(alpha_M):
        return NLAMContribution(tracer, alpha_M, 0.5).compute_kernel(z).sum()

    grad = jax.grad(total_kernel)(2.0)
    np.testing.assert_allclose(float(grad), float(total_kernel(1.0)), rtol=1e-6)
