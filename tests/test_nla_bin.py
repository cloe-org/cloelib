"""Tests for NLA-bin: NLA with a free intrinsic-alignment amplitude per
tomographic bin (`ia_model="NLA-bin"`).

`cosmo_setup` comes from `conftest.py`.
"""

import jax
import numpy as np
import pytest

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import IntrinsicAlignmentContribution
from cloelib.observables.photo.shear import NLABinContribution, _nla_bin_amplitudes
from cloelib.observables.photo.spectrum_engine import (
    SpectraBank,
    get_pk_terms,
    needs_generalized_engine,
)
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from test_nonlinear_bias import _StubNLBiasLoopComputer
from test_tatt import _StubTATTLoopComputer, _shear_nuisance
from test_tatt_m import _dndz, _pos_nuisance, _tatt_m_nuisance


def _nla_bin_nuisance(a_ia, **extra):
    """`_shear_nuisance` plus one `AIA_{i}` per bin.

    The NLA keys `_shear_nuisance` sets (`AIA`, `EtaIA`) are ignored by
    NLA-bin.
    """
    base = _shear_nuisance(len(a_ia))
    base.update({f"AIA_{i + 1}": a for i, a in enumerate(a_ia)})
    base.update(extra)
    return base


def _nla_bin_tracer(perturbations, z, a_ia, **extra):
    return ShearTracer(
        perturbations=perturbations,
        dndz=_dndz(len(a_ia), z),
        z=z,
        nuisance_params=_nla_bin_nuisance(a_ia, **extra),
        ia_model="NLA-bin",
    )


def _nla_tracer(perturbations, z, n_z_bins, A_IA):
    """Plain NLA with amplitude `A_IA` and no redshift evolution."""
    return ShearTracer(
        perturbations=perturbations,
        dndz=_dndz(n_z_bins, z),
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins, AIA=A_IA, EtaIA=0.0),
        ia_model="NLA",
    )


def _bank(perturbations):
    ks = np.logspace(-3, 1, 20)
    zs = perturbations.z
    return SpectraBank(perturbations.matter_power_spectrum(zs, ks), ks, zs)


def test_nla_bin_amplitudes_read_per_bin_keys():
    """`_nla_bin_amplitudes(nuisance, n_bins)` returns `[AIA_1, ..., AIA_n]`
    in bin order. The plain NLA key `AIA` is set to a different value to
    check that it is ignored."""
    nuisance = {"AIA_1": 0.5, "AIA_2": 1.2, "AIA_3": -0.3, "AIA": 9.0}
    np.testing.assert_array_equal(
        np.asarray(_nla_bin_amplitudes(nuisance, 3)), [0.5, 1.2, -0.3]
    )


def test_ia_model_nla_bin_builds_nla_bin_contribution(cosmo_setup):
    """NLA-bin is a plain kernel model: no extra spectra, so the engine never
    needs one-loop kernels for it."""
    perturbations, z = cosmo_setup
    tracer = _nla_bin_tracer(perturbations, z, [0.8, 1.3])
    assert isinstance(tracer.ia, NLABinContribution)
    assert isinstance(tracer.ia, IntrinsicAlignmentContribution)
    assert tracer.get_contributions() == (tracer.lensing, tracer.ia)
    np.testing.assert_array_equal(np.asarray(tracer.ia.A_IA), [0.8, 1.3])
    assert not needs_generalized_engine(
        tracer.get_contributions(), tracer.get_contributions()
    )


def test_nla_bin_missing_amplitude_raises(cosmo_setup):
    """Every bin needs its own `AIA_{i}`; there is no fallback to `AIA`."""
    perturbations, z = cosmo_setup
    nuisance = _nla_bin_nuisance([0.8, 1.3])
    del nuisance["AIA_2"]
    with pytest.raises(KeyError, match="AIA_2"):
        ShearTracer(
            perturbations=perturbations,
            dndz=_dndz(2, z),
            z=z,
            nuisance_params=nuisance,
            ia_model="NLA-bin",
        )


def test_nla_bin_wrong_number_of_amplitudes_raises(cosmo_setup):
    """`NLABinContribution` needs exactly one amplitude per tomographic bin:
    three amplitudes for a two-bin tracer raise `ValueError`."""
    perturbations, z = cosmo_setup
    tracer = _nla_bin_tracer(perturbations, z, [0.8, 1.3])
    with pytest.raises(ValueError, match="one amplitude per bin"):
        NLABinContribution(tracer, [0.8, 1.3, 2.0])


def test_nla_bin_rejects_tatt_loop_computer(cosmo_setup):
    """`tatt_loop_computer` is only accepted with ia_model='TATT' or
    'TATT-M'. NLA-bin needs no one-loop kernels, so passing one raises, as
    it does for NLA. `_StubTATTLoopComputer` is the fake loop computer the
    TATT/TATT-M tests use; any non-None object would trigger the error."""
    perturbations, z = cosmo_setup
    with pytest.raises(ValueError, match="only used when ia_model='TATT' or 'TATT-M'"):
        ShearTracer(
            perturbations=perturbations,
            dndz=_dndz(1, z),
            z=z,
            nuisance_params=_nla_bin_nuisance([1.0]),
            ia_model="NLA-bin",
            tatt_loop_computer=_StubTATTLoopComputer(perturbations),
        )


def test_nla_bin_kernel_equals_nla_kernel_when_amplitudes_identical(cosmo_setup):
    """With the same amplitude in every bin (and EtaIA = 0 for NLA), the
    NLA-bin IA kernel equals the NLA one, including D(z). The SHE-SHE C_ell
    (lensing + IA) are then compared too: both models pair the kernel with
    the same plain P_dd, so they must agree as well."""
    perturbations, z = cosmo_setup
    a = 1.7
    nla_bin = _nla_bin_tracer(perturbations, z, [a, a, a])
    nla = _nla_tracer(perturbations, z, 3, A_IA=a)
    np.testing.assert_allclose(
        np.asarray(nla_bin.ia.compute_kernel(z)),
        np.asarray(nla.ia.compute_kernel(z)),
        rtol=1e-10,
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    cl_nla_bin = AngularTwoPoint(nla_bin, nla_bin).get_Cl(ells, 0, ks)
    cl_nla = AngularTwoPoint(nla, nla).get_Cl(ells, 0, ks)
    # SHE-SHE is symmetric in the bins: `get_Cl` only returns j >= i.
    for i in range(1, 4):
        for j in range(i, 4):
            np.testing.assert_allclose(
                np.asarray(cl_nla_bin[("SHE", "SHE", i, j)].array),
                np.asarray(cl_nla[("SHE", "SHE", i, j)].array),
                rtol=1e-10,
            )


def test_nla_bin_kernel_scales_with_each_bin_amplitude(cosmo_setup):
    """Bin i's kernel is the A_IA = 1 NLA kernel of that bin times A_IA^i,
    independently of the other bins. A_IA^i = 0 switches bin i off, and a
    negative amplitude flips the sign."""
    perturbations, z = cosmo_setup
    a_ia = np.array([0.0, 1.5, -0.7])
    nla_bin = _nla_bin_tracer(perturbations, z, list(a_ia))
    # Keep the tracer alive: its contribution only holds a weak reference.
    unit_nla_tracer = _nla_tracer(perturbations, z, 3, A_IA=1.0)
    unit_nla = np.asarray(unit_nla_tracer.ia.compute_kernel(z))
    kernel = np.asarray(nla_bin.ia.compute_kernel(z))

    np.testing.assert_allclose(kernel, a_ia[:, None] * unit_nla, rtol=1e-10, atol=0.0)
    np.testing.assert_array_equal(kernel[0], 0.0)


def test_nla_bin_tatt_m_pairing_raises(cosmo_setup):
    """Pairing NLA-bin with TATT-M raises TATT-M's ValueError, in either
    order and through the full `get_Cl`."""
    perturbations, z = cosmo_setup
    nla_bin = _nla_bin_tracer(perturbations, z, [1.0])
    tatt_m = ShearTracer(
        perturbations=perturbations,
        dndz=_dndz(1, z),
        z=z,
        nuisance_params=_tatt_m_nuisance(1, [13.5], [1.0]),
        ia_model="TATT-M",
        tatt_loop_computer=_StubTATTLoopComputer(perturbations),
    )
    bank = _bank(perturbations)
    message = "cannot be paired with NLABinContribution"

    with pytest.raises(ValueError, match=message):
        get_pk_terms(tatt_m.ia, nla_bin.ia, z, bank)
    with pytest.raises(ValueError, match=message):
        get_pk_terms(nla_bin.ia, tatt_m.ia, z, bank)

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    for t1, t2 in ((nla_bin, tatt_m), (tatt_m, nla_bin)):
        with pytest.raises(ValueError, match=message):
            AngularTwoPoint(t1, t2).get_Cl(ells, 0, ks)


def test_nla_bin_with_nonlinear_bias_uses_b1_times_nla_bin_kernel(cosmo_setup):
    """Against `galaxy_bias_model="nonlinear"`, the galaxy-intrinsic cross is
    a single term: the b1 kernel times NLA-bin's own (amplitude-carrying)
    kernel times P_dd, in either tracer order. b2 is non-zero to check that
    higher-order bias is ignored for this cross."""
    perturbations, z = cosmo_setup
    nla_bin = _nla_bin_tracer(perturbations, z, [0.8, 1.3])
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
    # Only the b1 part of the nonlinear-bias kernel, b1_i * n_i(z) * H/c.
    b1_kernel = np.asarray(pos.bias.compute_kernel(z))
    ia_kernel = np.asarray(nla_bin.ia.compute_kernel(z))

    terms = get_pk_terms(pos.bias, nla_bin.ia, z, bank)
    assert len(terms) == 1
    np.testing.assert_allclose(np.asarray(terms[0].kernel1), b1_kernel)
    np.testing.assert_allclose(np.asarray(terms[0].kernel2), ia_kernel)
    # The very same P_dd array from the bank, not a one-loop spectrum.
    assert terms[0].pk is bank.matter_pk

    swapped = get_pk_terms(nla_bin.ia, pos.bias, z, bank)
    assert len(swapped) == 1
    np.testing.assert_allclose(np.asarray(swapped[0].kernel1), ia_kernel)
    np.testing.assert_allclose(np.asarray(swapped[0].kernel2), b1_kernel)
    assert swapped[0].pk is bank.matter_pk


def test_nla_bin_pos_she_nonlinear_bias_matches_linear_bias(cosmo_setup):
    """End-to-end: with only b1 non-zero, POS-SHE against NLA-bin is the
    same for nonlinear bias (generalized engine, `get_pk_terms`) and linear
    `per_bin` bias with the same b1 (legacy engine). Mirrors
    `test_pos_she_matches_legacy_linear_bias_when_higher_order_zero` for
    NLA in `test_nonlinear_bias.py`, with the same tolerance."""
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)
    nla_bin = _nla_bin_tracer(perturbations, z, [0.8, 1.3])
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

    cl_linear = AngularTwoPoint(linear, nla_bin).get_Cl(ells, 0, ks)
    cl_nonlinear = AngularTwoPoint(nonlinear, nla_bin).get_Cl(ells, 0, ks)

    for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        np.testing.assert_allclose(
            np.asarray(cl_nonlinear[("POS", "SHE", i, j)].array),
            np.asarray(cl_linear[("POS", "SHE", i, j)].array),
            rtol=5e-3,
        )


def test_nla_bin_get_ia_effective_spectra_raises(cosmo_setup):
    """`get_ia_effective_spectra` returns a model's separate P_II/P_deltaI
    grids. NLA-bin, like NLA, has none: its amplitude is in the kernel and
    the spectrum is plain P_dd, so the method raises."""
    perturbations, z = cosmo_setup
    tracer = _nla_bin_tracer(perturbations, z, [1.0])
    with pytest.raises(ValueError, match="defines an effective power spectrum"):
        tracer.get_ia_effective_spectra()


def test_nla_bin_kernel_differentiable_in_amplitudes(cosmo_setup):
    """`jax.grad` flows through `NLABinContribution`. The kernel of bin i is
    proportional to A_IA^i only, so d/d(A_IA^i) of the summed kernel is the
    sum of bin i's kernel at A_IA^i = 1."""
    perturbations, z = cosmo_setup
    tracer = _nla_bin_tracer(perturbations, z, [0.8, 1.3])

    def total_kernel(a_ia):
        return NLABinContribution(tracer, a_ia).compute_kernel(z).sum()

    grad = jax.grad(total_kernel)(np.array([0.8, 1.3]))
    unit_kernel = np.asarray(NLABinContribution(tracer, [1.0, 1.0]).compute_kernel(z))
    np.testing.assert_allclose(np.asarray(grad), unit_kernel.sum(axis=1), rtol=1e-6)
