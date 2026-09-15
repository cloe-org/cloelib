"""
Tests for `NonLinearGalaxyBiasContribution` (see `CONTRIBUTION_ARCHITECTURE.md`
and `test_tatt.py`, which this mirrors closely).

These check the *architecture*: that `galaxy_bias_model="nonlinear"` wires
up correctly (requires `nl_bias_loop_computer`, rejects it for the other
models, reads per-bin nuisance params), that its request pruning against an
IA contribution is correct, that `get_pk_terms` collapses to the exact
linear-bias limit when `b2=bs2=b3nl=bk2=0` for both same-bin (P_gg) and
cross-bin (a genuine two-bin P_gg term, and P_g,delta) pairings (checkable
exactly, independent of the one-loop kernel *values*, since every extra
term is multiplied by a zero amplitude - same trick `test_tatt.py` uses),
and that the generalized Cl path produces finite, correctly-shaped output
end-to-end.

`NonLinearGalaxyBiasContribution` requires a real `loop_computer` (no
illustrative default - see `cloelib/observables/photo/positions.py`), so
most tests here use `_StubNLBiasLoopComputer` (below - a minimal,
dependency-free stand-in, not part of the public API) purely to exercise
the architecture independent of any PT backend; tests that specifically
validate the real backend use `PBJNonlinearBiasLoopComputer` (real FAST-PT
`one_loop_dd_bias_b3nl` kernels) and are skipped unless `fast-pt` is
installed.
"""

import importlib.util

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import IntrinsicAlignmentContribution
from cloelib.observables.photo.positions import (
    GalaxyBiasContribution,
    NonLinearGalaxyBiasContribution,
    _NLBIAS_KERNELS,
)
from cloelib.observables.photo.spectrum_engine import (
    SpectraBank,
    needs_generalized_engine,
)
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

_FASTPT_INSTALLED = importlib.util.find_spec("fastpt") is not None
if _FASTPT_INSTALLED:
    from cloelib.observables.photo.positions import PBJNonlinearBiasLoopComputer


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
    """A *linear* Perturbations object, as `PBJNonlinearBiasLoopComputer`
    requires (FAST-PT's one-loop integrals are only valid starting from the
    linear Pk) - `cosmo_setup`'s own `perturbations` is nonlinear.
    """
    perturbations, _ = cosmo_setup
    return CAMBLinearPerturbations(perturbations.background, perturbations.z)


class _StubNLBiasLoopComputer:
    """Minimal, dependency-free stand-in for a real nonlinear-bias
    `loop_computer`.

    Used only to exercise the Contribution/generalized-engine architecture
    (shapes, request pruning, the per-contribution-pair Limber integral) in
    tests that don't need `fast-pt` installed - not part of the public API,
    and not a claim of physical accuracy. Real kernels come from
    `PBJNonlinearBiasLoopComputer` (see the `_FASTPT_INSTALLED`-gated tests
    below).
    """

    def compute(self, name):
        del name  # same kernel shape regardless of which one is requested

        def _compute(matter_pk, ks, zs):
            del ks, zs
            return 1e-2 * matter_pk[0]

        return _compute


def _pos_nuisance(n_z_bins, **extra):
    return {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(n_z_bins)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(n_z_bins)},
        **{f"magnification_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
        **extra,
    }


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


def _dndz(n_z_bins, z):
    dndz = np.ones((n_z_bins, len(z)))
    return dndz / np.trapezoid(dndz, z, axis=1)[:, None]


def _nlbias_tracer(perturbations, dndz, z, loop_computer, **nuisance_extra):
    return PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="nonlinear",
        nuisance_params=_pos_nuisance(dndz.shape[0], **nuisance_extra),
        nl_bias_loop_computer=loop_computer,
    )


def test_galaxy_bias_model_nonlinear_builds_per_bin_contribution(cosmo_setup):
    perturbations, z = cosmo_setup
    dndz = _dndz(2, z)

    tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        _StubNLBiasLoopComputer(),
        b1_photo_nl_bin0=1.7,
        b1_photo_nl_bin1=2.1,
        b2_photo_nl_bin0=0.3,
    )
    assert isinstance(tracer.bias, NonLinearGalaxyBiasContribution)
    np.testing.assert_allclose(np.asarray(tracer.bias.b1), [1.7, 2.1])
    np.testing.assert_allclose(np.asarray(tracer.bias.b2), [0.3, 0.0])
    np.testing.assert_allclose(np.asarray(tracer.bias.bs2), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(tracer.bias.b3nl), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(tracer.bias.bk2), [0.0, 0.0])
    assert tracer.bias_array is None


def test_nonlinear_requires_loop_computer(cosmo_setup):
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)
    with pytest.raises(ValueError):
        PositionsTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="nonlinear",
            nuisance_params=_pos_nuisance(1),
        )


def test_loop_computer_rejected_for_non_nonlinear_model(cosmo_setup):
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)
    with pytest.raises(ValueError):
        PositionsTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params=_pos_nuisance(1, b1_photo_bin0=1.1),
            nl_bias_loop_computer=_StubNLBiasLoopComputer(),
        )


def test_requirements_pruned_for_ia_pairing(cosmo_setup):
    """No one-loop kernel is needed when paired with an IA contribution -
    only `b1 * matter_pk` is used there (see class docstring)."""
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)
    pos_tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        _StubNLBiasLoopComputer(),
        b1_photo_nl_bin0=1.5,
        b2_photo_nl_bin0=0.4,
    )
    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    bias = pos_tracer.bias
    nla = shear_tracer.ia
    lensing = shear_tracer.lensing

    assert bias.get_requirements_for_interaction(nla) == ()
    assert isinstance(nla, IntrinsicAlignmentContribution)
    assert {r.name for r in bias.get_requirements_for_interaction(lensing)} == set(
        _NLBIAS_KERNELS
    )
    assert needs_generalized_engine(
        pos_tracer.get_contributions(), shear_tracer.get_contributions()
    )


def test_pos_pos_matches_legacy_linear_bias_when_higher_order_zero(cosmo_setup):
    """Strongest available check, mirroring `test_tatt.py`'s analogous
    `test_tatt_matches_legacy_nla_end_to_end_at_z0_zero`: with
    `b2=bs2=b3nl=bk2=0`, `galaxy_bias_model="nonlinear"` must reproduce
    `galaxy_bias_model="per_bin"`'s `get_Cl` output for the *same* per-bin
    b1 values - through the entire pipeline (legacy `_compute_cl_legacy`
    vs generalized `_compute_cl_generalized`, including the two different
    Pk-interpolation code paths and the `PkTerm`/`bank.zs` z-grid handling
    `get_pk_terms` relies on), not just an isolated formula check. Uses two
    *different* per-bin b1 values (bins 1-1 and 2-2 differ, plus the
    genuine cross-bin pair 1-2) so this exercises real per-bin support, not
    just a global-scalar-equivalent special case.
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)

    linear_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_bin0=1.8, b1_photo_bin1=1.3),
    )
    nlbias_tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        _StubNLBiasLoopComputer(),
        b1_photo_nl_bin0=1.8,
        b1_photo_nl_bin1=1.3,
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    cl_linear = AngularTwoPoint(linear_tracer, linear_tracer).get_Cl(ells, 0, ks)
    cl_nlbias = AngularTwoPoint(nlbias_tracer, nlbias_tracer).get_Cl(ells, 0, ks)

    for i, j in [(1, 1), (1, 2), (2, 2)]:
        np.testing.assert_allclose(
            np.asarray(cl_nlbias[("POS", "POS", i, j)].array),
            np.asarray(cl_linear[("POS", "POS", i, j)].array),
            rtol=5e-3,
        )


def test_pos_she_matches_legacy_linear_bias_when_higher_order_zero(cosmo_setup):
    """Same check as `test_pos_pos_matches_legacy_linear_bias_when_
    higher_order_zero`, but for the POS-SHE cross with an NLA `ShearTracer`
    - exercises *both* `get_pk_terms` branches at once: the (bias, lensing)
    galaxy-matter cross (reduces to `b1 * P_dd`) and the (bias, NLA)
    galaxy-intrinsic cross (also `b1 * P_dd`, deferring the IA amplitude to
    NLA's own kernel - see class docstring), both against the untouched
    legacy path (linear bias x NLA never triggers the generalized engine).
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)

    linear_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="per_bin",
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_bin0=1.8, b1_photo_bin1=1.3),
    )
    nlbias_tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        _StubNLBiasLoopComputer(),
        b1_photo_nl_bin0=1.8,
        b1_photo_nl_bin1=1.3,
    )
    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins),
        ia_model="NLA",
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    cl_linear = AngularTwoPoint(linear_tracer, shear_tracer).get_Cl(ells, 0, ks)
    cl_nlbias = AngularTwoPoint(nlbias_tracer, shear_tracer).get_Cl(ells, 0, ks)

    assert not needs_generalized_engine(
        linear_tracer.get_contributions(), shear_tracer.get_contributions()
    )
    assert needs_generalized_engine(
        nlbias_tracer.get_contributions(), shear_tracer.get_contributions()
    )
    for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        np.testing.assert_allclose(
            np.asarray(cl_nlbias[("POS", "SHE", i, j)].array),
            np.asarray(cl_linear[("POS", "SHE", i, j)].array),
            rtol=5e-3,
        )


def test_pk_terms_against_ia_ignores_higher_order_bias(cosmo_setup):
    """Paired with an IA contribution, exactly one `PkTerm` is returned
    (`b1 * matter_pk`) regardless of b2/bs2/b3nl/bk2 - a narrower, purely
    structural check complementing the end-to-end comparison above."""
    perturbations, z = cosmo_setup
    dndz = _dndz(1, z)
    pos_tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        _StubNLBiasLoopComputer(),
        b1_photo_nl_bin0=1.3,
        b2_photo_nl_bin0=5.0,
        bs2_photo_nl_bin0=-2.0,
        b3nl_photo_nl_bin0=7.0,
        bk2_photo_nl_bin0=3.0,
    )
    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(1),
        ia_model="NLA",
    )
    bias = pos_tracer.bias
    ks = np.logspace(-3, 1, 20)
    pert_zs = perturbations.z
    matter_pk = perturbations.matter_power_spectrum(pert_zs, ks)
    bank = SpectraBank(matter_pk, ks, pert_zs)

    terms = bias.get_pk_terms(shear_tracer.ia, z, bank)
    assert len(terms) == 1
    np.testing.assert_allclose(
        np.asarray(terms[0].pk), np.asarray(matter_pk), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(terms[0].kernel1[0]) / 1.3,
        np.asarray(pos_tracer.dndz_shifted[0])
        * np.asarray(perturbations.background.hubble_parameter(z) / 299792.458),
        rtol=1e-6,
    )


def test_generalized_cl_finite_and_correctly_shaped(cosmo_setup):
    """End-to-end: PositionsTracer(galaxy_bias_model="nonlinear") x itself
    (POS-POS, with genuinely different per-bin biases) and x a plain
    ShearTracer (POS-SHE) through AngularTwoPoint.get_Cl produces finite
    output with the same shape contract as the legacy (linear-bias) path.
    """
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)

    pos_tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        _StubNLBiasLoopComputer(),
        b1_photo_nl_bin0=1.4,
        b1_photo_nl_bin1=1.7,
        b2_photo_nl_bin0=0.5,
        b2_photo_nl_bin1=0.2,
        bs2_photo_nl_bin0=-0.3,
        bk2_photo_nl_bin0=0.1,
    )
    shear_tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=_shear_nuisance(n_z_bins),
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)

    pos_pos = AngularTwoPoint(pos_tracer, pos_tracer).get_Cl(ells, 0, ks)
    pos_she = AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells, 0, ks)

    assert ("POS", "POS", 1, 1) in pos_pos
    assert ("POS", "POS", 1, 2) in pos_pos
    for key, spectrum in {**pos_pos, **pos_she}.items():
        arr = np.asarray(spectrum.array)
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"

    assert pos_pos[("POS", "POS", 1, 1)].array.shape == (len(ells),)
    assert pos_she[("POS", "SHE", 1, 1)].array.shape == (2, len(ells))


def test_generalized_cl_finite_with_cmb_lensing(cosmo_setup):
    """POS(nonlinear) x CMBLensingTracer must also go through the
    generalized engine (`CMBLensingContribution`, `cloelib/observables/
    cmb.py`) rather than silently falling back to the legacy path with an
    amplitude-free (and therefore wrong) window."""
    from cloelib.observables.cmb import CMBLensingTracer

    perturbations, z = cosmo_setup
    n_z_bins = 1
    dndz = _dndz(n_z_bins, z)

    pos_tracer = _nlbias_tracer(
        perturbations, dndz, z, _StubNLBiasLoopComputer(), b1_photo_nl_bin0=1.4
    )
    cmbl_tracer = CMBLensingTracer(perturbations=perturbations, z=z)

    assert needs_generalized_engine(
        pos_tracer.get_contributions(), cmbl_tracer.get_contributions()
    )

    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    cl = AngularTwoPoint(pos_tracer, cmbl_tracer).get_Cl(ells, 0, ks)
    arr = np.asarray(cl[("CMBL", "POS", 1, 1)].array)
    assert np.all(np.isfinite(arr))


def test_rsd_with_generalized_engine_raises_not_implemented(cosmo_setup):
    perturbations, z = cosmo_setup
    n_z_bins = 1
    dndz = _dndz(n_z_bins, z)

    pos_tracer = PositionsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        galaxy_bias_model="nonlinear",
        nuisance_params=_pos_nuisance(n_z_bins, b1_photo_nl_bin0=1.4),
        nl_bias_loop_computer=_StubNLBiasLoopComputer(),
        include_rsd=True,
    )
    ells = np.logspace(1.0, np.log10(200), 6)
    ks = np.asarray(perturbations.k)
    with pytest.raises(NotImplementedError):
        AngularTwoPoint(pos_tracer, pos_tracer).get_Cl(ells, 0, ks)


def test_linear_galaxy_bias_contribution_unaffected():
    """Sanity check: the plain linear `GalaxyBiasContribution` is untouched
    by any of this - it's a distinct class, selected by every
    `galaxy_bias_model` other than `"nonlinear"`."""
    assert not hasattr(GalaxyBiasContribution, "get_spectrum_requests")
    assert not hasattr(GalaxyBiasContribution, "get_pk_terms")


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_nlbias_loop_computer_kernels_finite_and_correctly_shaped(
    linear_perturbations,
):
    """The real FAST-PT backend must return all seven named kernels, each
    finite and shaped like the requested k-grid - `NonLinearGalaxyBiasContribution`'s
    `get_pk_terms` relies on this shape contract."""
    computer = PBJNonlinearBiasLoopComputer(linear_perturbations)
    ks = np.logspace(-3, 1, 40)
    zs = np.linspace(0.0, 2.0, 5)
    matter_pk = np.ones((len(zs), len(ks)))  # unused by this computer

    for name in _NLBIAS_KERNELS:
        values = np.asarray(computer.compute(name)(matter_pk, ks, zs))
        assert values.shape == ks.shape, name
        assert np.all(np.isfinite(values)), name


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_nlbias_loop_computer_caches_across_kernel_names(linear_perturbations):
    """Seven kernel names share one (expensive) FAST-PT call per k-grid,
    not one FAST-PT call each - `_kernels_for` must be memoized."""
    computer = PBJNonlinearBiasLoopComputer(linear_perturbations)
    ks = np.logspace(-3, 1, 40)
    zs = np.linspace(0.0, 2.0, 5)
    matter_pk = np.ones((len(zs), len(ks)))

    computer.compute(_NLBIAS_KERNELS[0])(matter_pk, ks, zs)
    cached_after_first = computer._cached_kernels
    computer.compute(_NLBIAS_KERNELS[1])(matter_pk, ks, zs)
    assert computer._cached_kernels is cached_after_first


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_pbj_nlbias_generalized_cl_finite_with_real_fastpt(
    cosmo_setup, linear_perturbations
):
    """End-to-end with the real FAST-PT backend, per-bin biases and all
    five bias parameters nonzero - guards against the same class of
    numerical-stability issue `test_tatt.py`'s analogous test documents for
    TATT (steep/sign-changing kernels near the k-grid edges interacting
    badly with the Limber grid's `k_l = (ell+0.5)/chi` at chi -> 0)."""
    perturbations, z = cosmo_setup
    n_z_bins = 2
    dndz = _dndz(n_z_bins, z)

    pos_tracer = _nlbias_tracer(
        perturbations,
        dndz,
        z,
        PBJNonlinearBiasLoopComputer(linear_perturbations),
        b1_photo_nl_bin0=1.4,
        b1_photo_nl_bin1=1.7,
        b2_photo_nl_bin0=0.5,
        b2_photo_nl_bin1=0.2,
        bs2_photo_nl_bin0=-0.3,
        b3nl_photo_nl_bin1=0.2,
        bk2_photo_nl_bin0=0.1,
    )

    ells = np.logspace(1.0, np.log10(3000), 40)  # wide range - exercises chi~0 edge
    ks = np.asarray(perturbations.k)

    pos_pos = AngularTwoPoint(pos_tracer, pos_tracer).get_Cl(ells, 0, ks)
    for key, spectrum in pos_pos.items():
        arr = np.asarray(spectrum.array)
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"
