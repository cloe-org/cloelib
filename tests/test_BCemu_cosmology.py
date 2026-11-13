"""Tests of the BCemu baryonic boost (`cloelib.cosmology.BCemu_cosmology`)."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("BCemu") is None
    or importlib.util.find_spec("camb") is None,
    reason="BCemu or camb not installed",
)

H0 = 67.3
h = H0 / 100.0
ZS = np.linspace(1e-3, 3.0, 30)
PARS = {
    1: dict(log10Mc=13.32),
    3: dict(log10Mc=13.32, thej=4.235, deta=0.14),
    7: dict(
        log10Mc=13.32, thej=4.235, deta=0.14, mu=0.93, gamma=2.25, delta=6.4, eta=0.15
    ),
}


def _background():
    from cloelib.cosmology.camb_cosmology import CAMBBackground

    return CAMBBackground(
        H0=H0,
        Omega_b0=0.05,
        Omega_cdm0=0.27,
        Omega_k0=0.0,
        As=2.1e-9,
        ns=0.966,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0,
    )


@pytest.fixture(scope="module")
def linear():
    from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations

    return CAMBLinearPerturbations(_background(), ZS)


def _build(linear, **baryon_kwargs):
    from cloelib.cosmology.BCemu_cosmology import (
        CAMBNonLinearBCemuBaryonicPerturbations,
    )

    return CAMBNonLinearBCemuBaryonicPerturbations(
        _background(), linear, ZS, baryon_kwargs=baryon_kwargs
    )


@pytest.fixture(scope="module", params=[1, 3, 7], ids=lambda n: f"{n}param")
def bcemu(request, linear):
    return request.param, _build(linear, **PARS[request.param])


def test_protocol_and_model(bcemu):
    from cloelib.cosmology.cosmology import Perturbations

    n_params, pert = bcemu
    assert isinstance(pert, Perturbations)
    assert pert.n_params == n_params


def test_matches_bcemu_on_nodes(bcemu):
    """On the BCemu training redshifts and wavenumbers the table is BCemu itself."""
    import BCemu
    from cloelib.cosmology.BCemu_cosmology import _BCEMU_K, _MODELS

    n_params, pert = bcemu
    k = np.asarray(_BCEMU_K)[5:-5]
    emulator = getattr(BCemu, _MODELS[n_params][0])(verbose=False)
    for z in (0.0, 1.0, 2.0):
        ref = emulator.get_boost(z, dict(PARS[n_params]), k, fb=pert.fb)
        np.testing.assert_allclose(
            pert.baryonic_suppression([z], k, k_hunit=True)[0], ref, rtol=1e-6
        )


def test_limits(bcemu):
    _, pert = bcemu
    # B = 1 on large scales, finite and suppressing on small scales.
    np.testing.assert_allclose(pert.baryonic_suppression(ZS, [1e-3], k_hunit=True), 1.0)
    S = pert.baryonic_suppression(ZS, np.logspace(-1, 1.5, 20), k_hunit=True)
    assert np.all(np.isfinite(S)) and np.all((S > 0.5) & (S < 1.1))
    # Above z = 2 the boost is held at its z = 2 value by default.
    np.testing.assert_allclose(
        pert.baryonic_suppression([2.0], [5.0], k_hunit=True),
        pert.baryonic_suppression([2.7], [5.0], k_hunit=True),
    )
    # k units: 1/Mpc by default.
    np.testing.assert_allclose(
        pert.baryonic_suppression([0.5], [3.0 * h]),
        pert.baryonic_suppression([0.5], [3.0], k_hunit=True),
    )


def test_spectrum_is_suppressed(bcemu, linear):
    from cloelib.cosmology.camb_cosmology import CAMBNonLinearPerturbations

    _, pert = bcemu
    dmo = CAMBNonLinearPerturbations(_background(), linear, ZS)
    ks = np.logspace(-2, 0.8, 15)
    np.testing.assert_allclose(
        pert.matter_power_spectrum(ZS[:3], ks),
        dmo.matter_power_spectrum(ZS[:3], ks) * pert.baryonic_suppression(ZS[:3], ks),
        rtol=1e-6,
    )


def test_redshift_evolution(linear):
    """nu_Mc > 0 lowers log10Mc at z > 0, i.e. weakens the suppression."""
    base = _build(linear, **PARS[3])
    evolved = _build(linear, **PARS[3], nu_Mc=0.5)
    np.testing.assert_allclose(
        base.baryonic_suppression([0.0], [5.0]),
        evolved.baryonic_suppression([0.0], [5.0]),
    )
    assert not np.allclose(
        base.baryonic_suppression([1.0], [5.0]),
        evolved.baryonic_suppression([1.0], [5.0]),
    )


def test_state_does_not_leak_between_instances(linear):
    """BCemu keeps parameters on its (cached) instance; each build must reset them."""
    first = _build(linear, **PARS[3], nu_Mc=0.5)
    again = _build(linear, **PARS[3])
    fresh = _build(linear, **PARS[3])
    np.testing.assert_allclose(
        again.baryonic_suppression(ZS, [5.0]), fresh.baryonic_suppression(ZS, [5.0])
    )
    assert first.n_params == again.n_params


def test_extrapolation_options(linear):
    pert = _build(
        linear, **PARS[3], k_extrapolation="power_law", z_extrapolation="linear"
    )
    const = _build(linear, **PARS[3])
    assert np.all(np.isfinite(pert.baryonic_suppression([2.5], [30.0], k_hunit=True)))
    assert not np.allclose(
        pert.baryonic_suppression([0.0], [30.0], k_hunit=True),
        const.baryonic_suppression([0.0], [30.0], k_hunit=True),
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(log10Mc=13.0, thej=4.0), "exactly one of its models"),
        (dict(log10Mc=13.0, n_params=3), "exactly one of its models"),
        (dict(log10Mc=16.0), "training box"),
        (dict(log10Mc=13.0, k_extrapolation="spline"), "k_extrapolation"),
    ],
)
def test_invalid_inputs_raise(linear, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build(linear, **kwargs)
