"""Tests of the EuclidEmulator3 backend (`cloelib.cosmology.EE3_cosmology`)."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("euclidemulator3") is None
    or importlib.util.find_spec("camb") is None,
    reason="euclidemulator3 or camb not installed",
)

H0 = 67.7
h = H0 / 100.0
ZS = np.linspace(0.0, 3.0, 40)
ZQ = ZS[[0, 7, 20, 39]]  # redshifts on the grid, so no z interpolation


def _background(mnu=0.0):
    from cloelib.cosmology.camb_cosmology import CAMBBackground

    return CAMBBackground(
        H0=H0,
        Omega_b0=0.022 / h**2,
        Omega_cdm0=0.12 / h**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=mnu,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0 if mnu == 0.0 else 1,
    )


@pytest.fixture(scope="module", params=[0.0, 0.12], ids=["massless", "mnu0.12"])
def ee3_all(request):
    from cloelib.cosmology.EE3_cosmology import (
        EE3HMcodePerturbations,
        EE3LinearPerturbations,
        EE3NonLinearPerturbations,
    )

    background = _background(request.param)
    lin = EE3LinearPerturbations(background=background, redshifts=ZS)
    nonlin = EE3NonLinearPerturbations(
        background=background, linearperturbations=lin, redshifts=ZS
    )
    hmcode = EE3HMcodePerturbations(
        background=background, linearperturbations=lin, redshifts=ZS
    )
    return lin, nonlin, hmcode


@pytest.fixture(scope="module")
def ee3(ee3_all):
    return ee3_all[:2]


def test_protocol(ee3_all):
    from cloelib.cosmology.cosmology import Perturbations

    for obj in ee3_all:
        assert isinstance(obj, Perturbations)


@pytest.mark.parametrize("kind", ["cold", "total"])
def test_reproduces_emulator(ee3, kind):
    """With EE3 linear input the backend is the full EE3 prediction, emu.sim.pk_nl."""
    from cloelib.cosmology.EE3_cosmology import emu

    _, nonlin = ee3
    method = (
        nonlin.matter_power_spectrum_cb
        if kind == "cold"
        else nonlin.matter_power_spectrum
    )
    ref = emu.sim.pk_nl(nonlin.params_emu, ZQ, kind=kind) / h**3
    np.testing.assert_allclose(method(ZQ, emu.k * h), ref, rtol=1e-5)


def test_linear_matches_emulator(ee3):
    from cloelib.cosmology.EE3_cosmology import emu

    lin, _ = ee3
    k = np.asarray(emu.lin.k)[50:450]
    for z in ZQ:
        ref = emu.lin.pk(lin.params_emu, z, kind="total") / h**3
        np.testing.assert_allclose(
            lin.matter_power_spectrum([z], k * h)[0], ref[50:450], rtol=1e-5
        )


def test_shapes_and_limits(ee3_all):
    lin, nonlin, hmcode = ee3_all
    ks = np.logspace(-4, 1.5, 25)
    for obj in (lin, nonlin, hmcode):
        for method in (obj.matter_power_spectrum, obj.matter_power_spectrum_cb):
            out = method(ZQ, ks)
            assert out.shape == (len(ZQ), len(ks))
            assert np.all(np.isfinite(out)) and np.all(out > 0)
        assert obj.growth_factor(np.array([0.5]), ks).shape == (1, len(ks))
    # B = 1 on large scales (EE3 is anchored to linear theory there).
    np.testing.assert_allclose(nonlin.nonlinear_boost(ZQ, [1e-3]), 1.0, atol=1e-10)
    # The boost is extended beyond the emulated z and k ranges without blowing up.
    assert np.all(np.isfinite(nonlin.nonlinear_boost([6.5], [50.0])))


def test_growth_rate(ee3):
    lin, nonlin = ee3
    zs = np.array([0.3, 1.1])
    ks = np.array([0.01, 0.1])
    for obj in (lin, nonlin):
        f = obj.growth_rate(zs, ks)
        assert f.shape == (2, 2)
        np.testing.assert_allclose(
            f[:, 0], obj.background.Omega_m(zs) ** 0.55, rtol=3e-2
        )
    with pytest.raises(ValueError, match="ks"):
        lin.growth_rate(zs)


def test_sigma8_close_to_camb():
    from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations
    from cloelib.cosmology.EE3_cosmology import EE3LinearPerturbations

    background = _background()
    s8_ee3 = EE3LinearPerturbations(background, ZS).sigma8_0()
    s8_camb = CAMBLinearPerturbations(background, ZS).sigma8_0()
    assert abs(s8_ee3 / s8_camb - 1) < 5e-3


def test_boost_on_other_linear_backend():
    """The EE3 boost rescales any linear spectrum, as done for EE2."""
    from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations
    from cloelib.cosmology.EE3_cosmology import EE3NonLinearPerturbations

    background = _background()
    camb_lin = CAMBLinearPerturbations(background, ZS)
    nonlin = EE3NonLinearPerturbations(background, camb_lin, ZS)
    ks = np.logspace(-3, 0.5, 20)
    np.testing.assert_allclose(
        nonlin.matter_power_spectrum(ZQ, ks),
        nonlin.nonlinear_boost(ZQ, ks) * camb_lin.matter_power_spectrum(ZQ, ks),
        rtol=1e-10,
    )


def test_growth_from_emulator(ee3):
    """The growth factors are the EE3 growth emulator (emu.lin.growth), only
    interpolated in k, and agree with sqrt(P(z)/P(0)) of the linear emulator to
    the emulation accuracy."""
    from cloelib.cosmology.EE3_cosmology import emu

    lin, _ = ee3
    ks = np.asarray(emu.lin.k)[50:450] * h  # native nodes: no k interpolation
    for kind, method, P in (
        ("total", lin.growth_factor, lin.Pk_interp),
        ("cold", lin.growth_factor_cb, lin.Pk_cb_interp),
    ):
        D = method(ZQ, ks)
        ref = emu.lin.growth(lin.params_emu, ZQ, kind=kind)[:, 50:450]
        np.testing.assert_allclose(D, ref, rtol=1e-12)
        np.testing.assert_allclose(D[0], 1.0, atol=2e-4)  # ZQ[0] = 0
        np.testing.assert_allclose(D, np.sqrt(P(ZQ, ks) / P(0.0, ks)), rtol=1e-3)


def test_growth_beyond_emulator_range():
    """Above z = 9 the growth continues as D(9) sqrt(P(z) / P(9)), with the
    extrapolated linear spectrum: continuous at z = 9."""
    from cloelib.cosmology.EE3_cosmology import EE3LinearPerturbations

    lin = EE3LinearPerturbations(_background(), np.linspace(0.0, 12.0, 61))
    ks = np.array([1e-3, 0.1, 1.0])
    D = lin.growth_factor([9.0, 9.0 + 1e-8, 12.0], ks)
    np.testing.assert_allclose(D[1], D[0], rtol=1e-7)
    np.testing.assert_allclose(
        D[2] / D[0],
        np.sqrt(lin.Pk_interp(12.0, ks) / lin.Pk_interp(9.0, ks))[0],
        rtol=1e-12,
    )


def test_boost_at_every_redshift(ee3):
    """The boost is emulated at every requested redshift in one call (no
    sub-sampling): exact against emu.sim.lin_boost on the native grid."""
    from cloelib.cosmology.EE3_cosmology import emu

    _, nonlin = ee3
    np.testing.assert_array_equal(nonlin.z, ZS)
    np.testing.assert_allclose(
        nonlin.nonlinear_boost(ZS, emu.k * h),
        emu.sim.lin_boost(nonlin.params_emu, ZS),
        rtol=1e-10,
    )


@pytest.mark.parametrize("kind", ["cold", "total"])
def test_hmcode_reproduces_emulator(ee3_all, kind):
    """With EE3 linear input the HMcode backend is the emulated HMcode-2020,
    emu.hmc.pk, on the native k grid."""
    from cloelib.cosmology.EE3_cosmology import emu

    _, _, hmcode = ee3_all
    method = (
        hmcode.matter_power_spectrum_cb
        if kind == "cold"
        else hmcode.matter_power_spectrum
    )
    ks = np.asarray(emu.hmc.k)[50:450] * h
    if kind == "total" and hmcode.f_nu == 0.0:
        kind = "cold"  # massless: total == cold by construction
    ref = emu.hmc.pk(hmcode.params_emu, ZQ, kind=kind)[:, 50:450] / h**3
    np.testing.assert_allclose(method(ZQ, ks), ref, rtol=1e-5)


def test_hmcode_on_other_linear_backend():
    """The HMcode boosts rescale any linear spectrum."""
    from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations
    from cloelib.cosmology.EE3_cosmology import EE3HMcodePerturbations

    background = _background(0.12)
    camb_lin = CAMBLinearPerturbations(background, ZS)
    hmcode = EE3HMcodePerturbations(background, camb_lin, ZS)
    ks = np.logspace(-3, 0.5, 20)
    np.testing.assert_allclose(
        hmcode.matter_power_spectrum_cb(ZQ, ks),
        hmcode.nonlinear_boost(ZQ, ks) * camb_lin.matter_power_spectrum_cb(ZQ, ks),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        hmcode.matter_power_spectrum(ZQ, ks),
        hmcode.nonlinear_boost_mm(ZQ, ks) * camb_lin.matter_power_spectrum(ZQ, ks),
        rtol=1e-10,
    )
    # HMcode is linear on large scales
    np.testing.assert_allclose(hmcode.nonlinear_boost(ZQ, [1e-3]), 1.0, atol=2e-3)


def test_out_of_box_raises():
    from cloelib.cosmology.camb_cosmology import CAMBBackground
    from cloelib.cosmology.EE3_cosmology import (
        EE3LinearPerturbations,
        EE3NonLinearPerturbations,
    )

    bad = CAMBBackground(
        H0=H0,
        Omega_b0=0.022 / h**2,
        Omega_cdm0=0.12 / h**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.0,
        w0=-2.5,  # outside the EE3 l-space box (-2 <= w0 <= 0)
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0,
    )
    with pytest.raises(ValueError):
        lin = EE3LinearPerturbations(bad, ZS)
        EE3NonLinearPerturbations(bad, lin, ZS)


def test_boost_error(ee3):
    _, nonlin = ee3
    err = nonlin.boost_error([0.5, 7.0], [0.1, 1.0])
    assert err.shape == (2, 2)
    assert np.all(err[0] > 0) and np.all(err[0] < 0.05)
    assert np.all(err[1] == 0.0)  # outside the emulated redshift range
