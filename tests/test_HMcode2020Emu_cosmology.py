"""Unit tests for the HMcode2020Emu perturbations backend."""

import numpy as np
import pytest

pytest.importorskip("HMcode2020Emu")

from cloelib.cosmology.cosmology import Perturbations  # noqa: E402
from cloelib.cosmology.camb_cosmology import (  # noqa: E402
    CAMBBackground,
    CAMBLinearPerturbations,
)
from cloelib.cosmology.HMcode2020Emu_cosmology import (  # noqa: E402
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)

H0 = 67.7
h = H0 / 100.0

# HMcode2020Emu only supports one massive neutrino species with
# N_ur = 2.0308 (N_eff = 3.044), which CAMBBackground infers for N_mnu=1.
BACKGROUND_KWARGS = dict(
    H0=H0,
    Omega_b0=0.022 / h**2,
    Omega_cdm0=0.12 / h**2,
    Omega_k0=0.0,
    As=2e-9,
    ns=0.96,
    mnu=0.06,
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.0,
    N_mnu=1,
)


@pytest.fixture(scope="module")
def camb_background_instance():
    """Fixture to create an instance of CAMBBackground."""
    return CAMBBackground(**BACKGROUND_KWARGS)


@pytest.fixture(scope="module")
def zs():
    return np.linspace(0, 2, 20)


@pytest.fixture(scope="module")
def ks():
    return np.logspace(-3, 0.5, 30)


@pytest.fixture(scope="module")
def hmemu_perturbation_instance(camb_background_instance, zs):
    """Fixture to create the Linear and NonLinear instances of HMemuPerturbations."""
    hmemu_lin = HMemuLinearPerturbations(
        background=camb_background_instance, redshifts=zs
    )
    hmemu_nonlin = HMemuNonLinearPerturbations(
        background=camb_background_instance,
        linearperturbations=hmemu_lin,
        redshifts=zs,
    )
    return {"Linear": hmemu_lin, "NonLinear": hmemu_nonlin}


@pytest.fixture(scope="module")
def camb_linear_instance(camb_background_instance, zs):
    """Fixture to create a CAMB linear reference for the emulator."""
    return CAMBLinearPerturbations(background=camb_background_instance, redshifts=zs)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_perturbation_implements_protocol(hmemu_perturbation_instance, key):
    """Test that the Perturbation instances adhere to the protocol."""
    assert isinstance(hmemu_perturbation_instance[key], Perturbations)


@pytest.mark.parametrize(
    "method",
    [
        "matter_power_spectrum",
        "matter_power_spectrum_cb",
        "growth_factor",
        "growth_factor_cb",
    ],
)
@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_zk_methods_shape(hmemu_perturbation_instance, key, method, zs, ks):
    """Test that (z, k) methods return finite, positive (nz, nk) arrays."""
    result = getattr(hmemu_perturbation_instance[key], method)(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(zs), len(ks))
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_growth_factor_normalised_at_z0(hmemu_perturbation_instance, key, ks):
    """Test that the growth factor is normalised to 1 at z=0 and decreases with z."""
    perturbations = hmemu_perturbation_instance[key]
    D0 = perturbations.growth_factor(np.array([0.0]), ks)
    np.testing.assert_allclose(D0, 1.0, rtol=1e-12)
    D = perturbations.growth_factor(np.array([0.0, 1.0, 2.0]), np.array([1e-3]))
    assert np.all(np.diff(D[:, 0]) < 0)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_sigma8_0(hmemu_perturbation_instance, camb_linear_instance, key):
    """Test that sigma8_0 is a float consistent with CAMB."""
    result = hmemu_perturbation_instance[key].sigma8_0()
    assert isinstance(result, (float, np.floating))
    np.testing.assert_allclose(result, camb_linear_instance.sigma8_0(), rtol=5e-3)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_growth_rate(hmemu_perturbation_instance, camb_background_instance, key):
    """Test that the growth rate is close to the Omega_m(z)^0.55 approximation."""
    perturbations = hmemu_perturbation_instance[key]
    f = perturbations.growth_rate()
    assert f.shape == perturbations.z.shape
    Om_z = camb_background_instance.Omega_m(perturbations.z)
    np.testing.assert_allclose(f, Om_z**0.55, rtol=2e-2)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_growth_rate_at_requested_redshifts(hmemu_perturbation_instance, key, ks):
    """Test that growth_rate follows the protocol's optional (zs, ks) signature."""
    perturbations = hmemu_perturbation_instance[key]
    z_grid = perturbations.z
    np.testing.assert_allclose(
        perturbations.growth_rate(z_grid), perturbations.growth_rate()
    )
    zs = np.array([0.3, 1.7, 5.0])
    f = perturbations.growth_rate(zs)
    assert f.shape == (len(zs),)
    assert np.all((f > 0) & (f <= 1))
    f_zk = perturbations.growth_rate(zs, ks)
    assert f_zk.shape == (len(zs), len(ks))
    np.testing.assert_allclose(f_zk, np.tile(f[:, None], (1, len(ks))))


@pytest.mark.parametrize(
    "method", ["matter_power_spectrum", "matter_power_spectrum_cb", "growth_factor"]
)
@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_hmemu_single_redshift_keeps_z_axis(
    hmemu_perturbation_instance, key, method, ks
):
    """Test that a length-1 redshift array returns (1, nk), as for CAMB."""
    result = getattr(hmemu_perturbation_instance[key], method)(np.array([0.5]), ks)
    assert result.shape == (1, len(ks))


def test_hmemu_linear_matches_camb(hmemu_perturbation_instance, camb_linear_instance):
    """Test that the linear emulator reproduces CAMB within its range of validity."""
    zs = np.linspace(0, 2, 5)
    ks = np.logspace(-3, 0.5, 30)
    lin = hmemu_perturbation_instance["Linear"]
    np.testing.assert_allclose(
        lin.matter_power_spectrum(zs, ks),
        camb_linear_instance.matter_power_spectrum(zs, ks),
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        lin.matter_power_spectrum_cb(zs, ks),
        camb_linear_instance.matter_power_spectrum_cb(zs, ks),
        rtol=1e-2,
    )


def test_hmemu_cb_exceeds_total_with_massive_neutrinos(hmemu_perturbation_instance):
    """Test that neutrino free-streaming suppresses P_mm relative to P_cb."""
    lin = hmemu_perturbation_instance["Linear"]
    ks = np.array([1e-1, 1.0])
    ratio = lin.matter_power_spectrum_cb(np.array([0.0]), ks) / (
        lin.matter_power_spectrum(np.array([0.0]), ks)
    )
    assert np.all(ratio > 1.0)


def test_hmemu_nonlinear_matches_linear_on_large_scales(hmemu_perturbation_instance):
    """Test that the nonlinear spectrum reduces to linear at low k and is boosted at high k."""
    lin = hmemu_perturbation_instance["Linear"]
    nonlin = hmemu_perturbation_instance["NonLinear"]
    z = np.array([0.0])
    k_low = np.logspace(-3, -2.5, 5)
    np.testing.assert_allclose(
        nonlin.matter_power_spectrum(z, k_low),
        lin.matter_power_spectrum(z, k_low),
        rtol=1e-3,
    )
    k_high = np.array([1.0, 3.0])
    assert np.all(
        nonlin.matter_power_spectrum(z, k_high) > lin.matter_power_spectrum(z, k_high)
    )


def test_hmemu_baryonic_boost_suppresses_small_scales(
    camb_background_instance, hmemu_perturbation_instance, zs
):
    """Test that the log10TAGN baryonic feedback suppresses small-scale power."""
    lin = hmemu_perturbation_instance["Linear"]
    nonlin = hmemu_perturbation_instance["NonLinear"]
    nonlin_baryons = HMemuNonLinearPerturbations(
        background=camb_background_instance,
        linearperturbations=lin,
        redshifts=zs,
        log10TAGN=7.8,
    )
    z = np.array([0.0])
    ratio_low = nonlin_baryons.matter_power_spectrum(
        z, np.array([1e-3])
    ) / nonlin.matter_power_spectrum(z, np.array([1e-3]))
    np.testing.assert_allclose(ratio_low, 1.0, rtol=1e-3)
    k_high = np.array([1.0, 3.0])
    ratio_high = nonlin_baryons.matter_power_spectrum(
        z, k_high
    ) / nonlin.matter_power_spectrum(z, k_high)
    assert np.all(ratio_high < 1.0)


def test_hmemu_extrapolates_beyond_emulator_redshift(camb_background_instance, ks):
    """Test that redshifts above the emulator range (z > 4) are extrapolated."""
    zs = np.linspace(0, 5, 11)
    lin = HMemuLinearPerturbations(background=camb_background_instance, redshifts=zs)
    assert lin.z.max() == pytest.approx(5.0)
    result = lin.matter_power_spectrum(zs, ks)
    assert result.shape == (len(zs), len(ks))
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)


def test_hmemu_out_of_range_raises(zs):
    """Test that parameters outside the emulator bounds raise a ValueError."""
    background = CAMBBackground(**{**BACKGROUND_KWARGS, "w0": -3.5})
    with pytest.raises(ValueError, match="out of range"):
        HMemuLinearPerturbations(background=background, redshifts=zs)


def test_hmemu_multiple_massive_neutrinos_raises(zs):
    """Test that more than one massive neutrino species is rejected."""
    background = CAMBBackground(
        **{**BACKGROUND_KWARGS, "mnu": [0.03, 0.03], "N_mnu": 2}
    )
    with pytest.raises(ValueError, match="single species"):
        HMemuLinearPerturbations(background=background, redshifts=zs)


def test_hmemu_non_flat_raises(zs):
    """Test that non-flat geometries are rejected."""
    background = CAMBBackground(**{**BACKGROUND_KWARGS, "Omega_k0": 0.01})
    with pytest.raises(AssertionError, match="Non flat"):
        HMemuLinearPerturbations(background=background, redshifts=zs)
