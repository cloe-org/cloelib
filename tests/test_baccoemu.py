import pytest
import numpy as np

from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.baccoemu_cosmology import (
    BACCOemuLinearPerturbations,
    BACCOemuNonLinearPerturbations,
)


@pytest.fixture
def camb_background_instance(scope="module"):
    """Fixture to create an instance of CAMBBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    camb_instance = CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0,
    )
    return camb_instance


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


@pytest.fixture
def baccoemu_perturbation_instance(camb_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of BACCOemuPerturbations."""
    baccoemu_lin = BACCOemuLinearPerturbations(
        background=camb_background_instance, redshifts=zs
    )
    baccoemu_nonlin = BACCOemuNonLinearPerturbations(
        background=camb_background_instance,
        linearperturbations=baccoemu_lin,
        redshifts=zs,
    )
    return {"Linear": baccoemu_lin, "NonLinear": baccoemu_nonlin}


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_perturbation_implements_protocol(baccoemu_perturbation_instance, key):
    """Test that the Perturbation instances adhere to the protocol."""
    assert isinstance(baccoemu_perturbation_instance[key], Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_matter_power_spectrum(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu matter_power_spectrum."""
    assert hasattr(baccoemu_perturbation_instance[key], "matter_power_spectrum")
    assert callable(baccoemu_perturbation_instance[key].matter_power_spectrum)
    result = baccoemu_perturbation_instance[key].matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_matter_power_spectrum_cb(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu matter_power_spectrum_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "matter_power_spectrum_cb")
    assert callable(baccoemu_perturbation_instance[key].matter_power_spectrum_cb)
    result = baccoemu_perturbation_instance[key].matter_power_spectrum_cb(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_factor(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_factor."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_factor")
    assert callable(baccoemu_perturbation_instance[key].growth_factor)
    result = baccoemu_perturbation_instance[key].growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_factor_cb(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_factor_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_factor_cb")
    assert callable(baccoemu_perturbation_instance[key].growth_factor_cb)
    result = baccoemu_perturbation_instance[key].growth_factor_cb(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_rate(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_rate."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_rate")
    assert callable(baccoemu_perturbation_instance[key].growth_rate)
    result = baccoemu_perturbation_instance[key].growth_rate(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_rate_cb(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_rate_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_rate_cb")
    assert callable(baccoemu_perturbation_instance[key].growth_rate_cb)
    result = baccoemu_perturbation_instance[key].growth_rate_cb(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma8_0(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma8_0."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma8_0")
    assert callable(baccoemu_perturbation_instance[key].sigma8_0)
    result = baccoemu_perturbation_instance[key].sigma8_0()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma8_0_cb(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma8_0_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma8_0_cb")
    assert callable(baccoemu_perturbation_instance[key].sigma8_0)
    result = baccoemu_perturbation_instance[key].sigma8_0_cb()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma12_0(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma12_0."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma12_0")
    assert callable(baccoemu_perturbation_instance[key].sigma12_0)
    result = baccoemu_perturbation_instance[key].sigma12_0()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma12_0_cb(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma12_0_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma12_0_cb")
    assert callable(baccoemu_perturbation_instance[key].sigma12_0_cb)
    result = baccoemu_perturbation_instance[key].sigma12_0_cb()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value
