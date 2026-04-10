import pytest
import numpy as np

from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.EE2_cosmology import EE2NonLinearPerturbations


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
def EE2_perturbation_instance(camb_background_instance, zs, scope="module"):
    """Fixture to create an EE2NonLinearPerturbations instance."""
    return EE2NonLinearPerturbations(
        background=camb_background_instance, redshifts=zs
    )


def test_EE2_perturbation_implements_protocol(EE2_perturbation_instance):
    """Test that the Perturbation instances adhere to the protocol."""
    assert isinstance(EE2_perturbation_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


def test_EE2_matter_power_spectrum(EE2_perturbation_instance, zs, ks):
    """Test EE2 matter_power_spectrum."""
    assert hasattr(EE2_perturbation_instance, "matter_power_spectrum")
    assert callable(EE2_perturbation_instance.matter_power_spectrum)
    result = EE2_perturbation_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


def test_EE2_growth_factor(EE2_perturbation_instance, zs, ks):
    """Test EE2 growth_factor."""
    assert hasattr(EE2_perturbation_instance, "growth_factor")
    assert callable(EE2_perturbation_instance.growth_factor)
    result = EE2_perturbation_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


def test_camb_growth_rate(EE2_perturbation_instance, zs, ks):
    """Test EE2 growth_rate."""
    assert hasattr(EE2_perturbation_instance, "growth_rate")
    assert callable(EE2_perturbation_instance.growth_rate)
    result = EE2_perturbation_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
