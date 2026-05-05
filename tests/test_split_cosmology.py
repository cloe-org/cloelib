import pytest
import numpy as np

from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology.class_cosmology import (
    CLASSBackground,
    CLASSLinearPerturbations,
    CLASSNonLinearPerturbations,
)
from cloelib.cosmology.split_cosmology import (
    SplitLinearPerturbations,
    SplitNonLinearPerturbations,
)


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.fixture
def class_background_instance_geo(scope="module"):
    """Fixture to create an instance of CLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    Omega_m = 0.3
    Omega_cdm0 = Omega_m - Omega_b0
    class_instance = CLASSBackground(
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
    return class_instance


@pytest.fixture
def class_background_instance_growth(scope="module"):
    """Fixture to create an instance of CLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    Omega_m = 0.35
    Omega_cdm0 = Omega_m - Omega_b0
    class_instance = CLASSBackground(
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
    return class_instance


@pytest.fixture
def class_perturbation_instances_geo(class_background_instance_geo, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of
    CLASSPerturbations which will stand in for the geometric regime in this
    test with a test value of Omega_m = 0.3."""
    class_lin = CLASSLinearPerturbations(
        background=class_background_instance_geo, redshifts=zs
    )
    class_non = CLASSNonLinearPerturbations(
        background=class_background_instance_geo,
        redshifts=zs,
        nonlinear_model="halofit",
    )
    return {"Linear": class_lin, "NonLinear": class_non}


@pytest.fixture
def class_perturbation_instances_growth(
    class_background_instance_growth, zs, scope="module"
):
    """Fixture to create the Linear and NonLinear instances of
    CLASSPerturbations which will stand in for the growth regime in this test
    with a test value of Omega_m = 0.35."""
    class_lin = CLASSLinearPerturbations(
        background=class_background_instance_growth, redshifts=zs
    )
    class_non = CLASSNonLinearPerturbations(
        background=class_background_instance_growth,
        redshifts=zs,
        nonlinear_model="halofit",
    )
    return {"Linear": class_lin, "NonLinear": class_non}


@pytest.fixture
def omega_m_growth(scope="module"):
    return 0.35


@pytest.fixture
def split_perturbation_instances(
    class_background_instance_geo,
    class_perturbation_instances_geo,
    class_perturbation_instances_growth,
    zs,
    ks,
    omega_m_growth,
    scope="module",
):
    """Fixture to create the Linear and NonLinear instances of
    SplitPerturbations."""
    class_lin_geo = class_perturbation_instances_geo["Linear"]
    class_lin_growth = class_perturbation_instances_growth["Linear"]
    class_nl_growth = class_perturbation_instances_growth["NonLinear"]
    split_lin = SplitLinearPerturbations(
        background=class_background_instance_geo,
        omega_m_growth=omega_m_growth,
        redshifts=zs,
        lin_perturbations=class_lin_geo,
    )
    split_pk_linear = (split_lin.matter_power_spectrum(zs, ks),)
    split_non = SplitNonLinearPerturbations(
        redshifts=zs,
        pk_linear=split_pk_linear,
        perturbations_lin=class_lin_growth,
        perturbations_NL=class_nl_growth,
    )
    return {"Linear": split_lin, "NonLinear": split_non}


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_split_perturbation_implements_protocol(split_perturbation_instances, key):
    """Test that the SplitPerturbation instances adhere to the protocol."""
    split_instance = split_perturbation_instances[key]
    assert isinstance(split_instance, Perturbations)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_split_matter_power_spectrum(split_perturbation_instances, key, zs, ks):
    """Test split matter power spectrum."""
    split_instance = split_perturbation_instances[key]
    assert hasattr(split_instance, "matter_power_spectrum")
    assert callable(split_instance.matter_power_spectrum)
    result = split_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_split_growth_factor(split_perturbation_instances, key, zs, ks):
    """Test split growth_factor."""
    split_instance = split_perturbation_instances[key]
    assert hasattr(split_instance, "growth_factor")
    assert callable(split_instance.growth_factor)
    result = split_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_split_growth_rate(split_perturbation_instances, key):
    """Test split growth_rate."""
    split_instance = split_perturbation_instances[key]
    assert hasattr(split_instance, "growth_rate")
    assert callable(split_instance.growth_rate)
    result = split_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
