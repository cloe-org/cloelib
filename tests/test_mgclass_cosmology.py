import pytest
import numpy as np

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.mgclass_cosmology import (
    MGCLASSBackground,
    MGCLASSLinearPerturbations,
    MGCLASSNonLinearPerturbations,
)


@pytest.fixture
def mgclass_background_instance(scope="module"):
    """Fixture to create an instance of MGCLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    mgclass_instance = MGCLASSBackground(
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
        mg_ansatz="plk_musigma_norm_late",
        mg_z_init=0.0,
        mg_params={"mg_E11": 0.0, "mg_E22": 0.0},
        N_mnu=0,
    )
    return mgclass_instance


def test_mgclass_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {
        name
        for name, value in Background.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    contents = MGCLASSBackground.__dict__.items()
    methods_found = {
        name for name, value in contents if callable(value) and not name.startswith("_")
    }
    assert methods_required <= methods_found


def test_mgclass_background_required_attributes(mgclass_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {
        name
        for name, value in Background.__dict__.items()
        if not callable(value) and not name.startswith("_")
    }
    contents = (
        (name, getattr(mgclass_background_instance, name))
        for name in dir(mgclass_background_instance)
    )
    attributes_found = {
        name
        for name, value in contents
        if not callable(value) and not name.startswith("_")
    }
    assert attributes_required <= attributes_found


def test_mgclass_background_implements_protocol(mgclass_background_instance):
    """Test that the MGCLASSBackground instance adheres to the Background protocol."""
    assert isinstance(mgclass_background_instance, Background)


def test_mgclass_background_H0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "H0")
    assert isinstance(mgclass_background_instance.H0, float)
    assert mgclass_background_instance.H0 == 67.7


def test_mgclass_background_h(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "h")
    assert isinstance(mgclass_background_instance.h, float)
    assert mgclass_background_instance.h == 0.677


def test_mgclass_background_Omega_b0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "Omega_b0")
    assert isinstance(mgclass_background_instance.Omega_b0, float)
    assert mgclass_background_instance.Omega_b0 == 0.022 / (0.677 * 0.677)


def test_mgclass_background_Omega_cdm0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "Omega_cdm0")
    assert isinstance(mgclass_background_instance.Omega_cdm0, float)
    assert mgclass_background_instance.Omega_cdm0 == 0.12 / (0.677 * 0.677)


def test_mgclass_background_mnu(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "mnu")
    assert isinstance(mgclass_background_instance.mnu, float)
    assert mgclass_background_instance.mnu == 0.0


def test_mgclass_background_Omega_k0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "Omega_k0")
    assert isinstance(mgclass_background_instance.Omega_k0, float)
    assert mgclass_background_instance.Omega_k0 == 0.0


def test_mgclass_background_As(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "As")
    assert isinstance(mgclass_background_instance.As, float)
    assert mgclass_background_instance.As == 2e-9


def test_mgclass_background_ns(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "ns")
    assert isinstance(mgclass_background_instance.ns, float)
    assert mgclass_background_instance.ns == 0.96


def test_mgclass_background_w0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "w0")
    assert isinstance(mgclass_background_instance.w0, float)
    assert mgclass_background_instance.w0 == -1


def test_mgclass_background_wa(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "wa")
    assert isinstance(mgclass_background_instance.wa, float)
    assert mgclass_background_instance.wa == 0.0


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


def test_mgclass_omega_m(mgclass_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(mgclass_background_instance, "Omega_m")
    assert callable(mgclass_background_instance.Omega_m)
    result = mgclass_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_mgclass_hubble_parameter(mgclass_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(mgclass_background_instance, "hubble_parameter")
    assert callable(mgclass_background_instance.hubble_parameter)
    result = mgclass_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - mgclass_background_instance.H0) < 1e-4


def test_mgclass_comoving_distance(mgclass_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(mgclass_background_instance, "comoving_distance")
    assert callable(mgclass_background_instance.comoving_distance)
    result = mgclass_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mgclass_transverse_comoving_distance(mgclass_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(mgclass_background_instance, "transverse_comoving_distance")
    assert callable(mgclass_background_instance.transverse_comoving_distance)
    result = mgclass_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mgclass_angular_diameter_distance(mgclass_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(mgclass_background_instance, "angular_diameter_distance")
    assert callable(mgclass_background_instance.angular_diameter_distance)
    result = mgclass_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.fixture
def mgclass_perturbation_instances(mgclass_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of MGCLASSPerturbations."""
    mgclass_lin = MGCLASSLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )
    mgclass_non = MGCLASSNonLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs, nonlinear_model="halofit"
    )
    return {"Linear": mgclass_lin, "NonLinear": mgclass_non}


@pytest.mark.parametrize("key", ["Linear"])
def test_mgclass_perturbation_implements_protocol(mgclass_perturbation_instances, key):
    """Test that the MGCLASSPerturbation instances adhere to the protocol."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert isinstance(mgclass_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_mgclass_matter_power_spectrum(mgclass_perturbation_instances, key, zs, ks):
    """Test MGCLASS matter_power_spectrum."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert hasattr(mgclass_instance, "matter_power_spectrum")
    assert callable(mgclass_instance.matter_power_spectrum)
    result = mgclass_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_mgclass_growth_factor(mgclass_perturbation_instances, key, zs, ks):
    """Test MGCLASS growth_factor."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert hasattr(mgclass_instance, "growth_factor")
    assert callable(mgclass_instance.growth_factor)
    result = mgclass_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear"])
def test_mgclass_growth_rate(mgclass_perturbation_instances, key, zs, ks):
    """Test MGCLASS growth_rate."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert hasattr(mgclass_instance, "growth_rate")
    assert callable(mgclass_instance.growth_rate)
    result = mgclass_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
