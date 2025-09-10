import pytest
import jax.numpy as np

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)


@pytest.fixture
def jax_background_instance(scope="module"):
    """Fixture to create an instance of JAXBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    jax_instance = JAXBackground(
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
    )
    return jax_instance


def test_jax_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {
        name
        for name, value in Background.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    contents = JAXBackground.__dict__.items()
    methods_found = {
        name for name, value in contents if callable(value) and not name.startswith("_")
    }
    assert methods_required <= methods_found


def test_jax_background_required_attributes(jax_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {
        name
        for name, value in Background.__dict__.items()
        if not callable(value) and not name.startswith("_")
    }
    contents = (
        (name, getattr(jax_background_instance, name))
        for name in dir(jax_background_instance)
    )
    attributes_found = {
        name
        for name, value in contents
        if not callable(value) and not name.startswith("_")
    }
    assert attributes_required <= attributes_found


def test_jax_background_implements_protocol(jax_background_instance):
    """Test that the JAXBackground instance adheres to the Background protocol."""
    assert isinstance(jax_background_instance, Background)


def test_jax_background_H0(jax_background_instance):
    assert hasattr(jax_background_instance, "H0")
    assert isinstance(jax_background_instance.H0, float)
    assert jax_background_instance.H0 == 67.7


def test_jax_background_h(jax_background_instance):
    assert hasattr(jax_background_instance, "h")
    assert isinstance(jax_background_instance.h, float)
    assert jax_background_instance.h == 0.677


def test_jax_background_Omega_b0(jax_background_instance):
    assert hasattr(jax_background_instance, "Omega_b0")
    assert isinstance(jax_background_instance.Omega_b0, float)
    assert jax_background_instance.Omega_b0 == 0.022 / (0.677 * 0.677)


def test_jax_background_Omega_cdm0(jax_background_instance):
    assert hasattr(jax_background_instance, "Omega_cdm0")
    assert isinstance(jax_background_instance.Omega_cdm0, float)
    assert jax_background_instance.Omega_cdm0 == 0.12 / (0.677 * 0.677)


def test_jax_background_mnu(jax_background_instance):
    assert hasattr(jax_background_instance, "mnu")
    assert isinstance(jax_background_instance.mnu, float)
    assert jax_background_instance.mnu == 0.0


def test_jax_background_Omega_k0(jax_background_instance):
    assert hasattr(jax_background_instance, "Omega_k0")
    assert isinstance(jax_background_instance.Omega_k0, float)
    assert jax_background_instance.Omega_k0 == 0.0


def test_jax_background_As(jax_background_instance):
    assert hasattr(jax_background_instance, "As")
    assert isinstance(jax_background_instance.As, float)
    assert jax_background_instance.As == 2e-9


def test_jax_background_ns(jax_background_instance):
    assert hasattr(jax_background_instance, "ns")
    assert isinstance(jax_background_instance.ns, float)
    assert jax_background_instance.ns == 0.96


def test_jax_background_w0(jax_background_instance):
    assert hasattr(jax_background_instance, "w0")
    assert isinstance(jax_background_instance.w0, float)
    assert jax_background_instance.w0 == -1


def test_jax_background_wa(jax_background_instance):
    assert hasattr(jax_background_instance, "wa")
    assert isinstance(jax_background_instance.wa, float)
    assert jax_background_instance.wa == 0.0


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


def test_jax_omega_m(jax_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(jax_background_instance, "Omega_m")
    assert callable(jax_background_instance.Omega_m)
    result = jax_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)


def test_jax_omega_b(jax_background_instance, zs):
    """Test Omega_b.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(jax_background_instance, "Omega_b")
    assert callable(jax_background_instance.Omega_b)
    result = jax_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)
    assert np.abs(result[0] - jax_background_instance.Omega_b0) < 1e-4


@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_jax_hubble_parameter(jax_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(jax_background_instance, "hubble_parameter")
    assert callable(jax_background_instance.hubble_parameter)
    result = jax_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - jax_background_instance.H0) < 1e-4


def test_jax_comoving_distance(jax_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(jax_background_instance, "comoving_distance")
    assert callable(jax_background_instance.comoving_distance)
    result = jax_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)


def test_jax_transverse_comoving_distance(jax_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(jax_background_instance, "transverse_comoving_distance")
    assert callable(jax_background_instance.transverse_comoving_distance)
    result = jax_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)


def test_jax_angular_diameter_distance(jax_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(jax_background_instance, "angular_diameter_distance")
    assert callable(jax_background_instance.angular_diameter_distance)
    result = jax_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)


@pytest.fixture
def jax_perturbation_instances(jax_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of jaxPerturbations."""
    jax_lin = JAXLinearPerturbations(background=jax_background_instance)
    jax_non = JAXNonLinearPerturbations(background=jax_background_instance)
    return {"Linear": jax_lin, "NonLinear": jax_non}


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_jax_perturbation_implements_protocol(jax_perturbation_instances, key):
    """Test that the CAMBPerturbation instances adhere to the protocol."""
    jax_instance = jax_perturbation_instances[key]
    assert isinstance(jax_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_jax_matter_power_spectrum(jax_perturbation_instances, key, zs, ks):
    """Test JAX matter_power_spectrum."""
    jax_instance = jax_perturbation_instances[key]
    assert hasattr(jax_instance, "matter_power_spectrum")
    assert callable(jax_instance.matter_power_spectrum)
    result = jax_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_jax_growth_factor(jax_perturbation_instances, key, zs, ks):
    """Test JAX growth_factor."""
    jax_instance = jax_perturbation_instances[key]
    assert hasattr(jax_instance, "growth_factor")
    assert callable(jax_instance.growth_factor)
    result = jax_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    # allow the following test after jax cosmology homogenization
    # assert result.ndim == 2
    # assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_jax_growth_rate(jax_perturbation_instances, key, zs, ks):
    """Test JAX growth_rate."""
    jax_instance = jax_perturbation_instances[key]
    assert hasattr(jax_instance, "growth_rate")
    assert callable(jax_instance.growth_rate)
    result = jax_instance.growth_rate(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
