import pytest
import numpy as np

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.class_cosmology import CLASSBackground, CLASSLinearPerturbations, CLASSNonLinearPerturbations

@pytest.fixture
def class_background_instance(scope="module"):
    """Fixture to create an instance of CLASSBackground."""
    H0 =  67.7
    h = H0/100.
    omch2 = 0.12
    Omega_cdm0 = omch2/h**2
    ombh2 = 0.022
    Omega_b0 = ombh2/h**2
    class_instance = CLASSBackground(H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0, 
                                   Omega_k0=0.,
                                   As=2e-9, ns=0.96, mnu=0., w0=-1.0, wa=0.0, 
                                   gamma_MG=0.0, N_mnu=0)
    return class_instance

def test_class_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {name for name, value in Background.__dict__.items() if callable(value) and not name.startswith('_')}
    contents = CLASSBackground.__dict__.items()
    methods_found = {name for name, value in contents if callable(value) and not name.startswith('_')}
    assert methods_required <= methods_found

def test_class_background_required_attributes(class_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {name for name, value in Background.__dict__.items() if not callable(value) and not name.startswith('_')}
    contents = ((name, getattr(class_background_instance, name)) for name in dir(class_background_instance))
    attributes_found = {name for name, value in contents if not callable(value) and not name.startswith('_')}
    assert attributes_required <= attributes_found

def test_class_background_implements_protocol(class_background_instance):
    """Test that the CLASSBackground instance adheres to the Background protocol."""
    assert isinstance(class_background_instance, Background)

def test_class_background_H0(class_background_instance):
    assert hasattr(class_background_instance, 'H0')
    assert isinstance(class_background_instance.H0, float)
    assert class_background_instance.H0 == 67.7

def test_class_background_H0(class_background_instance):
    assert hasattr(class_background_instance, 'h')
    assert isinstance(class_background_instance.h, float)
    assert class_background_instance.h == 0.677

def test_class_background_Omega_b0(class_background_instance):
    assert hasattr(class_background_instance, 'Omega_b0')
    assert isinstance(class_background_instance.Omega_b0, float)
    assert class_background_instance.Omega_b0 == 0.022 / (0.677*0.677)

def test_class_background_Omega_cdm0(class_background_instance):
    assert hasattr(class_background_instance, 'Omega_cdm0')
    assert isinstance(class_background_instance.Omega_cdm0, float)
    assert class_background_instance.Omega_cdm0 == 0.12 / (0.677*0.677)

def test_class_background_mnu(class_background_instance):
    assert hasattr(class_background_instance, 'mnu')
    assert isinstance(class_background_instance.mnu, float)
    assert class_background_instance.mnu == 0.

def test_class_background_Omega_k0(class_background_instance):
    assert hasattr(class_background_instance, 'Omega_k0')
    assert isinstance(class_background_instance.Omega_k0, float)
    assert class_background_instance.Omega_k0 == 0.

def test_class_background_As(class_background_instance):
    assert hasattr(class_background_instance, 'As')
    assert isinstance(class_background_instance.As, float)
    assert class_background_instance.As == 2e-9

def test_class_background_ns(class_background_instance):
    assert hasattr(class_background_instance, 'ns')
    assert isinstance(class_background_instance.ns, float)
    assert class_background_instance.ns == 0.96

def test_class_background_w0(class_background_instance):
    assert hasattr(class_background_instance, 'w0')
    assert isinstance(class_background_instance.w0, float)
    assert class_background_instance.w0 == -1

def test_class_background_wa(class_background_instance):
    assert hasattr(class_background_instance, 'wa')
    assert isinstance(class_background_instance.wa, float)
    assert class_background_instance.wa == 0.

@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)

def test_class_omega_m(class_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(class_background_instance, 'Omega_m')
    assert callable(class_background_instance.Omega_m)
    result = class_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)

def test_class_omega_b(class_background_instance, zs):
    """Test Omega_b. 

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(class_background_instance, 'Omega_b')
    assert callable(class_background_instance.Omega_b)
    result = class_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    assert np.abs(result[0] - class_background_instance.Omega_b0) < 1e-4

@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_class_hubble_parameter(class_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(class_background_instance, 'hubble_parameter')
    assert callable(class_background_instance.hubble_parameter)
    result = class_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - class_background_instance.H0) < 1e-4

def test_class_comoving_distance(class_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(class_background_instance, 'comoving_distance')
    assert callable(class_background_instance.comoving_distance)
    result = class_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)

def test_class_transverse_comoving_distance(class_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(class_background_instance, 'transverse_comoving_distance')
    assert callable(class_background_instance.transverse_comoving_distance)
    result = class_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)

def test_class_angular_diameter_distance(class_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(class_background_instance, 'angular_diameter_distance')
    assert callable(class_background_instance.angular_diameter_distance)
    result = class_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.fixture
def class_perturbation_instances(class_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of CLASSPerturbations."""
    class_lin = CLASSLinearPerturbations(background=class_background_instance, redshifts=zs)
    class_non = CLASSNonLinearPerturbations(background=class_background_instance, redshifts=zs,
                                            nonlinear_model='halofit')
    return {"Linear": class_lin, "NonLinear": class_non}

@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_perturbation_implements_protocol(class_perturbation_instances, key):
    """Test that the CLASSPerturbation instances adhere to the protocol."""
    class_instance = class_perturbation_instances[key]
    assert isinstance(class_instance, Perturbations)

@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)

@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_matter_power_spectrum(class_perturbation_instances, key, zs, ks):
    """Test CLASS matter_power_spectrum."""
    class_instance = class_perturbation_instances[key]
    assert hasattr(class_instance, 'matter_power_spectrum')
    assert callable(class_instance.matter_power_spectrum)
    result = class_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))

@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_growth_factor(class_perturbation_instances, key, zs, ks):
    """Test CLASS growth_factor."""
    class_instance = class_perturbation_instances[key]
    assert hasattr(class_instance, 'growth_factor')
    assert callable(class_instance.growth_factor)
    result = class_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))

@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_growth_rate(class_perturbation_instances, key, zs, ks):
    """Test CLASS growth_rate."""
    class_instance = class_perturbation_instances[key]
    assert hasattr(class_instance, 'growth_rate')
    assert callable(class_instance.growth_rate)
    result = class_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
