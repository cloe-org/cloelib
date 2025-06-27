import pytest
import numpy as np

from cloelib.cosmology.cosmology import Background
from cloelib.cosmology.camb_cosmology import CAMBBackground

@pytest.fixture
def camb_background_instance(scope="module"):
    """Fixture to create an instance of CAMBBackground."""
    H0 =  67.7
    h = H0/100.
    omch2 = 0.12
    Omega_cdm0 = omch2/h**2
    ombh2 = 0.022
    Omega_b0 = ombh2/h**2
    camb_instance = CAMBBackground(H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0, 
                                   Omega_k0=0.,
                                   As=2e-9, ns=0.96, mnu=0., w0=-1.0, wa=0.0, 
                                   gamma_MG=0.0)
    return camb_instance

def test_camb_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {name for name, value in Background.__dict__.items() if callable(value) and not name.startswith('_')}
    contents = CAMBBackground.__dict__.items()
    methods_found = {name for name, value in contents if callable(value) and not name.startswith('_')}
    assert methods_required <= methods_found

def test_camb_background_required_attributes(camb_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {name for name, value in Background.__dict__.items() if not callable(value) and not name.startswith('_')}
    contents = camb_background_instance.__dict__.items()
    attributes_found = {name for name, value in contents if not callable(value) and not name.startswith('_')}
    assert attributes_required <= attributes_found

def test_camb_background_implements_protocol(camb_background_instance):
    """Test that the CAMBBackground instance adheres to the Background protocol."""
    assert isinstance(camb_background_instance, Background)

def test_camb_background_H0(camb_background_instance):
    assert hasattr(camb_background_instance, 'H0')
    assert isinstance(camb_background_instance.H0, float)
    assert camb_background_instance.H0 == 67.7

def test_camb_background_H0(camb_background_instance):
    assert hasattr(camb_background_instance, 'h')
    assert isinstance(camb_background_instance.h, float)
    assert camb_background_instance.h == 0.677

def test_camb_background_Omega_b0(camb_background_instance):
    assert hasattr(camb_background_instance, 'Omega_b0')
    assert isinstance(camb_background_instance.Omega_b0, float)
    assert camb_background_instance.Omega_b0 == 0.022 / (0.677*0.677)

def test_camb_background_Omega_cdm0(camb_background_instance):
    assert hasattr(camb_background_instance, 'Omega_cdm0')
    assert isinstance(camb_background_instance.Omega_cdm0, float)
    assert camb_background_instance.Omega_cdm0 == 0.12 / (0.677*0.677)

def test_camb_background_mnu(camb_background_instance):
    assert hasattr(camb_background_instance, 'mnu')
    assert isinstance(camb_background_instance.mnu, float)
    assert camb_background_instance.mnu == 0.

def test_camb_background_Omega_k0(camb_background_instance):
    assert hasattr(camb_background_instance, 'Omega_k0')
    assert isinstance(camb_background_instance.Omega_k0, float)
    assert camb_background_instance.Omega_k0 == 0.

def test_camb_background_As(camb_background_instance):
    assert hasattr(camb_background_instance, 'As')
    assert isinstance(camb_background_instance.As, float)
    assert camb_background_instance.As == 2e-9

def test_camb_background_ns(camb_background_instance):
    assert hasattr(camb_background_instance, 'ns')
    assert isinstance(camb_background_instance.ns, float)
    assert camb_background_instance.ns == 0.96

def test_camb_background_w0(camb_background_instance):
    assert hasattr(camb_background_instance, 'w0')
    assert isinstance(camb_background_instance.w0, float)
    assert camb_background_instance.w0 == -1

def test_camb_background_wa(camb_background_instance):
    assert hasattr(camb_background_instance, 'wa')
    assert isinstance(camb_background_instance.wa, float)
    assert camb_background_instance.wa == 0.

@pytest.fixture
def zs(scope="module"):
    return np.array([0.0, 0.1, 0.5, 1.0])

def test_omega_m(camb_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(camb_background_instance, 'Omega_m')
    assert callable(camb_background_instance.Omega_m)
    result = camb_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)

def test_omega_b(camb_background_instance, zs):
    """Test Omega_b. 

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(camb_background_instance, 'Omega_b')
    assert callable(camb_background_instance.Omega_b)
    result = camb_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)
    assert np.abs(result[0] - camb_background_instance.Omega_b0) < 1e-4

@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_hubble_parameter(camb_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(camb_background_instance, 'hubble_parameter')
    assert callable(camb_background_instance.hubble_parameter)
    result = camb_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - camb_background_instance.H0) < 1e-4

def test_comoving_distance(camb_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(camb_background_instance, 'comoving_distance')
    assert callable(camb_background_instance.comoving_distance)
    result = camb_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)

def test_transverse_comoving_distance(camb_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(camb_background_instance, 'transverse_comoving_distance')
    assert callable(camb_background_instance.transverse_comoving_distance)
    result = camb_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)

def test_angular_diameter_distance(camb_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(camb_background_instance, 'angular_diameter_distance')
    assert callable(camb_background_instance.angular_diameter_distance)
    result = camb_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(zs)

