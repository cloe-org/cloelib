import pytest
from cloelib.cosmology.cosmology import Background


def test_background_runtime():
    assert hasattr(Background, '_is_runtime_protocol')

def test_background_required_methods():
    contents = Background.__dict__.items()
    methods_found = {name for name, value in contents if callable(value) and not name.startswith('_')}
    methods_required = {'comoving_distance', 'hubble_parameter', 'angular_diameter_distance', 'Omega_b', 'Omega_m', 'transverse_comoving_distance'}
    assert methods_required == methods_found

def test_background_required_attributes():
    contents = Background.__dict__.items()
    attributes_found = {name for name, value in contents if not callable(value) and not name.startswith('_')}
    attributes_required = {'wa', 'As', 'w0', 'Omega_k0', 'h', 'Omega_b0', 'gamma_MG', 'mnu', 'Omega_cdm0', 'H0', 'ns', 'N_ur', 'N_mnu', 'N_eff'}
    assert attributes_required == attributes_found

