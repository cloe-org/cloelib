import numpy as np
import pytest
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.cosmology.cosmology import Background
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.class_cosmology import CLASSBackground
from cloelib.cosmology.jax_cosmology import JAXBackground
from cloelib.cosmology import derived_cosmology


def test_background_runtime():
    assert hasattr(Background, "_is_runtime_protocol")


def test_background_required_methods():
    contents = Background.__dict__.items()
    methods_found = {name for name, value in contents if callable(value)
                     and not name.startswith('_')}
    methods_required = {'comoving_distance', 'hubble_parameter', 'angular_diameter_distance',
                        'Omega_b', 'Omega_m', 'transverse_comoving_distance'}
    assert methods_required == methods_found


def test_background_required_attributes():
    contents = Background.__dict__.items()
    attributes_found = {name for name, value in contents if not callable(value)
                        and not name.startswith('_')}
    attributes_required = {'wa', 'As', 'w0', 'Omega_k0', 'h', 'Omega_b0', 'gamma_MG',
                           'mnu', 'Omega_cdm0', 'H0', 'ns', 'interface_args', 'rdrag'}
    assert attributes_required == attributes_found


def test_cosmo():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.1,
        As=2e-9,
        gamma_MG=0.0,
    )

    _z_test = np.zeros(1)

    for _Background in (CAMBBackground, CLASSBackground, JAXBackground):

        background = _Background(**_cosmo_pars)
        assert_allclose(
            derived_cosmology.rho_crit(background, _z_test), 1.27203085e11, rtol=2e-05
        )
        assert_allclose(
            derived_cosmology.dV_dzdO(background, _z_test+1), 2.853696e10, rtol=3e-04
        )

        # to be fixed in another PR
        if _Background != JAXBackground:
            assert_allclose(background.rdrag, 147.50225, rtol=1e-1)
