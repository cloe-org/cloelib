import numpy as np
from numpy.testing import assert_allclose

from cloelib.cosmology.cosmology import Background
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.class_cosmology import CLASSBackground
from cloelib.cosmology.jax_cosmology import JAXBackground
from cloelib.cosmology import derived_cosmology


def test_background_runtime():
    assert hasattr(Background, "_is_runtime_protocol")


def test_background_required_methods():
    contents = Background.__dict__.items()
    methods_found = {
        name for name, value in contents if callable(value) and not name.startswith("_")
    }
    methods_required = {
        "comoving_distance",
        "hubble_parameter",
        "angular_diameter_distance",
        "Omega_b",
        "Omega_m",
        "Omega_m_cb",
        "transverse_comoving_distance",
    }
    assert methods_required == methods_found


def test_background_required_attributes():
    contents = Background.__dict__.items()
    attributes_found = {
        name
        for name, value in contents
        if not callable(value) and not name.startswith("_")
    }
    attributes_required = {
        "wa",
        "As",
        "w0",
        "Omega_k0",
        "h",
        "Omega_b0",
        "gamma_MG",
        "mnu",
        "Omega_cdm0",
        "H0",
        "ns",
        "N_ur",
        "N_mnu",
        "N_eff",
        "interface_args",
        "rdrag",
    }
    assert attributes_required == attributes_found


def test_derived_cosmology():
    # Cosmology parameters
    print("# Cosmology parameters")
    _cosmo_pars = dict(
        H0=67.7,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_b0=0.022 / 0.677**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.1,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    _z_test = np.zeros(1)

    for _Background in (CAMBBackground, CLASSBackground, JAXBackground):
        background = _Background(**_cosmo_pars)
        assert_allclose(
            derived_cosmology.rho_crit(background, _z_test), 1.27203085e11, rtol=2e-05
        )
        assert_allclose(
            derived_cosmology.dV_dzdO(background, _z_test + 1), 2.853696e10, rtol=3e-04
        )

        # to be fixed in another PR
        if _Background != JAXBackground:
            assert_allclose(background.rdrag, 147.50225, rtol=1e-1)


def test_cb():
    # Cosmology parameters
    print("# Cosmology parameters")
    _cosmo_pars = dict(
        H0=67.7,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_b0=0.022 / 0.677**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.1,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    _z_test = np.zeros(1)
    # not implemented for CLASSBackground yet
    for _Background in (CAMBBackground, JAXBackground):
        background = _Background(**_cosmo_pars)
        assert (background.Omega_m_cb(_z_test) < background.Omega_m(_z_test)).all()
        assert_allclose(
            background.Omega_m_cb(_z_test)[0],
            _cosmo_pars["Omega_cdm0"] + _cosmo_pars["Omega_b0"],
            rtol=1e-03,
        )
