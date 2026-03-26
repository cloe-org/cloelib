import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)
from cloelib.cosmology.cosmology import Background, Perturbations


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


def test_camb_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {
        name
        for name, value in Background.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    contents = CAMBBackground.__dict__.items()
    methods_found = {
        name for name, value in contents if callable(value) and not name.startswith("_")
    }
    assert methods_required <= methods_found


def test_camb_background_required_attributes(camb_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {
        name
        for name, value in Background.__dict__.items()
        if not callable(value) and not name.startswith("_")
    }
    contents = (
        (name, getattr(camb_background_instance, name))
        for name in dir(camb_background_instance)
    )
    attributes_found = {
        name
        for name, value in contents
        if not callable(value) and not name.startswith("_")
    }
    assert attributes_required <= attributes_found


def test_camb_background_implements_protocol(camb_background_instance):
    """Test that the CAMBBackground instance adheres to the Background protocol."""
    assert isinstance(camb_background_instance, Background)


def test_camb_background_H0(camb_background_instance):
    assert hasattr(camb_background_instance, "H0")
    assert isinstance(camb_background_instance.H0, float)
    assert camb_background_instance.H0 == 67.7


def test_camb_background_h(camb_background_instance):
    assert hasattr(camb_background_instance, "h")
    assert isinstance(camb_background_instance.h, float)
    assert camb_background_instance.h == 0.677


def test_camb_background_Omega_b0(camb_background_instance):
    assert hasattr(camb_background_instance, "Omega_b0")
    assert isinstance(camb_background_instance.Omega_b0, float)
    assert camb_background_instance.Omega_b0 == 0.022 / (0.677 * 0.677)


def test_camb_background_Omega_cdm0(camb_background_instance):
    assert hasattr(camb_background_instance, "Omega_cdm0")
    assert isinstance(camb_background_instance.Omega_cdm0, float)
    assert camb_background_instance.Omega_cdm0 == 0.12 / (0.677 * 0.677)


def test_camb_background_mnu(camb_background_instance):
    assert hasattr(camb_background_instance, "mnu")
    assert isinstance(camb_background_instance.mnu, float)
    assert camb_background_instance.mnu == 0.0


def test_camb_background_Omega_k0(camb_background_instance):
    assert hasattr(camb_background_instance, "Omega_k0")
    assert isinstance(camb_background_instance.Omega_k0, float)
    assert camb_background_instance.Omega_k0 == 0.0


def test_camb_background_As(camb_background_instance):
    assert hasattr(camb_background_instance, "As")
    assert isinstance(camb_background_instance.As, float)
    assert camb_background_instance.As == 2e-9


def test_camb_background_ns(camb_background_instance):
    assert hasattr(camb_background_instance, "ns")
    assert isinstance(camb_background_instance.ns, float)
    assert camb_background_instance.ns == 0.96


def test_camb_background_w0(camb_background_instance):
    assert hasattr(camb_background_instance, "w0")
    assert isinstance(camb_background_instance.w0, float)
    assert camb_background_instance.w0 == -1


def test_camb_background_wa(camb_background_instance):
    assert hasattr(camb_background_instance, "wa")
    assert isinstance(camb_background_instance.wa, float)
    assert camb_background_instance.wa == 0.0


def test_camb_background_N_mnu(camb_background_instance):
    assert hasattr(camb_background_instance, "N_mnu")
    assert isinstance(camb_background_instance.N_mnu, int)
    assert camb_background_instance.N_mnu == 0


def test_camb_background_N_ur(camb_background_instance):
    assert hasattr(camb_background_instance, "N_ur")
    assert isinstance(camb_background_instance.N_ur, float)
    assert camb_background_instance.N_ur == 3.044


def test_camb_background_N_eff(camb_background_instance):
    assert hasattr(camb_background_instance, "N_eff")
    assert isinstance(camb_background_instance.N_eff, float)
    assert camb_background_instance.N_eff == 3.044


def test_camb_set_neutrino_parameters_degenerate():
    bg = CAMBBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.3,
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=3,
    )
    bg._set_neutrino_parameters()
    params = bg.interface_args["CAMBparams"]
    assert params.nu_mass_eigenstates == 3
    assert list(params.nu_mass_fractions) == [1 / 3] * 3
    assert list(params.nu_mass_degeneracies) == [1.0] * 3
    assert list(params.nu_mass_numbers) == [1] * 3


def test_camb_set_neutrino_parameters_non_degenerate():
    bg = CAMBBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=np.array([0.05, 0.03]),
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=2,
    )
    bg._set_neutrino_parameters()
    params = bg.interface_args["CAMBparams"]
    assert params.nu_mass_eigenstates == 2
    assert params.Transfer.accurate_massive_neutrinos is True
    assert np.isclose(sum(list(params.nu_mass_fractions)), 1.0)
    assert list(params.nu_mass_degeneracies) == [1.0] * 2
    assert list(params.nu_mass_numbers) == [1] * 2


def test_camb_set_neutrino_parameters_zero():
    bg = CAMBBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.0,
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=0,
    )
    bg._set_neutrino_parameters()
    params = bg.interface_args["CAMBparams"]
    assert params.nu_mass_eigenstates == 0
    assert list(params.nu_mass_fractions) == []
    assert list(params.nu_mass_degeneracies) == []
    assert list(params.nu_mass_numbers) == []


def test_camb_set_neutrino_parameters_wrong_length():
    with pytest.raises(ValueError):
        bg = CAMBBackground(
            H0=67.7,
            Omega_b0=0.022 / 0.677**2,
            Omega_cdm0=0.12 / 0.677**2,
            Omega_k0=0.0,
            As=2e-9,
            ns=0.96,
            mnu=[0.02, 0.04],
            w0=-1,
            wa=0,
            gamma_MG=0,
            N_mnu=3,
        )
        bg._set_neutrino_parameters()


def test_camb_set_neutrino_parameters_wrong_type():
    with pytest.raises(TypeError):
        bg = CAMBBackground(
            H0=67.7,
            Omega_b0=0.022 / 0.677**2,
            Omega_cdm0=0.12 / 0.677**2,
            Omega_k0=0.0,
            As=2e-9,
            ns=0.96,
            mnu=None,
            w0=-1,
            wa=0,
            gamma_MG=0,
            N_mnu=1,
        )
        bg._set_neutrino_parameters()


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


def test_camb_omega_m(camb_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(camb_background_instance, "Omega_m")
    assert callable(camb_background_instance.Omega_m)
    result = camb_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_camb_omega_b(camb_background_instance, zs):
    """Test Omega_b.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(camb_background_instance, "Omega_b")
    assert callable(camb_background_instance.Omega_b)
    result = camb_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    assert np.abs(result[0] - camb_background_instance.Omega_b0) < 1e-4


@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_camb_hubble_parameter(camb_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(camb_background_instance, "hubble_parameter")
    assert callable(camb_background_instance.hubble_parameter)
    result = camb_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - camb_background_instance.H0) < 1e-4


def test_camb_comoving_distance(camb_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(camb_background_instance, "comoving_distance")
    assert callable(camb_background_instance.comoving_distance)
    result = camb_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_camb_transverse_comoving_distance(camb_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(camb_background_instance, "transverse_comoving_distance")
    assert callable(camb_background_instance.transverse_comoving_distance)
    result = camb_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_camb_angular_diameter_distance(camb_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(camb_background_instance, "angular_diameter_distance")
    assert callable(camb_background_instance.angular_diameter_distance)
    result = camb_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.fixture
def camb_perturbation_instances(camb_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of CAMBPerturbations."""
    camb_lin = CAMBLinearPerturbations(
        background=camb_background_instance, redshifts=zs
    )
    camb_non = CAMBNonLinearPerturbations(
        background=camb_background_instance, redshifts=zs, nonlinear_model="mead2016"
    )
    return {"Linear": camb_lin, "NonLinear": camb_non}


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_camb_perturbation_implements_protocol(camb_perturbation_instances, key):
    """Test that the CAMBPerturbation instances adhere to the protocol."""
    camb_instance = camb_perturbation_instances[key]
    assert isinstance(camb_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_camb_matter_power_spectrum(camb_perturbation_instances, key, zs, ks):
    """Test CAMB matter_power_spectrum."""
    camb_instance = camb_perturbation_instances[key]
    assert hasattr(camb_instance, "matter_power_spectrum")
    assert callable(camb_instance.matter_power_spectrum)
    result = camb_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_camb_growth_factor(camb_perturbation_instances, key, zs, ks):
    """Test CAMB growth_factor."""
    camb_instance = camb_perturbation_instances[key]
    assert hasattr(camb_instance, "growth_factor")
    assert callable(camb_instance.growth_factor)
    result = camb_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_camb_growth_rate(camb_perturbation_instances, key, zs, ks):
    """Test CAMB growth_rate."""
    camb_instance = camb_perturbation_instances[key]
    assert hasattr(camb_instance, "growth_rate")
    assert callable(camb_instance.growth_rate)
    result = camb_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_camb_linear_perturbations_z_zero_automatic_inclusion(camb_background_instance):
    """Test that z=0 is automatically included when not in user's redshift array."""
    # User provides redshifts WITHOUT z=0
    user_redshifts = np.array([0.5, 1.0, 1.5, 2.0])

    linear_pert = CAMBLinearPerturbations(camb_background_instance, user_redshifts)

    # Check that z=0 was automatically added to internal array
    assert 0.0 in linear_pert.z or linear_pert.z.min() < 0.001
    assert len(linear_pert.z) == len(user_redshifts) + 1  # One extra element

    # Check that sigma8_0 was computed
    assert hasattr(linear_pert, "sigma8_0")
    assert isinstance(linear_pert.sigma8_0(), (float, np.floating))
    assert linear_pert.sigma8_0() > 0  # Physical value


def test_camb_nonlinear_perturbations_z_zero_automatic_inclusion(
    camb_background_instance,
):
    """Test that z=0 is automatically included in NonLinear perturbations."""
    # User provides redshifts WITHOUT z=0
    user_redshifts = np.array([0.5, 1.0, 1.5, 2.0])

    nonlinear_pert = CAMBNonLinearPerturbations(
        camb_background_instance, user_redshifts, nonlinear_model="mead2016"
    )

    # Check that z=0 was automatically added to internal array
    assert 0.0 in nonlinear_pert.z or nonlinear_pert.z.min() < 0.001
    assert len(nonlinear_pert.z) == len(user_redshifts) + 1  # One extra element

    # Check that sigma8_0 was computed
    assert hasattr(nonlinear_pert, "sigma8_0")
    assert isinstance(nonlinear_pert.sigma8_0(), (float, np.floating))
    assert nonlinear_pert.sigma8_0() > 0  # Physical value


def test_camb_perturbations_z_zero_already_present(camb_background_instance):
    """Test that z=0 is not duplicated if already in user's redshift array."""
    # User provides redshifts WITH z=0
    user_redshifts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    linear_pert = CAMBLinearPerturbations(camb_background_instance, user_redshifts)

    # Check that z array was not modified (no duplicate z=0)
    assert len(linear_pert.z) == len(user_redshifts)  # Same length
    assert 0.0 in linear_pert.z
    assert hasattr(linear_pert, "sigma8_0")
    assert linear_pert.sigma8_0() > 0


def test_camb_sigma8_consistency_linear_vs_nonlinear(camb_background_instance):
    """Test that Linear and NonLinear give consistent sigma8(z=0) values."""
    user_redshifts = np.array([0.5, 1.0, 2.0])

    linear_pert = CAMBLinearPerturbations(camb_background_instance, user_redshifts)
    nonlinear_pert = CAMBNonLinearPerturbations(
        camb_background_instance, user_redshifts, nonlinear_model="mead2016"
    )

    # Both should compute sigma8_0
    assert hasattr(linear_pert, "sigma8_0")
    assert hasattr(nonlinear_pert, "sigma8_0")

    # Values should be very close (same linear sigma8)
    assert np.abs(linear_pert.sigma8_0() - nonlinear_pert.sigma8_0()) < 1e-3


def test_matter_power_spectrum_cb():
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
    background = CAMBBackground(**_cosmo_pars)

    # linear
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    assert_allclose(perturbations.matter_power_spectrum(0, 1), 80.534861)
    assert_allclose(perturbations.matter_power_spectrum_cb(0, 1), 81.748209, rtol=1e-03)

    # non-linear
    perturbations_nl = CAMBNonLinearPerturbations(
        background, np.linspace(0.0, 2.0, 100)
    )
    assert_allclose(
        perturbations_nl.matter_power_spectrum(0, 1), 736.010737, rtol=1.0e-03
    )
    assert_allclose(
        perturbations_nl.matter_power_spectrum_cb(0, 1), 747.017036, rtol=1.0e-03
    )
