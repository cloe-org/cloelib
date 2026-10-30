import pytest
import numpy as np

from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology.class_cosmology import (
    CLASSBackground,
    CLASSLinearPerturbations,
    CLASSNonLinearPerturbations,
)
from cloelib.cosmology.Weyl_cosmology import WeylPerturbations


@pytest.fixture(scope="module")
def zs():
    return np.linspace(0, 2, 20)


@pytest.fixture(scope="module")
def ks():
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.fixture(scope="module")
def z_ini():
    return 4.0


@pytest.fixture(scope="module")
def class_background_instance():
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


@pytest.fixture(scope="module")
def class_perturbation_instances(class_background_instance, zs, z_ini):
    """Fixture to create the Linear and NonLinear instances of
    CLASSPerturbations which will stand in for the geometric regime in this
    test with a test value of Omega_m = 0.3."""

    if np.max(zs) < z_ini:
        zs_source = np.append(zs, z_ini)
    else:
        zs_source = zs

    class_lin = CLASSLinearPerturbations(
        background=class_background_instance, redshifts=zs_source
    )
    class_nl = CLASSNonLinearPerturbations(
        background=class_background_instance,
        linearperturbations=class_lin,
        redshifts=zs_source,
        nonlinear_model="halofit",
    )
    return {"Linear": class_lin, "NonLinear": class_nl}


@pytest.fixture(scope="module")
def Weyl_perturbation_instance(
    class_perturbation_instances,
    zs,
    z_ini,
):
    """Fixture to create the instance of WeylPerturbations."""
    class_lin = class_perturbation_instances["Linear"]
    class_nl = class_perturbation_instances["NonLinear"]

    Weyl_pert = WeylPerturbations(
        nonlinearperturbations=class_nl,
        linearperturbations=class_lin,
        redshifts=zs,
        z_ini=z_ini,
    )

    return Weyl_pert


def test_Weyl_perturbation_implements_protocol(Weyl_perturbation_instance):
    """Test that the WeylPerturbation instance adhere to the protocol."""
    assert isinstance(Weyl_perturbation_instance, Perturbations)


def test_Weyl_matter_power_spectrum(Weyl_perturbation_instance, zs, ks):
    """Test Weyl matter power spectrum."""
    assert hasattr(Weyl_perturbation_instance, "matter_power_spectrum")
    assert callable(Weyl_perturbation_instance.matter_power_spectrum)
    result = Weyl_perturbation_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


def test_Weyl_growth_factor(Weyl_perturbation_instance, zs, ks):
    """Test Weyl growth_factor."""
    assert hasattr(Weyl_perturbation_instance, "growth_factor")
    assert callable(Weyl_perturbation_instance.growth_factor)
    result = Weyl_perturbation_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


def test_Weyl_growth_rate(Weyl_perturbation_instance, zs):
    """Test Weyl growth_rate."""
    assert hasattr(Weyl_perturbation_instance, "growth_rate")
    assert callable(Weyl_perturbation_instance.growth_rate)
    result = Weyl_perturbation_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape == (len(zs),)


def test_Weyl_power_spectrum_boost(
    Weyl_perturbation_instance,
    class_perturbation_instances,
    zs,
    ks,
    z_ini,
):
    """Test that WeylPerturbations applies the correct power-spectrum boost."""

    class_lin = class_perturbation_instances["Linear"]
    class_nl = class_perturbation_instances["NonLinear"]

    # Calculate the power spectra directly from CLASS
    P_lin = class_lin.matter_power_spectrum(zs, ks)
    P_nl = class_nl.matter_power_spectrum(zs, ks)

    # Calculate the linear power spectrum at z_ini
    zs_ini = np.full_like(zs, z_ini)
    P_lin_ini = class_lin.matter_power_spectrum(zs_ini, ks)

    # Expected Weyl power spectrum
    P_expected = P_lin_ini * (P_nl / P_lin)

    # Power spectrum returned by the Weyl wrapper
    P_weyl = Weyl_perturbation_instance.matter_power_spectrum(zs, ks)

    # Compare the two results
    np.testing.assert_allclose(P_weyl, P_expected, rtol=1e-6)


def test_Weyl_matter_power_spectrum_cb(
    Weyl_perturbation_instance,
    class_perturbation_instances,
    zs,
    ks,
    z_ini,
):
    """Test the Weyl CDM+baryon power-spectrum boost."""

    class_lin = class_perturbation_instances["Linear"]
    class_nl = class_perturbation_instances["NonLinear"]

    # Calculate CDM+baryon power spectra directly from CLASS
    P_lin_cb = class_lin.matter_power_spectrum_cb(zs, ks)
    P_nl_cb = class_nl.matter_power_spectrum_cb(zs, ks)

    # Calculate the initial CDM+baryon power spectrum
    zs_ini = np.full_like(zs, z_ini)
    P_lin_ini_cb = class_lin.matter_power_spectrum_cb(zs_ini, ks)

    # Expected Weyl CDM+baryon power spectrum
    P_expected = P_lin_ini_cb * (P_nl_cb / P_lin_cb)

    # Actual Weyl CDM+baryon power spectrum
    P_weyl = Weyl_perturbation_instance.matter_power_spectrum_cb(zs, ks)

    np.testing.assert_allclose(P_weyl, P_expected, rtol=1e-6)


def test_Weyl_growth_rate_redshift_selection(
    Weyl_perturbation_instance,
    class_perturbation_instances,
):
    """Test that growth_rate selects the correct redshifts."""

    class_nl = class_perturbation_instances["NonLinear"]

    # Growth rates and redshifts from the nonlinear CLASS object
    growth_rates_source = class_nl.growth_rate()
    zs_source = np.asarray(class_nl.z)

    # Redshifts requested by the Weyl wrapper
    zs_weyl = np.asarray(Weyl_perturbation_instance.z)

    # Find the indices corresponding to the Weyl redshifts
    indices = [
        np.where(np.isclose(zs_source, z, rtol=0, atol=1e-12))[0][0] for z in zs_weyl
    ]

    # Expected growth rates
    expected = growth_rates_source[indices]

    # Actual growth rates from the Weyl wrapper
    result = Weyl_perturbation_instance.growth_rate()

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_Weyl_growth_rate_missing_redshift(
    class_perturbation_instances,
    z_ini,
):
    """Test that unavailable redshifts raise an error."""

    class_lin = class_perturbation_instances["Linear"]
    class_nl = class_perturbation_instances["NonLinear"]

    zs_weyl = np.array([0.123456789])

    weyl = WeylPerturbations(
        nonlinearperturbations=class_nl,
        linearperturbations=class_lin,
        redshifts=zs_weyl,
        z_ini=z_ini,
    )

    with pytest.raises(ValueError, match="Some redshifts"):
        weyl.growth_rate()


def test_Weyl_sigma8_0(
    Weyl_perturbation_instance,
    class_perturbation_instances,
):
    """Test that sigma8_0 returns the value from nonlinear CLASS perturbations."""

    class_nl = class_perturbation_instances["NonLinear"]

    expected = class_nl.sigma8_0()
    result = Weyl_perturbation_instance.sigma8_0()

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_Weyl_sigma8_zini(
    Weyl_perturbation_instance,
    class_perturbation_instances,
    z_ini,
):
    """Test the calculation of sigma8 at the initial redshift."""

    class_lin = class_perturbation_instances["Linear"]
    class_nl = class_perturbation_instances["NonLinear"]

    # sigma8 at z = 0 from nonlinear CLASS perturbations
    sigma8_0 = class_nl.sigma8_0()

    # Linear growth factor at z_ini
    D_ini = class_lin.growth_factor(
        np.array([z_ini]),
        np.array([0.1]),
    )

    # Expected sigma8 at z_ini
    expected = sigma8_0 * D_ini[0, 0]

    # Result from the Weyl wrapper
    result = Weyl_perturbation_instance.sigma8_zini()

    np.testing.assert_allclose(result, expected, rtol=1e-6)
