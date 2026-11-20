import pytest
import numpy as np

from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.baccoemu_cosmology import (
    BACCOemuLinearPerturbations,
    BACCOemuNonLinearPerturbations,
)


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
        alpha_s=0.0,
    )
    return camb_instance


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


@pytest.fixture
def baccoemu_perturbation_instance(camb_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of BACCOemuPerturbations."""
    baccoemu_lin = BACCOemuLinearPerturbations(
        background=camb_background_instance, redshifts=zs
    )
    baccoemu_nonlin = BACCOemuNonLinearPerturbations(
        background=camb_background_instance,
        linearperturbations=baccoemu_lin,
        redshifts=zs,
    )
    return {"Linear": baccoemu_lin, "NonLinear": baccoemu_nonlin}


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_perturbation_implements_protocol(baccoemu_perturbation_instance, key):
    """Test that the Perturbation instances adhere to the protocol."""
    assert isinstance(baccoemu_perturbation_instance[key], Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_matter_power_spectrum(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu matter_power_spectrum."""
    assert hasattr(baccoemu_perturbation_instance[key], "matter_power_spectrum")
    assert callable(baccoemu_perturbation_instance[key].matter_power_spectrum)
    result = baccoemu_perturbation_instance[key].matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_matter_power_spectrum_cb(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu matter_power_spectrum_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "matter_power_spectrum_cb")
    assert callable(baccoemu_perturbation_instance[key].matter_power_spectrum_cb)
    result = baccoemu_perturbation_instance[key].matter_power_spectrum_cb(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_factor(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_factor."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_factor")
    assert callable(baccoemu_perturbation_instance[key].growth_factor)
    result = baccoemu_perturbation_instance[key].growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_factor_cb(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_factor_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_factor_cb")
    assert callable(baccoemu_perturbation_instance[key].growth_factor_cb)
    result = baccoemu_perturbation_instance[key].growth_factor_cb(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_rate(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_rate."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_rate")
    assert callable(baccoemu_perturbation_instance[key].growth_rate)
    result = baccoemu_perturbation_instance[key].growth_rate(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_growth_rate_cb(baccoemu_perturbation_instance, key, zs, ks):
    """Test baccoemu growth_rate_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "growth_rate_cb")
    assert callable(baccoemu_perturbation_instance[key].growth_rate_cb)
    result = baccoemu_perturbation_instance[key].growth_rate_cb(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma8_0(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma8_0."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma8_0")
    assert callable(baccoemu_perturbation_instance[key].sigma8_0)
    result = baccoemu_perturbation_instance[key].sigma8_0()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma8_0_cb(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma8_0_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma8_0_cb")
    assert callable(baccoemu_perturbation_instance[key].sigma8_0_cb)
    result = baccoemu_perturbation_instance[key].sigma8_0_cb()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma12_0(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma12_0."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma12_0")
    assert callable(baccoemu_perturbation_instance[key].sigma12_0)
    result = baccoemu_perturbation_instance[key].sigma12_0()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_baccoemu_sigma12_0_cb(baccoemu_perturbation_instance, key):
    """Test baccoemu sigma12_0_cb."""
    assert hasattr(baccoemu_perturbation_instance[key], "sigma12_0_cb")
    assert callable(baccoemu_perturbation_instance[key].sigma12_0_cb)
    result = baccoemu_perturbation_instance[key].sigma12_0_cb()
    assert isinstance(result, (float, np.floating))
    assert result > 0  # Physical value


# --- Redshift-dependent halo mass scale, log10 M_c(z) = log10 M_c - alpha_z z ---

BCM = dict(eta=-0.25, beta=-0.15, M1_z0_cen=11.0, theta_inn=-1.0)
Z_BCM = np.array([0.0, 0.5, 1.0, 2.0])  # inside the Burger2025 range (z <= 3)


def _native(background, lin, M_c, alpha_z=0.0, zs=Z_BCM):
    return BACCOemuNonLinearPerturbations(
        background=background,
        linearperturbations=lin,
        redshifts=zs,
        baryonic_boost=True,
        M_c=M_c,
        alpha_z=alpha_z,
        **BCM,
    )


def _mixin(background, M_c, alpha_z=0.0, zs=Z_BCM):
    from cloelib.cosmology.baccoemu_cosmology import (
        BACCOemuNonLinearBaryonicPerturbations,
    )

    lin = BACCOemuLinearPerturbations(background=background, redshifts=zs)
    return BACCOemuNonLinearBaryonicPerturbations(
        background=background,
        linearperturbations=lin,
        redshifts=zs,
        baryon_kwargs=dict(M_c=M_c, alpha_z=alpha_z, **BCM),
    )


def test_alpha_z_zero_is_unchanged(camb_background_instance):
    """alpha_z = 0 (the default) is the redshift-independent model, bit for bit."""
    bg = camb_background_instance
    lin = BACCOemuLinearPerturbations(background=bg, redshifts=Z_BCM)
    ks = np.logspace(-2, 1, 15)
    ref = BACCOemuNonLinearPerturbations(
        background=bg,
        linearperturbations=lin,
        redshifts=Z_BCM,
        baryonic_boost=True,
        M_c=13.5,
        **BCM,
    )
    np.testing.assert_array_equal(
        _native(bg, lin, 13.5).matter_power_spectrum(Z_BCM, ks),
        ref.matter_power_spectrum(Z_BCM, ks),
    )
    m = _mixin(bg, 13.5)
    from cloelib.cosmology.baccoemu_cosmology import (
        BACCOemuNonLinearBaryonicPerturbations,
    )

    m_ref = BACCOemuNonLinearBaryonicPerturbations(
        background=bg,
        linearperturbations=lin,
        redshifts=Z_BCM,
        baryon_kwargs=dict(M_c=13.5, **BCM),
    )
    np.testing.assert_array_equal(
        m.baryonic_suppression(Z_BCM, ks), m_ref.baryonic_suppression(Z_BCM, ks)
    )


def test_alpha_z_shifts_M_c_per_redshift(camb_background_instance):
    """With alpha_z, each redshift is the redshift-independent model at
    M_c - alpha_z z: both the native path and the mixin."""
    bg = camb_background_instance
    lin = BACCOemuLinearPerturbations(background=bg, redshifts=Z_BCM)
    ks = np.logspace(-2, 1, 15)
    M_c, alpha_z = 13.5, 0.4
    native = _native(bg, lin, M_c, alpha_z)
    mixin = _mixin(bg, M_c, alpha_z)
    for z in Z_BCM[1:]:
        shifted = M_c - alpha_z * z
        np.testing.assert_allclose(
            native.matter_power_spectrum([z], ks),
            _native(bg, lin, shifted).matter_power_spectrum([z], ks),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            mixin.baryonic_suppression([z], ks),
            _mixin(bg, shifted).baryonic_suppression([z], ks),
            rtol=1e-10,
        )
    # and it does change the prediction
    assert not np.allclose(
        mixin.baryonic_suppression([2.0], ks),
        _mixin(bg, M_c).baryonic_suppression([2.0], ks),
    )


def test_alpha_z_clips_to_emulator_box(camb_background_instance):
    """A large |alpha_z| saturates M_c(z > 0) at the edge of the trained box
    instead of raising; M_c at z = 0 is left as given."""
    from cloelib.cosmology.baccoemu_cosmology import _redshift_dependent_M_c, emu

    be = emu["Arico2023"]["Burger2025"]
    info = be.emulator["baryon"]
    lo, hi = info["bounds"][list(info["keys"]).index("M_c")]
    M_c_z = _redshift_dependent_M_c(15.5, -3.0, Z_BCM, be)
    np.testing.assert_allclose(M_c_z, [15.5, hi, hi, hi])
    np.testing.assert_allclose(
        _redshift_dependent_M_c(10.5, 3.0, Z_BCM, be), [10.5, lo, lo, lo]
    )
    assert _redshift_dependent_M_c(13.0, 0.0, Z_BCM, be) == 13.0  # scalar kept
    bg = camb_background_instance
    ks = np.logspace(-2, 1, 5)
    clipped = _mixin(bg, 15.5, -3.0).baryonic_suppression([2.0], ks)
    np.testing.assert_allclose(
        clipped, _mixin(bg, float(hi)).baryonic_suppression([2.0], ks), rtol=1e-10
    )


def test_alpha_z_requires_baryonic_boost(camb_background_instance):
    bg = camb_background_instance
    lin = BACCOemuLinearPerturbations(background=bg, redshifts=Z_BCM)
    with pytest.raises(ValueError, match="baryonic_boost"):
        BACCOemuNonLinearPerturbations(
            background=bg, linearperturbations=lin, redshifts=Z_BCM, alpha_z=0.3
        )
