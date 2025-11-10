import pytest
import numpy as np
from cloelib.cosmology.cosmopower_cosmology import (
    w0waCDM_Linear,
    wCDM_Linear,
    LCDM_Linear,
    w0waCDM_Pcb_Linear,
    wCDM_Pcb_Linear,
    LCDM_Pcb_Linear,
)
from cloelib.cosmology.camb_cosmology import CAMBBackground

# Test cosmology parameters
H0 = 67.7
h = H0 / 100.0
omch2 = 0.12
Omega_cdm0 = omch2 / h**2
ombh2 = 0.022
Omega_b0 = ombh2 / h**2
Omega_k0 = 0.0
w0 = -1.0
wa = 0.0
ns = 0.96
mnu = 0.0  # All emulators have zero neutrino mass
N_mnu = 0  # Number of massive neutrino species
As = 2e-9


@pytest.fixture
def k_array():
    """Logarithmic k array covering typical scales"""
    return np.logspace(-3, 1, 50)


@pytest.fixture
def z_array():
    """Redshift array from z=0 to z=4 (within emulator range)"""
    return np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def background_w0wa():
    """Background for w0waCDM tests"""
    return CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu,
        N_mnu=N_mnu,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_wcdm():
    """Background for wCDM tests (wa=0 implicitly)"""
    return CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu,
        N_mnu=N_mnu,
        w0=-0.9,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_lcdm():
    """Background for LCDM tests (w=-1)"""
    return CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu,
        N_mnu=N_mnu,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
    )


# ============= w0waCDM Tests =============


def test_w0wa_linear_initialization(background_w0wa, z_array):
    """Test w0waCDM emulator initializes correctly"""
    emulator = w0waCDM_Linear(background=background_w0wa, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min


def test_w0wa_linear_power_spectrum(background_w0wa, z_array, k_array):
    """Test w0waCDM power spectrum output - following the pattern: pk = emulator.matter_power_spectrum(z, k)[0,:]"""
    emulator = w0waCDM_Linear(background=background_w0wa, redshifts=z_array)

    # Use the documented calling pattern
    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0), "Power spectrum must be positive"
    assert np.all(np.isfinite(pk)), "Power spectrum must be finite"


def test_w0wa_linear_redshift_evolution(background_w0wa, z_array, k_array):
    """Test power spectrum decreases with redshift"""
    emulator = w0waCDM_Linear(background=background_w0wa, redshifts=z_array)

    pk_z0 = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    pk_z2 = emulator.matter_power_spectrum(2.0, k_array)[0, :]

    # Power should be larger at z=0 than z=2
    assert np.all(pk_z0 > pk_z2), "P(k) should decrease with redshift"


def test_w0wa_linear_growth_factor(background_w0wa, z_array, k_array):
    """Test growth factor properties"""
    emulator = w0waCDM_Linear(background=background_w0wa, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]
    D_z2 = emulator.growth_factor(2.0, k_array)[0, :]

    # Growth factor at z=0 should be 1
    np.testing.assert_allclose(D_z0, 1.0, rtol=1e-6)

    # Growth factor should be < 1 at higher redshift
    assert np.all(D_z2 < 1.0), "D(z>0) should be < 1"
    assert np.all(D_z2 > 0), "D(z) should be positive"


def test_w0wa_pcb_linear(background_w0wa, z_array, k_array):
    """Test w0waCDM Pcb emulator"""
    emulator = w0waCDM_Pcb_Linear(background=background_w0wa, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_cb, np.ndarray)
    assert np.all(pk_cb > 0)
    assert pk_cb.shape[0] == len(k_array)


# ============= wCDM Tests =============


def test_wcdm_linear_power_spectrum(background_wcdm, z_array, k_array):
    """Test wCDM power spectrum"""
    emulator = wCDM_Linear(background=background_wcdm, redshifts=z_array)

    pk = emulator.matter_power_spectrum(1.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


def test_wcdm_pcb_linear(background_wcdm, z_array, k_array):
    """Test wCDM Pcb emulator"""
    emulator = wCDM_Pcb_Linear(background=background_wcdm, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= LCDM Tests =============


def test_lcdm_linear_power_spectrum(background_lcdm, z_array, k_array):
    """Test LCDM power spectrum"""
    emulator = LCDM_Linear(background=background_lcdm, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert pk.shape[0] == len(k_array)


def test_lcdm_growth_factor_normalization(background_lcdm, z_array, k_array):
    """Test LCDM growth factor is properly normalized"""
    emulator = LCDM_Linear(background=background_lcdm, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        D_z0, 1.0, rtol=1e-6, err_msg="Growth factor should be 1 at z=0"
    )


def test_lcdm_pcb_linear(background_lcdm, z_array, k_array):
    """Test LCDM Pcb emulator"""
    emulator = LCDM_Pcb_Linear(background=background_lcdm, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(1.5, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= Boundary & Error Tests =============


def test_parameter_out_of_bounds():
    """Test that out-of-bounds parameters raise ValueError"""
    # Create background with H0 outside bounds (should be 20-100)
    bad_background = CAMBBackground(
        H0=150,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu,
        N_mnu=N_mnu,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )

    with pytest.raises(ValueError, match="out of range"):
        w0waCDM_Linear(background=bad_background, redshifts=np.array([0.0, 1.0]))


def test_non_flat_geometry():
    """Test that non-flat geometries raise assertion error"""
    curved_background = CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.1,  # Non-zero curvature
        As=As,
        ns=ns,
        mnu=mnu,
        N_mnu=N_mnu,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )

    with pytest.raises(AssertionError, match="Non flat geometries"):
        w0waCDM_Linear(background=curved_background, redshifts=np.array([0.0]))


def test_redshift_filtering(background_lcdm):
    """Test that emulator handles valid redshifts correctly"""
    # Only use redshifts within valid range (<=5)
    z_valid = np.array([0.0, 1.0, 3.0, 4.5])
    emulator = LCDM_Linear(background=background_lcdm, redshifts=z_valid)

    # Internal z should be within range
    assert np.all(emulator.z <= 5.0)


# ============= Comparison Tests =============


def test_pcb_vs_total_matter(background_lcdm, z_array, k_array):
    """Test that Pcb and total matter spectra are similar (no massive neutrinos)"""
    emulator_total = LCDM_Linear(background=background_lcdm, redshifts=z_array)
    emulator_pcb = LCDM_Pcb_Linear(background=background_lcdm, redshifts=z_array)

    pk_total = emulator_total.matter_power_spectrum(0.0, k_array)[0, :]
    pk_cb = emulator_pcb.matter_power_spectrum(0.0, k_array)[0, :]

    # With mnu=0, they should be very close
    np.testing.assert_allclose(
        pk_total, pk_cb, rtol=0.01, err_msg="Pcb and total should match when mnu=0"
    )


def test_str_representation(background_w0wa, z_array):
    """Test __str__ method returns useful info"""
    emulator = w0waCDM_Linear(background=background_w0wa, redshifts=z_array)

    info_str = str(emulator)

    assert "w0waCDM" in info_str
    assert "k_min" in info_str
    assert "k_max" in info_str
    assert "neutrino" in info_str.lower()
