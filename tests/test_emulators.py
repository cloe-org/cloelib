import pytest
import numpy as np
import importlib.util

HAS_COSMOPOWER = importlib.util.find_spec("cosmopower") is not None
if HAS_COSMOPOWER:
    from cloelib.cosmology.cosmopower_cosmology import (
        CosmoPowerw0waCDMPerturbations as w0waCDM,
        CosmoPowerwCDMPerturbations as wCDM,
        CosmoPowerLCDMPerturbations as LCDM,
    )
else:
    pytest.skip(
        "cosmopower not installed; skipping emulator tests", allow_module_level=True
    )

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
mnu_1mass = 0.06  # For 1 massive neutrino tests
mnu_3degen = 0.06  # For 3 degenerate neutrinos tests
N_mnu = 0  # Number of massive neutrino species
As = 2e-9


class DummyBackground:
    def __init__(
        self, H0, Omega_b0, Omega_cdm0, Omega_k0, As, ns, mnu, N_mnu, w0, wa, gamma_MG
    ):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.mnu = mnu
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG
        self.N_mnu = N_mnu
        self._interface_args = {}

    def Omega_b(self, zs):
        return np.ones_like(zs)

    def Omega_m(self, zs):
        return np.ones_like(zs)

    def hubble_parameter(self, zs, units="km/s/Mpc"):
        return np.ones_like(zs)

    def comoving_distance(self, zs):
        return np.ones_like(zs)

    def transverse_comoving_distance(self, zs):
        return np.ones_like(zs)

    def angular_diameter_distance(self, zs):
        return np.ones_like(zs)


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
    return DummyBackground(
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
def background_w0wa_1mass():
    """Background for w0waCDM tests with 1 massive neutrino"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu_1mass,
        N_mnu=1,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_w0wa_3degen():
    """Background for w0waCDM tests with 3 degenerate neutrinos"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu_3degen,
        N_mnu=3,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_wcdm():
    """Background for wCDM tests (wa=0 implicitly)"""
    return DummyBackground(
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
def background_wcdm_1mass():
    """Background for wCDM tests with 1 massive neutrino"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu_1mass,
        N_mnu=1,
        w0=-0.9,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_wcdm_3degen():
    """Background for wCDM tests with 3 degenerate neutrinos"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu_3degen,
        N_mnu=3,
        w0=-0.9,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_lcdm():
    """Background for LCDM tests (w=-1)"""
    return DummyBackground(
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


@pytest.fixture
def background_lcdm_1mass():
    """Background for LCDM tests with 1 massive neutrino"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu_1mass,
        N_mnu=1,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_lcdm_3degen():
    """Background for LCDM tests with 3 degenerate neutrinos"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=mnu_3degen,
        N_mnu=3,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
    )


# ============= w0waCDM Tests (0 massive neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_initialization(background_w0wa, z_array):
    """Test w0waCDM emulator initializes correctly"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_power_spectrum(background_w0wa, z_array, k_array):
    """Test w0waCDM power spectrum output"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0), "Power spectrum must be positive"
    assert np.all(np.isfinite(pk)), "Power spectrum must be finite"


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_redshift_evolution(background_w0wa, z_array, k_array):
    """Test power spectrum decreases with redshift"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    pk_z0 = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    pk_z2 = emulator.matter_power_spectrum(2.0, k_array)[0, :]

    assert np.all(pk_z0 > pk_z2), "P(k) should decrease with redshift"


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_growth_factor(background_w0wa, z_array, k_array):
    """Test growth factor properties"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]
    D_z2 = emulator.growth_factor(2.0, k_array)[0, :]

    np.testing.assert_allclose(D_z0, 1.0, rtol=1e-6)
    assert np.all(D_z2 < 1.0), "D(z>0) should be < 1"
    assert np.all(D_z2 > 0), "D(z) should be positive"


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_pcb_linear(background_w0wa, z_array, k_array):
    """Test w0waCDM Pcb emulator"""
    emulator = w0waCDM.LinearCB(background=background_w0wa, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_cb, np.ndarray)
    assert np.all(pk_cb > 0)
    assert pk_cb.shape[0] == len(k_array)


# ============= w0waCDM Tests (1 massive neutrino) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_1mass_initialization(background_w0wa_1mass, z_array):
    """Test w0waCDM 1mass emulator initializes correctly"""
    emulator = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min
    assert emulator.has_neutrinos is True


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_1mass_power_spectrum(background_w0wa_1mass, z_array, k_array):
    """Test w0waCDM 1mass power spectrum output"""
    emulator = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0), "Power spectrum must be positive"
    assert np.all(np.isfinite(pk)), "Power spectrum must be finite"


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linearcb_1mass_power_spectrum(background_w0wa_1mass, z_array, k_array):
    """Test w0waCDM 1mass Pcb power spectrum"""
    emulator = w0waCDM.LinearCB(background=background_w0wa_1mass, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_cb, np.ndarray)
    assert np.all(pk_cb > 0)
    assert pk_cb.shape[0] == len(k_array)


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_1mass_suppression(
    background_w0wa, background_w0wa_1mass, z_array, k_array
):
    """Test that massive neutrinos suppress power spectrum"""
    emulator_0mass = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)
    emulator_1mass = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    pk_0mass = emulator_0mass.matter_power_spectrum(0.0, k_array)[0, :]
    pk_1mass = emulator_1mass.matter_power_spectrum(0.0, k_array)[0, :]

    # Massive neutrinos should suppress power on small scales
    # Test on scales where suppression is expected (k > 0.1)
    high_k_mask = k_array > 0.1
    assert np.all(pk_0mass[high_k_mask] > pk_1mass[high_k_mask]), (
        "Massive neutrinos should suppress power on small scales"
    )


# ============= w0waCDM Tests (3 degenerate neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_3degen_initialization(background_w0wa_3degen, z_array):
    """Test w0waCDM 3degen emulator initializes correctly"""
    emulator = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert emulator.has_neutrinos is True
    assert emulator.background.N_mnu == 3


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linear_3degen_power_spectrum(background_w0wa_3degen, z_array, k_array):
    """Test w0waCDM 3degen power spectrum output"""
    emulator = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_linearcb_3degen_power_spectrum(background_w0wa_3degen, z_array, k_array):
    """Test w0waCDM 3degen Pcb power spectrum"""
    emulator = w0waCDM.LinearCB(background=background_w0wa_3degen, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= wCDM Tests (0 massive neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_linear_power_spectrum(background_wcdm, z_array, k_array):
    """Test wCDM power spectrum"""
    emulator = wCDM.Linear(background=background_wcdm, redshifts=z_array)

    pk = emulator.matter_power_spectrum(1.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_pcb_linear(background_wcdm, z_array, k_array):
    """Test wCDM Pcb emulator"""
    emulator = wCDM.LinearCB(background=background_wcdm, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= wCDM Tests (1 massive neutrino) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_linear_1mass_initialization(background_wcdm_1mass, z_array):
    """Test wCDM 1mass emulator initializes correctly"""
    emulator = wCDM.Linear(background=background_wcdm_1mass, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_linear_1mass_power_spectrum(background_wcdm_1mass, z_array, k_array):
    """Test wCDM 1mass power spectrum output"""
    emulator = wCDM.Linear(background=background_wcdm_1mass, redshifts=z_array)

    pk = emulator.matter_power_spectrum(1.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_linearcb_1mass_power_spectrum(background_wcdm_1mass, z_array, k_array):
    """Test wCDM 1mass Pcb power spectrum"""
    emulator = wCDM.LinearCB(background=background_wcdm_1mass, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= wCDM Tests (3 degenerate neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_linear_3degen_initialization(background_wcdm_3degen, z_array):
    """Test wCDM 3degen emulator initializes correctly"""
    emulator = wCDM.Linear(background=background_wcdm_3degen, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert emulator.background.N_mnu == 3


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_wcdm_linear_3degen_power_spectrum(background_wcdm_3degen, z_array, k_array):
    """Test wCDM 3degen power spectrum output"""
    emulator = wCDM.Linear(background=background_wcdm_3degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk > 0)


# ============= LCDM Tests (0 massive neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linear_power_spectrum(background_lcdm, z_array, k_array):
    """Test LCDM power spectrum"""
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert pk.shape[0] == len(k_array)


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_growth_factor_normalization(background_lcdm, z_array, k_array):
    """Test LCDM growth factor is properly normalized"""
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        D_z0, 1.0, rtol=1e-6, err_msg="Growth factor should be 1 at z=0"
    )


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_pcb_linear(background_lcdm, z_array, k_array):
    """Test LCDM Pcb emulator"""
    emulator = LCDM.LinearCB(background=background_lcdm, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(1.5, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= LCDM Tests (1 massive neutrino) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linear_1mass_initialization(background_lcdm_1mass, z_array):
    """Test LCDM 1mass emulator initializes correctly"""
    emulator = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linear_1mass_power_spectrum(background_lcdm_1mass, z_array, k_array):
    """Test LCDM 1mass power spectrum output"""
    emulator = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert pk.shape[0] == len(k_array)


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linearcb_1mass_power_spectrum(background_lcdm_1mass, z_array, k_array):
    """Test LCDM 1mass Pcb power spectrum"""
    emulator = LCDM.LinearCB(background=background_lcdm_1mass, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(1.5, k_array)[0, :]

    assert np.all(pk_cb > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_1mass_growth_factor(background_lcdm_1mass, z_array, k_array):
    """Test LCDM 1mass growth factor"""
    emulator = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        D_z0, 1.0, rtol=1e-6, err_msg="Growth factor should be 1 at z=0"
    )


# ============= LCDM Tests (3 degenerate neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linear_3degen_initialization(background_lcdm_3degen, z_array):
    """Test LCDM 3degen emulator initializes correctly"""
    emulator = LCDM.Linear(background=background_lcdm_3degen, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert emulator.background.N_mnu == 3


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linear_3degen_power_spectrum(background_lcdm_3degen, z_array, k_array):
    """Test LCDM 3degen power spectrum output"""
    emulator = LCDM.Linear(background=background_lcdm_3degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_lcdm_linearcb_3degen_power_spectrum(background_lcdm_3degen, z_array, k_array):
    """Test LCDM 3degen Pcb power spectrum"""
    emulator = LCDM.LinearCB(background=background_lcdm_3degen, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= Boundary & Error Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_parameter_out_of_bounds():
    """Test that out-of-bounds parameters raise ValueError"""
    bad_background = DummyBackground(
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
        w0waCDM.Linear(background=bad_background, redshifts=np.array([0.0, 1.0]))


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_non_flat_geometry():
    """Test that non-flat geometries raise assertion error"""
    curved_background = DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.1,
        As=As,
        ns=ns,
        mnu=mnu,
        N_mnu=N_mnu,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )

    with pytest.raises(AssertionError, match="Non flat geometries"):
        w0waCDM.Linear(background=curved_background, redshifts=np.array([0.0]))


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_unsupported_neutrino_configuration():
    """Test that unsupported N_mnu raises ValueError"""
    bad_nu_background = DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=0.06,
        N_mnu=2,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )

    with pytest.raises(ValueError, match="Unsupported N_mnu"):
        w0waCDM.Linear(background=bad_nu_background, redshifts=np.array([0.0, 1.0]))


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_redshift_filtering(background_lcdm):
    """Test that emulator handles valid redshifts correctly"""
    z_valid = np.array([0.0, 1.0, 3.0, 4.5])
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_valid)

    assert np.all(emulator.z <= 5.0)


# ============= Comparison Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_pcb_vs_total_matter(background_lcdm, z_array, k_array):
    """Test that Pcb and total matter spectra are similar (no massive neutrinos)"""
    emulator_total = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    emulator_pcb = LCDM.LinearCB(background=background_lcdm, redshifts=z_array)

    pk_total = emulator_total.matter_power_spectrum(0.0, k_array)[0, :]
    pk_cb = emulator_pcb.matter_power_spectrum(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        pk_total, pk_cb, rtol=0.01, err_msg="Pcb and total should match when mnu=0"
    )


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_str_representation_no_neutrinos(background_w0wa, z_array):
    """Test __str__ method for massless neutrinos"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    info_str = str(emulator)

    assert "w0waCDM" in info_str
    assert "k_min" in info_str
    assert "k_max" in info_str
    assert "no massive neutrinos" in info_str


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_str_representation_1mass(background_w0wa_1mass, z_array):
    """Test __str__ method for 1 massive neutrino"""
    emulator = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    info_str = str(emulator)

    assert "w0waCDM" in info_str
    assert "Casas et al. 2023" in info_str
    assert "one massive neutrino" in info_str


@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_str_representation_3degen(background_w0wa_3degen, z_array):
    """Test __str__ method for 3 degenerate neutrinos"""
    emulator = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)

    info_str = str(emulator)

    assert "w0waCDM" in info_str
    assert "Archidiacono et al. (2024)" in info_str
    assert "three" in info_str
    assert "degenerate" in info_str
