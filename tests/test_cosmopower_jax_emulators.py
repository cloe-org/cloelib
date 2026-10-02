import pytest
import numpy as np
import importlib.util

HAS_COSMOPOWER_JAX = importlib.util.find_spec("cosmopower_jax") is not None
if HAS_COSMOPOWER_JAX:
    from cloelib.cosmology.cosmopower_jax_cosmology import (
        CosmoPowerJAXw0waCDMPerturbations as w0waCDM,
        CosmoPowerJAXwCDMPerturbations as wCDM,
        CosmoPowerJAXLCDMPerturbations as LCDM,
        CosmoPowerJAXLCDMCurvaturePerturbations as Curvature,
        CosmoPowerJAXw0waCurvaturePerturbations as w0waCurvature,
        CosmoPowerJAXLCDMRunningIndexPerturbations as RunningIndex,
        CosmoPowerJAXw0waRunningIndexPerturbations as w0waRunningIndex,
    )
else:
    pytest.skip(
        "cosmopower_jax not installed; skipping JAX emulator tests",
        allow_module_level=True,
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
log10TAGN = 7.8  # AGN feedback parameter


class DummyBackground:
    def __init__(
        self,
        H0,
        Omega_b0,
        Omega_cdm0,
        Omega_k0,
        As,
        ns,
        mnu,
        N_mnu,
        w0,
        wa,
        gamma_MG,
        alpha_s=0.0,
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
        self.alpha_s = alpha_s
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


@pytest.fixture
def background_w0wa_2degen():
    """Background for w0waCDM tests with 2 degenerate neutrinos"""
    return DummyBackground(
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


@pytest.fixture
def background_wcdm_2degen():
    """Background for wCDM tests with 2 degenerate neutrinos"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=0.06,
        N_mnu=2,
        w0=-0.9,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_lcdm_2degen():
    """Background for LCDM tests with 2 degenerate neutrinos"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=0.06,
        N_mnu=2,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_curvature():
    """Background for curvature tests (LCDM + non-zero Omega_k0)"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.02,
        As=As,
        ns=ns,
        mnu=0.06,
        N_mnu=1,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
    )


@pytest.fixture
def background_running():
    """Background for running spectral index tests (LCDM + alpha_s)"""
    return DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.0,
        As=As,
        ns=ns,
        mnu=0.06,
        N_mnu=1,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        alpha_s=0.01,
    )


def _extended_bg(N_mnu, mnu, Omega_k0, alpha_s, w0=-1.0, wa=0.0):
    """Helper: DummyBackground for the extended (curvature / running) families."""
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
        alpha_s=alpha_s,
    )


# --- LCDM curvature neutrino variants (N_mnu = 0, 3) ---
@pytest.fixture
def background_curvature_0mass():
    return _extended_bg(N_mnu=0, mnu=0.0, Omega_k0=0.02, alpha_s=0.0)


@pytest.fixture
def background_curvature_3degen():
    return _extended_bg(N_mnu=3, mnu=0.06, Omega_k0=0.02, alpha_s=0.0)


# --- LCDM running neutrino variants (N_mnu = 0, 3) ---
@pytest.fixture
def background_running_0mass():
    return _extended_bg(N_mnu=0, mnu=0.0, Omega_k0=0.0, alpha_s=0.01)


@pytest.fixture
def background_running_3degen():
    return _extended_bg(N_mnu=3, mnu=0.06, Omega_k0=0.0, alpha_s=0.01)


# --- w0waCDM curvature (dynamical dark energy + curvature) ---
@pytest.fixture
def background_w0wa_curvature():
    return _extended_bg(N_mnu=1, mnu=0.06, Omega_k0=0.02, alpha_s=0.0, w0=-0.9, wa=0.1)


@pytest.fixture
def background_w0wa_curvature_0mass():
    return _extended_bg(N_mnu=0, mnu=0.0, Omega_k0=0.02, alpha_s=0.0, w0=-0.9, wa=0.1)


# --- w0waCDM running (dynamical dark energy + running index) ---
@pytest.fixture
def background_w0wa_running():
    return _extended_bg(N_mnu=1, mnu=0.06, Omega_k0=0.0, alpha_s=0.01, w0=-0.9, wa=0.1)


@pytest.fixture
def background_w0wa_running_3degen():
    return _extended_bg(N_mnu=3, mnu=0.06, Omega_k0=0.0, alpha_s=0.01, w0=-0.9, wa=0.1)


# ============= w0waCDM Linear Tests (0 massive neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_initialization(background_w0wa, z_array):
    """Test w0waCDM JAX emulator initializes correctly"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min
    # Linear now has sigma8 and fsigma8
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_power_spectrum(background_w0wa, z_array, k_array):
    """Test w0waCDM JAX power spectrum output"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0), "Power spectrum must be positive"
    assert np.all(np.isfinite(pk)), "Power spectrum must be finite"


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_redshift_evolution(background_w0wa, z_array, k_array):
    """Test power spectrum decreases with redshift"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    pk_z0 = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    pk_z2 = emulator.matter_power_spectrum(2.0, k_array)[0, :]

    assert np.all(pk_z0 > pk_z2), "P(k) should decrease with redshift"


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_growth_factor(background_w0wa, z_array, k_array):
    """Test growth factor properties"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]
    D_z2 = emulator.growth_factor(2.0, k_array)[0, :]

    np.testing.assert_allclose(D_z0, 1.0, rtol=1e-6)
    assert np.all(D_z2 < 1.0), "D(z>0) should be < 1"
    assert np.all(D_z2 > 0), "D(z) should be positive"


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_sigma8(background_w0wa, z_array):
    """Test Linear class sigma8 and growth_rate"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    sigma8_0 = emulator.sigma8_0()
    growth_rate = emulator.growth_rate()

    assert isinstance(sigma8_0, (float, np.floating))
    assert sigma8_0 > 0
    assert isinstance(growth_rate, np.ndarray)
    assert np.all(growth_rate > 0)
    assert len(growth_rate) == len(z_array)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_pcb_linear(background_w0wa, z_array, k_array):
    """Test w0waCDM Pcb JAX emulator"""
    emulator = w0waCDM.LinearCB(background=background_w0wa, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_cb, np.ndarray)
    assert np.all(pk_cb > 0)
    assert pk_cb.shape[0] == len(k_array)
    # LinearCB now has sigma8 and fsigma8
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


# ============= w0waCDM Linear Tests (1 massive neutrino) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_1mass_initialization(background_w0wa_1mass, z_array):
    """Test w0waCDM 1mass JAX emulator initializes correctly"""
    emulator = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min
    assert emulator.has_neutrinos is True
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_1mass_power_spectrum(background_w0wa_1mass, z_array, k_array):
    """Test w0waCDM 1mass JAX power spectrum output"""
    emulator = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0), "Power spectrum must be positive"
    assert np.all(np.isfinite(pk)), "Power spectrum must be finite"


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linearcb_1mass_power_spectrum(background_w0wa_1mass, z_array, k_array):
    """Test w0waCDM 1mass Pcb JAX power spectrum"""
    emulator = w0waCDM.LinearCB(background=background_w0wa_1mass, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_cb, np.ndarray)
    assert np.all(pk_cb > 0)
    assert pk_cb.shape[0] == len(k_array)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
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


# ============= w0waCDM Linear Tests (3 degenerate neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_3degen_initialization(background_w0wa_3degen, z_array):
    """Test w0waCDM 3degen JAX emulator initializes correctly"""
    emulator = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert emulator.has_neutrinos is True
    assert emulator.background.N_mnu == 3


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_3degen_power_spectrum(background_w0wa_3degen, z_array, k_array):
    """Test w0waCDM 3degen JAX power spectrum output"""
    emulator = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linearcb_3degen_power_spectrum(background_w0wa_3degen, z_array, k_array):
    """Test w0waCDM 3degen Pcb JAX power spectrum"""
    emulator = w0waCDM.LinearCB(background=background_w0wa_3degen, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= w0waCDM NonLinear Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_initialization(background_w0wa, z_array):
    """Test w0waCDM nonlinear JAX emulator initializes correctly"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    assert hasattr(nonlinear, "Pk_int")
    assert hasattr(nonlinear, "sigma8")
    assert hasattr(nonlinear, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_power_spectrum(background_w0wa, z_array, k_array):
    """Test w0waCDM nonlinear JAX power spectrum"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_nl, np.ndarray)
    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_boost(background_w0wa, z_array, k_array):
    """Test that nonlinear power is boosted relative to linear on small scales"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_lin = linear.matter_power_spectrum(0.0, k_array)[0, :]
    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    # Nonlinear corrections should boost power on small scales
    high_k_mask = k_array > 1.0
    assert np.all(pk_nl[high_k_mask] > pk_lin[high_k_mask]), (
        "Nonlinear power should be boosted on small scales"
    )


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_sigma8(background_w0wa, z_array):
    """Test nonlinear sigma8 and growth_rate"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    sigma8_0 = nonlinear.sigma8_0()
    growth_rate = nonlinear.growth_rate()

    assert isinstance(sigma8_0, (float, np.floating))
    assert sigma8_0 > 0
    assert isinstance(growth_rate, np.ndarray)
    assert np.all(growth_rate > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinearcb(background_w0wa, z_array, k_array):
    """Test w0waCDM NonLinearCB JAX emulator"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)
    nonlinear_cb = w0waCDM.NonLinearCB(
        background=background_w0wa,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_cb_nl = nonlinear_cb.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb_nl > 0)
    assert hasattr(nonlinear_cb, "sigma8")
    assert hasattr(nonlinear_cb, "fsigma8")


# ============= wCDM Linear Tests (0 massive neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linear_power_spectrum(background_wcdm, z_array, k_array):
    """Test wCDM JAX power spectrum"""
    emulator = wCDM.Linear(background=background_wcdm, redshifts=z_array)

    pk = emulator.matter_power_spectrum(1.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))
    # Check sigma8 attributes
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_pcb_linear(background_wcdm, z_array, k_array):
    """Test wCDM Pcb JAX emulator"""
    emulator = wCDM.LinearCB(background=background_wcdm, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


# ============= wCDM Linear Tests (1 massive neutrino) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linear_1mass_initialization(background_wcdm_1mass, z_array):
    """Test wCDM 1mass JAX emulator initializes correctly"""
    emulator = wCDM.Linear(background=background_wcdm_1mass, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linear_1mass_power_spectrum(background_wcdm_1mass, z_array, k_array):
    """Test wCDM 1mass JAX power spectrum output"""
    emulator = wCDM.Linear(background=background_wcdm_1mass, redshifts=z_array)

    pk = emulator.matter_power_spectrum(1.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linearcb_1mass_power_spectrum(background_wcdm_1mass, z_array, k_array):
    """Test wCDM 1mass Pcb JAX power spectrum"""
    emulator = wCDM.LinearCB(background=background_wcdm_1mass, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= wCDM Linear Tests (3 degenerate neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linear_3degen_initialization(background_wcdm_3degen, z_array):
    """Test wCDM 3degen JAX emulator initializes correctly"""
    emulator = wCDM.Linear(background=background_wcdm_3degen, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert emulator.background.N_mnu == 3


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linear_3degen_power_spectrum(background_wcdm_3degen, z_array, k_array):
    """Test wCDM 3degen JAX power spectrum output"""
    emulator = wCDM.Linear(background=background_wcdm_3degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk > 0)


# ============= wCDM NonLinear Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_nonlinear_initialization(background_wcdm, z_array):
    """Test wCDM nonlinear JAX emulator initializes correctly"""
    linear = wCDM.Linear(background=background_wcdm, redshifts=z_array)
    nonlinear = wCDM.NonLinear(
        background=background_wcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    assert hasattr(nonlinear, "Pk_int")
    assert hasattr(nonlinear, "sigma8")
    assert hasattr(nonlinear, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_nonlinearcb(background_wcdm, z_array, k_array):
    """Test wCDM NonLinearCB JAX emulator"""
    linear = wCDM.Linear(background=background_wcdm, redshifts=z_array)
    nonlinear_cb = wCDM.NonLinearCB(
        background=background_wcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_cb_nl = nonlinear_cb.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb_nl > 0)


# ============= LCDM Linear Tests (0 massive neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_power_spectrum(background_lcdm, z_array, k_array):
    """Test LCDM JAX power spectrum"""
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert pk.shape[0] == len(k_array)
    # Check sigma8 attributes
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_growth_factor_normalization(background_lcdm, z_array, k_array):
    """Test LCDM growth factor is properly normalized"""
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        D_z0, 1.0, rtol=1e-6, err_msg="Growth factor should be 1 at z=0"
    )


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_sigma8(background_lcdm, z_array):
    """Test LCDM Linear class sigma8 and growth_rate"""
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_array)

    sigma8_0 = emulator.sigma8_0()
    growth_rate = emulator.growth_rate()

    assert isinstance(sigma8_0, (float, np.floating))
    assert sigma8_0 > 0
    assert isinstance(growth_rate, np.ndarray)
    assert np.all(growth_rate > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_pcb_linear(background_lcdm, z_array, k_array):
    """Test LCDM Pcb JAX emulator"""
    emulator = LCDM.LinearCB(background=background_lcdm, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(1.5, k_array)[0, :]

    assert np.all(pk_cb > 0)
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")


# ============= LCDM Linear Tests (1 massive neutrino) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_1mass_initialization(background_lcdm_1mass, z_array):
    """Test LCDM 1mass JAX emulator initializes correctly"""
    emulator = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_1mass_power_spectrum(background_lcdm_1mass, z_array, k_array):
    """Test LCDM 1mass JAX power spectrum output"""
    emulator = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert np.all(pk > 0)
    assert pk.shape[0] == len(k_array)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linearcb_1mass_power_spectrum(background_lcdm_1mass, z_array, k_array):
    """Test LCDM 1mass Pcb JAX power spectrum"""
    emulator = LCDM.LinearCB(background=background_lcdm_1mass, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(1.5, k_array)[0, :]

    assert np.all(pk_cb > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_1mass_growth_factor(background_lcdm_1mass, z_array, k_array):
    """Test LCDM 1mass growth factor"""
    emulator = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)

    D_z0 = emulator.growth_factor(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        D_z0, 1.0, rtol=1e-6, err_msg="Growth factor should be 1 at z=0"
    )


# ============= LCDM Linear Tests (3 degenerate neutrinos) =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_3degen_initialization(background_lcdm_3degen, z_array):
    """Test LCDM 3degen JAX emulator initializes correctly"""
    emulator = LCDM.Linear(background=background_lcdm_3degen, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert emulator.background.N_mnu == 3


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_3degen_power_spectrum(background_lcdm_3degen, z_array, k_array):
    """Test LCDM 3degen JAX power spectrum output"""
    emulator = LCDM.Linear(background=background_lcdm_3degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linearcb_3degen_power_spectrum(background_lcdm_3degen, z_array, k_array):
    """Test LCDM 3degen Pcb JAX power spectrum"""
    emulator = LCDM.LinearCB(background=background_lcdm_3degen, redshifts=z_array)

    pk_cb = emulator.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb > 0)


# ============= LCDM NonLinear Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_nonlinear_initialization(background_lcdm, z_array):
    """Test LCDM nonlinear JAX emulator initializes correctly"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nonlinear = LCDM.NonLinear(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    assert hasattr(nonlinear, "Pk_int")
    assert hasattr(nonlinear, "sigma8")
    assert hasattr(nonlinear, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_nonlinear_power_spectrum(background_lcdm, z_array, k_array):
    """Test LCDM nonlinear JAX power spectrum"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nonlinear = LCDM.NonLinear(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_nl, np.ndarray)
    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_nonlinearcb(background_lcdm, z_array, k_array):
    """Test LCDM NonLinearCB JAX emulator"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nonlinear_cb = LCDM.NonLinearCB(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_cb_nl = nonlinear_cb.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb_nl > 0)
    assert hasattr(nonlinear_cb, "sigma8")


# ============= Boundary & Error Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
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

    with pytest.raises(ValueError, match="out of emulator range"):
        w0waCDM.Linear(background=bad_background, redshifts=np.array([0.0, 1.0]))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
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


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_unsupported_neutrino_configuration():
    """Test that unsupported N_mnu (e.g. 4) raises ValueError"""
    bad_nu_background = DummyBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=0.06,
        N_mnu=4,
        w0=w0,
        wa=wa,
        gamma_MG=0.0,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        w0waCDM.Linear(background=bad_nu_background, redshifts=np.array([0.0, 1.0]))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_redshift_filtering(background_lcdm):
    """Test that emulator handles valid redshifts correctly"""
    z_valid = np.array([0.0, 1.0, 3.0, 4.5])
    emulator = LCDM.Linear(background=background_lcdm, redshifts=z_valid)

    assert np.all(emulator.z <= 5.0)


# ============= Comparison Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_pcb_vs_total_matter(background_lcdm, z_array, k_array):
    """Test that Pcb and total matter spectra are similar (no massive neutrinos)"""
    emulator_total = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    emulator_pcb = LCDM.LinearCB(background=background_lcdm, redshifts=z_array)

    pk_total = emulator_total.matter_power_spectrum(0.0, k_array)[0, :]
    pk_cb = emulator_pcb.matter_power_spectrum(0.0, k_array)[0, :]

    np.testing.assert_allclose(
        pk_total, pk_cb, rtol=0.01, err_msg="Pcb and total should match when mnu=0"
    )


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_str_representation_no_neutrinos(background_w0wa, z_array):
    """Test __str__ method for massless neutrinos"""
    emulator = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    info_str = str(emulator)

    assert "Cosmopower-JAX" in info_str
    assert "w0waCDM" in info_str
    assert "linear" in info_str
    assert "no massive neutrinos" in info_str


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_str_representation_1mass(background_w0wa_1mass, z_array):
    """Test __str__ method for 1 massive neutrino"""
    emulator = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)

    info_str = str(emulator)

    assert "Cosmopower-JAX" in info_str
    assert "w0waCDM" in info_str
    assert "one massive neutrino" in info_str


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_str_representation_3degen(background_w0wa_3degen, z_array):
    """Test __str__ method for 3 degenerate neutrinos"""
    emulator = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)

    info_str = str(emulator)

    assert "Cosmopower-JAX" in info_str
    assert "w0waCDM" in info_str
    assert "three degenerate massive neutrinos" in info_str


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_str_representation_nonlinear(background_lcdm, z_array):
    """Test __str__ method for nonlinear class"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nonlinear = LCDM.NonLinear(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    info_str = str(nonlinear)

    assert "Cosmopower-JAX" in info_str
    assert "LCDM" in info_str
    assert "nonlinear" in info_str


# ============= NonLinear with Massive Neutrinos Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_1mass(background_w0wa_1mass, z_array, k_array):
    """Test w0waCDM nonlinear with 1 massive neutrino"""
    linear = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa_1mass,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert isinstance(pk_nl, np.ndarray)
    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))
    assert hasattr(nonlinear, "sigma8")
    assert hasattr(nonlinear, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinearcb_1mass(background_w0wa_1mass, z_array, k_array):
    """Test w0waCDM NonLinearCB with 1 massive neutrino"""
    linear = w0waCDM.Linear(background=background_w0wa_1mass, redshifts=z_array)
    nonlinear_cb = w0waCDM.NonLinearCB(
        background=background_w0wa_1mass,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_cb_nl = nonlinear_cb.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb_nl > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_3degen(background_w0wa_3degen, z_array, k_array):
    """Test w0waCDM nonlinear with 3 degenerate neutrinos"""
    linear = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa_3degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinearcb_3degen(background_w0wa_3degen, z_array, k_array):
    """Test w0waCDM NonLinearCB with 3 degenerate neutrinos"""
    linear = w0waCDM.Linear(background=background_w0wa_3degen, redshifts=z_array)
    nonlinear_cb = w0waCDM.NonLinearCB(
        background=background_w0wa_3degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_cb_nl = nonlinear_cb.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_cb_nl > 0)


# ============= wCDM NonLinear with Massive Neutrinos Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_nonlinear_1mass(background_wcdm_1mass, z_array, k_array):
    """Test wCDM nonlinear with 1 massive neutrino"""
    linear = wCDM.Linear(background=background_wcdm_1mass, redshifts=z_array)
    nonlinear = wCDM.NonLinear(
        background=background_wcdm_1mass,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_nonlinear_3degen(background_wcdm_3degen, z_array, k_array):
    """Test wCDM nonlinear with 3 degenerate neutrinos"""
    linear = wCDM.Linear(background=background_wcdm_3degen, redshifts=z_array)
    nonlinear = wCDM.NonLinear(
        background=background_wcdm_3degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)


# ============= LCDM NonLinear with Massive Neutrinos Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_nonlinear_1mass(background_lcdm_1mass, z_array, k_array):
    """Test LCDM nonlinear with 1 massive neutrino"""
    linear = LCDM.Linear(background=background_lcdm_1mass, redshifts=z_array)
    nonlinear = LCDM.NonLinear(
        background=background_lcdm_1mass,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_nonlinear_3degen(background_lcdm_3degen, z_array, k_array):
    """Test LCDM nonlinear with 3 degenerate neutrinos"""
    linear = LCDM.Linear(background=background_lcdm_3degen, redshifts=z_array)
    nonlinear = LCDM.NonLinear(
        background=background_lcdm_3degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)


# ============= 2-Degen Neutrino Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_linear_2degen(background_w0wa_2degen, z_array, k_array):
    """Test w0waCDM linear with 2 degenerate neutrinos"""
    emulator = w0waCDM.Linear(background=background_w0wa_2degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_nonlinear_2degen(background_w0wa_2degen, z_array, k_array):
    """Test w0waCDM nonlinear with 2 degenerate neutrinos"""
    linear = w0waCDM.Linear(background=background_w0wa_2degen, redshifts=z_array)
    nonlinear = w0waCDM.NonLinear(
        background=background_w0wa_2degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_linear_2degen(background_wcdm_2degen, z_array, k_array):
    """Test wCDM linear with 2 degenerate neutrinos"""
    emulator = wCDM.Linear(background=background_wcdm_2degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_nonlinear_2degen(background_wcdm_2degen, z_array, k_array):
    """Test wCDM nonlinear with 2 degenerate neutrinos"""
    linear = wCDM.Linear(background=background_wcdm_2degen, redshifts=z_array)
    nonlinear = wCDM.NonLinear(
        background=background_wcdm_2degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_linear_2degen(background_lcdm_2degen, z_array, k_array):
    """Test LCDM linear with 2 degenerate neutrinos"""
    emulator = LCDM.Linear(background=background_lcdm_2degen, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_nonlinear_2degen(background_lcdm_2degen, z_array, k_array):
    """Test LCDM nonlinear with 2 degenerate neutrinos"""
    linear = LCDM.Linear(background=background_lcdm_2degen, redshifts=z_array)
    nonlinear = LCDM.NonLinear(
        background=background_lcdm_2degen,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


# ============= log10TAGN Bounds Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_log10TAGN_below_bounds(background_w0wa, z_array):
    """Test that log10TAGN below bounds raises error"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    with pytest.raises(ValueError, match="out of emulator range"):
        w0waCDM.NonLinear(
            background=background_w0wa,
            linearperturbations=linear,
            redshifts=z_array,
            log10TAGN=7.0,  # Below 7.3 bound
        )


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_log10TAGN_above_bounds(background_w0wa, z_array):
    """Test that log10TAGN above bounds raises error"""
    linear = w0waCDM.Linear(background=background_w0wa, redshifts=z_array)

    with pytest.raises(ValueError, match="out of emulator range"):
        w0waCDM.NonLinear(
            background=background_w0wa,
            linearperturbations=linear,
            redshifts=z_array,
            log10TAGN=9.0,  # Above 8.5 bound
        )


# ============= Curvature Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_linear_initialization(background_curvature, z_array):
    """Test LCDM+curvature linear emulator initializes correctly"""
    emulator = Curvature.Linear(background=background_curvature, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_linear_power_spectrum(background_curvature, z_array, k_array):
    """Test LCDM+curvature linear power spectrum output"""
    emulator = Curvature.Linear(background=background_curvature, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_linear_redshift_evolution(background_curvature, z_array, k_array):
    """Test LCDM+curvature power spectrum decreases with redshift"""
    emulator = Curvature.Linear(background=background_curvature, redshifts=z_array)

    pk_z0 = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    pk_z2 = emulator.matter_power_spectrum(2.0, k_array)[0, :]

    assert np.all(pk_z0 > pk_z2)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_nonlinear(background_curvature, z_array, k_array):
    """Test LCDM+curvature nonlinear emulator"""
    linear = Curvature.Linear(background=background_curvature, redshifts=z_array)
    nonlinear = Curvature.NonLinear(
        background=background_curvature,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_nonlinear_exceeds_linear(background_curvature, z_array, k_array):
    """Test nonlinear P(k) exceeds linear at small scales for curvature model"""
    linear = Curvature.Linear(background=background_curvature, redshifts=z_array)
    nonlinear = Curvature.NonLinear(
        background=background_curvature,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    k_small = k_array[k_array > 0.5]
    pk_lin = linear.matter_power_spectrum(0.0, k_small)[0, :]
    pk_nl = nonlinear.matter_power_spectrum(0.0, k_small)[0, :]

    assert np.any(pk_nl > pk_lin)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_linear_pcb(background_curvature, z_array, k_array):
    """Test LCDM+curvature linear P_cb emulator"""
    emulator = Curvature.LinearCB(background=background_curvature, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_nonlinear_pcb(background_curvature, z_array, k_array):
    """Test LCDM+curvature nonlinear P_cb emulator"""
    linear = Curvature.LinearCB(background=background_curvature, redshifts=z_array)
    nonlinear = Curvature.NonLinearCB(
        background=background_curvature,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


# ============= Running Spectral Index Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_linear_initialization(background_running, z_array):
    """Test LCDM+running spectral index linear emulator initializes correctly"""
    emulator = RunningIndex.Linear(background=background_running, redshifts=z_array)

    assert hasattr(emulator, "Pk_int")
    assert hasattr(emulator, "k")
    assert hasattr(emulator, "z")
    assert hasattr(emulator, "sigma8")
    assert hasattr(emulator, "fsigma8")
    assert emulator.k_min > 0
    assert emulator.k_max > emulator.k_min


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_linear_power_spectrum(background_running, z_array, k_array):
    """Test LCDM+running spectral index linear power spectrum output"""
    emulator = RunningIndex.Linear(background=background_running, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_array)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_linear_redshift_evolution(background_running, z_array, k_array):
    """Test LCDM+running spectral index power spectrum decreases with redshift"""
    emulator = RunningIndex.Linear(background=background_running, redshifts=z_array)

    pk_z0 = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    pk_z2 = emulator.matter_power_spectrum(2.0, k_array)[0, :]

    assert np.all(pk_z0 > pk_z2)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_nonlinear(background_running, z_array, k_array):
    """Test LCDM+running spectral index nonlinear emulator"""
    linear = RunningIndex.Linear(background=background_running, redshifts=z_array)
    nonlinear = RunningIndex.NonLinear(
        background=background_running,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk_nl = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk_nl > 0)
    assert np.all(np.isfinite(pk_nl))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_nonlinear_exceeds_linear(background_running, z_array, k_array):
    """Test nonlinear P(k) exceeds linear at small scales for running index model"""
    linear = RunningIndex.Linear(background=background_running, redshifts=z_array)
    nonlinear = RunningIndex.NonLinear(
        background=background_running,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    k_small = k_array[k_array > 0.5]
    pk_lin = linear.matter_power_spectrum(0.0, k_small)[0, :]
    pk_nl = nonlinear.matter_power_spectrum(0.0, k_small)[0, :]

    assert np.any(pk_nl > pk_lin)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_linear_pcb(background_running, z_array, k_array):
    """Test LCDM+running spectral index linear P_cb emulator"""
    emulator = RunningIndex.LinearCB(background=background_running, redshifts=z_array)

    pk = emulator.matter_power_spectrum(z_array[0], k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_nonlinear_pcb(background_running, z_array, k_array):
    """Test LCDM+running spectral index nonlinear P_cb emulator"""
    linear = RunningIndex.LinearCB(background=background_running, redshifts=z_array)
    nonlinear = RunningIndex.NonLinearCB(
        background=background_running,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk = nonlinear.matter_power_spectrum(0.0, k_array)[0, :]

    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


# ============= Curvature / Running neutrino variants (N_mnu = 0, 3) =============
# Extended-cosmology classes branch on N_mnu in {0, 1, 3} (no 2-degenerate).


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_0mass(background_curvature_0mass, z_array, k_array):
    """LCDM+curvature with massless neutrinos"""
    emulator = Curvature.Linear(
        background=background_curvature_0mass, redshifts=z_array
    )

    assert emulator.has_neutrinos is False
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_3degen(background_curvature_3degen, z_array, k_array):
    """LCDM+curvature with 3 degenerate massive neutrinos"""
    emulator = Curvature.Linear(
        background=background_curvature_3degen, redshifts=z_array
    )

    assert emulator.has_neutrinos is True
    assert emulator.background.N_mnu == 3
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_0mass(background_running_0mass, z_array, k_array):
    """LCDM+running with massless neutrinos"""
    emulator = RunningIndex.Linear(
        background=background_running_0mass, redshifts=z_array
    )

    assert emulator.has_neutrinos is False
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_3degen(background_running_3degen, z_array, k_array):
    """LCDM+running with 3 degenerate massive neutrinos"""
    emulator = RunningIndex.Linear(
        background=background_running_3degen, redshifts=z_array
    )

    assert emulator.has_neutrinos is True
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_curvature_rejects_2degen():
    """Curvature emulators only support N_mnu in {0, 1, 3}; N_mnu=2 must raise"""
    bg = _extended_bg(N_mnu=2, mnu=0.06, Omega_k0=0.02, alpha_s=0.0)
    with pytest.raises(ValueError, match="Unsupported"):
        Curvature.Linear(background=bg, redshifts=np.array([0.0, 1.0]))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_running_rejects_2degen():
    """Running emulators only support N_mnu in {0, 1, 3}; N_mnu=2 must raise"""
    bg = _extended_bg(N_mnu=2, mnu=0.06, Omega_k0=0.0, alpha_s=0.01)
    with pytest.raises(ValueError, match="Unsupported"):
        RunningIndex.Linear(background=bg, redshifts=np.array([0.0, 1.0]))


# ============= w0waCDM + Curvature Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_curvature_linear(background_w0wa_curvature, z_array, k_array):
    """w0waCDM+curvature linear emulator"""
    emulator = w0waCurvature.Linear(
        background=background_w0wa_curvature, redshifts=z_array
    )

    assert emulator.k_min > 0 and emulator.k_max > emulator.k_min
    assert hasattr(emulator, "sigma8") and hasattr(emulator, "fsigma8")
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_curvature_nonlinear(background_w0wa_curvature, z_array, k_array):
    """w0waCDM+curvature nonlinear P(k) and P_cb(k)"""
    linear = w0waCurvature.Linear(
        background=background_w0wa_curvature, redshifts=z_array
    )
    nl = w0waCurvature.NonLinear(
        background=background_w0wa_curvature,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )
    cb = w0waCurvature.NonLinearCB(
        background=background_w0wa_curvature,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk = nl.matter_power_spectrum(0.0, k_array)[0, :]
    pk_cb = cb.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0) and np.all(np.isfinite(pk))
    assert np.all(pk_cb > 0) and np.all(np.isfinite(pk_cb))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_curvature_0mass(background_w0wa_curvature_0mass, z_array, k_array):
    """w0waCDM+curvature with massless neutrinos"""
    emulator = w0waCurvature.Linear(
        background=background_w0wa_curvature_0mass, redshifts=z_array
    )

    assert emulator.has_neutrinos is False
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_curvature_rejects_2degen():
    """w0waCDM+curvature must reject N_mnu=2"""
    bg = _extended_bg(N_mnu=2, mnu=0.06, Omega_k0=0.02, alpha_s=0.0, w0=-0.9, wa=0.1)
    with pytest.raises(ValueError, match="Unsupported"):
        w0waCurvature.Linear(background=bg, redshifts=np.array([0.0, 1.0]))


# ============= w0waCDM + Running Spectral Index Tests =============


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_running_linear(background_w0wa_running, z_array, k_array):
    """w0waCDM+running linear emulator"""
    emulator = w0waRunningIndex.Linear(
        background=background_w0wa_running, redshifts=z_array
    )

    assert hasattr(emulator, "sigma8") and hasattr(emulator, "fsigma8")
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_running_nonlinear(background_w0wa_running, z_array, k_array):
    """w0waCDM+running nonlinear P(k) and P_cb(k)"""
    linear = w0waRunningIndex.Linear(
        background=background_w0wa_running, redshifts=z_array
    )
    nl = w0waRunningIndex.NonLinear(
        background=background_w0wa_running,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )
    cb = w0waRunningIndex.NonLinearCB(
        background=background_w0wa_running,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )

    pk = nl.matter_power_spectrum(0.0, k_array)[0, :]
    pk_cb = cb.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0) and np.all(np.isfinite(pk))
    assert np.all(pk_cb > 0) and np.all(np.isfinite(pk_cb))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_running_3degen(background_w0wa_running_3degen, z_array, k_array):
    """w0waCDM+running with 3 degenerate massive neutrinos"""
    emulator = w0waRunningIndex.Linear(
        background=background_w0wa_running_3degen, redshifts=z_array
    )

    assert emulator.has_neutrinos is True
    pk = emulator.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)


# ============= Halofit nonlinear prescription Tests =============
# Halofit is a dark-matter-only nonlinear recipe available for LCDM/wCDM/w0waCDM.
# log10TAGN is accepted for interface compatibility with the HMcode NonLinear
# classes but ignored (no baryonic feedback).


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_halofit_nonlinear(background_lcdm, z_array, k_array):
    """LCDM halofit nonlinear P(k) (massless)"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nl = LCDM.NonLinearHalofit(
        background=background_lcdm, linearperturbations=linear, redshifts=z_array
    )

    pk = nl.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))
    assert hasattr(nl, "sigma8")
    assert hasattr(nl, "fsigma8")


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_lcdm_halofit_nonlinearcb(background_lcdm, z_array, k_array):
    """LCDM halofit nonlinear P_cb(k)"""
    linear = LCDM.LinearCB(background=background_lcdm, redshifts=z_array)
    nl = LCDM.NonLinearHalofitCB(
        background=background_lcdm, linearperturbations=linear, redshifts=z_array
    )

    pk = nl.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
@pytest.mark.parametrize(
    "fixture_name",
    ["background_lcdm_1mass", "background_lcdm_2degen", "background_lcdm_3degen"],
)
def test_lcdm_halofit_neutrino_variants(fixture_name, request, z_array, k_array):
    """LCDM halofit supports N_mnu = 1, 2, 3 (halofit has a 2mass emulator)"""
    bg = request.getfixturevalue(fixture_name)
    linear = LCDM.Linear(background=bg, redshifts=z_array)
    nl = LCDM.NonLinearHalofit(
        background=bg, linearperturbations=linear, redshifts=z_array
    )

    pk = nl.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_halofit_ignores_log10TAGN(background_lcdm, z_array, k_array):
    """halofit is DM-only: different log10TAGN must give identical P(k)"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nl_a = LCDM.NonLinearHalofit(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=7.5,
    )
    nl_b = LCDM.NonLinearHalofit(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=8.4,
    )

    pk_a = nl_a.matter_power_spectrum(0.0, k_array)[0, :]
    pk_b = nl_b.matter_power_spectrum(0.0, k_array)[0, :]
    np.testing.assert_allclose(
        pk_a, pk_b, err_msg="halofit must ignore log10TAGN (dark-matter-only)"
    )


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_halofit_accepts_out_of_hmcode_log10TAGN(background_lcdm, z_array, k_array):
    """halofit accepts a log10TAGN outside the HMcode [7.3, 8.5] box (ignored)"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    # 9.0 would raise for the HMcode NonLinear class, but halofit ignores it.
    nl = LCDM.NonLinearHalofit(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=9.0,
    )

    pk = nl.matter_power_spectrum(0.0, k_array)[0, :]
    assert np.all(pk > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_halofit_differs_from_hmcode(background_lcdm, z_array, k_array):
    """halofit and HMcode are distinct nonlinear prescriptions"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    hm = LCDM.NonLinear(
        background=background_lcdm,
        linearperturbations=linear,
        redshifts=z_array,
        log10TAGN=log10TAGN,
    )
    hf = LCDM.NonLinearHalofit(
        background=background_lcdm, linearperturbations=linear, redshifts=z_array
    )

    pk_hm = hm.matter_power_spectrum(0.0, k_array)[0, :]
    pk_hf = hf.matter_power_spectrum(0.0, k_array)[0, :]
    high_k = k_array > 1.0
    assert not np.allclose(pk_hm[high_k], pk_hf[high_k])


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_wcdm_halofit(background_wcdm_1mass, z_array, k_array):
    """wCDM halofit nonlinear P(k) and P_cb(k)"""
    linear = wCDM.Linear(background=background_wcdm_1mass, redshifts=z_array)
    nl = wCDM.NonLinearHalofit(
        background=background_wcdm_1mass, linearperturbations=linear, redshifts=z_array
    )
    cb = wCDM.NonLinearHalofitCB(
        background=background_wcdm_1mass, linearperturbations=linear, redshifts=z_array
    )

    assert np.all(nl.matter_power_spectrum(0.0, k_array)[0, :] > 0)
    assert np.all(cb.matter_power_spectrum(0.0, k_array)[0, :] > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_w0wa_halofit(background_w0wa_2degen, z_array, k_array):
    """w0waCDM halofit nonlinear P(k) and P_cb(k) (2 massive neutrinos)"""
    linear = w0waCDM.Linear(background=background_w0wa_2degen, redshifts=z_array)
    nl = w0waCDM.NonLinearHalofit(
        background=background_w0wa_2degen, linearperturbations=linear, redshifts=z_array
    )
    cb = w0waCDM.NonLinearHalofitCB(
        background=background_w0wa_2degen, linearperturbations=linear, redshifts=z_array
    )

    assert np.all(nl.matter_power_spectrum(0.0, k_array)[0, :] > 0)
    assert np.all(cb.matter_power_spectrum(0.0, k_array)[0, :] > 0)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
def test_halofit_str(background_lcdm, z_array):
    """halofit __str__ identifies the prescription"""
    linear = LCDM.Linear(background=background_lcdm, redshifts=z_array)
    nl = LCDM.NonLinearHalofit(
        background=background_lcdm, linearperturbations=linear, redshifts=z_array
    )

    info_str = str(nl)
    assert "Cosmopower-JAX" in info_str
    assert "halofit" in info_str
    assert "LCDM" in info_str
