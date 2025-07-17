import pytest
import numpy as np
from cloelib.cosmology.cosmopower_cosmology import w0waCDM_Linear, w0waCDM_1mass_Linear, w0waCDM_3degen_Linear, LCDM_Linear, LCDM_1mass_Linear, LCDM_3degen_Linear
# from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.cosmology import Background
try:
    import cosmopower
    HAS_COSMOPOWER = True
except ImportError:
    HAS_COSMOPOWER = False

H0 =  67.7
h = H0/100.
sigma8 = 0.8277
omch2 = 0.12
Omega_cdm0 = omch2/h**2
ombh2 = 0.022
Omega_b0 = ombh2/h**2
Omega_k0 = 0.0
w = -1.
wa = 0.
ns = 0.96
mnu = 0.06
As=2e-9



class DummyBackground:
    def __init__(self, H0, Omega_b0, Omega_cdm0, Omega_k0, As, ns, mnu, w0, wa):
        self.H0 = H0
        self.h = H0 / 100.
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.mnu = mnu
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = 0.0
        self._interface_args = {}

    def Omega_b(self, zs): return np.ones_like(zs)
    def Omega_m(self, zs): return np.ones_like(zs)
    def hubble_parameter(self, zs, units="km/s/Mpc"): return np.ones_like(zs)
    def comoving_distance(self, zs): return np.ones_like(zs)
    def transverse_comoving_distance(self, zs): return np.ones_like(zs)
    def angular_diameter_distance(self, zs): return np.ones_like(zs)

@pytest.fixture
def background_instance():
    return DummyBackground(H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0,
                           Omega_k0=Omega_k0, As=As, ns=ns, mnu=mnu, w0=w, wa=wa)

@pytest.fixture
def k_emu():
    return np.logspace(-3, 1, 100)  # 100 k-values between 1e-3 and 10

@pytest.fixture
def z():
    return np.linspace(0, 2, 10)  # 10 redshifts from z=0 to z=2

@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_emulator(background_instance, z, k_emu):
    emulator = w0waCDM_Linear(background=background_instance, redshifts=z)
    
    # Call matter power spectrum
    pk = emulator.matter_power_spectrum(0, k_emu)[0, :]

    # Assertions
    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_emu)
    assert np.all(pk > 0), "Power spectrum should be positive"

@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_1mass_emulator(background_instance, z, k_emu):
    emulator = w0waCDM_1mass_Linear(background=background_instance, redshifts=z)
    
    # Call matter power spectrum
    pk = emulator.matter_power_spectrum(0, k_emu)[0, :]

    # Assertions
    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_emu)
    assert np.all(pk > 0), "Power spectrum should be positive"

@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_w0wa_3degen_emulator(background_instance, z, k_emu):
    emulator = w0waCDM_3degen_Linear(background=background_instance, redshifts=z)
    
    # Call matter power spectrum
    pk = emulator.matter_power_spectrum(0, k_emu)[0, :]

    # Assertions
    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_emu)
    assert np.all(pk > 0), "Power spectrum should be positive"

@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_LCDM_emulator(background_instance, z, k_emu):
    emulator = LCDM_Linear(background=background_instance, redshifts=z)
    
    # Call matter power spectrum
    pk = emulator.matter_power_spectrum(0, k_emu)[0, :]

    # Assertions
    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_emu)
    assert np.all(pk > 0), "Power spectrum should be positive" 

@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_LCDM_1mass_emulator(background_instance, z, k_emu):
    emulator = LCDM_1mass_Linear(background=background_instance, redshifts=z)
    
    # Call matter power spectrum
    pk = emulator.matter_power_spectrum(0, k_emu)[0, :]

    # Assertions
    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_emu)
    assert np.all(pk > 0), "Power spectrum should be positive"

@pytest.mark.skipif(not HAS_COSMOPOWER, reason="cosmopower not installed")
def test_LCDM_3degen_emulator(background_instance, z, k_emu):
    emulator = LCDM_3degen_Linear(background=background_instance, redshifts=z)
    
    # Call matter power spectrum
    pk = emulator.matter_power_spectrum(0, k_emu)[0, :]

    # Assertions
    assert isinstance(pk, np.ndarray)
    assert pk.shape[0] == len(k_emu)
    assert np.all(pk > 0), "Power spectrum should be positive"