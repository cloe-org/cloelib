import numpy as np
import pytest
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.cosmology.ReACTEmu_cosmology import (
    MGemuNonlinearBoost,
    BoostedPerturbations,
)

import urllib.request
import zipfile
from pathlib import Path


# -------------------------------
# Download utility
# -------------------------------

VALIDATION_URL = "https://drive.google.com/uc?id=16IftTSG1g7bVGhaajWJPSAln06XVOijI"


def ensure_validation_data() -> Path:
    """Download & extract validation_data.zip into the same directory as this test file."""
    here = Path(__file__).parent
    zip_path = here / "validation_data.zip"
    extract_dir = here  # extract right here, so we get here/validation_data/...

    if not (extract_dir / "validation_data").exists():
        if not zip_path.exists():
            print(f"Downloading validation dataset from {VALIDATION_URL} → {zip_path}")
            urllib.request.urlretrieve(VALIDATION_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir / "validation_data"


# -------------------------------
# Fixtures using actual CAMB class
# -------------------------------


@pytest.fixture(scope="module")
def camb_background():
    return CAMBBackground(
        H0=67.39774575153639,
        Omega_b0=0.05062684354655124,
        Omega_cdm0=0.3164470361777248 - 0.05062684354655124,
        Omega_k0=0.0,
        As=2.1977194699875245e-9,
        ns=0.9423180532642602,
        mnu=0.0,
        w0=-1,
        wa=0,
        gamma_MG=0.0,
        N_mnu=0.0,
    )


@pytest.fixture(scope="module")
def camb_linear(camb_background):
    zs = np.array([0.1, 0.478, 0.785, 1.5])
    linear = CAMBLinearPerturbations(camb_background, zs)
    return linear, zs


# -------------------------
# Test: MGemu boost runs
# -------------------------
def test_mgemu_initializes_correctly(camb_background, camb_linear):
    linear_pert, zs = camb_linear

    # New required MG parameters (use benign defaults for unused ones)
    mgpars = {
        "fr0": 1e-4,  # was MGp1
        "omega_rc": 0.0,  # for DGP; unused for 'fr'
        "gamma0": 0.55,  # for Linder gamma; unused for 'fr'
        "q1": 0.0,  # whatever default is safe in your code
    }

    obj = MGemuNonlinearBoost(
        camb_background, linear_pert, zs, gravity_model="fr", mgpars=mgpars
    )

    val = obj.mg_spectrum_boost(0.5, 0.2)
    assert isinstance(val, float) or isinstance(val, np.ndarray)


# -------------------------
# Test: Boosted spectrum interface
# -------------------------
def test_boosted_spectrum_shape():
    class DummyBackground:
        Omega_k0 = 0.0
        w0 = -1.0
        wa = 0.0
        # (optional) other fields you often read
        h = 0.67
        H0 = 100 * h
        Omega_b0 = 0.05
        Omega_cdm0 = 0.25
        mnu = 0.0
        As = 2.1e-9
        ns = 0.965
        N_mnu = 0.0

    class DummyLin:
        background = DummyBackground()

    class DummyBase:
        background = DummyBackground()

        def matter_power_spectrum(self, z, k):
            z = np.atleast_1d(z)
            k = np.atleast_1d(k)
            return np.outer(np.ones_like(z), np.ones_like(k)) * 2.0

    def dummy_boost(z, k, grid=False):
        return np.outer(np.ones_like(z), np.ones_like(k)) * 1.5

    z = np.array([0.1, 0.5])
    k = np.array([0.01, 0.1, 1.0])
    boosted = BoostedPerturbations(DummyLin(), DummyBase(), dummy_boost)
    result = boosted.matter_power_spectrum(z, k)

    assert result.shape == (2, 3)
    assert np.allclose(result, 3.0)


def test_boosted_scalar_input():
    class DummyBackground:
        Omega_k0 = 0.0
        w0 = -1.0
        wa = 0.0
        # (optional) other fields you often read
        h = 0.67
        H0 = 100 * h
        Omega_b0 = 0.05
        Omega_cdm0 = 0.25
        mnu = 0.0
        As = 2.1e-9
        ns = 0.965

    class DummyLin:
        background = DummyBackground()

    class DummyBase:
        background = DummyBackground()

        def matter_power_spectrum(self, z, k):
            z = np.atleast_1d(z)
            k = np.atleast_1d(k)
            return np.outer(np.ones_like(z), np.ones_like(k)) * 2.0

    def dummy_boost(z, k, grid=False):
        return np.array([1.5])

    boosted = BoostedPerturbations(DummyLin(), DummyBase(), dummy_boost)
    val = boosted.matter_power_spectrum(0.5, 0.1)
    assert np.isclose(val, 3.0)


# -------------------------
# Validation against external data
# -------------------------
@pytest.mark.parametrize(
    "model_name,mgparam,filename",
    [
        ("fr", 1e-5, "fR_validation_data.dat"),
        ("dgp", 0.1, "dgp_validation_data.dat"),
    ],
)
def test_mg_boost_matches_validation(
    camb_background, camb_linear, model_name, mgparam, filename
):
    # Get external data directory (download & unzip if needed)
    data_dir = ensure_validation_data()
    data_path = data_dir / filename

    linear_pert, zs = camb_linear

    # Map the single scalar 'mgparam' into the new dict the class expects.
    # Provide benign defaults for unused keys (class reads them unconditionally).
    if model_name == "fr":
        mgpars = {"fr0": mgparam, "omega_rc": 0.0, "gamma0": 0.55, "q1": 0.0}
    elif model_name == "dgp":
        mgpars = {"fr0": 0.0, "omega_rc": mgparam, "gamma0": 0.55, "q1": 0.0}
    else:
        raise ValueError(f"Unsupported model in test: {model_name}")

    boost_obj = MGemuNonlinearBoost(
        camb_background, linear_pert, zs, gravity_model=model_name, mgpars=mgpars
    )
    interp = boost_obj.MGboost_interp

    # Load validation data
    data = np.loadtxt(str(data_path))

    # Original k in h/Mpc (from data)
    k_raw = data[:, 0]
    assert k_raw.min() <= 0.01 and k_raw.max() >= 3.0, "k range incomplete"

    # Convert to 1/Mpc
    k_vals_physical = k_raw * camb_background.h

    # Clamp k-values to emulator-supported range
    kmin = interp.get_knots()[1][0]
    kmax = interp.get_knots()[1][-1]
    k_vals = np.clip(k_vals_physical, kmin, kmax)

    z_map = {1.5: 1, 0.785: 2, 0.478: 3, 0.1: 4}  # column indices

    for z in zs:
        model_B = interp(z, k_vals, grid=False)
        data_B = data[:, z_map[z]]
        rel_err = np.abs((model_B - data_B) / data_B)

        mask = (k_vals >= 0.01) & (k_vals <= 3.0)
        max_err = rel_err[mask].max()

        assert max_err < 0.005, (
            f"{model_name} boost error > 0.5% at z={z} (max rel error = {max_err:.5f})"
        )
