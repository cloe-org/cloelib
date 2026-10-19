"""Check which emulator files each CosmoPower-JAX class loads and what it feeds them.

Every class shares one setup routine, and a wrong file name there would still load
a valid emulator and return a positive spectrum, so the spectrum tests in
``test_cosmopower_jax_emulators.py`` would keep passing for the wrong cosmology.
These tests pin, for every class and neutrino configuration, the exact emulator
files and the exact inputs passed to each of them, and compare those inputs with
the parameter names stored in the real emulator files. They also cover the
download helper and the bounds checker.
"""

import functools
import importlib.util
import os
import urllib.request
import zlib

import numpy as np
import pytest

import cloelib.cosmology.cosmopower_jax_cosmology as cpj
from cloelib.cosmology.cosmopower_jax_cosmology import (
    CosmoPowerJAXLCDMCurvaturePerturbations as LCDMCurvature,
    CosmoPowerJAXLCDMPerturbations as LCDM,
    CosmoPowerJAXLCDMRunningIndexPerturbations as LCDMRunning,
    CosmoPowerJAXw0waCDMPerturbations as W0WACDM,
    CosmoPowerJAXw0waCurvaturePerturbations as W0WACurvature,
    CosmoPowerJAXw0waRunningIndexPerturbations as W0WARunning,
    CosmoPowerJAXwCDMPerturbations as WCDM,
    check_emulator_bounds,
)

HAS_COSMOPOWER_JAX = importlib.util.find_spec("cosmopower_jax") is not None

GITHUB_RAW = (
    "https://raw.githubusercontent.com/cosmopower-organization/"
    "Euclid-DR1-matter-emulators/main/emulators"
)

# Background values. They are all different, so an input read from the wrong
# background attribute shows up as a wrong value.
H0, OMEGA_B0, OMEGA_CDM0, AS, NS = 68.0, 0.048, 0.26, 2.1e-9, 0.965
MNU, W0, WA, OMEGA_K0, ALPHA_S = 0.12, -0.9, 0.2, 0.05, 0.01
LOG10TAGN = 7.9  # passed to every nonlinear class; only HMcode may use it
FIDUCIAL_LOG10TAGN = 7.6  # what the code feeds sigma8 emulators that need it
REDSHIFTS = np.array([0.5, 1.0, 2.0])
Z_GRID = np.array([0.0, 0.5, 1.0, 2.0])  # z = 0 is always added

BASE_N_MNU = (0, 1, 2, 3)
EXTENDED_N_MNU = (0, 1, 3)

# Per family: supported N_mnu, Omega_k0 of the test background, and the inputs it
# adds to the base LCDM set, with the background value each one must carry.
FAMILIES = {
    LCDM: (BASE_N_MNU, 0.0, {}),
    WCDM: (BASE_N_MNU, 0.0, {"w": W0}),
    W0WACDM: (BASE_N_MNU, 0.0, {"w0": W0, "wa": WA}),
    LCDMCurvature: (EXTENDED_N_MNU, OMEGA_K0, {"omk": OMEGA_K0}),
    W0WACurvature: (EXTENDED_N_MNU, OMEGA_K0, {"w0": W0, "wa": WA, "omk": OMEGA_K0}),
    LCDMRunning: (EXTENDED_N_MNU, 0.0, {"alpha_s": ALPHA_S}),
    W0WARunning: (EXTENDED_N_MNU, 0.0, {"w0": W0, "wa": WA, "alpha_s": ALPHA_S}),
}

# Per class: the (P(k), cb P(k), sigma8/fsigma8) files it must load. Written out
# by hand on purpose, so it does not reuse the module's own filename templates.
EXPECTED_FILES = {
    LCDM.Linear: (
        "lcdm-{n}mass-linear.npz",
        "lcdm-{n}mass-cb-linear.npz",
        "lcdm-{n}mass-s8-fs8.npz",
    ),
    LCDM.NonLinear: (
        "lcdm-{n}mass-nonlinear.npz",
        "lcdm-{n}mass-cb-nonlinear.npz",
        "lcdm-{n}mass-s8-fs8.npz",
    ),
    LCDM.NonLinearHalofit: (
        "halofit-lcdm-{n}mass-nonlinear.npz",
        "halofit-lcdm-{n}mass-cb-nonlinear.npz",
        "halofit-lcdm-{n}mass-s8-fs8.npz",
    ),
    WCDM.Linear: (
        "wcdm-{n}mass-linear.npz",
        "wcdm-{n}mass-cb-linear.npz",
        "wcdm-{n}mass-s8-fs8.npz",
    ),
    WCDM.NonLinear: (
        "wcdm-{n}mass-nonlinear.npz",
        "wcdm-{n}mass-cb-nonlinear.npz",
        "wcdm-{n}mass-s8-fs8.npz",
    ),
    WCDM.NonLinearHalofit: (
        "halofit-wcdm-{n}mass-nonlinear.npz",
        "halofit-wcdm-{n}mass-cb-nonlinear.npz",
        "halofit-wcdm-{n}mass-s8-fs8.npz",
    ),
    W0WACDM.Linear: (
        "w0wa-{n}mass-linear.npz",
        "w0wa-{n}mass-cb-linear.npz",
        "w0wa-{n}mass-s8-fs8.npz",
    ),
    W0WACDM.NonLinear: (
        "w0wa-{n}mass-nonlinear.npz",
        "w0wa-{n}mass-cb-nonlinear.npz",
        "w0wa-{n}mass-s8-fs8.npz",
    ),
    W0WACDM.NonLinearHalofit: (
        "halofit-w0wa-{n}mass-nonlinear.npz",
        "halofit-w0wa-{n}mass-cb-nonlinear.npz",
        "halofit-w0wa-{n}mass-s8-fs8.npz",
    ),
    LCDMCurvature.Linear: (
        "curvature-lcdm-{n}mass-linear.npz",
        "curvature-lcdm-{n}mass-cb-linear.npz",
        "curvature-lcdm-{n}mass-s8-fs8.npz",
    ),
    LCDMCurvature.NonLinear: (
        "curvature-lcdm-{n}mass-nonlinear.npz",
        "curvature-lcdm-{n}mass-cb-nonlinear.npz",
        "curvature-lcdm-{n}mass-s8-fs8.npz",
    ),
    W0WACurvature.Linear: (
        "curvature-w0wa-{n}mass-linear.npz",
        "curvature-w0wa-{n}mass-cb-linear.npz",
        "curvature-w0wa-{n}mass-s8-fs8.npz",
    ),
    W0WACurvature.NonLinear: (
        "curvature-w0wa-{n}mass-nonlinear.npz",
        "curvature-w0wa-{n}mass-cb-nonlinear.npz",
        "curvature-w0wa-{n}mass-s8-fs8.npz",
    ),
    LCDMRunning.Linear: (
        "nrun-lcdm-{n}mass-linear.npz",
        "nrun-lcdm-{n}mass-cb-linear.npz",
        "nrun-lcdm-{n}mass-s8-fs8.npz",
    ),
    LCDMRunning.NonLinear: (
        "nrun-lcdm-{n}mass-nonlinear.npz",
        "nrun-lcdm-{n}mass-cb-nonlinear.npz",
        "nrun-lcdm-{n}mass-s8-fs8.npz",
    ),
    W0WARunning.Linear: (
        "nrun-w0wa-{n}mass-linear.npz",
        "nrun-w0wa-{n}mass-cb-linear.npz",
        "nrun-w0wa-{n}mass-s8-fs8.npz",
    ),
    W0WARunning.NonLinear: (
        "nrun-w0wa-{n}mass-nonlinear.npz",
        "nrun-w0wa-{n}mass-cb-nonlinear.npz",
        "nrun-w0wa-{n}mass-s8-fs8.npz",
    ),
}


def _family(cls):
    return next(f for f in FAMILIES if cls in vars(f).values())


def _cases(n_mnu_filter=None):
    cases = []
    for cls in EXPECTED_FILES:
        family = _family(cls)
        for n in FAMILIES[family][0]:
            if n_mnu_filter is None or n in n_mnu_filter:
                cases.append(
                    pytest.param(cls, n, id=f"{family.__name__}.{cls.__name__}-{n}")
                )
    return cases


class _Background:
    """Just the background attributes the CosmoPower-JAX classes read."""

    def __init__(self, n_mnu, omega_k0):
        self.H0, self.h = H0, H0 / 100.0
        self.Omega_b0, self.Omega_cdm0, self.Omega_k0 = OMEGA_B0, OMEGA_CDM0, omega_k0
        self.As, self.ns = AS, NS
        self.N_mnu, self.mnu = n_mnu, MNU if n_mnu else 0.0
        self.w0, self.wa, self.alpha_s = W0, WA, ALPHA_S


def _amplitude(name):
    """A different, fixed amplitude per file, to tell which file fed which output."""
    return 1.0 + zlib.crc32(name.encode()) % 100_000 / 10_000


class _FakeEmulator:
    """Stands in for a CosmoPower-JAX emulator and records what it is asked."""

    modes = np.logspace(-4.0, 1.0, 40)

    def __init__(self, path, kind, log):
        self.name, self.kind, self.log = os.path.basename(path), kind, log

    def predict(self, inputs):
        self.log[self.name] = (self.kind, {k: np.array(v) for k, v in inputs.items()})
        z = np.asarray(inputs["z"])[:, None]
        if self.kind == "sigma":
            return _amplitude(self.name) * np.hstack([0.8 / (1 + z), 0.45 / (1 + z)])
        shape = 2e4 * self.modes / (1 + (self.modes / 0.02) ** 2) / (1 + z) ** 2
        return _amplitude(self.name) * shape


@pytest.fixture
def fake_emulators(monkeypatch):
    """Replace downloading and loading with fakes; return what was requested."""
    record = {"requested": [], "predicted": {}}

    def emulator_data(filename, base_url=None):
        record["requested"].append(filename)
        return os.path.join("not-downloaded", filename)

    def loader(kind):
        return lambda path: _FakeEmulator(path, kind, record["predicted"])

    monkeypatch.setattr(cpj, "emulator_data", emulator_data)
    monkeypatch.setattr(cpj, "load_pk_emulator", loader("pk"))
    monkeypatch.setattr(cpj, "load_sigma_emulator", loader("sigma"))
    return record


def _build(cls, n_mnu):
    background = _Background(n_mnu, FAMILIES[_family(cls)][1])
    if cls.__name__ == "Linear":
        return cls(background, REDSHIFTS)
    return cls(background, object(), REDSHIFTS, log10TAGN=LOG10TAGN)


def _expected_inputs(cls, n_mnu, kind):
    inputs = {
        "ombh2": OMEGA_B0 * (H0 / 100.0) ** 2,
        "omch2": OMEGA_CDM0 * (H0 / 100.0) ** 2,
        "H0": H0,
        "ns": NS,
        "lnAs": np.log(AS * 1e10),
        **FAMILIES[_family(cls)][2],
    }
    if n_mnu > 0:
        inputs["mnu"] = MNU
    if cls.__name__ == "NonLinear":
        inputs["logT_AGN"] = LOG10TAGN
    elif kind == "sigma":
        inputs["logT_AGN"] = FIDUCIAL_LOG10TAGN
    inputs = {k: np.full(len(Z_GRID), v) for k, v in inputs.items()}
    inputs["z"] = Z_GRID
    return inputs


@pytest.mark.parametrize("cls, n_mnu", _cases())
def test_loads_expected_emulator_files(fake_emulators, cls, n_mnu):
    """Each class loads exactly its three files and uses each for the right output."""
    perturbations = _build(cls, n_mnu)

    pk, cb, sigma = (f.format(n=n_mnu) for f in EXPECTED_FILES[cls])
    # Each file is requested once, by its bare name.
    assert sorted(fake_emulators["requested"]) == sorted([pk, cb, sigma])
    kinds = {name: kind for name, (kind, _) in fake_emulators["predicted"].items()}
    assert kinds == {pk: "pk", cb: "pk", sigma: "sigma"}

    # P(k), P_cb(k) and sigma8 come from the P(k), cb and sigma8 files.
    k = _FakeEmulator.modes[20]
    shape = 2e4 * k / (1 + (k / 0.02) ** 2)
    total = perturbations.matter_power_spectrum(0.0, k).item()
    cb_only = perturbations.matter_power_spectrum_cb(0.0, k).item()
    np.testing.assert_allclose(total, _amplitude(pk) * shape, rtol=1e-6)
    np.testing.assert_allclose(cb_only, _amplitude(cb) * shape, rtol=1e-6)
    np.testing.assert_allclose(perturbations.sigma8_0(), _amplitude(sigma) * 0.8)


@pytest.mark.parametrize("cls, n_mnu", _cases())
def test_passes_expected_inputs(fake_emulators, cls, n_mnu):
    """Each emulator gets exactly the expected inputs, with the expected values."""
    _build(cls, n_mnu)

    for name, (kind, inputs) in fake_emulators["predicted"].items():
        expected = _expected_inputs(cls, n_mnu, kind)
        assert sorted(inputs) == sorted(expected), name
        for key, value in expected.items():
            np.testing.assert_allclose(inputs[key], value, rtol=1e-12, err_msg=name)


_REAL_EMULATOR_DATA = cpj.emulator_data
_REAL_LOADERS = {"pk": cpj.load_pk_emulator, "sigma": cpj.load_sigma_emulator}


@functools.lru_cache(maxsize=None)
def _real_emulator_inputs(name, kind):
    return frozenset(_REAL_LOADERS[kind](_REAL_EMULATOR_DATA(name)).parameters)


@pytest.mark.skipif(not HAS_COSMOPOWER_JAX, reason="cosmopower_jax not installed")
@pytest.mark.parametrize("cls, n_mnu", _cases(n_mnu_filter=(0, 1)))
def test_inputs_match_real_emulator_parameters(fake_emulators, cls, n_mnu):
    """The inputs each class sends match the parameter names stored in the file.

    A CosmoPower-JAX emulator silently ignores inputs it was not trained on, so a
    wrong file, for example LCDM loaded for wCDM, would drop ``w`` without an
    error. N_mnu = 0 and 1 are enough here, because the 1, 2 and 3 neutrino files
    take the same inputs; the file-name test above tells those apart.
    """
    _build(cls, n_mnu)

    for name, (kind, inputs) in fake_emulators["predicted"].items():
        real = _real_emulator_inputs(name, kind)
        assert real - set(inputs) == set(), f"{name}: inputs not provided"
        # The halofit sigma8 emulators are dark-matter only, but the code also
        # sends them the fiducial logT_AGN, which they ignore.
        halofit_sigma = kind == "sigma" and cls.__name__ == "NonLinearHalofit"
        ignorable = {"logT_AGN"} if halofit_sigma else set()
        assert set(inputs) - real <= ignorable, f"{name}: inputs ignored"


def test_emulator_data_downloads_missing_file_from_given_url(monkeypatch):
    """A file that is not cached is fetched once from the given base URL."""
    downloads = []
    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda url, path: downloads.append((url, path))
    )
    name = "not-cached-test-emulator.npz"

    path = cpj.emulator_data(name, base_url="https://example.org/emulators")

    assert os.path.basename(path) == name
    assert not os.path.exists(path)  # the fake download wrote nothing
    assert downloads == [(f"https://example.org/emulators/{name}", path)]


@pytest.mark.parametrize(
    "name, folder",
    [
        ("lcdm-9mass-linear.npz", "lcdm/hmcode"),
        ("w0wa-9mass-s8-fs8.npz", "w0wa/hmcode"),
        ("halofit-wcdm-9mass-nonlinear.npz", "wcdm/halofit"),
        ("curvature-w0wa-9mass-cb-linear.npz", "extended/curvature"),
        ("nrun-lcdm-9mass-nonlinear.npz", "extended/running"),
    ],
)
def test_emulator_data_default_url_follows_repo_layout(monkeypatch, name, folder):
    """Without a base URL, files come from the matching GitHub repo folder."""
    downloads = []
    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda url, path: downloads.append(url)
    )

    cpj.emulator_data(name)

    assert downloads == [f"{GITHUB_RAW}/{folder}/{name}"]


def test_emulator_data_reuses_cached_file(monkeypatch):
    """A file that is already cached is not downloaded again."""
    name = "lcdm-0mass-s8-fs8.npz"
    path = cpj.emulator_data(name)  # downloads it first if needed

    def fail(*args):
        raise AssertionError("cached emulator was downloaded again")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail)

    assert cpj.emulator_data(name) == path


def test_check_emulator_bounds_skips_unknown_parameters():
    """Inputs without a training range, such as z, are not bounds-checked."""
    check_emulator_bounds({"z": np.array([0.0, 10.0]), "not_an_input": 1e9})

    with pytest.raises(ValueError, match="H0 out of emulator range"):
        check_emulator_bounds({"z": 10.0, "H0": 150.0})
