"""Unit tests for the consolidated modified-gravity boost module
``cloelib.cosmology.mg_cosmopower_jax_cosmology``.

The MG boost emulators (``mg-boost-*.npz``) and the LCDM baseline are *mocked*,
so these tests exercise the boost / Sigma / bounds / clamp logic and the public
API without needing cosmopower-jax, a network connection, or any emulator data
files. ``_load_emu`` is monkeypatched to return a fake emulator; a fake LCDM
baseline supplies a known P(k, z).
"""

import numpy as np
import pytest

from cloelib.cosmology import mg_cosmopower_jax_cosmology as mg


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #
class _FakeEmu:
    """Stand-in for a CosmoPowerJAX boost emulator.

    ``predict`` returns a constant boost, so the resulting spline is flat and we
    can check the multiplicative application (and the redshift clamp) exactly.
    """

    def __init__(self, parameters, modes, boost):
        self.parameters = parameters
        self.modes = np.asarray(modes, dtype=float)
        self._boost = float(boost)

    def predict(self, X):  # X: (nz, n_params)
        nz = np.asarray(X).shape[0]
        return np.full((nz, self.modes.size), self._boost, dtype=float)


def _make_fake_load_emu(boost):
    """Return a drop-in for ``mg._load_emu`` that builds the right parameter list
    for single-bin (``bin_index`` int) vs multi-bin (``bin_index`` None), and for
    the linear (has eta) vs nonlinear (no eta) sector."""
    base = ["Omega_m", "Omega_b", "h", "ns", "lnAs"]
    modes = np.logspace(-3, 1, 40)  # h/Mpc

    def _fake(branch, bin_index=None):
        if bin_index is None:  # multi-bin
            params = base + [f"mu{i}" for i in range(1, mg.N_BINS + 1)]
            if branch == "linear":
                params += [f"eta{i}" for i in range(1, mg.N_BINS + 1)]
        else:  # single-bin
            params = base + ["mu"]
            if branch == "linear":
                params += ["eta"]
        params += ["z"]
        return _FakeEmu(params, modes, boost)

    return _fake


def _bg(**overrides):
    """Fake background; keyword overrides let a test push a parameter out of box.

    Defaults sit inside *both* the single-bin and multi-bin training boxes
    (Omega_m=0.308, Omega_b=0.048, h=0.67, ns=0.96, lnAs=ln(2.1e-9*1e10)=3.045).
    """

    class _Background:
        Omega_cdm0 = 0.26
        Omega_b0 = 0.048
        Omega_k0 = 0.0
        H0 = 67.0
        ns = 0.96
        As = 2.1e-9

    for key, value in overrides.items():
        setattr(_Background, key, value)
    return _Background()


_KGRID = np.logspace(-3, 1, 60)  # baseline (wide) grid, h/Mpc
_ZBUILD = np.linspace(0.05, 2.9, 10)  # construction grid (>=2 pts for the splines)


class _FakeBaselineLinear:
    """P_LCDM,lin(k, z) = (1+z)^-2 * k^-2 (separable, positive, monotone)."""

    def __init__(self, background, redshifts):
        self.background = background
        self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
        self.k = _KGRID

    def matter_power_spectrum(self, zs, ks):
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        return (1.0 / (1.0 + zs))[:, None] ** 2 * (ks**-2.0)[None, :]

    def sigma8_0(self):
        return 0.8


class _FakeBaselineNonLinear:
    def __init__(self, background, linperturbations, redshifts, log10TAGN=None):
        self.background = background
        self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
        self.k = _KGRID

    def matter_power_spectrum(self, zs, ks):
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        # a mild "nonlinear" bump so it differs from the linear baseline
        return (
            (1.0 / (1.0 + zs))[:, None] ** 2 * (ks**-2.0)[None, :] * (1.0 + ks)[None, :]
        )


@pytest.fixture
def zs():
    return np.linspace(0.05, 2.5, 12)


def _build(monkeypatch, mg_params, boost=1.0):
    monkeypatch.setattr(mg, "_load_emu", _make_fake_load_emu(boost))
    return mg.mg_perturbations(
        mg_params,
        baseline_linear=_FakeBaselineLinear,
        baseline_nonlinear=_FakeBaselineNonLinear,
    )


# --------------------------------------------------------------------------- #
# GR limit and boost application
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mg_params",
    [
        mg.MGParams(mu=1.0, eta=1.0, bin_index=2),  # single-bin
        mg.MGParams(mu=np.ones(mg.N_BINS), eta=np.ones(mg.N_BINS)),  # multi-bin
    ],
)
def test_gr_limit_linear(monkeypatch, zs, mg_params):
    Lin, _ = _build(monkeypatch, mg_params, boost=1.0)
    lp = Lin(_bg(), zs)
    base = _FakeBaselineLinear(_bg(), zs)
    got = lp.matter_power_spectrum(zs, _KGRID)
    ref = base.matter_power_spectrum(zs, _KGRID)
    assert np.allclose(got, ref, rtol=1e-10)


def test_boost_applied_linear_multibin(monkeypatch, zs):
    # Multi-bin: the modification is active up to the top of the last bin
    # (z_top = 3.0), so the constant boost applies across the whole zs grid.
    Lin, _ = _build(monkeypatch, mg.MGParams(), boost=1.5)
    lp = Lin(_bg(), zs)
    ref = _FakeBaselineLinear(_bg(), zs).matter_power_spectrum(zs, _KGRID)
    got = lp.matter_power_spectrum(zs, _KGRID)
    assert np.allclose(got, 1.5 * ref, rtol=1e-8)


def test_gr_limit_nonlinear(monkeypatch, zs):
    _, NonLin = _build(monkeypatch, mg.MGParams(bin_index=3), boost=1.0)
    nlp = NonLin(_bg(), None, zs)
    ref = _FakeBaselineNonLinear(_bg(), None, zs).matter_power_spectrum(zs, _KGRID)
    got = nlp.matter_power_spectrum(zs, _KGRID)
    assert np.allclose(got, ref, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Redshift clamp (B = 1 above the active bin's upper edge)
# --------------------------------------------------------------------------- #
def test_redshift_clamp_single_bin(monkeypatch):
    # Bin 0 active (z_top = 0.43): the boost applies inside the bin and is forced
    # to 1 above it, even though the (fake) emulator returns a constant 2.0.
    Lin, _ = _build(monkeypatch, mg.MGParams(mu=1.05, eta=1.0, bin_index=0), boost=2.0)
    lp = Lin(_bg(), _ZBUILD)
    base = _FakeBaselineLinear(_bg(), _ZBUILD)
    kk = np.array([0.1])
    r_in = lp.matter_power_spectrum(0.2, kk) / base.matter_power_spectrum(
        0.2, kk
    )  # inside bin 0
    r_above = lp.matter_power_spectrum(1.0, kk) / base.matter_power_spectrum(
        1.0, kk
    )  # above bin 0
    assert np.isclose(r_in[0, 0], 2.0, rtol=1e-6)
    assert np.isclose(r_above[0, 0], 1.0, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Training-box enforcement
# --------------------------------------------------------------------------- #
def test_cosmology_out_of_single_bin_box_raises(monkeypatch):
    # ns=0.90 is outside the single-bin box [0.95, 1.00].
    Lin, _ = _build(monkeypatch, mg.MGParams(mu=1.0, eta=1.0, bin_index=2), boost=1.0)
    with pytest.raises(ValueError, match="ns.*single-bin"):
        Lin(_bg(ns=0.90), _ZBUILD)


def test_cosmology_in_multi_bin_box_ok(monkeypatch):
    # The same ns=0.90 is inside the wider multi-bin box [0.80, 1.20].
    Lin, _ = _build(monkeypatch, mg.MGParams(), boost=1.0)
    Lin(_bg(ns=0.90), _ZBUILD)  # must not raise


def test_mu_out_of_box_raises(monkeypatch):
    # mu=1.2 is outside [0.9, 1.1] for both variants.
    Lin, _ = _build(monkeypatch, mg.MGParams(mu=1.2, eta=1.0, bin_index=1), boost=1.0)
    with pytest.raises(ValueError, match="mu"):
        Lin(_bg(), _ZBUILD)


def test_multibin_per_component_eta_bound(monkeypatch):
    # One out-of-range eta component must trip the check.
    mu = np.full(mg.N_BINS, 1.05)
    eta = np.array([1.0, 1.0, 1.3, 1.0, 1.0])  # eta[2] out of [0.9, 1.1]
    Lin, _ = _build(monkeypatch, mg.MGParams(mu=mu, eta=eta), boost=1.0)
    with pytest.raises(ValueError, match="eta.*multi-bin"):
        Lin(_bg(), _ZBUILD)


# --------------------------------------------------------------------------- #
# Sigma(z)  (uses in-box mu/eta)
# --------------------------------------------------------------------------- #
def test_sigma_single_bin(monkeypatch):
    # mu=1.1, eta=0.9 -> Sigma = 1.1*(1+0.9)/2 = 1.045 inside bin 2, else 1.
    _, NonLin = _build(
        monkeypatch, mg.MGParams(mu=1.1, eta=0.9, bin_index=2), boost=1.0
    )
    nlp = NonLin(_bg(), None, _ZBUILD)
    zmin, zmax = mg._BIN_EDGES[2]
    z_in = 0.5 * (zmin + zmax)
    z_out = mg._BIN_EDGES[0][1] * 0.5  # inside bin 0
    assert np.isclose(nlp.Sigma(z_in)[0], 1.045)
    assert np.isclose(nlp.Sigma(z_out)[0], 1.0)


def test_sigma_multibin(monkeypatch):
    mu = np.array([1.1, 1.0, 0.9, 1.05, 0.95])
    eta = np.array([0.9, 1.0, 1.1, 0.95, 1.05])
    _, NonLin = _build(monkeypatch, mg.MGParams(mu=mu, eta=eta), boost=1.0)
    nlp = NonLin(_bg(), None, _ZBUILD)
    for i, (zmin, zmax) in enumerate(mg._BIN_EDGES):
        z_mid = 0.5 * (zmin + zmax)
        expected = mu[i] * (1.0 + eta[i]) / 2.0
        assert np.isclose(nlp.Sigma(z_mid)[0], expected), f"bin {i}"


def test_sigma_gr_gives_unity(monkeypatch):
    _, NonLin = _build(monkeypatch, mg.MGParams(), boost=1.0)  # all-GR multibin
    nlp = NonLin(_bg(), None, _ZBUILD)
    assert np.allclose(nlp.Sigma(np.linspace(0.1, 2.9, 20)), 1.0)


# --------------------------------------------------------------------------- #
# API / MGParams / Zenodo wiring
# --------------------------------------------------------------------------- #
def test_mgparams_modes():
    single = mg.MGParams(mu=1.1, eta=0.95, bin_index=4)
    assert single.bin_index == 4 and single.mu == 1.1 and single.eta == 0.95

    multi = mg.MGParams()  # defaults
    assert multi.bin_index is None
    assert multi.mu.shape == (mg.N_BINS,) and np.allclose(multi.mu, 1.0)
    assert np.allclose(multi.eta, 1.0)


def test_factory_aliases_are_the_same():
    assert mg.binned_mg_perturbations is mg.mg_perturbations
    assert mg.multibin_mg_perturbations is mg.mg_perturbations


@pytest.mark.parametrize(
    "branch, bin_index, expected",
    [
        ("linear", 0, "mg-boost-linear-bin0.npz"),
        ("nonlinear", 4, "mg-boost-nonlinear-bin4.npz"),
        ("linear", None, "mg-boost-linear-multibin.npz"),
        ("nonlinear", None, "mg-boost-nonlinear-multibin.npz"),
    ],
)
def test_emu_filename(branch, bin_index, expected):
    assert mg._emu_filename(branch, bin_index) == expected


def test_zenodo_url_is_extended_record():
    # The MG boost emulators live in the extended-cosmologies Zenodo record.
    assert "22967046" in mg.MG_EMULATOR_ZENODO_URL


def test_bounds_boxes_present():
    for variant in ("single", "multi"):
        box = mg.MG_EMULATOR_BOUNDS[variant]
        for key in ("Omega_m", "Omega_b", "h", "ns", "lnAs", "mu", "eta", "z"):
            low, high = box[key]
            assert low < high


def test_derived_quantities_run(monkeypatch, zs):
    _, NonLin = _build(
        monkeypatch, mg.MGParams(mu=1.1, eta=0.95, bin_index=1), boost=1.3
    )
    nlp = NonLin(_bg(), None, zs)
    s8 = nlp.sigma8_0()
    assert np.isfinite(s8) and s8 > 0
    f = nlp.growth_rate()
    assert f.shape == zs.shape and np.all(np.isfinite(f))
    D = nlp.growth_factor(zs, _KGRID)
    assert D.shape == (zs.size, _KGRID.size) and np.all(np.isfinite(D))
