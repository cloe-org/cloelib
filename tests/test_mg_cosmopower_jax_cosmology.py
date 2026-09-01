"""Unit tests for the consolidated modified-gravity boost module
``cloelib.cosmology.mg_cosmopower_jax_cosmology``.

The MG boost emulators (``mg-boost-*.npz``) and the LCDM baseline are *mocked*,
so these tests exercise the boost/Sigma logic and the public API without needing
cosmopower-jax or any emulator data files. ``_load_emu`` is monkeypatched to
return a fake emulator; a fake LCDM baseline supplies a known P(k, z).
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
    can check the multiplicative application exactly.
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

    def _fake(branch, model_dir, bin_index=None):
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


class _DummyBackground:
    Omega_cdm0 = 0.26
    Omega_b0 = 0.048
    Omega_k0 = 0.0
    H0 = 67.0
    ns = 0.96
    As = 2.1e-9


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
        model_dir="/does/not/exist",
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
        mg.MGParams(mu=np.ones(mg.N_BINS), eta=np.ones(mg.N_BINS)),
    ],  # multi-bin
)
def test_gr_limit_linear(monkeypatch, zs, mg_params):
    Lin, _ = _build(monkeypatch, mg_params, boost=1.0)
    lp = Lin(_DummyBackground(), zs)
    base = _FakeBaselineLinear(_DummyBackground(), zs)
    got = lp.matter_power_spectrum(zs, _KGRID)
    ref = base.matter_power_spectrum(zs, _KGRID)
    assert np.allclose(got, ref, rtol=1e-10)


def test_boost_applied_linear(monkeypatch, zs):
    Lin, _ = _build(monkeypatch, mg.MGParams(bin_index=0), boost=1.5)
    lp = Lin(_DummyBackground(), zs)
    ref = _FakeBaselineLinear(_DummyBackground(), zs).matter_power_spectrum(zs, _KGRID)
    got = lp.matter_power_spectrum(zs, _KGRID)
    assert np.allclose(got, 1.5 * ref, rtol=1e-8)


def test_gr_limit_nonlinear(monkeypatch, zs):
    _, NonLin = _build(monkeypatch, mg.MGParams(bin_index=3), boost=1.0)
    nlp = NonLin(_DummyBackground(), None, zs)
    ref = _FakeBaselineNonLinear(_DummyBackground(), None, zs).matter_power_spectrum(
        zs, _KGRID
    )
    got = nlp.matter_power_spectrum(zs, _KGRID)
    assert np.allclose(got, ref, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Sigma(z)
# --------------------------------------------------------------------------- #
def test_sigma_single_bin(monkeypatch):
    # mu=1.2, eta=0.5 -> Sigma = 1.2*(1+0.5)/2 = 0.9 inside bin 2, else 1
    _, NonLin = _build(
        monkeypatch, mg.MGParams(mu=1.2, eta=0.5, bin_index=2), boost=1.0
    )
    nlp = NonLin(_DummyBackground(), None, _ZBUILD)
    zmin, zmax = mg._BIN_EDGES[2]
    z_in = 0.5 * (zmin + zmax)
    z_out = mg._BIN_EDGES[0][1] * 0.5  # inside bin 0
    assert np.isclose(nlp.Sigma(z_in)[0], 0.9)
    assert np.isclose(nlp.Sigma(z_out)[0], 1.0)


def test_sigma_multibin(monkeypatch):
    mu = np.array([1.1, 1.0, 0.9, 1.0, 1.2])
    eta = np.array([0.0, 0.0, 0.4, 0.0, 0.0])
    _, NonLin = _build(monkeypatch, mg.MGParams(mu=mu, eta=eta), boost=1.0)
    nlp = NonLin(_DummyBackground(), None, _ZBUILD)
    for i, (zmin, zmax) in enumerate(mg._BIN_EDGES):
        z_mid = 0.5 * (zmin + zmax)
        expected = mu[i] * (1.0 + eta[i]) / 2.0
        assert np.isclose(nlp.Sigma(z_mid)[0], expected), f"bin {i}"


def test_sigma_gr_gives_unity(monkeypatch):
    _, NonLin = _build(monkeypatch, mg.MGParams(), boost=1.0)  # all-GR multibin
    nlp = NonLin(_DummyBackground(), None, _ZBUILD)
    assert np.allclose(nlp.Sigma(np.linspace(0.1, 2.9, 20)), 1.0)


# --------------------------------------------------------------------------- #
# API / MGParams
# --------------------------------------------------------------------------- #
def test_mgparams_modes():
    single = mg.MGParams(mu=1.2, eta=0.5, bin_index=4)
    assert single.bin_index == 4 and single.mu == 1.2 and single.eta == 0.5

    multi = mg.MGParams()  # defaults
    assert multi.bin_index is None
    assert multi.mu.shape == (mg.N_BINS,) and np.allclose(multi.mu, 1.0)
    assert np.allclose(multi.eta, 1.0)


def test_factory_aliases_are_the_same():
    assert mg.binned_mg_perturbations is mg.mg_perturbations
    assert mg.multibin_mg_perturbations is mg.mg_perturbations


def test_derived_quantities_run(monkeypatch, zs):
    _, NonLin = _build(
        monkeypatch, mg.MGParams(mu=1.1, eta=0.2, bin_index=1), boost=1.3
    )
    nlp = NonLin(_DummyBackground(), None, zs)
    s8 = nlp.sigma8_0()
    assert np.isfinite(s8) and s8 > 0
    f = nlp.growth_rate()
    assert f.shape == zs.shape and np.all(np.isfinite(f))
    D = nlp.growth_factor(zs, _KGRID)
    assert D.shape == (zs.size, _KGRID.size) and np.all(np.isfinite(D))


def test_missing_emulator_raises(tmp_path):
    # real _load_emu (not mocked) should raise a clear error on a missing file
    with pytest.raises(FileNotFoundError):
        mg._load_emu("linear", str(tmp_path), bin_index=0)
