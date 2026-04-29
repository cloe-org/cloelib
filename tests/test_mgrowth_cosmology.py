import importlib
import sys
from types import ModuleType

import numpy as np


class DummyBackground:
    Omega_k0 = 0.0
    h = 0.7
    H0 = 70.0
    Omega_cdm0 = 0.25
    Omega_b0 = 0.05
    mnu = 0.0
    w0 = -0.9
    wa = 0.1

    def Omega_m(self, zs):
        return np.full_like(np.atleast_1d(zs), 0.3, dtype=float)


class DummyBaseLinearPerturbations:
    background = DummyBackground()
    z = np.array([0.0, 0.5, 1.0])
    k = np.array([0.05, 0.1, 0.2])

    def matter_power_spectrum(self, zs, ks):
        z = np.atleast_1d(zs)
        k = np.atleast_1d(ks)
        return np.ones((len(z), len(k))) * 2.0

    def sigma8_0(self):
        return 0.8


def install_fake_mgrowth(monkeypatch):
    class FakeBaseModel:
        def __init__(self, background):
            self.background = background

        def growth_parameters(self, **kwargs):
            a_arr = np.asarray(self.background["a_arr"])
            d = a_arr.copy()
            f = np.full_like(a_arr, 0.6)
            if "xi" in kwargs:
                d = d * (1.0 + 0.01 * kwargs["xi"])
            if "gamma" in kwargs:
                f = f * (1.0 + 0.1 * kwargs["gamma"])
            if "gamma0" in kwargs:
                f = f * (1.0 + 0.1 * kwargs["gamma0"])
            if "mu_interp" in kwargs:
                d = d * 1.1
            return d, f

    class FakeFRModel(FakeBaseModel):
        def growth_parameters(self, **kwargs):
            a_arr = np.asarray(self.background["a_arr"])
            k_arr = np.asarray(kwargs["k_arr"])
            d = np.outer(1.0 + 0.01 * k_arr, a_arr)
            f = np.outer(0.5 + 0.0 * k_arr, np.ones_like(a_arr))
            return d, f

    fake_module = ModuleType("MGrowth")
    fake_module.w0waCDM = FakeBaseModel
    fake_module.IDE = FakeBaseModel
    fake_module.fR_HS = FakeFRModel
    fake_module.nDGP = FakeBaseModel
    fake_module.Linder_gamma = FakeBaseModel
    fake_module.Linder_gamma_a = FakeBaseModel
    fake_module.mu_a = FakeBaseModel
    monkeypatch.setitem(sys.modules, "MGrowth", fake_module)
    return fake_module


def test_mgrowth_linear_perturbations_ide(monkeypatch):
    install_fake_mgrowth(monkeypatch)
    module = importlib.import_module("cloelib.cosmology.mgrowth_cosmology")
    module = importlib.reload(module)

    pert = module.MGrowthLinearPerturbations(
        DummyBackground(),
        DummyBaseLinearPerturbations(),
        "ide",
        {"xi": 20.0},
    )

    growth = pert.growth_factor(np.array([0.0, 0.5]), np.array([0.05, 0.1]))
    rate = pert.growth_rate(np.array([0.0, 0.5]), np.array([0.05, 0.1]))
    power = pert.matter_power_spectrum(np.array([0.0, 0.5]), np.array([0.05, 0.1]))

    assert growth.shape == (2, 2)
    assert rate.shape == (2, 2)
    assert power.shape == (2, 2)
    assert pert.sigma8_0() > 0.0


def test_mgrowth_linear_perturbations_fr(monkeypatch):
    install_fake_mgrowth(monkeypatch)
    module = importlib.import_module("cloelib.cosmology.mgrowth_cosmology")
    module = importlib.reload(module)

    pert = module.MGrowthLinearPerturbations(
        DummyBackground(),
        DummyBaseLinearPerturbations(),
        "fr",
        {"fr0": 1e-5},
    )

    growth = pert.growth_factor(np.array([0.0, 1.0]), np.array([0.05, 0.1]))
    assert growth.shape == (2, 2)
