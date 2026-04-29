import importlib
import sys
from types import ModuleType

import numpy as np


class DummyBackground:
    h = 0.7
    Omega_cdm0 = 0.25
    Omega_b0 = 0.05
    As = 2.1e-9
    ns = 0.965
    mnu = 0.0
    w0 = -0.9
    wa = 0.1


class DummyLinearPerturbations:
    background = DummyBackground()
    z = np.array([0.0, 0.5, 1.0])

    def matter_power_spectrum(self, zs, ks, hubble_units=False, k_hunit=False):
        k = np.atleast_1d(ks)
        return np.ones_like(k) * 3.0

    def growth_rate(self):
        return np.array([0.5, 0.7, 0.9])

    def growth_factor(self, zs, ks):
        z = np.atleast_1d(zs)
        return np.ones_like(z, dtype=float) * 0.8


class DummyGrowthPerturbations:
    gravity_model = "ide"
    mgpars = {"xi": 20.0}
    z = np.array([0.0, 0.5, 1.0])

    @staticmethod
    def fz_interp(zs, ks):
        z = np.atleast_1d(zs)
        k = np.atleast_1d(ks)
        return np.ones((len(z), len(k))) * 0.75

    @staticmethod
    def dz_norm_lcdm_interp(zs, ks):
        z = np.atleast_1d(zs)
        k = np.atleast_1d(ks)
        return np.ones((len(z), len(k))) * 1.1


def install_fake_pbj(monkeypatch):
    class FakeTheory:
        def __init__(self):
            self.kL = np.array([0.01, 0.02, 0.05])
            self.last_call = None

        def _Pgg_kmu_terms(self, plinear, cosmo, units="1/Mpc"):
            self.last_call = {"plinear": plinear, "cosmo": cosmo, "units": units}

        def P_kmu_2D(self, redshift, flag, **kwargs):
            self.last_call.update(
                {"redshift": redshift, "flag": flag, "kwargs": kwargs}
            )
            kgrid = np.asarray(kwargs["kgrid"])
            mu = np.asarray(kwargs["mu"])
            return np.ones((len(kgrid), len(mu)))

        def P_kmu_2D_marg_dict(self, redshift, flag, **kwargs):
            kgrid = np.asarray(kwargs["kgrid"])
            mu = np.asarray(kwargs["mu"])
            arr = np.ones((len(kgrid), len(mu)))
            return {"c0": arr, "c2": 2.0 * arr}

    pbj_module = ModuleType("pbjcosmo")
    theory_module = ModuleType("pbjcosmo.theory")
    theory_module.Theory = FakeTheory
    pbj_module.theory = theory_module
    monkeypatch.setitem(sys.modules, "pbjcosmo", pbj_module)
    monkeypatch.setitem(sys.modules, "pbjcosmo.theory", theory_module)


def test_pbj_beyond_lcdm_growth_inputs(monkeypatch):
    install_fake_pbj(monkeypatch)
    module = importlib.import_module("cloelib.observables.PBJ_spectro_beyond_LCDM")
    module = importlib.reload(module)

    spectro = module.PBJSpectroPower(
        linear_perturbations=DummyLinearPerturbations(),
        nuisance_parameters={"b1": 1.5, "c0": 0.0},
        growth_perturbations=DummyGrowthPerturbations(),
        redshift=0.5,
    )

    k = np.array([0.05, 0.1])
    mu = np.array([0.0, 0.5, 1.0])
    pkmu = spectro.Pk2d_rsd(k, mu)

    assert pkmu.shape == (2, 3)
    assert spectro.growth_model == "darkscattering"
    assert spectro.cosmo["xi"] == 20.0
    assert module.pbj_obj.last_call["kwargs"]["growth_model"] == "darkscattering"


def test_pbj_beyond_lcdm_term_dictionary(monkeypatch):
    install_fake_pbj(monkeypatch)
    module = importlib.import_module("cloelib.observables.PBJ_spectro_beyond_LCDM")
    module = importlib.reload(module)

    spectro = module.PBJSpectroPower(
        linear_perturbations=DummyLinearPerturbations(),
        nuisance_parameters={"b1": 1.5, "c0": 0.0},
        growth_perturbations=DummyGrowthPerturbations(),
        redshift=0.5,
    )

    terms = spectro.Pk2d_term_rsd(np.array([0.05]), np.array([0.0, 1.0]), ["c0", "c2"])
    assert terms.shape == (2, 1, 2)
