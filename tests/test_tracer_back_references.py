"""
Tracers and their Contributions must not form reference cycles (cloelib#714).

A tracer owns its contributions (`ShearTracer.lensing`, `PositionsTracer.bias`,
...) and each contribution calls back into the tracer through `self._tracer`.
If that back-reference were strong, the pair would be a reference cycle: a
discarded tracer - and with it its perturbations, background, P(k) grids and
CAMB results - would only be freed when Python's cyclic garbage collector
runs, so memory grows with every new cosmology in a chain (cloerun#18).

Each contribution therefore stores `self._tracer = weakref.proxy(tracer)`, a
weak reference used exactly like the tracer. These tests check that every tracer type
(and every IA / galaxy-bias model) is freed by reference counting alone, with
the cyclic GC disabled.
"""

import gc
import weakref
from contextlib import contextmanager

import numpy as np
import pytest
from test_tatt import _StubTATTLoopComputer

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.gw import GWNumberCountsTracer, GWWeakLensingTracer
from cloelib.observables.photo import PositionsTracer, ShearTracer

N_BINS = 2


@contextmanager
def no_cyclic_gc():
    """Only reference counting can free objects inside this block."""
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def assert_freed_by_refcount(make_tracer):
    """Build a tracer with its contributions, drop it: it must be gone at once."""
    with no_cyclic_gc():
        tracer = make_tracer()
        contributions = list(tracer.get_contributions())
        assert contributions
        ref = weakref.ref(tracer)
        del tracer, contributions
        assert ref() is None, (
            "tracer kept alive after its last reference was dropped: "
            "it is part of a reference cycle"
        )


@pytest.fixture(scope="module")
def perturbations():
    H0 = 67.7
    h = H0 / 100.0
    background = CAMBBackground(
        H0=H0,
        Omega_b0=0.022 / h**2,
        Omega_cdm0=0.12 / h**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        alpha_s=0.0,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )
    return CAMBNonLinearPerturbations(background, None, np.linspace(0.01, 1100.0, 100))


@pytest.fixture(scope="module")
def z_dndz():
    z = np.linspace(0.2, 2.0, 15)
    dndz = np.ones((N_BINS, len(z)))
    return z, dndz / np.trapezoid(dndz, z, axis=1)[:, None]


class _FakeBackground:
    H0 = 70.0

    def Omega_m(self, z):
        return 0.3 + 0.0 * z

    def hubble_parameter(self, z, units="km/s/Mpc"):
        return self.H0 * np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)

    def comoving_distance(self, z):
        return 3000.0 * z


class _FakePerturbations:
    """Minimal perturbations for the GW tracers (as in test_gw_observable.py)."""

    def __init__(self, z):
        self.background = _FakeBackground()
        self.z = z
        self.k = np.geomspace(1.0e-3, 1.0, 32)

    def matter_power_spectrum(self, z, k):
        return 1.0 / (1.0 + z[:, None]) ** 2 / (1.0 + k[None, :] ** 2)


_TATT_M_NUISANCE = {
    "alphaM": 1.0,
    "betaM": 0.5,
    "k1": 0.073,
    "k2": 0.268,
    **{f"log10_Mh_{i + 1}": 13.5 for i in range(N_BINS)},
    **{f"f_r_{i + 1}": 1.0 for i in range(N_BINS)},
}


@pytest.mark.parametrize("ia_model", [None, "NLA", "NLA-M", "TATT", "TATT-M"])
def test_shear_tracer_freed_by_refcount(perturbations, z_dndz, ia_model):
    z, dndz = z_dndz
    nuisance = {
        **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(N_BINS)},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(N_BINS)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(N_BINS)},
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": -0.41,
        "A2IA": 0.5,
        "bTA": 0.3,
        **(_TATT_M_NUISANCE if ia_model in ("TATT-M", "NLA-M") else {}),
    }
    loop = {}
    if ia_model in ("TATT", "TATT-M"):
        loop["tatt_loop_computer"] = _StubTATTLoopComputer(perturbations)
    assert_freed_by_refcount(
        lambda: ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            nuisance_params=nuisance,
            ia_model=ia_model,
            **loop,
        )
    )


class _StubNLBias:
    def compute(self, name):
        return lambda matter_pk, ks, zs: 1e-2 * matter_pk[0]


@pytest.mark.parametrize("include_rsd", [False, True])
@pytest.mark.parametrize("bias_model", ["per_bin", "nonlinear"])
def test_positions_tracer_freed_by_refcount(
    perturbations, z_dndz, bias_model, include_rsd
):
    z, dndz = z_dndz
    nuisance = {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(N_BINS)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(N_BINS)},
        **{f"magnification_bias_{i + 1}": 0.3 for i in range(N_BINS)},
        **{f"b1_photo_bin{i}": 1.0 + 0.2 * i for i in range(N_BINS)},
        **{f"b1_photo_nl_bin{i}": 1.0 + 0.2 * i for i in range(N_BINS)},
    }
    loop = {}
    if bias_model == "nonlinear":
        loop["nl_bias_loop_computer"] = _StubNLBias()
    assert_freed_by_refcount(
        lambda: PositionsTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model=bias_model,
            nuisance_params=nuisance,
            include_rsd=include_rsd,
            **loop,
        )
    )


def test_cmb_lensing_tracer_freed_by_refcount(perturbations):
    assert_freed_by_refcount(
        lambda: CMBLensingTracer(
            perturbations=perturbations, z=np.linspace(0.01, 1100.0, 100)
        )
    )


@pytest.mark.parametrize("tracer_cls", [GWNumberCountsTracer, GWWeakLensingTracer])
def test_gw_tracers_freed_by_refcount(tracer_cls):
    z = np.linspace(0.1, 1.1, 21)
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    nuisance = {"b1_gw_bin0": 1.7, "dz_gw_1": 0.0, "width_gw_1": 1.0}
    kwargs = {"gw_bias_model": "per_bin"} if tracer_cls is GWNumberCountsTracer else {}
    assert_freed_by_refcount(
        lambda: tracer_cls(
            perturbations=_FakePerturbations(z),
            dndz=dndz,
            z=z,
            nuisance_params=nuisance,
            **kwargs,
        )
    )
