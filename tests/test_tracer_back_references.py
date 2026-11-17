"""
Tracers and their Contributions must not form reference cycles (cloelib#714).

A tracer owns its contributions (`ShearTracer.lensing`, `PositionsTracer.bias`,
...) and each contribution calls back into the tracer through `self._tracer`.
If that back-reference were strong, the pair would be a reference cycle: a
discarded tracer - and with it its perturbations, background, P(k) grids and
CAMB results - would only be freed when Python's cyclic garbage collector
runs, so memory grows with every new cosmology in a chain (cloerun#18).

`TracerBackRefMixin` keeps the back-reference weak. These tests check that a
tracer is freed by reference counting alone (cyclic GC disabled), for every
tracer type, and that the mixin behaves sensibly.
"""

import gc
import weakref
from contextlib import contextmanager

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBNonLinearPerturbations,
)
from cloelib.observables.cmb import CMBLensingContribution, CMBLensingTracer
from cloelib.observables.gw import GWNumberCountsTracer, GWWeakLensingTracer
from cloelib.observables.gw.number_counts import GWSourceBiasContribution
from cloelib.observables.gw.weak_lensing import GWWeakLensingContribution
from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import TracerBackRefMixin
from cloelib.observables.photo.positions import (
    GalaxyBiasContribution,
    MagnificationContribution,
    NonLinearGalaxyBiasContribution,
)
from cloelib.observables.photo.shear import (
    LensingContribution,
    NLAContribution,
    TATTContribution,
    TATTMContribution,
)

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
    """Build a tracer, touch its contributions, drop it: it must be gone at once."""
    with no_cyclic_gc():
        tracer = make_tracer()
        for contribution in tracer.get_contributions():
            assert contribution._tracer is tracer
        ref = weakref.ref(tracer)
        del tracer, contribution
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


def test_shear_tracer_freed_by_refcount(perturbations, z_dndz):
    z, dndz = z_dndz
    nuisance = {
        **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(N_BINS)},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(N_BINS)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(N_BINS)},
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": -0.41,
    }
    assert_freed_by_refcount(
        lambda: ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            nuisance_params=nuisance,
            ia_model="NLA",
        )
    )


def test_positions_tracer_freed_by_refcount(perturbations, z_dndz):
    z, dndz = z_dndz
    nuisance = {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(N_BINS)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(N_BINS)},
        **{f"magnification_bias_{i + 1}": 0.3 for i in range(N_BINS)},
        **{f"b1_photo_bin{i}": 1.0 + 0.2 * i for i in range(N_BINS)},
    }
    assert_freed_by_refcount(
        lambda: PositionsTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params=nuisance,
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


@pytest.mark.parametrize(
    "contribution_cls",
    [
        LensingContribution,
        NLAContribution,
        TATTContribution,
        TATTMContribution,
        GalaxyBiasContribution,
        MagnificationContribution,
        NonLinearGalaxyBiasContribution,
        CMBLensingContribution,
        GWSourceBiasContribution,
        GWWeakLensingContribution,
    ],
)
def test_contributions_hold_weak_tracer_reference(contribution_cls):
    """Every contribution that calls back into its tracer uses the mixin
    (also covers TATT / TATT-M / nonlinear bias, which need loop computers
    to be built as part of a tracer)."""
    assert issubclass(contribution_cls, TracerBackRefMixin)


def test_mixin_weak_reference_semantics():
    class Owner:
        pass

    class Child(TracerBackRefMixin):
        def __init__(self, tracer):
            self._tracer = tracer

    owner = Owner()
    child = Child(owner)
    assert child._tracer is owner
    assert isinstance(child.__dict__["_tracer_ref"], weakref.ref)
    assert "_tracer" not in child.__dict__  # no strong reference stored

    del owner
    with pytest.raises(ReferenceError, match="Child"):
        _ = child._tracer
