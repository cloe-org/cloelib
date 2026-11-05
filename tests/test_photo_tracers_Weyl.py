"""Unit tests for Weyl photometric tracers (initialization and kernels).

These are deliberately small unit tests, not golden-value or CLASS/CAMB
integration tests. The background and growth factor have controlled values.
Run from the repository root:
    python -m pytest tests/test_photo_tracers_Weyl.py -v
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.observables.photo.positions import PositionsTracer
from cloelib.observables.photo.positions_Weyl import (
    PositionsTracer_Weyl_GC,
    PositionsTracer_Weyl_GGL,
    ShearTracer_IAOnly,
)

C_KM_S = SPEED_OF_LIGHT / 1000.0


@pytest.fixture
def tracer_setup():
    """Two-bin tracers with a simple, explicitly controlled growth history."""
    z = np.linspace(0.2, 1.8, 9)
    dndz = np.vstack((np.ones_like(z), 1.0 + z))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

    background = SimpleNamespace(
        H0=70.0,
        hubble_parameter=lambda zz: 70.0 * (1.0 + np.asarray(zz)),
        Omega_m=lambda zz: 0.3 + 0.1 * np.asarray(zz),
        comoving_distance=lambda zz: 3000.0 * np.asarray(zz),
    )

    def growth_factor(zs, ks):
        zs, ks = np.atleast_1d(zs), np.atleast_1d(ks)
        return np.broadcast_to((1.0 / (1.0 + zs))[:, None], (len(zs), len(ks)))

    perturbations = SimpleNamespace(
        background=background,
        z_ini=3.0,
        sigma8_zini=lambda: 0.4,
        growth_factor=growth_factor,
    )
    nuisance = {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(2)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(2)},
        "magnification_bias_1": 0.2,
        "magnification_bias_2": 0.4,
        "bhat_bin0": 1.2,
        "bhat_bin1": 1.5,
    }
    jhat = {"Jhat_bin0": 0.6, "Jhat_bin1": 0.8}
    gc = PositionsTracer_Weyl_GC(perturbations, dndz, z, nuisance)
    ggl = PositionsTracer_Weyl_GGL(perturbations, dndz, z, nuisance, jhat)
    return SimpleNamespace(
        z=z,
        dndz=dndz,
        background=background,
        perturbations=perturbations,
        nuisance=nuisance,
        jhat=jhat,
        gc=gc,
        ggl=ggl,
    )


@pytest.mark.parametrize("name", ["gc", "ggl"])
def test_initialization_and_bias(tracer_setup, name):
    """Both subclasses enforce per-bin bias and read Weyl normalization."""
    s = tracer_setup
    tracer = getattr(s, name)
    assert isinstance(tracer, PositionsTracer)
    assert tracer.flags["galaxy_bias_model"] == "per_bin"
    assert tracer.perturbations is s.perturbations
    assert tracer.z_ini == pytest.approx(3.0)
    assert tracer.sigma8_ini == pytest.approx(0.4)
    assert tracer.n_z_bins == 2
    np.testing.assert_allclose(np.asarray(tracer.bias_array)[:2], [1.2, 1.5])
    assert np.asarray(tracer.dndz_shifted).shape == (2, len(s.z))
    if name == "ggl":
        np.testing.assert_allclose(np.asarray(tracer.Jhat_array)[:2], [0.6, 0.8])


# The next part is to verify that both Weyl tracer classes raise an AttributeError if the supplied perturbations object is missing either z_ini or sigma8_zini().
@pytest.mark.parametrize("cls", [PositionsTracer_Weyl_GC, PositionsTracer_Weyl_GGL])
@pytest.mark.parametrize("missing", ["z_ini", "sigma8_zini"])
def test_required_perturbation_attributes(tracer_setup, cls, missing):
    """Fail clearly when a Weyl-specific perturbation capability is absent."""
    s = tracer_setup
    attrs = vars(s.perturbations).copy()
    attrs.pop(missing)
    perturbations = SimpleNamespace(**attrs)
    kwargs = {"Jhat_params": s.jhat} if cls is PositionsTracer_Weyl_GGL else {}
    with pytest.raises(AttributeError, match=missing):
        cls(perturbations, s.dndz, s.z, s.nuisance, **kwargs)


@pytest.mark.parametrize("name", ["gc", "ggl"])
def test_growth_since_zini(tracer_setup, name):
    """D(z)/D(z_ini) has the correct normalization and redshift dependence."""
    s = tracer_setup
    tracer = getattr(s, name)
    expected = (1.0 + tracer.z_ini) / (1.0 + s.z)
    np.testing.assert_allclose(np.asarray(tracer.growth_since_zini(s.z)), expected)
    np.testing.assert_allclose(
        np.asarray(tracer.growth_since_zini(np.array([tracer.z_ini]))), 1.0
    )


@pytest.mark.parametrize("name", ["gc", "ggl"])
def test_positions_window_formula(tracer_setup, name):
    """Check the distinct bhat/Jhat, Omega_m, and sigma8 powers."""
    s = tracer_setup
    tracer = getattr(s, name)
    H = s.background.hubble_parameter(s.z)
    bhat = np.array([1.2, 1.5])[:, None]
    expected = bhat * np.asarray(tracer.dndz_shifted) * H[None, :] / C_KM_S
    if name == "gc":
        expected /= 0.4
    else:
        expected *= np.array([0.6, 0.8])[:, None]
        expected /= s.background.Omega_m(s.z)[None, :] * 0.4**2
    result = np.asarray(tracer.get_window_positions(s.z))
    assert result.shape == (2, len(s.z))
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_gc_ggl_positions_window_ratio(tracer_setup):
    """Independent relation between the two windows for common dndz/bhat."""
    s = tracer_setup
    gc = np.asarray(s.gc.get_window_positions(s.z))
    ggl = np.asarray(s.ggl.get_window_positions(s.z))
    expected_ratio = np.array([0.6, 0.8])[:, None] / (
        0.4 * s.background.Omega_m(s.z)[None, :]
    )
    np.testing.assert_allclose(ggl / gc, expected_ratio, rtol=1e-6)


@pytest.mark.parametrize("name,power", [("gc", 1), ("ggl", 2)])
def test_rsd_source_growth_power(tracer_setup, monkeypatch, name, power):
    """Verify the source passed to the shared RSD kernel, not its internals."""
    import cloelib.observables.photo.positions_Weyl as weyl_module

    s = tracer_setup
    tracer = getattr(s, name)
    ells = np.array([10.0, 20.0])
    H = s.background.hubble_parameter(s.z)
    f = np.full_like(s.z, 0.75)
    chi = s.background.comoving_distance(s.z)
    captured = {}

    def fake_get_photo_rsd(given_ells, given_chi, source):
        captured["ells"] = np.asarray(given_ells)
        captured["chi"] = np.asarray(given_chi)
        captured["source"] = np.asarray(source)
        return np.zeros((len(given_ells), tracer.n_z_bins, len(given_chi)))

    monkeypatch.setattr(weyl_module, "get_photo_rsd", fake_get_photo_rsd)
    result = tracer.get_window_rsd(ells, H, f, chi)
    growth = (1.0 + tracer.z_ini) / (1.0 + s.z)
    expected_source = (
        H[None, :]
        * f[None, :]
        / C_KM_S
        * np.asarray(tracer.dndz_shifted)
        * growth[None, :] ** power
    )
    np.testing.assert_allclose(captured["source"], expected_source, rtol=1e-6)
    np.testing.assert_array_equal(captured["ells"], ells)
    np.testing.assert_array_equal(captured["chi"], chi)
    assert np.asarray(result).shape == (len(ells), 2, len(s.z))


@pytest.mark.parametrize("name,power", [("gc", 1), ("ggl", 2)])
def test_magnification_window_growth_power(tracer_setup, name, power):
    """Verify the full Weyl magnification prefactor using parent efficiency."""
    s = tracer_setup
    tracer = getattr(s, name)
    efficiency = np.asarray(tracer.get_magnification_efficiency(s.z))
    growth = (1.0 + tracer.z_ini) / (1.0 + s.z)
    factor = (
        1.5
        * growth**power
        * (s.background.H0 / C_KM_S) ** 2
        * s.background.Omega_m(0.0)
        * (1.0 + s.z)
        * s.background.comoving_distance(s.z)
    )
    expected = efficiency * factor[None, :] * np.array([0.2, 0.4])[:, None]
    actual = np.asarray(tracer.get_window_magnification(s.z))
    assert actual.shape == (2, len(s.z))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


def test_ia_only_contributions(monkeypatch):
    """IA-only shear exposes only IA, not the usual lensing contribution."""
    from cloelib.observables.photo.shear import ShearTracer

    ia = object()

    def fake_parent_init(self, *args, **kwargs):
        self.ia = ia

    monkeypatch.setattr(ShearTracer, "__init__", fake_parent_init)
    tracer = ShearTracer_IAOnly()
    assert tracer.get_contributions() == (ia,)


def test_ia_only_rejects_missing_ia(monkeypatch):
    """An IA-only shear tracer cannot be constructed without an IA term."""
    from cloelib.observables.photo.shear import ShearTracer

    def fake_parent_init(self, *args, **kwargs):
        self.ia = None

    monkeypatch.setattr(ShearTracer, "__init__", fake_parent_init)
    with pytest.raises(ValueError, match="needs an IA model"):
        ShearTracer_IAOnly()
