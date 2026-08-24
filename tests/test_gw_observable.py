"""Unit tests for gravitational-wave observable windows."""

import numpy as np

from cloelib.observables.gw import GWNumberCountsTracer, GWWeakLensingTracer


class FakeBackground:
    """Minimal background implementation needed by the GW tracers."""

    H0 = 70.0

    def Omega_m(self, z):
        return 0.3 + 0.0 * z

    def hubble_parameter(self, z):
        return self.H0 * np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)

    def comoving_distance(self, z):
        return 3000.0 * z


class FakePerturbations:
    """Minimal perturbations implementation needed by the GW tracers."""

    background = FakeBackground()


def _normalised_dndz(z):
    centers = (0.4, 0.8)
    dndz = np.asarray(
        [np.exp(-0.5 * ((np.asarray(z) - center) / 0.15) ** 2) for center in centers]
    )
    return dndz / np.trapezoid(dndz, np.asarray(z), axis=1)[:, None]


def _gw_nuisance():
    return {
        "b1_GW_bin0": 1.0,
        "b1_GW_bin1": 1.5,
        "dz_gw_1": 0.0,
        "dz_gw_2": 0.0,
        "width_gw_1": 1.0,
        "width_gw_2": 1.0,
    }


def test_gw_number_counts_window_is_finite_and_non_negative():
    z = np.linspace(0.1, 1.1, 21)
    dndz = _normalised_dndz(z)
    tracer = GWNumberCountsTracer(
        perturbations=FakePerturbations(),
        dndz=dndz,
        z=z,
        gw_bias_model="per_bin",
        nuisance_params=_gw_nuisance(),
    )

    window = np.asarray(tracer.get_window(z))
    tolerance = 1.0e-12

    assert window.shape == dndz.shape
    assert np.all(np.isfinite(window))
    assert np.all(window >= -tolerance)
    assert np.any(window > tolerance)


def test_gw_weak_lensing_window_without_ia_is_finite_and_non_negative():
    z = np.linspace(0.1, 1.1, 21)
    dndz = _normalised_dndz(z)
    nuisance_params = {
        "dz_gw_1": 0.0,
        "dz_gw_2": 0.0,
        "width_gw_1": 1.0,
        "width_gw_2": 1.0,
    }
    tracer = GWWeakLensingTracer(
        perturbations=FakePerturbations(),
        dndz=dndz,
        z=z,
        nuisance_params=nuisance_params,
    )

    window = np.asarray(tracer.get_window(z))
    tolerance = 1.0e-12

    assert window.shape == dndz.shape
    assert np.all(np.isfinite(window))
    assert np.all(window >= -tolerance)
    assert np.any(window > tolerance)
