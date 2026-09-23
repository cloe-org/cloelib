"""Unit tests for gravitational-wave observables."""

import numpy as np

from cloelib.observables.gw import GWNumberCountsTracer, GWWeakLensingTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint


class FakeBackground:
    """Minimal background implementation needed by the GW tracers."""

    H0 = 70.0

    def Omega_m(self, z):
        return 0.3 + 0.0 * z

    def hubble_parameter(self, z, units="km/s/Mpc"):
        return self.H0 * np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)

    def comoving_distance(self, z):
        return 3000.0 * z


class FakePerturbations:
    """Minimal perturbation implementation needed by ``AngularTwoPoint``."""

    def __init__(self, z):
        self.background = FakeBackground()
        self.z = z
        self.k = np.geomspace(1.0e-3, 1.0, 32)

    def matter_power_spectrum(self, z, k):
        return 1.0 / (1.0 + z[:, None]) ** 2 / (1.0 + k[None, :] ** 2)


def test_gw_windows_and_angular_power_spectra():
    z = np.linspace(0.1, 1.1, 21)
    dndz = np.ones((1, len(z)))
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    perturbations = FakePerturbations(z)
    nuisance_gw = {
        "b1_GW_bin0": 1.0,
        "dz_gw_1": 0.0,
        "width_gw_1": 1.0,
    }

    gw_number_counts = GWNumberCountsTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        gw_bias_model="per_bin",
        nuisance_params=nuisance_gw,
    )
    gw_weak_lensing = GWWeakLensingTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=z,
        nuisance_params=nuisance_gw,
    )

    window_number_counts = np.asarray(gw_number_counts.get_window(z))
    window_weak_lensing = np.asarray(gw_weak_lensing.get_window(z))
    tolerance = 1.0e-12

    assert window_number_counts.shape == dndz.shape
    assert window_weak_lensing.shape == dndz.shape
    assert np.all(np.isfinite(window_number_counts))
    assert np.all(np.isfinite(window_weak_lensing))
    assert np.all(window_number_counts >= -tolerance)
    assert np.all(window_weak_lensing >= -tolerance)
    assert np.any(window_number_counts > tolerance)
    assert np.any(window_weak_lensing > tolerance)

    nl = 10
    ells = np.logspace(1.0, np.log10(100), nl)

    cells = {
        **AngularTwoPoint(gw_number_counts, gw_number_counts).get_Cl(
            ells, 0, perturbations.k
        ),
        **AngularTwoPoint(gw_weak_lensing, gw_weak_lensing).get_Cl(
            ells, 0, perturbations.k
        ),
        **AngularTwoPoint(gw_number_counts, gw_weak_lensing).get_Cl(
            ells, 0, perturbations.k
        ),
    }

    # Check that cells has the right number of elements and right keys
    assert len(cells.keys()) == 3
    assert ("GWNC", "GWNC", 1, 1) in cells.keys()
    assert ("GWWL", "GWWL", 1, 1) in cells.keys()
    assert ("GWNC", "GWWL", 1, 1) in cells.keys()
    assert cells[("GWNC", "GWNC", 1, 1)].shape == (nl,)
    assert cells[("GWWL", "GWWL", 1, 1)].shape == (nl,)
    assert cells[("GWNC", "GWWL", 1, 1)].shape == (nl,)

    # We also check everything works well if we reverse the order of the probes
    cells_reverse = AngularTwoPoint(gw_weak_lensing, gw_number_counts).get_Cl(
        ells, 0, perturbations.k
    )

    assert len(cells_reverse.keys()) == 1
    assert ("GWNC", "GWWL", 1, 1) in cells_reverse.keys()
    assert cells_reverse[("GWNC", "GWWL", 1, 1)].shape == (nl,)
