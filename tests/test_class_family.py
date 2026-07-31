import inspect

import numpy as np

from cloelib.cosmology._class_family import _ClassFamilyBackground
from cloelib.cosmology.class_cosmology import (
    CLASSLinearPerturbations,
    CLASSNonLinearPerturbations,
)
from cloelib.cosmology.hi_class_cosmology import (
    hi_classLinearPerturbations,
    hi_classNonLinearPerturbations,
)


class _DistanceOnlyBackground(_ClassFamilyBackground):
    def __init__(self, H0, Omega_k0, chi):
        self.H0 = H0
        self.Omega_k0 = Omega_k0
        self._chi = np.asarray(chi)

    def comoving_distance(self, zs):
        return self._chi


def test_transverse_comoving_distance_is_flat_in_the_flat_limit():
    background = _DistanceOnlyBackground(70.0, 0.0, [1000.0, 3300.0])
    np.testing.assert_allclose(
        background.transverse_comoving_distance(np.array([0.5, 1.0])), [1000.0, 3300.0]
    )


def test_transverse_comoving_distance_uses_hubble_distance_for_open_geometry():
    background = _DistanceOnlyBackground(70.0, 0.01, [3300.0])
    hubble_distance = background.c0 / background.H0
    expected = hubble_distance / 0.1 * np.sinh(0.1 * 3300.0 / hubble_distance)
    np.testing.assert_allclose(
        background.transverse_comoving_distance(np.array([1.0])), expected
    )


def test_transverse_comoving_distance_uses_hubble_distance_for_closed_geometry():
    background = _DistanceOnlyBackground(70.0, -0.01, [3300.0])
    hubble_distance = background.c0 / background.H0
    expected = hubble_distance / 0.1 * np.sin(0.1 * 3300.0 / hubble_distance)
    np.testing.assert_allclose(
        background.transverse_comoving_distance(np.array([1.0])), expected
    )


def test_class_family_growth_rate_accepts_protocol_arguments():
    for perturbations_class in (
        CLASSLinearPerturbations,
        CLASSNonLinearPerturbations,
        hi_classLinearPerturbations,
        hi_classNonLinearPerturbations,
    ):
        parameters = inspect.signature(perturbations_class.growth_rate).parameters
        assert "zs" in parameters
        assert "ks" in parameters
