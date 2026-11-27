import numpy as np
import pytest

from cloelib.cosmology.TabulatedBoost_cosmology import (
    TabulatedBoostedPerturbations,
    TabulatedNonlinearBoost,
)


class DummyBackground:
    Omega_k0 = 0.0
    h = 0.7


class DummyLinearPerturbations:
    background = DummyBackground()
    k = np.array([0.07, 0.14, 0.21])


class DummyBasePerturbations:
    background = DummyBackground()

    def matter_power_spectrum(self, z, k):
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)
        return np.full((len(z), len(k)), 2.0)

    def sigma8_0(self):
        return 0.8


def write_boost_table(path):
    data = np.array(
        [
            [0.1, 1.10, 1.00],
            [0.2, 1.20, 1.10],
            [0.3, 1.30, 1.20],
        ]
    )
    np.savetxt(path, data)


def test_tabulated_boost_interpolates_and_freezes_high_z(tmp_path):
    boost_file = tmp_path / "tabulated_boost.txt"
    write_boost_table(boost_file)

    zs = np.array([0.0, 0.5, 1.0, 2.0])
    z_cols = [1.0, 0.0]  # intentionally unsorted

    tabulated_boost = TabulatedNonlinearBoost(
        DummyBackground(),
        DummyLinearPerturbations(),
        zs,
        str(boost_file),
        z_cols,
        high_z_policy="freeze",
    )

    assert np.isclose(tabulated_boost.mg_spectrum_boost(0.0, 0.14), 1.10)
    assert np.isclose(tabulated_boost.mg_spectrum_boost(1.0, 0.14), 1.20)
    assert np.isclose(tabulated_boost.mg_spectrum_boost(0.5, 0.14), 1.15)
    assert np.isclose(tabulated_boost.mg_spectrum_boost(2.0, 0.14), 1.20)


def test_tabulated_boosted_perturbations_apply_boost(tmp_path):
    boost_file = tmp_path / "tabulated_boost.txt"
    write_boost_table(boost_file)

    zs = np.array([0.0, 1.0])
    z_cols = [1.0, 0.0]

    tabulated_boost = TabulatedNonlinearBoost(
        DummyBackground(),
        DummyLinearPerturbations(),
        zs,
        str(boost_file),
        z_cols,
        high_z_policy="freeze",
    )

    boosted_perturbations = TabulatedBoostedPerturbations(
        DummyLinearPerturbations(),
        DummyBasePerturbations(),
        tabulated_boost.MGboost_interp,
    )

    result = boosted_perturbations.matter_power_spectrum(np.array([0.0, 1.0]), 0.14)
    assert np.allclose(result, np.array([2.2, 2.4]))

    sigma8 = boosted_perturbations.sigma8_0()
    assert np.allclose(sigma8, np.array([0.8]))


@pytest.mark.parametrize("includes_baryons", [False, True])
def test_tabulated_boosted_perturbations_report_baryons(includes_baryons):
    """The boosted spectrum includes baryonic feedback exactly when its base
    does, so that no baryonic mixin is added on top of a base that has it."""
    from cloelib.cosmology.cosmology import ensure_gravity_only

    base = DummyBasePerturbations()
    base.includes_baryons = includes_baryons
    boosted_perturbations = TabulatedBoostedPerturbations(
        DummyLinearPerturbations(), base, lambda z, k, grid=False: 1.0
    )
    assert boosted_perturbations.includes_baryons is includes_baryons
    if includes_baryons:
        with pytest.raises(ValueError, match="twice"):
            ensure_gravity_only(boosted_perturbations, "SomeBaryonBoostMixin")
