import pytest
import numpy as np

from cloelib.cosmology.class_cosmology import (
    CLASSBackground, CLASSLinearPerturbations
)
from cloelib.cosmology.derived_cosmology import (
    hubble_rate, growth_function_ODE
)

@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.fixture
def class_background_instance(scope="module"):
    """Fixture to create an instance of CLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    Omega_m = 0.3
    Omega_cdm0 = Omega_m - Omega_b0
    class_instance = CLASSBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0,
    )
    return class_instance


@pytest.fixture
def class_perturbation_instances(
    class_background_instance, zs, scope="module"
):
    """Fixture to create the Linear CLASSPerturbations."""
    class_lin = CLASSLinearPerturbations(
        background=class_background_instance, redshifts=zs
    )
    return class_lin


def test_hubble_rate(zs, class_background_instance, scope="module"):
    bg = class_background_instance
    lna = -np.log(zs +1)
    h_z_class = bg.hubble_parameter(zs)
    h_z_derived = 100 * hubble_rate(
        lna, bg.h, bg.Omega_cdm0 + bg.Omega_b0, bg.Omega_k0, bg.w0, bg.wa
    )
    assert h_z_derived == pytest.approx(h_z_class, rel=1e-3)


def test_growth_function_ODE(
    zs, ks, class_background_instance, class_perturbation_instances
):
    growth_ODE = growth_function_ODE(class_background_instance, zs)
    growth_class = (
        1 + zs) * class_perturbation_instances.growth_factor(zs, ks)[:, 0]
    # The growth factor from CLASS is normalised to 1 at z=0
    assert growth_ODE / growth_ODE[0] == pytest.approx(growth_class, rel=2e-2)