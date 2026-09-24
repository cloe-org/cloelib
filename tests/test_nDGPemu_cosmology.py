import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)


try:
    from nDGPemu import BoostPredictor
    from cloelib.cosmology.nDGPemu_cosmology import NDGPemuNonLinearPerturbations

except ModuleNotFoundError:
    _NDGEMU_INSTALLED = False

else:
    _NDGEMU_INSTALLED = True


@pytest.fixture(scope="module")
def zs():
    return np.linspace(0, 2, 30)


@pytest.fixture(scope="module")
def ks():
    return np.linspace(1e-4, 6, 100)


@pytest.fixture(scope="module")
def camb_background_instance():
    """Fixture to create an instance of CAMBBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    camb_instance = CAMBBackground(
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
    return camb_instance


@pytest.fixture(scope="module")
def camb_linearperturbations_instance(camb_background_instance, zs):
    """Fixture to create an instance of CAMBLinearPerturbations."""
    camb_instance = CAMBLinearPerturbations(camb_background_instance, zs)
    return camb_instance


@pytest.fixture(scope="module")
def camb_nonlinearperturbations_instance(camb_background_instance, zs):
    """Fixture to create an instance of CAMBNonLinearPerturbations."""
    camb_instance = CAMBNonLinearPerturbations(camb_background_instance, None, zs)
    return camb_instance


@pytest.mark.skipif(
    not _NDGEMU_INSTALLED,
    reason="nDGPemu is not installed",
)
def test_mg_boost_cloelib_vs_external(
    zs,
    ks,
    camb_background_instance,
    camb_linearperturbations_instance,
    camb_nonlinearperturbations_instance,
):
    """Validate the nonlinear matter power spectrum boost.

    The boost obtained with the cloelib interface of nDGPemu is compared to the boost
    obtained directly from nDGPemu.
    """

    # Extended parameter value.
    omega_rc = 1

    # Init. nDGP nonlinear perturbations.
    # Note: we use a LCDM linear instance, instead of an nDGP one.
    # This makes no difference for this test, but a linear nDGP instance should be used in all generality.
    ndgpemu_cloe = NDGPemuNonLinearPerturbations(
        background=camb_background_instance,
        linearperturbations=camb_linearperturbations_instance,
        nonlinearperturbations_lcdm=camb_nonlinearperturbations_instance,
        redshifts=zs,
        omega_rc=omega_rc,
    )

    # Init. nDGP emulator.
    ndpgemu_ext = BoostPredictor()

    ndgpemu_ext_params = {
        "Om": camb_background_instance.Omega_m(0),
        "Ob": camb_background_instance.Omega_b0,
        "As": camb_background_instance.As,
        "ns": camb_background_instance.ns,
        "h": camb_background_instance.h,
    }

    # Compute nDGPemu boost by calling the emulator directly (k in units of h/Mpc).
    boost_ext = np.zeros((zs.shape[0], ks.shape[0]))
    for i, zi in enumerate(zs):
        boost_ext[i, :] = ndpgemu_ext.predict(
            (1 / 4 / omega_rc) ** (1 / 2),
            zi,
            ndgpemu_ext_params,
            k_out=ks / camb_background_instance.h,
            ext=3,
        )
    np.min(boost_ext)

    # Compute nDGP power spectrum.
    pk_fr = ndgpemu_cloe.matter_power_spectrum(zs, ks)

    # Compute LCDM power spectrum.
    pk_lcdm = camb_nonlinearperturbations_instance.matter_power_spectrum(zs, ks)

    # Compute power spectrum boost.
    boost_cloe = pk_fr / pk_lcdm

    # Compute absolute relative difference.
    abs_rel_diff = np.abs(boost_cloe - boost_ext) / boost_ext

    # Maximum absolute relative difference is smaller than 0.1%.
    assert np.max(abs_rel_diff) < 1e-3

    # Mean absolute relative difference is smaller than 0.01%.
    assert np.mean(abs_rel_diff) < 1e-4
