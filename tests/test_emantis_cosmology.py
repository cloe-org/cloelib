import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)


try:
    from emantis.matter_power_spectrum import NonLinearMGBoostEmulator
    from cloelib.cosmology.emantis_cosmology import EmantisFofrNonLinearPerturbations

except ModuleNotFoundError:
    _EMANTIS_INSTALLED = False

else:
    _EMANTIS_INSTALLED = True


@pytest.fixture(scope="module")
def zs():
    return np.linspace(0, 3, 30)


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
    not _EMANTIS_INSTALLED,
    reason="emantis is not installed",
)
def test_mg_boost_cloelib_vs_external(
    zs,
    ks,
    camb_background_instance,
    camb_linearperturbations_instance,
    camb_nonlinearperturbations_instance,
):
    """Validate the nonlinear matter power spectrum boost.

    The boost obtained with the cloelib interface of emantis is compared to the boost
    obtained directly from emantis.
    """

    # Extended parameter value.
    fR0 = -1e-5

    # Init. f(R) nonlinear perturbations.
    # Note: we use a LCDM linear instance, instead of an f(R) one.
    # This makes no difference for this test, but a linear f(R) instance should be used in all generality.
    emantis_cloe = EmantisFofrNonLinearPerturbations(
        background=camb_background_instance,
        linearperturbations=camb_linearperturbations_instance,
        nonlinearperturbations_lcdm=camb_nonlinearperturbations_instance,
        redshifts=zs,
        fR0=fR0,
    )

    # Init. e-MANTIS emulator from emantis directly.
    emantis_ext = NonLinearMGBoostEmulator(model="fR")

    emantis_ext_params = {
        "Omega_m": camb_background_instance.Omega_m(0),
        "Omega_b": camb_background_instance.Omega_b0,
        "A_s": camb_background_instance.As,
        "n_s": camb_background_instance.ns,
        "h": camb_background_instance.h,
        "logfR0": -np.log10(np.abs(fR0)),
    }

    # Compute e-MANTIS boost by calling emantis directly (k in units of h/Mpc).
    boost_ext = emantis_ext.predict_boost(
        emantis_ext_params, aexp=1 / (1 + zs), k=ks / camb_background_instance.h
    )

    # Compute f(R) power spectrum.
    pk_fr = emantis_cloe.matter_power_spectrum(zs, ks)

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


@pytest.mark.skipif(
    not _EMANTIS_INSTALLED,
    reason="emantis is not installed",
)
@pytest.mark.parametrize("prescription", ["correlated", "linear_neutrinos", None])
def test_cb_cross_prescription_is_the_baseline_one(prescription):
    """The boost rescales the baseline P_mm and P_cb alike, so the cb x matter
    prescription is the one of the LCDM baseline (`None` if it declares none)."""

    class _Baseline:
        pass

    if prescription is not None:
        _Baseline.cb_cross_prescription = prescription
    # Skip the emulator set-up: the prescription only needs the baseline.
    emantis_cloe = object.__new__(EmantisFofrNonLinearPerturbations)
    emantis_cloe.nonlinearpertubations_lcdm = _Baseline()
    assert emantis_cloe.cb_cross_prescription == prescription


class _FakeBoost:
    """Constant e-MANTIS boost, standing in for `boost_interp`."""

    def __call__(self, zs, log_ks):
        return np.full((np.size(zs), np.size(log_ks)), 1.5)


class _FakeBaselineLCDM:
    """LCDM baseline with distinct total matter and CDM+baryons spectra."""

    def matter_power_spectrum(self, zs, ks):
        return np.outer(1.0 + np.atleast_1d(zs), np.atleast_1d(ks) ** -1)

    def matter_power_spectrum_cb(self, zs, ks):
        return 0.9 * self.matter_power_spectrum(zs, ks)


def _emantis_without_emulator(mnu):
    """Instance with a constant boost and a fake baseline, skipping the emulator."""

    class _Background:
        pass

    _Background.mnu = mnu
    emantis_cloe = object.__new__(EmantisFofrNonLinearPerturbations)
    emantis_cloe._background = _Background()
    emantis_cloe.boost_interp = _FakeBoost()
    emantis_cloe.nonlinearpertubations_lcdm = _FakeBaselineLCDM()
    return emantis_cloe


@pytest.mark.skipif(
    not _EMANTIS_INSTALLED,
    reason="emantis is not installed",
)
def test_pcb_boosts_baseline_pcb():
    """Without massive neutrinos, P_cb = boost x baseline P_cb (not P_mm)."""
    zs, ks = np.array([0.0, 1.0]), np.logspace(-2, 0, 5)
    emantis_cloe = _emantis_without_emulator(mnu=0.0)
    np.testing.assert_allclose(
        emantis_cloe.matter_power_spectrum_cb(zs, ks),
        1.5 * _FakeBaselineLCDM().matter_power_spectrum_cb(zs, ks),
        rtol=1e-12,
    )


@pytest.mark.skipif(
    not _EMANTIS_INSTALLED,
    reason="emantis is not installed",
)
@pytest.mark.parametrize("mnu", [0.06, np.array([0.06, 0.0, 0.0])])
def test_pcb_raises_with_massive_neutrinos(mnu):
    """emantis is calibrated without massive neutrinos: P_cb raises, P_mm doesn't."""
    zs, ks = np.array([0.0, 1.0]), np.logspace(-2, 0, 5)
    emantis_cloe = _emantis_without_emulator(mnu)
    with pytest.raises(ValueError, match="massive neutrinos"):
        emantis_cloe.matter_power_spectrum_cb(zs, ks)
    assert np.all(np.isfinite(emantis_cloe.matter_power_spectrum(zs, ks)))
