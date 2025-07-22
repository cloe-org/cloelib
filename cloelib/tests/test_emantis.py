import numpy as np
import pytest


try:
    from emantis.matter_power_spectrum import NonLinearMGBoostEmulator
    from cloelib.cosmology.emantis_cosmology import EmantisFofrNonLinearPerturbations

except ModuleNotFoundError:
    _EMANTIS_INSTALLED = False

else:
    _EMANTIS_INSTALLED = True


class LCDMBackground:

    def __init__(self) -> None:

        self.h = 0.7
        self.Omega_b0 = 0.05
        self.Omega_cdm0 = 0.27
        self.mnu = 0
        self.As = 2e-9
        self.ns = 0.96


class LCDMNonLinearPerturbations:

    def __init__(self) -> None:
        pass

    def matter_power_spectrum(self, zs, ks):
        return np.ones((zs.shape[0], ks.shape[0]))


@pytest.mark.skipif(
    not _EMANTIS_INSTALLED,
    reason="emantis is not installed",
)
def test_mg_boost_cloelib_vs_external():
    """Validate the nonlinear matter power spectrum boost.

    The boost obtained with the cloelib interface of emantis is compared to the boost
    obtained directly from emantis.
    """

    # Redshift and wavenumber arrays.
    zs = np.linspace(0, 3, 100)
    ks = np.geomspace(1e-4, 5, 100)

    fR0 = -1e-5

    # Init. background.
    background = LCDMBackground()

    # Init. LCDM nonlinear perturbations.
    lcdm_nonlinear = LCDMNonLinearPerturbations()

    # Init. f(R) nonlinear perturbations.
    emantis_cloe = EmantisFofrNonLinearPerturbations(
        background=background,
        nonlinearperturbations_lcdm=lcdm_nonlinear,
        redshifts=zs,
        fR0=fR0,
    )

    # Init. e-MANTIS emulator from emantis directly.
    emantis_ext = NonLinearMGBoostEmulator(model="fR")

    emantis_ext_params = {
        "Omega_m": background.Omega_cdm0
        + background.Omega_b0
        + background.mnu / 93.14 / background.h**2,
        "Omega_b": background.Omega_b0,
        "A_s": background.As,
        "n_s": background.ns,
        "h": background.h,
        "logfR0": -np.log10(np.abs(fR0)),
    }

    # Compute e-MANTIS boost by calling emantis directly (k in units of h/Mpc).
    boost_ext = emantis_ext.predict_boost(
        emantis_ext_params, aexp=1 / (1 + zs), k=ks / background.h
    )

    # Compute f(R) power spectrum.
    pk_fr = emantis_cloe.matter_power_spectrum(zs, ks)

    # Compute LCDM power spectrum.
    pk_lcdm = lcdm_nonlinear.matter_power_spectrum(zs, ks)

    # Compute power spectrum boost.
    boost_cloe = pk_fr / pk_lcdm

    # Compute absolute relative difference.
    abs_rel_diff = np.abs(boost_cloe - boost_ext) / boost_ext

    # Maximum absolute relative difference is smaller than 0.2%.
    assert np.max(abs_rel_diff) < 2e-3

    # Mean absolute relative difference is smaller than 0.02%.
    assert np.mean(abs_rel_diff) < 2e-4
