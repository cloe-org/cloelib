"""Unit tests for the spectroscopic Legendre multipoles."""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles

H0 = 67.7
h = H0 / 100.0

BACKGROUND_KWARGS = dict(
    H0=H0,
    Omega_b0=0.022 / h**2,
    Omega_cdm0=0.12 / h**2,
    Omega_k0=0.0,
    As=2e-9,
    ns=0.96,
    alpha_s=0.0,
    mnu=0.0,
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.0,
    N_mnu=0,
)

# Shot noise and observational systematics of the multipoles
NUISANCE_PARAMETERS = dict(NP0=1.0, NP20=0.0, NP22=0.0, sigmaz=0.0, fcorr=1.0)


class _AnalyticSpectroPower:
    """Minimal SpectroPower place holder.

    The analytical form used here is only needed for testing the fcorr rescaling.
    More accurate unit tests of the Legendre multipoles should be added in a future PR.
    """

    def __init__(self, background, redshift):
        self.background = background
        self.redshift = redshift

    def Pk2d_rsd(self, k, mu):
        return k**2 * mu**4

    def Pk2d_term_rsd(self, k, mu, term_list):
        return np.array([self.Pk2d_rsd(k, mu) for _ in term_list])


@pytest.fixture(scope="module")
def spectro_power():
    """Fixture to create an instance of the analytic SpectroPower."""
    return _AnalyticSpectroPower(CAMBBackground(**BACKGROUND_KWARGS), redshift=1.0)


def _multipoles(spectro_power, fcorr):
    # Noise switched off, so that only the signal is rescaled by fcorr
    parameters = {**NUISANCE_PARAMETERS, "NP0": 0.0, "fcorr": fcorr}
    return LegendreMultipoles(
        spectro_power=spectro_power,
        background_fiducial=spectro_power.background,
        parameters=parameters,
        nbar=1e-3,
    )


@pytest.mark.parametrize(
    "method, kwargs",
    [("power_multipoles", {}), ("power_term_multipoles", {"term_list": ["b1-b1"]})],
)
def test_fcorr_rescales_signal(spectro_power, method, kwargs):
    """Test that fcorr rescales the multipoles of the signal by fcorr**2."""
    k = np.logspace(-2, np.log10(0.3), 20)
    reference = getattr(_multipoles(spectro_power, fcorr=1.0), method)(k, **kwargs)
    rescaled = getattr(_multipoles(spectro_power, fcorr=0.8), method)(k, **kwargs)
    for ell in ["ell0", "ell2", "ell4"]:
        np.testing.assert_allclose(rescaled[ell], 0.8**2 * reference[ell], rtol=1e-10)
