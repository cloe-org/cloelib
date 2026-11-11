"""
Consistency of each backend's `cb_cross_prescription` with its own spectra.

Each prescription for the cb x matter cross spectrum is exact under a
condition on the backend's non-linear P_mm and P_cb (see
`cloelib.cosmology.cosmology`). Given the backend's P_mm^NL, each condition
predicts its P_cb^NL:

- CB_CROSS_CORRELATED (same non-linear boost for all fields):
      P_cb^NL = P_mm^NL P_cb^L / P_mm^L
- CB_CROSS_LINEAR_NU (neutrinos stay linear):
      P_cb^NL = P_cb^L + (P_mm^NL - P_mm^L) / f_cb^2

For the backends where the two predictions can be told apart, the declared
prescription must reproduce the backend's P_cb^NL clearly better than the
other one. For the CLASS family, whose non-linear model runs on P_cb
separately, and for the CosmoPowerJAX halofit sector, where the emulation
error hides the difference, neither does: only the size of the error is
bounded, at the level quoted in `docs/code_structure/perturbations.md`.

The comparison uses k in [0.05, 3] 1/Mpc, where the non-linear corrections
are, and the largest neutrino mass in each emulator's range, where the two
conditions differ most. Every case also checks that `SpectraBank` builds the
cross spectrum with the declared formula. JAX (no massive neutrinos), emantis
(m_nu = 0 only) and MG-CosmoPowerJAX (the prescription of its baseline) are
not included.

The tests are skipped for the backends that are not installed. hi_class and
mochi_class use the bounds of CLASS, whose non-linear modules they share.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloelib.cosmology.cosmology import CB_CROSS_CORRELATED, CB_CROSS_LINEAR_NU
from cloelib.observables.photo.contributions import CB, MATTER
from cloelib.observables.photo.spectrum_engine import SpectraBank

H = 0.677
Z_PK = np.linspace(0.0, 3.0, 31)
Z_TEST = np.array([0.0, 0.5, 1.0, 1.5])
K_TEST = np.logspace(np.log10(0.05), np.log10(3.0), 40)  # 1/Mpc


def _cosmo(mnu):
    return dict(
        H0=100 * H,
        Omega_b0=0.022 / H**2,
        Omega_cdm0=0.12 / H**2,
        Omega_k0=0.0,
        As=2.1e-9,
        ns=0.96,
        alpha_s=0.0,
        mnu=mnu,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )


# --- Backends: (background, linear, nonlinear) at a given neutrino mass -------------


def _camb_linear(mnu):
    pytest.importorskip("camb")
    from cloelib.cosmology.camb_cosmology import (
        CAMBBackground,
        CAMBLinearPerturbations,
    )

    background = CAMBBackground(**_cosmo(mnu))
    return background, CAMBLinearPerturbations(background, Z_PK)


def _camb(mnu):
    from cloelib.cosmology.camb_cosmology import CAMBNonLinearPerturbations

    background, linear = _camb_linear(mnu)
    nonlinear = CAMBNonLinearPerturbations(
        background, linear, Z_PK, nonlinear_model="mead2020"
    )
    return background, linear, nonlinear


def _ee2(mnu):
    pytest.importorskip("euclidemu2")
    from cloelib.cosmology.EE2_cosmology import EE2NonLinearPerturbations

    background, linear = _camb_linear(mnu)
    return background, linear, EE2NonLinearPerturbations(background, linear, Z_PK)


def _baccoemu(mnu):
    pytest.importorskip("baccoemu")
    from cloelib.cosmology.baccoemu_cosmology import (
        BACCOemuLinearPerturbations,
        BACCOemuNonLinearPerturbations,
    )
    from cloelib.cosmology.camb_cosmology import CAMBBackground

    background = CAMBBackground(**_cosmo(mnu))
    linear = BACCOemuLinearPerturbations(background, Z_PK)
    return (
        background,
        linear,
        BACCOemuNonLinearPerturbations(background, linear, Z_PK),
    )


def _hmcode2020emu(mnu):
    pytest.importorskip("HMcode2020Emu")
    from cloelib.cosmology.camb_cosmology import CAMBBackground
    from cloelib.cosmology.HMcode2020Emu_cosmology import (
        HMemuLinearPerturbations,
        HMemuNonLinearPerturbations,
    )

    background = CAMBBackground(**_cosmo(mnu))
    linear = HMemuLinearPerturbations(background, Z_PK)
    return background, linear, HMemuNonLinearPerturbations(background, linear, Z_PK)


def _cosmopower_jax(sector):
    def build(mnu):
        pytest.importorskip("cosmopower_jax")
        from cloelib.cosmology.camb_cosmology import CAMBBackground
        from cloelib.cosmology.cosmopower_jax_cosmology import (
            CosmoPowerJAXLCDMPerturbations as CPJ,
        )

        background = CAMBBackground(**_cosmo(mnu))
        linear = CPJ.Linear(background, Z_PK)
        return background, linear, getattr(CPJ, sector)(background, linear, Z_PK)

    return build


def _class(nonlinear_model, **options):
    def build(mnu):
        pytest.importorskip("classy")
        from cloelib.cosmology.class_cosmology import (
            CLASSBackground,
            CLASSLinearPerturbations,
            CLASSNonLinearPerturbations,
        )

        background = CLASSBackground(**_cosmo(mnu))
        linear = CLASSLinearPerturbations(background, Z_PK)
        nonlinear = CLASSNonLinearPerturbations(
            background, linear, Z_PK, nonlinear_model=nonlinear_model, **options
        )
        return background, linear, nonlinear

    return build


def _mgclass(nonlinear_model):
    def build(mnu):
        pytest.importorskip("mgclassy")
        from cloelib.cosmology.mgclass_cosmology import (
            MGCLASSBackground,
            MGCLASSLinearPerturbations,
            MGCLASSNonLinearPerturbations,
        )

        # GR limit of the mu-Sigma parametrisation.
        background = MGCLASSBackground(
            **_cosmo(mnu),
            Y_He=0.25,
            mg_ansatz="plk_musigma_norm_late",
            mg_z_init=0.0,
            mg_params={"mg_E11": 0.0, "mg_E22": 0.0},
        )
        linear = MGCLASSLinearPerturbations(background, Z_PK)
        nonlinear = MGCLASSNonLinearPerturbations(
            background, linear, Z_PK, nonlinear_model=nonlinear_model
        )
        return background, linear, nonlinear

    return build


def _hi_class(mnu):
    pytest.importorskip("hiclassy")
    from cloelib.cosmology.hi_class_cosmology import (
        hi_classBackground,
        hi_classLinearPerturbations,
        hi_classNonLinearPerturbations,
    )

    background = hi_classBackground(**_cosmo(mnu))
    linear = hi_classLinearPerturbations(background, Z_PK)
    nonlinear = hi_classNonLinearPerturbations(
        background, linear, Z_PK, nonlinear_model="halofit"
    )
    return background, linear, nonlinear


def _mochi_class(mnu):
    pytest.importorskip("mochi_classy")
    from cloelib.cosmology.mochi_class_cosmology import (
        mochiCLASSBackground,
        mochiCLASSLinearPerturbations,
        mochiCLASSNonLinearPerturbations,
    )

    # LCDM background, where mochi_class keeps its non-linear model.
    background = mochiCLASSBackground(
        **_cosmo(mnu),
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    linear = mochiCLASSLinearPerturbations(background, Z_PK)
    nonlinear = mochiCLASSNonLinearPerturbations(
        background, linear, Z_PK, nonlinear_model="halofit"
    )
    return background, linear, nonlinear


# --- Cases ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable
    mnu: float
    #: Prescription the backend must declare.
    prescription: str
    #: Bound on the error of the declared prescription (both, if not `separates`).
    bound: float
    #: Whether the declared prescription must beat the other one.
    separates: bool = True


SEPARATING = [
    Case("CAMB-mead2020", _camb, 0.45, CB_CROSS_CORRELATED, 1e-5),
    Case("EE2", _ee2, 0.15, CB_CROSS_LINEAR_NU, 1e-6),
    Case("BACCOemu", _baccoemu, 0.30, CB_CROSS_LINEAR_NU, 3e-4),
    Case("HMcode2020Emu", _hmcode2020emu, 0.45, CB_CROSS_CORRELATED, 4e-3),
    Case("CPJ-HMcode", _cosmopower_jax("NonLinear"), 0.45, CB_CROSS_CORRELATED, 5e-3),
]
# Neither prescription is exact: today ~5e-3 for the CosmoPowerJAX halofit
# sector and 1.4e-2 (halofit) to 2.1e-2 (HMcode2020) for the CLASS family.
INEXACT = [
    Case(
        "CPJ-halofit",
        _cosmopower_jax("NonLinearHalofit"),
        0.15,
        CB_CROSS_CORRELATED,
        1e-2,
        separates=False,
    ),
    Case(
        "CLASS-halofit",
        _class("halofit"),
        0.15,
        CB_CROSS_LINEAR_NU,
        3e-2,
        separates=False,
    ),
    Case(
        "CLASS-HMcode2020",
        _class("hmcode", hmcode_version="2020"),
        0.15,
        CB_CROSS_LINEAR_NU,
        3e-2,
        separates=False,
    ),
    Case(
        "MGCLASS-halofit",
        _mgclass("halofit"),
        0.15,
        CB_CROSS_LINEAR_NU,
        3e-2,
        separates=False,
    ),
    Case(
        "MGCLASS-HMcode",
        _mgclass("hmcode"),
        0.15,
        CB_CROSS_LINEAR_NU,
        3e-2,
        separates=False,
    ),
    Case(
        "hi_class-halofit",
        _hi_class,
        0.15,
        CB_CROSS_LINEAR_NU,
        3e-2,
        separates=False,
    ),
    Case(
        "mochi_class-halofit",
        _mochi_class,
        0.15,
        CB_CROSS_LINEAR_NU,
        3e-2,
        separates=False,
    ),
]


@pytest.fixture(scope="module", params=SEPARATING + INEXACT, ids=lambda case: case.name)
def backend(request):
    case = request.param
    return case, *case.build(case.mnu)


def _f_cb(background):
    return float(np.squeeze(background.Omega_cb(0.0) / background.Omega_m(0.0)))


def _spectra(background, linear, nonlinear):
    return (
        _f_cb(background),
        linear.matter_power_spectrum(Z_TEST, K_TEST),
        linear.matter_power_spectrum_cb(Z_TEST, K_TEST),
        nonlinear.matter_power_spectrum(Z_TEST, K_TEST),
        nonlinear.matter_power_spectrum_cb(Z_TEST, K_TEST),
    )


def _errors(background, linear, nonlinear):
    """Max relative error on P_cb^NL predicted by each prescription."""
    f_cb, pmm_l, pcb_l, pmm, pcb = _spectra(background, linear, nonlinear)
    predicted = {
        CB_CROSS_CORRELATED: pmm * pcb_l / pmm_l,
        CB_CROSS_LINEAR_NU: pcb_l + (pmm - pmm_l) / f_cb**2,
    }
    return {name: np.max(np.abs(p / pcb - 1)) for name, p in predicted.items()}


def test_declared_prescription(backend):
    case, _, _, nonlinear = backend
    assert nonlinear.cb_cross_prescription == case.prescription


def test_declared_prescription_matches_spectra(backend):
    case, background, linear, nonlinear = backend
    errors = _errors(background, linear, nonlinear)
    declared = errors[nonlinear.cb_cross_prescription]
    if case.separates:
        (other,) = (e for name, e in errors.items() if name != case.prescription)
        assert declared < case.bound, errors
        assert declared < 0.5 * other, errors
    else:
        assert max(errors.values()) < case.bound, errors


def test_cross_spectrum_follows_declared_prescription(backend):
    _, background, linear, nonlinear = backend
    f_cb, pmm_l, pcb_l, pmm, pcb = _spectra(background, linear, nonlinear)
    if nonlinear.cb_cross_prescription == CB_CROSS_CORRELATED:
        expected = np.sqrt(pcb * pmm)
    else:
        expected = f_cb * (pcb - pcb_l) + np.sqrt(pcb_l * pmm_l)
    bank = SpectraBank.from_perturbations(nonlinear, K_TEST, Z_TEST)
    assert_allclose(bank.base(CB, MATTER), expected, rtol=1e-8)
