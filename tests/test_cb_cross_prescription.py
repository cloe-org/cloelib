"""
The cb x matter cross spectrum prescription (`cb_cross_prescription`).

Every backend providing `matter_power_spectrum_cb` declares which formula the
cb x matter cross spectrum follows (see `cloelib.cosmology.cosmology`), and
`SpectraBank` applies it. The declaration is checked statically, so the test
needs none of the external backends.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import cloelib.cosmology
from cloelib.cosmology.cosmology import (
    CB_CROSS_CORRELATED,
    CB_CROSS_LINEAR_NU,
)
from cloelib.observables.photo.contributions import CB, MATTER
from cloelib.observables.photo.spectrum_engine import SpectraBank

COSMOLOGY_DIR = Path(cloelib.cosmology.__file__).parent

# Backends that keep massive neutrinos linear on top of a linear backend use
# CB_CROSS_LINEAR_NU; Boltzmann codes (uniform non-linear boost) and linear
# backends use CB_CROSS_CORRELATED. A new backend providing
# `matter_power_spectrum_cb` must be added here.
EXPECTED = {
    "camb_cosmology.py": {
        "CAMBLinearPerturbations": "CB_CROSS_CORRELATED",
        "CAMBNonLinearPerturbations": "CB_CROSS_CORRELATED",
    },
    "class_cosmology.py": {
        "CLASSLinearPerturbations": "CB_CROSS_CORRELATED",
        "CLASSNonLinearPerturbations": "CB_CROSS_CORRELATED",
    },
    "hi_class_cosmology.py": {
        "hi_classLinearPerturbations": "CB_CROSS_CORRELATED",
        "hi_classNonLinearPerturbations": "CB_CROSS_CORRELATED",
    },
    "mgclass_cosmology.py": {
        "MGCLASSLinearPerturbations": "CB_CROSS_CORRELATED",
        "MGCLASSNonLinearPerturbations": "CB_CROSS_CORRELATED",
    },
    "mochi_class_cosmology.py": {
        "mochiCLASSLinearPerturbations": "CB_CROSS_CORRELATED",
        "mochiCLASSNonLinearPerturbations": "CB_CROSS_CORRELATED",
    },
    "HMcode2020Emu_cosmology.py": {
        "HMemuLinearPerturbations": "CB_CROSS_CORRELATED",
        "HMemuNonLinearPerturbations": "CB_CROSS_LINEAR_NU",
    },
    "EE2_cosmology.py": {
        "EE2NonLinearPerturbations": "CB_CROSS_LINEAR_NU",
    },
    "baccoemu_cosmology.py": {
        "BACCOemuLinearPerturbations": "CB_CROSS_CORRELATED",
        "BACCOemuNonLinearPerturbations": "CB_CROSS_LINEAR_NU",
    },
    "emantis_cosmology.py": {
        "EmantisFofrNonLinearPerturbations": "CB_CROSS_LINEAR_NU",
    },
    "jax_cosmology.py": {
        "JAXLinearPerturbations": "CB_CROSS_CORRELATED",
        "JAXNonLinearPerturbations": "CB_CROSS_LINEAR_NU",
    },
    "mg_cosmopower_jax_cosmology.py": {
        "MGLinearPerturbations": "CB_CROSS_CORRELATED",
        "MGNonLinearPerturbations": "CB_CROSS_LINEAR_NU",
    },
    "cosmopower_jax_cosmology.py": {
        f"{outer}.{inner}": value
        for outer, inners in {
            "CosmoPowerJAXw0waCDMPerturbations": ("NonLinear", "NonLinearHalofit"),
            "CosmoPowerJAXwCDMPerturbations": ("NonLinear", "NonLinearHalofit"),
            "CosmoPowerJAXLCDMPerturbations": ("NonLinear", "NonLinearHalofit"),
            "CosmoPowerJAXLCDMCurvaturePerturbations": ("NonLinear",),
            "CosmoPowerJAXw0waCurvaturePerturbations": ("NonLinear",),
            "CosmoPowerJAXLCDMRunningIndexPerturbations": ("NonLinear",),
            "CosmoPowerJAXw0waRunningIndexPerturbations": ("NonLinear",),
        }.items()
        for inner, value in [("Linear", "CB_CROSS_CORRELATED")]
        + [(name, "CB_CROSS_LINEAR_NU") for name in inners]
    },
}


def _declared_prescriptions(path):
    """`{qualname: declared value}` for every class of `path` defining
    `matter_power_spectrum_cb`, the value being the name assigned to its
    `cb_cross_prescription` class attribute (`None` if not declared).

    Classes with base classes (the `Perturbations` Protocol, the baryon
    `Combined` wrapper, subclasses of a backend) are skipped: they inherit
    the declaration.
    """
    found = {}

    def walk(node, prefix):
        for child in node.body:
            if not isinstance(child, ast.ClassDef):
                continue
            qualname = prefix + child.name
            methods = {n.name for n in child.body if isinstance(n, ast.FunctionDef)}
            if "matter_power_spectrum_cb" in methods and not child.bases:
                found[qualname] = next(
                    (
                        ast.unparse(n.value)
                        for n in child.body
                        if isinstance(n, ast.Assign)
                        and [ast.unparse(t) for t in n.targets]
                        == ["cb_cross_prescription"]
                    ),
                    None,
                )
            walk(child, qualname + ".")

    walk(ast.parse(path.read_text()), "")
    return found


def test_every_cb_backend_declares_its_prescription():
    declared = {
        path.name: classes
        for path in sorted(COSMOLOGY_DIR.glob("*.py"))
        if (classes := _declared_prescriptions(path))
    }
    assert declared == EXPECTED


# --- SpectraBank applies the declared prescription ----------------------------------

ZS = np.array([0.0, 1.0])
KS = np.logspace(-2, 0, 5)
F_CB = 0.99


class _Background:
    def Omega_cb(self, z):
        return np.array([0.3 * F_CB])

    def Omega_m(self, z):
        return np.array([0.3])


class _Perturbations:
    """Toy backend with distinct, positive spectra on `(ZS, KS)`."""

    background = _Background()

    def __init__(self, scale=1.0, linear=None, **attributes):
        self._scale = scale
        if linear is not None:
            self.linearperturbations = linear
        for name, value in attributes.items():
            setattr(self, name, value)

    def matter_power_spectrum(self, zs, ks):
        return self._scale * np.outer(1.0 + zs, ks**-1)

    def matter_power_spectrum_cb(self, zs, ks):
        return 1.1 * self._scale * np.outer(1.0 + zs, ks**-1.2)


def _cross(perturbations):
    return np.asarray(
        SpectraBank.from_perturbations(perturbations, KS, ZS).base(CB, MATTER)
    )


def test_correlated_prescription():
    pert = _Perturbations(cb_cross_prescription=CB_CROSS_CORRELATED)
    expected = np.sqrt(
        pert.matter_power_spectrum_cb(ZS, KS) * pert.matter_power_spectrum(ZS, KS)
    )
    assert_allclose(_cross(pert), expected, rtol=1e-12)


def test_linear_neutrinos_prescription():
    linear = _Perturbations(scale=0.5)
    pert = _Perturbations(linear=linear, cb_cross_prescription=CB_CROSS_LINEAR_NU)
    pcb_l = linear.matter_power_spectrum_cb(ZS, KS)
    expected = F_CB * (pert.matter_power_spectrum_cb(ZS, KS) - pcb_l) + np.sqrt(
        pcb_l * linear.matter_power_spectrum(ZS, KS)
    )
    assert_allclose(_cross(pert), expected, rtol=1e-12)


def test_prescription_ignores_linearperturbations():
    # The formula follows the declaration, not the presence of
    # `linearperturbations`.
    linear = _Perturbations(scale=0.5)
    with_linear = _Perturbations(
        linear=linear, cb_cross_prescription=CB_CROSS_CORRELATED
    )
    without_linear = _Perturbations(cb_cross_prescription=CB_CROSS_CORRELATED)
    assert_allclose(_cross(with_linear), _cross(without_linear), rtol=1e-12)


@pytest.mark.parametrize(
    "attributes",
    [{}, {"cb_cross_prescription": "uniform"}, {"cb_cross_prescription": None}],
)
def test_missing_or_unknown_prescription_raises(attributes):
    with pytest.raises(ValueError, match="cb_cross_prescription"):
        _cross(_Perturbations(**attributes))


def test_linear_neutrinos_without_linear_spectra_raises():
    pert = _Perturbations(cb_cross_prescription=CB_CROSS_LINEAR_NU)
    with pytest.raises(ValueError, match="linearperturbations"):
        _cross(pert)


def test_undeclared_backend_still_provides_mm_and_cb():
    # Only the cross spectrum needs the declaration.
    bank = SpectraBank.from_perturbations(_Perturbations(), KS, ZS)
    pert = _Perturbations()
    assert_allclose(bank.base(MATTER, MATTER), pert.matter_power_spectrum(ZS, KS))
    assert_allclose(bank.base(CB, CB), pert.matter_power_spectrum_cb(ZS, KS))
