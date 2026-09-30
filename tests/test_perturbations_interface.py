"""Tests of the interface shared by all `Perturbations` implementations.

Every backend is built through the same constructor keywords
(`LinearPerturbationsFactory`/`NonLinearPerturbationsFactory`) and must follow
the same `growth_rate(zs=None, ks=None)` convention. Backends whose optional
dependency is not installed are skipped. The static counterpart of these
checks, `tests/typing/perturbations_conformance.py`, is run through ty.
"""

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from cloelib.cosmology.cosmology import Perturbations

H0 = 67.7
h = H0 / 100.0

# One massive neutrino with N_ur inferred as 2.0308, which every backend,
# including the emulators, supports.
COSMO = dict(
    H0=H0,
    Omega_b0=0.022 / h**2,
    Omega_cdm0=0.12 / h**2,
    Omega_k0=0.0,
    As=2e-9,
    ns=0.96,
    mnu=0.06,
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.0,
    N_mnu=1,
)
REDSHIFTS = np.linspace(0.0, 2.0, 11)
ZS = np.array([0.3, 1.1])
KS = np.logspace(-3, 0, 5)


def _camb_background():
    from cloelib.cosmology.camb_cosmology import CAMBBackground

    return CAMBBackground(**COSMO)


def _camb():
    from cloelib.cosmology.camb_cosmology import (
        CAMBLinearPerturbations,
        CAMBNonLinearPerturbations,
    )

    lin = CAMBLinearPerturbations(background=_camb_background(), redshifts=REDSHIFTS)
    nonlin = CAMBNonLinearPerturbations(
        background=_camb_background(), linearperturbations=lin, redshifts=REDSHIFTS
    )
    return lin, nonlin


def _class():
    from cloelib.cosmology.class_cosmology import (
        CLASSBackground,
        CLASSLinearPerturbations,
        CLASSNonLinearPerturbations,
    )

    background = CLASSBackground(**COSMO)
    lin = CLASSLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = CLASSNonLinearPerturbations(
        background=background,
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        nonlinear_model="halofit",
    )
    return lin, nonlin


def _hi_class():
    from cloelib.cosmology.hi_class_cosmology import (
        hi_classBackground,
        hi_classLinearPerturbations,
        hi_classNonLinearPerturbations,
    )

    background = hi_classBackground(**COSMO)
    lin = hi_classLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = hi_classNonLinearPerturbations(
        background=background,
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        nonlinear_model="halofit",
    )
    return lin, nonlin


def _mgclass():
    from cloelib.cosmology.mgclass_cosmology import (
        MGCLASSBackground,
        MGCLASSLinearPerturbations,
        MGCLASSNonLinearPerturbations,
    )

    background = MGCLASSBackground(
        **COSMO,
        alpha_s=0.0,
        Y_He=0.25,
        mg_ansatz="plk_musigma_norm_late",
        mg_z_init=0.0,
        mg_params={"mg_E11": 0.0, "mg_E22": 0.0},
    )
    lin = MGCLASSLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = MGCLASSNonLinearPerturbations(
        background=background, linearperturbations=lin, redshifts=REDSHIFTS
    )
    return lin, nonlin


def _mochi_class():
    from cloelib.cosmology.mochi_class_cosmology import (
        mochiCLASSBackground,
        mochiCLASSLinearPerturbations,
        mochiCLASSNonLinearPerturbations,
    )

    background = mochiCLASSBackground(
        **COSMO,
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    lin = mochiCLASSLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = mochiCLASSNonLinearPerturbations(
        background=background,
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        nonlinear_model="halofit",
    )
    return lin, nonlin


def _hmemu():
    from cloelib.cosmology.HMcode2020Emu_cosmology import (
        HMemuLinearPerturbations,
        HMemuNonLinearPerturbations,
    )

    background = _camb_background()
    lin = HMemuLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = HMemuNonLinearPerturbations(
        background=background, linearperturbations=lin, redshifts=REDSHIFTS
    )
    return lin, nonlin


def _bacco():
    from cloelib.cosmology.baccoemu_cosmology import (
        BACCOemuLinearPerturbations,
        BACCOemuNonLinearPerturbations,
    )

    background = _camb_background()
    lin = BACCOemuLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = BACCOemuNonLinearPerturbations(
        background=background, linearperturbations=lin, redshifts=REDSHIFTS
    )
    return lin, nonlin


def _ee2():
    from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations
    from cloelib.cosmology.EE2_cosmology import EE2NonLinearPerturbations

    background = _camb_background()
    lin = CAMBLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = EE2NonLinearPerturbations(
        background=background, linearperturbations=lin, redshifts=REDSHIFTS
    )
    return lin, nonlin


def _jax():
    from cloelib.cosmology.jax_cosmology import (
        JAXBackground,
        JAXLinearPerturbations,
        JAXNonLinearPerturbations,
    )

    background = JAXBackground(**COSMO)
    lin = JAXLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = JAXNonLinearPerturbations(
        background=background, linearperturbations=lin, redshifts=REDSHIFTS
    )
    return lin, nonlin


# name: (builder, module that must be importable, scale-dependent growth rate)
BACKENDS = {
    "CAMB": (_camb, "camb", False),
    "CLASS": (_class, "classy", False),
    "hi_class": (_hi_class, "hiclassy", False),
    "MGCLASS": (_mgclass, "mgclassy", False),
    "mochi_class": (_mochi_class, "mochi_classy", False),
    "HMcode2020Emu": (_hmemu, "HMcode2020Emu", False),
    "BACCOemu": (_bacco, "baccoemu", True),
    "EE2": (_ee2, "euclidemu2", False),
    "JAX": (_jax, "jax", False),
}


@pytest.fixture(scope="module", params=list(BACKENDS))
def backend(request):
    """Linear and nonlinear perturbations of each installed backend."""
    builder, module, scale_dependent = BACKENDS[request.param]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    lin, nonlin = builder()
    return {"Linear": lin, "NonLinear": nonlin, "scale_dependent": scale_dependent}


def _grid(perturbations):
    """Redshift grid `growth_rate()` is evaluated on by default."""
    z = getattr(perturbations, "z", None)
    return REDSHIFTS if z is None else z


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_implements_protocol(backend, key):
    """Every backend built through the shared constructor keywords is a `Perturbations`."""
    assert isinstance(backend[key], Perturbations)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_growth_rate_default_grid(backend, key):
    """`growth_rate()` returns one value per redshift of its own grid."""
    if backend["scale_dependent"]:
        pytest.skip("scale-dependent growth rate requires ks")
    perturbations = backend[key]
    f = np.asarray(perturbations.growth_rate())
    assert f.shape == _grid(perturbations).shape
    assert np.all((f > 0) & (f <= 1))
    np.testing.assert_allclose(
        np.asarray(perturbations.growth_rate(_grid(perturbations))), f, rtol=1e-6
    )


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_growth_rate_at_requested_redshifts(backend, key):
    """`growth_rate(zs)` has shape (nz,) and `growth_rate(zs, ks)` (nz, nk)."""
    perturbations = backend[key]
    f_zk = np.asarray(perturbations.growth_rate(ZS, KS))
    assert f_zk.shape == (len(ZS), len(KS))
    assert np.all((f_zk > 0) & (f_zk <= 1))
    if backend["scale_dependent"]:
        with pytest.raises(ValueError, match="ks"):
            perturbations.growth_rate(ZS)
        return
    f = np.asarray(perturbations.growth_rate(ZS))
    assert f.shape == (len(ZS),)
    # Scale-independent growth rates are broadcast along k.
    np.testing.assert_allclose(f_zk[:, 0], f, rtol=1e-6)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_growth_rate_matches_approximation(backend, key):
    """The growth rate agrees across backends with f ~ Omega_m(z)^0.55."""
    perturbations = backend[key]
    f = np.asarray(perturbations.growth_rate(ZS, KS))[:, 0]
    Om = np.asarray(perturbations.background.Omega_m(ZS))
    np.testing.assert_allclose(f, Om**0.55, rtol=3e-2)


requires_camb = pytest.mark.skipif(
    importlib.util.find_spec("camb") is None, reason="camb not installed"
)
requires_classy = pytest.mark.skipif(
    importlib.util.find_spec("classy") is None, reason="classy not installed"
)


def _camb_nonlinear_pk(**kwargs):
    from cloelib.cosmology.camb_cosmology import CAMBNonLinearPerturbations

    # CAMBNonLinearPerturbations configures the background's CAMB parameters,
    # so each spectrum is computed from a fresh background.
    nonlin = CAMBNonLinearPerturbations(_camb_background(), None, REDSHIFTS, **kwargs)
    return nonlin.matter_power_spectrum(np.array([0.0]), np.array([1.0, 3.0]))


def _class_nonlinear_pk(**kwargs):
    from cloelib.cosmology.class_cosmology import (
        CLASSBackground,
        CLASSNonLinearPerturbations,
    )

    nonlin = CLASSNonLinearPerturbations(
        CLASSBackground(**COSMO), None, REDSHIFTS, **kwargs
    )
    return nonlin.matter_power_spectrum(np.array([0.0]), np.array([1.0, 3.0]))


@requires_camb
def test_camb_log10TAGN_selects_feedback_model():
    """log10TAGN without a nonlinear model uses mead2020_feedback instead of being ignored."""
    pk_default = _camb_nonlinear_pk()
    pk_feedback = _camb_nonlinear_pk(log10TAGN=7.8)
    np.testing.assert_allclose(
        pk_feedback,
        _camb_nonlinear_pk(nonlinear_model="mead2020_feedback", log10TAGN=7.8),
    )
    assert np.all(pk_feedback < pk_default)
    assert np.all(_camb_nonlinear_pk(log10TAGN=8.2) < pk_feedback)


@requires_camb
def test_camb_log10TAGN_with_other_model_raises():
    """log10TAGN with a nonlinear model that ignores it raises."""
    with pytest.raises(ValueError, match="mead2020_feedback"):
        _camb_nonlinear_pk(nonlinear_model="mead2016", log10TAGN=7.8)


@requires_classy
def test_class_accepts_log10TAGN():
    """CLASS takes log10TAGN like CAMB and HMcode2020Emu (issue #543)."""
    pk_no_feedback = _class_nonlinear_pk(
        nonlinear_model="hmcode", hmcode_version="2020"
    )
    pk_feedback = _class_nonlinear_pk(log10TAGN=7.8)
    np.testing.assert_allclose(
        pk_feedback,
        _class_nonlinear_pk(
            nonlinear_model="hmcode",
            hmcode_version="2020_baryonic_feedback",
            log10TAGN=7.8,
        ),
    )
    assert np.all(pk_feedback < pk_no_feedback)
    assert np.all(_class_nonlinear_pk(log10TAGN=8.2) < pk_feedback)


@requires_camb
@requires_classy
def test_class_and_camb_log10TAGN_agree():
    """The same log10TAGN gives the same HMcode2020 feedback spectrum in CLASS and CAMB."""
    np.testing.assert_allclose(
        _class_nonlinear_pk(log10TAGN=7.5), _camb_nonlinear_pk(log10TAGN=7.5), rtol=1e-2
    )


@requires_classy
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nonlinear_model="halofit", log10TAGN=7.8),
        dict(nonlinear_model="hmcode", hmcode_version="2020", log10TAGN=7.8),
    ],
)
def test_class_log10TAGN_with_other_model_raises(kwargs):
    """log10TAGN with a CLASS model or HMcode version that ignores it raises."""
    with pytest.raises(ValueError, match="2020_baryonic_feedback"):
        _class_nonlinear_pk(**kwargs)


def test_jax_nonlinear_uses_given_linear_perturbations():
    """JAXNonLinearPerturbations takes linearperturbations like the other backends."""
    from cloelib.cosmology.jax_cosmology import (
        JAXBackground,
        JAXLinearPerturbations,
        JAXNonLinearPerturbations,
    )

    background = JAXBackground(**COSMO)
    lin = JAXLinearPerturbations(background)
    assert JAXNonLinearPerturbations(background, lin).linearperturbations is lin
    with pytest.raises(ValueError, match="same `background`"):
        JAXNonLinearPerturbations(JAXBackground(**COSMO), lin)
    with pytest.raises(ValueError, match="redshift grid"):
        JAXNonLinearPerturbations(background).growth_rate()


@pytest.mark.skipif(importlib.util.find_spec("ty") is None, reason="ty not installed")
def test_ty_conformance():
    """ty accepts every backend as implementing the shared interfaces.

    See `tests/typing/perturbations_conformance.py`; `unused-ignore-comment`
    makes the documented exceptions fail too once they are resolved.
    """
    root = pathlib.Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--python",
            sys.prefix,
            "--error",
            "unused-ignore-comment",
            "--output-format",
            "concise",
            "tests/typing/perturbations_conformance.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
