"""Tests of the interface shared by all `Perturbations` implementations.

Every backend is built through the same constructor keywords
(`LinearPerturbationsFactory`/`NonLinearPerturbationsFactory`) and must follow
the same `growth_rate(zs=None, ks=None)` convention. Backends whose optional
dependency is not installed are skipped. The static counterpart of these
checks, `tests/typing/perturbations_conformance.py`, is run through ty.
"""

import ast
import importlib.util
import pathlib
import subprocess
import sys
from types import SimpleNamespace

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


def _hmemu_baryons():
    from cloelib.cosmology.cosmology import with_baryon_boost
    from cloelib.cosmology.HMcode2020Emu_cosmology import (
        HMcode2020BaryonBoostMixin,
        HMemuLinearPerturbations,
        HMemuNonLinearPerturbations,
    )

    background = _camb_background()
    lin = HMemuLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = with_baryon_boost(HMemuNonLinearPerturbations, HMcode2020BaryonBoostMixin)(
        background=background,
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        baryon_kwargs=dict(log10TAGN=7.8),
    )
    return lin, nonlin


def _bacco_baryons():
    from cloelib.cosmology.baccoemu_cosmology import (
        BACCOemuLinearPerturbations,
        BACCOemuNonLinearBaryonicPerturbations,
    )

    background = _camb_background()
    lin = BACCOemuLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = BACCOemuNonLinearBaryonicPerturbations(
        background=background,
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        baryon_kwargs=dict(
            M_c=13.0, eta=-0.25, beta=-0.15, M1_z0_cen=11.0, theta_inn=-1.0
        ),
    )
    return lin, nonlin


def _camb_with(BaryonMixinClass, **baryon_kwargs):
    """CAMB nonlinear perturbations with a baryonic boost from another backend."""
    from cloelib.cosmology.camb_cosmology import (
        CAMBLinearPerturbations,
        CAMBNonLinearPerturbations,
    )
    from cloelib.cosmology.cosmology import with_baryon_boost

    background = _camb_background()
    lin = CAMBLinearPerturbations(background=background, redshifts=REDSHIFTS)
    nonlin = with_baryon_boost(CAMBNonLinearPerturbations, BaryonMixinClass)(
        background=background,
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        baryon_kwargs=baryon_kwargs,
    )
    return lin, nonlin


def _camb_bacco_baryons():
    from cloelib.cosmology.baccoemu_cosmology import BACCOemuBaryonBoostMixin

    return _camb_with(
        BACCOemuBaryonBoostMixin,
        M_c=13.0,
        eta=-0.25,
        beta=-0.15,
        M1_z0_cen=11.0,
        theta_inn=-1.0,
    )


def _camb_hmcode_baryons():
    from cloelib.cosmology.HMcode2020Emu_cosmology import HMcode2020BaryonBoostMixin

    return _camb_with(HMcode2020BaryonBoostMixin, log10TAGN=7.8)


def _camb_flamingo():
    from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations
    from cloelib.cosmology.FlamingoBaryonResponseEmulator_cosmology import (
        CAMBNonLinearFLAMINGOBaryonicPerturbations,
    )

    lin = CAMBLinearPerturbations(background=_camb_background(), redshifts=REDSHIFTS)
    nonlin = CAMBNonLinearFLAMINGOBaryonicPerturbations(
        background=_camb_background(),
        linearperturbations=lin,
        redshifts=REDSHIFTS,
        baryon_kwargs=dict(fgas_sigma=0.0, Mstar_sigma=0.0, jet_fraction=0.0),
    )
    return lin, nonlin


# How `growth_rate` treats `ks`:
SCALE_INDEPENDENT = "scale-independent"  # broadcast along k
PIVOT_K = 1.0  # scale dependent, evaluated at this k [1/Mpc] if ks is None
KS_REQUIRED = "ks required"  # scale dependent, ks must be given

# name: (builder, module that must be importable, growth rate k-dependence)
BACKENDS = {
    "CAMB": (_camb, "camb", SCALE_INDEPENDENT),
    "CLASS": (_class, "classy", SCALE_INDEPENDENT),
    "hi_class": (_hi_class, "hiclassy", PIVOT_K),
    "MGCLASS": (_mgclass, "mgclassy", SCALE_INDEPENDENT),
    "mochi_class": (_mochi_class, "mochi_classy", PIVOT_K),
    "HMcode2020Emu": (_hmemu, "HMcode2020Emu", SCALE_INDEPENDENT),
    "BACCOemu": (_bacco, "baccoemu", KS_REQUIRED),
    "EE2": (_ee2, "euclidemu2", SCALE_INDEPENDENT),
    "JAX": (_jax, "jax", SCALE_INDEPENDENT),
    # Nonlinear perturbations with a baryonic boost (`with_baryon_boost`).
    "HMcode2020Emu+HMcode2020": (_hmemu_baryons, "HMcode2020Emu", SCALE_INDEPENDENT),
    "BACCOemu+BACCOemu": (_bacco_baryons, "baccoemu", KS_REQUIRED),
    "CAMB+FLAMINGO": (
        _camb_flamingo,
        "FlamingoBaryonResponseEmulator",
        SCALE_INDEPENDENT,
    ),
    "CAMB+BACCOemu": (_camb_bacco_baryons, "baccoemu", SCALE_INDEPENDENT),
    "CAMB+HMcode2020": (_camb_hmcode_baryons, "HMcode2020Emu", SCALE_INDEPENDENT),
}


@pytest.fixture(scope="module", params=list(BACKENDS))
def backend(request):
    """Linear and nonlinear perturbations of each installed backend."""
    builder, module, k_dependence = BACKENDS[request.param]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    lin, nonlin = builder()
    return {"Linear": lin, "NonLinear": nonlin, "k_dependence": k_dependence}


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
    if backend["k_dependence"] == KS_REQUIRED:
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
    if backend["k_dependence"] == KS_REQUIRED:
        with pytest.raises(ValueError, match="ks"):
            perturbations.growth_rate(ZS)
        return
    f = np.asarray(perturbations.growth_rate(ZS))
    assert f.shape == (len(ZS),)
    if backend["k_dependence"] == SCALE_INDEPENDENT:
        np.testing.assert_allclose(f_zk, np.tile(f[:, None], (1, len(KS))), rtol=1e-6)
    else:
        # Without ks, the scale-dependent growth rate at the pivot scale.
        f_pivot = np.asarray(perturbations.growth_rate(ZS, np.array([PIVOT_K])))
        np.testing.assert_allclose(f_pivot[:, 0], f, rtol=1e-6)


# Known exceptions to the (1, nk) shape for a single redshift.
SINGLE_REDSHIFT_EXCEPTIONS = {
    ("JAX", "Linear", "matter_power_spectrum"): "JAX linear spectrum is squeezed",
    ("JAX", "Linear", "growth_factor"): "JAX growth factor is scale-independent D(z)",
    (
        "JAX",
        "NonLinear",
        "growth_factor",
    ): "JAX growth factor is scale-independent D(z)",
}


@pytest.mark.parametrize(
    "method", ["matter_power_spectrum", "matter_power_spectrum_cb", "growth_factor"]
)
@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_single_redshift_keeps_z_axis(request, backend, key, method):
    """A single redshift gives shape (1, nk), as for any other number of redshifts."""
    name = request.node.callspec.params["backend"]
    reason = SINGLE_REDSHIFT_EXCEPTIONS.get((name, key, method))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    try:
        result = getattr(backend[key], method)(np.array([0.5]), KS)
    except NotImplementedError:
        pytest.skip(f"{method} not implemented")
    assert np.shape(result) == (1, len(KS))


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


def test_jax_nonlinear_keeps_its_redshifts_with_given_linear_perturbations():
    """The `redshifts` of JAXNonLinearPerturbations are its default grid, also
    when its linear perturbations were built without any."""
    from cloelib.cosmology.jax_cosmology import (
        JAXBackground,
        JAXLinearPerturbations,
        JAXNonLinearPerturbations,
    )

    background = JAXBackground(**COSMO)
    lin = JAXLinearPerturbations(background)
    nonlin = JAXNonLinearPerturbations(background, lin, redshifts=ZS)
    np.testing.assert_allclose(nonlin.growth_rate(), lin.growth_rate(ZS))
    # Its own redshifts take precedence over those of the linear perturbations.
    lin_with_z = JAXLinearPerturbations(background, redshifts=REDSHIFTS)
    nonlin = JAXNonLinearPerturbations(background, lin_with_z, redshifts=ZS)
    assert np.shape(nonlin.growth_rate()) == ZS.shape


class _RecordingLinear:
    """Linear perturbations stand-in recording the arguments of `growth_rate`."""

    def __init__(self):
        self.background = SimpleNamespace(Omega_k0=0.0)
        self.calls = []

    def growth_rate(self, zs=None, ks=None):
        self.calls.append(zs)
        return np.zeros(0 if zs is None else len(zs))


@pytest.mark.parametrize("with_grid", [True, False])
def test_boosted_perturbations_growth_rate_on_own_grid(with_grid):
    """Without zs, the boosted wrapper uses the grid of its base perturbations."""
    from cloelib.cosmology.TabulatedBoost_cosmology import (
        TabulatedBoostedPerturbations,
    )

    lin = _RecordingLinear()
    base = SimpleNamespace(background=lin.background)
    if with_grid:
        base.z = ZS
    boosted = TabulatedBoostedPerturbations(lin, base, boost_interp=None)
    boosted.growth_rate()
    boosted.growth_rate(REDSHIFTS)
    if with_grid:
        np.testing.assert_array_equal(lin.calls[0], ZS)
    else:
        # No grid of its own: the linear perturbations use theirs.
        assert lin.calls[0] is None
    np.testing.assert_array_equal(lin.calls[1], REDSHIFTS)


@pytest.mark.skipif(
    importlib.util.find_spec("mgclassy") is None, reason="mgclassy not installed"
)
def test_mgclass_growth_rate_needs_three_redshifts():
    """The MGCLASS finite-difference growth rate rejects grids that are too short."""
    from cloelib.cosmology.mgclass_cosmology import (
        MGCLASSLinearPerturbations,
        MGCLASSNonLinearPerturbations,
    )

    for cls in (MGCLASSLinearPerturbations, MGCLASSNonLinearPerturbations):
        with pytest.raises(ValueError, match="at least 3 redshifts"):
            cls.growth_rate(SimpleNamespace(z=np.array([0.0, 1.0])))


@pytest.mark.skipif(
    importlib.util.find_spec("mochi_classy") is None,
    reason="mochi_classy not installed",
)
def test_mochi_class_log10TAGN_with_stable_basis_raises():
    """mochi_class has no nonlinear corrections with the stable MG basis, so
    log10TAGN is rejected instead of being ignored."""
    from cloelib.cosmology.mochi_class_cosmology import (
        mochiCLASSNonLinearPerturbations,
    )

    background = SimpleNamespace(mg_stable_basis_on=True)
    with pytest.raises(ValueError, match="mg_stable_basis_on"):
        mochiCLASSNonLinearPerturbations(background, None, REDSHIFTS, log10TAGN=7.8)


def test_partial_protocols_match_perturbations():
    """The partial protocols of the ty checks lack exactly one `Perturbations` member."""
    conformance = (
        pathlib.Path(__file__).parent / "typing" / "perturbations_conformance.py"
    )
    members = {
        node.name: {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
        }
        for node in ast.parse(conformance.read_text()).body
        if isinstance(node, ast.ClassDef)
    }
    protocol = {
        name
        for name, value in vars(Perturbations).items()
        if not name.startswith("_") and (callable(value) or isinstance(value, property))
    }
    assert members["PerturbationsWithoutCb"] == protocol - {"matter_power_spectrum_cb"}
    assert members["PerturbationsWithoutBackground"] == protocol - {"background"}


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


class _NoBaryonsNonLinear:
    """Nonlinear perturbations stand-in recording whether it was built."""

    built = False

    def __init__(self, *args, **kwargs):
        _NoBaryonsNonLinear.built = True


class _UnitBoostMixin:
    """Baryonic mixin stand-in with a unit boost."""

    def __init__(self):
        pass

    def baryonic_suppression(self, zs, ks, k_hunit=False):
        return np.ones((np.size(zs), np.size(ks)))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(log10TAGN=7.8),
        dict(baryonic_boost="Burger2025"),
        dict(nonlinear_model="mead2020_feedback"),
        dict(hmcode_version="2020_baryonic_feedback"),
    ],
)
def test_with_baryon_boost_rejects_native_baryons(kwargs):
    """A baryonic mixin is not applied on top of a backend's own baryonic boost."""
    from cloelib.cosmology.cosmology import with_baryon_boost

    _NoBaryonsNonLinear.built = False
    Combined = with_baryon_boost(_NoBaryonsNonLinear, _UnitBoostMixin)
    with pytest.raises(ValueError, match="twice"):
        Combined(None, None, REDSHIFTS, **kwargs)
    assert not _NoBaryonsNonLinear.built
    Combined(None, None, REDSHIFTS, log10TAGN=None)
    assert _NoBaryonsNonLinear.built


def test_with_baryon_boost_rejects_stacked_mixins():
    """Two baryonic mixins are not stacked on the same nonlinear class."""
    from cloelib.cosmology.cosmology import with_baryon_boost

    Combined = with_baryon_boost(_NoBaryonsNonLinear, _UnitBoostMixin)
    with pytest.raises(TypeError, match="twice"):
        with_baryon_boost(Combined, _UnitBoostMixin)
