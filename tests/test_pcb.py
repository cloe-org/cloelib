"""
Validation of the cold dark matter + baryons (cb) power spectrum, P_cb, and of
its use in the photometric angular power spectra (`PositionsTracer(use_Pcb=True)`).

Everything is checked against references that do not go through the code
under test:

- the linear P_cb of each backend against its physical limits (P_cb -> P_mm
  on large scales, P_cb -> P_mm / f_cb^2 on small scales where massive
  neutrinos don't cluster, P_cb = P_mm without massive neutrinos) and against
  the other backends;
- the nonlinear P_cb of each backend against its own nonlinear P_mm, under
  the assumption that massive neutrinos stay linear (as in HMcode2020);
- the cb x matter cross spectrum used for (cb-positions) x (matter tracer)
  pairs against CAMB's own delta_cb x delta_m spectrum (linear) and against
  the cross spectrum reconstructed from the backend's P_mm^NL (nonlinear);
- the full C_ell against CCL, an independent Limber implementation fed with
  the same reference spectra.

With f_cb = Omega_cb0 / Omega_m0 and neutrinos following linear theory,
delta_m = f_cb delta_cb + f_nu delta_nu gives

    P_mm = f_cb^2 P_cb + 2 f_cb f_nu P_cb,nu + f_nu^2 P_nu
    P_cb,m = f_cb P_cb + f_nu P_cb,nu

where P_cb,nu and P_nu are linear. In linear theory delta_cb and delta_nu are
perfectly correlated, so f_nu^2 P_nu = (sqrt(P_mm^L) - f_cb sqrt(P_cb^L))^2,
and eliminating P_cb,nu gives the reference cross spectrum

    P_cb,m = [P_mm + f_cb^2 P_cb - (sqrt(P_mm^L) - f_cb sqrt(P_cb^L))^2] / (2 f_cb).

The tests need the external backends (CAMB, CCL, and optionally CLASS and
HMcode2020Emu) and are skipped when they are not installed.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.observables.photo.contributions import CB, MATTER
from cloelib.observables.photo.spectrum_engine import SpectraBank
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

camb = pytest.importorskip("camb")

from cloelib.cosmology.camb_cosmology import (  # noqa: E402
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)

H = 0.677
COSMO_PARS = {
    "H0": 100 * H,
    "Omega_b0": 0.022 / H**2,
    "Omega_cdm0": 0.12 / H**2,
    "Omega_k0": 0.0,
    "As": 2.1e-9,
    "ns": 0.96,
    "alpha_s": 0.0,
    "mnu": 0.15,
    "w0": -1.0,
    "wa": 0.0,
    "gamma_MG": 0.0,
    "N_mnu": 1,
}
# Redshift grid of the perturbations: HMcode2020Emu is trained up to z = 3.
Z_PK = np.linspace(0.0, 3.0, 60)
# Grids where the spectra are compared.
Z_TEST = np.array([0.0, 0.5, 1.0, 2.0])
K_TEST = np.logspace(-3, np.log10(5.0), 30)
# Large / small scales for the linear asymptotics (1/Mpc).
K_LARGE_SCALES = 1e-4
K_SMALL_SCALES = 5.0


def _f_cb(background):
    return float(background.Omega_cb(0.0) / background.Omega_m(0.0))


def _reference_cross(nonlinear, linear, background, zs, ks):
    """cb x matter spectrum from the backend's P_mm and P_cb (see module docstring)."""
    f_cb = _f_cb(background)
    p_mm = nonlinear.matter_power_spectrum(zs, ks)
    p_cb = nonlinear.matter_power_spectrum_cb(zs, ks)
    f_nu2_p_nu = (
        np.sqrt(linear.matter_power_spectrum(zs, ks))
        - f_cb * np.sqrt(linear.matter_power_spectrum_cb(zs, ks))
    ) ** 2
    return (p_mm + f_cb**2 * p_cb - f_nu2_p_nu) / (2 * f_cb)


def _camb_cross_linear(background, zs, ks):
    """CAMB's own linear delta_cb x delta_m power spectrum."""
    return camb.get_matter_power_interpolator(
        background.interface_args["CAMBparams"],
        nonlinear=False,
        extrap_kmax=300,
        hubble_units=False,
        k_hunit=False,
        var1="delta_nonu",
        var2="delta_tot",
    ).P(zs, ks)


# --- Backends ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def camb_background():
    return CAMBBackground(**COSMO_PARS)


@pytest.fixture(scope="module")
def camb_linear(camb_background):
    return CAMBLinearPerturbations(camb_background, Z_PK)


@pytest.fixture(scope="module")
def camb_nonlinear(camb_background):
    return CAMBNonLinearPerturbations(camb_background, None, Z_PK)


@pytest.fixture(scope="module")
def class_linear():
    pytest.importorskip("classy")
    from cloelib.cosmology.class_cosmology import (
        CLASSBackground,
        CLASSLinearPerturbations,
    )

    return CLASSLinearPerturbations(CLASSBackground(**COSMO_PARS), Z_PK)


@pytest.fixture(scope="module")
def hmcode(camb_background):
    pytest.importorskip("HMcode2020Emu")
    from cloelib.cosmology.HMcode2020Emu_cosmology import (
        HMemuLinearPerturbations,
        HMemuNonLinearPerturbations,
    )

    linear = HMemuLinearPerturbations(background=camb_background, redshifts=Z_PK)
    nonlinear = HMemuNonLinearPerturbations(
        background=camb_background,
        linearperturbations=linear,
        redshifts=Z_PK,
        log10TAGN=7.8,
    )
    return linear, nonlinear


@pytest.fixture(scope="module")
def linear_backends(request):
    """Name -> linear perturbations, for the backends that are installed."""
    backends = {"CAMB": request.getfixturevalue("camb_linear")}
    for name, fixture in [("CLASS", "class_linear"), ("HMcode2020Emu", "hmcode")]:
        try:
            value = request.getfixturevalue(fixture)
        except pytest.skip.Exception:
            continue
        backends[name] = value[0] if isinstance(value, tuple) else value
    return backends


@pytest.fixture(scope="module")
def jax_nonlinear():
    from cloelib.cosmology.jax_cosmology import (
        JAXBackground,
        JAXNonLinearPerturbations,
    )

    nonlinear = JAXNonLinearPerturbations(JAXBackground(**COSMO_PARS))
    # Evaluated on the fly: no tabulated grids, so set the ones
    # `AngularTwoPoint` reads.
    nonlinear.z = Z_PK
    nonlinear.k = np.logspace(-4, np.log10(50.0), 200)
    return nonlinear


@pytest.fixture(scope="module", params=["CAMB", "HMcode2020Emu", "JAX"])
def nonlinear_case(request, camb_background, camb_linear, camb_nonlinear):
    """(nonlinear, linear, background) for each nonlinear backend.

    CAMB's nonlinear perturbations have no `linearperturbations`, so their
    cross spectrum goes through `sqrt(P_cb^NL P_mm^NL)`; HMcode2020Emu's and
    JAX's have one and go through `f_cb (P_cb^NL - P_cb^L) + sqrt(P_cb^L P_mm^L)`.
    """
    if request.param == "CAMB":
        return camb_nonlinear, camb_linear, camb_background
    if request.param == "JAX":
        nonlinear = request.getfixturevalue("jax_nonlinear")
        return nonlinear, nonlinear.linearperturbations, nonlinear.background
    linear, nonlinear = request.getfixturevalue("hmcode")
    return nonlinear, linear, camb_background


# --- Linear P_cb --------------------------------------------------------------------


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_camb_pcb_equals_pmm_without_massive_neutrinos(nonlinear):
    background = CAMBBackground(**{**COSMO_PARS, "mnu": 0.0, "N_mnu": 0})
    if nonlinear:
        perturbations = CAMBNonLinearPerturbations(background, None, Z_PK)
    else:
        perturbations = CAMBLinearPerturbations(background, Z_PK)
    assert_allclose(_f_cb(background), 1.0, rtol=1e-12)
    assert_allclose(
        perturbations.matter_power_spectrum_cb(Z_TEST, K_TEST),
        perturbations.matter_power_spectrum(Z_TEST, K_TEST),
        rtol=1e-10,
    )


def test_linear_pcb_asymptotics(linear_backends, camb_background):
    # Above the neutrino free-streaming scale delta_nu = delta_cb, so P_cb = P_mm;
    # well below it delta_nu -> 0, so P_mm = f_cb^2 P_cb.
    f_cb = _f_cb(camb_background)
    for name, linear in linear_backends.items():
        for k, expected, rtol in [
            # 1e-3: HMcode2020Emu's emulation error on the largest scales.
            (K_LARGE_SCALES, 1.0, 1e-3),
            (K_SMALL_SCALES, 1.0 / f_cb**2, 2e-3),
        ]:
            ks = np.array([k])
            ratio = linear.matter_power_spectrum_cb(Z_TEST, ks) / (
                linear.matter_power_spectrum(Z_TEST, ks)
            )
            assert_allclose(ratio, expected, rtol=rtol, err_msg=f"{name}, k={k}")


def test_linear_pcb_agrees_between_backends(linear_backends):
    if len(linear_backends) < 2:
        pytest.skip("needs CLASS or HMcode2020Emu besides CAMB")
    reference = linear_backends["CAMB"].matter_power_spectrum_cb(Z_TEST, K_TEST)
    for name, linear in linear_backends.items():
        assert_allclose(
            linear.matter_power_spectrum_cb(Z_TEST, K_TEST),
            reference,
            rtol=3e-3,
            err_msg=name,
        )


def test_jax_linear_pcb_ratio_matches_camb(camb_linear, jax_nonlinear):
    # The JAX P_mm is an Eisenstein & Hu fit, so only the cb / matter ratio
    # (EH99 scale-dependent growth) is compared. It differs from 1 by up to
    # 1 / f_cb^2 - 1 ~ 2e-2 here.
    linear = jax_nonlinear.linearperturbations
    jax_ratio = linear.matter_power_spectrum_cb(Z_TEST, K_TEST) / (
        linear.matter_power_spectrum(Z_TEST, K_TEST)
    )
    camb_ratio = camb_linear.matter_power_spectrum_cb(Z_TEST, K_TEST) / (
        camb_linear.matter_power_spectrum(Z_TEST, K_TEST)
    )
    assert_allclose(jax_ratio, camb_ratio, rtol=1e-3)


# --- Nonlinear P_cb -----------------------------------------------------------------


def test_nonlinear_pcb_is_linear_on_large_scales(nonlinear_case):
    nonlinear, linear, _ = nonlinear_case
    ks = np.logspace(-4, -3, 5)
    assert_allclose(
        nonlinear.matter_power_spectrum_cb(Z_TEST, ks),
        linear.matter_power_spectrum_cb(Z_TEST, ks),
        rtol=2e-3,
    )


def test_nonlinear_pcb_consistent_with_nonlinear_pmm(nonlinear_case):
    # With linear neutrinos, P_mm - f_cb^2 P_cb only contains neutrino terms
    # and is the same in linear and nonlinear theory.
    nonlinear, linear, background = nonlinear_case
    f_cb = _f_cb(background)
    expected_pmm = f_cb**2 * nonlinear.matter_power_spectrum_cb(Z_TEST, K_TEST) + (
        linear.matter_power_spectrum(Z_TEST, K_TEST)
        - f_cb**2 * linear.matter_power_spectrum_cb(Z_TEST, K_TEST)
    )
    assert_allclose(
        nonlinear.matter_power_spectrum(Z_TEST, K_TEST), expected_pmm, rtol=2e-3
    )


# --- cb x matter cross spectrum -----------------------------------------------------


def _positions(
    perturbations, dndz, z, use_Pcb, magnification_bias=0.0, bias=(1.3, 1.7)
):
    n_bins = dndz.shape[0]
    nuisance = {
        **{f"dz_pos_{i + 1}": 0.0 for i in range(n_bins)},
        **{f"width_pos_{i + 1}": 1.0 for i in range(n_bins)},
        **{f"magnification_bias_{i + 1}": magnification_bias for i in range(n_bins)},
        **{f"b1_photo_bin{i}": bias[i] for i in range(n_bins)},
    }
    return PositionsTracer(perturbations, dndz, z, "per_bin", nuisance, use_Pcb=use_Pcb)


def _shear(perturbations, dndz, z):
    n_bins = dndz.shape[0]
    nuisance = {
        **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(n_bins)},
        **{f"dz_shear_{i + 1}": 0.0 for i in range(n_bins)},
        **{f"width_shear_{i + 1}": 1.0 for i in range(n_bins)},
    }
    # No IA: the shear kernel is pure lensing, as CCL's `WeakLensingTracer` below.
    return ShearTracer(
        perturbations=perturbations, dndz=dndz, z=z, nuisance_params=nuisance
    )


@pytest.fixture(scope="module")
def tomography():
    z = np.linspace(0.01, 3.0, 300)
    dndz = np.array([np.exp(-0.5 * ((z - zc) / 0.15) ** 2) for zc in (0.6, 1.2)])
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    return z, dndz


def test_cross_spectrum_linear_matches_camb(camb_linear, camb_background):
    bank = SpectraBank.from_perturbations(camb_linear, K_TEST, Z_TEST)
    expected = _camb_cross_linear(camb_background, Z_TEST, K_TEST)
    for fields in [(CB, MATTER), (MATTER, CB)]:
        assert_allclose(bank.base(*fields), expected, rtol=1e-5)


def test_cross_spectrum_nonlinear_matches_reference(nonlinear_case):
    nonlinear, linear, background = nonlinear_case
    bank = SpectraBank.from_perturbations(nonlinear, K_TEST, Z_TEST)
    assert_allclose(
        bank.base(CB, MATTER),
        _reference_cross(nonlinear, linear, background, Z_TEST, K_TEST),
        rtol=2e-3,
    )


class _CountingPerturbations:
    """Wraps a backend, counting the calls to its matter and cb spectra.

    Every other attribute is the wrapped backend's; `linearperturbations`
    is wrapped too when present, sharing the same counter.
    """

    def __init__(self, base, counts, suffix=""):
        self._base = base
        self._counts = counts
        self._suffix = suffix
        if hasattr(base, "linearperturbations"):
            self.linearperturbations = _CountingPerturbations(
                base.linearperturbations, counts, "_lin"
            )

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _count(self, name):
        key = name + self._suffix
        self._counts[key] = self._counts.get(key, 0) + 1

    def matter_power_spectrum(self, zs, ks):
        self._count("P_mm")
        return self._base.matter_power_spectrum(zs, ks)

    def matter_power_spectrum_cb(self, zs, ks):
        self._count("P_cb")
        return self._base.matter_power_spectrum_cb(zs, ks)


@pytest.mark.parametrize("engine", ["legacy", "generalized"])
def test_each_spectrum_computed_once(nonlinear_case, tomography, engine):
    # A cb galaxy auto spectrum with magnification needs every base spectrum:
    # P_cb (bias x bias), P_mm (magnification x magnification) and the cross
    # spectrum (bias x magnification), which reuses P_cb and, when the
    # backend has linear perturbations, the linear P_mm and P_cb.
    nonlinear, _, _ = nonlinear_case
    z, dndz = tomography
    counts = {}
    pert = _CountingPerturbations(nonlinear, counts)
    pos = _positions(pert, dndz, z, use_Pcb=True, magnification_bias=0.8)
    ells = np.logspace(1.0, np.log10(2000.0), 4)
    a2p = AngularTwoPoint(pos, pos)
    if engine == "generalized":
        contributions = pos.get_contributions()
        a2p._compute_cl_generalized(ells, nonlinear.k, contributions, contributions)
    else:
        a2p.get_Cl_tensor(ells, 0, nonlinear.k)

    expected = ["P_mm", "P_cb"]
    if hasattr(nonlinear, "linearperturbations"):
        expected += ["P_mm_lin", "P_cb_lin"]
    assert counts == dict.fromkeys(expected, 1)


# --- Spectrum per contribution ------------------------------------------------------


@pytest.mark.parametrize("partner", ["self", "SHE", "POS-mm"])
@pytest.mark.parametrize("engine", ["legacy", "generalized"])
def test_magnification_traces_total_matter(camb_nonlinear, tomography, partner, engine):
    # Only the galaxy-bias term of a `use_Pcb=True` tracer follows cb; the
    # magnification term is lensing and follows the total matter. The windows
    # are linear, so the full C_ell must split into a bias-only cb tracer plus
    # a magnification-only tracer (zero galaxy bias, no cb).
    z, dndz = tomography
    pert = camb_nonlinear
    ells = np.logspace(1.0, np.log10(2000.0), 8)
    bias, s_mag = (1.3, 1.7), 0.8

    def pos(b, s, use_Pcb):
        return _positions(pert, dndz, z, use_Pcb, magnification_bias=s, bias=b)

    full = pos(bias, s_mag, True)
    bias_only = pos(bias, 0.0, True)
    mag_only = pos((0.0, 0.0), s_mag, False)

    if partner == "SHE":
        others = [_shear(pert, dndz, z)]
    elif partner == "POS-mm":
        others = [pos(bias, s_mag, False)]
    else:
        others = None  # auto spectrum: the partner splits as well

    def cl(t1, t2):
        a2p = AngularTwoPoint(t1, t2)
        if engine == "generalized":
            c1, c2 = t1.get_contributions(), t2.get_contributions()
            return np.asarray(a2p._compute_cl_generalized(ells, pert.k, c1, c2))
        return np.asarray(a2p.get_Cl_tensor(ells, 0, pert.k))

    if others is None:
        expected = sum(
            cl(a, b) for a in (bias_only, mag_only) for b in (bias_only, mag_only)
        )
        actual = cl(full, full)
    else:
        expected = cl(bias_only, others[0]) + cl(mag_only, others[0])
        actual = cl(full, others[0])
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-30)


# --- Angular power spectra vs CCL ---------------------------------------------------


@pytest.fixture(
    scope="module", params=["CAMB-linear", "CAMB-nonlinear", "HMcode2020Emu"]
)
def cl_case(request, camb_background, camb_linear, camb_nonlinear):
    """(perturbations, cb x matter reference callable, rtol on C_cb / C_mm).

    The C_cb / C_mm tolerance is set by the reference: exact for CAMB, while
    HMcode2020Emu's nonlinear P_mm and P_cb are emulated separately and obey
    the linear-neutrino relation behind `_reference_cross` only to ~1e-3.
    """
    if request.param == "CAMB-linear":
        return (
            camb_linear,
            lambda zs, ks: _camb_cross_linear(camb_background, zs, ks),
            1e-5,
        )
    if request.param == "CAMB-nonlinear":
        nonlinear, linear, rtol = camb_nonlinear, camb_linear, 1e-5
    else:
        (linear, nonlinear), rtol = request.getfixturevalue("hmcode"), 1e-3
    return (
        nonlinear,
        lambda zs, ks: _reference_cross(nonlinear, linear, camb_background, zs, ks),
        rtol,
    )


def test_cl_matches_ccl(cl_case, tomography):
    ccl = pytest.importorskip("pyccl")
    perturbations, cross, rtol_ratio = cl_case
    z, dndz = tomography
    bias = (1.3, 1.7)
    ells = np.logspace(1.0, np.log10(2000.0), 12)
    n_bins = dndz.shape[0]

    cosmo = ccl.Cosmology(
        Omega_c=COSMO_PARS["Omega_cdm0"],
        Omega_b=COSMO_PARS["Omega_b0"],
        h=H,
        A_s=COSMO_PARS["As"],
        n_s=COSMO_PARS["ns"],
        m_nu=COSMO_PARS["mnu"],
        mass_split="single",
        T_CMB=2.7255,
        Neff=3.044,
    )
    ccl_pos = [
        ccl.NumberCountsTracer(
            cosmo, has_rsd=False, dndz=(z, dndz[i]), bias=(z, np.full_like(z, bias[i]))
        )
        for i in range(n_bins)
    ]
    ccl_shear = [ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[i])) for i in range(n_bins)]
    k_grid = np.logspace(-4, 1.5, 400)

    def pk2d(pk_fn):
        return ccl.Pk2D(
            a_arr=1 / (1 + Z_PK[::-1]),
            lk_arr=np.log(k_grid),
            pk_arr=np.log(pk_fn(Z_PK, k_grid))[::-1],
            is_logp=True,
        )

    def ccl_cl(tracers1, tracers2, pk):
        cl = [
            [ccl.angular_cl(cosmo, t1, t2, ells, p_of_k_a=pk) for t2 in tracers2]
            for t1 in tracers1
        ]
        return np.moveaxis(np.array(cl), -1, 0)  # (n_ell, n_bins, n_bins)

    def cloelib_cl(t1, t2):
        return np.asarray(
            AngularTwoPoint(t1, t2).get_Cl_tensor(ells, 0, perturbations.k)
        )

    pos_cb = _positions(perturbations, dndz, z, use_Pcb=True, bias=bias)
    pos_mm = _positions(perturbations, dndz, z, use_Pcb=False, bias=bias)
    shear = _shear(perturbations, dndz, z)
    pk_mm = pk2d(perturbations.matter_power_spectrum)

    diag = np.arange(n_bins)
    # Lens bin i x source bin j >= i: a source bin in front of the lenses gives
    # a tiny C_ell dominated by the integration accuracy.
    lens, source = np.triu_indices(n_bins)
    cases = {
        # name: (C_ell^cb cloelib, C_ell^mm cloelib, C_ell^cb CCL, C_ell^mm CCL, selection)
        "POS-POS": (
            cloelib_cl(pos_cb, pos_cb),
            cloelib_cl(pos_mm, pos_mm),
            ccl_cl(ccl_pos, ccl_pos, pk2d(perturbations.matter_power_spectrum_cb)),
            ccl_cl(ccl_pos, ccl_pos, pk_mm),
            # Non-overlapping bin pairs are tiny and dominated by the integration.
            (slice(None), diag, diag),
        ),
        "POS-SHE": (
            cloelib_cl(pos_cb, shear),
            cloelib_cl(pos_mm, shear),
            ccl_cl(ccl_pos, ccl_shear, pk2d(cross)),
            ccl_cl(ccl_pos, ccl_shear, pk_mm),
            (slice(None), lens, source),
        ),
    }
    for name, (cl_cb, cl_mm, ref_cb, ref_mm, sel) in cases.items():
        # The cb spectrum must make a difference, or the test checks nothing.
        assert np.max(np.abs(cl_cb[sel] / cl_mm[sel] - 1)) > 1e-2, name
        assert_allclose(cl_cb[sel], ref_cb[sel], rtol=1e-3, err_msg=name)
        # Integration differences between the two codes cancel in the ratio.
        assert_allclose(
            cl_cb[sel] / cl_mm[sel],
            ref_cb[sel] / ref_mm[sel],
            rtol=rtol_ratio,
            err_msg=name,
        )
