"""Tests for cloelib.cosmology.mgrowth_cosmology.MGrowthLinearPerturbations.

Covers every gravity model exposed by the class and a few interface contracts
(shapes, LCDM limits, error handling). 
"""

import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations

try:
    import MGrowth  # noqa: F401

    from cloelib.cosmology.MGrowth_cosmology import MGrowthLinearPerturbations
except ImportError:  # pragma: no cover
    pytest.skip(
        "MGrowth is not installed; skipping MGrowth tests.",
        allow_module_level=True,
    )


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture(scope="module")
def camb_background():
    return CAMBBackground(
        H0=67.39774575153639,
        Omega_b0=0.05062684354655124,
        Omega_cdm0=0.3164470361777248 - 0.05062684354655124,
        Omega_k0=0.0,
        As=2.1977194699875245e-9,
        ns=0.9423180532642602,
        mnu=0.0,
        w0=-1,
        wa=0,
        gamma_MG=0.0,
        N_mnu=0.0,
    )


@pytest.fixture(scope="module")
def camb_linear(camb_background):
    zs = np.array([0.1, 0.478, 0.785, 1.5])
    linear = CAMBLinearPerturbations(camb_background, zs)
    return linear, zs


# All supported MGrowth gravity models and a representative parameter dict for each.
# Values are chosen to be physically reasonable but well inside any validity range
# so that initialization is robust.
SUPPORTED_MODELS = [
    ("w0wacdm", {}),
    ("fr", {"fr0": 1e-5}),
    ("dgp", {"omega_rc": 0.25}),
    ("ds", {"xi": 5.0}),
    ("gamma", {"gamma0": 0.55}),
    ("gammaz", {"gamma0": 0.55, "gamma1": 0.0}),
    ("musigma-de", {"mu0": 0.1, "sigma0": 0.1}),
    ("mu", {"mu0": 0.1, "sigma0": 0.1, "c1": 0.0, "c2": 0.0, "lam": 0.0}),
]

# Models whose growth is expected to be k-independent.
SCALE_INDEPENDENT_MODELS = {"w0wacdm", "dgp", "ds", "gamma", "gammaz", "musigma-de"}

# Models that carry a mu / sigma_lensing interpolator.
MU_SIGMA_MODELS = {"musigma-de", "mu"}


# -------------------------------
# Helpers
# -------------------------------


def _make_mg(camb_background, camb_linear, model_name, mgpars):
    linear_pert, _ = camb_linear
    return MGrowthLinearPerturbations(
        background=camb_background,
        base_linear_perturbations=linear_pert,
        gravity_model=model_name,
        mgpars=mgpars,
    )


# -------------------------------
# Initialization
# -------------------------------


@pytest.mark.parametrize("model_name,mgpars", SUPPORTED_MODELS, ids=[m for m, _ in SUPPORTED_MODELS])
def test_initializes_for_all_models(camb_background, camb_linear, model_name, mgpars):
    """Every advertised gravity model should initialize without error."""
    mg = _make_mg(camb_background, camb_linear, model_name, mgpars)

    # Core attributes that downstream code relies on.
    assert mg.gravity_model == model_name
    assert mg.mgpars == mgpars
    assert hasattr(mg, "dz_interp")
    assert hasattr(mg, "fz_interp")
    assert hasattr(mg, "dz_norm_dz0_interp")
    assert hasattr(mg, "dz_norm_w0wacdm_interp")

    # mu-Sigma models expose extra interpolators.
    if model_name in MU_SIGMA_MODELS:
        assert hasattr(mg, "mu_interp")
        assert hasattr(mg, "sigma_lensing")


def test_case_insensitive_model_name(camb_background, camb_linear):
    """`gravity_model` argument should be normalized to lower case."""
    mg = _make_mg(camb_background, camb_linear, "FR", {"fr0": 1e-5})
    assert mg.gravity_model == "fr"


# -------------------------------
# Errors / input validation
# -------------------------------


def test_unsupported_model_raises(camb_background, camb_linear):
    with pytest.raises(ValueError, match="Unsupported gravity model"):
        _make_mg(camb_background, camb_linear, "not_a_model", {})


def test_musigma_de_requires_mu0_sigma0(camb_background, camb_linear):
    with pytest.raises(ValueError, match="Mu-Sigma parameters"):
        _make_mg(camb_background, camb_linear, "musigma-de", {"mu0": 0.0})


def test_mu_requires_full_parameter_set(camb_background, camb_linear):
    # Missing c2 and lam.
    with pytest.raises(ValueError, match="Mu-Sigma parameters"):
        _make_mg(
            camb_background,
            camb_linear,
            "mu",
            {"mu0": 0.1, "sigma0": 0.1, "c1": 0.0},
        )


def test_nonflat_geometry_rejected(camb_linear):
    """Constructor asserts a flat background."""
    nonflat_bg = CAMBBackground(
        H0=67.5,
        Omega_b0=0.05,
        Omega_cdm0=0.25,
        Omega_k0=0.01,  # nonzero curvature
        As=2.1e-9,
        ns=0.965,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0.0,
    )
    linear_pert, _ = camb_linear
    with pytest.raises(AssertionError, match="Non flat"):
        MGrowthLinearPerturbations(
            background=nonflat_bg,
            base_linear_perturbations=linear_pert,
            gravity_model="w0wacdm",
            mgpars={},
        )


# -------------------------------
# Output shapes & basic sanity
# -------------------------------


@pytest.mark.parametrize("model_name,mgpars", SUPPORTED_MODELS, ids=[m for m, _ in SUPPORTED_MODELS])
def test_growth_factor_shape_and_normalization(
    camb_background, camb_linear, model_name, mgpars
):
    mg = _make_mg(camb_background, camb_linear, model_name, mgpars)

    zs = np.array([0.0, 0.3, 1.0, 2.5])
    ks = np.logspace(np.log10(mg.k[0]) + 1e-4, np.log10(mg.k[-1]) - 1e-4, 6)

    D = mg.growth_factor(zs, ks)
    assert D.shape == (len(zs), len(ks))
    assert np.all(np.isfinite(D))

    # By construction `dz_norm_dz0_interp` is D(z, k) / D(0, k) → 1 at z=0.
    np.testing.assert_allclose(D[0], 1.0, atol=1e-6)

    # Growth factor must decrease with redshift at every k.
    assert np.all(np.diff(D, axis=0) <= 1e-10)


@pytest.mark.parametrize("model_name,mgpars", SUPPORTED_MODELS, ids=[m for m, _ in SUPPORTED_MODELS])
def test_growth_rate_shape_and_sign(camb_background, camb_linear, model_name, mgpars):
    mg = _make_mg(camb_background, camb_linear, model_name, mgpars)

    zs = np.array([0.0, 0.5, 1.5])
    ks = np.logspace(np.log10(mg.k[0]) + 1e-4, np.log10(mg.k[-1]) - 1e-4, 4)

    f = mg.growth_rate(zs, ks)
    assert f.shape == (len(zs), len(ks))
    assert np.all(np.isfinite(f))
    # f = dln D / dln a is positive for growing modes in (w0wa)CDM / standard MG.
    assert np.all(f > 0.0)
    # For these standard models f(z=0) is somewhere in [Omega_m^0.55, ~1].
    assert np.all(f[0] < 1.5)


@pytest.mark.parametrize("model_name,mgpars", SUPPORTED_MODELS, ids=[m for m, _ in SUPPORTED_MODELS])
def test_matter_power_spectrum_shape_and_positivity(
    camb_background, camb_linear, model_name, mgpars
):
    mg = _make_mg(camb_background, camb_linear, model_name, mgpars)

    zs = np.array([0.0, 0.5, 1.5])
    ks = np.logspace(np.log10(mg.k[0]) + 1e-4, np.log10(mg.k[-1]) - 1e-4, 8)

    pk = mg.matter_power_spectrum(zs, ks)
    assert pk.shape == (len(zs), len(ks))
    assert np.all(np.isfinite(pk))
    assert np.all(pk > 0.0)

    # P(z, k) must decrease with redshift (linear matter PS).
    assert np.all(np.diff(pk, axis=0) <= 1e-10)


@pytest.mark.parametrize("model_name,mgpars", SUPPORTED_MODELS, ids=[m for m, _ in SUPPORTED_MODELS])
def test_sigma8_0_is_positive_float(camb_background, camb_linear, model_name, mgpars):
    mg = _make_mg(camb_background, camb_linear, model_name, mgpars)
    s8 = mg.sigma8_0()
    assert np.isfinite(s8)
    assert s8 > 0.0
    # σ8 in the broad ballpark of plausible cosmologies.
    assert 0.3 < s8 < 1.5


# -------------------------------
# Scale (in)dependence
# -------------------------------


@pytest.mark.parametrize(
    "model_name,mgpars",
    [(m, p) for m, p in SUPPORTED_MODELS if m in SCALE_INDEPENDENT_MODELS],
    ids=[m for m, _ in SUPPORTED_MODELS if m in SCALE_INDEPENDENT_MODELS],
)
def test_growth_is_scale_independent_for_non_k_models(
    camb_background, camb_linear, model_name, mgpars
):
    """Models without k-dependent growth must produce D(z, k) flat in k."""
    mg = _make_mg(camb_background, camb_linear, model_name, mgpars)

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(np.log10(mg.k[0]) + 1e-4, np.log10(mg.k[-1]) - 1e-4, 10)

    D = mg.growth_factor(zs, ks)
    # Each row should be (approximately) constant across k.
    rel_spread = (D.max(axis=1) - D.min(axis=1)) / D.mean(axis=1)
    np.testing.assert_array_less(rel_spread, 1e-6)


def test_fr_growth_is_scale_dependent(camb_background, camb_linear):
    """f(R) growth must depend on k.

    `growth_factor` returns D(z, k)/D(0, k), so for f(R) — where MG enhances
    growth more on small scales (large k) — the *normalized* growth at z>0
    is **smaller** at large k than at small k.
    """
    mg = _make_mg(camb_background, camb_linear, "fr", {"fr0": 1e-5})

    zs = np.array([0.5])
    k_small = max(mg.k[0] * 1.1, 1e-3)
    k_large = min(mg.k[-1] / 1.1, 5.0)
    ks = np.array([k_small, k_large])

    D = mg.growth_factor(zs, ks)
    # The k-dependence must be clearly non-trivial (>0.1% spread).
    rel_diff = abs(D[0, 1] - D[0, 0]) / D[0, 0]
    assert rel_diff > 1e-3, f"Expected k-dependent growth in f(R), got rel_diff={rel_diff:.2e}"
    # Direction check: D(z, k_large)/D(0, k_large) < D(z, k_small)/D(0, k_small).
    assert D[0, 1] < D[0, 0]


# -------------------------------
# LCDM limit consistency
# -------------------------------


# Each entry maps to a parameter dict at (or very close to) the LCDM limit.
LCDM_LIMIT_CASES = [
    ("w0wacdm", {}),
    ("dgp", {"omega_rc": 1e-12}),
    ("gamma", {"gamma0": 0.55}),
    ("gammaz", {"gamma0": 0.55, "gamma1": 0.0}),
    ("musigma-de", {"mu0": 0.0, "sigma0": 0.0}),
    ("ds", {"xi": 0.0}),
]


@pytest.mark.parametrize("model_name,mgpars", LCDM_LIMIT_CASES, ids=[m for m, _ in LCDM_LIMIT_CASES])
def test_lcdm_limit_matches_w0wacdm_baseline(
    camb_background, camb_linear, model_name, mgpars
):
    """At the LCDM limit every model should produce the same D(z) as 'w0wacdm'."""
    mg_baseline = _make_mg(camb_background, camb_linear, "w0wacdm", {})
    mg_limit = _make_mg(camb_background, camb_linear, model_name, mgpars)

    zs = np.array([0.0, 0.3, 1.0, 2.0])
    k_ref = np.array([0.05])  # use a single scale; these models are k-independent.

    D_base = mg_baseline.growth_factor(zs, k_ref)
    D_limit = mg_limit.growth_factor(zs, k_ref)

    # Allow ~0.5% slack — Linder gamma is an approximation to the exact w0waCDM growth.
    tol = 5e-3 if model_name in {"gamma", "gammaz"} else 1e-4
    np.testing.assert_allclose(D_limit, D_base, rtol=tol)


def test_fr_reduces_to_lcdm_for_tiny_fr0(camb_background, camb_linear):
    """f(R) with fR0 → 0 must recover the scale-independent baseline growth."""
    mg_baseline = _make_mg(camb_background, camb_linear, "w0wacdm", {})
    mg_fr_tiny = _make_mg(camb_background, camb_linear, "fr", {"fr0": 1e-12})

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.array([1e-2, 1e-1, 1.0])

    D_base = mg_baseline.growth_factor(zs, ks)
    D_fr = mg_fr_tiny.growth_factor(zs, ks)

    np.testing.assert_allclose(D_fr, D_base, rtol=1e-3)


# -------------------------------
# Coupling between mgpars and outputs
# -------------------------------


def test_fr_growth_increases_with_fR0(camb_background, camb_linear):
    """Stronger f(R) should give larger growth at low z on small scales."""
    mg_weak = _make_mg(camb_background, camb_linear, "fr", {"fr0": 1e-6})
    mg_strong = _make_mg(camb_background, camb_linear, "fr", {"fr0": 1e-4})

    zs = np.array([0.0])
    k_small_scale = np.array([min(mg_strong.k[-1] / 1.1, 1.0)])

    # Compare unnormalized growth via the matter power spectrum,
    # which encodes the full D_MG(z, k) modification.
    pk_weak = mg_weak.matter_power_spectrum(zs, k_small_scale)
    pk_strong = mg_strong.matter_power_spectrum(zs, k_small_scale)
    assert pk_strong[0, 0] > pk_weak[0, 0]


def test_check_ranges_flag_flips_for_extreme_musigma(camb_background, camb_linear):
    """`check_ranges` should be flipped to False when mu0 > 2*sigma0 + 1."""
    mg_ok = _make_mg(camb_background, camb_linear, "musigma-de", {"mu0": 0.1, "sigma0": 0.1})
    mg_extreme = _make_mg(
        camb_background, camb_linear, "musigma-de", {"mu0": 2.0, "sigma0": 0.1}
    )
    assert mg_ok.check_ranges is True
    assert mg_extreme.check_ranges is False


# -------------------------------
# mu / sigma_lensing interpolators
# -------------------------------


def test_musigma_de_interpolators_are_lcdm_at_origin(camb_background, camb_linear):
    """At mu0=sigma0=0 the mu(a) and Sigma(z,k) interpolators must equal 1."""
    mg = _make_mg(camb_background, camb_linear, "musigma-de", {"mu0": 0.0, "sigma0": 0.0})

    a_samples = np.array([0.5, 0.8, 1.0])
    z_samples = np.array([0.0, 1.0])
    k_samples = np.array([0.01, 0.1])

    mu_vals = mg.mu_interp(a_samples)
    sigma_vals = mg.sigma_lensing(z_samples, k_samples)

    np.testing.assert_allclose(mu_vals, 1.0, atol=1e-10)
    np.testing.assert_allclose(sigma_vals, 1.0, atol=1e-10)


def test_mu_scaledep_interpolators_present(camb_background, camb_linear):
    """The 'mu' (mu_ak) model should expose a list of mu(a) interpolators and Sigma(z,k)."""
    mgpars = {"mu0": 0.1, "sigma0": 0.1, "c1": 0.0, "c2": 0.0, "lam": 0.0}
    mg = _make_mg(camb_background, camb_linear, "mu", mgpars)

    assert isinstance(mg.mu_interp, list)
    assert len(mg.mu_interp) == mg.k_len
    # Each entry callable on a-values.
    sample_a = np.array([0.5, 1.0])
    for interp in mg.mu_interp[:3]:
        out = interp(sample_a)
        assert out.shape == sample_a.shape
        assert np.all(np.isfinite(out))

    # Sigma_lensing should be a 2D spline over (z, k).
    sigma_vals = mg.sigma_lensing(np.array([0.0, 0.5]), np.array([0.05, 0.5]))
    assert sigma_vals.shape == (2, 2)
    assert np.all(np.isfinite(sigma_vals))
