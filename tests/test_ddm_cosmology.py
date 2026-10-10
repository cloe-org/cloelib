"""Tests for the decaying dark matter (1bDDM and 2bDDM) cosmology modules.

Most tests use the CosmoPower-JAX emulators (fast). The `use_emulator=False` code paths are
tested with a mock of `classy.Class`: a stand-in that does not run CLASS, only records the
parameters it is given and returns a made-up smooth P(k). This checks the parameters sent to
CLASS and the k grid it is evaluated on, but not the CLASS numbers themselves. The comparisons
against real CLASS runs are slow (about 1.5 minutes) and are only run when the environment
variable CLOE_RUN_SLOW_TESTS is set.
"""

import importlib.util
import inspect
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("cosmopower_jax", reason="cosmopower_jax not installed")
pytest.importorskip("classy", reason="classy not installed")

from cloelib.cosmology import class_DDM_cosmology as ddm  # noqa: E402
from cloelib.cosmology.class_DDM_cosmology import (  # noqa: E402
    obDDMBackground,
    obDDMLinearPerturbations,
    obDDMNonLinearPerturbations,
    tbDDMNonLinearPerturbations,
)
from cloelib.cosmology.class_cosmology import CLASSBackground  # noqa: E402

HAS_2B = importlib.util.find_spec("DMemu") is not None  # the 2bDDM boost emulator

requires_2b = pytest.mark.skipif(not HAS_2B, reason="DMemu not installed")
slow = pytest.mark.skipif(
    not os.environ.get("CLOE_RUN_SLOW_TESTS"),
    reason="slow (runs CLASS); set CLOE_RUN_SLOW_TESTS=1 to run",
)

# DUSTGRAIN baseline cosmology
COSMO = dict(
    H0=67.31,
    Omega_b0=0.0481,
    Omega_cdm0=0.31345 - 0.0481,
    Omega_k0=0.0,
    As=2.2e-9,
    ns=0.9658,
    mnu=0.06,
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.545,
    N_mnu=3,
    N_ur=0.00441,
)
F_DCDM = 0.4
GAMMA_TIMES_F = 0.01  # 1/Gyr
ZS = np.array([0.0, 0.1, 0.5, 1.0])
KS = np.logspace(-3, 1, 40)  # 1/Mpc
PRESCRIPTIONS = ["hmcode", "halofit"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def background_1b():
    return obDDMBackground(f_dcdm=F_DCDM, Gamma_times_f=GAMMA_TIMES_F, **COSMO)


@pytest.fixture(scope="module")
def linear_1b(background_1b):
    return obDDMLinearPerturbations(background_1b, ZS)


@pytest.fixture(scope="module")
def nl_1b(background_1b, linear_1b):
    """1bDDM non-linear perturbations with the emulators, for both prescriptions."""
    return {
        p: obDDMNonLinearPerturbations(
            background_1b, linear_1b, ZS, use_emulator=True, non_linear_lcdm=p
        )
        for p in PRESCRIPTIONS
    }


@pytest.fixture(scope="module")
def background_2b():
    return CLASSBackground(**COSMO)


@pytest.fixture(scope="module")
def nl_2b(background_2b):
    """2bDDM non-linear perturbations with the emulators, for both prescriptions."""
    return {
        p: tbDDMNonLinearPerturbations(
            background_2b,
            ZS,
            f_dcdm=F_DCDM,
            epsilon=0.01,
            Gamma=GAMMA_TIMES_F,
            use_emulator=True,
            non_linear_lcdm=p,
        )
        for p in PRESCRIPTIONS
    }


class MockClass:
    """Mock of classy.Class: runs no CLASS, records the parameters and returns a made-up Pk."""

    K_GRID = np.geomspace(1e-5, 44.5, 130)  # like CLASS's own grid
    last = None

    def __init__(self):
        self.params = {}
        self.pk_calls = []
        MockClass.last = self

    def set(self, params):
        self.params = dict(params)

    def compute(self):
        pass

    def get_pk_and_k_and_z(self, **kwargs):
        return None, MockClass.K_GRID, None

    def pk(self, k, z):
        if k > MockClass.K_GRID[-1]:  # what the real CLASS does at the edge
            raise ValueError("k out of bounds")
        self.pk_calls.append(k)
        return 1e3 * (k / 0.02) ** -1.5 / (1.0 + z)

    pk_lin = pk


# ---------------------------------------------------------------------------
# Defaults and arguments
# ---------------------------------------------------------------------------
def test_use_emulator_defaults_to_true():
    classes = [obDDMBackground, obDDMLinearPerturbations, obDDMNonLinearPerturbations]
    if HAS_2B:
        classes.append(tbDDMNonLinearPerturbations)
    for cls in classes:
        default = inspect.signature(cls).parameters["use_emulator"].default
        assert default is True, f"{cls.__name__}: use_emulator default is {default}"


def test_non_linear_lcdm_defaults_to_hmcode():
    classes = [obDDMNonLinearPerturbations]
    if HAS_2B:
        classes.append(tbDDMNonLinearPerturbations)
    for cls in classes:
        default = inspect.signature(cls).parameters["non_linear_lcdm"].default
        assert default == "hmcode"


def test_1b_default_is_emulator_hmcode(background_1b, linear_1b, nl_1b):
    default = obDDMNonLinearPerturbations(background_1b, linear_1b, ZS)
    assert default.use_emulator is True
    assert default.non_linear_lcdm == "hmcode"
    np.testing.assert_allclose(
        default.matter_power_spectrum(ZS, KS),
        nl_1b["hmcode"].matter_power_spectrum(ZS, KS),
    )


def test_1b_invalid_prescription(background_1b, linear_1b):
    with pytest.raises(ValueError, match="non_linear_lcdm"):
        obDDMNonLinearPerturbations(background_1b, linear_1b, ZS, non_linear_lcdm="foo")


def test_1b_prescription_is_case_insensitive(background_1b, linear_1b):
    perts = obDDMNonLinearPerturbations(
        background_1b, linear_1b, ZS, non_linear_lcdm="HMcode"
    )
    assert perts.non_linear_lcdm == "hmcode"


def test_1b_log10TAGN_warning_only_for_halofit(background_1b, linear_1b):
    with pytest.warns(UserWarning, match="log10TAGN is ignored"):
        obDDMNonLinearPerturbations(
            background_1b, linear_1b, ZS, log10TAGN=8.0, non_linear_lcdm="halofit"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        obDDMNonLinearPerturbations(
            background_1b, linear_1b, ZS, log10TAGN=8.0, non_linear_lcdm="hmcode"
        )
        obDDMNonLinearPerturbations(
            background_1b, linear_1b, ZS, non_linear_lcdm="halofit"
        )


# ---------------------------------------------------------------------------
# 1bDDM with the emulators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prescription", PRESCRIPTIONS)
def test_1b_emulator_power_spectrum(nl_1b, prescription):
    pk = nl_1b[prescription].matter_power_spectrum(ZS, KS)
    assert pk.shape == (len(ZS), len(KS))
    assert np.all(np.isfinite(pk)) and np.all(pk > 0)
    # structure growth: the power spectrum decreases with redshift
    assert np.all(pk[0] > pk[-1])


def test_1b_prescriptions_differ(nl_1b):
    pk_hm = nl_1b["hmcode"].matter_power_spectrum(ZS, KS)
    pk_hf = nl_1b["halofit"].matter_power_spectrum(ZS, KS)
    assert np.max(np.abs(pk_hm / pk_hf - 1)) > 0.01
    # they agree in the linear regime
    low_k = KS < 1e-2
    np.testing.assert_allclose(pk_hm[:, low_k], pk_hf[:, low_k], rtol=0.03)


@pytest.mark.parametrize("prescription", PRESCRIPTIONS)
def test_1b_ddm_suppresses_power(nl_1b, prescription):
    """The decaying DM Pk is below the equivalent LCDM one at small scales."""
    perts = nl_1b[prescription]
    k_small_scale = np.array([2.0])
    pk_ddm = perts.matter_power_spectrum(np.array([0.0]), k_small_scale)[0, 0]
    pk_lcdm = perts.Pk_int_lcdm(0.0, k_small_scale)[0, 0]
    assert pk_ddm < pk_lcdm


def test_1b_log10TAGN_only_affects_hmcode(background_1b, linear_1b):
    def pk(prescription, log10TAGN):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            perts = obDDMNonLinearPerturbations(
                background_1b,
                linear_1b,
                ZS,
                log10TAGN=log10TAGN,
                non_linear_lcdm=prescription,
            )
        return perts.matter_power_spectrum(ZS, KS)

    assert np.max(np.abs(pk("hmcode", 8.2) / pk("hmcode", 7.4) - 1)) > 0.01
    np.testing.assert_allclose(pk("halofit", 8.2), pk("halofit", 7.4))


def test_1b_linear_lcdm_emulator_is_prescription_independent(nl_1b):
    """The linear LCDM Pk does not depend on the non-linear prescription."""
    lin_hm = nl_1b["hmcode"].Pk_lin_int_lcdm(ZS, KS)
    lin_hf = nl_1b["halofit"].Pk_lin_int_lcdm(ZS, KS)
    np.testing.assert_allclose(
        lin_hm, lin_hf, rtol=1e-10
    )  # k grids differ at rounding level


def test_1b_emulators_are_downloaded_from_github(nl_1b):
    from cloelib.cosmology.cosmopower_jax_cosmology import emulator_data

    assert ddm.DDM_EMULATOR_URL.endswith("/extended/1bddm")
    for filename, url in [
        ("halofit-w0wa-3mass-nonlinear.npz", None),  # url derived from the filename
        ("w0wa-3mass-nonlinear.npz", None),
        ("w0wa-3mass-linear.npz", None),
        ("ddm-1body-neutrino-linear.npz", ddm.DDM_EMULATOR_URL),
    ]:
        assert os.path.exists(emulator_data(filename, url))


def test_1b_m_ncdm_is_the_sum_of_the_neutrino_masses(monkeypatch):
    """The emulators take the neutrino mass sum (3 degenerate species) as input."""
    monkeypatch.setattr(ddm, "Class", MockClass)  # no CLASS run needed here

    def background(**kwargs):
        cosmo = {**COSMO, **kwargs}
        return obDDMBackground(
            f_dcdm=F_DCDM, Gamma_times_f=GAMMA_TIMES_F, use_emulator=False, **cosmo
        )

    assert background(mnu=0.06, N_mnu=3).m_ncdm == pytest.approx(0.06)
    assert background(mnu=[0.02, 0.02, 0.02], N_mnu=3).m_ncdm == pytest.approx(0.06)
    assert background(mnu=0.0, N_mnu=0, N_ur=None).m_ncdm == 0.0
    with pytest.raises(ValueError, match="3 degenerate"):
        background(mnu=0.06, N_mnu=1, N_ur=None).m_ncdm
    with pytest.raises(ValueError, match="3 degenerate"):
        background(mnu=[0.01, 0.02, 0.03], N_mnu=3).m_ncdm
    with pytest.raises(ValueError, match="out of emulator range"):
        background(mnu=1.5, N_mnu=3).m_ncdm


def test_1b_emulators_depend_on_the_neutrino_mass():
    """The neutrino mass is an input of the emulators (distances, sigma8 and Pk)."""

    def emulated(mnu):
        cosmo = {**COSMO, "mnu": mnu}
        bg = obDDMBackground(f_dcdm=F_DCDM, Gamma_times_f=GAMMA_TIMES_F, **cosmo)
        lin = obDDMLinearPerturbations(bg, ZS)
        return bg, lin

    bg_light, lin_light = emulated(0.06)
    bg_heavy, lin_heavy = emulated(0.3)
    pk_light = lin_light.matter_power_spectrum(ZS, KS)
    pk_heavy = lin_heavy.matter_power_spectrum(ZS, KS)
    assert np.max(np.abs(pk_heavy / pk_light - 1)) > 0.05  # free-streaming suppression
    assert lin_heavy.sigma8_0() < lin_light.sigma8_0()
    assert not np.allclose(
        bg_light.angular_diameter_distance(ZS),
        bg_heavy.angular_diameter_distance(ZS),
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# 1bDDM, CLASS path (mocked CLASS: checks the parameters sent to CLASS)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_class_1b(monkeypatch):
    monkeypatch.setattr(ddm, "Class", MockClass)


def test_1b_class_params_hmcode(mock_class_1b, background_1b, linear_1b):
    obDDMNonLinearPerturbations(
        background_1b,
        linear_1b,
        ZS,
        use_emulator=False,
        log10TAGN=8.1,
        non_linear_lcdm="hmcode",
    )
    params = MockClass.last.params
    assert params["non linear"] == "hmcode"
    assert params["hmcode_version"] == "2020_baryonic_feedback"
    assert params["log10T_heat_hmcode"] == 8.1


def test_1b_class_params_halofit(mock_class_1b, background_1b, linear_1b):
    obDDMNonLinearPerturbations(
        background_1b, linear_1b, ZS, use_emulator=False, non_linear_lcdm="halofit"
    )
    params = MockClass.last.params
    assert params["non linear"] == "halofit"
    assert "hmcode_version" not in params
    assert "log10T_heat_hmcode" not in params


def test_1b_class_default_prescription_is_hmcode(
    mock_class_1b, background_1b, linear_1b
):
    obDDMNonLinearPerturbations(background_1b, linear_1b, ZS, use_emulator=False)
    params = MockClass.last.params
    assert params["non linear"] == "hmcode"
    assert params["log10T_heat_hmcode"] == 7.6  # default log10TAGN


def test_1b_class_params_are_equivalent_lcdm(mock_class_1b, background_1b, linear_1b):
    """The DDM parameters are removed and the total CDM is restored."""
    perts = obDDMNonLinearPerturbations(
        background_1b, linear_1b, ZS, use_emulator=False
    )
    params = MockClass.last.params
    assert "omega_ini_dcdm" not in params and "Gamma_dcdm" not in params
    assert params["omega_cdm"] == pytest.approx(perts.wdm)


# ---------------------------------------------------------------------------
# 2bDDM
# ---------------------------------------------------------------------------
@requires_2b
def test_2b_invalid_prescription(background_2b):
    with pytest.raises(ValueError, match="non_linear_lcdm"):
        tbDDMNonLinearPerturbations(
            background_2b,
            ZS,
            f_dcdm=F_DCDM,
            epsilon=0.01,
            Gamma=GAMMA_TIMES_F,
            non_linear_lcdm="foo",
        )


@requires_2b
def test_2b_default_is_emulator_hmcode(background_2b, nl_2b):
    default = tbDDMNonLinearPerturbations(
        background_2b, ZS, f_dcdm=F_DCDM, epsilon=0.01, Gamma=GAMMA_TIMES_F
    )
    assert default.use_emulator is True
    assert default.non_linear_lcdm == "hmcode"
    np.testing.assert_allclose(
        default.matter_power_spectrum(ZS, KS),
        nl_2b["hmcode"].matter_power_spectrum(ZS, KS),
    )


@requires_2b
def test_2b_log10TAGN_warning_only_for_halofit(background_2b):
    kwargs = dict(f_dcdm=F_DCDM, epsilon=0.01, Gamma=GAMMA_TIMES_F, log10TAGN=8.0)
    with pytest.warns(UserWarning, match="log10TAGN is ignored"):
        tbDDMNonLinearPerturbations(
            background_2b, ZS, non_linear_lcdm="halofit", **kwargs
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        tbDDMNonLinearPerturbations(
            background_2b, ZS, non_linear_lcdm="hmcode", **kwargs
        )


@requires_2b
@pytest.mark.parametrize("prescription", PRESCRIPTIONS)
def test_2b_emulator_power_spectrum(nl_2b, prescription):
    pk = nl_2b[prescription].matter_power_spectrum(ZS, KS)
    assert pk.shape == (len(ZS), len(KS))
    assert np.all(np.isfinite(pk)) and np.all(pk > 0)
    assert np.all(pk[0] > pk[-1])


@requires_2b
def test_2b_prescriptions_differ(nl_2b):
    pk_hm = nl_2b["hmcode"].matter_power_spectrum(ZS, KS)
    pk_hf = nl_2b["halofit"].matter_power_spectrum(ZS, KS)
    assert np.max(np.abs(pk_hm / pk_hf - 1)) > 0.01


@requires_2b
def test_2b_log10TAGN_only_affects_hmcode(background_2b):
    def pk(prescription, log10TAGN):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            perts = tbDDMNonLinearPerturbations(
                background_2b,
                ZS,
                f_dcdm=F_DCDM,
                epsilon=0.01,
                Gamma=GAMMA_TIMES_F,
                log10TAGN=log10TAGN,
                non_linear_lcdm=prescription,
            )
        return perts.matter_power_spectrum(ZS, KS)

    assert np.max(np.abs(pk("hmcode", 8.2) / pk("hmcode", 7.4) - 1)) > 0.01
    np.testing.assert_allclose(pk("halofit", 8.2), pk("halofit", 7.4))


# ---------------------------------------------------------------------------
# 2bDDM, CLASS path (mocked CLASS: checks the parameters and the k grid)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_class_2b(monkeypatch):
    monkeypatch.setattr(ddm, "Class", MockClass)


@requires_2b
def test_2b_class_params(mock_class_2b, background_2b):
    kwargs = dict(f_dcdm=F_DCDM, epsilon=0.01, Gamma=GAMMA_TIMES_F, use_emulator=False)
    tbDDMNonLinearPerturbations(
        background_2b, ZS, log10TAGN=8.1, non_linear_lcdm="hmcode", **kwargs
    )
    params = MockClass.last.params
    assert params["non linear"] == "hmcode"
    assert params["hmcode_version"] == "2020_baryonic_feedback"
    assert params["log10T_heat_hmcode"] == 8.1

    tbDDMNonLinearPerturbations(background_2b, ZS, non_linear_lcdm="halofit", **kwargs)
    params = MockClass.last.params
    assert params["non linear"] == "halofit"
    assert "hmcode_version" not in params and "log10T_heat_hmcode" not in params


@requires_2b
def test_2b_class_k_grid(mock_class_2b, background_2b):
    """CLASS is evaluated on a dense log-spaced grid, never beyond CLASS's own k range."""
    perts = tbDDMNonLinearPerturbations(
        background_2b,
        ZS,
        f_dcdm=F_DCDM,
        epsilon=0.01,
        Gamma=GAMMA_TIMES_F,
        use_emulator=False,
    )
    ks_used = np.unique(MockClass.last.pk_calls)  # sorted; evaluated for every redshift
    assert ks_used[0] == pytest.approx(MockClass.K_GRID[0])
    assert ks_used[-1] == pytest.approx(perts.kmax)
    assert ks_used[-1] <= MockClass.K_GRID[-1]
    # 100 points per decade, much denser than CLASS's own grid
    dlogk = np.diff(np.log10(ks_used))
    assert dlogk.max() == pytest.approx(0.01, rel=0.05)
    assert len(ks_used) > 5 * len(MockClass.K_GRID) / 2
    assert len(MockClass.last.pk_calls) == len(ZS) * len(ks_used)


# ---------------------------------------------------------------------------
# Comparison with real CLASS runs (slow)
# ---------------------------------------------------------------------------
@slow
@pytest.mark.parametrize("prescription", PRESCRIPTIONS)
def test_1b_emulator_vs_class(nl_1b, background_1b, linear_1b, prescription):
    perts_class = obDDMNonLinearPerturbations(
        background_1b, linear_1b, ZS, use_emulator=False, non_linear_lcdm=prescription
    )
    pk_emu = nl_1b[prescription].matter_power_spectrum(ZS, KS)
    pk_class = perts_class.matter_power_spectrum(ZS, KS)
    assert np.max(np.abs(pk_emu / pk_class - 1)) < 0.015


@requires_2b
@slow
@pytest.mark.parametrize("prescription", PRESCRIPTIONS)
def test_2b_emulator_vs_class(nl_2b, background_2b, prescription):
    perts_class = tbDDMNonLinearPerturbations(
        background_2b,
        ZS,
        f_dcdm=F_DCDM,
        epsilon=0.01,
        Gamma=GAMMA_TIMES_F,
        use_emulator=False,
        non_linear_lcdm=prescription,
    )
    pk_emu = nl_2b[prescription].matter_power_spectrum(ZS, KS)
    pk_class = perts_class.matter_power_spectrum(ZS, KS)
    assert np.max(np.abs(pk_emu / pk_class - 1)) < 0.015
