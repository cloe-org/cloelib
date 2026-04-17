import pytest
import numpy as np

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.mgclass_cosmology import (
    MGCLASSBackground,
    MGCLASSLinearPerturbations,
    MGCLASSNonLinearPerturbations,
)


@pytest.fixture
def mgclass_background_instance(scope="module"):
    """Fixture to create an instance of MGCLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    mgclass_instance = MGCLASSBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        Y_He=0.25,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        mg_ansatz="plk_musigma_norm_late",
        mg_z_init=0.0,
        mg_params={"mg_E11": 0.0, "mg_E22": 0.0},
        N_mnu=0,
    )
    return mgclass_instance


def test_mgclass_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {
        name
        for name, value in Background.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    contents = MGCLASSBackground.__dict__.items()
    methods_found = {
        name for name, value in contents if callable(value) and not name.startswith("_")
    }
    assert methods_required <= methods_found


def test_mgclass_background_required_attributes(mgclass_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {
        name
        for name, value in Background.__dict__.items()
        if not callable(value) and not name.startswith("_")
    }
    contents = (
        (name, getattr(mgclass_background_instance, name))
        for name in dir(mgclass_background_instance)
    )
    attributes_found = {
        name
        for name, value in contents
        if not callable(value) and not name.startswith("_")
    }
    assert attributes_required <= attributes_found


def test_mgclass_background_implements_protocol(mgclass_background_instance):
    """Test that the MGCLASSBackground instance adheres to the Background protocol."""
    assert isinstance(mgclass_background_instance, Background)


def test_mgclass_background_H0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "H0")
    assert isinstance(mgclass_background_instance.H0, float)
    assert mgclass_background_instance.H0 == 67.7


def test_mgclass_background_h(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "h")
    assert isinstance(mgclass_background_instance.h, float)
    assert mgclass_background_instance.h == 0.677


def test_mgclass_background_Omega_b0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "Omega_b0")
    assert isinstance(mgclass_background_instance.Omega_b0, float)
    assert mgclass_background_instance.Omega_b0 == 0.022 / (0.677 * 0.677)


def test_mgclass_background_Omega_cdm0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "Omega_cdm0")
    assert isinstance(mgclass_background_instance.Omega_cdm0, float)
    assert mgclass_background_instance.Omega_cdm0 == 0.12 / (0.677 * 0.677)


def test_mgclass_background_mnu(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "mnu")
    assert isinstance(mgclass_background_instance.mnu, float)
    assert mgclass_background_instance.mnu == 0.0


def test_mgclass_background_Omega_k0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "Omega_k0")
    assert isinstance(mgclass_background_instance.Omega_k0, float)
    assert mgclass_background_instance.Omega_k0 == 0.0


def test_mgclass_background_As(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "As")
    assert isinstance(mgclass_background_instance.As, float)
    assert mgclass_background_instance.As == 2e-9


def test_mgclass_background_ns(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "ns")
    assert isinstance(mgclass_background_instance.ns, float)
    assert mgclass_background_instance.ns == 0.96


def test_mgclass_background_w0(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "w0")
    assert isinstance(mgclass_background_instance.w0, float)
    assert mgclass_background_instance.w0 == -1


def test_mgclass_background_wa(mgclass_background_instance):
    assert hasattr(mgclass_background_instance, "wa")
    assert isinstance(mgclass_background_instance.wa, float)
    assert mgclass_background_instance.wa == 0.0


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


def test_mgclass_omega_m(mgclass_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(mgclass_background_instance, "Omega_m")
    assert callable(mgclass_background_instance.Omega_m)
    result = mgclass_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_class_omega_b(mgclass_background_instance, zs):
    """Test Omega_b.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(mgclass_background_instance, "Omega_b")
    assert callable(mgclass_background_instance.Omega_b)
    result = mgclass_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    assert np.abs(result[0] - mgclass_background_instance.Omega_b0) < 1e-4


@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_mgclass_hubble_parameter(mgclass_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(mgclass_background_instance, "hubble_parameter")
    assert callable(mgclass_background_instance.hubble_parameter)
    result = mgclass_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - mgclass_background_instance.H0) < 1e-4


def test_mgclass_comoving_distance(mgclass_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(mgclass_background_instance, "comoving_distance")
    assert callable(mgclass_background_instance.comoving_distance)
    result = mgclass_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mgclass_transverse_comoving_distance(mgclass_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(mgclass_background_instance, "transverse_comoving_distance")
    assert callable(mgclass_background_instance.transverse_comoving_distance)
    result = mgclass_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mgclass_angular_diameter_distance(mgclass_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(mgclass_background_instance, "angular_diameter_distance")
    assert callable(mgclass_background_instance.angular_diameter_distance)
    result = mgclass_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.fixture
def mgclass_perturbation_instances(mgclass_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of MGCLASSPerturbations."""
    mgclass_lin = MGCLASSLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )
    mgclass_non = MGCLASSNonLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs, nonlinear_model="halofit"
    )
    return {"Linear": mgclass_lin, "NonLinear": mgclass_non}


@pytest.mark.parametrize("key", ["Linear"])
def test_mgclass_perturbation_implements_protocol(mgclass_perturbation_instances, key):
    """Test that the MGCLASSPerturbation instances adhere to the protocol."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert isinstance(mgclass_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_mgclass_matter_power_spectrum(mgclass_perturbation_instances, key, zs, ks):
    """Test MGCLASS matter_power_spectrum."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert hasattr(mgclass_instance, "matter_power_spectrum")
    assert callable(mgclass_instance.matter_power_spectrum)
    result = mgclass_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_mgclass_growth_factor(mgclass_perturbation_instances, key, zs, ks):
    """Test MGCLASS growth_factor."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert hasattr(mgclass_instance, "growth_factor")
    assert callable(mgclass_instance.growth_factor)
    result = mgclass_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear"])
def test_mgclass_growth_rate(mgclass_perturbation_instances, key, zs, ks):
    """Test MGCLASS growth_rate."""
    mgclass_instance = mgclass_perturbation_instances[key]
    assert hasattr(mgclass_instance, "growth_rate")
    assert callable(mgclass_instance.growth_rate)
    result = mgclass_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_mgclass_sigma8_consistency_linear_vs_nonlinear(
    mgclass_background_instance, zs
):
    """Test that Linear and NonLinear give consistent sigma8(z=0) values."""
    mgclass_lin = MGCLASSLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )
    mgclass_non = MGCLASSNonLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs, nonlinear_model="halofit"
    )
    assert np.abs(mgclass_lin.sigma8_0() - mgclass_non.sigma8_0()) < 1e-3


def test_Omega_cb_returns_Om_b_plus_Om_cdm(mgclass_background_instance):
    """
    Ensure that `Omega_cb(zs)` correctly returns the sum
    of the baryon and CDM density parameters for an array of redshifts.
    """
    import numpy as np

    # A set of redshifts (including z=0) to test the vectorised path
    zs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Reference values from MGCLASS
    Om_b = mgclass_background_instance.results.Omega_b()
    Om_cdm = mgclass_background_instance.results.Omega0_cdm()
    Ez = mgclass_background_instance.hubble_parameter(zs) / mgclass_background_instance.H0
    expected = (Om_b + Om_cdm) * (1.0 + zs) ** 3.0 / Ez**0.5

    # Call the wrapper under test
    omega_cb = mgclass_background_instance.Omega_cb(zs)

    # Basic shape / type checks
    assert isinstance(omega_cb, np.ndarray)
    assert omega_cb.shape == zs.shape

    # Verify element‑wise equality to MGCLASS precision
    assert np.allclose(omega_cb, expected, rtol=1e-12, atol=1e-15)


###########################
# PERTURBATIONS UNIT TESTS
###########################


@pytest.fixture
def mgclass_lin_perturb_instance(mgclass_background_instance):
    """
    Fixture that builds a fully‑initialised MGCLASSLinearPerturbations instance.
    It re‑uses the existing `mgclass_cosmo` background fixture (if you already have
    one) or creates a fresh MGCLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations mgclass itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # ----- instantiate perturbations ---------------------------------
    pert = MGCLASSLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )

    return pert


@pytest.fixture
def mgclass_lin_perturb_instance_nu(mgclass_background_instance):
    """
    Fixture that builds a fully‑initialised MGCLASSLinearPerturbations instance.
    It re‑uses the existing `mgclass_cosmo` background fixture (if you already have
    one) or creates a fresh MGCLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations mgclass itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # add neutrinos:
    mgclass_background_instance.interface_args["MGCLASSparams"]["N_ncdm"] = 1
    mgclass_background_instance.interface_args["MGCLASSparams"]["m_ncdm"] = 0.2
    # ----- instantiate perturbations ---------------------------------
    pert = MGCLASSLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )

    return pert


def test_matter_power_spectrum_cb_no_neutrinos(mgclass_lin_perturb_instance):
    """
    Verify that with N_ncdm == 0 the CB power spectrum is identical
    to the total matter power spectrum.
    """
    import warnings

    # Ensure the perturbation object is in the “no‑neutrino” configuration.
    # (The fixture may already provide this; otherwise we explicitly set it.)
    mgclass_lin_perturb_instance.interface_args["MGCLASSparams"]["N_ncdm"] = 0

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 15)  # 15 k‑values spanning 10⁻³–10¹ Mpc⁻¹

    # Expected: ordinary matter power spectrum
    pk_total = mgclass_lin_perturb_instance.matter_power_spectrum(zs, ks)

    # CB spectrum – should trigger the warning and fall back to pk_total
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pk_cb = mgclass_lin_perturb_instance.matter_power_spectrum_cb(zs, ks)

        # Verify the warning was emitted
        assert any("no massive neutrinos" in str(warn.message) for warn in w), (
            "Expected warning about N_ncdm == 0 not raised"
        )

    # Shape and value checks
    assert pk_cb.shape == pk_total.shape, "Shape mismatch between cb and total spectra"
    assert np.allclose(pk_cb, pk_total, rtol=1e-12, atol=1e-15), (
        "CB spectrum should equal total matter spectrum when N_ncdm == 0"
    )


def test_matter_power_spectrum_cb_with_neutrinos(mgclass_lin_perturb_instance_nu):
    """
    With massive neutrinos present, check that the CB spectrum:
      * has the correct (nz, nk) shape,
      * matches the low‑level MGCLASS `pk_cb` values for a few random points.
    """

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 20)

    pk_cb = mgclass_lin_perturb_instance_nu.matter_power_spectrum_cb(zs, ks)

    # Basic shape check
    assert pk_cb.shape == (len(zs), len(ks)), "Unexpected shape for CB power spectrum"

    # Spot‑check a few random (z, k) entries against the direct MGCLASS call
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        i = rng.integers(0, len(zs))
        j = rng.integers(0, len(ks))
        z_test = zs[i]
        k_test = ks[j]

        # Direct MGCLASS low‑level call
        pk_direct = mgclass_lin_perturb_instance_nu.results.pk_cb(k_test, z_test)  # type: ignore[union-attr]

        assert np.isclose(pk_cb[i, j], pk_direct, rtol=1e-12, atol=1e-15), (
            f"Mismatch at z={z_test}, k={k_test}"
        )


@pytest.fixture
def mgclass_nonlin_perturb_instance(mgclass_background_instance):
    """
    Fixture that builds a fully‑initialised MGCLASSLinearPerturbations instance.
    It re‑uses the existing `mgclass_cosmo` background fixture (if you already have
    one) or creates a fresh MGCLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations mgclass itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # ----- instantiate perturbations ---------------------------------
    pert = MGCLASSNonLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )

    return pert


@pytest.fixture
def mgclass_nonlin_perturb_instance_nu(mgclass_background_instance):
    """
    Fixture that builds a fully‑initialised MGCLASSLinearPerturbations instance.
    It re‑uses the existing `mgclass_cosmo` background fixture (if you already have
    one) or creates a fresh MGCLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations mgclass itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # add neutrinos:
    mgclass_background_instance.interface_args["MGCLASSparams"]["N_ncdm"] = 1
    mgclass_background_instance.interface_args["MGCLASSparams"]["m_ncdm"] = 0.2
    # ----- instantiate perturbations ---------------------------------
    pert = MGCLASSNonLinearPerturbations(
        background=mgclass_background_instance, redshifts=zs
    )

    return pert


def test_nl_matter_power_spectrum_cb_no_neutrinos(mgclass_nonlin_perturb_instance):
    """
    Verify that with N_ncdm == 0 the CB power spectrum raises a warning!
    """
    import warnings

    # Ensure the perturbation object is in the “no‑neutrino” configuration.
    # (The fixture may already provide this; otherwise we explicitly set it.)
    mgclass_nonlin_perturb_instance.interface_args["MGCLASSparams"]["N_ncdm"] = 0

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 15)  # 15 k‑values spanning 10⁻³–10¹ Mpc⁻¹

    # Expected: ordinary matter power spectrum
    mgclass_nonlin_perturb_instance.matter_power_spectrum(zs, ks)

    # CB spectrum – should trigger the warning and fall back to pk_total
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mgclass_nonlin_perturb_instance.matter_power_spectrum_cb(zs, ks)

        # Verify the warning was emitted
        assert any("no massive neutrinos" in str(warn.message) for warn in w), (
            "Expected warning about N_ncdm == 0 not raised"
        )


def test_nl_matter_power_spectrum_cb_with_neutrinos(mgclass_nonlin_perturb_instance_nu):
    """
    With massive neutrinos present, check that the CB spectrum:
      * has the correct (nz, nk) shape,
      * matches the low‑level MGCLASS `pk_cb` values for a few random points.
    """

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 20)

    pk_cb = mgclass_nonlin_perturb_instance_nu.matter_power_spectrum_cb(zs, ks)

    # Basic shape check
    assert pk_cb.shape == (len(zs), len(ks)), "Unexpected shape for CB power spectrum"

    # Spot‑check a few random (z, k) entries against the direct MGCLASS call
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        i = rng.integers(0, len(zs))
        j = rng.integers(0, len(ks))
        z_test = zs[i]
        k_test = ks[j]

        # Direct MGCLASS low‑level call
        pk_direct = mgclass_nonlin_perturb_instance_nu.results.pk_cb(k_test, z_test)  # type: ignore[union-attr]

        assert np.isclose(pk_cb[i, j], pk_direct, rtol=1e-12, atol=1e-15), (
            f"Mismatch at z={z_test}, k={k_test}"
        )
