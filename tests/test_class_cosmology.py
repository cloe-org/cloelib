import numpy as np
import pytest

from cloelib.cosmology.class_cosmology import (
    CLASSBackground,
    CLASSLinearPerturbations,
    CLASSNonLinearPerturbations,
)
from cloelib.cosmology.cosmology import Background, Perturbations


@pytest.fixture
def class_background_instance(scope="module"):
    """Fixture to create an instance of CLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    class_instance = CLASSBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
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
    return class_instance


def test_class_background_required_methods(class_background_instance):
    """Test that all required methods are present."""
    methods_required = {
        name
        for name, value in Background.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    methods_found = {
        name
        for name in dir(class_background_instance)
        if callable(getattr(class_background_instance, name))
        and not name.startswith("_")
    }
    assert methods_required <= methods_found


def test_class_background_required_attributes(class_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {
        name
        for name, value in Background.__dict__.items()
        if not callable(value) and not name.startswith("_")
    }
    contents = (
        (name, getattr(class_background_instance, name))
        for name in dir(class_background_instance)
    )
    attributes_found = {
        name
        for name, value in contents
        if (not callable(value) or (callable(value) and isinstance(value, float)))
        and not name.startswith("_")
    }
    assert attributes_required <= attributes_found


def test_class_background_implements_protocol(class_background_instance):
    """Test that the CLASSBackground instance adheres to the Background protocol."""
    assert isinstance(class_background_instance, Background)


def test_class_background_H0(class_background_instance):
    assert hasattr(class_background_instance, "H0")
    assert isinstance(class_background_instance.H0, float)
    assert class_background_instance.H0 == 67.7


def test_class_background_h(class_background_instance):
    assert hasattr(class_background_instance, "h")
    assert isinstance(class_background_instance.h, float)
    assert class_background_instance.h == 0.677


def test_class_background_Omega_b0(class_background_instance):
    assert hasattr(class_background_instance, "Omega_b0")
    assert isinstance(class_background_instance.Omega_b0, float)
    assert class_background_instance.Omega_b0 == 0.022 / (0.677 * 0.677)


def test_class_background_Omega_cdm0(class_background_instance):
    assert hasattr(class_background_instance, "Omega_cdm0")
    assert isinstance(class_background_instance.Omega_cdm0, float)
    assert class_background_instance.Omega_cdm0 == 0.12 / (0.677 * 0.677)


def test_class_background_mnu(class_background_instance):
    assert hasattr(class_background_instance, "mnu")
    assert isinstance(class_background_instance.mnu, float)
    assert class_background_instance.mnu == 0.0


def test_class_background_Omega_k0(class_background_instance):
    assert hasattr(class_background_instance, "Omega_k0")
    assert isinstance(class_background_instance.Omega_k0, float)
    assert class_background_instance.Omega_k0 == 0.0


def test_class_background_As(class_background_instance):
    assert hasattr(class_background_instance, "As")
    assert isinstance(class_background_instance.As, float)
    assert class_background_instance.As == 2e-9


def test_class_background_ns(class_background_instance):
    assert hasattr(class_background_instance, "ns")
    assert isinstance(class_background_instance.ns, float)
    assert class_background_instance.ns == 0.96


def test_class_background_w0(class_background_instance):
    assert hasattr(class_background_instance, "w0")
    assert isinstance(class_background_instance.w0, float)
    assert class_background_instance.w0 == -1


def test_class_background_wa(class_background_instance):
    assert hasattr(class_background_instance, "wa")
    assert isinstance(class_background_instance.wa, float)
    assert class_background_instance.wa == 0.0


def test_class_background_N_mnu(class_background_instance):
    assert hasattr(class_background_instance, "N_mnu")
    assert isinstance(class_background_instance.N_mnu, int)
    assert class_background_instance.N_mnu == 0


def test_class_background_N_ur(class_background_instance):
    assert hasattr(class_background_instance, "N_ur")
    assert isinstance(class_background_instance.N_ur, float)
    assert class_background_instance.N_ur == 3.044


def test_class_background_N_eff(class_background_instance):
    assert hasattr(class_background_instance, "N_eff")
    assert isinstance(class_background_instance.N_eff, float)
    assert class_background_instance.N_eff == 3.044


def test_set_neutrino_masses_single_float():
    bg = CLASSBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.1,
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=1,
    )
    assert bg._set_neutrino_masses() == "0.1"


def test_set_neutrino_masses_degenerate():
    bg = CLASSBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.3,
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=3,
    )
    assert bg._set_neutrino_masses() == "0.1,0.1,0.1"


def test_set_neutrino_masses_array():
    bg = CLASSBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=np.array([0.05, 0.03]),
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=2,
    )
    assert bg._set_neutrino_masses() == "0.05,0.03"


def test_set_neutrino_masses_sequence():
    bg = CLASSBackground(
        H0=67.7,
        Omega_b0=0.022 / 0.677**2,
        Omega_cdm0=0.12 / 0.677**2,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=[0.02, 0.04, 0.06],
        w0=-1,
        wa=0,
        gamma_MG=0,
        N_mnu=3,
    )
    assert bg._set_neutrino_masses() == "0.02,0.04,0.06"


def test_set_neutrino_masses_wrong_length():
    with pytest.raises(ValueError):
        bg = CLASSBackground(
            H0=67.7,
            Omega_b0=0.022 / 0.677**2,
            Omega_cdm0=0.12 / 0.677**2,
            Omega_k0=0.0,
            As=2e-9,
            ns=0.96,
            mnu=[0.02, 0.04],
            w0=-1,
            wa=0,
            gamma_MG=0,
            N_mnu=3,
        )
        bg._set_neutrino_masses()


def test_set_neutrino_masses_wrong_type():
    with pytest.raises(TypeError):
        bg = CLASSBackground(
            H0=67.7,
            Omega_b0=0.022 / 0.677**2,
            Omega_cdm0=0.12 / 0.677**2,
            Omega_k0=0.0,
            As=2e-9,
            ns=0.96,
            mnu=None,
            w0=-1,
            wa=0,
            gamma_MG=0,
            N_mnu=1,
        )
        bg._set_neutrino_masses()


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


def test_class_omega_m(class_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(class_background_instance, "Omega_m")
    assert callable(class_background_instance.Omega_m)
    result = class_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_class_omega_b(class_background_instance, zs):
    """Test Omega_b.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(class_background_instance, "Omega_b")
    assert callable(class_background_instance.Omega_b)
    result = class_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    assert np.abs(result[0] - class_background_instance.Omega_b0) < 1e-4


@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_class_hubble_parameter(class_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(class_background_instance, "hubble_parameter")
    assert callable(class_background_instance.hubble_parameter)
    result = class_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - class_background_instance.H0) < 1e-4


def test_class_comoving_distance(class_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(class_background_instance, "comoving_distance")
    assert callable(class_background_instance.comoving_distance)
    result = class_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_class_transverse_comoving_distance(class_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(class_background_instance, "transverse_comoving_distance")
    assert callable(class_background_instance.transverse_comoving_distance)
    result = class_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_class_angular_diameter_distance(class_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(class_background_instance, "angular_diameter_distance")
    assert callable(class_background_instance.angular_diameter_distance)
    result = class_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.fixture
def class_perturbation_instances(class_background_instance, zs, scope="module"):
    """Fixture to create the Linear and NonLinear instances of CLASSPerturbations."""
    class_lin = CLASSLinearPerturbations(
        background=class_background_instance, redshifts=zs
    )
    class_non = CLASSNonLinearPerturbations(
        background=class_background_instance,
        linearperturbations=None,
        redshifts=zs,
        nonlinear_model="halofit",
    )
    return {"Linear": class_lin, "NonLinear": class_non}


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_perturbation_implements_protocol(class_perturbation_instances, key):
    """Test that the CLASSPerturbation instances adhere to the protocol."""
    class_instance = class_perturbation_instances[key]
    assert isinstance(class_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_matter_power_spectrum(class_perturbation_instances, key, zs, ks):
    """Test CLASS matter_power_spectrum."""
    class_instance = class_perturbation_instances[key]
    assert hasattr(class_instance, "matter_power_spectrum")
    assert callable(class_instance.matter_power_spectrum)
    result = class_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_growth_factor(class_perturbation_instances, key, zs, ks):
    """Test CLASS growth_factor."""
    class_instance = class_perturbation_instances[key]
    assert hasattr(class_instance, "growth_factor")
    assert callable(class_instance.growth_factor)
    result = class_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear", "NonLinear"])
def test_class_growth_rate(class_perturbation_instances, key, zs, ks):
    """Test CLASS growth_rate."""
    class_instance = class_perturbation_instances[key]
    assert hasattr(class_instance, "growth_rate")
    assert callable(class_instance.growth_rate)
    result = class_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_class_sigma8_consistency_linear_vs_nonlinear(class_background_instance, zs):
    """Test that Linear and NonLinear give consistent sigma8(z=0) values."""
    class_lin = CLASSLinearPerturbations(
        background=class_background_instance, redshifts=zs
    )
    class_non = CLASSNonLinearPerturbations(
        background=class_background_instance,
        linearperturbations=None,
        redshifts=zs,
        nonlinear_model="halofit",
    )
    assert np.abs(class_lin.sigma8_0() - class_non.sigma8_0()) < 1e-3


def test_Omega_cb_returns_Om_b_plus_Om_cdm(class_background_instance):
    """
    Ensure that `Omega_cb(zs)` correctly returns the sum
    of the baryon and CDM density parameters for an array of redshifts.
    """
    import numpy as np

    # A set of redshifts (including z=0) to test the vectorised path
    zs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Reference values directly from CLASS
    Om_b = class_background_instance.results.Om_b(zs)
    Om_cdm = class_background_instance.results.Om_cdm(zs)
    expected = Om_b + Om_cdm

    # Call the wrapper under test
    omega_cb = class_background_instance.Omega_cb(zs)

    # Basic shape / type checks
    assert isinstance(omega_cb, np.ndarray)
    assert omega_cb.shape == zs.shape

    # Verify element‑wise equality to CLASS precision
    assert np.allclose(omega_cb, expected, rtol=1e-12, atol=1e-15)


###########################
# PERTURBATIONS UNIT TESTS
###########################


@pytest.fixture
def class_lin_perturb_instance(class_background_instance):
    """
    Fixture that builds a fully‑initialised CLASSLinearPerturbations instance.
    It re‑uses the existing `class_cosmo` background fixture (if you already have
    one) or creates a fresh CLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations class itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # ----- instantiate perturbations ---------------------------------
    pert = CLASSLinearPerturbations(background=class_background_instance, redshifts=zs)

    return pert


@pytest.fixture
def class_lin_perturb_instance_nu(class_background_instance):
    """
    Fixture that builds a fully‑initialised CLASSLinearPerturbations instance.
    It re‑uses the existing `class_cosmo` background fixture (if you already have
    one) or creates a fresh CLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations class itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # add neutrinos:
    class_background_instance.interface_args["CLASSparams"]["N_ncdm"] = 1
    class_background_instance.interface_args["CLASSparams"]["m_ncdm"] = 0.2
    # ----- instantiate perturbations ---------------------------------
    pert = CLASSLinearPerturbations(background=class_background_instance, redshifts=zs)

    return pert


def test_matter_power_spectrum_cb_no_neutrinos(class_lin_perturb_instance):
    """
    Verify that with N_ncdm == 0 the CB power spectrum is identical
    to the total matter power spectrum.
    """
    import warnings

    # Ensure the perturbation object is in the “no‑neutrino” configuration.
    # (The fixture may already provide this; otherwise we explicitly set it.)
    class_lin_perturb_instance.interface_args["CLASSparams"]["N_ncdm"] = 0

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 15)  # 15 k‑values spanning 10⁻³–10¹ Mpc⁻¹

    # Expected: ordinary matter power spectrum
    pk_total = class_lin_perturb_instance.matter_power_spectrum(zs, ks)

    # CB spectrum – should trigger the warning and fall back to pk_total
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pk_cb = class_lin_perturb_instance.matter_power_spectrum_cb(zs, ks)

        # Verify the warning was emitted
        assert any("no massive neutrinos" in str(warn.message) for warn in w), (
            "Expected warning about N_ncdm == 0 not raised"
        )

    # Shape and value checks
    assert pk_cb.shape == pk_total.shape, "Shape mismatch between cb and total spectra"
    assert np.allclose(pk_cb, pk_total, rtol=1e-12, atol=1e-15), (
        "CB spectrum should equal total matter spectrum when N_ncdm == 0"
    )


def test_matter_power_spectrum_cb_with_neutrinos(class_lin_perturb_instance_nu):
    """
    With massive neutrinos present, check that the CB spectrum:
      * has the correct (nz, nk) shape,
      * matches the low‑level CLASS `pk_cb` values for a few random points.
    """

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 20)

    pk_cb = class_lin_perturb_instance_nu.matter_power_spectrum_cb(zs, ks)

    # Basic shape check
    assert pk_cb.shape == (len(zs), len(ks)), "Unexpected shape for CB power spectrum"

    # Spot‑check a few random (z, k) entries against the direct CLASS call
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        i = rng.integers(0, len(zs))
        j = rng.integers(0, len(ks))
        z_test = zs[i]
        k_test = ks[j]

        # Direct CLASS low‑level call
        pk_direct = class_lin_perturb_instance_nu.results.pk_cb(k_test, z_test)  # type: ignore[union-attr]

        assert np.isclose(pk_cb[i, j], pk_direct, rtol=1e-12, atol=1e-15), (
            f"Mismatch at z={z_test}, k={k_test}"
        )


@pytest.fixture
def class_nonlin_perturb_instance(class_background_instance):
    """
    Fixture that builds a fully‑initialised CLASSLinearPerturbations instance.
    It re‑uses the existing `class_cosmo` background fixture (if you already have
    one) or creates a fresh CLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations class itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # ----- instantiate perturbations ---------------------------------
    pert = CLASSNonLinearPerturbations(
        background=class_background_instance, linearperturbations=None, redshifts=zs
    )

    return pert


@pytest.fixture
def class_nonlin_perturb_instance_nu(class_background_instance):
    """
    Fixture that builds a fully‑initialised CLASSLinearPerturbations instance.
    It re‑uses the existing `class_cosmo` background fixture (if you already have
    one) or creates a fresh CLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations class itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # add neutrinos:
    class_background_instance.interface_args["CLASSparams"]["N_ncdm"] = 1
    class_background_instance.interface_args["CLASSparams"]["m_ncdm"] = 0.2
    # ----- instantiate perturbations ---------------------------------
    pert = CLASSNonLinearPerturbations(
        background=class_background_instance, linearperturbations=None, redshifts=zs
    )

    return pert


def test_nl_matter_power_spectrum_cb_no_neutrinos(class_nonlin_perturb_instance):
    """
    Verify that with N_ncdm == 0 the CB power spectrum raises a warning!
    """
    import warnings

    # Ensure the perturbation object is in the “no‑neutrino” configuration.
    # (The fixture may already provide this; otherwise we explicitly set it.)
    class_nonlin_perturb_instance.interface_args["CLASSparams"]["N_ncdm"] = 0

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 15)  # 15 k‑values spanning 10⁻³–10¹ Mpc⁻¹

    # Expected: ordinary matter power spectrum
    class_nonlin_perturb_instance.matter_power_spectrum(zs, ks)

    # CB spectrum – should trigger the warning and fall back to pk_total
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        class_nonlin_perturb_instance.matter_power_spectrum_cb(zs, ks)

        # Verify the warning was emitted
        assert any("no massive neutrinos" in str(warn.message) for warn in w), (
            "Expected warning about N_ncdm == 0 not raised"
        )


def test_nl_matter_power_spectrum_cb_with_neutrinos(class_nonlin_perturb_instance_nu):
    """
    With massive neutrinos present, check that the CB spectrum:
      * has the correct (nz, nk) shape,
      * matches the low‑level CLASS `pk_cb` values for a few random points.
    """

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 20)

    pk_cb = class_nonlin_perturb_instance_nu.matter_power_spectrum_cb(zs, ks)

    # Basic shape check
    assert pk_cb.shape == (len(zs), len(ks)), "Unexpected shape for CB power spectrum"

    # Spot‑check a few random (z, k) entries against the direct CLASS call
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        i = rng.integers(0, len(zs))
        j = rng.integers(0, len(ks))
        z_test = zs[i]
        k_test = ks[j]

        # Direct CLASS low‑level call
        pk_direct = class_nonlin_perturb_instance_nu.results.pk_cb(k_test, z_test)  # type: ignore[union-attr]

        assert np.isclose(pk_cb[i, j], pk_direct, rtol=1e-12, atol=1e-15), (
            f"Mismatch at z={z_test}, k={k_test}"
        )
