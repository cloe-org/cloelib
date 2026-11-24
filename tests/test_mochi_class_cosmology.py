import numpy as np
import pytest

from cloelib.cosmology.mochi_class_cosmology import (
    mochiCLASSBackground,
    mochiCLASSLinearPerturbations,
)
from cloelib.cosmology.cosmology import Background, Perturbations


@pytest.fixture
def mochiCLASS_background_instance(scope="module"):
    """Fixture to create an instance of mochiCLASSBackground."""
    H0 = 67.7
    h = H0 / 100.0
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    mochiCLASS_instance = mochiCLASSBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=0,
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    return mochiCLASS_instance


def test_mochiCLASS_background_required_methods():
    """Test that all required methods are present."""
    methods_required = {
        name
        for name, value in Background.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    contents = mochiCLASSBackground.__dict__.items()
    methods_found = {
        name for name, value in contents if callable(value) and not name.startswith("_")
    }
    assert methods_required <= methods_found


def test_mochiCLASS_background_required_attributes(mochiCLASS_background_instance):
    """Test that all required attributes are present."""
    attributes_required = {
        name
        for name, value in Background.__dict__.items()
        if not callable(value) and not name.startswith("_")
    }
    contents = (
        (name, getattr(mochiCLASS_background_instance, name))
        for name in dir(mochiCLASS_background_instance)
    )
    attributes_found = {
        name
        for name, value in contents
        if (not callable(value) or (callable(value) and isinstance(value, float)))
        and not name.startswith("_")
    }
    assert attributes_required <= attributes_found


def test_mochiCLASS_background_implements_protocol(mochiCLASS_background_instance):
    """Test that the mochiCLASSBackground instance adheres to the Background protocol."""
    assert isinstance(mochiCLASS_background_instance, Background)


def test_mochiCLASS_background_H0(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "H0")
    assert isinstance(mochiCLASS_background_instance.H0, float)
    assert mochiCLASS_background_instance.H0 == 67.7


def test_mochiCLASS_background_h(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "h")
    assert isinstance(mochiCLASS_background_instance.h, float)
    assert mochiCLASS_background_instance.h == 0.677


def test_mochiCLASS_background_Omega_b0(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "Omega_b0")
    assert isinstance(mochiCLASS_background_instance.Omega_b0, float)
    assert mochiCLASS_background_instance.Omega_b0 == 0.022 / (0.677 * 0.677)


def test_mochiCLASS_background_Omega_cdm0(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "Omega_cdm0")
    assert isinstance(mochiCLASS_background_instance.Omega_cdm0, float)
    assert mochiCLASS_background_instance.Omega_cdm0 == 0.12 / (0.677 * 0.677)


def test_mochiCLASS_background_mnu(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "mnu")
    assert isinstance(mochiCLASS_background_instance.mnu, float)
    assert mochiCLASS_background_instance.mnu == 0.0


def test_mochiCLASS_background_Omega_k0(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "Omega_k0")
    assert isinstance(mochiCLASS_background_instance.Omega_k0, float)
    assert mochiCLASS_background_instance.Omega_k0 == 0.0


def test_mochiCLASS_background_As(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "As")
    assert isinstance(mochiCLASS_background_instance.As, float)
    assert mochiCLASS_background_instance.As == 2e-9


def test_mochiCLASS_background_ns(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "ns")
    assert isinstance(mochiCLASS_background_instance.ns, float)
    assert mochiCLASS_background_instance.ns == 0.96


def test_mochiCLASS_background_w0(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "w0")
    assert isinstance(mochiCLASS_background_instance.w0, float)
    assert mochiCLASS_background_instance.w0 == -1


def test_mochiCLASS_background_wa(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "wa")
    assert isinstance(mochiCLASS_background_instance.wa, float)
    assert mochiCLASS_background_instance.wa == 0.0


def test_mochiCLASS_background_N_mnu(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "N_mnu")
    assert isinstance(mochiCLASS_background_instance.N_mnu, int)
    assert mochiCLASS_background_instance.N_mnu == 0


def test_mochiCLASS_background_N_ur(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "N_ur")
    assert isinstance(mochiCLASS_background_instance.N_ur, float)
    assert mochiCLASS_background_instance.N_ur == 3.044


def test_mochiCLASS_background_N_eff(mochiCLASS_background_instance):
    assert hasattr(mochiCLASS_background_instance, "N_eff")
    assert isinstance(mochiCLASS_background_instance.N_eff, float)
    assert mochiCLASS_background_instance.N_eff == 3.044


def test_set_neutrino_masses_single_float():
    bg = mochiCLASSBackground(
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
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    assert bg._set_neutrino_masses() == "0.1"


def test_set_neutrino_masses_degenerate():
    bg = mochiCLASSBackground(
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
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    assert bg._set_neutrino_masses() == "0.1,0.1,0.1"


def test_set_neutrino_masses_array():
    bg = mochiCLASSBackground(
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
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    assert bg._set_neutrino_masses() == "0.05,0.03"


def test_set_neutrino_masses_sequence():
    bg = mochiCLASSBackground(
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
        mg_stable_basis_on=False,
        stable_MG_dict={},
        mg_background_model="lcdm",
    )
    assert bg._set_neutrino_masses() == "0.02,0.04,0.06"


def test_set_neutrino_masses_wrong_length():
    with pytest.raises(ValueError):
        bg = mochiCLASSBackground(
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
            mg_stable_basis_on=False,
            stable_MG_dict={},
            mg_background_model="lcdm",
        )
        bg._set_neutrino_masses()


def test_set_neutrino_masses_wrong_type():
    with pytest.raises(TypeError):
        bg = mochiCLASSBackground(
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
            mg_stable_basis_on=False,
            stable_MG_dict={},
            mg_background_model="lcdm",
        )
        bg._set_neutrino_masses()


@pytest.fixture
def zs(scope="module"):
    return np.linspace(0, 2, 20)


def test_mochiCLASS_omega_m(mochiCLASS_background_instance, zs):
    """Test Omega_m returns an np.ndarray object of correct size."""
    assert hasattr(mochiCLASS_background_instance, "Omega_m")
    assert callable(mochiCLASS_background_instance.Omega_m)
    result = mochiCLASS_background_instance.Omega_m(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mochiCLASS_omega_b(mochiCLASS_background_instance, zs):
    """Test Omega_b.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to Omega_b0.
    """
    assert hasattr(mochiCLASS_background_instance, "Omega_b")
    assert callable(mochiCLASS_background_instance.Omega_b)
    result = mochiCLASS_background_instance.Omega_b(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    assert np.abs(result[0] - mochiCLASS_background_instance.Omega_b0) < 1e-4


@pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
def test_mochiCLASS_hubble_parameter(mochiCLASS_background_instance, zs, units):
    """
    Test hubble_parameter.

    Check the method returns a np.ndarray of correct size,
    and at redshift zero the value is almost equal to H0.
    """
    assert hasattr(mochiCLASS_background_instance, "hubble_parameter")
    assert callable(mochiCLASS_background_instance.hubble_parameter)
    result = mochiCLASS_background_instance.hubble_parameter(zs, units)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)
    if units == "km/s/Mpc":
        assert np.abs(result[0] - mochiCLASS_background_instance.H0) < 1e-4


def test_mochiCLASS_comoving_distance(mochiCLASS_background_instance, zs):
    """Test comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(mochiCLASS_background_instance, "comoving_distance")
    assert callable(mochiCLASS_background_instance.comoving_distance)
    result = mochiCLASS_background_instance.comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mochiCLASS_transverse_comoving_distance(mochiCLASS_background_instance, zs):
    """Test transverse_comoving_distance returns a np.ndarray of correct size."""
    assert hasattr(mochiCLASS_background_instance, "transverse_comoving_distance")
    assert callable(mochiCLASS_background_instance.transverse_comoving_distance)
    result = mochiCLASS_background_instance.transverse_comoving_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


def test_mochiCLASS_angular_diameter_distance(mochiCLASS_background_instance, zs):
    """Test angular_diameter_distance returns a np.ndarray of correct size."""
    assert hasattr(mochiCLASS_background_instance, "angular_diameter_distance")
    assert callable(mochiCLASS_background_instance.angular_diameter_distance)
    result = mochiCLASS_background_instance.angular_diameter_distance(zs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == len(zs)


@pytest.fixture
def mochiCLASS_perturbation_instances(
    mochiCLASS_background_instance, zs, scope="module"
):
    """Fixture to create the Linear instance of mochiCLASSPerturbations."""
    mochiCLASS_lin = mochiCLASSLinearPerturbations(
        background=mochiCLASS_background_instance, redshifts=zs
    )
    return {"Linear": mochiCLASS_lin}


@pytest.mark.parametrize("key", ["Linear"])
def test_mochiCLASS_perturbation_implements_protocol(
    mochiCLASS_perturbation_instances, key
):
    """Test that the mochiCLASSPerturbation instances adhere to the protocol."""
    mochiCLASS_instance = mochiCLASS_perturbation_instances[key]
    mochiCLASS_instance = mochiCLASS_perturbation_instances["Linear"]
    print([f for f in dir(mochiCLASS_instance) if not f.startswith("_")])
    # print(mochiCLASS_instance)
    # print(key)
    assert isinstance(mochiCLASS_instance, Perturbations)


@pytest.fixture
def ks(scope="module"):
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.mark.parametrize("key", ["Linear"])
def test_mochiCLASS_matter_power_spectrum(
    mochiCLASS_perturbation_instances, key, zs, ks
):
    """Test mochiCLASS matter_power_spectrum."""
    mochiCLASS_instance = mochiCLASS_perturbation_instances[key]
    assert hasattr(mochiCLASS_instance, "matter_power_spectrum")
    assert callable(mochiCLASS_instance.matter_power_spectrum)
    result = mochiCLASS_instance.matter_power_spectrum(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear"])
def test_mochiCLASS_growth_factor(mochiCLASS_perturbation_instances, key, zs, ks):
    """Test mochiCLASS growth_factor."""
    mochiCLASS_instance = mochiCLASS_perturbation_instances[key]
    assert hasattr(mochiCLASS_instance, "growth_factor")
    assert callable(mochiCLASS_instance.growth_factor)
    result = mochiCLASS_instance.growth_factor(zs, ks)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (len(zs), len(ks))


@pytest.mark.parametrize("key", ["Linear"])
def test_mochiCLASS_growth_rate(mochiCLASS_perturbation_instances, key, zs, ks):
    """Test mochiCLASS growth_rate."""
    mochiCLASS_instance = mochiCLASS_perturbation_instances[key]
    assert hasattr(mochiCLASS_instance, "growth_rate")
    assert callable(mochiCLASS_instance.growth_rate)
    result = mochiCLASS_instance.growth_rate()
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_Omega_cb_returns_Om_b_plus_Om_cdm(mochiCLASS_background_instance):
    """
    Ensure that `Omega_cb(zs)` correctly returns the sum
    of the baryon and CDM density parameters for an array of redshifts.
    """
    import numpy as np

    # A set of redshifts (including z=0) to test the vectorised path
    zs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Reference values directly from mochiCLASS
    Om_b = mochiCLASS_background_instance.results.Om_b(zs)
    Om_cdm = mochiCLASS_background_instance.results.Om_cdm(zs)
    expected = Om_b + Om_cdm

    # Call the wrapper under test
    omega_cb = mochiCLASS_background_instance.Omega_cb(zs)

    # Basic shape / type checks
    assert isinstance(omega_cb, np.ndarray)
    assert omega_cb.shape == zs.shape

    # Verify element‑wise equality to mochiCLASS precision
    assert np.allclose(omega_cb, expected, rtol=1e-12, atol=1e-15)


###########################
# PERTURBATIONS UNIT TESTS
###########################


@pytest.fixture
def mochiCLASS_lin_perturb_instance(mochiCLASS_background_instance):
    """
    Fixture that builds a fully‑initialised mochiCLASSLinearPerturbations instance.
    It re‑uses the existing `mochiCLASS_cosmo` background fixture (if you already have
    one) or creates a fresh mochiCLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations mochiCLASS itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # ----- instantiate perturbations ---------------------------------
    pert = mochiCLASSLinearPerturbations(
        background=mochiCLASS_background_instance, redshifts=zs
    )

    return pert


@pytest.fixture
def mochiCLASS_lin_perturb_instance_nu(mochiCLASS_background_instance):
    """
    Fixture that builds a fully‑initialised mochiCLASSLinearPerturbations instance.
    It re‑uses the existing `mochiCLASS_cosmo` background fixture (if you already have
    one) or creates a fresh mochiCLASSCosmology object with default parameters.
    """
    # ----- redshift & k grid -----------------------------------------
    zs = np.array([0.0, 0.5, 1.0])  # a few test redshifts
    # The perturbations mochiCLASS itself will set its own k‑grid, so we just
    # pass the redshifts here.

    # add neutrinos:
    mochiCLASS_background_instance.interface_args["CLASSparams"]["N_ncdm"] = 1
    mochiCLASS_background_instance.interface_args["CLASSparams"]["m_ncdm"] = 0.2
    # ----- instantiate perturbations ---------------------------------
    pert = mochiCLASSLinearPerturbations(
        background=mochiCLASS_background_instance, redshifts=zs
    )

    return pert


def test_matter_power_spectrum_cb_no_neutrinos(mochiCLASS_lin_perturb_instance):
    """
    Verify that with N_ncdm == 0 the CB power spectrum is identical
    to the total matter power spectrum.
    """
    import warnings

    # Ensure the perturbation object is in the “no‑neutrino” configuration.
    # (The fixture may already provide this; otherwise we explicitly set it.)
    mochiCLASS_lin_perturb_instance.interface_args["CLASSparams"]["N_ncdm"] = 0

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 15)  # 15 k‑values spanning 10⁻³–10¹ Mpc⁻¹

    # Expected: ordinary matter power spectrum
    pk_total = mochiCLASS_lin_perturb_instance.matter_power_spectrum(zs, ks)

    # CB spectrum – should trigger the warning and fall back to pk_total
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pk_cb = mochiCLASS_lin_perturb_instance.matter_power_spectrum_cb(zs, ks)

        # Verify the warning was emitted
        assert any("no massive neutrinos" in str(warn.message) for warn in w), (
            "Expected warning about N_ncdm == 0 not raised"
        )

    # Shape and value checks
    assert pk_cb.shape == pk_total.shape, "Shape mismatch between cb and total spectra"
    assert np.allclose(pk_cb, pk_total, rtol=1e-12, atol=1e-15), (
        "CB spectrum should equal total matter spectrum when N_ncdm == 0"
    )


def test_matter_power_spectrum_cb_with_neutrinos(mochiCLASS_lin_perturb_instance_nu):
    """
    With massive neutrinos present, check that the CB spectrum:
      * has the correct (nz, nk) shape,
      * matches the low‑level mochiCLASS `pk_cb` values for a few random points.
    """

    zs = np.array([0.0, 0.5, 1.0])
    ks = np.logspace(-3, 1, 20)

    pk_cb = mochiCLASS_lin_perturb_instance_nu.matter_power_spectrum_cb(zs, ks)

    # Basic shape check
    assert pk_cb.shape == (len(zs), len(ks)), "Unexpected shape for CB power spectrum"

    # Spot‑check a few random (z, k) entries against the direct mochiCLASS call
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        i = rng.integers(0, len(zs))
        j = rng.integers(0, len(ks))
        z_test = zs[i]
        k_test = ks[j]

        # Direct mochiCLASS low‑level call
        pk_direct = mochiCLASS_lin_perturb_instance_nu.results.pk_cb(k_test, z_test)  # type: ignore[union-attr]

        assert np.isclose(pk_cb[i, j], pk_direct, rtol=1e-12, atol=1e-15), (
            f"Mismatch at z={z_test}, k={k_test}"
        )
