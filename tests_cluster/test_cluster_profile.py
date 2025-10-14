# import jax.numpy as np

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.observables.clusters.castro_hmf_bias import CastroHMFBias
from cloelib.observables.clusters.profile import ProfileNFW, ProfileBMO
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations


def _get_castro_hs():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    HS = HaloStatistics(perturbations, overdensity_type="vir")

    return CastroHMFBias(HS)


def test_array_shapes():

    HS_castro = _get_castro_hs()
    # Profiles
    print("# Profiles ")
    _prof_kwargs = dict(
        two_halo="None",
        offcentering=False,
        rms_off=0.0,
        f_off=0.0,
        trunc_fact=3.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )

    profile_nfw = ProfileNFW(HS_castro, **_prof_kwargs)

    R_test = np.linspace(0.01, 1.0, 9)
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.linspace(1e14, 5e14, 6)
    c_test = 4.0

    _kwargs = {"R": R_test, "z": z_test, "M": M_test}
    out_shape = (z_test.size, M_test.size, R_test.size)

    for radius_units in ("Mpc/h", "radians", "degrees", "arcmin", "arcsec"):
        print(radius_units)
        _kwargs["radius_units"] = radius_units

        _r, _z, _m = profile_nfw._surface_mass_density_args(**_kwargs)
        assert (_r * _z * _m).shape == out_shape

        _kwargs["c"] = c_test
        for bias_z in (None, np.ones((z_test.size, M_test.size))):

            _kwargs["bias_z"] = bias_z

            for two_halo in ("None", "sum", "max"):
                profile_nfw.two_halo = two_halo

                assert profile_nfw.surface_mass_density(**_kwargs).shape == out_shape
                assert (
                    profile_nfw.excess_surface_mass_density(**_kwargs).shape
                    == out_shape
                )

            _kwargs.pop("bias_z")
        _kwargs.pop("c")


def _test_profile(profile, reference_vals):

    R_test = np.array([1])
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.array([5e14])
    c_test = 4.0
    z_sources_test = np.linspace(0.6, 1, 5)
    zbin_test = 1

    print("    sigma_crit")
    assert_allclose(
        profile.sigma_crit(z_test, z_sources_test)[0], **reference_vals["sigma_crit"]
    )
    print("    n_zs_norM")
    assert_allclose(profile.n_zs_norM(z_test), **reference_vals["n_zs_norM"])
    print("    n_zs")
    assert_allclose(profile.n_zs(z_test)[0][:5], **reference_vals["n_zs"])
    print("    surface_mass_density")
    assert_allclose(
        profile.surface_mass_density(R_test, z_test, M_test, c_test)[:, 0, 0],
        **reference_vals["surface_mass_density"],
    )
    print("    excess_surface_mass_density")
    assert_allclose(
        profile.excess_surface_mass_density(R_test, z_test, M_test, c_test)[:, 0, 0],
        **reference_vals["excess_surface_mass_density"],
    )
    print("    _surface_mass_density_2h")
    assert_allclose(
        profile._surface_mass_density_2h(R_test, z_test, M_test)[:, 0, 0],
        **reference_vals["surface_mass_density_2h"],
    )
    print("    _excess_surface_mass_density_2h")
    assert_allclose(
        profile._excess_surface_mass_density_2h(R_test, z_test, M_test)[:, 0, 0],
        **reference_vals["excess_surface_mass_density_2h"],
    )


def test_profiles():
    HS_castro = _get_castro_hs()

    # Profiles
    print("# Profiles ")
    _prof_kwargs = dict(
        two_halo="None",
        offcentering=False,
        rms_off=0.0,
        f_off=0.0,
        trunc_fact=3.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )

    print("  NFW")
    profile_nfw = ProfileNFW(HS_castro, **_prof_kwargs)
    _reference_vals = {
        # All validation values have to be updated with extarnal values
        "sigma_crit": {
            "desired": [
                57269.106048,
                57134.290198,
                57034.179918,
                56956.999411,
                56895.740123,
            ],
            "rtol": 1e-5,
        },
        "n_zs_norM": {"desired": [1.04925, 1.156374, 1.424675, 2.067442], "rtol": 1e-5},
        "n_zs": {
            "desired": [
                3.445074e-01,
                3.52638e-01,
                3.6089e-01,
                3.69262e-01,
                3.77753e-01,
            ],
            "rtol": 1e-3,
        },
        "surface_mass_density": {
            "desired": [57.782346, 60.278322, 62.62685, 64.791899],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density": {
            "desired": [85.397604, 94.3122, 103.483482, 112.700292],
            "rtol": 1e-3,
        },
        "surface_mass_density_2h": {
            "desired": [13.276354, 16.761777, 20.598522, 24.711948],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density_2h": {
            "desired": [1.515384, 2.157451, 2.844042, 3.643293],
            "rtol": 1e-3,
        },
    }
    _test_profile(profile_nfw, _reference_vals)

    print("  BMO")
    _reference_vals.update(
        # All validation values have to be updated with extarnal values
        {
            "surface_mass_density": {
                "desired": [49.754672, 50.309348, 50.479286, 50.276593],
                "rtol": 1e-3,
            },
            "excess_surface_mass_density": {
                "desired": [90.134602, 99.25608, 108.537932, 117.745383],
                "rtol": 1e-3,
            },
        }
    )
    profile_bmo = ProfileBMO(HS_castro, **_prof_kwargs)
    _test_profile(profile_bmo, _reference_vals)
