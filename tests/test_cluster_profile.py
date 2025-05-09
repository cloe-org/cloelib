#import jax.numpy as np

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.observables.clusters.halo_statistics import (
    HaloStatisticsCastro,
)
from cloelib.observables.clusters.profile import ProfileNFW, ProfileBMO
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations


def _test_profile(profile, reference_vals):

    R_test = 1
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = 5e14
    M_test_arr = np.array([5e14])
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
        profile.surface_mass_density(
            R_test, z_test, c_test, M_test, force_no_2h=False, force_no_off=False
        )[0],
        **reference_vals["surface_mass_density"], 
    )
    print("    excess_surface_mass_density")
    assert_allclose(
        profile.excess_surface_mass_density(R_test, z_test, c_test, M_test)[0],
        **reference_vals["excess_surface_mass_density"], 
    )
    print("    surface_mass_density_2h")
    assert_allclose(
        profile.surface_mass_density_2h(R_test, z_test, M_test_arr)[:, 0],
        **reference_vals["surface_mass_density_2h"], 
    )
    print("    excess_surface_mass_density_2h")
    assert_allclose(
        profile.excess_surface_mass_density_2h(R_test, z_test, M_test_arr)[:, 0],
        **reference_vals["excess_surface_mass_density_2h"], 
    )


def test_profiles():
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

    HS_castro = HaloStatisticsCastro(perturbations, "vir")

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
                57267.891792,
                57133.08175,
                57032.977397,
                56955.802944,
                56894.549792,
            ],
            "rtol": 1e-5,
        },
        "n_zs_norM": {"desired": [1.04925, 1.156374, 1.424675, 2.067442], "rtol": 1e-5},
        "n_zs": {
            "desired": [
                3.445074e-01,
                4.309865e-01,
                5.286852e-01,
                6.359309e-01,
                7.500898e-01,
            ],
            "rtol": 1e-5,
        },
        "surface_mass_density": {
            "desired": [85.326055, 86.91781, 88.153822, 89.082218],
            "rtol": 1e-5,
        },
        "excess_surface_mass_density": {
            "desired": [126.057512, 131.59368, 136.043568, 139.476555],
            "rtol": 1e-5,
        },
        "surface_mass_density_2h": {
            "desired": [2.247808, 3.142143, 4.20271, 5.433065],
            "rtol": 1e-5,
        },
        "excess_surface_mass_density_2h": {
            "desired": [19.611728, 24.812692, 30.431719, 36.436455],
            "rtol": 1e-5,
        },
    }
    _test_profile(profile_nfw, _reference_vals)

    print("  BMO")
    _reference_vals.update(
        # All validation values have to be updated with extarnal values
        {
            "surface_mass_density": {
                "desired": [73.485728, 73.904413, 74.163978, 74.31884],
                "rtol": 1e-5,
            },
            "excess_surface_mass_density": {
                "desired": [133.052194, 138.733475, 143.283091, 146.782184],
                "rtol": 1e-5,
            },
        }
    )
    profile_bmo = ProfileBMO(HS_castro, **_prof_kwargs)
    _test_profile(profile_bmo, _reference_vals)
