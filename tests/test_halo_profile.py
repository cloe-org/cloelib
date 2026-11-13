# import jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.halos.halo_abundance import CastroHaloAbundance
from cloelib.observables.halos.halo_profile import BMOHaloProfile, NFWHaloProfile
from cloelib.observables.halos.halo_model_properties import HaloModelProperties
from cloelib.observables.halos.halo_profile import EmulatorMiscenteredHaloProfile


def _get_halo_model_properties():
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
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    return HaloModelProperties(perturbations)


def _get_castro():
    return CastroHaloAbundance(_get_halo_model_properties())


def test_array_shapes():
    # Profiles
    print("# Profiles ")
    _prof_kwargs = dict(
        two_halo="None",
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )

    profile_nfw = NFWHaloProfile(_get_halo_model_properties(), **_prof_kwargs)

    R_test = np.linspace(0.01, 1.0, 9)
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.linspace(1e14, 5e14, 6)
    c_test = 4.0
    HS = _get_halo_model_properties()
    profile = NFWHaloProfile(HS)

    _kwargs = {"R": R_test, "z": z_test, "M": M_test}
    out_shape = (z_test.size, M_test.size, R_test.size)

    for radius_units in ("Mpc/h", "radians", "degrees", "arcmin", "arcsec"):
        print(radius_units)
        _kwargs["radius_units"] = radius_units

        _r, _z, _m = profile.surface_mass_density_args(**_kwargs)
        assert (_r * _z * _m).shape == out_shape

        _kwargs["c"] = c_test

        _kwargs["halo_bias"] = np.ones((z_test.size, M_test.size))

        for two_halo in ("None", "sum", "max"):
            profile_nfw.two_halo = two_halo

            assert profile_nfw.surface_mass_density(**_kwargs).shape == out_shape
            assert profile_nfw.excess_surface_mass_density(**_kwargs).shape == out_shape

        _kwargs.pop("halo_bias")
        _kwargs.pop("c")


def _test_profile(profile, reference_vals):
    R_test = np.array([1])
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.array([5e14])
    c_test = 4.0
    z_sources_test = np.linspace(0.6, 1, 5)

    castro = _get_castro()
    halo_bias = castro.bias(z_test, M_test)

    print("    sigma_crit")
    assert_allclose(
        profile.sigma_crit(z_test, z_sources_test)[0],
        **reference_vals["sigma_crit"],
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
    profile_2h = (
        profile.halo_model_properties.surface_mass_density_2h(R_test, z_test)[
            :, np.newaxis, :
        ]
        * halo_bias[:, :, np.newaxis]
    )
    assert_allclose(
        profile_2h[:, 0, 0],
        **reference_vals["surface_mass_density_2h"],
    )
    print("    _excess_surface_mass_density_2h")
    profile_2h = (
        profile.halo_model_properties.excess_surface_mass_density_2h(
            R_test, z_test
        )[:, np.newaxis, :]
        * halo_bias[:, :, np.newaxis]
    )
    assert_allclose(
        profile_2h[:, 0, 0],
        **reference_vals["excess_surface_mass_density_2h"],
    )


def test_profiles():
    # Profiles
    print("# Profiles ")

    print("  NFW")
    _prof_kwargs = dict(
        two_halo="None",
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )
    profile_nfw = NFWHaloProfile(_get_halo_model_properties(), **_prof_kwargs)
    _reference_vals = {
        # All validation values have to be updated with extarnal values
        "sigma_crit": {
            "desired": [
                57269.705861,
                57134.890066,
                57034.779821,
                56957.599339,
                56896.34007,
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
            "desired": [13.316477, 16.810355, 20.655992, 24.778575],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density_2h": {
            "desired": [1.520876, 2.16498, 2.853623, 3.655179],
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

    _prof_kwargs = dict(
        two_halo="None",
        trunc_fact=3.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )
    profile_bmo = BMOHaloProfile(_get_halo_model_properties(), **_prof_kwargs)
    _test_profile(profile_bmo, _reference_vals)


def _test_misc_profile(profile, reference_vals):
    r"""
    Test miscentered Sigma_off and DeltaSigma_off profiles against benchmark
    values.

    Parameters
    ----------
    profile : MiscBMOHaloProfileEmu
        Instantiated emulator profile object.
    reference_vals : dict
        Dictionary with keys ``"surface_mass_density"`` and
        ``"excess_surface_mass_density"``, each containing ``"desired"``
        (h Msun / pc²) and ``"rtol"``.
    """
    # Input grid matching the benchmark generation
    R_test = 10.0 ** np.arange(2.0, 4.5, 0.05) / 1.0e3  # physical Mpc/h, (50,)
    z_test = np.array([1.0])  # (1,)
    M_test = np.array([1.0e14])  # Msun/h, (1,)
    c_test = 2.0
    sigma_off_test = 0.4  # physical Mpc/h

    out_shape = (z_test.size, M_test.size, R_test.size)  # (1, 1, 50)

    print("    surface_mass_density (misc)")
    Sigma_off = profile.surface_mass_density(
        R_test, z_test, M_test, c_test, sigma_off_test
    )
    assert Sigma_off.shape == out_shape, (
        f"surface_mass_density shape mismatch: got {Sigma_off.shape}, "
        f"expected {out_shape}"
    )
    assert_allclose(Sigma_off[0, 0, :], **reference_vals["surface_mass_density"])

    print("    excess_surface_mass_density (misc)")
    DeltaSigma_off = profile.excess_surface_mass_density(
        R_test, z_test, M_test, c_test, sigma_off_test
    )
    assert DeltaSigma_off.shape == out_shape, (
        f"excess_surface_mass_density shape mismatch: got {DeltaSigma_off.shape}, "
        f"expected {out_shape}"
    )
    assert_allclose(
        DeltaSigma_off[0, 0, :], **reference_vals["excess_surface_mass_density"]
    )
 
 
def test_misc_profile():
    r"""
    Integration test for MiscBMOHaloProfileEmu.
 
    Benchmark arrays were produced at:
        M = 1e14 Msun/h,  z = 1.0,  c = 2.0,  sigma_off = 0.4 Mpc/h,
        H0 = 67.7,  omch2 = 0.120,  ombh2 = 0.022
        R = 10^arange(2, 4.5, 0.05) / 1e3  physical Mpc/h.
 
    The benchmark values are in units of Msun h / physical Mpc², which
    convert to h Msun / pc² via the factor 1e-12
    (1 Mpc = 1e6 pc  →  1/Mpc² = 1e-12/pc²).
 
    Weights are loaded automatically from EmuNetWeights (no path arguments
    required).
    """
    print("# MiscBMOHaloProfileEmu")
 
    # Benchmark in Msun h / physical Mpc²; convert to h Msun / pc² via 1e-12.
    _sigma_off_mpc2 = np.array([
        1.03437782e+14, 1.02892023e+14, 1.02208253e+14, 1.01353450e+14,
        1.00287558e+14, 9.89624947e+13, 9.73213312e+13, 9.52978910e+13,
        9.28171457e+13, 8.97969655e+13, 8.61519774e+13, 8.18004608e+13,
        7.66752572e+13, 7.07393939e+13, 6.40062278e+13, 5.65620751e+13,
        4.85863345e+13, 4.03605128e+13, 3.22550638e+13, 2.46848866e+13,
        1.80343758e+13, 1.25710012e+13, 8.38335067e+12, 5.37794615e+12,
        3.33890203e+12, 2.01484296e+12, 1.18579349e+12, 6.81083267e+11,
        3.82348622e+11, 2.10344854e+11, 1.13748288e+11, 6.06457586e+10,
        3.19638895e+10, 1.66916789e+10, 8.65202212e+09, 4.45809141e+09,
        2.28611968e+09, 1.16779650e+09, 5.94661984e+08, 3.02036626e+08,
        1.53086286e+08, 7.74570941e+07, 3.91348831e+07, 1.97492122e+07,
        9.95643275e+06, 5.01528637e+06, 2.52454382e+06, 1.27002555e+06,
        6.38591907e+05, 3.20958012e+05,
    ])
 
    _dsigma_off_mpc2 = np.array([
        1.21628769e+12, 1.49424755e+12, 1.84603967e+12, 2.26596256e+12,
        2.79707329e+12, 3.44687095e+12, 4.25073964e+12, 5.24109630e+12,
        6.42031402e+12, 7.83788536e+12, 9.52616609e+12, 1.14948945e+13,
        1.37443295e+13, 1.62413303e+13, 1.89310467e+13, 2.17273282e+13,
        2.44462940e+13, 2.68214658e+13, 2.84883561e+13, 2.93805054e+13,
        2.92776593e+13, 2.81449114e+13, 2.60604369e+13, 2.33498010e+13,
        2.03703578e+13, 1.73444167e+13, 1.45191592e+13, 1.19739835e+13,
        9.78197961e+12, 7.91753709e+12, 6.37809774e+12, 5.11051992e+12,
        4.08501406e+12, 3.25849912e+12, 2.59579994e+12, 2.06687228e+12,
        1.64387867e+12, 1.30401149e+12, 1.04122801e+12, 8.23317572e+11,
        6.55088494e+11, 5.19934196e+11, 4.13581168e+11, 3.28310063e+11,
        2.60977427e+11, 2.07169421e+11, 1.64662861e+11, 1.30755777e+11,
        1.03861854e+11, 8.25733749e+10,
    ])
 
    _reference_vals = {
        "surface_mass_density": {
            "desired": _sigma_off_mpc2 * 1.0e-12,   # h Msun / pc²
            "rtol": 1e-2,                            # emulator-level accuracy (~1%)
        },
        "excess_surface_mass_density": {
            "desired": _dsigma_off_mpc2 * 1.0e-12,  # h Msun / pc²
            "rtol": 1.5e-2,
        },
    }
 
    _prof_kwargs = dict(
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    )
    profile_misc = EmulatorMiscenteredHaloProfile(_get_halo_model_properties(), **_prof_kwargs)
    profile_misc.set_weights()
    _test_misc_profile(profile_misc, _reference_vals)
