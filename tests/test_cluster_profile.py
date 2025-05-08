import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.observables.clusters.halo_statistics import (
    HaloStatisticsCastro,
)
from cloelib.observables.clusters.profile import ProfileNFW, ProfileBMO
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations


def _test_profiles(profile_nfw, profile_bmo):
    R_test = 1
    z_test = np.linspace(0.01, 0.5, 20)
    M_test = 5e14
    c_test = 4.0
    z_sources_test = np.linspace(0.6, 1, 21)
    zbin_test = 1

    for _name, _prof in zip(("NFW", "BMO"), (profile_nfw, profile_bmo)):
        print(f"  {_name}")
        print("    sigma_crit")
        assert_allclose(_prof.sigma_crit(z_test, z_sources_test), XXX)
        print("    n_zs_norM")
        assert_allclose(_prof.n_zs_norM(z_test), XXX)
        print("    n_zs")
        assert_allclose(_prof.n_zs(z_test), XXX)
        print("    surface_mass_density")
        assert_allclose(
            _prof.surface_mass_density(
                R_test, z_test, c_test, M_test, force_no_2h=False, force_no_off=False
            ),
            XXX,
        )
        print("    excess_surface_mass_density")
        assert_allclose(
            _prof.excess_surface_mass_density(R_test, z_test, c_test, M_test), XXX
        )
        print("    surface_mass_density_2h")
        assert_allclose(_prof.surface_mass_density_2h(R_test, z_test, M_test), XXX)
        print("    excess_surface_mass_density_2h")
        assert_allclose(
            _prof.excess_surface_mass_density_2h(R_test, z_test, M_test), XXX
        )


if __name__ == "__main__":
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
        mnu=0.0,
        As=2e-9,
        gamma_MG=0.0,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    HS_castro = HaloStatisticsCastro(perturbations, "vir")
    test_halostatistics(HS, HS_tinker, HS_castro)

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
    profile_bmo = ProfileBMO(HS_castro, **_prof_kwargs)
    _test_profiles(profile_nfw, profile_bmo)
