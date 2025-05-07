from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.observables.clusters.halo_statistics import (
    HaloStatistics,
    HaloStatisticsTinker,
    HaloStatisticsCastro,
)
from cloelib.observables.clusters.profile import ProfileNFW, ProfileBMO
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.clustering import HaloClustering


import numpy as np


def test_selectionfunction(SF):
    z_test = np.linspace(0.01, 1.0, 20)
    zob_test = np.linspace(0.1, 1.1, 20)
    M_test = 1.0e14
    l_test = np.logspace(0.0, 2.0, 20)
    lob_test = np.logspace(0.2, 2.2, 20)

    print("    lnlambda")
    SF.lnlambda(z_test, M_test)
    print("    scatter_lnl")
    SF.scatter_lnl(z_test, M_test)
    print("    P_lnlbd")
    SF.P_lnlbd(z_test, M_test, l_test)
    print("    scatter_lbdobs_lbd")
    SF.scatter_lbdobs_lbd(z_test, l_test)
    print("    P_lbdobs_lbd")
    SF.P_lbdobs_lbd(z_test, l_test, lob_test)
    print("    scatter_zobs_z")
    SF.scatter_zobs_z(lob_test, z_test)
    print("    P_zobs_z")
    SF.P_zobs_z(zob_test, lob_test, z_test)


def test_halostatistics(HS, HS_tinker, HS_castro):
    z_test = np.linspace(0.01, 1.0, 20)
    k_test = np.logspace(-2, 1, 100)
    R_test = np.logspace(-1, 1, 20)
    M_test = np.logspace(14, 15, 50)

    print("    window")
    HS.window(k_test, R_test)
    print("    radius_M")
    HS.radius_M(M_test)
    print("    delta_c")
    HS.delta_c(z_test)
    print("    get_Delta_crit")
    HS.get_Delta_crit(z_test)
    print("    sigma_z_R")
    HS.sigma_z_R(z_test, R_test)
    print("    sigma_z_M")
    HS.sigma_z_M(z_test, M_test)
    print("    nu_z_M")
    HS.nu_z_M(z_test, M_test)
    print("    dlns_dlnR")
    HS.dlns_dlnR(z_test, M_test)

    # HS_tinker.bias(z_test, M_test)

    HS_castro.dn_dm(z_test, M_test)
    HS_castro.bias(z_test, M_test)


def test_profiles(profile_nfw, profile_bmo):
    R_test = np.linspace(1e-5, 2, 21)
    z_test = np.linspace(0.01, 0.5, 20)
    M_test = 5e14
    c_test = 4.0
    z_sources_test = np.linspace(0.6, 1, 21)
    zbin_test = 1

    for _prof in (profile_nfw, profile_bmo):
        print("    sigma_crit")
        _prof.sigma_crit(z_test, z_sources_test)
        print("    n_zs_norM")
        _prof.n_zs_norM(z_test)
        print("    n_zs")
        _prof.n_zs(z_test)
        print("    surface_mass_density")
        _prof.surface_mass_density(
            R_test[10], z_test, c_test, M_test, force_no_2h=False, force_no_off=False
        )
        print("    excess_surface_mass_density")
        _prof.excess_surface_mass_density(R_test[10], z_test, c_test, M_test)
        print("    surface_mass_density_2h")
        _prof.surface_mass_density_2h(R_test[10], z_test, M_test)
        print("    excess_surface_mass_density_2h")
        _prof.excess_surface_mass_density_2h(R_test[10], z_test, M_test)


def test_clustering(CL):
    z_test = np.linspace(0.0, 2.0, 20)
    r_test = np.geomspace(20, 150, 30)
    lob_test = np.logspace(0.2, 2.2, 20)

    print("    APcorr_func")
    CL.APcorr_func(z_test)
    print("    WF_ra")
    CL.WF_ra(z_test, r_test)

    print("    Pk_IR_func")
    Pk_test = perturbations.matter_power_spectrum(
        z_test, CL.k, hubble_units=True, k_hunit=True
    )
    CL.Pk_IR_func(Pk_test)

    print("    photoz_rsd_correction")
    sigma_zob = SF.scatter_zobs_z(lob_test, z_test)
    CL.photoz_rsd_correction(z_test, sigma_zob)


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

    # SelectionFunction
    print("# SelectionFunction")
    _sel_pars = dict(
        A_l=0.5,
        B_l=0.6,
        C_l=0.5,
        sig_A_l=0.1,
        sig_B_l=0.0,
        sig_C_l=0.0,
        sig_lambda_norm=0.1,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.1,
        sig_z_z=0.1,
        sig_z_lambda=0.1,
    )
    SF = SelectionFunction(**_sel_pars)
    test_selectionfunction(SF)

    # HaloStatistics
    print("# HaloStatistics")
    HS = HaloStatistics(perturbations, "vir")
    HS_tinker = HaloStatisticsTinker(perturbations, "vir")
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
    test_profiles(profile_nfw, profile_bmo)

    # clustering
    print("# clustering")

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    perturbations_fid = CAMBLinearPerturbations(
        background_fid, np.linspace(0.0, 2.0, 100)
    )
    k_min = 1e-4
    k_max = 2e0
    k_div = 300
    nonu = True
    CL = HaloClustering(perturbations, perturbations_fid, nonu, k_div, k_min, k_max)
    test_clustering(CL)
