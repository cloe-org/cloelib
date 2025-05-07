from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.observables.clusters.halo_statistics import (
    HaloStatistics,
    HaloStatisticsTinker,
    HaloStatisticsCastro,
)
from cloelib.observables.clusters.profile import ProfileNFW, ProfileBMO
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations


import numpy as np

if __name__ == "__main__":
    # Cosmology parameters
    print("# Cosmology parameters")
    H0 = 67.7
    h = H0 / 100.0
    sigma8 = 0.8277
    omch2 = 0.12
    Omega_cdm0 = omch2 / h**2
    ombh2 = 0.022
    Omega_b0 = ombh2 / h**2
    Omega_k0 = 0.0
    w = -1.0
    wa = 0.0
    ns = 0.96
    mnu = 0.0
    As = 2e-9

    # sel. function parameters
    print("# sel. function parameters")
    A_l = 0.5
    B_l = 0.6
    C_l = 0.5
    sig_A_l = 0.1
    sig_B_l = 0.0
    sig_C_l = 0.0
    sig_lambda_norm = 0.1
    sig_lambda_z = 0.1
    sig_lambda_exponent = 0.1
    sig_z_z = 0.1
    sig_z_lambda = 0.1

    #
    background = CAMBBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=0.0,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
    )
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    # SelectionFunction
    print("# SelectionFunction")
    SF = SelectionFunction(
        A_l,
        B_l,
        C_l,
        sig_A_l,
        sig_B_l,
        sig_C_l,
        sig_lambda_norm,
        sig_lambda_z,
        sig_lambda_exponent,
        sig_z_z,
        sig_z_lambda,
    )

    z_test = np.linspace(0.01, 1.0, 20)
    zob_test = np.linspace(0.1, 1.1, 20)
    M_test = 1.0e14
    l_test = np.logspace(0.0, 2.0, 20)
    lob_test = np.logspace(0.2, 2.2, 20)

    SF.lnlambda(z_test, M_test)
    SF.scatter_lnl(z_test, M_test)
    SF.P_lnlbd(z_test, M_test, l_test)
    SF.scatter_lbdobs_lbd(z_test, l_test)
    SF.P_lbdobs_lbd(z_test, l_test, lob_test)
    SF.scatter_zobs_z(lob_test, z_test)
    SF.P_zobs_z(zob_test, lob_test, z_test)

    # HaloStatistics
    print("# HaloStatistics")
    HS = HaloStatistics(perturbations, "vir")

    k_test = np.logspace(-2, 1, 100)
    R_test = np.logspace(-1, 1, 20)
    M_test = np.logspace(14, 15, 50)

    """
    HS.window(k_test, R_test)
    HS.radius_M(M_test)
    HS.delta_c(z_test)
    HS.get_Delta_crit(z_test)
    HS.sigma_z_R(z_test, R_test)
    HS.sigma_z_M(z_test, M_test)
    HS.nu_z_M(z_test, M_test)
    HS.dlns_dlnR(z_test, M_test)
    """

    HS_tinker = HaloStatisticsTinker(perturbations, "vir")
    # HS_tinker.bias(z_test, M_test)

    HS_castro = HaloStatisticsCastro(perturbations, "vir")
    """
    HS_castro.dn_dm(z_test, M_test)
    HS_castro.bias(z_test, M_test)
    """

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

    R_test = np.linspace(1e-5, 2, 21)
    # z_test = 0.5
    M_test = 5e14
    c_test = 4.0
    z_sources_test = np.linspace(0.5, 1, 21)
    zbin_test = 1

    for _prof in (profile_nfw, profile_bmo):
        _prof.sigma_crit(z_test, z_sources_test)
        _prof.n_zs_norM(z_test)
        _prof.n_zs(z_test)
        _prof.m_sig_crit_m1(z_test, zbin_test)
        # _prof._surface_mass_density_cen(R_test, z_test, c_test, M_test, force_no_2h=False)
        # _prof._surface_mass_density_profile(R_test, RDelta_test, c_test, Delta_crit_test, rho_c_test)
        _prof.surface_mass_density(
            R_test, z_test, c_test, M_test, force_no_2h=False, force_no_off=False
        )
        _prof.excess_surface_mass_density(R_test, z_test, c_test, M_test)
        # _prof._mean_surface_mass_density_profile(R_test, RDelta_test, c_test, Delta_crit_test, rho_c_test)
        _prof.surface_mass_density_2h(R_test, z_test, M_test)
        _prof.excess_surface_mass_density_2h(R_test, z_test, M_test)
        # _prof.F_term(x_test)
        # _prof.G_term(x_test)
        # _prof._surface_mass_density_profile(R_test, RDelta_test, c_test, Delta_crit_test, rho_c_test)
        # _prof._mean_surface_mass_density_profile(R_test, RDelta_test, c_test, Delta_crit_test, rho_c_test)
