# import jax.numpy as np
import benchmark_values
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.castro_hmf_bias import CastroHMFBias
from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.observables.clusters.profile import ProfileNFW
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.summary_statistics.clusters.cluster_statistics import ClusterStatistics


def test_clustersummmarystatitistics():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.0
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

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    perturbations_fid = CAMBLinearPerturbations(
        background_fid, np.linspace(0.0, 2.0, 100)
    )

    area = 10313
    halo_concentration = 0.1
    overdensity_type = "vir"
    CG_like_selection = "CC_CWL_Cxi2"
    CG_xi2_cov_selection = "covCC_covCxi2"
    bias = "castro23"
    neutrino_cdm = True

    zed_obs_edges = np.linspace(0.2, 1.8, 9)
    Lambda_obs_edges = np.array([20.0, 30.0, 45.0, 60.0, 500.0])
    Rad_obs_edges = np.linspace(5.0, 100.0, 11)
    Lambda_obs_Cxi2_edges = np.array([20, 30, 500])
    Rad_obs_Cxi2_edges = np.geomspace(20.0, 130.0, 31)
    zed_obs_Cxi2_edges = np.arange(0.2, 1.81, 0.4)

    k = np.geomspace(1e-4, 10, 500)
    Mass = np.logspace(12.0, 16.0, 51)
    Lambda = np.geomspace(5.0, 250.0, 51)
    zed = np.linspace(1.0e-5, 6.0 - 1.0e-5, 200)

    _sel_pars = dict(
        A_l=52.0,
        B_l=0.9,
        C_l=0.5,
        sig_A_l=0.2,
        sig_B_l=-0.05,
        sig_C_l=0.001,
        sig_lambda_norm=0.9,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.4,
        sig_z_z=0.025,
        sig_z_lambda=5.0e-6,
    )

    _prof_pars = dict(
        r_interp=np.logspace(-10, 2.5, 200),
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

    HS = HaloStatistics(perturbations, z=zed, k=k, overdensity_type=overdensity_type)
    HSCastro = CastroHMFBias(HS)

    profileNFW = ProfileNFW(HSCastro, k=k, z=zed, **_prof_pars)

    selectionFunction = SelectionFunction(**_sel_pars)

    haloClustering = HaloClustering(
        perturbations, perturbations_fid, selectionFunction, k=k
    )

    covariance = HaloCovariance(
        perturbations, area=area, nbins_zob=len(zed_obs_edges), k=k
    )

    clusterStatistics = ClusterStatistics(
        perturbations,
        HS,
        selectionFunction,
        HSCastro,
        profileNFW,
        haloClustering,
        covariance,
        z_obs_edges=zed_obs_edges,
        Lambda_obs_edges=Lambda_obs_edges,
        Rad_obs_edges=Rad_obs_edges,
        Lambda_obs_Cxi2_edges=Lambda_obs_Cxi2_edges,
        Rad_obs_Cxi2_edges=Rad_obs_Cxi2_edges,
        z_obs_Cxi2_edges=zed_obs_Cxi2_edges,
        halo_concentration=halo_concentration,
        k=k,
        Mass=Mass,
        Lambda=Lambda,
        z=zed,
        area=area,
        CG_like_selection=CG_like_selection,
        CG_xi2_cov_selection=CG_xi2_cov_selection,
        bias=bias,
        neutrino_cdm=neutrino_cdm,
    )

    N_zbin_Lbin, g_zbin_Lbin_Rbin, Cxi2_zbin_Lbin_Rbin, cov_zbin_Lbin, cov_Cxi2 = (
        clusterStatistics.N_zbin_Lbin_Rbin()
    )

    assert_allclose(N_zbin_Lbin, benchmark_values.NC_ref, rtol=1e-2)

    assert_allclose(g_zbin_Lbin_Rbin[0:2], benchmark_values.gWL, rtol=1e-2)

    assert_allclose(Cxi2_zbin_Lbin_Rbin[0:2], benchmark_values.Cxi2, rtol=1e-2)

    assert_allclose(cov_zbin_Lbin[1:2], benchmark_values.NC_cov, rtol=5e-2)

    assert_allclose(
        cov_Cxi2[1, 1, 1:3, 1:3, 10:20, 10:20], benchmark_values.Cxi2_cov, rtol=5e-2
    )
