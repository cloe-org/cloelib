# import jax.numpy as np
import time

import benchmark_values
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_clustering import TwoPoint3DHaloClustering
from cloelib.observables.clusters.halo_profile import NFWHaloProfile
from cloelib.observables.clusters.matter_statistics import MatterStatistics
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterClustering,
    ClusterCounts,
    ClusterStatisticsModeling,
    ClusterWeakLensing,
)


def get_values():

    ###########
    # Cosmology
    ###########

    print("# Cosmology parameters")
    t0 = time.time()
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
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    print(f"cosmo     :  {time.time()-t0:.4f} seconds")
    t0 = time.time()

    #############
    # Observables
    #############

    # Parameters

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

    integ_k_arr = np.geomspace(1e-4, 10, 500)
    integ_mass_arr = np.logspace(12.0, 16.0, 51)
    integ_lambda_true_arr = np.geomspace(5.0, 250.0, 51)
    integ_ztrue_arr = np.linspace(1.0e-5, 6.0 - 1.0e-5, 200)

    halo_concentration = 0.1
    area = 10313

    # Integration bins

    z_obs_nc_edges = np.linspace(0.2, 1.8, 9)
    lambda_obs_nc_edges = np.array([20.0, 30.0, 45.0, 60.0, 500.0])
    z_obs_profile_edges = np.linspace(0.2, 1.8, 9)
    lambda_obs_profile_edges = np.array([20.0, 30.0, 45.0, 60.0, 500.0])
    radius_profile_edges = np.linspace(5.0, 100.0, 11)
    lambda_obs_clustering_edges = np.array([20, 30, 500])
    radius_clustering_edges = np.geomspace(20.0, 130.0, 31)
    zed_obs_clustering_edges = np.arange(0.2, 1.81, 0.4)

    # Istanciate objects

    selectionFunction = SelectionFunction(**_sel_pars)
    matter_stat = MatterStatistics(
        perturbations,
        z=integ_ztrue_arr,
        k=integ_k_arr,
    )
    HSCastro = CastroHaloAbundance(matter_statistics=matter_stat)
    covariance = HaloCovariance(
        perturbations, area=area, nbins_zob=len(z_obs_nc_edges), k=integ_k_arr
    )
    profileNFW = NFWHaloProfile(matter_stat, two_halo="None")
    haloClustering = TwoPoint3DHaloClustering(matter_stat, background_fid)

    print(f"init obs  :  {time.time()-t0:.4f} seconds")
    t0 = time.time()

    ####################
    # Summary Statistics
    ####################

    print("---------------------------")
    t1 = time.time()

    # Istanciate objects
    integ_ztrue_arr_new = integ_ztrue_arr.copy()
    integ_ztrue_arr_new[0] += 1.0e-10
    integ_ztrue_arr_new[-1] -= 1.0e-10

    integ_k_arr_new = integ_k_arr.copy()
    integ_k_arr_new[0] += 1.0e-10
    integ_k_arr_new[-1] -= 1.0e-10

    cluster_statitstics_modeling = ClusterStatisticsModeling(
        HSCastro,
        selectionFunction,
        integ_k_arr=integ_k_arr_new,
        integ_mass_arr=integ_mass_arr,
        integ_lambda_true_arr=integ_lambda_true_arr,
        integ_ztrue_arr=integ_ztrue_arr_new,
        area=area,
    )
    cluster_counts_statistics = ClusterCounts(
        cluster_statitstics_modeling,
        covariance,
    )
    cluster_wl_statistics = ClusterWeakLensing(
        cluster_statitstics_modeling,
        profileNFW,
        halo_concentration=halo_concentration,
    )
    cluster_clustering_statistics = ClusterClustering(
        cluster_statitstics_modeling,
        haloClustering,
    )

    print(f"init stat :  {time.time()-t0:.4f} seconds")
    t0 = time.time()

    # Compute values

    cluster_counts, counts_intermediate_integration_products = (
        cluster_counts_statistics.get_NC(
            z_obs_edges=z_obs_nc_edges,
            lambda_obs_edges=lambda_obs_nc_edges,
        )
    )
    print(f"nc        :  {time.time()-t0:.4f} seconds")
    t0 = time.time()
    cov_cluster_counts = cluster_counts_statistics.get_NC_covariance(
        z_obs_nc_edges,
        cluster_counts,
        counts_intermediate_integration_products["window_lambda_obs"],
        counts_intermediate_integration_products["window_z_obs"],
    )
    print(f"nc_cov    :  {time.time()-t0:.4f} seconds")
    t0 = time.time()
    gt_mean_values = cluster_wl_statistics.get_gt(
        z_obs_edges=z_obs_profile_edges,
        lambda_obs_edges=lambda_obs_profile_edges,
        radius_edges=radius_profile_edges,
    )
    print(f"dsig      :  {time.time()-t0:.4f} seconds")
    t0 = time.time()
    cluster_clustering, clustering_intermediate_integration_products = (
        cluster_clustering_statistics.get_xi(
            lambda_obs_edges=lambda_obs_clustering_edges,
            radius_edges=radius_clustering_edges,
            z_obs_edges=zed_obs_clustering_edges,
        )
    )
    print(f"xi        :  {time.time()-t0:.4f} seconds")
    t0 = time.time()
    cov_cluster_clustering = cluster_clustering_statistics.get_xi_covariance(
        clustering_intermediate_integration_products["pk_mean_values"],
        clustering_intermediate_integration_products["radial_shell_window"],
        clustering_intermediate_integration_products["radial_shell_volume"],
        clustering_intermediate_integration_products["window_z_obs"],
        clustering_intermediate_integration_products["cluster_counts"],
    )
    print(f"xi_cov    :  {time.time()-t0:.4f} seconds")
    t0 = time.time()
    print("---------------------------")
    print(f"tot like  :  {time.time()-t1:.4f} seconds")

    # Max relative difference
    print()
    print("Max relative differences to the reference values:")
    print_diff(
        "nc",
        cluster_counts,
        benchmark_values.cluster_counts,
    )
    print_diff(
        "dsig",
        gt_mean_values[0:2],
        benchmark_values.deltasigma,
    )
    print_diff(
        "xi",
        cluster_clustering[0:2],
        benchmark_values.cluster_clustering,
    )
    print_diff(
        "nc_cov",
        cov_cluster_counts[1:2],
        benchmark_values.cov_cluster_counts,
    )
    print_diff(
        "xi_cov",
        cov_cluster_clustering[1, 1, 1:3, 1:3, 10:20, 10:20],
        benchmark_values.cov_cluster_clustering,
    )
    return (
        cluster_counts,
        gt_mean_values,
        cluster_clustering,
        cov_cluster_counts,
        cov_cluster_clustering,
    )


def print_diff(name, value_test, value_ref, min_comparison_value=0):
    _msk = abs(value_ref) > min_comparison_value
    print(f"  {name:6} : {abs(value_test[_msk]/value_ref[_msk]-1).max():.2e}")


def test_clustersummmarystatitistics():
    (
        cluster_counts,
        gt_mean_values,
        cluster_clustering,
        cov_cluster_counts,
        cov_cluster_clustering,
    ) = get_values()

    assert_allclose(cluster_counts, benchmark_values.cluster_counts, rtol=1e-2)

    assert_allclose(gt_mean_values[0:2], benchmark_values.deltasigma, rtol=1e-2)

    assert_allclose(
        cluster_clustering[0:2], benchmark_values.cluster_clustering, rtol=1e-2
    )

    assert_allclose(
        cov_cluster_counts[1:2], benchmark_values.cov_cluster_counts, rtol=5e-2
    )

    assert_allclose(
        cov_cluster_clustering[1, 1, 1:3, 1:3, 10:20, 10:20],
        benchmark_values.cov_cluster_clustering,
        rtol=5e-2,
    )


if __name__ == "__main__":
    get_values()
