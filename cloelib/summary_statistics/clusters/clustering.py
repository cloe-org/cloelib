# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.observables.clusters.auxiliary import photoz_rsd_correction
from cloelib.observables.clusters.halo_clustering import HaloClustering
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax

"""

## Notes :

- Clusters clustering class

"""


class ClusterClustering:
    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        clustering: HaloClustering,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        clustering : HaloClusteringCore
            Halo clustering object
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.clustering = clustering

        # hardcoded quantities for integration
        self.l_m_tab_sig = [31, 51]
        self.z_tab_sig = 31

    def get_xi(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        return_intermediate_products=True,
    ):
        """Computes binned quantities (clustering+aux)

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the clustering.

        Returns
        -------
        cluster_clustering : numpy.ndarray
            Two point correlation function in richness, redshift and radial bins
        intermediate_integration_products (optional) : dict
            Dictionary with intermidate products that can be used for other computations.
            Returned only when `return_intermediate_products` is true.
            Contains :

                * pk_mean_values (numpy.ndarray) : Power spectrum averaged on redshift and richnesses bins (with IR-resummation).
                * radial_shell_window (numpy.ndarray) : Cluster count covariance window (z_obs, lambda_obs, k).
                * radial_shell_volume (numpy.ndarray) : Spherical shell volume (z_obs, radius).
                * window_z_obs (numpy.ndarray) : Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
                * cluster_counts (numpy.ndarray) :  Number counts in redshift and richness bins
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # integral of P(z_obs|lambda_obs, z) on z_obs bins : (z_obs, lambda_obs, ztrue)
        window_z_obs = self.cluster_statitstics_modeling.window_z_observed(
            z_obs_edges, lambda_obs_edges, self.z_tab_sig
        )
        # integral of P(lambda_obs|M, z) on lambda_obs bins : (lambda_obs_edges, M, ztrue)
        _window_lambda_obs = self.cluster_statitstics_modeling.window_richness_observed(
            lambda_obs_edges, self.l_m_tab_sig
        )
        # integral of P(lambda_obs|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
        window_lambda_obs_mass_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), _window_lambda_obs
            )
        )
        # integral of P(lambda_obs|M, z)*dn/dM*bias on lambda_obs bins and mass : (lambda_obs, ztrue)
        halo_bias_in_window_lambda_obs_mass_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                self.cluster_statitstics_modeling.tabulated_integrands["bias(ztrue,M)"],
                _window_lambda_obs,
            )
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                window_lambda_obs_mass_integrated, window_z_obs
            )
        )

        ################################################
        # Computes radial shells for the 3D 2ptcf
        ################################################

        # radial_shell_window : (z_obs, radius, k)
        # radial_shell_volume : (z_obs, radius)
        radial_shell_window, radial_shell_volume = (
            self.clustering.core.radial_shell_window_and_volume(
                0.5 * (z_obs_edges[1:] + z_obs_edges[:-1]),
                self.cluster_statitstics_modeling.tabulated_integrands["k"],
                radius_edges,
            )
        )
        # * radial_shell_volume is used only by covariance

        ################################################
        # Computes the 3D two-point correlation function
        ################################################

        # halo matter power spectrum + IR resummation : (lambda_obs, ztrue, k) dimension
        _pk_halo = self.clustering.power_spectrum_RSD_corrected(
            z=self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
            k=self.cluster_statitstics_modeling.tabulated_integrands["k"],
            z_obs_scatter=(
                self.cluster_statitstics_modeling.selectionfunction.scatter_zobs_z(
                    0.5 * (lambda_obs_edges[1:] + lambda_obs_edges[:-1])[np.newaxis, :],
                    self.cluster_statitstics_modeling.tabulated_integrands["ztrue"][
                        :, np.newaxis
                    ],
                )
            ),  # (ztrue, lambda_obs)
            b_eff=(
                halo_bias_in_window_lambda_obs_mass_integrated
                / window_lambda_obs_mass_integrated
            ).T,  # (ztrue, lambda_obs)
        ).transpose(2, 0, 1)

        # integrate the square root of power spectrum in redshift bins (z_obs, lambda_obs, k)
        _sqrt_pk_z_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                np.sqrt(_pk_halo) * window_lambda_obs_mass_integrated[:, :, np.newaxis],
                window_z_obs,
            )
        )

        # Compute Pk convoluted in lambda_obs: (z_obs, lambda_obs, lambda_obs, k)
        pk_mean_values = (
            _sqrt_pk_z_integrated[:, :, np.newaxis, :]
            * _sqrt_pk_z_integrated[:, np.newaxis, :, :]
        ) / (
            cluster_counts[:, np.newaxis, :, np.newaxis]
            * cluster_counts[:, :, np.newaxis, np.newaxis]
        )

        # compute 2point correlation function : (z_obs, lambda_obs, lambda_obs, radius)
        _2pt_3d_cf = self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
            radial_shell_window[:, np.newaxis, np.newaxis, :, :]
            * pk_mean_values[:, :, :, np.newaxis, :]
        )

        # xi(lambda_obs_i, lambda_obs_j) = xi(lambda_obs_j, lambda_obs_i) so we reshape
        # and keep only one of them, with a (z_obs, lambda_obs, radius) output
        triangle_indexes = np.triu_indices(len(lambda_obs_edges) - 1)
        cluster_clustering = _2pt_3d_cf[:, triangle_indexes[0], triangle_indexes[1], :]

        if not return_intermediate_products:
            return cluster_clustering

        intermediate_integration_products = {
            "pk_mean_values": pk_mean_values,
            "radial_shell_window": radial_shell_window,
            "radial_shell_volume": radial_shell_volume,
            "cluster_counts": cluster_counts,
            "window_z_obs": window_z_obs,
        }
        return cluster_clustering, intermediate_integration_products

    # ----------------------
    # clustering covariance
    # ----------------------

    def get_xi_covariance(
        self,
        pk_mean_values,
        radial_shell_window,
        radial_shell_volume,
        window_z_obs,
        cluster_counts,
    ):
        """Computes clustering covariance.

        Parameters
        ----------
        pk_mean_values : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation).
            Is in the intermediate_integration_products output of get_xi.
        radial_shell_window : numpy.ndarray
            Cluster count covariance window (z_obs, lambda_obs, k),
            with (k) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_xi.
        radial_shell_volume : numpy.ndarray
            Spherical shell volume (z_obs, radius).
            Is in the intermediate_integration_products output of get_xi.
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
            Dimentions: (z_obs, lambda_obs, ztrue) with (ztrue) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_xi.
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins

        Returns
        -------
        cov_cluster_clustering : numpy.ndarray
            Covariance of the two point correlation function in richness, redshift and radial bins
        """
        z_obs_edges_size, lambda_obs_edges_size = cluster_counts.shape
        _, radius_edges_size = radial_shell_volume.shape

        ########################################
        # Cluster statistics modeling quantities
        ########################################

        # Compute observed volume in each redshift bin : (z_obs, lambda_obs)
        volume_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                np.ones((1, 1)), window_z_obs
            )
        )
        # Compute output shot-noise terms : (z_obs, lambda_obs, lambda_obs)
        vol_over_cluster_counts = (
            volume_mean_values[:, :, np.newaxis]
            / cluster_counts[:, :, np.newaxis]
            * np.identity(lambda_obs_edges_size)[np.newaxis, :, :]
        )

        #############################
        # Compute nuisance parameters
        #############################

        #    alpha(ztrue,lambda_obs), beta(ztrue,lambda_obs),
        #    gamma(ztrue,lambda_obs) are nuisance parameters to be fitted on
        #    (few, ~100) simulations to correct for bias model inaccuracy,
        #    non-poissonian shot-noise and high-order terms ref values are
        #    alpha=0, beta=1, gamma=0 (see Euclid Collaboration : Fumagalli et
        #    al. 2022)
        alpha = np.zeros((z_obs_edges_size, lambda_obs_edges_size))
        beta = np.ones((z_obs_edges_size, lambda_obs_edges_size))
        gamma = np.zeros((z_obs_edges_size, lambda_obs_edges_size))

        # Combine alpha, beta with pk, vol and reshape to be used
        # in cov_g, cov_ng integral
        # shape (z_obs_edges, lambda_obs, lambda_obs, k)
        beta_pk_mean_values = (
            beta[:, :, np.newaxis, np.newaxis]
            * beta[:, np.newaxis, :, np.newaxis]
            * pk_mean_values
        )
        # shape (z_obs_edges, lambda_obs, lambda_obs, k)
        avol_bpk_mean_values = (
            # reshape alpha to be (z_obs, lambda_obs, lambda_obs)
            (1 + alpha)[:, :, np.newaxis, np.newaxis]
            * (1 + alpha)[:, np.newaxis, :, np.newaxis]
            * vol_over_cluster_counts[:, :, :, np.newaxis]
            + beta_pk_mean_values
        )

        ####################
        # Compute covariance
        ####################

        # define cluster clustering bin numbers for loops
        z_bin_loop = range(z_obs_edges_size)
        lambda_bin_loop = range(lambda_obs_edges_size)
        rad_bin_loop = range(radius_edges_size)

        # cov_g, cov_ng are TWO TERMS OF EQ. 73
        _cov_gaussian = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                radius_edges_size,
                radius_edges_size,
            )
        )
        _cov_nongaussian = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                radius_edges_size,
                radius_edges_size,
            )
        )

        # note : this could be reduced to compute only half of the matrix
        for ind_lambda_i in lambda_bin_loop:
            for ind_lambda_j in lambda_bin_loop:
                for ind_radius in rad_bin_loop:
                    _cov_nongaussian[
                        :,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_radius,
                        ind_radius,
                    ] = (
                        self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
                            radial_shell_window[:, ind_radius, :]
                            * beta_pk_mean_values[:, ind_lambda_i, ind_lambda_j, :],
                        )
                        * (1 + gamma[:, ind_lambda_i])
                        * vol_over_cluster_counts[:, ind_lambda_i, ind_lambda_i]
                        * (1 + gamma[:, ind_lambda_j])
                        * vol_over_cluster_counts[:, ind_lambda_j, ind_lambda_j]
                        / radial_shell_volume[:, ind_radius]
                    )

                for ind_lambda_k in lambda_bin_loop:
                    for ind_lambda_h in lambda_bin_loop:

                        # somehow using integrate_probe_function_in_k is much faster then
                        # integrate_probe_function_in_dk here, to be investigated

                        # gaussian term
                        _cov_gaussian[
                            :,
                            ind_lambda_i,
                            ind_lambda_j,
                            ind_lambda_k,
                            ind_lambda_h,
                            :,
                            :,
                        ] = self.cluster_statitstics_modeling.integrate_probe_function_in_k(
                            radial_shell_window[:, np.newaxis, :, :]
                            * radial_shell_window[:, :, np.newaxis, :]
                            * avol_bpk_mean_values[
                                :, ind_lambda_i, ind_lambda_k, np.newaxis, np.newaxis, :
                            ]
                            * avol_bpk_mean_values[
                                :, ind_lambda_j, ind_lambda_h, np.newaxis, np.newaxis, :
                            ]
                            * self.cluster_statitstics_modeling.tabulated_integrands[
                                "dk"
                            ],
                        )

        # Compute the covariance : (z_obs, lambda_obs,  lambda_obs, lambda_obs, lambda_obs, radius, radius)
        _cov_clustering_4_lambda_obs_edges = (
            (_cov_gaussian + _cov_nongaussian)
            + (_cov_gaussian + _cov_nongaussian).transpose(
                0, 1, 2, 4, 3, 5, 6  # tranposing lambda_obs_clustering bins
            )
        ) / volume_mean_values[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(lambda_obs_i, lambda_obs_j, lambda_obs_k, lambda_obs_l) =
        # cov_xi(lambda_obs_j, lambda_obs_i, lambda_obs_l, lambda_obs_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(lambda_obs_edges_size)
        # simplify first pair
        _cov_clustering_3_lambda_obs_edges = _cov_clustering_4_lambda_obs_edges[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        _cov_clustering_2_lambda_obs_edges = _cov_clustering_3_lambda_obs_edges[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (z_obs, lambda_obs, lambda_obs, radius, radius),
        # make it (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius),
        # being diagonal in (z_obs, z_obs)
        cov_cluster_clustering = (
            np.identity(z_obs_edges_size)[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * _cov_clustering_2_lambda_obs_edges
        )
        return cov_cluster_clustering
