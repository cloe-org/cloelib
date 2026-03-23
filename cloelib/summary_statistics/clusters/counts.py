# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.observables.clusters.auxiliary import photoz_rsd_correction
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax

"""

## Notes :

- Cluster counts

"""


class ClusterCounts:
    """Object to compute cluster counts"""

    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        covariance: HaloCovariance,
        selection_function: SelectionFunction,
    ):
        """
        Initializes the cluster counts

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        covariance : HaloCovariance
            Halo covariance object
        selection_function : SelectionFunction
            Selection function object
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.covariance = covariance
        self.selection_function = selection_function

    def get_NC(
        self,
        z_obs_edges,
        lambda_obs_edges,
        return_intermediate_products=True,
    ):
        """Computes binned quantities (counts+aux).

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        return_intermediate_products : bool
            If true, also returns `intermediate_integration_products`, a dictionary
            with the intermediate products computed.

        Returns
        -------
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins
        intermediate_integration_products (optional) : dict
            Dictionary with intermidate products that can be used for other computations.
            Returned only when `return_intermediate_products` is true.
            Contains :

                * window_lambda_obs (numpy.ndarray) : Integral of P(lamda_obs|M, ztrue) in lambda_obs bins.
                * window_z_obs (numpy.ndarray) : Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################
        # integral of P(lambda_obs|M, z) on lambda_obs bins : (lambda_obs, M, ztrue)
        window_lambda_obs = self.cluster_statitstics_modeling.window_richness_observed(
            self.selection_function, lambda_obs_edges
        )
        # integral of P(z_obs|lambda_obs, z) on z_obs bins : (z_obs, lambda_obs, ztrue)
        window_z_obs = self.cluster_statitstics_modeling.window_z_observed(
            self.selection_function, z_obs_edges, lambda_obs_edges
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_lambda_obs
            ),
            window_z_obs,
        )

        if not return_intermediate_products:
            return cluster_counts

        return cluster_counts, {
            "window_lambda_obs": window_lambda_obs,
            "window_z_obs": window_z_obs,
        }

    # -------------------
    # cluster counts cov
    # -------------------

    def _compute_spatial_cov(self, z_obs_edges):
        """Computes only spatial part of the covariance.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.

        Returns
        -------
        spatial_cov : numpy.ndarray
            Spatial part of the covariance
        """

        z_obs_edges_size = len(z_obs_edges) - 1
        z_mid = 0.5 * (z_obs_edges[1:] + z_obs_edges[:-1])

        # power spectrum at the center of observed redshift bins (z_obs, k)
        pk = self.cluster_statitstics_modeling.matter_statistics.matter_power_spectrum_cb(
            z_mid, self.cluster_statitstics_modeling.tabulated_integrands["k"]
        )

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        # can neglect richness dependence here
        pk *= photoz_rsd_correction(
            self.cluster_statitstics_modeling.matter_statistics.background,
            z_mid,
            self.cluster_statitstics_modeling.tabulated_integrands["k"],
            self.cluster_statitstics_modeling.selectionfunction.scatter_zobs_z(
                0, z_mid
            ),
        )[0]

        # spherical harmonic expansion coefficients (covariance)
        KL = self.covariance.Kl_coeff()

        # compute spatial covariance (z_obs, z_obs)
        spatial_cov = np.zeros((z_obs_edges_size, z_obs_edges_size))
        for ind_z in range(z_obs_edges_size):
            spatial_cov[ind_z, : (ind_z + 1)] = (
                self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
                    np.sqrt(pk[ind_z] * pk[: (ind_z + 1)])
                    * self.covariance.cov_window(
                        ind_z,
                        (
                            z_obs_edges[ind_z],
                            z_obs_edges[ind_z + 1],
                        ),
                        KL,
                    ),
                )
            )
            # fill 2nd half of symmetrical matrix
            spatial_cov[: (ind_z + 1), ind_z] = spatial_cov[ind_z, : (ind_z + 1)]
        return spatial_cov

    def get_NC_covariance(
        self, z_obs_edges, cluster_counts, window_lambda_obs, window_z_obs
    ):
        """Computes theoretical covariance for cluster counts, including shot noise and sample covariance

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins
        window_lambda_obs : numpy.ndarray
            Integral of P(lamda_obs|M, ztrue) in lambda_obs bins.
            Dimensions: (lambda_obs, ztrue, M) with (ztrue, M) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_NC.
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
            Dimensions: (z_obs, lambda_obs, ztrue) with (ztrue) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_NC.

        Returns
        -------
        cov_cluster_counts : numpy.ndarray
            Covariance number counts in redshift and richness bins
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # integral of P(lambda_obs|M, z)*dn/dM*bias on lambda_obs bins and mass : (lambda_obs, ztrue)
        # cluster integrated bias : (z_obs, lambda_obs)
        halo_bias_mean_values = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs|M, z)*dn/dM*bias on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                self.cluster_statitstics_modeling.tabulated_integrands["bias(ztrue,M)"],
                window_lambda_obs,
            ),
            window_z_obs,
        )

        ####################
        # Compute covraiance
        ####################

        # spatial component of covariance (z_obs, z_obs)
        spatial_cov = self._compute_spatial_cov(z_obs_edges)

        # shot noise (z_obs, z_obs, lambda_obs, lambda_obs)
        _shot_noise = (
            np.diag(cluster_counts.flatten())
            .reshape(*cluster_counts.shape, *cluster_counts.shape)
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance (z_obs, z_obs, lambda_obs, lambda_obs)
        cov_cluster_counts = _shot_noise + (
            halo_bias_mean_values[np.newaxis, :, np.newaxis, :]
            * halo_bias_mean_values[:, np.newaxis, :, np.newaxis]
            * spatial_cov[:, :, np.newaxis, np.newaxis]
        )

        return cov_cluster_counts
