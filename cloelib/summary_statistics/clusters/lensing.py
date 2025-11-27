# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.observables.clusters.profile import Profile
from cloelib.summary_statistics.clusters.statistics_modeling import (
    ClusterStatisticsModeling,
)

# import jax

"""

## Notes :

- Cluster profile lensing

"""


class ClusterWeakLensing:
    def __init__(
        self,
        cluster_statitstics_modeling: ClusterStatisticsModeling,
        profile: Profile,
        halo_concentration: float,
    ):
        """
        Initializes the cluster profile lensing

        Parameters
        ----------
        cluster_statitstics_modeling : ClusterStatisticsModeling
            Cluster summary statistics modeling object, it contains functions
            for cluster statistics and tabled values for integration.
        profile : Profile
            Halo weak lensing radial profile object
        halo_concentration : float
            Halo concentration
        """
        # cluster counts summary statistics, contains tables for integrals
        # and functions to compute binned integrals of counts
        self.cluster_statitstics_modeling = cluster_statitstics_modeling

        # observable objects
        self.profile = profile

        # internal values
        self.halo_concentration = halo_concentration

        # hardcoded quantities for integration
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

    def get_DeltaSigma(self, z_obs_edges, lambda_obs_edges, radius_edges):
        """Compute excess surface density.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile.

        Returns
        -------
        deltasigma_mean_values : numpy.ndarray
            Excess surface density in redshift, richness, and radial bins.
        """
        z_obs_edges_size = len(z_obs_edges) - 1

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # integral of P(z_obs|lambda_obs, z) on z_obs bins : (z_obs, lambda_obs, ztrue)
        window_z_obs = self.cluster_statitstics_modeling.window_z_observed(
            z_obs_edges, lambda_obs_edges, self.z_tab_sig
        )
        # integral of P(lambda_obs|M, z) on lambda_obs bins : (lambda_obs, M, ztrue)
        window_lambda_obs = self.cluster_statitstics_modeling.window_richness_observed(
            lambda_obs_edges, self.l_m_tab_sig
        )
        # cluster counts : (z_obs, lambda_obs)
        cluster_counts = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_lambda_obs
            ),
            window_z_obs,
        )

        ########################
        # Compute binned profile
        ########################

        # mass/richness part

        # excess surface mass density : (ztrue, M, radius)
        excess_surface_mass_density = self.profile.excess_surface_mass_density(
            radius_edges[:-1],
            self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
            self.cluster_statitstics_modeling.tabulated_integrands["M"],
            self.halo_concentration,
        )
        # excess surface mass density in a richness bin, integrated on mass w HMF : (lambda_obs, ztrue, radius)
        excess_surface_density_in_window_lambda_obs_mass_integrated = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                excess_surface_mass_density, window_lambda_obs
            )
        )

        # redshift part

        # Effective inverse critical surface mass density : (z_obs, ztrue)
        effective_inverse_critical_surface_mass_density = np.zeros(
            (
                z_obs_edges_size,
                self.cluster_statitstics_modeling.tabulated_integrands["ztrue"].size,
            )
        )
        for ind_z in range(z_obs_edges_size):
            effective_inverse_critical_surface_mass_density[ind_z] = (
                self.profile.m_sig_crit_m1(
                    self.cluster_statitstics_modeling.tabulated_integrands["ztrue"],
                    ind_z,
                )
            )

        # output : (z_obs, lambda_obs, radius)
        deltasigma_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                excess_surface_density_in_window_lambda_obs_mass_integrated,
                window_z_obs
                * effective_inverse_critical_surface_mass_density[:, np.newaxis, :],
            )
        ) / cluster_counts[:, :, np.newaxis]
        return deltasigma_mean_values
