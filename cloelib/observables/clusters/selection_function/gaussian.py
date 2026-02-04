# General imports
# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.selection_function.lambda_true_distribution import (
    LambdaTrueDistribution,
)


class GaussianSelectionFunction:
    def __init__(
        self,
        lambda_true_distribution: LambdaTrueDistribution,
        sig_lambda_norm: float,
        sig_lambda_z: float,
        sig_lambda_exponent: float,
        sig_z_z: float,
        sig_z_lambda: float,
    ):
        r"""
        Class defining the selection function of galaxy clusters, including
        sample purity, completeness, mass-observable relation, and
        uncertainties on observed quantities.

        Parameters
        ----------
        lambda_true_distribution: LambdaTrueDistribution,
            Object that contains the distribution of true richness given mass
        sig_lambda_norm: float
            Amplitude of the observed proxy - true proxy relation
        sig_lambda_z: float
            Redshift evolution of the observed proxy - true proxy relation
        sig_lambda_exponent: float
            Exponential evolution of the observed proxy - true proxy relation
        sig_z_z: float
            Amplitude of the observed redshift - true redshift relation
        sig_z_lambda: float
            Proxy evolution of the observed redshift - true redshift relation
        """
        self.lambda_true_distribution = lambda_true_distribution
        self.sig_lambda_norm = sig_lambda_norm
        self.sig_lambda_z = sig_lambda_z
        self.sig_lambda_exponent = sig_lambda_exponent
        self.sig_z_z = sig_z_z
        self.sig_z_lambda = sig_z_lambda

    def _scatter_lambda_obs(self, z, lambda_true):
        r"""
        Statistical uncertainty on the observed mass proxy.

        Computes the scatter of the observed richness PDF
        at the requested true redshift and richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        scatter_lbobs_lbdz: numpy.ndarray
            scatter_lbobs_lbdz[i,j], where i is the true redshift axis
            and j is the true richness axis
        """
        return (
            self.sig_lambda_norm + self.sig_lambda_z * z[:, np.newaxis]
        ) * lambda_true**self.sig_lambda_exponent

    def _prob_lambda_obs(self, z, lambda_true, lambda_obs):
        r"""
        Observed mass proxy PDF.

        Computes the observed richness PDF at the requested
        true richness, true redshift, and observed richness points

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_true: numpy.ndarray
            True richness points.
        lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        prob_lambda_obs: numpy.ndarray
            prob_lambda_obs[i,j,k], where i is the redshift axis,
            j is the the theoretical richness axis,
            and k is the observed richness
        """
        sigma_lambda_obs = self._scatter_lambda_obs(z, lambda_true)[:, :, np.newaxis]

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigma_lambda_obs**2.0))
            * np.exp(
                -(
                    (
                        lambda_obs[np.newaxis, np.newaxis, :]
                        - lambda_true[np.newaxis, :, np.newaxis]
                    )
                    ** 2.0
                )
                / (2.0 * sigma_lambda_obs**2.0)
            )
        )

    def scatter_z_obs(self, lambda_obs, z):
        r"""
        Statistical uncertainty on the observed redshift.

        Computes the scatter of the observed redshift PDF
        at the requested true redshift and observed richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        scatter_z_obs: numpy.ndarray
            scatter_z_obs[i,j] where i is the true redshift axis
            and j the observed richness axis
        """
        return self.sig_z_z * z + self.sig_z_lambda * lambda_obs

    def _prob_zobs(self, z_obs, lambda_obs, z):
        r"""
        Observed redshift PDF.

        Computes the observed redshift PDF at the requested
        true richness, true redshift, and observed redshift points.

        Parameters
        ----------
        z_obs: numpy.ndarray
            Observed redshift points.
        lambda_obs: numpy.ndarray
            Observed richness points.
        z: numpy.ndarray
            True redshift points.

        Returns
        -------
        prob_zobs: numpy.ndarray
            prob_zobs[i,j,k] where i is the observed redshift axis,
            j is the observed richness axis,
            and k is the true redshift axis
        """
        sigmazobsz = self.scatter_z_obs(lambda_obs, z)

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigmazobsz**2.0))
            * np.exp(-((z_obs[:, np.newaxis] - z) ** 2.0) / (2.0 * sigmazobsz**2.0))
        )

    def window_z_observed(self, z_obs_edges, lambda_obs_edges, z_tab_sig, z_true):
        r"""Compute the window function of each observed redshift bin, given by:

        ..math:
            W_{\Delta z_{\rm obs}}(\lambda_{\rm obs}, z_{\rm true}) = \int_{\Delta z_{\rm obs}}dz_{\rm obs} P(z_{\rm obs}|\lambda_{\rm obs}, z_{\rm true})

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_tab_sig : int, None
            Number of points to be used for z_obs integration.
        z_true : numpy.ndarray
            True redshift to compute the window.

        Returns
        -------
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, z_true) in z_obs bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true).
        """

        z_obs_edges_size = len(z_obs_edges) - 1
        lambda_obs_edges_size = len(lambda_obs_edges) - 1

        # for z_obs integration
        z_obs_tabs = np.linspace(z_obs_edges[:-1], z_obs_edges[1:], z_tab_sig)

        # reshape for multiplication
        _z_obs_tabs = z_obs_tabs[:, :, np.newaxis]
        _lambda_obs = lambda_obs_edges[np.newaxis, :-1, np.newaxis]

        # Window function
        window_z_obs = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                z_true.size,
            )
        )
        for ind_z in range(z_obs_edges_size):
            window_z_obs[ind_z] = simps(
                self._prob_zobs(_z_obs_tabs[:, ind_z], _lambda_obs, z_true),
                x=z_obs_tabs[:, ind_z],
                axis=0,
            )
        return window_z_obs

    def window_richness_observed_richness_true(
        self,
        lambda_obs_edges,
        l_m_tab_sig,
        z_true,
        lambda_true,
    ):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda_{\rm obs}}(\lambda_{\rm true}, z_{\rm true}) = \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs} P(\lambda_{\rm obs}|\lambda_{\rm true}, z_{\rm true})

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        l_m_tab_sig : List, None
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_edges.
        z_true : numpy.ndarray
            True redshift to compute the window.
        lambda_true : numpy.ndarray
            True richness to compute the window.

        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|\lambda_{\rm true}, z_true) in lambda_obs bins.
            Dimensions: (lambda_obs_edges, z_true, \lambda_{\rm true}).
        """

        # if external_richness_selection_function == 'CG_ESF' :
        #     window_lambda_obs  = self.int_Plobltr_Dlob[lambda_bin](self.tabulated_integrands["z_true"], self.tabulated_integrands["lambda_true"]).T

        lambda_obs_edges_size = len(lambda_obs_edges) - 1
        window_lambda_obs_lambda_true = np.zeros(
            (
                lambda_obs_edges_size,
                z_true.size,
                lambda_true.size,
            )
        )
        for ind_lambda in range(lambda_obs_edges_size):
            # integrate P(lambda_obs|lambda_true, z) in lambda_obs
            l_tab = np.geomspace(
                lambda_obs_edges[ind_lambda],
                lambda_obs_edges[ind_lambda + 1],
                l_m_tab_sig[ind_lambda],
            )
            window_lambda_obs_lambda_true[ind_lambda] = simps(
                self._prob_lambda_obs(
                    z_true,
                    lambda_true,
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )
        return window_lambda_obs_lambda_true

    def window_redshift_richness_observed(
        self,
        z_obs_edges,
        lambda_obs_edges,
        z_true,
        mass,
        lambda_true,
        z_tab_sig,
        l_m_tab_sig,
    ):
        r"""
        Computes the window function for observed redshift and richness bins, i. e.:


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.
        z_tab_sig : int, None
            Number of points to be used for z_obs integration.
        l_m_tab_sig : List, None
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_edges.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true, mass)
        """
        # Dimensions: (z_obs_edges, lambda_obs_edges, z_true, lambda_true)
        window_lambda_true = (
            self.window_z_observed(z_obs_edges, lambda_obs_edges, z_tab_sig, z_true)[
                :, :, :, np.newaxis
            ]
            # Dimensions: (z_obs_edges, lambda_obs_edges, z_true, 1).
            * self.window_richness_observed_richness_true(
                self,
                lambda_obs_edges,
                l_m_tab_sig,
                z_true,
                lambda_true,
            )[np.newaxis, :, :, :]
            # Dimensions: (1, lambda_obs_edges, z_true, lambda_true)
        )
