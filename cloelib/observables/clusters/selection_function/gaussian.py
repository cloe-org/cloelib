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
        z_tab_integ: int,
        lambda_tab_integ: list,
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
        z_tab_integ : int
            Number of points to be used for z_obs integration.
        lambda_tab_integ : List
            Number of points to be used for the lambda_obs integration
            in each lambda_obs bin. Must be same size of lambda_obs_edges.
        """
        self.lambda_true_distribution = lambda_true_distribution
        self.sig_lambda_norm = sig_lambda_norm
        self.sig_lambda_z = sig_lambda_z
        self.sig_lambda_exponent = sig_lambda_exponent
        self.sig_z_z = sig_z_z
        self.sig_z_lambda = sig_z_lambda
        self.z_tab_integ = z_tab_integ
        self.lambda_tab_integ = lambda_tab_integ

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
            Statistical uncertainty on the observed mass proxy.
        """
        return (
            self.sig_lambda_norm + self.sig_lambda_z * z
        ) * lambda_true**self.sig_lambda_exponent

    @staticmethod
    def _gaussian(value, mean, sigma):
        return np.exp(-((value - mean) ** 2.0) / (2.0 * sigma**2.0)) / (
            np.sqrt(2.0 * np.pi * sigma**2.0)
        )

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
            Observed mass proxy PDF.
        """
        return self._gaussian(
            lambda_obs, lambda_true, self._scatter_lambda_obs(z, lambda_true)
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
            Statistical uncertainty on the observed redshift.
        """
        return self.sig_z_z * z + self.sig_z_lambda * lambda_obs

    def _prob_z_obs(self, z_obs, lambda_obs, z):
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
        numpy.ndarray
            Observed redshift PDF.
        """
        return self._gaussian(z_obs, z, self.scatter_z_obs(lambda_obs, z))

    def window_z_observed(
        self, z_obs_edges, lambda_obs_edges, z_true, lambda_true=None
    ):
        r"""Compute the window function of each observed redshift bin, given by:

        ..math:
            W_{\Delta z_{\rm obs}}(\lambda_{\rm obs}, z_{\rm true}) = \int_{\Delta z_{\rm obs}}dz_{\rm obs} P(z_{\rm obs}|\lambda_{\rm obs}, z_{\rm true})

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        lambda_true : None
            Values to be used for marginalization over true richness, not used here.

        Returns
        -------
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, z_true) in z_obs bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true).
        """

        z_obs_bins_size = len(z_obs_edges) - 1
        lambda_obs_bins_size = len(lambda_obs_edges) - 1

        # for z_obs integration
        z_obs_tabs = np.linspace(z_obs_edges[:-1], z_obs_edges[1:], self.z_tab_integ).T

        # reshape for multiplication
        _z_obs_tabs = z_obs_tabs[:, :, np.newaxis, np.newaxis]
        _lambda_obs = lambda_obs_edges[np.newaxis, :-1, np.newaxis]
        _z_true = z_true[np.newaxis, np.newaxis, :]

        # Window function
        window_z_obs = np.zeros(
            (
                z_obs_bins_size,
                lambda_obs_bins_size,
                z_true.size,
            )
        )
        for ind_z in range(z_obs_bins_size):
            window_z_obs[ind_z] = simps(
                self._prob_z_obs(_z_obs_tabs[ind_z], _lambda_obs, _z_true),
                x=z_obs_tabs[ind_z],
                axis=0,
            )
        return window_z_obs

    def window_richness_observed(
        self,
        lambda_obs_edges,
        z_true,
        mass,
        lambda_true,
    ):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            P(\lambda_{\rm obs}|\lambda_{\rm true}, z_{\rm true})

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.

        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|\lambda_{\rm true}, z_true) in lambda_obs bins.
            Dimensions: (lambda_obs_edges, z_true, \lambda_{\rm true}).
        """

        lambda_obs_bins_size = len(lambda_obs_edges) - 1
        if len(self.lambda_tab_integ) != lambda_obs_bins_size:
            raise ValueError(
                f"Number of bins from lambda_obs_edges ({lambda_obs_bins_size})"
                " is different from internal lambda_tab_integ"
                f" ({(len(self.lambda_tab_integ))}) setup!"
            )

        ################################################
        # Compute P(Delta lobs|ltrue, ztrue)
        ################################################

        windows_lambda_obs_lambda_true = np.zeros(
            (
                lambda_obs_bins_size,
                z_true.size,
                lambda_true.size,
            )
        )
        for ind_lambda in range(len(windows_lambda_obs_lambda_true)):
            # integrate P(lambda_obs|lambda_true, z) in lambda_obs
            l_tab = np.geomspace(
                lambda_obs_edges[ind_lambda],
                lambda_obs_edges[ind_lambda + 1],
                self.lambda_tab_integ[ind_lambda],
            )
            windows_lambda_obs_lambda_true[ind_lambda] = simps(
                self._prob_lambda_obs(
                    z_true[:, np.newaxis, np.newaxis],
                    lambda_true[np.newaxis, :, np.newaxis],
                    l_tab[np.newaxis, np.newaxis, :],
                ),
                x=l_tab,
                axis=-1,
            )

        ################################################
        # Compute P(Delta lobs|mass, ztrue)
        ################################################
        pdf_mass_richness_scaling = self.lambda_true_distribution.prob_richness(
            z_true, mass, lambda_true
        )

        return simps(
            pdf_mass_richness_scaling[np.newaxis, :, :, :]  # (1, z, M, ltr)
            * windows_lambda_obs_lambda_true[:, :, np.newaxis, :],  # (lobs, z, 1, ltr)
            x=lambda_true,
            axis=-1,
        )

    def window_redshift_richness_observed(
        self,
        z_obs_edges,
        lambda_obs_edges,
        z_true,
        mass,
        lambda_true,
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

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true, mass)
        """
        # Dimensions: (z, M, lambda_true)
        pdf_mass_richness_scaling = self.lambda_true_distribution.prob_richness(
            z_true, mass, lambda_true
        )
        # Dimensions: (z_obs_edges, lambda_obs_edges, z_true, lambda_true)
        window_lambda_true = (
            # Dimensions: (z_obs_edges, lambda_obs_edges, z_true, 1).
            self.window_z_observed(z_obs_edges, lambda_obs_edges, z_true)[
                :, :, :, np.newaxis
            ]
            # Dimensions: (1, lambda_obs_edges, z_true, lambda_true)
            * self.window_richness_observed(
                lambda_obs_edges,
                z_true,
                lambda_true,
                mass,
            )[np.newaxis, :, :, :]
        )
        return simps(
            pdf_mass_richness_scaling[np.newaxis, np.newaxis, :, :, :]
            * window_lambda_true[:, :, :, np.newaxis, :],
            x=lambda_true,
            axis=-1,
        )
