# General imports
# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps

from cloelib.observables.clusters.selection_function.mass_lambda_true.gaussian import (
    GaussianMassLambdaTrue,
)


class GaussianSelectionFunction:
    def __init__(
        self,
        A_l: float,
        B_l: float,
        C_l: float,
        sig_A_l: float,
        sig_B_l: float,
        sig_C_l: float,
        sig_lambda_norm: float,
        sig_lambda_z: float,
        sig_lambda_exponent: float,
        sig_z_z: float,
        sig_z_lambda: float,
        M_piv: float = 3.0e14,
        z_piv: float = 0.45,
    ):
        r"""
        Class defining the selection function of galaxy clusters, including
        sample purity, completeness, mass-observable relation, and
        uncertainties on observed quantities.

        Parameters
        ----------
        A_l : float
            Amplitude of the proxy - mass scaling relation
        B_l : float
            Mass slope of the proxy - mass scaling relation
        C_l : float
            Redshift slope of the proxy - mass scaling relation
        sig_A_l : float
            Amplitude of the proxy - mass scaling relation
            intrinsic scatter
        sig_B_l : float
            Mass slope of the proxy - mass scaling relation
            intrinsic scatter
        sig_C_l : float
            Redshift slope of the proxy - mass scaling relation
            intrinsic scatter
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
        M_piv: float
            Mass pivot in the proxy - mass relation, in h^{-1} Msun
        z_piv: float
            Redshift pivot in the proxy - mass relation
        """
        self.mass_lambda_true = GaussianMassLambdaTrue(
            A_l, B_l, C_l, sig_A_l, sig_B_l, sig_C_l, M_piv, z_piv
        )
        self.sig_lambda_norm = sig_lambda_norm
        self.sig_lambda_z = sig_lambda_z
        self.sig_lambda_exponent = sig_lambda_exponent
        self.sig_z_z = sig_z_z
        self.sig_z_lambda = sig_z_lambda

    def P_lnlbd(self, z, M, Lambda):
        r"""
        Proxy - mass relation PDF.

        Computes the theoretical richness probability distribution
        at the requested true mass, redshift, and richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.
        Lambda: numpy.ndarray
            True richness points.

        Returns
        -------
        P_lnlbd: numpy.ndarray
            P_lnlbd[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        return self.mass_lambda_true.P_lnlbd(z, M, Lambda)

    def scatter_lbdobs_lbd(self, z, Lambda):
        r"""
        Statistical uncertainty on the observed mass proxy.

        Computes the scatter of the observed richness PDF
        at the requested true redshift and richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        Lambda: numpy.ndarray
            True richness points.

        Returns
        -------
        scatter_lbobs_lbdz: numpy.ndarray
            scatter_lbobs_lbdz[i,j], where i is the true redshift axis
            and j is the true richness axis
        """
        return (
            self.sig_lambda_norm + self.sig_lambda_z * z[:, np.newaxis]
        ) * Lambda**self.sig_lambda_exponent

    def P_lbdobs_lbd(self, z, Lambda, Lambda_obs):
        r"""
        Observed mass proxy PDF.

        Computes the observed richness PDF at the requested
        true richness, true redshift, and observed richness points

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        Lambda: numpy.ndarray
            True richness points.
        Lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        P_lbdobs_lbd: numpy.ndarray
            P_lbdobs_lbd[i,j,k], where i is the redshift axis,
            j is the the theoretical richness axis,
            and k is the observed richness
        """
        sigma_lbdobslbd = self.scatter_lbdobs_lbd(z, Lambda)[:, :, np.newaxis]

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigma_lbdobslbd**2.0))
            * np.exp(
                -(
                    (
                        Lambda_obs[np.newaxis, np.newaxis, :]
                        - Lambda[np.newaxis, :, np.newaxis]
                    )
                    ** 2.0
                )
                / (2.0 * sigma_lbdobslbd**2.0)
            )
        )

    def scatter_zobs_z(self, Lambda_obs, z):
        r"""
        Statistical uncertainty on the observed redshift.

        Computes the scatter of the observed redshift PDF
        at the requested true redshift and observed richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        Lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        scatter_zobs_z: numpy.ndarray
            scatter_zobs_z[i,j] where i is the true redshift axis
            and j the observed richness axis
        """
        return self.sig_z_z * z + self.sig_z_lambda * Lambda_obs

    def P_zobs_z(self, z_obs, Lambda_obs, z):
        r"""
        Observed redshift PDF.

        Computes the observed redshift PDF at the requested
        true richness, true redshift, and observed redshift points.

        Parameters
        ----------
        z_obs: numpy.ndarray
            Observed redshift points.
        Lambda_obs: numpy.ndarray
            Observed richness points.
        z: numpy.ndarray
            True redshift points.

        Returns
        -------
        P_zobs_z: numpy.ndarray
            P_zobs_z[i,j,k] where i is the observed redshift axis,
            j is the observed richness axis,
            and k is the true redshift axis
        """
        sigmazobsz = self.scatter_zobs_z(Lambda_obs, z)

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigmazobsz**2.0))
            * np.exp(-((z_obs[:, np.newaxis] - z) ** 2.0) / (2.0 * sigmazobsz**2.0))
        )

    def window_z_observed(self, z_obs_edges, lambda_obs_edges, z_tab_sig, z_true):
        r"""Compute the window function of each observed redshift bin, given by:

        ..math:
            W_{\Delta z^{\rm obs}}(\lambda^{\rm obs}, z^{\rm true}) = \int_{\Delta z^{\rm obs}}dz^{\rm obs} P(z^{\rm obs}|\lambda^{\rm obs}, z^{\rm true})

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
            Dimentions: (z_obs_edges, lambda_obs_edges, z_true).
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
                self.P_zobs_z(_z_obs_tabs[:, ind_z], _lambda_obs, z_true),
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
            W_{\Delta\lambda^{\rm obs}}(\lambda_{\rm true}, z^{\rm true}) = \int_{\Delta\lambda^{\rm obs}}d\lambda^{\rm obs} P(\lambda^{\rm obs}|\lambda_{\rm true}, z^{\rm true})

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
            Dimentions: (lambda_obs_edges, z_true, \lambda_{\rm true}).
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
                self.P_lbdobs_lbd(
                    z_true,
                    lambda_true,
                    l_tab,
                ),
                x=l_tab,
                axis=-1,
            )
        return window_lambda_obs_lambda_true

    def window_richness_observed(
        self,
        z_true,
        lambda_true,
        mass,
        windows_lambda_obs_lambda_true,
    ):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda^{\rm obs}}(M, z^{\rm true}) = \int_{\Delta\lambda^{\rm obs}}d\lambda^{\rm obs} P(\lambda^{\rm obs}|M, z^{\rm true})

        Parameters
        ----------
        lambda_true : numpy.ndarray
            True richness to integrate the window.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        windows_lambda_obs_lambda_true : numpy.ndarray
            Integral P(lambda_obs|lambda_true, z_true) in lambda_obs bins,
            must be shape (lambda_obs_edges, z_true, lambda_true).


        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|M, z_true) in lambda_obs bins.
            Dimentions: (lambda_obs_edges, z_true, M).
        """

        # if external_richness_selection_function == 'CG_ESF' :
        #     window_lambda_obs  = self.int_Plobltr_Dlob[lambda_bin](self.tabulated_integrands["z_true"], self.tabulated_integrands["lambda_true"]).T

        window_lambda_obs = np.zeros(
            (
                *windows_lambda_obs_lambda_true.shape[:-1],
                mass.size,
            )
        )
        for ind_lambda, _window_lambda_obs in enumerate(windows_lambda_obs_lambda_true):
            # Window function
            window_lambda_obs[ind_lambda] = simps(
                self.P_lnlbd(z_true, mass, lambda_true)
                * _window_lambda_obs[:, np.newaxis, :],
                x=lambda_true,
                axis=-1,
            )
        return window_lambda_obs
