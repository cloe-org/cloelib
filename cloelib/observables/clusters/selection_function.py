# General imports
# import jax.numpy as np
import numpy as np


class SelectionFunction:
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
        self.A_l = A_l
        self.B_l = B_l
        self.C_l = C_l
        self.sig_A_l = sig_A_l
        self.sig_B_l = sig_B_l
        self.sig_C_l = sig_C_l
        self.sig_lambda_norm = sig_lambda_norm
        self.sig_lambda_z = sig_lambda_z
        self.sig_lambda_exponent = sig_lambda_exponent
        self.sig_z_z = sig_z_z
        self.sig_z_lambda = sig_z_lambda
        self.M_piv = M_piv
        self.z_piv = z_piv

    def lnlambda(self, z, M):
        r"""
        Mean of the richness-mass relation PDF.

        Computes the theoretical richness at
        the requested true redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        lnlambda : numpy.ndarray
            lnlambda[i,j], where i is the true redhshift axis and j the mass axis
        """
        return (
            np.log(self.A_l)
            + self.B_l * np.log(M / (self.M_piv))
            + self.C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_piv))
        )

    def scatter_lnl(self, z, M):
        r"""
        Intrinsic scatter of the proxy - mass relation.

        Computes the scatter of the theoretical richness probability distribution
        at the requested true redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        scatter_lnl : numpy.ndarray
            scatter_lnl[i,j], where i is the true redhshift axis and j the mass axis
        """

        return (
            self.sig_A_l
            + self.sig_B_l * np.log(M / (self.M_piv))
            + self.sig_C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_piv))
        )

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
        lnlambda1 = self.lnlambda(z, M)[:, :, np.newaxis]
        sigmalnl = self.scatter_lnl(z, M)[:, :, np.newaxis]
        Lambda = Lambda[np.newaxis, np.newaxis, :]

        return (
            1.0
            / (Lambda * np.sqrt(2.0 * np.pi * sigmalnl**2.0))
            * np.exp(-((np.log(Lambda) - lnlambda1) ** 2.0) / (2.0 * sigmalnl**2.0))
        )

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
