# General imports
import jax.numpy as np # type: ignore


class SelectionFunction:
    def __init__(self, A_l: float, B_l: float, C_l: float,
                 sig_A_l: float, sig_B_l: float, sig_C_l: float, 
                 sig_lambda_norm: float, sig_lambda_z: float, 
                 sig_lambda_exponent: float, sig_z_z: float,
                 sig_z_lambda: float):
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
        
    def lnlambda(self, z, M):
        r"""
        Computes the theoretical richness
        at the true redshift and mass requested


        Parameters
        ----------
        z: numpy.ndarray
                   True redshift at which to evaluate the theoretical richness
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness
               Units: h^{-1} Ms

        Returns
        -------
        lnlambda : numpy.ndarray
                lnlambda[i,j] where i is the true redhshift axis and
                                    j the mass axis
        """

        # 3e14 is in [Ms h^-1] units
        return (
            np.log(self.A_l)
            + self.B_l * np.log(M / (3.0e14))
            + self.C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + 0.45))
        )



    def scatter_lnl(self, z, M):
        r"""
        Computes the scatter of the theoretical richness probability distribution
        at the true redshift and mass requested

        Parameters
        ----------
        z: numpy.ndarray
                   True redshift at which to evaluate the theoretical richness scatter
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness scatter
               Units: h^{-1} Ms

        Returns
        -------
        scatter_lnl : float or numpy.ndarray
                scatter_lnl[i,j] where i is the true redhshift axis and
                                       j the mass axis
        """

        return (
            self.sig_A_l
            + self.sig_B_l * np.log(M / (3.0e14))
            + self.sig_C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + 0.45))
        )

    def P_lnlbd(self, z, M, Lambda):
        r"""
        Computes the theoretical richness probability distribution
        at the Mass, true redshift and theoretical richness requested


        Parameters
        ----------
        z: numpy.ndarray
                   True redshift at which to evaluate the theoretical richness scatter
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness scatter
               Units: Ms h^{-1}
        Lambda: numpy.ndarray
               Theoretical richness at which to evaluate the theoretical richness

        Returns
        -------
        P_lnlbd: float or numpy.ndarray
                P_lnlbd[i,j,k] where i is the redshift,
                                     j is the mass,
                                     k is the observed richness
        """

        lnlambda1 = self.lnlambda(z, M)[:, :, np.newaxis]
        sigmalnl = self.scatter_lnl(z, M)[:, :, np.newaxis]
        Lambda = Lambda[np.newaxis, np.newaxis, :]

        return (
            1.0
            / (Lambda * np.sqrt(2.0 * np.pi * sigmalnl**2.0))
            * np.exp(-((np.log(Lambda) - lnlambda1) ** 2.0) / (2.0 * sigmalnl**2.0))
        )

    def scatter_lbdobs_lbd(self, z, lbd):
        r"""
        Computes the scatter of the observed richness probability distribution
        at the true redshift and theoretical richness requested


        Parameters
        ----------
        z: numpy.ndarray
                   True redshift at which to evaluate the scatter of the observed richness probability distribution
        lbd: float or numpy.ndarray
               Theoretical richness at which to evaluate the scatter of the observed richness probability distribution

        Returns
        -------
        scatter_lbobs_lbdz: float or numpy.ndarray
                scatter_lbobs_lbdz[i,j] where i is the redshift axis
                                              j is the theoretical richness axis
        """

        return (
            self.sig_lambda_norm + self.sig_lambda_z * z[:, np.newaxis]
        ) * lbd**self.sig_lambda_exponent

    def P_lbdobs_lbd(self, z, lbd, lbd_obs):
        r"""
        Computes the observed richness probability distribution
        at the theoretical richness and the true redshift and observed richness requested


        Parameters
        ----------
        z: numpy.ndarray
               True redshift at which to evaluate the observed richness probability distribution
        lbd: numpy.ndarray
                   Theoretical richness at which to evaluate the observed richness probability distribution
        lbd_obs: numpy.ndarray
               Observed richness at which to evaluate the observed richness probability distribution

        Returns
        -------
        P_lbdobs_lbd: float or numpy.ndarray
                P_lbdobs_lbd[i,j,k] where i is the redshift axis
                                          j is the the theoretical richness axis,
                                          k is the observed richness
        """

        sigma_lbdobslbd = self.scatter_lbdobs_lbd(z, lbd)[:, :, np.newaxis]

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigma_lbdobslbd**2.0))
            * np.exp(
                -(
                    (
                        lbd_obs[np.newaxis, np.newaxis, :]
                        - lbd[np.newaxis, :, np.newaxis]
                    )
                    ** 2.0
                )
                / (2.0 * sigma_lbdobslbd**2.0)
            )
        )

    def scatter_zobs_z(self, lbd_obs, z):
        r"""
        Computes the scatter of the observed redshift probability distribution
        at the true redshift and observed richness requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate scatter_zobs_z
        lbd: float or numpy.ndarray
               Observed richness at which to evaluate scatter_zobs_z

        Returns
        -------
        scatter_zobs_z: float or numpy.ndarray
                scatter_zobs_z[i,j] where i is the true redshift axis
                                          j the observed richness axis
        """

        return self.sig_z_z * z + self.sig_z_lambda * lbd_obs

    def P_zobs_z(self, zed_obs, lbd_obs, z):
        r"""
        Computes the observed redshift probability distribution
        at the theoretical richness and the true redshift and observed redshift requested


        Parameters
        ----------
        z: numpy.ndarray
               True redshift at which to evaluate the observed redshift probability distribution
        lbd: numpy.ndarray
                   Observed richness at which to evaluate the observed redshift probability distribution
        zed_obs: numpy.ndarray
               Observed redshift at which to evaluate the observed redshift probability distribution

        Returns
        -------
        P_zobs_z: float or numpy.ndarray
                P_zobs_z[i,j,k] where i is the observed redshift axis
                                      j is the observed richness axis
                                      k is the true redshift axis
        """

        sigmazobsz = self.scatter_zobs_z(lbd_obs, z)

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigmazobsz**2.0))
            * np.exp(-((zed_obs[:, np.newaxis] - z) ** 2.0) / (2.0 * sigmazobsz**2.0))
        )
