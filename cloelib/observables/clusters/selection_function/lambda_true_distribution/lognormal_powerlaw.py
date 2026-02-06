# General imports
# import jax.numpy as np
import numpy as np

from cloelib.observables.clusters.auxiliary import tabulated_return


class LognormalPowerLawLambdaTrueDistribution:
    def __init__(
        self,
        A_l: float,
        B_l: float,
        C_l: float,
        sig_A_l: float,
        sig_B_l: float,
        sig_C_l: float,
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
        self.M_piv = M_piv
        self.z_piv = z_piv

        # to avoid recomputing prob_richness
        self._tabulated_prob_richness = {
            "inputs": {
                "M": None,
                "z": None,
                "lambda_true": None,
            },
            "values": None,
        }

    def mean_lnrichness(self, z, M):
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
        lnrichness : numpy.ndarray
            lnrichness[i,j], where i is the true redhshift axis and j the mass axis
        """
        return (
            np.log(self.A_l)
            + self.B_l * np.log(M / (self.M_piv))
            + self.C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_piv))
        )

    def scatter_lnrichness(self, z, M):
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
        scatter_lnrichness : numpy.ndarray
            scatter_lnrichness[i,j], where i is the true redhshift axis and j the mass axis
        """

        return (
            self.sig_A_l
            + self.sig_B_l * np.log(M / (self.M_piv))
            + self.sig_C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_piv))
        )

    def _prob_richness(self, z, M, lambda_true):
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
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        prob_richness: numpy.ndarray
            prob_richness[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        _mean_lnlambda = self.mean_lnrichness(z, M)[:, :, np.newaxis]
        _sigma_lnrichness = self.scatter_lnrichness(z, M)[:, :, np.newaxis]
        _lambda_true = lambda_true[np.newaxis, np.newaxis, :]

        return (
            1.0
            / (_lambda_true * np.sqrt(2.0 * np.pi * _sigma_lnrichness**2.0))
            * np.exp(
                -((np.log(_lambda_true) - _mean_lnlambda) ** 2.0)
                / (2.0 * _sigma_lnrichness**2.0)
            )
        )

    def prob_richness(self, z, M, lambda_true):
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
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        prob_richness: numpy.ndarray
            prob_richness[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        return tabulated_return(
            self._tabulated_prob_richness,
            self._prob_richness,
            {"z": z, "M": M, "lambda_true": lambda_true},
        )
