# General imports
# import jax.numpy as np
import numpy as np


class GaussianMassLambdaTrue:
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
        tabulte_P_lnlbd: bool = True,
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

        self.tabulte_P_lnlbd = tabulte_P_lnlbd
        # to avoid recomputing P_lnlbd
        self._tabulated_P_lnlbd_args = {
            "M": None,
            "z": None,
            "lambda_true": None,
        }
        self._tabulated_P_lnlbd = None

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

    def _P_lnlbd(self, z, M, Lambda):
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

    def _are_args_tabulated(self, z, M, lambda_true):
        """Check if args are the tabluated values"""
        if any(value is None for key, value in self._tabulated_P_lnlbd_args.items()):
            return False
        _locals = locals()
        for name, ref_val in self._tabulated_P_lnlbd_args.items():
            test_val = _locals[name]
            if len(ref_val) != len(test_val):
                return False
            elif (ref_val != test_val).any():
                return False
        return True

    def P_lnlbd(self, z, M, lambda_true):
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
        P_lnlbd: numpy.ndarray
            P_lnlbd[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        if not self.tabulte_P_lnlbd or not self._are_args_tabulated(z, M, lambda_true):
            self._tabulated_P_lnlbd_args["M"] = M
            self._tabulated_P_lnlbd_args["z"] = z
            self._tabulated_P_lnlbd_args["lambda_true"] = lambda_true
            self._tabulated_P_lnlbd = self._P_lnlbd(z, M, lambda_true)

        return self._tabulated_P_lnlbd
