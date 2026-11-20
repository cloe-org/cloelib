# General imports
# import jax.numpy as np
import numpy as np
import scipy.special as spc

from cloelib.observables.halos.halo_abundance.halo_abundance_base import (
    tabulated_return,
)


class ShiftedPoissonHaloMassObservable:
    def __init__(
        self,
        M_min_cen: float,
        M_min_sat: float,
        alpha: float,
        epsilon: float,
        sigma_lnltr: float,
        z_piv: float = 1.25,
    ):
        r"""
        Class defining the observable-mass relation as
        a shifted continuos Poisson distribution.

        Parameters
        ----------
        M_min_cen : float
            Minimum halo mass to host a central galaxy
        M_min_sat : float
            Minimum halo mass to host one satellite galaxy
        alpha : float
            Mass slope of the proxy - mass scaling relation
        epsilon : float
            Redshift evolution of the proxy-mass scaling relation
        sigma_lnltr: float
            Variance of the proxy-mass scaling relation
        z_piv: float
            Redshift pivot in the proxy - mass relation
        """
        self.M_min_cen = M_min_cen
        self.M_min_sat = M_min_sat
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma_lnltr = sigma_lnltr
        self.z_piv = z_piv

        # to avoid recomputing pdf_richness
        self._tabulated_pdf_richness = {
            "inputs": {
                "M": None,
                "z": None,
                "lambda_true": None,
            },
            "values": None,
        }

    def _mean_richness(self, z, M):
        r"""
        Mean of the richness-mass relation PDF.

        Computes the theoretical richness at
        the requested true redshift and mass points.

        .. math::
            \lambda_{\rm true}(z, M) = 1 + \left(\frac{M - M_{\text{min}}}{M_1 - M_{\text{min}}}\right)^{\alpha}
            \left(\frac{1 + z}{1 + z_{\text{piv}}}\right)^{\epsilon}

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        lnrichness : numpy.ndarray
            ln(richness), with the same dimensions of the
            operation z x M.
        """

        lsat = (
            (M - self.M_min_cen) / (self.M_min_sat - self.M_min_cen)
        ) ** self.alpha * ((1.0 + z) / (1.0 + self.z_piv)) ** self.epsilon
        return 1.0 + lsat

    def scatter_richness(self, z, M):
        r"""
        Intrinsic scatter of the proxy - mass relation.

        Computes the scatter of the theoretical richness probability distribution
        at the requested true redshift and mass points.

        .. math::
            \text{scatter} = \sigma_{\text{lnltr}} \cdot l_{\text{sat}} =
            \sigma_{\text{lnltr}} \cdot (\text{\_mean\_lnrichness}(z, M) - 1)

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        scatter_lnrichness : numpy.ndarray
            Scatter of ln(richness), with the same dimensions of the
            operation z x M.
        """

        return self.sigma_lnltr * (self._mean_richness(z, M) - 1.0)

    def _pdf_richness(self, z, M, lambda_true):
        r"""
        Proxy - mass relation PDF.

        Computes the theoretical richness probability distribution
        at the requested true mass, redshift, and richness points.

        Continuous Poisson shift-corrected by the variance.
        .. math::
            P(\lambda_{\text{true}} \mid z, M) = \frac{e^{-\Lambda} \cdot
            \Lambda^{X - 1}}{\Gamma(X)}

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
        pdf_richness: numpy.ndarray
            PDF of richness.
        """

        m = self._mean_richness(z, M) - 1.0  # lsat
        std_richness = self.scatter_richness(z, M)
        std = np.sqrt(m + std_richness**2.0)
        x = lambda_true + std_richness**2.0
        lam = std**2.0
        ln_gamma_fun = spc.gammaln(x)
        return np.exp(
            -lam + (x - 1.0) * np.log(lam) - ln_gamma_fun, dtype=np.longdouble
        )

    def pdf_richness(self, z, M, lambda_true):
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
        pdf_richness: numpy.ndarray
            pdf_richness[i,j,k], where i is the redshift, j is the mass,
            and k is the true richness index
        """
        return tabulated_return(
            self._tabulated_pdf_richness,
            self._pdf_richness,
            {
                "z": np.atleast_1d(z)[:, np.newaxis, np.newaxis],
                "M": np.atleast_1d(M)[np.newaxis, :, np.newaxis],
                "lambda_true": np.atleast_1d(lambda_true)[np.newaxis, np.newaxis, :],
            },
        )
