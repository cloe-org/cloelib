import numpy as np

from cloelib.observables.clusters.halo_statistics import HaloStatistics
from cloelib.cosmology import derived_cosmology


class TinkerHMFBias:

    def __init__(self, halo_statistics: HaloStatistics):

        self.halo_statistics = halo_statistics

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.halo_statistics.perturbations.background

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes the Tinker et al. (2008) multiplicity function
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        f_sigma_nu: numpy.ndarray
            f_sigma_nu[i,j], where i is the redshift axis and j the mass axis
        """
        raise NotImplementedError

    def bias(self, z, M):
        r"""
        Computation of the halo bias.

        Computes the Tinker et al. (2010) halo bias
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        bias: numpy.ndarray
            bias[i,j], where i is the redshift axis and j the mass axis
        """
        Delta = self.halo_statistics.get_Delta_crit(z) / self.halo_statistics._Omega_m(
            z
        )

        # parameters
        p = [1.0, 0.24, 0.44, 0.88, 0.183, 1.5, 0.019, 0.107, 0.19, 2.4]
        y = np.log10(Delta)
        A_par = p[0] + p[1] * y * np.e ** (-((4.0 / y) ** 4))
        a_par = p[2] * y - p[3]
        B_par = p[4]
        b_par = p[5]
        C_par = p[6] + p[7] * y + p[8] * np.e ** (-((4.0 / y) ** 4))
        c_par = p[9]

        # bias
        nu = self.halo_statistics.nu_z_M(z, M).T
        return (
            1.0
            - A_par * nu**a_par / (nu**a_par + self.halo_statistics.delta_c(z) ** a_par)
            + B_par * nu**b_par
            + C_par * nu**c_par
        ).T

    def dn_dm(self, z, M):
        r"""Derivative of the number density.

        Computes the derivative of the number density
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        dn_dm: numpy.ndarray
            dn_dm[i,j], where i is the redshift axis and j the mass axis.
            Units: h^4 Mpc^{-3} Ms^{-1}.
        """
        dlnsigmadlnR = self.halo_statistics.dlns_dlnR(z, M)
        rho_mean_0 = self.halo_statistics._Omega_m(0) * derived_cosmology.rho_crit(
            self.background, 0.0
        )
        rho_mean_0 /= self.background.h**2.0

        return rho_mean_0 / M**2.0 * self.f_sigma_nu(z, M) * dlnsigmadlnR / (-3)
