import numpy as np
from scipy import interpolate
from scipy.special import gamma

from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.halo_statistics import HaloStatistics


class CastroHMFBias:

    def __init__(self, halo_statistics: HaloStatistics):

        self.halo_statistics = halo_statistics

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.halo_statistics.perturbations.background

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes the Castro et al. (2023) multiplicity function
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
        a1 = 0.7962
        a2 = 0.1449
        az = -0.0658
        p1 = -0.5612
        p2 = -0.4743
        q1 = 0.3688
        q2 = -0.2804
        qz = 0.0251

        dlnsigmadlnR = self.halo_statistics.dlns_dlnR(z, M)
        Ommz = self.halo_statistics._Omega_m(z)[:, np.newaxis]
        nu = self.halo_statistics.nu_z_M(z, M)

        aR = a1 + a2 * (dlnsigmadlnR + 0.6125) ** 2.0
        a = aR * Ommz**az
        p = p1 + p2 * (dlnsigmadlnR + 0.5)
        qR = q1 + q2 * (dlnsigmadlnR + 0.5)
        q = qR * Ommz**qz
        A = 1.0 / (
            2.0 ** (-0.5 - p + q / 2.0)
            / np.sqrt(np.pi)
            * (2.0**p * gamma(q / 2.0) + gamma(-p + q / 2.0))
        )

        return (
            A
            * np.sqrt(2.0 * a / (np.pi))
            * np.exp(-a * nu**2.0 / 2.0)
            * (1.0 + 1.0 / (a * nu**2.0) ** p)
            * (nu * np.sqrt(a)) ** (q - 1.0)
        ) * nu

    def bias(self, z, M):
        r"""
        Computation of the halo bias.

        Computes the Castro et al. (2024) halo bias
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

        Notes
        -------
        If the mass array has less than 4 entries, this causes problem with the derivative
        """
        if not hasattr(M, "__len__"):
            M = [M]
        M = np.asarray(M)
        lenM_orig = M.size
        if lenM_orig < 4:
            M = np.append(M, M[-1] * np.arange(2, 6))

        dlnsigmadlnR = self.halo_statistics.dlns_dlnR(z, M)
        Ommz = self.halo_statistics._Omega_m(z)[:, np.newaxis]
        S8 = self.halo_statistics.sigma8 * np.sqrt(
            self.halo_statistics._Omega_m(0.0) / 0.3
        )

        nu = self.halo_statistics.nu_z_M(z, M)
        nufnu = self.f_sigma_nu(z, M)
        dlnnufnu_dlnnu = np.zeros(nufnu.shape)
        for i in range(len(z)):
            nufnu_int = interpolate.splrep(np.log(nu[i]), np.log(nufnu[i]), s=0)
            dlnnufnu_dlnnu[i] = interpolate.splev(np.log(nu[i]), nufnu_int, der=1)

        # parameters
        A0, a1, b1, b2, c1 = 1.150, 0.0929, 0.256, 0.173, -0.0372
        b_pbs = 1 - 1 / self.halo_statistics.delta_c(z)[:, np.newaxis] * dlnnufnu_dlnnu
        f0 = 1 + a1 * Ommz
        f1 = 1 + b1 * dlnsigmadlnR + b2 * dlnsigmadlnR**2
        f2 = 1 + c1 * S8

        # bias
        bias = A0 * f0 * f1 * f2 * b_pbs

        # original mass array size
        if lenM_orig < len(M):
            bias = bias[:, :lenM_orig]

        return bias

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
