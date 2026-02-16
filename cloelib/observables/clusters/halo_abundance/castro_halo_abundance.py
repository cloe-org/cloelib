import numpy as np
from scipy import interpolate
from scipy.special import gamma

from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.matter_statistics import MatterStatistics

from .halo_abundance_core import HaloAbundanceCore


class CastroHaloAbundance:

    def __init__(self, matter_statistics: MatterStatistics):        
        """
        Class implementing the Castro et al. mass abundance models.

        Castro et al. followed the cold dark matter
        prescription by Costanzi+13 (https://arxiv.org/abs/1311.1514) and
        Castorina+13 (https://arxiv.org/pdf/1311.1212). That is, the halo mass function
        and halo bias do not include the massive neutrino contribution in the
        computation of mass variance and power spectrum.
        """
        self.core = HaloAbundanceCore(matter_statistics)

    # For sanity checks in summary_statistics
    @property
    def overdensity_type(self):
        return "vir"

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes the Castro et al. (2023) multiplicity function
        at the requested redshift and mass points. Valid only for
        virial overdensities.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        f_sigma_nu: numpy.ndarray
            f_sigma_nu[i,j], where i is the redshift axis and j the mass axis
        """
        # compute inputs
        Omega_m = self.core.matter_statistics.background.Omega_m(z)
        dlnsigmadlnM = self.core.dlns_dlnM(z, M)
        nu = self.core.nu_z_M(z, M)

        ##################
        # HMF computations
        ##################

        a1 = 0.7962
        a2 = 0.1449
        az = -0.0658
        p1 = -0.5612
        p2 = -0.4743
        q1 = 0.3688
        q2 = -0.2804
        qz = 0.0251

        # Compute main quantities
        dlnsigmadlnR = 3 * dlnsigmadlnM

        aR = a1 + a2 * (dlnsigmadlnR + 0.6125) ** 2.0
        a = aR * Omega_m[:, np.newaxis] ** az
        p = p1 + p2 * (dlnsigmadlnR + 0.5)
        qR = q1 + q2 * (dlnsigmadlnR + 0.5)
        q = qR * Omega_m[:, np.newaxis] ** qz
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
        at the requested redshift and mass points. Valid only
        for virial overdensities.

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
        # Format mass shape
        if not hasattr(M, "__len__"):
            M = [M]
        M = np.asarray(M)
        lenM_orig = M.size
        if lenM_orig < 4:
            M = np.append(M, M[-1] * np.arange(2, 6))

        # compute inputs
        Omega_m = self.core.matter_statistics.background.Omega_m(z)
        delta_c = self.core.delta_c(z)
        dlnsigmadlnM = self.core.dlns_dlnM(z, M)
        nu = self.core.nu_z_M(z, M)

        ###################
        # Bias computations
        ###################

        # Compute main quantities
        dlnsigmadlnR = 3 * dlnsigmadlnM
        fsigmanu = self.f_sigma_nu(z, M)
        S8 = self.core.sigma8 * np.sqrt(self.core.matter_statistics.Omega_m / 0.3)

        dlnfsigmanu_dlnnu = np.zeros(fsigmanu.shape)
        for i in range(len(Omega_m)):
            fsigmanu_int = interpolate.splrep(np.log(nu[i]), np.log(fsigmanu[i]), s=0)
            dlnfsigmanu_dlnnu[i] = interpolate.splev(np.log(nu[i]), fsigmanu_int, der=1)

        # parameters
        A0, a1, b1, b2, c1 = 1.150, 0.0929, 0.256, 0.173, -0.0372
        b_pbs = 1 - 1 / delta_c[:, np.newaxis] * dlnfsigmanu_dlnnu
        f0 = 1 + a1 * Omega_m[:, np.newaxis]
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
        return self.core.dn_dm_fsigmanu(z, M, self.f_sigma_nu(z, M))
