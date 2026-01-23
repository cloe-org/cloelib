import numpy as np

from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.halo_model import HaloModel

from .hmf_bias_auxiliary import HMFBiasAuxiliary


class TinkerHMFBias:

    def __init__(self, halo_model: HaloModel):

        self.halo_model = halo_model
        self.auxiliary = HMFBiasAuxiliary(halo_model)

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes the Tinker et al. (2008) multiplicity function
        at the requested redshift and mass points.

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
        # compute inputs
        delta_c = self.halo_model.delta_c(z)
        nu = self.halo_model.nu_z_M(z, M)
        Delta = self.halo_model.get_Delta_crit(z) / self.halo_model._Omega_m(z)

        ###################
        # Bias computations
        ###################

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
        _nu = nu.T
        return (
            1.0
            - A_par * _nu**a_par / (_nu**a_par + delta_c**a_par)
            + B_par * _nu**b_par
            + C_par * _nu**c_par
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
        return self.auxiliary.dn_dm_fsigmanu(z, M, self.f_sigma_nu(z, M))
