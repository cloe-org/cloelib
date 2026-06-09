import numpy as np

from cloelib.auxiliary.cluster_helpers import convert_to_Delta_crit
from cloelib.observables.clusters.halo_model_properties import HaloModelProperties

from .halo_abundance_base import HaloAbundanceBase


class TinkerHaloAbundance(HaloAbundanceBase):
    """
    Class implementing the Tinker et al. mass abundance models.

    Following the cold dark matter
    prescription by Costanzi+13 (https://arxiv.org/abs/1311.1514) and
    Castorina+13 (https://arxiv.org/pdf/1311.1212), the halo mass function
    and halo bias do not include the massive neutrino contribution in the
    computation of mass variance, power spectrum, and overdensity.
    """

    def __init__(
        self,
        halo_model_properties: HaloModelProperties,
        overdensity_type: str = "vir",
        overdensity: int = 200,
    ):
        HaloAbundanceBase.__init__(self, halo_model_properties)
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity

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
        delta_c = self.delta_c(z)
        nu = self.nu_z_M(z, M)
        Delta = convert_to_Delta_crit(
            self.overdensity_type,
            self.overdensity,
            self.halo_model_properties.background,
            z,
        ) / self.halo_model_properties.background.Omega_cb(z)

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
