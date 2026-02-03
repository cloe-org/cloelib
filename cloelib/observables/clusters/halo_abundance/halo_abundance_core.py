import numpy as np
from scipy.integrate import simpson as simps

from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.matter_statistics import MatterStatistics


class HaloAbundanceCore:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
    ):
        r"""Auxiliary class computing quantities used in halo mass function and halo bias models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        matter_statistics : MatterStatistics
            An object from the `MatterStatistics` class.
        """
        self.matter_statistics = matter_statistics

        # to avoid recomputing sigma & dsigmadlnM
        self._tabulated_sigma = {
            "M": None,
            "z": None,
            "values": None,
        }
        self._tabulated_dlnsigmadlnM = {
            "M": None,
            "z": None,
            "values": None,
        }

        # internal value of sigma8
        self.__sigma8 = None

    def _are_mass_and_z_tabulated(self, z, M, reference_table):
        """Check if mass and redshift are the tabluated values"""
        if any(reference_table[key] is None for key in "Mz"):
            return False
        for name, test_val in (("M", M), ("z", z)):
            if len(reference_table[name]) != len(test_val):
                return False
            elif (reference_table[name] != test_val).any():
                return False
        return True

    @property
    def sigma8(self):
        r"""Returns the `sigma_8` value at redshift `z=0`. If `sigma_8`
        is not set as a base parameter, it is computed from the power spectrum.
        """
        if self.__sigma8 is None:
            self.__sigma8 = self.sigma_z_R([0.0], np.array([8.0]))
        return self.__sigma8

    def dn_dm_fsigmanu(self, z, M, fsigmanu):
        r"""Derivative of the number density with pre-computed
        halo mass function.

        Computes the derivative of the number density
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.
        fsigmanu: numpy.ndarray
            Multiplicity function.

        Returns
        -------
        dn_dm: numpy.ndarray
            dn_dm[i,j], where i is the redshift axis and j the mass axis.
            Units: h^4 Mpc^{-3} Ms^{-1}.
        """
        rho_mean_0 = self.matter_statistics._Omega_m(0) * derived_cosmology.rho_crit(
            self.matter_statistics.background, 0.0
        )
        rho_mean_0 /= self.matter_statistics.background.h**2.0

        return -rho_mean_0 / M**2.0 * fsigmanu * self.dlns_dlnM(z, M)

    def window(self, k, R):
        r"""Top-hat window and its derivative.

        Computes the top-hat window function and its derivative.

        Parameters
        ----------
        k: numpy.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}
        R: numpy.ndarray
               Radius where W(kR) is evaluated.
               Units: h^{-1} Mpc

        Returns
        -------
        W: numpy.ndarray
            W[i,j], where i is the wavenumber axis and j the radius axis
        dWdx: numpy.ndarray
            dWdx[i,j], where i is the wavenumber axis and j the radius axis
        """
        x = R[:, np.newaxis] * k
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3.0
        dWdx = 3.0 * (np.sin(x) * (x**2.0 - 3.0) + 3.0 * x * np.cos(x)) / x**4.0

        return W, dWdx

    def radius_M(self, M):
        r"""Radius from a mass.

        Converts a mass into a radius.

        Parameters
        ----------
        M: numpy.ndarray
            Mass in h^{-1} Msun

        Returns
        -------
        radius_M: array
            Radius in h^{-1} Mpc
        """
        rho_m_0 = (
            derived_cosmology.rho_crit(self.matter_statistics.background, 0.0)
            * self.matter_statistics._Omega_m(0.0)
            / self.matter_statistics.background.h**2.0
        )
        return (M / rho_m_0 * (3.0 / (4.0 * np.pi))) ** (1 / 3.0)

    def sigma_z_R(self, z, R):
        r"""Standard deviation of perturbations given a redshift and radius.

        Computes the rms at the radii requested from
        the table given by the Boltzman code.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        R: numpy.ndarray
            Radius points in h^{-1} Mpc.

        Returns
        -------
        sigma_z_R: numpy.ndarray
            sigma_z_R[i,j], where i is the redshift axis and j the radius axis.
        """
        k = self.matter_statistics.k  # h/Mpc
        W, _ = self.window(k, R)
        return np.sqrt(
            (
                1
                / (2.0 * np.pi**2)
                * simps(
                    (k**2.0).reshape(1, 1, len(k))
                    * self.matter_statistics.matter_power_spectrum(z, k).reshape(
                        len(z), 1, len(k)
                    )
                    * (W**2.0).reshape(1, len(R), len(k)),
                    x=k,
                    axis=-1,
                )
            )
        )

    def sigma_z_M(self, z, M):
        r"""Standard deviation of perturbations given a redshift and mass.

        Computes the rms at the masses requested from
        the table given by the Boltzman code.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        sigma_z_M: numpy.ndarray
            sigma_z_M[i,j], where i is the redshift axis and j the mass axis.
        """

        if not self._are_mass_and_z_tabulated(z, M, self._tabulated_sigma):
            R = self.radius_M(M)  # Mpc/h
            self._tabulated_sigma["M"] = M
            self._tabulated_sigma["z"] = z
            self._tabulated_sigma["values"] = self.sigma_z_R(z, R)

        return self._tabulated_sigma["values"]

    def delta_c(self, z):
        r"""Critical overdensity.

        Computes the critical overdensity at a given redshift
        following an approximation from Kitayama & Suto (1999).

        Parameters
        ----------
        z: float or np.ndarray
            Redshift at which to evaluate the delta_c

        Returns
        -------
        delta_c:  float or numpy.ndarray
            Value of the critical overdensity a given redshift.
        """
        return (
            3.0
            / 20.0
            * (12.0 * np.pi) ** (2.0 / 3.0)
            * (1.0 + 0.012299 * np.log10(self.matter_statistics.background.Omega_m(z)))
        )

    def nu_z_M(self, z, M):
        r"""Peak height.

        Computes the critical overdensity over the rms,
        delta_c/sigma, at a given redshift and mass.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points.

        Returns
        -------
        nu_z_M: numpy.ndarray
            nu_z_M[i,j], where i is the redshift axis and j the mass axis.
        """
        return self.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def dlns_dlnM(self, z, M):
        r"""Derivative of the logarithmic rms.

        Computes the derivative of the ln rms
        with respect to the ln of mass
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        dlns_dlnM: numpy.ndarray
            dlns_dlnM[i,j], where i is the redshift axis and j the mass axis.
        """

        if not self._are_mass_and_z_tabulated(z, M, self._tabulated_dlnsigmadlnM):

            k = self.matter_statistics.k  # h/Mpc
            R = self.radius_M(M)  # Mpc/h
            W, dWdx = self.window(k, R)
            dsigma2_dlnR = (
                R
                * np.pi**-2
                * simps(
                    k.reshape(1, 1, len(k)) ** 3
                    * self.matter_statistics.matter_power_spectrum(z, k).reshape(
                        len(z), 1, len(k)
                    )
                    * W.reshape(1, len(R), len(k))
                    * dWdx.reshape(1, len(R), len(k)),
                    x=k,
                    axis=-1,
                )
            )
            dsigma2_dlnM = dsigma2_dlnR / 3
            sigma = self.sigma_z_M(z, M)

            self._tabulated_dlnsigmadlnM["M"] = M
            self._tabulated_dlnsigmadlnM["z"] = z
            self._tabulated_dlnsigmadlnM["values"] = dsigma2_dlnM / (2 * sigma**2)

        return self._tabulated_dlnsigmadlnM["values"]
