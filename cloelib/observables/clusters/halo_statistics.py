from cloelib.cosmology.cosmology_clusters import Perturbations

import jax.numpy as np
from scipy.integrate import simpson as simps
from scipy.special import gamma
from scipy import integrate, interpolate


class HaloStatistics:
    @property
    def background(self):
        return self.perturbations.background

    @property
    def sigma8(self):
        if self.__sigma8 is None:
            self.__sigma8 = self.sigma_z_R(z, np.array([8.0]))
        return self.__sigma8

    def __init__(
        self,
        perturbations: Perturbations,
        overdensity_type: str,
        overdensity: int = 200,
        nonu: bool = False,
        k_div: int = 500,
        k_min: float = 1.0e-4,
        k_max: float = 1.0e2,
    ):
        self.perturbations = perturbations

        if overdensity_type not in ["crit", "mean", "vir"]:
            raise ValueError("Invalid overdensity definition, %s." % overdensity_type)
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity

        self.nonu = nonu

        # wavelength array (integration variable)
        self.k = np.geomspace(k_min, k_max, k_div)

        self.nonu = nonu  # self.theory["obs_specifications"]["CG"]["nonu"]
        self._camb_delta = "delta_nonu" if nonu else "delta_tot"

        # internal value of sigma8
        self.__sigma8 = None

    def window(self, k, R):
        r"""
        Computes the top-hat window function and its derivative

        Parameters
        ----------
        k: numpy.ndarray
               Wavenumber at which to evaluate W(kR)
               Units:  h Mpc^{-1}
        R: numpy.ndarray
               Radius at which to evaluate W(kR)
               Units: h^{-1} Mpc

        Returns
        -------
        W:   numpy.ndarray
             W[i,j] where i is the wavenumber axis and
                j the radius axis
        dWdx: numpy.ndarray
              dWdx[i,j] where i is the wavenumber axis and
                j the radius axis
        """

        x = R[:, np.newaxis] * k
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3.0
        dWdx = 3.0 * (np.sin(x) * (x**2.0 - 3.0) + 3.0 * x * np.cos(x)) / x**4.0

        return W, dWdx

    def radius_M(self, M):
        r"""
        Convert the requested mass in the associated radius_M

        Parameters
        ----------
        M: numpy.ndarray
              Mass at which the radius is
              to be estimated in h^{-1} Ms

        Returns
        -------
        radius_M: array
                Radius_M in h^{-1} Mpc
        """
        rho_m_0 = self.background.rho_crit(0.0) * self.background.Omega_m(
            0.0, self.nonu
        )
        return (M / rho_m_0 * (3.0 / (4.0 * np.pi))) ** (1 / 3.0)

    def delta_c(self, z):
        r"""
        Computes the critical overdensity at a given redshift
        following an approximation from Kitayama & Suto (1999)

        Parameters
        ----------
        z: float or np.ndarray
            Redshift at which to evaluate the delta_c

        Returns
        -------
        delta_c:  float or numpy.ndarray
            Value of the critical overdensity a given redshift
        """

        return (
            3.0
            / 20.0
            * (12.0 * np.pi) ** (2.0 / 3.0)
            * (1.0 + 0.012299 * np.log10(self.background.Omega_m(z)))
        )

    def get_Delta_crit(self, z):
        r"""
        Critical overdensity factor.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.

        Returns
        -------
        overdensity: float or np.ndarray
            The overdensity factor which needs
            to be multiplied to the critical
            density in order to define an overdensity.

        Notes
        -----
        The function is returned for :math:`\rm \rho_c` in a density definition
        at a given redshift. The function returns :math:`\rm \Delta` for the
        critical density of the universe, :math:`\rm \Delta \Omega_{m}` for
        the mean matter density of the universe, :math:`\rm \Delta` determined
        by `Bryan & Norman 1998
        <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`_ Equation 6 for
        the virial density.
        """
        if self.overdensity_type == "crit":
            Delta = self.overdensity

        elif self.overdensity_type == "mean":
            Delta = self.overdensity * self.background.Omega_m(z, self.nonu)

        elif self.overdensity_type == "vir":
            x = self.background.Omega_m(z, self.nonu) - 1.0
            Delta = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

        return Delta

    def sigma_z_R(self, z, R):
        r"""
        Computes the rms at the radii requested from
        the table given by the Boltzman code

        Parameters
        ----------
        z: numpy.ndarray
                   Redshift at which to evaluate sigma_z_M
        R: numpy.ndarray
               Radius at which to evaluate sigma_z_R in h^{-1} Mpc

        Returns
        -------
        sigma_z_R: numpy.ndarray
                sigma_z_R[i,j] where i is the redshift axis and
                j the radius axis
        """
        k = self.k  # h/Mpc
        W, _ = self.window(k, R)
        return np.sqrt(
            (
                1
                / (2.0 * np.pi**2)
                * simps(
                    (k**2.0).reshape(1, 1, len(k))
                    * self.perturbations.matter_power_spectrum(
                        z, k, delta=self._camb_delta
                    ).reshape(len(z), 1, len(k))
                    * (W**2.0).reshape(1, len(R), len(k)),
                    k,
                    axis=-1,
                )
            )
        )

    def sigma_z_M(self, z, M):
        r"""
        Computes the rms at the masses requested from
        the table given by the Boltzman code

        Parameters
        ----------
        z: numpy.ndarray
                   Redshift at which to evaluate sigma_z_M
        M: numpy.ndarray
               Mass at which to evaluate sigma_z_M in h^{-1} Mpc

        Returns
        -------
        sigma_z_M: numpy.ndarray
                sigma_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """
        R = self.radius_M(M)  # Mpc/h
        return self.sigma_z_R(z, R)

    def nu_z_M(self, z, M):
        r"""
        Computes the critical overdensity over the rms
        delta_c/sigma at a given redshift and mass

        Parameters
        ----------
        z: numpy.ndarray
                   Redshift at which to evaluate nu_z_M
        M: numpy.ndarray
               Mass at which to evaluate nu_z_M

        Returns
        -------
        nu_z_M:   numpy.ndarray
            nu_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        return self.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def dlns_dlnR(self, z, M):
        r"""
        Computes the derivatives of the log rms
        with respect to the radius
        at the redshift and mass requested

        Parameters
        ----------
        z: numpy.ndarray
                   Redshift at which to evaluate dlns_dlnR
        M: numpy.ndarray
               Mass at which to evaluate dlns_dlnR
               Units: Ms h^{-1}

        Returns
        -------
        dlns_dlnR: numpy.ndarray
                dlns_dlnR[i,j] where i is the redshift axis and
                j the mass axis
        """

        k = self.k  # h/Mpc
        R = self.radius_M(M)  # Mpc/h
        W, dWdx = self.window(k, R)
        dsigma2_dR = np.pi**-2 * simps(
            k.reshape(1, 1, len(k)) ** 3
            * self.perturbations.matter_power_spectrum(
                z, k, delta=self._camb_delta
            ).reshape(len(z), 1, len(k))
            * W.reshape(1, len(R), len(k))
            * dWdx.reshape(1, len(R), len(k)),
            k,
            axis=-1,
        )

        return R / (2 * self.sigma_z_M(z, M) ** 2) * dsigma2_dR

    def f_sigma_nu(self, z, M):
        r"""
        Computes the multiplicity function
        at the redshift and mass requested
        Computation of the Multiplicity function

        Parameters
        ----------
        z: numpy.ndarray
                   Redshift at which to evaluate f_sigma_nu
        M: numpy.ndarray
               Mass at which to evaluate f_sigma_nu in h^{-1} Ms

        Returns
        -------
        f_sigma_nu: numpy.ndarray
                f_sigma_nu[i,j] where i is the redshift axis and
                j the mass axis
        """
        raise NotImplementedError

    def dn_dm(self, z, M):
        r"""
        Computes the derivative of the number density
        at the redshift and mass requested


        Parameters
        ----------
        z: numpy.ndarray
                   Redshift at which to evaluate dn_dm
        M: numpy.ndarray
               Mass at which to evaluate dn_dm
               Units: h^{-1} Ms

        Returns
        -------
        dn_dm: numpy.ndarray
                dn_dm[i,j] where i is the redshift axis and
                j the mass axis h^4 Mpc^{-3} Ms^{-1}
        """

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        rho_mean_0 = self.background.Omega_m(0, self.nonu) * self.background.rho_crit(0)

        return rho_mean_0 / M**2.0 * self.f_sigma_nu(z, M) * dlnsigmadlnR / (-3)


class HaloStatisticsTinker(HaloStatistics):
    def f_sigma_nu(self, z, M):
        r"""
        Computes the multiplicity function
        at the redshift and mass requested
        Computation of the Multiplicity function

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate f_sigma_nu
        M: float or numpy.ndarray
               Mass at which to evaluate f_sigma_nu in h^{-1} Ms

        Returns
        -------
        f_sigma_nu: numpy.ndarray
                f_sigma_nu[i,j] where i is the redshift axis and
                j the mass axis
        """
        raise NotImplementedError

    def bias(self, z, M):
        Delta = self.get_Delta_crit(z) / self.background.Omega_m(z)

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
        nu = self.nu_z_M(z, M).T
        return (
            1.0
            - A_par * nu**a_par / (nu**a_par + self.delta_c(z) ** a_par)
            + B_par * nu**b_par
            + C_par * nu**c_par
        ).T


class HaloStatisticsCastro(HaloStatistics):
    def f_sigma_nu(self, z, M):
        r"""
        Computes the multiplicity function
        at the redshift and mass requested
        Computation of the Multiplicity function

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate f_sigma_nu
        M: float or numpy.ndarray
               Mass at which to evaluate f_sigma_nu in h^{-1} Ms

        Returns
        -------
        f_sigma_nu: numpy.ndarray
                f_sigma_nu[i,j] where i is the redshift axis and
                j the mass axis
        """
        a1 = 0.7962
        a2 = 0.1449
        az = -0.0658
        p1 = -0.5612
        p2 = -0.4743
        q1 = 0.3688
        q2 = -0.2804
        qz = 0.0251

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        Ommz = self.background.Omega_m(z, self.nonu)[:, np.newaxis]
        nu = self.nu_z_M(z, M)

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
        # if the mass array has less than 4 entries, this causes problem with the derivative
        M = np.asarray(M)
        lenM_orig = M.size
        if lenM_orig < 4:
            M = np.append(M, M[-1] * np.arange(2, 6))

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        Ommz = self.background.Omega_m(z, self.nonu)[:, np.newaxis]
        S8 = self.sigma8 * np.sqrt(self.background.Omega_m(0.0) / 0.3)

        nu = self.nu_z_M(z, M)
        nufnu = self.f_sigma_nu(z, M)
        dlnnufnu_dlnnu = np.zeros(nufnu.shape)
        for i in range(len(z)):
            nufnu_int = interpolate.splrep(np.log(nu[i]), np.log(nufnu[i]), s=0)
            dlnnufnu_dlnnu[i] = interpolate.splev(np.log(nu[i]), nufnu_int, der=1)

        # parameters
        A0, a1, b1, b2, c1 = 1.150, 0.0929, 0.256, 0.173, -0.0372
        b_pbs = 1 - 1 / self.delta_c(z)[:, np.newaxis] * dlnnufnu_dlnnu
        f0 = 1 + a1 * Ommz
        f1 = 1 + b1 * dlnsigmadlnR + b2 * dlnsigmadlnR**2
        f2 = 1 + c1 * S8

        # bias
        bias = A0 * f0 * f1 * f2 * b_pbs

        # original mass array size
        if lenM_orig < len(M):
            bias = bias[:, :lenM_orig]

        return bias
