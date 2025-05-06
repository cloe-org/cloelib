import numpy as np  # type: ignore

from .cosmo_temp import _tempPerturbationsCluster


class HaloStatistics:
    def __init__(
        self,
        pertrurbations: Perturbations,
        overdensity_type: str,
        overdensity: int,
        neutrino_cdm: bool,
        k_div: int,
        k_min: float,
        k_max: float,
    ):
        self.cosmo = _tempPerturbationsCluster(pertrurbations)
        self.overdensity_type = overdensity_type  # self.theory['obs_specifications']['CG']['overdensity_type']
        self.overdensity = (
            overdensity  # self.theory['obs_specifications']['CG']['overdensity']
        )

        ################### ARRAYS FOR INTEGRATION VARIABLES ###################

        # wavelength array (integration variable)
        self.k = np.geomspace(k_min, k_max, k_div)

        self.neutrino_cdm = (
            neutrino_cdm  # self.theory["obs_specifications"]["CG"]["neutrino_cdm"]
        )

    def window(self, k, R):
        r"""
        Computes the top-hat window function and its derivative

        Parameters
        ----------
        k: float or numpy.ndarray
               Wavenumber at which to evaluate W(kR)
               Units:  h Mpc^{-1}
        R: float or numpy.ndarray
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

    def sigma_z_M(self, z, M):
        r"""
        Computes the rms at the masses requested from
        the table given by the Boltzman code

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate sigma_z_M
        M: float or numpy.ndarray
               Mass at which to evaluate sigma_z_M in h^{-1} Mpc

        Returns
        -------
        sigma_z_M: numpy.ndarray
                sigma_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        k = self.k  # h/Mpc
        R = self.cosmo.radius_M(M)  # Mpc/h
        W, dWdx = self.window(k, R)
        return np.sqrt(
            (
                1
                / (2.0 * np.pi**2)
                * simps(
                    (k**2.0).reshape(1, 1, len(k))
                    * self.cosmo.Pk_def(z, k, self.neutrino_cdm).reshape(
                        len(z), 1, len(k)
                    )
                    * (W**2.0).reshape(1, len(R), len(k)),
                    k,
                    axis=-1,
                )
            )
        )

    def nu_z_M(self, z, M):
        r"""
        Computes the critical overdensity over the rms
        delta_c/sigma at a given redshift and mass

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate nu_z_M
        M: float or numpy.ndarray
               Mass at which to evaluate nu_z_M

        Returns
        -------
        nu_z_M:   numpy.ndarray
            nu_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        return self.cosmo.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def dlns_dlnR(self, z, M):
        r"""
        Computes the derivatives of the log rms
        with respect to the radius
        at the redshift and mass requested

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate dlns_dlnR
        M: float or numpy.ndarray
               Mass at which to evaluate dlns_dlnR
               Units: Ms h^{-1}

        Returns
        -------
        dlns_dlnR: numpy.ndarray
                dlns_dlnR[i,j] where i is the redshift axis and
                j the mass axis
        """

        k = self.k  # h/Mpc
        R = self.cosmo.radius_M(M)  # Mpc/h
        W, dWdx = self.window(k, R)
        dsigma2_dR = np.pi**-2 * simps(
            k.reshape(1, 1, len(k)) ** 3
            * self.cosmo.Pk_def(z, k, self.neutrino_cdm).reshape(len(z), 1, len(k))
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

    def dn_dm(self, z, M):
        r"""
        Computes the derivative of the number density
        at the redshift and mass requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate dn_dm
        M: float or numpy.ndarray
               Mass at which to evaluate dn_dm
               Units: h^{-1} Ms

        Returns
        -------
        dn_dm: numpy.ndarray
                dn_dm[i,j] where i is the redshift axis and
                j the mass axis h^4 Mpc^{-3} Ms^{-1}
        """

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        rho_mean_0 = self.cosmo.Omega_m(0, self.nonu) * self.cosmo.rho_crit_z(0)

        return rho_mean_0 / M**2.0 * self.f_sigma_nu(z, M) * dlnsigmadlnR / (-3)


class HaloStatisticsTinker10(HaloStatistics):
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

        Delta = self.cosmo.get_Delta(
            self.overdensity_type, z, "tot", self.overdensity
        ) / self.cosmo.Omm_z(z, nu_cdm="tot")

        if type(M) is not np.ndarray:
            M = np.array([M])

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
            - A_par * nu**a_par / (nu**a_par + self.cosmo.delta_c(z) ** a_par)
            + B_par * nu**b_par
            + C_par * nu**c_par
        ).T


class HaloStatisticsCastro23(HaloStatistics):
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
        Ommz = self.cosmo.Omm_z(z, self.neutrino_cdm)[:, np.newaxis]
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
        Ommz = self.cosmo.Omm_z(z, self.neutrino_cdm)[:, np.newaxis]
        S8 = self.cosmo.parameter["sigma8_0"] * np.sqrt(
            self.cosmo.parameter["Omm"] / 0.3
        )

        nu = self.nu_z_M(z, M)
        nufnu = self.f_sigma_nu(z, M)
        dlnnufnu_dlnnu = np.zeros(nufnu.shape)
        for i in range(len(z)):
            nufnu_int = interpolate.splrep(np.log(nu[i]), np.log(nufnu[i]), s=0)
            dlnnufnu_dlnnu[i] = interpolate.splev(np.log(nu[i]), nufnu_int, der=1)

        # parameters
        A0, a1, b1, b2, c1 = 1.150, 0.0929, 0.256, 0.173, -0.0372
        b_pbs = 1 - 1 / self.cosmo.delta_c(z)[:, np.newaxis] * dlnnufnu_dlnnu
        f0 = 1 + a1 * Ommz
        f1 = 1 + b1 * dlnsigmadlnR + b2 * dlnsigmadlnR**2
        f2 = 1 + c1 * S8

        # bias
        bias = A0 * f0 * f1 * f2 * b_pbs

        # original mass array size
        if lenM_orig < len(M):
            bias = bias[:, :lenM_orig]

        return bias
