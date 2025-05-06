class Profile:
    def __init__(
        self,
        pertrurbations: Perturbations,
        halo_statistics: HaloStatistics,
        two_halo="None",
        offcentering=False,
        rms_off=0.0,
        f_off=0.0,
        trunc_fact=3.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    ):
        self.cosmo = _tempPerturbationsCluster(pertrurbations)
        self.halo_statistics = halo_statistics
        self.two_halo = two_halo  # self.theory['obs_specifications']['CG']['two_halo']
        if self.two_halo not in ["None", "sum", "max"]:
            raise ValueError("Invalid 'two_halo' definition, %s." % self.two_halo)

        # offcentering
        self.offcentering = offcentering  # self.theory['obs_specifications']['CG']['offcentering']  # offcentering
        self.rms_off = rms_off  # self.theory['obs_specifications']['CG']['rms_off']                # rms_off
        self.f_off = f_off  # self.theory['obs_specifications']['CG']['f_off']                  # f_off
        self.trunc_fact = trunc_fact

        # ???
        self.zs_max = zs_max  # self.theory['obs_specifications']['CG']['zs_max']
        self.mean_nz = mean_nz  # self.theory['obs_specifications']['CG']['mean_nz']
        self.sigma_nz = sigma_nz  # self.theory['obs_specifications']['CG']['sigma_nz']
        self.alpha_nz = alpha_nz  # self.theory['obs_specifications']['CG']['alpha_nz']

        # true redshift array (integration variable)
        z_min = 1e-5
        z_max = (
            self.zs_max - 1e-5
        )  # correction needed for avoiding zero values in n_zs_norM computation
        self.z_div = 50
        self.zed = np.linspace(z_min, z_max, self.z_div + 1)

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.zed)
        self.nzs = self.n_zs(self.zed)
        self.r_interp = np.logspace(-10, 2.5, 200)

    def sigma_crit(self, z, z_sources):
        r"""
        Computes the critical surface mass density


        Parameters
        ----------
        z: float
                   Redshift at which to evaluate the critical density
        z_sources: float
                   Redshift of the galaxy sources

        Returns
        -------
        sigma_crit : float
                     Critical surface mass density (unit : Msun/pc^2)
        """

        light_speed = self.cosmo["c"] * (units.km / units.s)
        fact = light_speed**2.0 / (4.0 * np.pi * G)
        fact = fact.to(units.Msun / units.pc).value
        d_a_sources = self.cosmo["d_z_func"](z_sources) * 1.0e6
        d_a_l = self.cosmo["d_z_func"](z) * 1.0e6
        d_m_l = (1.0 + z) * d_a_l
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_h = self.cosmo["c"] / (self.cosmo["H0"] * 1.0e-6)
        d_a_lens_source = (
            1.0
            / (1.0 + z_sources)
            * (
                d_m_sources
                * np.sqrt(
                    1.0 + self.cosmo["Omk"] * (d_m_l[:, np.newaxis] ** 2.0 / d_h**2.0)
                )
                - d_m_l[:, np.newaxis]
                * np.sqrt(1.0 + self.cosmo["Omk"] * (d_m_sources**2.0 / d_h**2.0))
            )
        )
        sig_crit = fact * (d_a_sources / (d_a_l[:, np.newaxis] * d_a_lens_source))

        return sig_crit / self.cosmo.h  # Ms pc^{-2} h

    def n_zs_norM(self, z):
        r"""
        galaxy number density normalization per redshift


        Parameters
        ----------
        z: float or np.ndarray
                   Redshift at which to evaluate the
                   normalization of the galaxy number density

        Returns
        -------
        n_zs_norM: float or np.ndarray
                   Galaxy number density normalization per redshift
        """

        n_zs_norM = 1.0 / (
            skewnorm.cdf(self.zs_max, self.alpha_nz, self.mean_nz, self.sigma_nz)
            - skewnorm.cdf(z, self.alpha_nz, self.mean_nz, self.sigma_nz)
        )

        return n_zs_norM

    def n_zs(self, z):
        r"""
        galaxy number density per redshift


        Parameters
        ----------
        z: float or np.ndarray
                   Redshift at which to evaluate the
                   galaxy number density

        Returns
        -------
        n_zs: float or np.ndarray
               Galaxy number density per redshift
        """

        n_zs = np.zeros((len(z), self.z_div + 1))
        for z_ind, zed in enumerate(z):
            z_s = np.linspace(zed + 1.0e-5, self.zs_max, self.z_div + 1)
            n_zs[z_ind] = skewnorm.pdf(z_s, self.alpha_nz, self.mean_nz, self.sigma_nz)

        return n_zs

    def m_sig_crit_m1(self, z, zbin):
        r"""
        Effective critical surface mass density


        Parameters
        ----------
        z: float
                Redshift at which to evaluate the
                effective critcal surface mass density

        Returns
        -------
        m_sigma_crit_m1: float
                Effective critical surface mass density (units : pc^2/Msun)
        """

        z_s = np.linspace(z + 1.0e-5, self.zs_max, self.z_div + 1, axis=1)
        sig_crit_m1 = self.nzs[zbin] * 1.0 / self.sigma_crit(z, z_s)

        return self.nzsnorM[zbin] * simps(sig_crit_m1, x=z_s)  # pc^2 / Msun / h

    def surface_mass_density_cen(self, R, z, c, M, force_no_2h=False):
        r"""
        Centered surface mass density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        c: float
            Concentration parameter of the cluster
        M: Float
            Mass of the cluster (Msun)
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term

        Returns
        -------
        surface_mass_density: np.ndarray
                              centered surface mass density profile (units : Msun / pc**2)

        """

        Delta_crit = self.halo_statistics.get_Delta_crit(z[:, np.newaxis])
        rho_c = self.rho_crit_z(z[:, np.newaxis]) / self.cosmo.h
        densityThreshold = Delta_crit * rho_c

        RDelta = (3.0 * M / 4.0 / np.pi / densityThreshold) ** (1.0 / 3.0)

        Sigma = self._surface_mass_density_cen(R, RDelta, c, Delta_crit, rho_c)

        if force_no_2h == False and self.two_halo == "sum":
            Sigma += self.surface_mass_density_2h(R, z, M)
        elif force_no_2h == False and self.two_halo == "max":
            Sigma_2h = self.surface_mass_density_2h(R, z, M)
            Sigma = np.maximum(Sigma, Sigma_2h)

        return Sigma

    def _surface_mass_density_cen(self, R, RDelta, c, Delta_crit, rho_c):
        return NotImplementedError

    def surface_mass_density(self, R, z, c, M, force_no_2h=False, force_no_off=False):
        r"""
        Surface mass density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        c: float
            Concentration parameter of the cluster
        M: Float
            Mass of the cluster (Msun)
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term
        force_no_off: bool
            if True, force the non-inclusion of the off-centering

        Returns
        -------
        surface_mass_density: np.ndarray
                              surface mass density profile (units : Msun / pc**2)

        """
        if force_no_off == False and self.offcentering and self.rms_off >= 1.0e-4:

            R = np.asarray(R)
            Sigma_off = np.zeros_like(R)

            ir.Sigma_off(
                R,
                self.r_interp,
                self.surface_mass_density_cen(
                    self.r_interp, z, c, M, force_no_2h=False
                ),
                self.rms_off,
                Sigma_off,
            )

            Sigma_cen = self.surface_mass_density_cen(R, z, c, M, force_no_2h=False)
            return (1.0 - self.f_off) * Sigma_cen + self.f_off * Sigma_off

        else:

            return self.surface_mass_density_cen(R, z, c, M, force_no_2h)

    def excess_surface_mass_density(self, R, z, c, M):
        r"""
        Excess surface mass density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        c: float
            Concentration parameter of the cluster
        M: Float
            Mass of the cluster (Msun)

        Returns
        -------
        excess_surface_mass_density: float or np.ndarray
                                     excess surface density (units : Msun / pc**2)
        """
        Delta_crit = self.halo_statistics.get_Delta_crit(z[:, np.newaxis])
        rho_c = self.cosmo.rho_crit_z(z[:, np.newaxis]) / self.cosmo.h
        densityThreshold = Delta_crit * rho_c

        RDelta = (3.0 * M / 4.0 / np.pi / densityThreshold) ** (1.0 / 3.0)
        Rs = RDelta / c
        x = R / Rs

        Sigma_mean = self._excess_surface_mass_density(R, RDelta, c, Delta_crit, rho_c)

        Sigma = self.surface_mass_density_cen(R, z, c, M, force_no_2h=True)
        DeltaSigma = Sigma_mean - Sigma

        if self.two_halo == "sum":
            DeltaSigma += self.excess_surface_mass_density_2h(R, z, M)
        elif self.two_halo == "max":
            DeltaSigma_2h = self.excess_surface_mass_density_2h(R, z, M)
            DeltaSigma = np.maximum(DeltaSigma, DeltaSigma_2h)

        if self.offcentering:

            if self.rms_off >= 1.0e-4 and self.f_off >= 1.0e-4:

                R = np.asarray(R)
                DeltaSigma_off = np.zeros_like(R)

                ir.DeltaSigma_off(
                    R,
                    self.r_interp,
                    self.r_interp,
                    self.surface_mass_density_cen(
                        self.r_interp, z, c, M, force_no_2h=False
                    ),
                    self.rms_off,
                    DeltaSigma_off,
                )

                DeltaSigma *= 1.0 - self.f_off
                DeltaSigma += self.f_off * DeltaSigma_off

            elif self.rms_off < 1.0e-4 and self.f_off >= 1.0:

                DeltaSigma = np.zeros(len(DeltaSigma))

        return DeltaSigma

    def _excess_surface_mass_density(self, R, RDelta, c, Delta_crit, rho_c):
        return NotImplementedError

    def surface_mass_density_2h(self, R, z, M=1e14):
        r"""
        Surface 2-halo density profile at radius R, generalized to handle arrays of z and M.

        Parameters
        ----------
        R: float
            Radius at which the profile is to be computed (units : Mpc)
        z: np.ndarray
            Redshift(s) at which the mean matter content is
            to be computed. Can be an array.
        M: np.ndarray
            Mass(es) of the cluster(s) (Msun), used only for the bias computation.
            Can be an array.

        Returns
        -------
        surface_mass_density_2h: np.ndarray
            2-halo surface mass density profile (units : Msun / pc**2),
            computed for each z and M.
        """

        # Ensure z and M are arrays for broadcasting
        if type(z) is not np.ndarray:
            z = np.array([z])
        if type(M) is not np.ndarray:
            M = np.array([M])

        # Define base quantities
        D_A = self.cosmo["d_z_func"](z)  # D_A should now be (Nz, 1)
        theta = R / D_A  # R is a scalar, so theta has shape (Nz, 1)

        kl_min = 1.0e-4
        kl_max = 1.0e2
        kl_array = np.logspace(np.log10(kl_min), np.log10(kl_max), 500)

        # Bias calculation
        bias_z = self.halo_statistics.bias(z, M)

        # Compute P(k) interpolation and Sigma for each redshift z
        Sigma = np.zeros((z.size, M.size))
        for i, z_val in enumerate(z):  # Loop over redshift values
            # Get P(k) for this redshift
            Pk_interp = interpolate.InterpolatedUnivariateSpline(
                kl_array, self.cosmo.Pk_def(z_val, kl_array, nu_cdm="tot")
            )

            # Define the integrand for this redshift
            def integrand(l):
                kl = l / (1.0 + z_val) / D_A[i]
                return j0(l * theta[i]) * l * Pk_interp(kl)

            # Compute rho_m for this redshift
            rho_m = (
                self.cosmo.Omega_m(z_val, nonu=False)
                * self.cosmo.rho_crit_z(z_val)
                / self.cosmo.h
            )

            # Compute Sigma for each mass M
            Sigma_z = quad_vec(
                integrand,
                kl_min * (1.0 + z_val) * D_A[i],
                kl_max * (1.0 + z_val) * D_A[i],
                epsrel=1e-1,
            )[0]

            Sigma_z *= (
                1.0e-12
                * rho_m
                * bias_z[i]
                / (2.0 * np.pi * (1.0 + z_val) ** 3.0 * D_A[i] ** 2.0)
            )

            # Store the result for this redshift
            Sigma[i, :] = Sigma_z

        return Sigma  # Shape: (Nz, Nm)

    def excess_surface_mass_density_2h(self, R, z, M=1e14):
        r"""
        Excess surface 2-halo density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        M: Float
            Mass of the cluster (Msun), used only for the bias computation

        Returns
        -------
        excess_surface_mass_density_2h: np.ndarray
                                 2-halo excess surface mass density profile (units : Msun / pc**2)

        """
        # Ensure z and M are arrays for broadcasting
        if type(z) is not np.ndarray:
            z = np.array([z])
        if type(M) is not np.ndarray:
            M = np.array([M])

        # Define base quantities
        D_A = self.cosmo["d_z_func"](z)  # D_A should now be (Nz, 1)
        theta = R / D_A  # R is a scalar, so theta has shape (Nz, 1)

        kl_min = 1.0e-4
        kl_max = 1.0e2
        kl_array = np.logspace(np.log10(kl_min), np.log10(kl_max), 500)

        # Bias calculation
        bias_z = self.halo_statistics.bias(z, M)

        # Compute P(k) interpolation and Sigma for each redshift z
        DeltaSigma = np.zeros((z.size, M.size))
        for i, z_val in enumerate(z):  # Loop over redshift values
            # Get P(k) for this redshift
            Pk_interp = interpolate.InterpolatedUnivariateSpline(
                kl_array, self.cosmo.Pk_def(z_val, kl_array, nu_cdm="tot")
            )

            # Define the integrand for this redshift
            def integrand(l):
                kl = l / (1.0 + z_val) / D_A[i]
                j2 = 2.0 / (l * theta[i]) * j1(l * theta[i]) - j0(l * theta[i])
                return j2 * l * Pk_interp(kl)

            # Compute rho_m for this redshift
            rho_m = (
                self.cosmo.Omega_m(z_val, nuno=False)
                * self.cosmo.rho_crit_z(z_val)
                / self.cosmo.h
            )

            # Compute Sigma for each mass M
            DeltaSigma_z = quad_vec(
                integrand,
                kl_min * (1.0 + z_val) * D_A[i],
                kl_max * (1.0 + z_val) * D_A[i],
                epsrel=1e-1,
            )[0]
            DeltaSigma_z *= (
                1.0e-12
                * rho_m
                * bias_z[i]
                / (2.0 * np.pi * (1.0 + z_val) ** 3.0 * D_A[i] ** 2.0)
            )

            # Store the result for this redshift
            DeltaSigma[i, :] = DeltaSigma_z

        return DeltaSigma  # Shape: (Nz, Nm)

    def F_term(self, x):
        r"""
        One-Halo profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        float
            One-Halo profile F term.

        """
        return NotImplementedError

    def G_term(self, x):
        r"""
        One-Halo profile G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        float
            One-Halo profile G term.

        """
        return NotImplementedError


class ProfileNFW(Profile):
    def F_term(self, x):
        r"""
        One-Halo NFW F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_NFW: float
                One-Halo NFW F term.

        Notes
        -----
        Implementation of second part of Eq. 4 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return (1.0 - np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)) / (x**2.0 - 1.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)) / (x**2.0 - 1.0)

    def G_term(self, x):
        r"""
        One-Halo NFW G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        F_NFW: float
                One-Halo NFW G term.

        Notes
        -----
        Implementation of Eq. 5 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0 + np.log(1.0 / 2.0)
        if x > 1.0:
            return np.log(x / 2.0) + np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def _surface_mass_density_cen(self, R, RDelta, c, Delta_crit, rho_c):

        Rs = RDelta / c
        x = R / Rs

        F = np.vectorize(self.F_term)(x)
        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta_crit * c**3.0 / (3.0 * m_nfw) * rho_c

        Sigma = 2.0 * rho_s * Rs * F * 1.0e-12

        return Sigma

    def _excess_surface_mass_density(self, R, RDelta, c, Delta_crit, rho_c):

        Rs = RDelta / c
        x = R / Rs

        G = np.vectorize(self.G_term)(x)

        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta_crit * c**3.0 / (3.0 * m_nfw) * rho_c

        Sigma_mean = 4.0 * rho_s * Rs * (G / x**2.0) * 1.0e-12
        return Sigma_mean


class ProfileBMO(Profile):
    def F_term(self, x):
        r"""
        One-Halo BMO F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_BMO : float
                One-Halo BMO F term.

        Notes
        -----
        Implementation of Eq. A.5 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0
        if x > 1.0:
            return np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def G_term(self, x):
        r"""
        One-Halo BMO G term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        G_BMO : float
                One-Halo BMO G term.

        Notes
        -----
        Implementation of Eq. A.28 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return (self.F_term(x) - 1.0) / (1.0 - x**2.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - self.F_term(x)) / (x**2.0 - 1.0)

    def _surface_mass_density_cen(self, R, RDelta, c, Delta_crit, rho_c):

        Rs = RDelta / c
        x = R / Rs

        Rt = self.trunc_fact * RDelta
        tau = Rt / Rs

        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (
                c
                * (tau**2.0 + 1.0)
                * (
                    c * (c + 1.0)
                    - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                    - 2.0 * tau**4.0
                )
                + tau
                * (c + 1.0)
                * (tau**2.0 + c**2.0)
                * (
                    2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                    + tau
                    * (tau**2.0 - 3.0)
                    * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                )
            )
        )

        rho_s_bmo = Delta_crit * c**3.0 / (3.0 * m_bmo) * rho_c

        const = rho_s_bmo * Rs

        G = np.vectorize(self.G_term)(x)
        F = np.vectorize(self.F_term)(x)

        term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0
        term2 = 2.0 * (tau**2.0 + 1.0) * G
        term3 = 8.0 * F
        term4 = (tau**4.0 - 1.0) / (tau**2.0 * (tau**2.0 + x**2.0))
        term5 = (
            np.pi
            * (4.0 * (tau**2.0 + x**2.0) + tau**2.0 + 1.0)
            / (tau**2.0 + x**2.0) ** (3.0 / 2.0)
        )
        term6 = (
            tau**2.0 * (tau**4.0 - 1.0)
            + (tau**2.0 + x**2.0) * (3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0)
        ) / (tau**3.0 * (tau**2.0 + x**2.0) ** (3.0 / 2.0))

        L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

        Sigma = 1e-12 * const * term1 * (term2 + term3 + term4 - term5 + term6 * L)
        return Sigma

    def _excess_surface_mass_density(self, R, RDelta, c, Delta_crit, rho_c):

        Rs = RDelta / c
        x = R / Rs

        Rt = self.trunc_fact * RDelta
        tau = Rt / Rs

        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (
                c
                * (tau**2.0 + 1.0)
                * (
                    c * (c + 1.0)
                    - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                    - 2.0 * tau**4.0
                )
                + tau
                * (c + 1.0)
                * (tau**2.0 + c**2.0)
                * (
                    2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                    + tau
                    * (tau**2.0 - 3.0)
                    * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                )
            )
        )

        rho_s_bmo = Delta_crit * c**3.0 / (3.0 * m_bmo) * rho_c

        const = 2.0 * np.pi * rho_s_bmo * Rs**3.0
        term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0

        F = np.vectorize(self.F_term)(x)
        term2 = 2.0 * (tau**2.0 + 1.0 + 4.0 * (x**2.0 - 1.0)) * F

        G = np.vectorize(self.G_term)(x)
        term3 = (
            np.pi * (3.0 * tau**2.0 - 1.0) + 2.0 * tau * (tau**2.0 - 3.0) * np.log(tau)
        ) / tau

        term4 = tau**3.0 * np.sqrt(tau**2.0 + x**2.0)
        term5 = -(tau**3.0) * np.pi * (4.0 * (tau**2.0 + x**2.0) - tau**2.0 - 1.0)
        term6 = -(tau**2.0) * (tau**4.0 - 1.0) + +(tau**2.0 + x**2.0) * (
            3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0
        )
        L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

        M_proj = const * term1 * (term2 + term3 + (term5 + term6 * L) / term4)

        Sigma_mean = M_proj / (np.pi * R**2.0) * 1.0e-12

        return Sigma_mean
