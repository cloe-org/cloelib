from ...cosmology.cosmology import Perturbations


class _tempPerturbationsCluster:
    def __init__(self, perturbations: Perturbations):
        self.perturbations = perturbations

    def Pk_def(self, z, k, nu_cdm):
        r"""
        Computes the power spectrum for the clusters probe.


        Parameters
        ----------
        z: float
            Redshift at which to evaluate the power spectrum.
        k: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum.
            Units: h Mpc^{-1}

        Returns
        -------
            float or numpy.ndarray
            Value of power spectrum
            at a given redshift and k-mode for galaxy clusters
            Units: h^{-3} Mpc^3
        """

        # general cloe works without h units
        k_noh = k * self.h

        if nu_cdm == "cb":
            return (self.theory["Pk_cb"].P(z, k_noh)) * self.h**3
        else:
            return (self.theory["Pk_delta"].P(z, k_noh)) * self.h**3

    def dV_dzdO(self, z):
        r"""
        Computes the volume element per redshit per solid angle
        at the redshift requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   redshift at which to evaluate dV_dzdO

        Returns
        -------
        dV_dzdO: numpy.ndarray
                dV_dzdO[i] where i is the redshift axis
                Units: Mpc^3 h^{-3}
        """

        return (
            self.theory["c"]
            * self.theory["r_z_func"](z) ** 2.0
            / self.theory["H_z_func"](z)
            * self.h**3.0
        )

    def rho_crit_0(self):
        r"""
        Critical density of the universe at redshift = 0
        Units: Mpc^{-3} Ms h^2
        """

        hubble_value = 100.0 / 3.085677581491367e19  # H0/h / unit conversion
        G_unit = G.to(units.Mpc**3.0 / (units.Msun * units.s**2)).value

        return 3.0 * hubble_value**2.0 / (8.0 * np.pi * G_unit)

    def rho_crit_z(self, z):
        r"""
        Critical density of the universe at a redshift
        Units: Mpc^{-3} Ms h^2
        """

        hubble_value = (
            self.theory["H_z_func"](z) / 3.085677581491367e19 / self.h
        )  # Hz/h / unit conversion
        G_unit = G.to(units.Mpc**3.0 / (units.Msun * units.s**2.0)).value

        return 3.0 * hubble_value**2.0 / (8.0 * np.pi * G_unit)

    def rho_mean_0(self):
        r"""
        Mean matter density at redshift=0
        Units: Mpc^{-3} Ms h^2
        """

        if self.neutrino_cdm == "cb":
            return (self.theory["Omm"] - self.theory["Omnu"]) * self.rho_crit_0()
        else:
            return self.theory["Omm"] * self.rho_crit_0()

    def radius_M(self, M):
        r"""
        Convert the requested mass in the associated radius_M

        Parameters
        ----------
        M: float or numpy.ndarray
              Mass at which the radius is
              to be estimated in h^{-1} Ms

        Returns
        -------
        radius_M: array
                Radius_M in h^{-1} Mpc
        """

        return (M / self.rho_mean_0() * (3.0 / (4.0 * np.pi))) ** (1 / 3.0)

    def Omm_z(self, z, nu_cdm):
        r"""
        Computes the evolution of the matter density
        parameter with reshift

        Parameters
        ----------
        z: float or numpy.ndarray
                  Redshifts at which the computation parameter is
                  to be estimated
        """

        if nu_cdm == "cb":
            return (
                (self.theory["Omm"] - self.theory["Omnu"])
                * (1.0 + z) ** 3.0
                * (self.theory["H0"] / self.theory["H_z_func"](z)) ** 2.0
            )
        else:
            return (
                self.theory["Omm"]
                * (1.0 + z) ** 3.0
                * (self.theory["H0"] / self.theory["H_z_func"](z)) ** 2.0
            )

    def rho_mean_z(self, z, nu_cdm):
        r"""
        Mean matter density at z
        Units: Mpc^{-3} Ms h^2
        """

        return self.Omm_z(z, nu_cdm) * self.rho_crit_z(z)

    def delta_c(self, z):
        r"""
        Computes the critical overdensity at a given redshift
        following an approximation from Kitayama & Suto (1999)

        Parameters
        ----------
        z: float
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
            * (1.0 + 0.012299 * np.log10(self.Omm_z(z, nu_cdm="tot")))
        )

    def get_Delta(self, overdensity_type, z, nu_cdm, overdensity=200):
        r"""
        Overdensity factor.

        Parameters
        ----------
        overdensity_type: str
            The overdensity type. Possibilities are: "crit", "mean", "vir".
        z: float
            Redshift.
        overdensity: int
            The overdensity. If a virial density is assumed, this input
            variable is not used.

        Returns
        -------
        overdensity: float
            The overdensity factor which needs to be multiplied to the critical
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
        if overdensity_type == "crit":
            Delta = overdensity

        elif overdensity_type == "mean":
            Delta = overdensity * self.Omm_z(z, nu_cdm)

        elif overdensity_type == "vir":
            x = self.Omm_z(z, nu_cdm) - 1.0
            Delta = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

        else:
            raise ValueError("Invalid overdensity definition, %s." % overdensity_type)

        return Delta
