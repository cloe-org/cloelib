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

    def photoz_rsd_correction(self, z, sigma_zob):
        # sigma_zob = self.scatter_zobs_z(Lambda, z)
        """
        Compute the correction that accounts for photo-z uncertainty and RSD

        Parameters
        ----------
        z:  float or numpy.ndarray
            redshift
        sigma_zob: float
            scatter of observed redshift

        Returns
        -------
        corr0, corr1, corr2: correction terms to the power spectrum

        """

        # growth rate
        f_gr = (self.cosmo.Omm_z(z, self.neutrino_cdm) ** 0.55)[:, np.newaxis]

        ks = self.k * (
            sigma_zob * self.theory["c"] / self.theory["H_z_func"](z) * self.h
        ).reshape(len(z), 1)
        erf_ks = erf(ks)

        corr0 = np.sqrt(np.pi) / (2 * ks) * erf_ks
        corr1 = f_gr / ks**3 * (np.sqrt(np.pi) / 2 * erf_ks - ks * np.exp(-(ks**2)))
        corr2 = (
            f_gr**2
            / ks**5
            * (
                3 * np.sqrt(np.pi) / 8 * erf_ks
                - ks / 4 * (2 * ks**2 + 3) * np.exp(-(ks**2))
            )
        )

        # correct for numerical inaccuracy
        corr1[erf_ks < 0.02] = 2 / 3.0
        corr2[erf_ks < 0.02] = 1 / 5.0

        return corr0, corr1, corr2
