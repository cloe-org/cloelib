from ...cosmology.cosmology import Perturbations


class _tempPerturbationsCluster:
    def __init__(self, perturbations: Perturbations):
        self.perturbations = perturbations

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
