class HaloCovariance:
    def __init__(
        self,
        pertrurbations: Perturbations,
    ):
        self.cosmo = _tempPerturbationsCluster(pertrurbations)

    def Kl_coeff(self, L):
        r"""
        Coefficients of the spherical harmonics expansion of the angular part of the window function

        Parameters
        ----------
        L: int
           Maximum number at which to evaluate the coefficients

        Returns
        -------
        KL: numpy.ndarray
            Coefficients up to L multipole
        """

        ell = np.linspace(0, L, L + 1, dtype=int)
        theta = np.arccos(1 - (self.area * (np.pi / 180.0) ** 2.0) / (2 * np.pi))
        KL = (
            np.sqrt(np.pi / (2.0 * ell + 1.0))
            * (
                eval_legendre(ell - 1, np.cos(theta))
                - eval_legendre(ell + 1, np.cos(theta))
            )
            / (2.0 * np.pi * (1 - np.cos(theta)))
        )
        KL[0] = 1 / (2.0 * np.sqrt(np.pi))
        return KL

    def cov_window(self, zbin, ztab, k, L, KL):
        r"""
        Computes the window function between redshifts bins

        Parameters
        ----------
        zbins: int
               Index of the redshift bins at which to evaluate the window function
        ztab: numpy.ndarray
              Array of redshifts (integration variable) between zbins[zbin] and zbins[zbin+1]
        k: numpy.ndarray
           Wavenumbers used to evaluate power spectrum in h Mpc^{-1}
        L: int
           Maximum number at which the coefficients are evaluated
        KL: numpy.ndarray
            Spherical harmonic expansion coefficients

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i and j are two redshift bin and k are the wavenumbers
        """

        rvec = self.cosmo["r_z_func"](ztab) * self.h  # Mpc  h^{-1}
        Vz = (rvec[-1] ** 3 - rvec[0] ** 3) / 3  # Mpc^3 h^{-3}
        kr = self.k[:, np.newaxis] * rvec
        self.rint[zbin] = (
            1
            / Vz
            * simps(
                rvec**2.0
                * np.array(
                    [spherical_jn(l, kr, derivative=False) for l in range(L + 1)]
                ),
                rvec,
                axis=-1,
            ).T
        )
        return (4 * np.pi) * np.sum(
            self.rint[:, :] * self.rint[: (zbin + 1), :, :] * KL[:] ** 2, axis=-1
        )
